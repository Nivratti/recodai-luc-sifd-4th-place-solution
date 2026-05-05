from __future__ import annotations

"""High-level CBIR engine.

The main performance trick is: **embed once, then rank with matrix multiplication**.

Typical use cases
-----------------
1) **Rank within a single figure** (e.g., 30 panels): use :meth:`PanelCBIR.rank_within`.
2) **Rank many queries vs a shared candidate set**: use :meth:`PanelCBIR.build_index` once, then :meth:`PanelCBIR.rank`.

All ranking uses cosine similarity (dot product of L2-normalized embeddings).
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

from .config import CBIRConfig
from .embedders import BaseEmbedder, build_embedder, normalize_model_output
from .index import load_index_npz, save_index_npz
from .types import CBIRIndex, Match, ImageInput
from .utils import (
    direct_resize,
    estimate_scores_bytes,
    l2_normalize,
    letterpad_to_square,
    pil_to_tensor,
    safe_load_image_pil,
    set_determinism,
    topk_indices_per_row,
    to_rgb_pil,
)


def _resolve_score_device(cfg: CBIRConfig, embedder: BaseEmbedder) -> torch.device:
    if cfg.score_device is None:
        return embedder.device
    dev = str(cfg.score_device)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("score_device requests CUDA but CUDA is not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(dev)


def _compute_scores(
    query_embs: np.ndarray,
    cand_embs: np.ndarray,
    *,
    device: torch.device,
    fp16: bool,
    chunk_size: int,
) -> np.ndarray:
    """Compute cosine scores between L2-normalized embeddings.

    Returns an array of shape (Q, N), dtype float32.
    """

    if query_embs.ndim != 2 or cand_embs.ndim != 2:
        raise ValueError("query_embs and cand_embs must be 2D")
    if query_embs.shape[1] != cand_embs.shape[1]:
        raise ValueError(f"Embedding dim mismatch: {query_embs.shape} vs {cand_embs.shape}")

    Q, _ = query_embs.shape
    N = cand_embs.shape[0]

    # Use torch matmul when possible (fast BLAS and/or CUDA).
    use_torch = device.type == "cuda" or (device.type == "cpu" and torch.get_num_threads() > 1)
    if not use_torch:
        return (query_embs @ cand_embs.T).astype(np.float32, copy=False)

    q = torch.from_numpy(query_embs).to(device, non_blocking=True)
    c = torch.from_numpy(cand_embs).to(device, non_blocking=True)

    use_fp16 = bool(fp16 and device.type == "cuda")
    if use_fp16:
        q = q.half()
        c = c.half()

    if chunk_size and int(chunk_size) > 0:
        parts: List[np.ndarray] = []
        for j0 in range(0, N, int(chunk_size)):
            j1 = min(N, j0 + int(chunk_size))
            s = q @ c[j0:j1].T
            parts.append(s.float().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(parts, axis=1)

    s = q @ c.T
    return s.float().cpu().numpy().astype(np.float32, copy=False)


class PanelCBIR:
    """CBIR engine for ranking panel images."""

    def __init__(self, cfg: CBIRConfig):
        self.cfg = cfg
        if cfg.deterministic and cfg.seed is not None:
            set_determinism(int(cfg.seed))

        self.embedder = build_embedder(cfg)
        logger.info(
            f"Embedder ready: backend={cfg.backend}, model={cfg.model_name}, dim={self.embedder.output_dim()}, "
            f"size={self.embedder.input_size}, resize_mode={cfg.resize_mode}, grayscale={cfg.grayscale}"
        )
        self._score_device = _resolve_score_device(cfg, self.embedder)

    # -----------------
    # Embedding / Index
    # -----------------

    def build_index(
        self,
        candidates: Sequence[ImageInput],
        *,
        ids: Optional[Sequence[str]] = None,
        cache_path: Optional[str | Path] = None,
        reuse_cache: bool = True,
        desc: str = "Indexing",
    ) -> CBIRIndex:
        """Embed candidate images and return a reusable `CBIRIndex`.

        If `cache_path` is provided and exists, the cached index is loaded (when `reuse_cache=True`).
        """

        if cache_path is not None and reuse_cache:
            p = Path(cache_path)
            if p.exists():
                try:
                    idx = load_index_npz(p)
                    logger.info(f"Loaded cached index: {p} (N={len(idx.paths)}, D={idx.embeddings.shape[1]})")
                    return idx
                except Exception as e:
                    logger.warning(f"Failed to load cache index at {p}: {type(e).__name__}: {e}. Rebuilding.")

        good_paths, embs = self.embedder.embed_inputs(candidates, ids=ids, desc=desc)
        if len(good_paths) == 0:
            raise RuntimeError("No candidate images could be embedded (all failed to load?)")

        meta = {
            "config": asdict(self.cfg),
            "backend": str(self.cfg.backend),
            "model_name": str(self.cfg.model_name),
            "embed_dim": int(embs.shape[1]),
        }
        index = CBIRIndex(paths=list(good_paths), embeddings=embs, meta=meta)

        if cache_path is not None:
            p = save_index_npz(index, cache_path)
            logger.info(f"Saved index cache: {p}")

        return index

    def embed_query(self, query: ImageInput, *, id: Optional[str] = None) -> Tuple[str, np.ndarray]:
        """Embed a single query image and return (path, embedding)."""

        qpath = Path(query)
        im = safe_load_image_pil(qpath)
        if im is None:
            raise RuntimeError(f"Query image cannot be read: {qpath}")

        pil = to_rgb_pil(im, grayscale=self.cfg.grayscale)
        if self.cfg.resize_mode == "resize":
            pil2 = direct_resize(pil, self.embedder.input_size)
        elif self.cfg.resize_mode == "letterpad":
            pil2 = letterpad_to_square(pil, self.embedder.input_size)
        else:
            raise ValueError(f"resize_mode must be 'resize' or 'letterpad', got: {self.cfg.resize_mode}")

        x = pil_to_tensor(pil2, self.embedder.mean, self.embedder.std).unsqueeze(0)
        x = x.to(self.embedder.device, non_blocking=True)

        model = self.embedder.model
        assert model is not None
        model.eval()

        use_amp = bool(self.cfg.fp16 and self.embedder.device.type == "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            with torch.inference_mode():
                out = model(x)
                feat = normalize_model_output(out)

        q = feat.float().cpu().numpy().astype(np.float32, copy=False)
        q = l2_normalize(q)[0]
        return str(qpath), q

    # -------
    # Ranking
    # -------

    def search(self, index: CBIRIndex, query: str | Path, *, topk: int = 6) -> List[Match]:
        """Rank candidate images for a single query."""

        _, q = self.embed_query(query)
        scores = _compute_scores(q[None, :], index.embeddings, device=self._score_device, fp16=self.cfg.score_fp16, chunk_size=self.cfg.score_chunk_size)[0]

        k = min(int(topk), len(scores))
        if k <= 0:
            return []

        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        return [Match(path=index.paths[int(j)], score=float(scores[int(j)]), rank=r) for r, j in enumerate(idx, start=1)]

    def rank(self, index: CBIRIndex, queries: Sequence[ImageInput], *, query_ids: Optional[Sequence[str]] = None, topk: int = 6) -> Dict[str, List[Match]]:
        """Rank multiple queries against a shared candidate index.

        This embeds all queries in batches and performs a single (or chunked) matrix multiplication.
        """

        good_q, q_embs = self.embedder.embed_inputs(queries, ids=query_ids, desc="Embedding queries")
        if len(good_q) == 0:
            raise RuntimeError("No queries could be embedded")

        scores = _compute_scores(
            q_embs,
            index.embeddings,
            device=self._score_device,
            fp16=self.cfg.score_fp16,
            chunk_size=self.cfg.score_chunk_size,
        )

        k = min(int(topk), index.embeddings.shape[0])
        if k <= 0:
            return {p: [] for p in good_q}

        # Memory warning for large matrices
        est_bytes = estimate_scores_bytes(len(good_q), len(index.paths))
        if est_bytes > 2_000_000_000:  # ~2GB
            logger.warning(f"Large score matrix: approx {est_bytes/1e9:.2f} GB")

        idx = topk_indices_per_row(scores, k)

        out: Dict[str, List[Match]] = {}
        for i, qpath in enumerate(good_q):
            matches: List[Match] = []
            for r, j in enumerate(idx[i], start=1):
                matches.append(Match(path=index.paths[int(j)], score=float(scores[i, int(j)]), rank=r))
            out[qpath] = matches
        return out


    def rank_within_index(self, index: CBIRIndex, *, topk: int = 6, exclude_self: bool = True) -> Dict[str, List[Match]]:
        # Rank candidates within the same index (no re-embedding).
        embs = index.embeddings
        n = embs.shape[0]
        if n == 0:
            return {}

        chunk = int(self.cfg.score_chunk_size) if int(self.cfg.score_chunk_size) > 0 else 0
        bytes_est = estimate_scores_bytes(n, n, 4)
        if chunk == 0 and bytes_est > 1_500_000_000:
            chunk = 4096

        scores = _compute_scores(embs, embs, device=self._score_device, fp16=bool(self.cfg.score_fp16), chunk_size=chunk)
        if exclude_self:
            np.fill_diagonal(scores, -np.inf)

        k = min(int(topk), n)
        idx = topk_indices_per_row(scores, k)

        out: Dict[str, List[Match]] = {}
        for i, qpath in enumerate(index.paths):
            matches: List[Match] = []
            for r, j in enumerate(idx[i], start=1):
                j = int(j)
                matches.append(Match(path=index.paths[j], score=float(scores[i, j]), rank=r))
            out[qpath] = matches
        return out

    def rank_within(self, images: Sequence[str | Path], *, topk: int = 6, exclude_self: bool = True) -> Dict[str, List[Match]]:
        """Rank each image against the other images in the same list.

        Optimized for the "~30 panels" use case.
        """

        paths = [Path(p) for p in images]
        good_paths, embs = self.embedder.embed_paths(paths, desc="Embedding")
        if len(good_paths) == 0:
            raise RuntimeError("No images could be embedded")

        N = embs.shape[0]
        scores = _compute_scores(
            embs,
            embs,
            device=self._score_device,
            fp16=self.cfg.score_fp16,
            chunk_size=self.cfg.score_chunk_size,
        )

        if exclude_self:
            np.fill_diagonal(scores, -np.inf)

        k = min(int(topk), N - (1 if exclude_self else 0))
        if k <= 0:
            return {p: [] for p in good_paths}

        idx = topk_indices_per_row(scores, k)

        out: Dict[str, List[Match]] = {}
        for i, qpath in enumerate(good_paths):
            matches: List[Match] = []
            for r, j in enumerate(idx[i], start=1):
                matches.append(Match(path=good_paths[int(j)], score=float(scores[i, int(j)]), rank=r))
            out[qpath] = matches
        return out

    # ---------
    # Utilities
    # ---------

    def save_rankings_json(self, rankings: Dict[str, List[Match]], out_path: str | Path) -> Path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {q: [m.__dict__ for m in ms] for q, ms in rankings.items()}
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return p
