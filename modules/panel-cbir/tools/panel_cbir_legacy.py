#!/usr/bin/env python3
"""
panel_cbir.py — simple, modular CBIR for scientific figure panels (single-script version).

Supports embedders:
  1) timm pretrained models (resnet/vgg/vit/dinov2 in timm if available)
  2) torch.hub pretrained models (optional; requires cache/internet)
  3) SSCD TorchScript model (local .pt/.torchscript)

Resizing:
  - resize      : direct resize to (img_size, img_size)
  - letterpad   : keep aspect ratio, pad to square (img_size, img_size)

CLI behavior:
  - If --query is provided: rank candidates in folder against query.
  - If --query is NOT provided: rank-all (leave-one-out) for every image.
  - By default, saves CLI args to outdir/run_args.json.
  - Optional visualization grids saved to outdir/viz/.

Importable usage:
  from panel_cbir import PanelCBIR, CBIRConfig
  cbir = PanelCBIR(CBIRConfig(...))
  cbir.index_folder("panels/")
  results = cbir.search("panels/img_0001.png", topk=6)
  cbir.save_viz_grid("panels/img_0001.png", results, "out/viz.png")

Dependencies:
  - required: torch, numpy, pillow, loguru, tqdm
  - optional: timm, torchvision
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import shutil
import urllib.request
import subprocess


# -----------------------------
# Config
# -----------------------------
@dataclass
class CBIRConfig:
    # I/O
    img_size: Optional[int] = None            # if None: auto from model defaults (timm) or 224 fallback
    resize_mode: str = "letterpad"           # "resize" | "letterpad"
    grayscale: bool = False                  # useful for blots

    # Runtime
    device: str = "cpu"                      # "cpu" | "cuda" | "cuda:0" ...
    batch_size: int = 32
    fp16: bool = False
    seed: Optional[int] = 0

    # Normalization overrides (if None, use model defaults when possible)
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None

    # Embedder selection
    backend: str = "timm"                    # "timm" | "torchhub" | "sscd"
    model_name: str = "resnet50"             # timm model name or torchhub model name
    hub_repo: str = "pytorch/vision"         # torchhub repo, e.g. "facebookresearch/dinov2"
    hub_source: str = "github"               # torch.hub.load source (keep default)

    # SSCD TorchScript
    sscd_torchscript_path: Optional[str] = None  # local path to TorchScript model

    # Logging/outputs
    deterministic: bool = True


# -----------------------------
# Helpers
# -----------------------------
IMG_EXTS_DEFAULT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"


def download_file(url: str, dst: Path, timeout: int = 60) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Prefer wget if available (matches your command), else fallback to urllib
    wget = shutil.which("wget")
    if wget:
        logger.info(f"Downloading with wget: {url} -> {dst}")
        cmd = [wget, "-O", str(dst), url]
        subprocess.check_call(cmd)
        return

    logger.info(f"Downloading with urllib: {url} -> {dst}")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def discover_images(folder: Path, recursive: bool = True, exts: Optional[Sequence[str]] = None) -> List[Path]:
    folder = Path(folder)
    if exts is None:
        exts_set = IMG_EXTS_DEFAULT
    else:
        exts_set = {e.lower().strip() if e.startswith(".") else f".{e.lower().strip()}" for e in exts}
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    paths: List[Path] = []
    if recursive:
        it = folder.rglob("*")
    else:
        it = folder.glob("*")

    for p in it:
        if p.is_file() and p.suffix.lower() in exts_set:
            paths.append(p)

    paths.sort()
    return paths


def _set_determinism(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _to_rgb_pil(img: Image.Image, grayscale: bool) -> Image.Image:
    if grayscale:
        # Convert to L then back to RGB (3ch) for most CNNs
        return img.convert("L").convert("RGB")
    return img.convert("RGB")


def _letterpad_to_square(img: Image.Image, size: int, fill: int = 0) -> Image.Image:
    """Keep aspect ratio, resize to fit within size x size, pad remaining area."""
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), (fill, fill, fill))
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _direct_resize(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.BILINEAR)


def _pil_to_tensor(img: Image.Image, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
    return torch.from_numpy(arr)


def _safe_load_image(path: Path) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            return im.copy()
    except Exception as e:
        logger.warning(f"Failed to read image: {path} | {type(e).__name__}: {e}")
        return None


def _maybe_log_override(name: str, default_val: Any, user_val: Any) -> Any:
    if user_val is None:
        return default_val
    if user_val != default_val:
        logger.info(f"Override: {name}={user_val} (default was {default_val})")
    return user_val


def _normalize_model_output(out: Any) -> torch.Tensor:
    """
    Try to turn various model outputs into a (B,D) tensor.
    Handles tensor / tuple / dict common cases.
    """
    if isinstance(out, torch.Tensor):
        x = out
    elif isinstance(out, (list, tuple)) and len(out) > 0:
        x = out[0] if isinstance(out[0], torch.Tensor) else out[-1]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported model output tuple/list types: {[type(t) for t in out]}")
    elif isinstance(out, dict):
        # common keys
        for k in ("x", "feat", "features", "embeddings", "last_hidden_state", "pooler_output"):
            if k in out and isinstance(out[k], torch.Tensor):
                x = out[k]
                break
        else:
            # pick first tensor value
            tensors = [v for v in out.values() if isinstance(v, torch.Tensor)]
            if not tensors:
                raise TypeError(f"Unsupported model output dict: keys={list(out.keys())}")
            x = tensors[0]
    else:
        raise TypeError(f"Unsupported model output type: {type(out)}")

    # squeeze common (B,D,1,1) shapes
    if x.ndim == 4 and x.shape[-1] == 1 and x.shape[-2] == 1:
        x = x[:, :, 0, 0]
    if x.ndim == 3:
        # e.g., (B, T, D) -> take CLS token
        x = x[:, 0, :]
    if x.ndim != 2:
        x = x.view(x.shape[0], -1)
    return x


# -----------------------------
# Embedders
# -----------------------------
class BaseEmbedder:
    def __init__(self, cfg: CBIRConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")

        self.model: Optional[torch.nn.Module] = None
        self.input_size: int = 224
        self.mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        self.std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def output_dim(self) -> int:
        raise NotImplementedError

    def preprocess_info(self) -> str:
        return f"size={self.input_size}, resize_mode={self.cfg.resize_mode}, mean={self.mean}, std={self.std}, grayscale={self.cfg.grayscale}"

    def _preprocess_one(self, img: Image.Image) -> torch.Tensor:
        img = _to_rgb_pil(img, grayscale=self.cfg.grayscale)
        if self.cfg.resize_mode not in ("resize", "letterpad"):
            raise ValueError(f"resize_mode must be 'resize' or 'letterpad', got: {self.cfg.resize_mode}")

        if self.cfg.resize_mode == "resize":
            img2 = _direct_resize(img, self.input_size)
        else:
            img2 = _letterpad_to_square(img, self.input_size)

        t = _pil_to_tensor(img2, self.mean, self.std)
        return t

    @torch.inference_mode()
    def embed_paths(self, paths: Sequence[Path], desc: str = "Embedding") -> np.ndarray:
        assert self.model is not None, "Model not initialized"
        self.model.eval()

        # batching
        embs: List[np.ndarray] = []
        bs = max(1, int(self.cfg.batch_size))

        for i in tqdm(range(0, len(paths), bs), desc=desc, unit="batch"):
            batch_paths = paths[i : i + bs]
            imgs: List[torch.Tensor] = []
            ok_count = 0

            for p in batch_paths:
                im = _safe_load_image(p)
                if im is None:
                    continue
                imgs.append(self._preprocess_one(im))
                ok_count += 1

            if ok_count == 0:
                continue

            x = torch.stack(imgs, dim=0).to(self.device, non_blocking=True)

            use_amp = bool(self.cfg.fp16 and self.device.type == "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = self.model(x)
                feat = _normalize_model_output(out)

            feat = feat.float().cpu().numpy()  # (B,D)
            feat = _l2_normalize(feat)
            embs.append(feat)

        if not embs:
            return np.zeros((0, self.output_dim()), dtype=np.float32)

        return np.concatenate(embs, axis=0).astype(np.float32)


class TimmEmbedder(BaseEmbedder):
    def __init__(self, cfg: CBIRConfig):
        super().__init__(cfg)
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except Exception as e:
            raise RuntimeError("timm backend requested but timm is not installed.") from e

        import timm  # type: ignore
        from timm.data import resolve_data_config  # type: ignore

        logger.info(f"Loading timm model: {cfg.model_name} (pretrained=True)")
        model = timm.create_model(cfg.model_name, pretrained=True, num_classes=0, global_pool="avg")
        model.to(self.device)
        self.model = model

        # timm data config
        dc = resolve_data_config({}, model=model)
        # dc has input_size like (3,224,224), mean/std
        default_size = int(dc.get("input_size", (3, 224, 224))[1])
        default_mean = tuple(float(x) for x in dc.get("mean", (0.485, 0.456, 0.406)))
        default_std = tuple(float(x) for x in dc.get("std", (0.229, 0.224, 0.225)))

        self.input_size = _maybe_log_override("img_size", default_size, cfg.img_size)
        self.mean = _maybe_log_override("mean", default_mean, cfg.mean)
        self.std = _maybe_log_override("std", default_std, cfg.std)

        # infer output dim via a tiny dry run
        self._out_dim = self._infer_dim()

    def _infer_dim(self) -> int:
        assert self.model is not None
        x = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device)
        with torch.inference_mode():
            out = self.model(x)
            feat = _normalize_model_output(out)
        return int(feat.shape[1])

    def output_dim(self) -> int:
        return self._out_dim


class TorchHubEmbedder(BaseEmbedder):
    """
    Generic torch.hub embedder.
    Note: torch.hub may need internet unless models are cached.
    """
    def __init__(self, cfg: CBIRConfig):
        super().__init__(cfg)

        logger.info(f"Loading torch.hub model: repo={cfg.hub_repo}, model={cfg.model_name}, source={cfg.hub_source}")
        model = torch.hub.load(cfg.hub_repo, cfg.model_name, pretrained=True, source=cfg.hub_source)
        model.to(self.device)
        model.eval()

        # remove classifier heads for common torchvision-like models
        # (best-effort; safe no-op if attributes not present)
        for attr in ("fc", "classifier", "head", "heads"):
            if hasattr(model, attr):
                try:
                    setattr(model, attr, torch.nn.Identity())
                    logger.info(f"Adjusted model head: set {attr}=Identity()")
                    break
                except Exception:
                    pass

        self.model = model

        # defaults (torchhub rarely exposes preprocess metadata)
        default_size = 224
        default_mean = (0.485, 0.456, 0.406)
        default_std = (0.229, 0.224, 0.225)

        # Heuristic for dinov2: many use 518, but keep overrideable
        if "dinov2" in cfg.model_name.lower() or "dinov2" in cfg.hub_repo.lower():
            default_size = 518

        self.input_size = _maybe_log_override("img_size", default_size, cfg.img_size)
        self.mean = _maybe_log_override("mean", default_mean, cfg.mean)
        self.std = _maybe_log_override("std", default_std, cfg.std)

        self._out_dim = self._infer_dim()

    def _infer_dim(self) -> int:
        assert self.model is not None
        x = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device)
        with torch.inference_mode():
            out = self.model(x)
            feat = _normalize_model_output(out)
        return int(feat.shape[1])

    def output_dim(self) -> int:
        return self._out_dim


class SSCDTorchScriptEmbedder(BaseEmbedder):
    def __init__(self, cfg: CBIRConfig):
        super().__init__(cfg)
        if not cfg.sscd_torchscript_path:
            raise ValueError("SSCD backend requires --sscd-torchscript-path")

        p = Path(cfg.sscd_torchscript_path)

        if not p.exists():
            # Auto-download if missing
            url = DEFAULT_SSCD_URL
            logger.warning(f"SSCD model not found at: {p}")
            logger.info(f"Auto-downloading SSCD TorchScript from: {url}")
            download_file(url, p)

        if not p.exists():
            raise FileNotFoundError(f"SSCD TorchScript model still not found after download: {p}")

        if not p.exists():
            raise FileNotFoundError(f"SSCD TorchScript model not found: {p}")

        logger.info(f"Loading SSCD TorchScript model: {p}")
        model = torch.jit.load(str(p), map_location=self.device)
        model.eval()
        self.model = model

        # SSCD README commonly uses 288 (small-edge) or 320 square.
        # You requested only resize or letterpad; we follow that with a square target size.
        default_size = 288
        default_mean = (0.485, 0.456, 0.406)
        default_std = (0.229, 0.224, 0.225)

        self.input_size = _maybe_log_override("img_size", default_size, cfg.img_size)
        self.mean = _maybe_log_override("mean", default_mean, cfg.mean)
        self.std = _maybe_log_override("std", default_std, cfg.std)

        self._out_dim = self._infer_dim()

    def _infer_dim(self) -> int:
        assert self.model is not None
        x = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device)
        with torch.inference_mode():
            out = self.model(x)
            feat = _normalize_model_output(out)
        return int(feat.shape[1])

    def output_dim(self) -> int:
        return self._out_dim


def build_embedder(cfg: CBIRConfig) -> BaseEmbedder:
    b = cfg.backend.lower()
    if b == "timm":
        return TimmEmbedder(cfg)
    if b == "torchhub":
        return TorchHubEmbedder(cfg)
    if b == "sscd":
        return SSCDTorchScriptEmbedder(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend}")


# -----------------------------
# CBIR Engine
# -----------------------------
@dataclass
class Match:
    path: str
    score: float
    rank: int


class PanelCBIR:
    """
    Importable CBIR engine.

    Typical flow:
      cbir = PanelCBIR(cfg)
      cbir.index_folder("panels/")
      results = cbir.search("panels/q.png", topk=6)
      cbir.save_viz_grid("panels/q.png", results, "out/viz.png")
    """
    def __init__(self, cfg: CBIRConfig):
        self.cfg = cfg
        if cfg.deterministic and cfg.seed is not None:
            _set_determinism(int(cfg.seed))

        self.embedder = build_embedder(cfg)
        logger.info(f"Embedder ready: backend={cfg.backend}, dim={self.embedder.output_dim()}, {self.embedder.preprocess_info()}")

        self.index_paths: List[Path] = []
        self.index_embs: Optional[np.ndarray] = None  # (N,D)

    def index_folder(self, folder: str | Path, recursive: bool = True, exts: Optional[Sequence[str]] = None) -> None:
        paths = discover_images(Path(folder), recursive=recursive, exts=exts)
        logger.info(f"Discovered {len(paths)} images in: {folder}")
        self.index_images(paths)

    def index_images(self, paths: Sequence[str | Path]) -> None:
        paths2 = [Path(p) for p in paths]
        if len(paths2) == 0:
            raise ValueError("No images to index.")
        self.index_paths = paths2

        # embed (and keep only successfully embedded images)
        # Note: embed_paths skips unreadable images; we must keep alignment.
        # We'll embed in a robust loop to maintain correct path->embedding mapping.
        good_paths: List[Path] = []
        tensors: List[torch.Tensor] = []

        # We will reuse embedder.preprocess and run batching ourselves to preserve mapping.
        model = self.embedder.model
        assert model is not None
        model.eval()

        device = self.embedder.device
        bs = max(1, int(self.cfg.batch_size))
        D = self.embedder.output_dim()

        all_embs: List[np.ndarray] = []

        pbar = tqdm(range(0, len(paths2), bs), desc="Indexing", unit="batch")
        for i in pbar:
            batch_paths = paths2[i : i + bs]
            batch_imgs: List[torch.Tensor] = []
            batch_ok_paths: List[Path] = []

            for p in batch_paths:
                im = _safe_load_image(p)
                if im is None:
                    continue
                batch_imgs.append(self.embedder._preprocess_one(im))
                batch_ok_paths.append(p)

            if not batch_imgs:
                continue

            x = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
            use_amp = bool(self.cfg.fp16 and device.type == "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                with torch.inference_mode():
                    out = model(x)
                    feat = _normalize_model_output(out)

            feat_np = feat.float().cpu().numpy().astype(np.float32)
            feat_np = _l2_normalize(feat_np)

            good_paths.extend(batch_ok_paths)
            all_embs.append(feat_np)

        if not good_paths:
            raise RuntimeError("No images could be embedded (all failed to load?).")

        embs = np.concatenate(all_embs, axis=0).astype(np.float32)
        if embs.shape[1] != D:
            logger.warning(f"Embedding dim mismatch? expected {D}, got {embs.shape[1]}")

        self.index_paths = good_paths
        self.index_embs = embs
        logger.info(f"Index built: N={len(self.index_paths)}, D={self.index_embs.shape[1]}")

    def search(self, query_path: str | Path, topk: int = 6) -> List[Match]:
        if self.index_embs is None or not self.index_paths:
            raise RuntimeError("Index is empty. Call index_folder/index_images first.")

        qpath = Path(query_path)
        im = _safe_load_image(qpath)
        if im is None:
            raise RuntimeError(f"Query image cannot be read: {qpath}")

        # embed query
        model = self.embedder.model
        assert model is not None
        device = self.embedder.device

        x = self.embedder._preprocess_one(im).unsqueeze(0).to(device)
        use_amp = bool(self.cfg.fp16 and device.type == "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            with torch.inference_mode():
                out = model(x)
                feat = _normalize_model_output(out)

        q = feat.float().cpu().numpy().astype(np.float32)
        q = _l2_normalize(q)[0]  # (D,)

        # cosine since all normalized
        scores = (self.index_embs @ q).astype(np.float32)  # (N,)

        # rank
        k = min(int(topk), len(scores))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        results: List[Match] = []
        for r, j in enumerate(idx, start=1):
            results.append(Match(path=str(self.index_paths[int(j)]), score=float(scores[int(j)]), rank=r))
        return results

    def rank_all(self, topk: int = 6) -> Dict[str, List[Match]]:
        if self.index_embs is None or not self.index_paths:
            raise RuntimeError("Index is empty. Call index_folder/index_images first.")

        embs = self.index_embs
        N = embs.shape[0]
        k = min(int(topk), max(0, N - 1))
        out: Dict[str, List[Match]] = {}

        pbar = tqdm(range(N), desc="Rank-all", unit="img")
        for i in pbar:
            q = embs[i]  # (D,)
            scores = (embs @ q).astype(np.float32)
            scores[i] = -np.inf  # no self-match

            if k <= 0:
                out[str(self.index_paths[i])] = []
                continue

            idx = np.argpartition(-scores, kth=k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]

            matches: List[Match] = []
            for r, j in enumerate(idx, start=1):
                matches.append(Match(path=str(self.index_paths[int(j)]), score=float(scores[int(j)]), rank=r))
            out[str(self.index_paths[i])] = matches

        return out

    def save_viz_grid(
        self,
        query_path: str | Path,
        matches: Sequence[Match],
        out_path: str | Path,
        thumb: int = 224,
        max_k: int = 6,
        title: Optional[str] = None,
    ) -> None:
        qpath = Path(query_path)
        out_path = Path(out_path)
        _ensure_dir(out_path.parent)

        # load images
        qimg0 = _safe_load_image(qpath)
        if qimg0 is None:
            logger.warning(f"Cannot visualize query (failed load): {qpath}")
            return
        qimg = _to_rgb_pil(qimg0, grayscale=False).copy()

        k = min(max_k, len(matches))
        cands: List[Tuple[Image.Image, str]] = []

        for m in matches[:k]:
            p = Path(m.path)
            im0 = _safe_load_image(p)
            if im0 is None:
                continue
            im = _to_rgb_pil(im0, grayscale=False).copy()
            cands.append((im, f"#{m.rank}  {m.score:.4f}"))

        # grid layout: query + topK
        n_items = 1 + len(cands)
        cols = min(4, n_items)  # simple default
        rows = int(math.ceil(n_items / cols))

        pad = 8
        label_h = 28
        cell_w = thumb
        cell_h = thumb + label_h

        grid_w = cols * cell_w + (cols + 1) * pad
        grid_h = rows * cell_h + (rows + 1) * pad + (label_h if title else 0)

        canvas = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)

        # font (best-effort)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        y0 = pad
        if title:
            draw.text((pad, y0), title, fill=(240, 240, 240), font=font)
            y0 += label_h

        def paste_cell(idx: int, img: Image.Image, label: str) -> None:
            r = idx // cols
            c = idx % cols
            x = pad + c * (cell_w + pad)
            y = y0 + pad + r * (cell_h + pad)

            # thumbnail
            im = img.copy()
            im.thumbnail((thumb, thumb))
            # center in thumb box
            bx = x + (thumb - im.size[0]) // 2
            by = y + (thumb - im.size[1]) // 2
            canvas.paste(im, (bx, by))

            # label
            draw.text((x, y + thumb + 4), label, fill=(240, 240, 240), font=font)

        paste_cell(0, qimg, "QUERY")
        for i, (im, lab) in enumerate(cands, start=1):
            paste_cell(i, im, lab)

        canvas.save(out_path)
        logger.info(f"Saved visualization: {out_path}")


# -----------------------------
# CLI
# -----------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="panel-cbir",
        description="CBIR for scientific panel images (timm/torchhub/sscd) — single script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folder", type=str, required=True, help="Folder containing candidate images to index.")
    p.add_argument("--recursive", action="store_true", help="Recursively search for images in folder.")
    p.add_argument("--exts", type=str, default=",".join(sorted(IMG_EXTS_DEFAULT)), help="Comma-separated extensions.")
    p.add_argument("--query", type=str, default=None, help="Query image path. If omitted, rank-all for each image.")

    p.add_argument("--outdir", type=str, default=None, help="Output directory. If omitted, auto-created under ./out/.")
    p.add_argument("--topk", type=int, default=6, help="Top-K matches to return.")
    p.add_argument("--viz", action="store_true", help="Save visualization grid images.")
    p.add_argument("--viz-thumb", type=int, default=224, help="Thumbnail size for visualization grids.")

    # Model / embedding
    p.add_argument("--backend", type=str, default="timm", choices=["timm", "torchhub", "sscd"], help="Embedder backend.")
    p.add_argument("--model", type=str, default="resnet50", help="Model name (timm or torchhub).")
    p.add_argument("--hub-repo", type=str, default="pytorch/vision", help="torch.hub repo (torchhub backend).")
    p.add_argument("--hub-source", type=str, default="github", help="torch.hub source.")
    p.add_argument("--sscd-torchscript-path", type=str, default="resources/models/sscd_disc_mixup.torchscript.pt", help="Path to SSCD TorchScript model file.")

    # Preprocess
    p.add_argument("--img-size", type=int, default=None, help="Override input size (square). If omitted, use model default.")
    p.add_argument("--resize-mode", type=str, default="letterpad", choices=["resize", "letterpad"], help="Resize policy.")
    p.add_argument("--grayscale", action="store_true", help="Convert images to grayscale then to RGB (useful for blots).")
    p.add_argument("--mean", type=str, default=None, help="Override mean as 'r,g,b' in 0..1.")
    p.add_argument("--std", type=str, default=None, help="Override std as 'r,g,b' in 0..1.")

    # Runtime
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda/cuda:0...")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for embedding.")
    p.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA.")
    p.add_argument("--seed", type=int, default=0, help="Seed for deterministic behavior.")
    p.add_argument("--no-deterministic", action="store_true", help="Disable deterministic settings.")
    p.add_argument("--no-save-args", action="store_true", help="Do not save CLI args JSON in outdir.")

    return p.parse_args(argv)


def _parse_triplet(s: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if s is None:
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {s}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    folder = Path(args.folder)
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    outdir = Path(args.outdir) if args.outdir else Path("out") / f"panel_cbir_{_now_tag()}"
    _ensure_dir(outdir)

    if not args.no_save_args:
        with open(outdir / "run_args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Saved CLI args: {outdir / 'run_args.json'}")

    cfg = CBIRConfig(
        img_size=args.img_size,
        resize_mode=args.resize_mode,
        grayscale=bool(args.grayscale),
        device=args.device,
        batch_size=int(args.batch_size),
        fp16=bool(args.fp16),
        seed=int(args.seed),
        deterministic=not bool(args.no_deterministic),
        mean=_parse_triplet(args.mean),
        std=_parse_triplet(args.std),
        backend=args.backend,
        model_name=args.model,
        hub_repo=args.hub_repo,
        hub_source=args.hub_source,
        sscd_torchscript_path=args.sscd_torchscript_path,
    )

    logger.info(f"Running CBIR: backend={cfg.backend}, model={cfg.model_name}, resize_mode={cfg.resize_mode}, device={cfg.device}")
    cbir = PanelCBIR(cfg)

    # discover + index
    paths = discover_images(folder, recursive=bool(args.recursive), exts=exts)
    logger.info(f"Detected images count: {len(paths)}")
    if len(paths) == 0:
        logger.error("No images found. Check --folder / --exts / --recursive.")
        return 2

    cbir.index_images(paths)

    viz_dir = outdir / "viz"
    if args.viz:
        _ensure_dir(viz_dir)

    if args.query:
        # single query mode
        q = Path(args.query)
        results = cbir.search(q, topk=int(args.topk))

        out_json = outdir / "ranking.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "query": str(q),
                    "topk": int(args.topk),
                    "results": [asdict(m) for m in results],
                    "config": asdict(cfg),
                },
                f,
                indent=2,
            )
        logger.info(f"Saved ranking: {out_json}")

        if args.viz:
            cbir.save_viz_grid(
                query_path=q,
                matches=results,
                out_path=viz_dir / f"query_{q.stem}_top{int(args.topk)}.png",
                thumb=int(args.viz_thumb),
                max_k=int(args.topk),
                title=f"{q.name} (top{int(args.topk)})",
            )

    else:
        # rank-all mode
        rankings = cbir.rank_all(topk=int(args.topk))
        out_jsonl = outdir / "rank_all.jsonl"

        with open(out_jsonl, "w", encoding="utf-8") as f:
            for qpath, matches in rankings.items():
                rec = {
                    "query": qpath,
                    "topk": int(args.topk),
                    "results": [asdict(m) for m in matches],
                }
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Saved rank-all JSONL: {out_jsonl}")

        if args.viz:
            # visualize each query
            for qpath, matches in tqdm(rankings.items(), desc="Saving viz", unit="img"):
                q = Path(qpath)
                safe_name = q.stem.replace(" ", "_")
                cbir.save_viz_grid(
                    query_path=q,
                    matches=matches,
                    out_path=viz_dir / f"{safe_name}_top{int(args.topk)}.png",
                    thumb=int(args.viz_thumb),
                    max_k=int(args.topk),
                    title=f"{q.name} (top{int(args.topk)})",
                )

    logger.info(f"Done. Output dir: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
