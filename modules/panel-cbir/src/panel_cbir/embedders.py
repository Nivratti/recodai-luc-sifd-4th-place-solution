from __future__ import annotations

"""Embedding backends for panel_cbir.

All embedders return L2-normalized float32 embeddings.

Supported backends
------------------
- **timm**: recommended for most use cases; good model coverage and preprocessing metadata.
- **torchhub**: flexible but may require internet access unless models are cached.
- **sscd**: uses a local TorchScript file (auto-download supported).

Performance notes
-----------------
- Set `CBIRConfig.batch_size` to the largest that fits memory.
- Set `CBIRConfig.num_workers>0` to parallelize image decode + preprocessing.
- For ranking many queries, embed queries in a batch and rank with a single matmul.
"""

import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from .config import CBIRConfig
from .types import ImageInput
from .utils import (
    default_item_id,
    direct_resize,
    is_pathlike,
    l2_normalize,
    letterpad_to_square,
    pil_to_tensor,
    safe_load_image_pil,
    to_rgb_pil,
)


DEFAULT_SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"


def download_file(url: str, dst: Path, timeout: int = 60) -> None:
    """Download `url` to `dst` (best-effort).

    Uses `wget` if available (often faster/more robust), falls back to urllib.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)

    wget = subprocess.run(["bash", "-lc", "command -v wget"], capture_output=True, text=True)
    if wget.returncode == 0:
        logger.info(f"Downloading with wget: {url} -> {dst}")
        subprocess.check_call(["wget", "-O", str(dst), url])
        return

    logger.info(f"Downloading with urllib: {url} -> {dst}")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)


def normalize_model_output(out: Any) -> torch.Tensor:
    """Turn a model output into a (B, D) tensor (best-effort)."""

    if isinstance(out, torch.Tensor):
        x = out
    elif isinstance(out, (list, tuple)) and len(out) > 0:
        x = out[0] if isinstance(out[0], torch.Tensor) else out[-1]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported model output tuple/list types: {[type(t) for t in out]}")
    elif isinstance(out, dict):
        for k in ("x", "feat", "features", "embeddings", "last_hidden_state", "pooler_output"):
            if k in out and isinstance(out[k], torch.Tensor):
                x = out[k]
                break
        else:
            tensors = [v for v in out.values() if isinstance(v, torch.Tensor)]
            if not tensors:
                raise TypeError(f"Unsupported model output dict: keys={list(out.keys())}")
            x = tensors[0]
    else:
        raise TypeError(f"Unsupported model output type: {type(out)}")

    # squeeze common (B,D,1,1) shapes
    if x.ndim == 4 and x.shape[-1] == 1 and x.shape[-2] == 1:
        x = x[:, :, 0, 0]

    # e.g. (B, T, D) -> take CLS token
    if x.ndim == 3:
        x = x[:, 0, :]

    if x.ndim != 2:
        x = x.view(x.shape[0], -1)

    return x


@dataclass(frozen=True)
class _PreprocessCfg:
    input_size: int
    resize_mode: str
    grayscale: bool
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


class _ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[Path], p: _PreprocessCfg):
        self.paths = list(paths)
        self.p = p

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        im = safe_load_image_pil(path)
        if im is None:
            return None

        img = to_rgb_pil(im, grayscale=self.p.grayscale)
        if self.p.resize_mode == "resize":
            img2 = direct_resize(img, self.p.input_size)
        elif self.p.resize_mode == "letterpad":
            img2 = letterpad_to_square(img, self.p.input_size)
        else:
            raise ValueError(f"resize_mode must be 'resize' or 'letterpad', got: {self.p.resize_mode}")

        t = pil_to_tensor(img2, self.p.mean, self.p.std)
        return str(path), t


def _collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], None
    paths, tensors = zip(*batch)
    return list(paths), torch.stack(list(tensors), dim=0)


class BaseEmbedder:
    def __init__(self, cfg: CBIRConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
        if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")

        self.model: Optional[torch.nn.Module] = None
        self.input_size: int = 224
        self.mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        self.std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def output_dim(self) -> int:
        raise NotImplementedError

    def preprocess_info(self) -> str:
        return (
            f"size={self.input_size}, resize_mode={self.cfg.resize_mode}, mean={self.mean}, std={self.std}, "
            f"grayscale={self.cfg.grayscale}"
        )

    def _infer_dim(self) -> int:
        assert self.model is not None
        x = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device)
        with torch.inference_mode():
            out = self.model(x)
            feat = normalize_model_output(out)
        return int(feat.shape[1])

    @torch.inference_mode()
    def embed_paths(self, paths: Sequence[Path], desc: str = "Embedding") -> tuple[list[str], np.ndarray]:
        """Embed a list of image paths.

        Returns (good_paths, embeddings).
        """

        assert self.model is not None, "Model not initialized"
        self.model.eval()

        if len(paths) == 0:
            return [], np.zeros((0, self.output_dim()), dtype=np.float32)

        p = _PreprocessCfg(
            input_size=self.input_size,
            resize_mode=self.cfg.resize_mode,
            grayscale=self.cfg.grayscale,
            mean=self.mean,
            std=self.std,
        )
        ds = _ImagePathDataset(paths, p)
        dl = DataLoader(
            ds,
            batch_size=max(1, int(self.cfg.batch_size)),
            shuffle=False,
            num_workers=max(0, int(self.cfg.num_workers)),
            pin_memory=bool(self.cfg.pin_memory and self.device.type == "cuda"),
            collate_fn=_collate_skip_none,
        )

        good_paths: List[str] = []
        embs: List[np.ndarray] = []

        use_amp = bool(self.cfg.fp16 and self.device.type == "cuda")
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

        for batch_paths, x in dl:
            if x is None:
                continue
            x = x.to(self.device, non_blocking=True)
            with autocast:
                out = self.model(x)
                feat = normalize_model_output(out)

            feat_np = feat.float().cpu().numpy().astype(np.float32, copy=False)
            feat_np = l2_normalize(feat_np)

            good_paths.extend(batch_paths)
            embs.append(feat_np)

        if not embs:
            return [], np.zeros((0, self.output_dim()), dtype=np.float32)

        return good_paths, np.concatenate(embs, axis=0).astype(np.float32)

@torch.inference_mode()
def embed_inputs(
    self,
    items: Sequence[ImageInput],
    *,
    ids: Optional[Sequence[str]] = None,
    desc: str = "Embedding",
) -> tuple[list[str], np.ndarray]:
    """Embed a list of images provided as paths, numpy arrays, or PIL images.

    Returns (good_ids, embeddings).

    Notes
    -----
    - When all inputs are paths, :meth:`embed_paths` is usually faster because it can
      use a multi-worker DataLoader for decode/preprocess.
    - For in-memory inputs, we process in-process (no extra worker processes) to
      avoid heavy pickling overhead.
    """
    if len(items) == 0:
        return [], np.zeros((0, self.output_dim()), dtype=np.float32)

    if ids is not None and len(ids) != len(items):
        raise ValueError(f"ids length mismatch: len(ids)={len(ids)} vs len(items)={len(items)}")

    # Fast path: all are path-like and user didn't provide custom ids
    if ids is None and all(is_pathlike(x) for x in items):
        paths = [Path(x) for x in items]  # type: ignore[arg-type]
        return self.embed_paths(paths, desc=desc)

    assert self.model is not None, "Model not initialized"
    self.model.eval()

    good_ids: list[str] = []
    embs: list[np.ndarray] = []

    bs = max(1, int(self.cfg.batch_size))
    batch_tensors: list[torch.Tensor] = []
    batch_ids: list[str] = []

    def flush_batch() -> None:
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors, dim=0).to(self.device, non_blocking=True)
        out = self.model(x)
        feat = normalize_model_output(out)
        feat = feat.detach().float().cpu().numpy()
        feat = l2_normalize(feat)
        embs.append(feat.astype(np.float32, copy=False))
        good_ids.extend(batch_ids)
        batch_tensors.clear()
        batch_ids.clear()

    for i, item in enumerate(items):
        im = safe_load_image_pil(item)
        if im is None:
            continue

        item_id = str(ids[i]) if ids is not None else default_item_id(item, i)

        pil = to_rgb_pil(im, grayscale=self.cfg.grayscale)
        if self.cfg.resize_mode == "resize":
            pil2 = direct_resize(pil, self.input_size)
        elif self.cfg.resize_mode == "letterpad":
            pil2 = letterpad_to_square(pil, self.input_size)
        else:
            raise ValueError(f"resize_mode must be 'resize' or 'letterpad', got: {self.cfg.resize_mode}")

        t = pil_to_tensor(pil2, self.mean, self.std)
        batch_tensors.append(t)
        batch_ids.append(item_id)

        if len(batch_tensors) >= bs:
            flush_batch()

    flush_batch()

    if len(good_ids) == 0:
        return [], np.zeros((0, self.output_dim()), dtype=np.float32)

    return good_ids, np.concatenate(embs, axis=0).astype(np.float32, copy=False)



class TimmEmbedder(BaseEmbedder):
    def __init__(self, cfg: CBIRConfig):
        super().__init__(cfg)
        try:
            import timm  # type: ignore
            from timm.data import resolve_data_config  # type: ignore
        except Exception as e:
            raise RuntimeError("timm backend requested but timm is not installed") from e

        import timm  # type: ignore
        from timm.data import resolve_data_config  # type: ignore

        logger.info(f"Loading timm model: {cfg.model_name} (pretrained=True)")
        model = timm.create_model(cfg.model_name, pretrained=True, num_classes=0, global_pool="avg")
        model.to(self.device)
        self.model = model

        dc = resolve_data_config({}, model=model)
        default_size = int(dc.get("input_size", (3, 224, 224))[1])
        default_mean = tuple(float(x) for x in dc.get("mean", (0.485, 0.456, 0.406)))
        default_std = tuple(float(x) for x in dc.get("std", (0.229, 0.224, 0.225)))

        self.input_size = int(cfg.img_size) if cfg.img_size is not None else default_size
        self.mean = cfg.mean if cfg.mean is not None else default_mean
        self.std = cfg.std if cfg.std is not None else default_std

        self._out_dim = self._infer_dim()

    def output_dim(self) -> int:
        return int(self._out_dim)


class TorchHubEmbedder(BaseEmbedder):
    def __init__(self, cfg: CBIRConfig):
        super().__init__(cfg)

        logger.info(f"Loading torch.hub model: repo={cfg.hub_repo}, model={cfg.model_name}, source={cfg.hub_source}")
        model = torch.hub.load(cfg.hub_repo, cfg.model_name, pretrained=True, source=cfg.hub_source)
        model.to(self.device)
        model.eval()

        for attr in ("fc", "classifier", "head", "heads"):
            if hasattr(model, attr):
                try:
                    setattr(model, attr, torch.nn.Identity())
                    logger.info(f"Adjusted model head: set {attr}=Identity()")
                    break
                except Exception:
                    pass

        self.model = model

        default_size = 224
        if "dinov2" in cfg.model_name.lower() or "dinov2" in cfg.hub_repo.lower():
            default_size = 518

        self.input_size = int(cfg.img_size) if cfg.img_size is not None else default_size
        self.mean = cfg.mean if cfg.mean is not None else (0.485, 0.456, 0.406)
        self.std = cfg.std if cfg.std is not None else (0.229, 0.224, 0.225)

        self._out_dim = self._infer_dim()

    def output_dim(self) -> int:
        return int(self._out_dim)


class SSCDTorchScriptEmbedder(BaseEmbedder):
    def __init__(self, cfg: CBIRConfig):
        super().__init__(cfg)

        if not cfg.sscd_torchscript_path:
            raise ValueError("SSCD backend requires sscd_torchscript_path")

        p = Path(cfg.sscd_torchscript_path)
        if not p.exists():
            logger.warning(f"SSCD model not found at: {p}")
            logger.info(f"Auto-downloading SSCD TorchScript from: {DEFAULT_SSCD_URL}")
            download_file(DEFAULT_SSCD_URL, p)

        if not p.exists():
            raise FileNotFoundError(f"SSCD TorchScript model not found: {p}")

        logger.info(f"Loading SSCD TorchScript model: {p}")
        model = torch.jit.load(str(p), map_location=self.device)
        model.eval()
        self.model = model

        self.input_size = int(cfg.img_size) if cfg.img_size is not None else 288
        self.mean = cfg.mean if cfg.mean is not None else (0.485, 0.456, 0.406)
        self.std = cfg.std if cfg.std is not None else (0.229, 0.224, 0.225)

        self._out_dim = self._infer_dim()

    def output_dim(self) -> int:
        return int(self._out_dim)


def build_embedder(cfg: CBIRConfig) -> BaseEmbedder:
    b = str(cfg.backend).lower()
    if b == "timm":
        return TimmEmbedder(cfg)
    if b == "torchhub":
        return TorchHubEmbedder(cfg)
    if b == "sscd":
        return SSCDTorchScriptEmbedder(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend}")
