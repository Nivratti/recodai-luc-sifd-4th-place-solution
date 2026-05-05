"""Configuration objects for panel_cbir."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CBIRConfig:
    """Configuration for :class:`panel_cbir.PanelCBIR`."""

    # Preprocess
    img_size: Optional[int] = None            # if None: use model default (timm) or fallback 224
    resize_mode: str = "letterpad"           # "resize" | "letterpad"
    grayscale: bool = False                  # useful for blots

    # Runtime (embedding)
    device: str = "cpu"                      # "cpu" | "cuda" | "cuda:0" ...
    batch_size: int = 32
    fp16: bool = False                       # autocast fp16 for embedding (CUDA only)
    seed: Optional[int] = 0
    deterministic: bool = True

    # Data loading
    num_workers: int = 0                     # DataLoader workers; 0 = main process
    pin_memory: bool = True                  # effective when using CUDA

    # Normalization overrides (if None, use model defaults when possible)
    mean: Optional[Tuple[float, float, float]] = None
    std: Optional[Tuple[float, float, float]] = None

    # Embedder selection
    backend: str = "timm"                    # "timm" | "torchhub" | "sscd"
    model_name: str = "resnet50"             # timm model name or torchhub model name
    hub_repo: str = "pytorch/vision"         # torchhub repo, e.g. "facebookresearch/dinov2"
    hub_source: str = "github"               # torch.hub.load source

    # SSCD TorchScript
    sscd_torchscript_path: Optional[str] = None

    # Scoring / ranking
    score_device: Optional[str] = None       # if None: defaults to `device`
    score_fp16: bool = False                 # matmul in fp16 (CUDA only). Scores returned as float32.
    score_chunk_size: int = 0                # 0 = auto. If >0, compute scores in blocks.
