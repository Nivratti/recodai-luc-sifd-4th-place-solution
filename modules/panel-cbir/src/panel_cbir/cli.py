from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger

from .config import CBIRConfig
from .engine import PanelCBIR
from .utils import discover_images, ensure_dir, IMG_EXTS_DEFAULT


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="panel-cbir",
        description="CBIR for scientific panel images (timm/torchhub/sscd)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--folder", type=str, required=True, help="Folder containing images to index.")
    p.add_argument("--recursive", action="store_true", help="Recursively search for images.")
    p.add_argument("--exts", type=str, default=",".join(sorted(IMG_EXTS_DEFAULT)), help="Comma-separated extensions.")

    p.add_argument("--query", type=str, default=None, help="Query image path. If omitted, rank-within for the folder.")
    p.add_argument("--topk", type=int, default=6, help="Top-K matches to return.")

    p.add_argument("--outdir", type=str, default=None, help="Output directory (JSON results).")
    p.add_argument("--index-cache", type=str, default=None, help="Optional path to a .npz cache for candidate embeddings.")

    # Model / embedding
    p.add_argument("--backend", type=str, default="timm", choices=["timm", "torchhub", "sscd"], help="Embedder backend.")
    p.add_argument("--model", type=str, default="resnet50", help="Model name (timm or torchhub).")
    p.add_argument("--hub-repo", type=str, default="pytorch/vision", help="torch.hub repo.")
    p.add_argument("--hub-source", type=str, default="github", help="torch.hub source.")
    p.add_argument("--sscd-torchscript-path", type=str, default=None, help="Path to SSCD TorchScript model file.")

    # Preprocess
    p.add_argument("--img-size", type=int, default=None, help="Override input size (square).")
    p.add_argument("--resize-mode", type=str, default="letterpad", choices=["resize", "letterpad"], help="Resize policy.")
    p.add_argument("--grayscale", action="store_true", help="Convert images to grayscale then to RGB.")

    # Runtime
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda/cuda:0...")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding.")
    p.add_argument("--fp16", action="store_true", help="Use fp16 autocast for embedding on CUDA.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for image decode/preprocess.")

    # Scoring
    p.add_argument("--score-device", type=str, default=None, help="Device for scoring matmul. Defaults to --device.")
    p.add_argument("--score-fp16", action="store_true", help="Use fp16 matmul for scoring on CUDA.")
    p.add_argument("--score-chunk-size", type=int, default=0, help="Chunk candidates during scoring to cap memory.")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    folder = Path(args.folder)
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    outdir = Path(args.outdir) if args.outdir else Path("out") / "panel_cbir"
    ensure_dir(outdir)

    cfg = CBIRConfig(
        img_size=args.img_size,
        resize_mode=args.resize_mode,
        grayscale=bool(args.grayscale),
        device=args.device,
        batch_size=int(args.batch_size),
        fp16=bool(args.fp16),
        num_workers=int(args.num_workers),
        backend=args.backend,
        model_name=args.model,
        hub_repo=args.hub_repo,
        hub_source=args.hub_source,
        sscd_torchscript_path=args.sscd_torchscript_path,
        score_device=args.score_device,
        score_fp16=bool(args.score_fp16),
        score_chunk_size=int(args.score_chunk_size),
    )

    cbir = PanelCBIR(cfg)

    paths = discover_images(folder, recursive=bool(args.recursive), exts=exts)
    if not paths:
        logger.error("No images found. Check --folder / --exts / --recursive.")
        return 2

    index = cbir.build_index(paths, cache_path=args.index_cache, reuse_cache=True)

    if args.query:
        results = cbir.search(index, args.query, topk=int(args.topk))
        out_json = outdir / "ranking.json"
        out_json.write_text(
            json.dumps({"query": str(args.query), "results": [m.__dict__ for m in results], "config": asdict(cfg)}, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved: {out_json}")
    else:
        rankings = cbir.rank_within_index(index, topk=int(args.topk), exclude_self=True)
        out_json = outdir / "rank_within.json"
        cbir.save_rankings_json(rankings, out_json)
        logger.info(f"Saved: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
