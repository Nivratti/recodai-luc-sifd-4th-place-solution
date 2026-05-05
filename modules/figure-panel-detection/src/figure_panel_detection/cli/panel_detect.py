from __future__ import annotations

import argparse
from typing import List, Optional
from pathlib import Path
import sys

from ..core.types import RunConfig, VizConfig
from ..pipeline.panel_detector import PanelDetector


def _parse_list_int(v: Optional[List[str]]) -> Optional[List[int]]:
    if not v:
        return None
    out: List[int] = []
    for t in v:
        for part in str(t).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
    return out or None

def _norm_ext(x: str) -> str:
    x = str(x).strip().lower()
    return x if x.startswith(".") else f".{x}"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="panel-detect", description="Figure panel detection (YOLOv5 ONNX)")

    p.add_argument("--source", required=True, help="Input image file or directory (recursive by default)")
    p.add_argument("--out", required=True, help="Output directory")

    p.add_argument("--model", required=True, help="Path to YOLOv5 ONNX model")
    p.add_argument(
        "--names",
        default=None,
        help="Optional names JSON. If omitted, auto-load <model_stem>.json next to the ONNX model (required).",
    )
    p.add_argument(
        "--keep-classes",
        nargs="+",
        default=None,
        help=(
            "Output filter: keep only these classes in the kept bucket (accepts ids or class names). "
            "Names must match the model names mapping (case/space-insensitive). "
            "Different from --classes (model-side filter)."
        ),
    )
    p.add_argument("--keep-apply", choices=["both", "vis"], default="both")

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf-thres", type=float, default=0.40)
    p.add_argument("--iou-thres", type=float, default=0.40)
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--classes", nargs="+", default=None, help="Model-side class filter (numeric ids). Applies before --keep-classes.")
    p.add_argument("--agnostic-nms", action="store_true")
    p.add_argument("--providers", nargs="+", default=["CUDAExecutionProvider", "CPUExecutionProvider"], help="ONNXRuntime providers list (default: CUDAExecutionProvider CPUExecutionProvider)")
    p.add_argument("--infer-batch-size", type=int, default=1, help="Batch size for ONNX inference forward pass (default: 1)")

    p.add_argument("--no-save-txt", dest="save_txt", action="store_false", help="Disable saving YOLO .txt labels")
    p.add_argument("--no-save-conf", dest="save_conf", action="store_false", help="Omit conf from labels")

    p.add_argument("--save-vis-img", action="store_true")
    p.add_argument(
        "--save-vis-mode",
        choices=["all_except_no_objects", "all", "kept_only"],
        default="all_except_no_objects",
        help="Which buckets get vis images when --save-vis-img is set. "
            "Default: all_except_no_objects (skips no_objects).",
    )
    p.add_argument("--copy-images", action="store_true")
    p.add_argument(
        "--copy-images-mode",
        choices=["all", "kept_only"],
        default="all",
        help="Which buckets get copied originals when --copy-images is set. "
            "Default: all. Use kept_only to reduce disk.",
    )

    # Crops
    p.add_argument("--save-crop", dest="save_crop", action="store_true", help="Save cropped detections under crops/")
    p.add_argument(
        "--crop-layout-mode",
        choices=["class", "image"],
        default="class",
        help="Crop layout: class (default, crops/<class>/...) or image (crops/<image_folder>/...)",
    )
    p.add_argument(
        "--crop-include-ignored",
        action="store_true",
        help="When using keep-classes, also save crops for ignored bucket (default: off)",
    )
    # Expansion controls (both)
    p.add_argument("--crop-pad-px", type=int, default=0, help="Expand crop by N pixels each side (default: 0)")
    p.add_argument("--crop-pad-pct", type=float, default=0.0, help="Expand crop by percentage of box size (default: 0.0)")
    p.add_argument(
        "--crop-expand-mode",
        choices=["margin", "context"],
        default="margin",
        help="Crop expansion behavior: margin (default) or context (avoid expanding into neighboring boxes)",
    )
    p.add_argument(
        "--crop-context-gap-px",
        type=int,
        default=0,
        help="In context mode, keep this many pixels away from neighboring boxes (default: 0)",
    )
    # Ranking (class mode)
    p.add_argument(
        "--crop-batch-size",
        type=int,
        default=None,
        help="Batch size within each class after ranking (default: uses --layout-batch-size)",
    )
    p.add_argument(
        "--crop-score-mode",
        choices=["area_conf", "area", "conf"],
        default="area_conf",
        help="Crop ranking score (default: area_conf = area_px * conf)",
    )
    p.add_argument(
        "--crop-score-area-exp",
        type=float,
        default=1.0,
        help="Area exponent in score (default: 1.0). Score=(area_px**area_exp)*(conf**conf_exp).",
    )
    p.add_argument(
        "--crop-score-conf-exp",
        type=float,
        default=1.0,
        help="Conf exponent in score (default: 1.0). Score=(area_px**area_exp)*(conf**conf_exp).",
    )
    p.add_argument(
        "--crop-max-cache-images",
        type=int,
        default=8,
        help="How many source images to keep in-memory while writing crops (default: 8)",
    )


    # Output format
    p.add_argument("--crop-ext", choices=["png", "jpg", "jpeg"], default="png", help="Crop image format (default: png)")
    p.add_argument("--crop-jpg-quality", type=int, default=95, help="JPEG quality (default: 95)")
    p.add_argument("--crop-jpg-subsampling", choices=["444", "422", "420"], default="444", help="JPEG chroma subsampling (default: 444 = none)")

    p.add_argument("--split-by-detections", action="store_true")
    p.add_argument("--layout", choices=["preserve", "flat", "batch"], default="preserve")
    p.add_argument(
        "--layout-batch-size",
        "--layout_batch_size",
        dest="layout_batch_size",
        type=int,
        default=100,
        help="Batch size for output layout grouping when --layout batch (default: 100).",
    )
    
    p.add_argument("--sort-by-objects", action="store_true")
    p.add_argument(
        "--sort-mode",
        choices=["objects", "area_sum", "area_max", "conf_mean", "conf_max", "quality"],
        default="quality",
        help="Sorting strategy when --sort-by-objects is enabled (default: quality).",
    )
    # composite score weights (only used for sort-mode=quality)
    p.add_argument("--sort-w-count", type=float, default=1.0, help="Weight for log1p(count) (quality mode).")
    p.add_argument("--sort-w-area-sum", type=float, default=4.0, help="Weight for sum(box_area/image_area) (quality mode).")
    p.add_argument("--sort-w-area-max", type=float, default=1.0, help="Weight for max(box_area/image_area) (quality mode).")
    p.add_argument("--sort-w-conf-mean", type=float, default=2.0, help="Weight for mean(conf) (quality mode).")
    p.add_argument("--sort-w-conf-max", type=float, default=0.5, help="Weight for max(conf) (quality mode).")
    p.add_argument("--sort-w-nclasses", type=float, default=0.5, help="Weight for number of unique classes (quality mode).")

    p.add_argument("--meta-file", default=None, help="Mapping JSON output path (only used for --layout flat|batch)")
    p.add_argument("--backup-original-labels", action="store_true")
    p.add_argument("--backup-labels-dirname", default="labels_full")

    # Viz args
    p.add_argument("--color-map", type=str, default="", help="JSON string or path to color-map JSON")
    p.add_argument("--line-thickness", type=int, default=2)
    p.add_argument("--hide-labels", action="store_true")
    p.add_argument("--hide-conf", action="store_true")
    p.add_argument("--min-font-scale", type=float, default=0.35)
    p.add_argument("--max-font-scale", type=float, default=0.9)
    p.add_argument("--label-bg-alpha", type=float, default=0.80)
    p.add_argument("--label-pad", type=int, default=2)
    p.add_argument("--touch-tol", type=int, default=0)

    # label sizing + outside-gap controls
    p.add_argument("--label-max-width-ratio", type=float, default=0.70, help="Max label width as fraction of box width (default: 0.70)")
    p.add_argument("--label-gap-ratio-w", type=float, default=0.05, help="Outside-placement gap ratio (label width multiplier) (default: 0.05)")
    p.add_argument("--label-gap-ratio-h", type=float, default=0.90, help="Outside-placement gap ratio (label height multiplier) (default: 0.90)")

    p.add_argument("--dry-run", action="store_true", help="Validate inputs and show what would run (no inference, no outputs)")

    p.add_argument("--dedup", action="store_true", help="Remove near-duplicate boxes by IoU (postprocess after NMS)")
    p.add_argument("--dedup-iou", type=float, default=0.90, help="IoU threshold for considering boxes duplicates (default: 0.90)")
    p.add_argument("--dedup-merge", action="store_true", help="Merge duplicates using conf-weighted box average (YOLOv5-style)")
    p.add_argument("--dedup-class-agnostic", action="store_true", help="Deduplicate across classes (default: per-class)")

    return p


def handle_dry_run(args):
    # -------------------------
    # Dry-run: validate + count
    # -------------------------
    from ..io.discover import gather_images
    from ..pipeline.panel_detector import load_names_required

    model_p = Path(args.model)
    if not model_p.exists():
        raise SystemExit(f"[ERR] model not found: {model_p}")

    names_map, names_src = load_names_required(str(model_p), args.names)

    # Validate keep-classes early in dry-run as well
    try:
        from ..filtering.keep_classes import parse_keep_classes_tokens
        if args.keep_classes:
            keep_ids = parse_keep_classes_tokens(args.keep_classes, names_map) or []
            missing = [i for i in keep_ids if i not in names_map]
            if missing:
                available = ", ".join([f"{k}:{v}" for k, v in sorted(names_map.items(), key=lambda x: x[0])])
                raise SystemExit(
                    f"[ERR] --keep-classes includes invalid id(s): {missing}\n"
                    f"      Available model classes: {available}\n"
                )
    except SystemExit:
        raise
    except Exception as e:
        print(f"[dry-run] keep-classes validation skipped: {e}")

    src = Path(args.source)
    files = gather_images(src, recursive=True)
    n = len(files)

    print("[dry-run] inputs validated")
    print(f"[dry-run] model: {model_p}")
    print(f"[dry-run] names_source: {names_src} ({len(names_map)} classes)")
    print(f"[dry-run] source: {src} (recursive=True)")
    print(f"[dry-run] found_images: {n}")
    print(f"[dry-run] out_dir: {args.out}")

    if n:
        print("[dry-run] sample files:")
        for p in files[: min(5, n)]:
            print(f"  - {p}")

    # ORT environment info (best-effort, does not create a session)
    try:
        import onnxruntime as ort  # type: ignore
        print(f"[dry-run] ort_device: {ort.get_device()}")
        print(f"[dry-run] ort_available_providers: {ort.get_available_providers()}")
    except Exception as e:
        print(f"[dry-run] onnxruntime not available: {e}")

    print("[dry-run] done (no inference performed)")


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.dry_run:
        handle_dry_run(args)
        return

    # -------------------------
    # CLI sanity checks / warnings
    # -------------------------
    layout_norm = str(args.layout).strip().lower()

    if args.sort_by_objects and layout_norm == "preserve":
        print("[WARN] sort-by-objects ignored because layout=preserve (use --layout flat or --layout batch).", file=sys.stderr)

    if args.meta_file and layout_norm == "preserve":
        print("[WARN] --meta-file only matters in --layout flat|batch (ignored for preserve).", file=sys.stderr)

    # Early validation: fail fast if --keep-classes has unknown names or out-of-range ids
    if args.keep_classes:
        from ..pipeline.panel_detector import load_names_required
        from ..filtering.keep_classes import parse_keep_classes_tokens

        names_map, _ = load_names_required(args.model, args.names)
        keep_ids = parse_keep_classes_tokens(args.keep_classes, names_map) or []

        # parse_keep_classes_tokens validates unknown *names*.
        # Also validate numeric ids are within the model's id set.
        missing_ids = [i for i in keep_ids if i not in names_map]
        if missing_ids:
            available = ", ".join([f"{k}:{v}" for k, v in sorted(names_map.items(), key=lambda x: x[0])])
            raise SystemExit(
                f"[ERR] Invalid numeric class id(s) in --keep-classes: {missing_ids}\n"
                f"      Available model classes: {available}\n"
                "      Tip: use class names (recommended) or valid numeric ids."
            )

    viz = VizConfig(
        color_map=args.color_map,
        line_thickness=args.line_thickness,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
        min_font_scale=args.min_font_scale,
        max_font_scale=args.max_font_scale,
        label_max_width_ratio=args.label_max_width_ratio,
        label_pad=args.label_pad,
        label_bg_alpha=args.label_bg_alpha,
        touch_tol=args.touch_tol,
        label_gap_ratio_w=args.label_gap_ratio_w,
        label_gap_ratio_h=args.label_gap_ratio_h,
    )

    cfg = RunConfig(
        model_onnx=args.model,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        classes=_parse_list_int(args.classes),
        agnostic_nms=args.agnostic_nms,
        providers=args.providers,
        infer_batch_size=int(args.infer_batch_size),

        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_vis_img=args.save_vis_img,
        save_vis_mode=args.save_vis_mode,
        copy_images=args.copy_images,
        copy_images_mode=args.copy_images_mode,

        save_crop=args.save_crop,
        crop_layout_mode=args.crop_layout_mode,
        crop_include_ignored=args.crop_include_ignored,
        crop_pad_px=args.crop_pad_px,
        crop_pad_pct=float(args.crop_pad_pct),
        crop_expand_mode=str(args.crop_expand_mode),
        crop_context_gap_px=int(args.crop_context_gap_px),
        crop_ext=_norm_ext(args.crop_ext),
        crop_jpg_quality=int(args.crop_jpg_quality),
        crop_jpg_subsampling=str(args.crop_jpg_subsampling),
        crop_batch_size=(int(args.crop_batch_size) if args.crop_batch_size is not None else int(args.layout_batch_size)),
        crop_score_mode=str(args.crop_score_mode),
        crop_score_area_exp=float(args.crop_score_area_exp),
        crop_score_conf_exp=float(args.crop_score_conf_exp),
        crop_max_cache_images=int(args.crop_max_cache_images),


        split_by_detections=args.split_by_detections,
        keep_classes_tokens=args.keep_classes,
        keep_apply=args.keep_apply,
        backup_original_labels=args.backup_original_labels,
        backup_labels_dirname=args.backup_labels_dirname,
        layout=args.layout,
        layout_batch_size=int(args.layout_batch_size),
        sort_by_objects=args.sort_by_objects,
        sort_mode=args.sort_mode,
        sort_w_count=float(args.sort_w_count),
        sort_w_area_sum=float(args.sort_w_area_sum),
        sort_w_area_max=float(args.sort_w_area_max),
        sort_w_conf_mean=float(args.sort_w_conf_mean),
        sort_w_conf_max=float(args.sort_w_conf_max),
        sort_w_nclasses=float(args.sort_w_nclasses),
        meta_file=args.meta_file,
        names=args.names,
        viz=viz,
    )

    runner = PanelDetector(cfg)
    res = runner.run(args.source, args.out)
    print(res)


if __name__ == "__main__":  # pragma: no cover
    main()