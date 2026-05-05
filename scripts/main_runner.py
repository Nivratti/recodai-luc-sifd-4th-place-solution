import argparse
from pathlib import Path

from loguru import logger

from recodai_sifd.utils.model_assets import ensure_panel_detector_model
from figure_panel_detection import FigurePanelDetector
from recodai_sifd.pipeline.figure_kind import classify_figure_kind
from recodai_sifd.pipeline.figure_pipeline import FigurePipeline, FigurePipelineConfig
from recodai_sifd.pipeline.mask_fusion import MaskFusionConfig
from recodai_sifd.pipeline.reuse_detection import ReusePruningConfig, ReuseSavePolicy
from recodai_sifd.utils.image_io import resolve_input_images, read_image_pil_rgb
from recodai_sifd.utils.submission_eval import SubmissionWriter


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run panel detection -> reuse detection -> grouping.")

    p.add_argument(
        "--input",
        dest="input_path",
        default="resources/samples",
        help="Single image file OR a folder of images",
    )
    p.add_argument("--out", default="runs/sifd/v1", help="Output folder (e.g., runs/sifd/v1)")

    # Panel detector
    p.add_argument(
        "--model",
        default="models/panel_detector/model_4_class.onnx",
        help=(
            "Path to panel detector .onnx model. If the model or class JSON is missing, "
            "the public release files are downloaded automatically."
        ),
    )
    p.add_argument("--conf-thres", type=float, default=0.4, help="Confidence threshold")
    p.add_argument("--iou-thres", type=float, default=0.4, help="IoU threshold for NMS")

    # Dev toggles
    p.add_argument("--debug", action="store_true", help="Verbose logs / keep extra artifacts")
    p.add_argument("--max-images", type=int, default=None, help="Process only first N images")

    # Submission + eval (local)
    p.add_argument("--submission-out", default=None, help="Where to write submission.csv (default: <out>/submission.csv)")
    p.add_argument(
        "--min-pixels",
        type=int,
        default=10,
        help="Filter out predicted instances smaller than this before RLE encoding (dev: helps remove noise)",
    )

    p.add_argument(
        "--gt-mask-dir",
        default=None,
        help="Optional GT mask folder for LOCAL evaluation (dev only). Mask filename must be <case_id>.png",
    )
    p.add_argument("--gt-channel-axis", type=int, default=0, choices=[-1, 0])  # -1: (H,W,K), 0: (K,H,W)

    # -------------------------
    # NEW: reuse I/O policy (speedups that should NOT change accuracy)
    # -------------------------
    p.add_argument(
        "--reuse-save-artifacts",
        default="matches",
        choices=["all", "matches", "none"],
        help="Per-pair artifact saving policy. "
             "'all' (legacy) saves match+no_match folders for every pair; "
             "'matches' saves only matched pairs; "
             "'none' saves nothing per pair (still writes reuse_summary.json).",
    )
    p.add_argument(
        "--reuse-write-no-match-json",
        action="store_true",
        help="When --reuse-save-artifacts=matches, also write minimal pair.json files for no-match pairs "
             "(creates many folders; usually keep off).",
    )

    # Pair-level debug
    p.add_argument("--reuse-debug-pairs", action="store_true", help="Log every attempted pair (very verbose).")

    # -------------------------
    # NEW: reuse pruning (may affect accuracy if too aggressive)
    # -------------------------
    p.add_argument("--reuse-prune", action="store_true", help="Enable candidate pruning (CBIR/group/geometry).")

    p.add_argument("--reuse-prune-disable-cbir", action="store_true", help="Disable CBIR shortlist (if prune enabled).")
    p.add_argument("--reuse-prune-disable-grouping", action="store_true", help="Disable type grouping (if prune enabled).")
    p.add_argument("--reuse-prune-disable-geometry", action="store_true", help="Disable aspect/area filters (if prune enabled).")
    p.add_argument("--reuse-prune-disable-early-stop", action="store_true", help="Disable early-stop (if prune enabled).")

    p.add_argument("--reuse-group-mode", default="broad", choices=["none", "class", "broad"])
    p.add_argument(
        "--reuse-only-within-group",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, only compare panels within the same (broad/class) group.",
    )

    # CBIR settings
    p.add_argument("--reuse-cbir-topk", type=int, default=12, help="Top-K candidates per panel from CBIR (if enabled).")
    p.add_argument("--reuse-cbir-device", default="cuda", help="CBIR device (e.g., cuda, cuda:0, cpu).")
    p.add_argument("--reuse-cbir-backend", default="timm", help="CBIR backend (panel_cbir).")
    p.add_argument("--reuse-cbir-model", default="resnet50", help="CBIR model name (panel_cbir).")
    p.add_argument("--reuse-cbir-batch-size", type=int, default=64, help="CBIR batch size.")
    p.add_argument("--reuse-cbir-fp16", action=argparse.BooleanOptionalAction, default=True, help="CBIR embedding fp16.")
    p.add_argument("--reuse-cbir-score-fp16", action=argparse.BooleanOptionalAction, default=True, help="CBIR scoring fp16.")

    # Geometry
    p.add_argument("--reuse-aspect-log-tol", type=float, default=0.9, help="Aspect ratio log tolerance (geometry prune).")
    p.add_argument("--reuse-area-ratio-min", type=float, default=0.20, help="Min area ratio (geometry prune).")

    # Early stop
    p.add_argument(
        "--reuse-stop-no-match-streak",
        type=int,
        default=0,
        help="If >0, stop checking more candidates for a source panel after this many consecutive no-matches "
             "(only applies when pruning is active).",
    )
    p.add_argument(
        "--reuse-stop-matches-per-source",
        type=int,
        default=0,
        help="If >0, stop after finding this many matches for a single source panel (only in pruning mode).",
    )

    # Optional: fast stage (resizing) for matching
    p.add_argument(
        "--reuse-max-side-sum-fast",
        type=int,
        default=0,
        help="If >0, run a FAST matching stage on resized panels where (W+H)<=this value. "
             "This can speed up CPU matching but MAY reduce recall.",
    )
    p.add_argument(
        "--reuse-rerun-fullres-on-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If using --reuse-max-side-sum-fast, rerun full-res matching for pairs that match in fast stage "
             "(recommended).",
    )

    return p


def main() -> int:
    args = build_argparser().parse_args()

    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)

    debug = bool(args.debug)

    model_path = ensure_panel_detector_model(args.model)
    panel_detector = FigurePanelDetector(model_onnx=model_path)
    sub_writer = SubmissionWriter()

    # Save policy (I/O only; should not affect matching decision)
    save_policy = ReuseSavePolicy(
        artifacts=args.reuse_save_artifacts,
        write_no_match_pair_json=bool(args.reuse_write_no_match_json),
    )

    # Prune config:
    # SAFETY: if --reuse-prune is NOT set, we pass prune=None (true legacy behavior).
    prune_cfg = None
    if args.reuse_prune:
        stop_matches = int(args.reuse_stop_matches_per_source or 0)
        prune_cfg = ReusePruningConfig(
            enabled=True,
            enable_cbir=not bool(args.reuse_prune_disable_cbir),
            enable_grouping=not bool(args.reuse_prune_disable_grouping),
            enable_geometry=not bool(args.reuse_prune_disable_geometry),
            enable_early_stop=not bool(args.reuse_prune_disable_early_stop),
            group_mode=args.reuse_group_mode,
            only_within_group=bool(args.reuse_only_within_group),
            cbir_topk=int(args.reuse_cbir_topk),
            cbir_cfg=dict(
                device=str(args.reuse_cbir_device),
                backend=str(args.reuse_cbir_backend),
                model_name=str(args.reuse_cbir_model),
                batch_size=int(args.reuse_cbir_batch_size),
                fp16=bool(args.reuse_cbir_fp16),
                score_fp16=bool(args.reuse_cbir_score_fp16),
            ),
            aspect_ratio_log_tol=float(args.reuse_aspect_log_tol),
            area_ratio_min=float(args.reuse_area_ratio_min),
            stop_after_no_match_streak=int(args.reuse_stop_no_match_streak or 0),
            stop_after_matches_per_source=(None if stop_matches <= 0 else stop_matches),
            enable_fast_resize=bool(int(args.reuse_max_side_sum_fast or 0) > 0),
            max_side_sum_fast=(None if int(args.reuse_max_side_sum_fast or 0) <= 0 else int(args.reuse_max_side_sum_fast)),
            enable_fullres_rerun=bool(args.reuse_rerun_fullres_on_match),
            rerun_fullres_on_match=bool(args.reuse_rerun_fullres_on_match),
        )

    pipeline = FigurePipeline(
        panel_detector=panel_detector,
        cfg=FigurePipelineConfig(
            output_root=output_root,
            debug=debug,
            edge_margin_ratio=0.02,
            min_panel_area_ratio=0.0,
            min_matched_keypoints=20,
            min_pixels_pred=int(args.min_pixels),
            fusion=MaskFusionConfig(
                overlap_intra_thresh=0.30,
                iou_thresh=0.15,
                dilate_radius_px=2,
                min_pixels=int(args.min_pixels),
            ),
            reuse_save_policy=save_policy,
            reuse_prune=prune_cfg,
            reuse_debug_pairs=bool(args.reuse_debug_pairs),
        ),
        intra_model=None,
    )

    try:
        image_paths = resolve_input_images(args.input_path)
    except Exception as e:
        logger.error(f"Failed to resolve input images: {e}")
        return 2

    if not image_paths:
        logger.warning("No images found. Exiting.")
        return 0

    if args.max_images:
        image_paths = image_paths[: args.max_images]

    for image_index, image_path in enumerate(image_paths):
        figure_path = Path(image_path)
        figure_id = figure_path.stem

        figure_image = read_image_pil_rgb(image_path)  # PIL image (RGB)
        figure_size_wh = figure_image.size  # (W,H)

        panel_detections = panel_detector.predict(
            figure_image,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            keep_classes=["Blots", "Microscopy"],
        )

        if debug:
            logger.debug(f"[{image_index}] figure={figure_id} path={image_path}")
            for det in panel_detections.detections:
                logger.debug(f"  {det.class_name} conf={det.conf:.3f} xyxy={det.xyxy}")

        decision = classify_figure_kind(
            image_size_wh=figure_size_wh,
            detections=panel_detections.detections,
            edge_margin_ratio=0.02,
            min_panel_area_ratio=0.0,
        )

        if debug:
            logger.debug(f"[figure_kind] {figure_id}: {decision.kind} ({decision.reason})")

        result = pipeline.process_figure(
            figure_id=figure_id,
            figure_image=figure_image,
            figure_size_wh=figure_size_wh,
            panel_detections=panel_detections,
        )

        sub_writer.add(figure_id, result.pred_annotation)

        if debug:
            logger.debug(f"[result] {figure_id}: kind={result.kind}  meta={result.meta}")

    # write submission.csv
    out_csv = Path(args.submission_out) if args.submission_out else Path(output_root, "submission.csv")
    sub_writer.write_csv(out_csv)
    logger.info(f"Wrote submission CSV: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
