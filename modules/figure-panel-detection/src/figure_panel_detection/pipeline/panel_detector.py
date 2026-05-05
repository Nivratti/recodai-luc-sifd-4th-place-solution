from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import cv2
from tqdm import tqdm
import math
import numpy as np

from ..core.layout import bucket_order, decide_pads, make_out_rel
from ..core.types import InferItem, LayoutCfg, RunConfig
from ..cropping.crop_regions import CropConfig, save_crops, save_ranked_class_crops
from ..filtering.buckets import bucket_for_image, choose_det_sources
from ..filtering.keep_classes import filter_by_class_ids, parse_keep_classes_tokens
from ..io.discover import gather_images, rel_path_under
from ..io.writers import copy2_safe, save_labels_yolo, write_layout_mapping, write_run_config
from ..viz.render import render_detections
from ..yolo.onnx_predictor import OnnxPredictorConfig, YoloOnnxPredictor
from ..filtering.dedup import dedup_detections


def _det_for_sort(it: InferItem, using_keep: bool) -> Optional[np.ndarray]:
    # When keep-classes is active, kept bucket should be sorted using kept detections.
    if using_keep and it.bucket == "kept" and it.det_kept is not None:
        return it.det_kept
    return it.det_all


def _det_metrics(det: Optional[np.ndarray], im0_shape_hw: tuple[int, int]) -> dict:
    h, w = im0_shape_hw
    img_area = float(max(1, h * w))

    if det is None or len(det) == 0:
        return {
            "count": 0,
            "area_sum": 0.0,
            "area_max": 0.0,
            "conf_mean": 0.0,
            "conf_max": 0.0,
            "nclasses": 0,
        }

    d = det
    x1 = d[:, 0]
    y1 = d[:, 1]
    x2 = d[:, 2]
    y2 = d[:, 3]
    conf = d[:, 4]
    cls = d[:, 5].astype(int)

    areas = np.clip((x2 - x1), 0, None) * np.clip((y2 - y1), 0, None)
    area_sum = float(areas.sum() / img_area)
    area_max = float((areas.max() / img_area) if len(areas) else 0.0)

    conf_mean = float(conf.mean()) if len(conf) else 0.0
    conf_max = float(conf.max()) if len(conf) else 0.0
    nclasses = int(len(set(cls.tolist()))) if len(cls) else 0

    return {
        "count": int(len(d)),
        "area_sum": area_sum,
        "area_max": area_max,
        "conf_mean": conf_mean,
        "conf_max": conf_max,
        "nclasses": nclasses,
    }


def _quality_score(m: dict, cfg: RunConfig) -> float:
    return (
        cfg.sort_w_area_sum * m["area_sum"]
        + cfg.sort_w_area_max * m["area_max"]
        + cfg.sort_w_conf_mean * m["conf_mean"]
        + cfg.sort_w_conf_max * m["conf_max"]
        + cfg.sort_w_nclasses * m["nclasses"]
        + cfg.sort_w_count * math.log1p(m["count"])
    )


def _sort_value(it: InferItem, using_keep: bool, cfg: RunConfig) -> float:
    mode = str(cfg.sort_mode).strip().lower()

    # current behavior
    if mode == "objects":
        return float(it.sort_n)

    det = _det_for_sort(it, using_keep)
    m = _det_metrics(det, it.im0_shape_hw)

    if mode == "area_sum":
        return float(m["area_sum"])
    if mode == "area_max":
        return float(m["area_max"])
    if mode == "conf_mean":
        return float(m["conf_mean"])
    if mode == "conf_max":
        return float(m["conf_max"])
    if mode == "quality":
        return float(_quality_score(m, cfg))

    raise ValueError(f"Unknown sort_mode: {cfg.sort_mode}")


def _update_run_config_json(run_cfg_path: Path, extra: Dict[str, Any]) -> None:
    """
    Merge extra fields into run_config.json under key 'extra' (preferred).
    Safe even if the schema changes slightly.
    """
    try:
        data = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("extra"), dict):
            data["extra"].update(extra)
        else:
            data = {"extra": extra, **(data if isinstance(data, dict) else {})}
        run_cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:  # pragma: no cover
        print(f"[warn] failed to update run_config.json with extras: {e}", file=sys.stderr)


def _parse_names_payload(data: Any, src_path: Path) -> Dict[int, str]:
    """
    Accepts:
      - {"class_names": {...}}  (recommended)
      - {"names": {...}}        (alias)
      - {"0":"Blots", "1":"Microscopy", ...} (legacy/simple)
      - {"class_names": ["Blots", ...]} (list form)
    Returns: {int_id: str_name}
    """
    if isinstance(data, dict):
        # Preferred keys
        for key in ("class_names", "names"):
            if key in data:
                data = data[key]
                break

        # Now data could be dict or list
        if isinstance(data, list):
            return {i: str(v) for i, v in enumerate(data)}

        if isinstance(data, dict):
            out: Dict[int, str] = {}
            for k, v in data.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            if out:
                return out

    raise SystemExit(
        f"[ERR] Invalid names JSON format in: {src_path}\n"
        f"      Expected one of:\n"
        f"        {{\"class_names\": {{\"0\":\"Blots\", \"1\":\"Microscopy\", ...}}}}\n"
        f"        {{\"class_names\": [\"Blots\", \"Microscopy\", ...]}}\n"
        f"        {{\"names\": ...}}  (alias)\n"
        f"        {{\"0\":\"Blots\", \"1\":\"Microscopy\", ...}}  (legacy)\n"
    )


def load_names_required(model_onnx_path: str, names_arg: Optional[str]) -> tuple[Dict[int, str], Path]:
    """
    If --names is provided, load it.
    Else auto-load <model_stem>.json next to the ONNX model.
    If nothing found -> ERROR (strict requirement).
    """
    if names_arg:
        p = Path(names_arg)
        if not p.exists():
            raise SystemExit(f"[ERR] --names file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return _parse_names_payload(data, p), p

    model_p = Path(model_onnx_path)
    sidecar = model_p.with_suffix(".json")
    if not sidecar.exists():
        raise SystemExit(
            f"[ERR] Class names mapping not provided and sidecar JSON not found.\n"
            f"      Looked for: {sidecar}\n"
            f"      Create it with:\n"
            f"        {{\"class_names\": {{\"0\":\"Blots\", \"1\":\"Microscopy\", ...}}}}\n"
            f"      Or pass --names <path_to_names.json>."
        )

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    return _parse_names_payload(data, sidecar), sidecar


def _chunks(seq, n: int):
    n = max(1, int(n))
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


class PanelDetector:
    """High-level wrapper: ONNX predictor + keep-filter + sorting + writers + viz + crops."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

        # Strict: must exist via --names or sidecar next to model
        self.names, self.names_source = load_names_required(cfg.model_onnx, cfg.names)

        self.predictor = YoloOnnxPredictor(
            model_path=cfg.model_onnx,
            cfg=OnnxPredictorConfig(imgsz=cfg.imgsz, fp16=False, providers=cfg.providers),
        )

    def run(self, source: str, out: str) -> Dict[str, Any]:
        src = Path(source)
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # recursive is DEFAULT
        files = gather_images(src, recursive=True)
        if not files:
            raise SystemExit(f"No images found under: {src}")

        skipped_samples: List[Dict[str, Any]] = []
        skipped_by_reason: Dict[str, int] = {}
        skipped_cap = 1000  # store up to 1000 examples in run_config.json

        def record_skip(p: Path, reason: str) -> None:
            skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
            if len(skipped_samples) < skipped_cap:
                skipped_samples.append({"path": str(p), "reason": reason})
            # print(f"[warn] skipping file (reason={reason}): {p}", file=sys.stderr)

        # Keep-classes now works for names (since names are required)
        keep_ids = parse_keep_classes_tokens(self.cfg.keep_classes_tokens, self.names)
        using_keep = bool(keep_ids)
        keep_set = set(keep_ids or [])

        pad_file, pad_batch = decide_pads(total_files=len(files), layout_batch_size=self.cfg.layout_batch_size)
        layout_cfg = LayoutCfg(
            layout=self.cfg.layout,
            layout_batch_size=self.cfg.layout_batch_size,
            pad_file=pad_file,
            pad_batch=pad_batch
        )
        
        need_sort = bool(self.cfg.sort_by_objects) and (layout_cfg.layout in ("flat", "batch"))
        need_collect = need_sort or (layout_cfg.layout in ("flat", "batch"))

        runtime_info = {
            "names_source": str(self.names_source),
            "ort_device": self.predictor.get_device(),
            "ort_providers": self.predictor.get_providers(),
        }

        print(f"[info] names_source: {runtime_info['names_source']}")
        print(f"[info] ort_device: {runtime_info['ort_device']}")
        print(f"[info] ort_providers: {runtime_info['ort_providers']}")

        run_cfg_path = write_run_config(out_dir, self.cfg, extra={"total_images": len(files), **runtime_info})

        meta_items: List[Dict[str, Any]] = []
        meta_path = Path(self.cfg.meta_file) if (self.cfg.meta_file and need_collect) else (out_dir / "layout_mapping.json")
        write_meta = bool(need_collect)

        # crop outputs
        # - image mode: crops are written immediately; mapping rows collected in crop_maps
        # - class mode: detections + per-image context are collected, and crops are written once at the end in ranked order
        crop_maps: Dict[Path, List[Dict[str, Any]]] = {}  # used only when crop_layout_mode == 'image'
        crop_plans: Dict[Path, Dict[str, Any]] = {}       # used only when crop_layout_mode == 'class'
        crop_bases: Dict[Path, Path] = {}  # base crops dir -> base crops_temp dir (for atomic swap)


        if need_collect:
            items: List[InferItem] = []
            pbar = tqdm(total=len(files), desc="Infer", unit="img")

            for batch_paths in _chunks(files, self.cfg.infer_batch_size):
                ims: List[np.ndarray] = []
                ok_paths: List[Path] = []

                for p in batch_paths:
                    im0 = cv2.imread(str(p))
                    if im0 is None:
                        record_skip(p, "imread_failed")
                        pbar.update(1)
                        continue
                    ims.append(im0)
                    ok_paths.append(p)

                if not ok_paths:
                    continue

                dets_all = self.predictor.predict_batch_bgr(
                    ims,
                    conf_thres=self.cfg.conf_thres,
                    iou_thres=self.cfg.iou_thres,
                    max_det=self.cfg.max_det,
                    classes=self.cfg.classes,
                    agnostic_nms=self.cfg.agnostic_nms,
                )

                if self.cfg.dedup:
                    dedup_args = {
                        "iou_thres": self.cfg.dedup_iou, 
                        "merge": self.cfg.dedup_merge, 
                        "class_agnostic": self.cfg.dedup_class_agnostic
                    }
                    dets_all = [
                        dedup_detections(d, **dedup_args) if d is not None else None 
                        for d in dets_all
                    ]
                    
                for p, im0, det_all in zip(ok_paths, ims, dets_all):
                    det_kept = filter_by_class_ids(det_all, keep_set) if using_keep else None

                    n_all = int(0 if det_all is None else len(det_all))
                    n_kept = int(0 if det_kept is None else len(det_kept))
                    bucket = bucket_for_image(has_any_det=n_all > 0, has_kept_det=n_kept > 0, using_keep=using_keep)

                    sort_n = n_all
                    if using_keep and bucket == "kept":
                        sort_n = n_kept

                    rel_in = rel_path_under(p, src) if src.is_dir() else Path(p.name)
                    items.append(
                        InferItem(
                            im_path=p,
                            rel_in=rel_in,
                            bucket=bucket,
                            bucket_key=bucket,
                            n_all=n_all,
                            n_kept=n_kept,
                            sort_n=sort_n,
                            im0_shape_hw=im0.shape[:2],
                            det_all=det_all,
                            det_kept=det_kept,
                        )
                    )
                    pbar.update(1)

            pbar.close()

            buckets = bucket_order(using_keep)
            for b in buckets:
                subset = [it for it in items if it.bucket == b]
                if need_sort:
                    # Best-first sorting (descending score). Tie-breaker by path for stability.
                    subset.sort(
                        key=lambda it: (-_sort_value(it, using_keep=using_keep, cfg=self.cfg), str(it.rel_in))
                    )
                    
                for idx, it in enumerate(subset):
                    self._write_one(it, out_dir, src, layout_cfg, idx_in_bucket=idx, using_keep=using_keep, crop_maps=crop_maps, crop_plans=crop_plans, crop_bases=crop_bases)
                    if write_meta:
                        out_rel = make_out_rel(it.rel_in, it.im_path.suffix, layout_cfg, idx)
                        meta_items.append({"bucket": b, "src": str(it.rel_in), "out": str(out_rel), "n_all": it.n_all, "n_kept": it.n_kept})

            if write_meta:
                write_layout_mapping(meta_path, meta_items)

        else:
            # preserve layout without sorting: stream write
            pbar = tqdm(total=len(files), desc="Infer+write", unit="img")

            for batch_paths in _chunks(files, self.cfg.infer_batch_size):
                ims: List[np.ndarray] = []
                ok_paths: List[Path] = []

                for p in batch_paths:
                    im0 = cv2.imread(str(p))
                    if im0 is None:
                        record_skip(p, "imread_failed")
                        pbar.update(1)
                        continue
                    ims.append(im0)
                    ok_paths.append(p)

                if not ok_paths:
                    continue

                dets_all = self.predictor.predict_batch_bgr(
                    ims,
                    conf_thres=self.cfg.conf_thres,
                    iou_thres=self.cfg.iou_thres,
                    max_det=self.cfg.max_det,
                    classes=self.cfg.classes,
                    agnostic_nms=self.cfg.agnostic_nms,
                )

                if self.cfg.dedup:
                    dedup_args = {
                        "iou_thres": self.cfg.dedup_iou, 
                        "merge": self.cfg.dedup_merge, 
                        "class_agnostic": self.cfg.dedup_class_agnostic
                    }
                    dets_all = [
                        dedup_detections(d, **dedup_args) if d is not None else None 
                        for d in dets_all
                    ]

                for p, im0, det_all in zip(ok_paths, ims, dets_all):
                    det_kept = filter_by_class_ids(det_all, keep_set) if using_keep else None

                    n_all = int(0 if det_all is None else len(det_all))
                    n_kept = int(0 if det_kept is None else len(det_kept))
                    bucket = bucket_for_image(has_any_det=n_all > 0, has_kept_det=n_kept > 0, using_keep=using_keep)

                    rel_in = rel_path_under(p, src) if src.is_dir() else Path(p.name)
                    it = InferItem(
                        im_path=p,
                        rel_in=rel_in,
                        bucket=bucket,
                        bucket_key=bucket,
                        n_all=n_all,
                        n_kept=n_kept,
                        sort_n=n_all,
                        im0_shape_hw=im0.shape[:2],
                        det_all=det_all,
                        det_kept=det_kept,
                    )
                    self._write_one(it, out_dir, src, layout_cfg, idx_in_bucket=0, using_keep=using_keep, crop_maps=crop_maps, crop_plans=crop_plans, crop_bases=crop_bases)
                    pbar.update(1)

            pbar.close()

        # finalize crop mapping(s)
        
        if self.cfg.save_crop:
            mode = str(self.cfg.crop_layout_mode or "class").strip().lower().replace("-", "_")
            if mode == "image":
                # image-mode crops were written during inference
                for crop_root, rows in crop_maps.items():
                    if rows:
                        write_layout_mapping(crop_root / "mapping.json", rows)
            else:
                # class-mode crops: write once in ranked order -> crops_temp, then swap to crops
                crop_cfg = CropConfig(
                    pad_px=int(self.cfg.crop_pad_px),
                    pad_pct=float(self.cfg.crop_pad_pct),
                    expand_mode=str(self.cfg.crop_expand_mode),
                    context_gap_px=int(self.cfg.crop_context_gap_px),
                    ext=str(self.cfg.crop_ext),
                    jpg_quality=int(self.cfg.crop_jpg_quality),
                    jpg_subsampling=str(self.cfg.crop_jpg_subsampling),
                )

                # ensure fresh crops_temp roots
                for base, base_tmp in crop_bases.items():
                    if base_tmp.exists():
                        shutil.rmtree(base_tmp)
                    base_tmp.mkdir(parents=True, exist_ok=True)

                # write each crop_root into its corresponding crops_temp subtree
                for crop_root, plan in crop_plans.items():
                    dets = plan.get("detections") or []
                    if not dets:
                        continue

                    base = crop_root
                    rel_under = Path()
                    while base.name != "crops" and base.parent != base:
                        rel_under = Path(base.name) / rel_under
                        base = base.parent
                    if base.name != "crops":
                        base = crop_root.parent
                        rel_under = Path(crop_root.name)

                    base_tmp = crop_bases.get(base, base.with_name("crops_temp"))
                    temp_root = base_tmp / rel_under

                    save_ranked_class_crops(
                        temp_root=temp_root,
                        final_root=crop_root,
                        detections=list(dets),
                        image_ctx=dict(plan.get("image_ctx") or {}),
                        names=self.names,
                        cfg=crop_cfg,
                        batch_size=int(getattr(self.cfg, "crop_batch_size", self.cfg.layout_batch_size)),
                        score_mode=str(getattr(self.cfg, "crop_score_mode", "area_conf")),
                        score_area_exp=float(getattr(self.cfg, "crop_score_area_exp", 1.0)),
                        score_conf_exp=float(getattr(self.cfg, "crop_score_conf_exp", 1.0)),
                        max_cache_images=int(getattr(self.cfg, "crop_max_cache_images", 8)),
                    )

                # swap crops_temp -> crops per base dir
                for base, base_tmp in crop_bases.items():
                    if not base_tmp.exists():
                        continue
                    if base.exists():
                        shutil.rmtree(base)
                    os.replace(str(base_tmp), str(base))

        skipped_total = sum(skipped_by_reason.values())
        if skipped_total > 0:
            _update_run_config_json(
                Path(run_cfg_path),
                {
                    "skipped_total": skipped_total,
                    "skipped_by_reason": skipped_by_reason,
                    "skipped_samples": skipped_samples,
                },
            )
            print(f"[info] skipped files: {skipped_total} ({skipped_by_reason})", file=sys.stderr)

        return {
            "out_dir": str(out_dir),
            "run_config": str(run_cfg_path),
            "layout_mapping": str(meta_path) if (need_collect and write_meta) else None,
        }

    def _bucket_root(self, out_dir: Path, bucket: str) -> Path:
        return (out_dir / bucket) if self.cfg.split_by_detections else out_dir

    def _crop_root(self, bucket_root: Path, bucket: str, using_keep: bool) -> Path:
        """
        crops folder location:
          - split_by_detections: <bucket_root>/crops (bucket_root already bucket-specific)
          - no split + using_keep: <out>/crops/<bucket> to avoid mixing kept/ignored
          - no split + no keep: <out>/crops
        """
        root = bucket_root / "crops"
        if using_keep and (not self.cfg.split_by_detections):
            root = root / bucket
        return root

    def _select_det_for_crops(self, it: InferItem, using_keep: bool) -> Optional[Any]:
        if not self.cfg.save_crop:
            return None

        if using_keep:
            if it.bucket == "kept":
                return it.det_kept
            if it.bucket == "ignored" and self.cfg.crop_include_ignored:
                return it.det_all
            return None

        # no keep filtering -> crop all detections
        return it.det_all

    def _write_one(
        self,
        it: InferItem,
        out_dir: Path,
        src: Path,
        layout_cfg: LayoutCfg,
        idx_in_bucket: int,
        using_keep: bool,
        crop_maps: Dict[Path, List[Dict[str, Any]]],
        crop_plans: Dict[Path, Dict[str, Any]],
        crop_bases: Dict[Path, Path],
    ) -> None:

        def _is_kept_bucket(bucket: str) -> bool:
            # If using keep-classes, "kept" is the kept bucket.
            # If not using keep-classes, the closest equivalent is "has_objects".
            return bucket == ("kept" if using_keep else "has_objects")

        bucket_root = self._bucket_root(out_dir, it.bucket)

        labels_dir = bucket_root / "labels"
        vis_dir = bucket_root / "vis"
        images_dir = bucket_root / "images"

        out_img_rel = make_out_rel(it.rel_in, it.im_path.suffix, layout_cfg, idx_in_bucket)

        im0 = cv2.imread(str(it.im_path))
        if im0 is None:
            return

        det_txt, det_vis = choose_det_sources(
            bucket=it.bucket,
            using_keep=using_keep,
            keep_apply=self.cfg.keep_apply,
            det_all=it.det_all,
            det_kept=it.det_kept,
        )

        # labels
        lbl_path: Optional[Path] = None
        if self.cfg.save_txt:
            lbl_path = labels_dir / out_img_rel.with_suffix(".txt")
            if using_keep and it.bucket == "kept" and self.cfg.keep_apply == "both" and self.cfg.backup_original_labels:
                backup_dir = bucket_root / self.cfg.backup_labels_dirname
                backup_lbl = backup_dir / out_img_rel.with_suffix(".txt")
                save_labels_yolo(backup_lbl, it.det_all, it.im0_shape_hw, save_conf=self.cfg.save_conf, save_empty=True)

            save_labels_yolo(lbl_path, det_txt, it.im0_shape_hw, save_conf=self.cfg.save_conf, save_empty=True)

        # copy images (so mapping can prefer copied image path)
        # ---- COPY IMAGES (mode-aware) ----
        out_img_path: Optional[Path] = None
        do_copy = bool(self.cfg.copy_images)
        if do_copy:
            cmode = (getattr(self.cfg, "copy_images_mode", None) or "all")
            cmode = str(cmode).strip().lower().replace("-", "_")

            kept_bucket = "kept" if using_keep else "has_objects"
            if cmode == "all":
                pass
            elif cmode == "kept_only":
                if it.bucket != kept_bucket:
                    do_copy = False
            else:
                raise ValueError(f"Unknown copy_images_mode: {cmode}")

        if do_copy:
            out_img_path = images_dir / out_img_rel
            copy2_safe(it.im_path, out_img_path)

        # vis: if split-by-detections, skip vis for no_objects bucket
        # ---- VIS (mode-aware) ----
        do_vis = bool(self.cfg.save_vis_img)
        if do_vis:
            mode = (getattr(self.cfg, "save_vis_mode", None) or "all_except_no_objects")
            mode = str(mode).strip().lower().replace("-", "_")

            kept_bucket = "kept" if using_keep else "has_objects"

            if mode == "all":
                pass
            elif mode == "all_except_no_objects":
                if it.bucket == "no_objects":
                    do_vis = False
            elif mode == "kept_only":
                if it.bucket != kept_bucket:
                    do_vis = False
            else:
                raise ValueError(f"Unknown save_vis_mode: {mode}")

        if do_vis:
            im_vis = im0.copy()
            render_detections(
                im_vis,
                det_vis,
                names=self.names,
                color_map=self.cfg.viz.color_map,
                line_thickness=self.cfg.viz.line_thickness,
                hide_labels=self.cfg.viz.hide_labels,
                hide_conf=self.cfg.viz.hide_conf,
                min_font_scale=self.cfg.viz.min_font_scale,
                max_font_scale=self.cfg.viz.max_font_scale,
                label_max_width_ratio=self.cfg.viz.label_max_width_ratio,
                label_pad=self.cfg.viz.label_pad,
                label_bg_alpha=self.cfg.viz.label_bg_alpha,
                touch_tol=self.cfg.viz.touch_tol,
                label_gap_ratio_w=self.cfg.viz.label_gap_ratio_w,
                label_gap_ratio_h=self.cfg.viz.label_gap_ratio_h,
            )
            vis_path = vis_dir / out_img_rel
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), im_vis)

        
        # crops
        det_crop = self._select_det_for_crops(it, using_keep=using_keep)
        if det_crop is not None and len(det_crop) > 0:
            crop_root = self._crop_root(bucket_root=bucket_root, bucket=it.bucket, using_keep=using_keep)

            # mapping wants label path if exists (higher priority), else image path (prefer copied if exists)
            label_ref = str(lbl_path) if (lbl_path is not None and lbl_path.exists()) else None
            image_ref = str(out_img_path) if (out_img_path is not None) else str(it.im_path)
            source_ref = label_ref if label_ref is not None else image_ref

            obstacles = it.det_all
            obstacles_xyxy = obstacles[:, 0:4] if (obstacles is not None and len(obstacles) > 0) else None

            crop_cfg = CropConfig(
                pad_px=int(self.cfg.crop_pad_px),
                pad_pct=float(self.cfg.crop_pad_pct),
                expand_mode=str(self.cfg.crop_expand_mode),
                context_gap_px=int(self.cfg.crop_context_gap_px),
                ext=str(self.cfg.crop_ext),
                jpg_quality=int(self.cfg.crop_jpg_quality),
                jpg_subsampling=str(self.cfg.crop_jpg_subsampling),
            )

            mode = str(self.cfg.crop_layout_mode or "class").strip().lower().replace("-", "_")

            if mode == "image":
                # legacy behavior: write crops immediately (per-image layout)
                rows = save_crops(
                    im0_bgr=im0,
                    det_xyxy_conf_cls=det_crop,
                    crop_root=crop_root,
                    layout_mode=mode,
                    names=self.names,
                    rel_in=it.rel_in,
                    out_img_rel=out_img_rel,
                    source_ref=source_ref,
                    label_ref=label_ref,
                    image_ref=image_ref,
                    bucket=it.bucket,
                    cfg=crop_cfg,
                    obstacles_xyxy=obstacles_xyxy,
                )
                if rows:
                    crop_maps.setdefault(crop_root, []).extend(rows)
            else:
                # ranked class-mode: collect (dets + per-image context) and write once at the end
                # Register base crops dir swap: <...>/crops_temp -> <...>/crops
                base = crop_root
                rel_under = Path()
                while base.name != "crops" and base.parent != base:
                    rel_under = Path(base.name) / rel_under
                    base = base.parent
                if base.name != "crops":
                    # fallback: treat immediate parent as base
                    base = crop_root.parent
                    rel_under = Path(crop_root.name)

                base_temp = base.with_name("crops_temp")
                crop_bases[base] = base_temp

                plan = crop_plans.setdefault(crop_root, {"image_ctx": {}, "detections": []})
                img_key = str(it.im_path)

                if img_key not in plan["image_ctx"]:
                    plan["image_ctx"][img_key] = {
                        "im_path": it.im_path,
                        "rel_in": it.rel_in,
                        "out_img_rel": out_img_rel,
                        "bucket": it.bucket,
                        "label_ref": label_ref,
                        "image_ref": image_ref,
                        "source_ref": source_ref,
                        "obstacles_xyxy": obstacles_xyxy,
                    }

                # add detections
                for di in range(int(len(det_crop))):
                    x1, y1, x2, y2, conf, cls = det_crop[di].tolist()
                    area_px = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
                    plan["detections"].append(
                        {
                            "img_key": img_key,
                            "class_id": int(cls),
                            "conf": float(conf),
                            "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "area_px": float(area_px),
                            "det_index": int(di),
                        }
                    )
