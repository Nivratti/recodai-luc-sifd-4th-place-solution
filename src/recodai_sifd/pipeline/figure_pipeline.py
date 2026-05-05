from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from recodai_sifd.pipeline.figure_kind import classify_figure_kind, FigureKind
from recodai_sifd.pipeline.reuse_detection import (
    ReusePruningConfig,
    ReuseSavePolicy,
    run_reuse_detection_all_pairs,
)
from recodai_sifd.pipeline.region_grouping import compute_grouping_result, save_grouping_outputs
from recodai_sifd.utils.mask_debug import summarize_instance_masks
from recodai_sifd.utils.submission_eval import (
    instance_dict_to_multichannel,
    recodai_annotation_from_multichannel,
)

from recodai_sifd.pipeline.intra_panel_api import IntraPanelCopyMoveModel, NoopIntraPanelModel
from recodai_sifd.pipeline.panel_crops_adapter import iter_panel_crops, panel_item_to_rgb_numpy
from recodai_sifd.pipeline.mask_geometry import paste_crop_mask_to_figure
from recodai_sifd.pipeline.mask_fusion import MaskFusionConfig, fuse_inter_intra_instances


@dataclass
class FigurePipelineConfig:
    output_root: Path
    debug: bool = False

    # classify heuristics
    edge_margin_ratio: float = 0.02
    min_panel_area_ratio: float = 0.0

    # reuse detection (matching decision)
    min_matched_keypoints: int = 20
    reuse_only_same_class: bool = False
    reuse_assume_bgr: bool = False
    reuse_keep_image: bool = True

    # NEW: output + pruning controls
    reuse_save_policy: Optional[ReuseSavePolicy] = None
    reuse_prune: Optional[ReusePruningConfig] = None
    reuse_debug_pairs: bool = False

    # submission encoding
    min_pixels_pred: int = 10

    # fusion
    fusion: MaskFusionConfig = MaskFusionConfig()


@dataclass(frozen=True)
class FigurePipelineResult:
    figure_id: str
    kind: str
    pred_annotation: str
    meta: Dict[str, Any]


class FigurePipeline:
    def __init__(
        self,
        panel_detector: Any,
        cfg: FigurePipelineConfig,
        intra_model: Optional[IntraPanelCopyMoveModel] = None,
    ):
        self.panel_detector = panel_detector
        self.cfg = cfg
        self.intra_model = intra_model or NoopIntraPanelModel()

    def _run_intra_on_crops(self, panel_crops: Any, figure_size_wh: Tuple[int, int]) -> List[np.ndarray]:
        """
        Runs intra model per panel crop and returns figure-space masks.
        (Today NoopIntraPanelModel returns empty.)
        """
        masks_fig: List[np.ndarray] = []
        for item in iter_panel_crops(panel_crops):
            panel_rgb = panel_item_to_rgb_numpy(item)
            masks_crop = self.intra_model.predict_instances(panel_rgb, panel_uid=item.uid)
            for m_crop in masks_crop:
                masks_fig.append(paste_crop_mask_to_figure(m_crop, item.xyxy, figure_size_wh))
        return masks_fig

    def process_figure(
        self,
        *,
        figure_id: str,
        figure_image: Any,  # PIL RGB
        figure_size_wh: Tuple[int, int],
        panel_detections: Any,
        panel_crops: Optional[Any] = None,  # lazily computed
    ) -> FigurePipelineResult:
        # classify
        decision = classify_figure_kind(
            image_size_wh=figure_size_wh,
            detections=panel_detections.detections,
            edge_margin_ratio=self.cfg.edge_margin_ratio,
            min_panel_area_ratio=self.cfg.min_panel_area_ratio,
        )

        meta: Dict[str, Any] = {
            "decision_kind": str(decision.kind),
            "decision_reason": decision.reason,
            "n_panels": decision.n_panels,
        }

        if self.cfg.debug:
            logger.debug(f"[figure_kind] {figure_id}: {decision.kind} ({decision.reason})")

        # SIMPLE / UNKNOWN: placeholder authentic (skip everything)
        if decision.kind in (FigureKind.SIMPLE, FigureKind.UNKNOWN):
            return FigurePipelineResult(
                figure_id=figure_id,
                kind=str(decision.kind),
                pred_annotation="authentic",
                meta=meta,
            )

        # We’ll need crops for both compound paths
        if panel_crops is None:
            panel_crops = self.panel_detector.extract_crops(figure_image, panel_detections)

        # Debug: print crop sizes once, right after extraction
        if self.cfg.debug:
            sizes_wh_by_uid = {}
            for item in iter_panel_crops(panel_crops):
                rgb = panel_item_to_rgb_numpy(item)
                h, w = rgb.shape[:2]
                sizes_wh_by_uid[item.uid] = (w, h)
                logger.debug(
                    f"[panel_crop] {figure_id} uid={item.uid} xyxy={tuple(round(v, 1) for v in item.xyxy)} size_wh=({w},{h})"
                )
            meta["panel_sizes_wh_by_uid"] = sizes_wh_by_uid

        # COMPOUND SINGLE: run crops now; later you’ll run intra here.
        if decision.kind == FigureKind.COMPOUND_SINGLE:
            intra_masks_fig = self._run_intra_on_crops(panel_crops, figure_size_wh)
            meta["intra_instances"] = len(intra_masks_fig)

            return FigurePipelineResult(
                figure_id=figure_id,
                kind=str(decision.kind),
                pred_annotation="authentic",
                meta=meta,
            )

        # COMPOUND MULTI: inter-panel pipeline + intra + fuse
        reuse_artifacts_dir = self.cfg.output_root / "reuse" / figure_id
        grouping_dir = self.cfg.output_root / "reuse_groups" / figure_id

        # SAFETY: if pruning is disabled, do NOT pass it (true legacy behavior)
        prune_to_pass: Optional[ReusePruningConfig] = None
        if self.cfg.reuse_prune is not None and bool(self.cfg.reuse_prune.enabled):
            prune_to_pass = self.cfg.reuse_prune

        reuse_result = run_reuse_detection_all_pairs(
            panel_crops,
            figure_id=figure_id,
            save_dir=reuse_artifacts_dir,
            assume_bgr=self.cfg.reuse_assume_bgr,
            keep_image=self.cfg.reuse_keep_image,
            only_same_class=self.cfg.reuse_only_same_class,
            min_matched_keypoints=self.cfg.min_matched_keypoints,
            save_policy=self.cfg.reuse_save_policy,
            prune=prune_to_pass,
            debug=self.cfg.debug,
            debug_pairs=self.cfg.reuse_debug_pairs,
        )

        panel_xyxy_by_uid = {uid: node.xyxy for uid, node in reuse_result.graph.panels.items()}

        # grouping -> inter instances in figure space
        grouping = compute_grouping_result(
            figure_shape_hw=figure_size_wh,  # (W,H)
            shape_is_wh=True,
            panel_xyxy_by_uid=panel_xyxy_by_uid,
            pairs=reuse_result.graph.pairs,
        )
        inter_instances: Dict[int, np.ndarray] = grouping.instance_masks_by_id
        meta["inter_instances"] = len(inter_instances)

        if self.cfg.debug:
            logger.debug(f"[inter] Instances: {sorted(inter_instances.keys())}")
            _ = summarize_instance_masks(inter_instances)

            save_grouping_outputs(
                figure_shape_hw=figure_size_wh,
                shape_is_wh=True,
                panel_xyxy_by_uid=panel_xyxy_by_uid,
                pairs=reuse_result.graph.pairs,
                out_dir=grouping_dir,
            )
            logger.debug("[inter] Saved grouping outputs")

        # intra-panel model per crop (stub now)
        intra_masks_fig = self._run_intra_on_crops(panel_crops, figure_size_wh)
        meta["intra_instances"] = len(intra_masks_fig)

        # fuse (overlap -> union; else append)
        fused_instances = fuse_inter_intra_instances(inter_instances, intra_masks_fig, self.cfg.fusion)
        meta["fused_instances"] = len(fused_instances)

        # submission annotation from fused instances
        if not fused_instances:
            pred_annotation = "authentic"
        else:
            pred_mc = instance_dict_to_multichannel(fused_instances, channel_axis=-1)
            pred_annotation = recodai_annotation_from_multichannel(
                pred_mc,
                channel_axis=-1,
                min_pixels_pred=self.cfg.min_pixels_pred,
            )

        return FigurePipelineResult(
            figure_id=figure_id,
            kind=str(decision.kind),
            pred_annotation=pred_annotation,
            meta=meta,
        )
