from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

from .types import ClassifierResult
from .prompts import PromptPack, DEFAULT_PROMPT_PACK
from .io_utils import ImageLike, to_pil_rgb


_INFER = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

@dataclass
class ModelConfig:
    model_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    context_len: int = 256

    # numerical stability guard
    logit_scale_max: float = 100.0

    # speed knobs
    use_amp: bool = True


class PanelTypeClassifier:
    """Zero-shot panel-type classifier using BiomedCLIP.

    Inputs:
      - image path (str / Path)
      - PIL.Image
      - numpy array (HxW, HxWx1, HxWx3, HxWx4)

    Output:
      - imaging vs non_imaging (group)
      - if imaging -> subtype + confidence

    Also exposes:
      - encode_image(...) -> unit-norm embedding (float32, shape [D])
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        prompt_pack: PromptPack = DEFAULT_PROMPT_PACK,
        config: Optional[ModelConfig] = None,
    ):
        self.config = config or ModelConfig()
        self.prompt_pack = prompt_pack

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load model + preprocess + tokenizer
        self.model, self.preprocess = create_model_from_pretrained(self.config.model_id)
        self.tokenizer = get_tokenizer(self.config.model_id)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Build and cache text features
        self.group_labels, self.group_text_feats = self._build_text_features(self.prompt_pack.group_prompts)
        self.subtype_labels, self.subtype_text_feats = self._build_text_features(self.prompt_pack.subtype_prompts)

    @_INFER()
    def _build_text_features(self, class_to_prompts: Dict[str, List[str]]) -> Tuple[List[str], torch.Tensor]:
        """Build one normalized text embedding per class by mean pooling prompts x templates."""
        labels: List[str] = []
        class_feats: List[torch.Tensor] = []

        for cls, prompts in class_to_prompts.items():
            all_texts: List[str] = []
            for p in prompts:
                for tmpl in self.prompt_pack.templates:
                    all_texts.append(tmpl.format(p))

            tokens = self.tokenizer(all_texts, context_length=self.config.context_len).to(self.device)

            # Text encoder is usually safe in fp16, but keep the pooled vector in fp32.
            amp_ctx = (
                torch.autocast(device_type=self.device.type, dtype=torch.float16)
                if (self.config.use_amp and self.device.type == "cuda")
                else nullcontext()
            )
            with amp_ctx:
                text_features = self.model.encode_text(tokens)

            text_features = text_features.float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            cls_feat = text_features.mean(dim=0)
            cls_feat = cls_feat / cls_feat.norm().clamp_min(1e-12)

            labels.append(cls)
            class_feats.append(cls_feat)

        feats = torch.stack(class_feats, dim=0)  # [C, D]
        return labels, feats

    def _preprocess_to_tensor(self, img: Image.Image) -> torch.Tensor:
        x = self.preprocess(img).unsqueeze(0)  # [1,3,H,W]
        return x.to(self.device)

    @_INFER()
    def encode_image(self, image: ImageLike) -> np.ndarray:
        """Return unit-norm embedding (float32, shape [D])."""
        pil = to_pil_rgb(image)
        images = self._preprocess_to_tensor(pil)

        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if (self.config.use_amp and self.device.type == "cuda")
            else nullcontext()
        )
        with amp_ctx:
            feat = self.model.encode_image(images)

        feat = feat.float()
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    @_INFER()
    def predict(
        self,
        image: ImageLike,
        return_probs: bool = False,
        return_embedding: bool = False,
        topk_probs: int = 0,
    ) -> ClassifierResult:
        """Predict on a single image."""
        pil = to_pil_rgb(image)
        images = self._preprocess_to_tensor(pil)

        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if (self.config.use_amp and self.device.type == "cuda")
            else nullcontext()
        )
        with amp_ctx:
            image_features = self.model.encode_image(images)

        image_features = image_features.float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        logit_scale = self.model.logit_scale.exp().float().clamp(max=float(self.config.logit_scale_max))

        # Group
        group_logits = logit_scale * (image_features @ self.group_text_feats.T)
        group_probs = torch.softmax(group_logits, dim=-1)

        if not torch.isfinite(group_probs).all():
            # fallback: reduce scale and recompute in fp32
            logit_scale = logit_scale.clamp(max=10.0)
            group_logits = logit_scale * (image_features @ self.group_text_feats.T)
            group_probs = torch.softmax(group_logits, dim=-1)

        g_idx = int(torch.argmax(group_probs, dim=-1).item())
        group_label = self.group_labels[g_idx]
        group_score = float(group_probs[0, g_idx].item())

        # Subtype
        subtype_logits = logit_scale * (image_features @ self.subtype_text_feats.T)
        subtype_probs = torch.softmax(subtype_logits, dim=-1)
        if not torch.isfinite(subtype_probs).all():
            logit_scale = logit_scale.clamp(max=10.0)
            subtype_logits = logit_scale * (image_features @ self.subtype_text_feats.T)
            subtype_probs = torch.softmax(subtype_logits, dim=-1)

        if group_label == "imaging":
            s_idx = int(torch.argmax(subtype_probs, dim=-1).item())
            subtype_label = self.subtype_labels[s_idx]
            subtype_score = float(subtype_probs[0, s_idx].item())
        else:
            subtype_label = "non_imaging"
            subtype_score = float(1.0 - group_score)

        out = ClassifierResult(
            group_label=group_label,
            group_score=group_score,
            subtype_label=subtype_label,
            subtype_score=subtype_score,
        )

        if return_probs or topk_probs > 0:
            # optionally reduce to topk to keep results small
            g = {lab: float(group_probs[0, j].item()) for j, lab in enumerate(self.group_labels)}
            s = {lab: float(subtype_probs[0, j].item()) for j, lab in enumerate(self.subtype_labels)}
            if topk_probs and topk_probs > 0:
                g = dict(sorted(g.items(), key=lambda kv: kv[1], reverse=True)[:topk_probs])
                s = dict(sorted(s.items(), key=lambda kv: kv[1], reverse=True)[:topk_probs])
            out.group_probs = g
            out.subtype_probs = s

        if return_embedding:
            out.embedding = image_features.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

        return out

    @_INFER()
    def predict_paths(
        self,
        paths: Sequence[Union[str, Path]],
        batch_size: int = 16,
        return_probs: bool = False,
        topk_probs: int = 0,
    ) -> List[Dict]:
        """Predict on many image paths efficiently."""
        results: List[Dict] = []
        buf: List[Path] = []

        def flush(batch: List[Path]):
            if not batch:
                return
            imgs = []
            valid_paths = []
            for p in batch:
                try:
                    im = Image.open(str(p)).convert("RGB")
                    imgs.append(self.preprocess(im))
                    valid_paths.append(p)
                except Exception:
                    continue

            if not imgs:
                return

            images = torch.stack(imgs, dim=0).to(self.device)

            amp_ctx = (
                torch.autocast(device_type=self.device.type, dtype=torch.float16)
                if (self.config.use_amp and self.device.type == "cuda")
                else nullcontext()
            )
            with amp_ctx:
                image_features = self.model.encode_image(images)

            image_features = image_features.float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            logit_scale = self.model.logit_scale.exp().float().clamp(max=float(self.config.logit_scale_max))

            group_logits = logit_scale * (image_features @ self.group_text_feats.T)
            group_probs = torch.softmax(group_logits, dim=-1)

            subtype_logits = logit_scale * (image_features @ self.subtype_text_feats.T)
            subtype_probs = torch.softmax(subtype_logits, dim=-1)

            for i, p in enumerate(valid_paths):
                g_idx = int(group_probs[i].argmax().item())
                group_label = self.group_labels[g_idx]
                group_score = float(group_probs[i, g_idx].item())

                if group_label == "imaging":
                    s_idx = int(subtype_probs[i].argmax().item())
                    subtype_label = self.subtype_labels[s_idx]
                    subtype_score = float(subtype_probs[i, s_idx].item())
                else:
                    subtype_label = "non_imaging"
                    subtype_score = float(1.0 - group_score)

                row = {
                    "path": str(p),
                    "filename": p.name,
                    "group_pred": group_label,
                    "group_conf": group_score,
                    "subtype_pred": subtype_label,
                    "subtype_conf": subtype_score,
                }

                if return_probs or topk_probs > 0:
                    g = {lab: float(group_probs[i, j].item()) for j, lab in enumerate(self.group_labels)}
                    s = {lab: float(subtype_probs[i, j].item()) for j, lab in enumerate(self.subtype_labels)}
                    if topk_probs and topk_probs > 0:
                        g = dict(sorted(g.items(), key=lambda kv: kv[1], reverse=True)[:topk_probs])
                        s = dict(sorted(s.items(), key=lambda kv: kv[1], reverse=True)[:topk_probs])
                    row.update({f"p_group_{k}": v for k, v in g.items()})
                    row.update({f"p_sub_{k}": v for k, v in s.items()})

                results.append(row)

        for p in paths:
            buf.append(Path(p))
            if len(buf) >= batch_size:
                flush(buf)
                buf = []
        flush(buf)
        return results