"""Step 2B: Batched GPU FP16 SigLIP 2 Scoring across 100% of images."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from .config import PipelineConfig

logger = logging.getLogger(__name__)

# Full-frame prompts: Wildlife action/pose + Landscape/Sun/Sunrise/Sunset
WILDLIFE_ACT_POS = [
    "a dramatic wildlife photograph of a bird in flight with spread wings",
    "an exciting action wildlife photo of a bird or animal hunting, catching prey, or interacting",
    "a dynamic wildlife photo of a bird or animal with an open beak, calling, running, or displaying",
]
WILDLIFE_COMP_POS = [
    "an award-winning wildlife photograph with clean creamy bokeh background and clear subject separation",
    "a sharp well-composed wildlife portrait in clear unobstructed view",
]
WILDLIFE_FULL_NEG = [
    "a cluttered wildlife photo obscured by dense branches, twigs, or messy foreground",
    "a blurry out-of-focus wildlife photo with motion blur",
    "an uninteresting static photo of a bird or animal from behind facing away",
]

SCENIC_POS = [
    "a breathtaking golden hour sunrise or sunset landscape photograph with glowing sun and dramatic sky",
    "a vivid sun disk or sunburst over a layered mountain, water, or desert horizon",
    "a moody scenic landscape with dramatic clouds, mist, warm light, or reflections",
    "dramatic golden hour rim light or warm sunset glow illuminating the scene",
]
SCENIC_NEG = [
    "a flat blown-out overexposed white sky with no color or detail",
    "an accidental misfire photo of empty blurry ground, dirt, grass, or plain blue sky",
    "an underexposed dark blurry test shot with no clear subject or composition",
]

# Tight head/subject crop prompts (verified wildlife only)
HEAD_POS = [
    "a tack-sharp close-up of a bird or wild animal's eye with bright catchlight and crisp detail",
    "a wild bird or animal making direct eye contact with its head angled toward the camera",
    "a crystal-clear unobstructed close-up of a wild bird or animal's face and sharp eye",
]
HEAD_NEG = [
    "a blurry out-of-focus animal or bird eye with soft indistinct feathers or fur",
    "a bird or animal with its head turned away or back of the head with no visible eye",
    "branches, twigs, or grass crossing directly in front of a bird or animal's eye or face",
]


def _extract_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    raise TypeError(f"Unexpected feature output type: {type(out)}")


def _load_pil_rgb_safe(path: Path) -> Image.Image:
    with open(path, "rb") as fh:
        img = Image.open(fh)
        img.load()
        return img.convert("RGB")


class SigLIP2Scorer:
    """Runs SigLIP 2 in FP16 on RTX 4060 across 100% of images + wildlife head crops."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        model_id = cfg.siglip2_model
        try:
            logger.info("Loading SigLIP 2 model %s on %s (%s)...", model_id, self.device, self.dtype)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
            self.model.eval()
            self.model_id = model_id
        except Exception as exc:
            logger.warning("Primary %s failed (%s), falling back to %s.", model_id, exc, cfg.siglip2_fallback_model)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model_id = cfg.siglip2_fallback_model
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
            self.model.eval()
            self.model_id = model_id

        self.full_prompts = WILDLIFE_ACT_POS + WILDLIFE_COMP_POS + WILDLIFE_FULL_NEG + SCENIC_POS + SCENIC_NEG
        self.head_prompts = HEAD_POS + HEAD_NEG

        self.full_text_embeds = self._encode_texts(self.full_prompts)
        self.head_text_embeds = self._encode_texts(self.head_prompts)

        self.logit_scale = (
            self.model.logit_scale.exp().detach().float()
            if hasattr(self.model, "logit_scale")
            else torch.tensor(10.0, device=self.device)
        )
        self.logit_bias = (
            self.model.logit_bias.detach().float()
            if hasattr(self.model, "logit_bias") and self.model.logit_bias is not None
            else torch.tensor(0.0, device=self.device)
        )

        if self.device == "cuda":
            logger.info(
                "SigLIP 2 (%s) ready. GPU VRAM allocated: %.0f MB",
                self.model_id,
                torch.cuda.memory_allocated() / (1024 * 1024),
            )

    @torch.no_grad()
    def _encode_texts(self, prompts: List[str]) -> torch.Tensor:
        # SigLIP 2 expects lowercase text with max_length=64
        clean_prompts = [p.lower() for p in prompts]
        inputs = self.processor(
            text=clean_prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = _extract_tensor(self.model.get_text_features(**inputs)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats

    @torch.no_grad()
    def _encode_images_and_score(
        self, pil_images: List[Image.Image], text_embeds: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        img_feats = _extract_tensor(self.model.get_image_features(pixel_values=pixel_values)).float()
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cos_sim = torch.matmul(img_feats, text_embeds.T)
        probs = torch.sigmoid(cos_sim * self.logit_scale + self.logit_bias)
        return (
            img_feats.cpu().numpy().astype(np.float16),
            cos_sim.cpu().numpy(),
            probs.cpu().numpy(),
        )

    def score_all_images(
        self, records: List[Dict[str, Any]], output_dir: Path
    ) -> np.ndarray:
        """
        Score 100% of images on Full Frame (Wildlife Action + Scenic/Sun/Landscape),
        and score Tight Head Crops for all verified Wildlife detections.
        Returns full_frame SigLIP 2 embedding matrix [N, D] as float16.
        """
        n = len(records)
        bs = self.cfg.siglip2_batch_size

        n_wact = len(WILDLIFE_ACT_POS)
        n_wcomp = len(WILDLIFE_COMP_POS)
        n_wneg = len(WILDLIFE_FULL_NEG)
        n_spos = len(SCENIC_POS)

        full_embs_list: List[np.ndarray] = []

        # 1) Full-frame batched inference across 100% of images
        for start in tqdm(range(0, n, bs), desc="Step 2B: SigLIP 2 Full-Frame (100%)", unit="batch"):
            batch = records[start : start + bs]
            imgs = [_load_pil_rgb_safe(output_dir / r["thumb_rel_path"]) for r in batch]
            embs, sims, probs = self._encode_images_and_score(imgs, self.full_text_embeds)
            full_embs_list.append(embs)

            for i, rec in enumerate(batch):
                s = sims[i]
                p = probs[i]

                w_act = s[:n_wact]
                w_comp = s[n_wact : n_wact + n_wcomp]
                w_neg = s[n_wact + n_wcomp : n_wact + n_wcomp + n_wneg]

                sc_pos_s = s[n_wact + n_wcomp + n_wneg : n_wact + n_wcomp + n_wneg + n_spos]
                sc_neg_s = s[n_wact + n_wcomp + n_wneg + n_spos :]
                sc_pos_p = p[n_wact + n_wcomp + n_wneg : n_wact + n_wcomp + n_wneg + n_spos]

                act_val = float(0.65 * np.max(w_act) + 0.35 * np.mean(w_act))
                comp_val = float(np.mean(w_comp))
                wneg_val = float(0.5 * np.mean(w_neg) + 0.5 * np.max(w_neg))
                rec["siglip_action_raw"] = round((0.55 * act_val + 0.45 * comp_val) - 0.70 * wneg_val, 5)

                sc_val = float(0.60 * np.max(sc_pos_s) + 0.40 * np.mean(sc_pos_s))
                sc_neg_val = float(0.50 * np.mean(sc_neg_s) + 0.50 * np.max(sc_neg_s))
                rec["siglip_scenic_raw"] = round(sc_val - 0.85 * sc_neg_val, 5)
                rec["siglip_scenic_prob"] = round(float(np.max(sc_pos_p)), 4)
                # Golden-hour / sunrise / sunset light signal for Hybrid Wildlife+Light bonus
                rec["siglip_light_prob"] = round(float(max(sc_pos_p[0], sc_pos_p[3])), 4)
                rec["siglip_eye_raw"] = 0.0

        # 2) Tight Head/Subject Crop inference for ALL verified Wildlife images
        wildlife_recs = [r for r in records if r.get("has_wildlife")]
        n_hpos = len(HEAD_POS)
        if wildlife_recs:
            for start in tqdm(
                range(0, len(wildlife_recs), bs),
                desc="Step 2B: SigLIP 2 Head/Eye Crops",
                unit="batch",
            ):
                batch = wildlife_recs[start : start + bs]
                imgs = [
                    _load_pil_rgb_safe(output_dir / r["_siglip_crop_rel_path"])
                    for r in batch
                ]
                _, sims, _ = self._encode_images_and_score(imgs, self.head_text_embeds)
                for i, rec in enumerate(batch):
                    hs = sims[i]
                    hpos = hs[:n_hpos]
                    hneg = hs[n_hpos:]
                    hpos_val = float(0.55 * np.mean(hpos) + 0.45 * np.max(hpos))
                    hneg_val = float(0.45 * np.mean(hneg) + 0.55 * np.max(hneg))
                    rec["siglip_eye_raw"] = round(hpos_val - 0.90 * hneg_val, 5)

        return np.vstack(full_embs_list) if full_embs_list else np.zeros((0, 1), dtype=np.float16)

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
