"""Stage 3: GPU SigLIP Two-View Eye Contact, Action & Interest Scoring (Layer 2 - To Edit)."""

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

# Full-frame prompts: action, wing spread, flight, prey, interaction, bokeh, lighting vs clutter/blur
FULL_POS_ACTION_PROMPTS = [
    "a dramatic wildlife photograph of a bird in flight with spread wings",
    "an exciting action wildlife photo of a bird or animal hunting, catching prey, or interacting",
    "a dynamic wildlife photo of a bird or animal with an open beak, calling, running, or displaying",
]

FULL_POS_COMP_PROMPTS = [
    "an award-winning wildlife photograph with clean creamy bokeh background and golden light",
    "a sharp, well-composed close portrait of a wild bird or animal in clear unobstructed view",
]

FULL_NEG_PROMPTS = [
    "a cluttered wildlife photo obscured by dense branches, twigs, or messy foreground",
    "a distant tiny bird lost in a busy distracting background",
    "a blurry out-of-focus wildlife photo with motion blur",
    "an uninteresting static photo of a bird or animal from behind facing away",
    "a poorly framed wildlife photo with the animal's head cut off",
]

# Tight bird/head crop prompts: sharp eye focus, catchlight, eye contact, unobstructed face
HEAD_POS_PROMPTS = [
    "a tack-sharp close-up of a bird or wild animal's eye with bright catchlight and crisp detail",
    "a wild bird or animal making direct eye contact with the camera with its head angled toward viewer",
    "a crystal-clear unobstructed close-up of a wild bird or animal's face and sharp eye",
]

HEAD_NEG_PROMPTS = [
    "a blurry out-of-focus animal or bird eye with soft indistinct feathers or fur",
    "a bird or animal with its head turned away or back of the head with no visible eye",
    "branches, twigs, or grass crossing directly in front of a bird or animal's eye or face",
    "empty blurry foliage, branches, sky, or background with no bird or animal visible",
]


def _extract_tensor(out: Any) -> torch.Tensor:
    """Safely extract feature tensor across transformers v4/v5 return types."""
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    raise TypeError(f"Unexpected feature output type: {type(out)}")


def _load_pil_rgb_safe(path: Path) -> Image.Image:
    """Unicode-safe PIL image reader using binary file handle."""
    with open(path, "rb") as fh:
        img = Image.open(fh)
        img.load()
        return img.convert("RGB")


def _robust_norm_0_100(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    if arr.size == 1:
        return np.array([75.0], dtype=np.float32)
    p5, p95 = np.percentile(arr, [5, 95])
    if p95 - p5 < 1e-6:
        return np.full_like(arr, 50.0, dtype=np.float32)
    scaled = (arr - p5) / (p95 - p5)
    return np.clip(scaled * 100.0, 0.0, 100.0).astype(np.float32)


class SigLIPScorer:
    """FP16 GPU SigLIP scorer staying well under 8GB VRAM on RTX 4060."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        model_id = cfg.siglip_model
        try:
            logger.info("Loading SigLIP model %s on %s (%s)...", model_id, self.device, self.dtype)
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
            self.model.eval()
            self.model_id = model_id
        except Exception as exc:
            logger.warning(
                "Primary SigLIP model %s failed (%s). Falling back to %s.",
                model_id,
                exc,
                cfg.siglip_fallback_model,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model_id = cfg.siglip_fallback_model
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(self.device)
            self.model.eval()
            self.model_id = model_id

        # Precompute text embeddings once
        self.full_prompts = FULL_POS_ACTION_PROMPTS + FULL_POS_COMP_PROMPTS + FULL_NEG_PROMPTS
        self.head_prompts = HEAD_POS_PROMPTS + HEAD_NEG_PROMPTS

        self.full_text_embeds = self._encode_texts(self.full_prompts)
        self.head_text_embeds = self._encode_texts(self.head_prompts)

        self.logit_scale = self.model.logit_scale.exp().detach().float()
        self.logit_bias = (
            self.model.logit_bias.detach().float()
            if hasattr(self.model, "logit_bias") and self.model.logit_bias is not None
            else torch.tensor(0.0, device=self.device)
        )

        if self.device == "cuda":
            alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info("SigLIP (%s) ready. GPU VRAM allocated: %.0f MB", self.model_id, alloc_mb)

    @torch.no_grad()
    def _encode_texts(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.processor(
            text=prompts, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = _extract_tensor(self.model.get_text_features(**inputs)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feats

    @torch.no_grad()
    def _score_images(
        self, pil_images: List[Image.Image], text_embeds: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (cosine_sim_matrix, sigmoid_prob_matrix) of shape [batch, num_prompts].
        Using cosine similarities for pos-vs-neg differentials avoids sigmoid saturation near 0,
        while sigmoid probabilities provide calibrated absolute detection checks (e.g., no-bird).
        """
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        img_feats = _extract_tensor(self.model.get_image_features(pixel_values=pixel_values)).float()
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cos_sim = torch.matmul(img_feats, text_embeds.T)
        logits = cos_sim * self.logit_scale + self.logit_bias
        probs = torch.sigmoid(logits)
        return cos_sim.cpu().numpy(), probs.cpu().numpy()

    def score_layer1_keepers(
        self,
        records: List[Dict[str, Any]],
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Score all Layer 1 keepers on full-frame view and tight head/bird crop view,
        then combine with Stage 2 sharpness to promote Layer 2 ('To Edit', ~3-7% overall).
        """
        l1_records = [r for r in records if r["layer"] == "layer1_keep"]
        if not l1_records:
            return records

        bs = self.cfg.siglip_batch_size
        n_act = len(FULL_POS_ACTION_PROMPTS)
        n_comp = len(FULL_POS_COMP_PROMPTS)
        n_hpos = len(HEAD_POS_PROMPTS)

        raw_eye_scores = np.zeros(len(l1_records), dtype=np.float32)
        raw_act_scores = np.zeros(len(l1_records), dtype=np.float32)
        raw_comp_scores = np.zeros(len(l1_records), dtype=np.float32)
        no_bird_probs = np.zeros(len(l1_records), dtype=np.float32)

        for start in tqdm(
            range(0, len(l1_records), bs),
            desc="Stage 3: SigLIP Eye/Action Scoring",
            unit="batch",
        ):
            batch = l1_records[start : start + bs]
            full_imgs: List[Image.Image] = []
            head_imgs: List[Image.Image] = []

            for rec in batch:
                thumb_path = output_dir / rec["thumb_rel_path"]
                sig_crop_path = output_dir / rec.get("_siglip_crop_rel_path", rec["crop_rel_path"])
                full_imgs.append(_load_pil_rgb_safe(thumb_path))
                head_imgs.append(_load_pil_rgb_safe(sig_crop_path))

            full_sims, _ = self._score_images(full_imgs, self.full_text_embeds)
            head_sims, head_probs = self._score_images(head_imgs, self.head_text_embeds)

            for i in range(len(batch)):
                fs = full_sims[i]
                hs = head_sims[i]
                hp = head_probs[i]

                # Full-frame action & composition (cosine similarity margin: non-saturating)
                act_pos = fs[:n_act]
                comp_pos = fs[n_act : n_act + n_comp]
                full_neg = fs[n_act + n_comp :]

                act_val = float(0.65 * np.max(act_pos) + 0.35 * np.mean(act_pos))
                comp_val = float(np.mean(comp_pos))
                fneg_val = float(0.5 * np.mean(full_neg) + 0.5 * np.max(full_neg))

                raw_act_scores[start + i] = act_val - 0.60 * fneg_val
                raw_comp_scores[start + i] = comp_val - 0.85 * fneg_val

                # Head crop: sharp eye + catchlight + eye contact vs blurry / turned away / branches / no bird
                hpos = hs[:n_hpos]
                hneg = hs[n_hpos:]
                hpos_val = float(0.55 * np.mean(hpos) + 0.45 * np.max(hpos))
                hneg_val = float(0.45 * np.mean(hneg) + 0.55 * np.max(hneg))
                raw_eye_scores[start + i] = hpos_val - 0.90 * hneg_val
                no_bird_probs[start + i] = float(hp[-1])  # Calibrated sigmoid probability of empty foliage

        # Normalize Stage 3 signals across Layer 1 keepers to 0..100
        eye_norm = _robust_norm_0_100(raw_eye_scores)
        act_comb_raw = 0.55 * raw_act_scores + 0.45 * raw_comp_scores
        act_norm = _robust_norm_0_100(act_comb_raw)
        sharp_vals = np.array([float(r["sharpness_score"]) for r in l1_records], dtype=np.float32)
        sharp_l1_norm = _robust_norm_0_100(sharp_vals)

        for i, rec in enumerate(l1_records):
            rec["eye_contact_score"] = round(float(eye_norm[i]), 2)
            rec["action_score"] = round(float(act_norm[i]), 2)
            rec["siglip_eye_raw"] = round(float(raw_eye_scores[i]), 4)
            rec["siglip_action_raw"] = round(float(act_comb_raw[i]), 4)
            rec["no_bird_prob"] = round(float(no_bird_probs[i]), 4)

            combined = (
                self.cfg.weight_sharpness * float(sharp_l1_norm[i])
                + self.cfg.weight_eye_contact * float(eye_norm[i])
                + self.cfg.weight_action_comp * float(act_norm[i])
            )
            # Strong penalty if SigLIP thinks the crop is empty foliage/no bird
            if no_bird_probs[i] > 0.25:
                combined -= min(30.0, (float(no_bird_probs[i]) - 0.25) * 60.0)
            # Mild confidence boost for confirmed YOLO wildlife vs fallback
            if rec["detection_source"] == "saliency_fallback":
                combined -= 5.0

            rec["overall_score"] = round(float(np.clip(combined, 0.0, 100.0)), 2)

        # For rejected records (not passed to SigLIP), set default Stage 3 scores
        for rec in records:
            if rec["layer"] != "layer1_keep":
                rec.setdefault("eye_contact_score", 0.0)
                rec.setdefault("action_score", 0.0)
                rec.setdefault("siglip_eye_raw", 0.0)
                rec.setdefault("siglip_action_raw", 0.0)
                rec.setdefault("no_bird_prob", 0.0)
                rec["overall_score"] = round(float(rec["sharpness_score"]) * 0.35, 2)

        # Promote top ~5.0% overall (within 3-7%, roughly top 25-35% of Layer 1) into Layer 2 ('To Edit')
        total = len(records)
        target_l2 = max(1, int(round(total * self.cfg.layer2_target_ratio)))
        if total <= 25:
            target_l2 = max(1, min(len(l1_records), max(2, int(round(total * 0.20)))))
        target_l2 = min(target_l2, len(l1_records))

        l1_sorted = sorted(l1_records, key=lambda r: r["overall_score"], reverse=True)
        burst_l2_counts: Dict[str, int] = {}
        l2_selected: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []

        for rec in l1_sorted:
            if len(l2_selected) >= target_l2:
                break
            bid = rec["burst_id"]
            if burst_l2_counts.get(bid, 0) < self.cfg.max_per_burst_layer2:
                rec["layer"] = "layer2_to_edit"
                burst_l2_counts[bid] = burst_l2_counts.get(bid, 0) + 1
                l2_selected.append(rec)
            else:
                deferred.append(rec)

        # Top up if burst cap left us slightly short of target_l2
        for rec in deferred:
            if len(l2_selected) >= target_l2:
                break
            rec["layer"] = "layer2_to_edit"
            l2_selected.append(rec)

        # Assign global overall_rank (Layer 2 first by overall_score, then Layer 1, then Rejected)
        layer_order = {"layer2_to_edit": 0, "layer1_keep": 1, "rejected": 2}
        all_ranked = sorted(
            records,
            key=lambda r: (layer_order[r["layer"]], -float(r["overall_score"]), -float(r["sharpness_score"])),
        )
        for rank_idx, rec in enumerate(all_ranked, start=1):
            rec["overall_rank"] = rank_idx

        logger.info(
            "Layer 2 (To Edit) selection complete: %d / %d images (%.2f%% overall, %.1f%% of Layer 1).",
            len(l2_selected),
            total,
            100.0 * len(l2_selected) / max(1, total),
            100.0 * len(l2_selected) / max(1, len(l1_records)),
        )
        return records

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
