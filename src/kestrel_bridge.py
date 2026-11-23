"""Headless, strictly read-only bridge to Project Kestrel's GPU ONNX modules."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _init_kestrel_imports(kestrel_dir: Path):
    analyzer_root = str((kestrel_dir / "analyzer").resolve())
    if analyzer_root not in sys.path:
        sys.path.insert(0, analyzer_root)

    # Import torch first so CUDA/cuDNN DLLs are loaded for onnxruntime-gpu
    import torch  # noqa: F401
    import kestrel_analyzer.ml as kml
    import kestrel_analyzer.ml.providers as kprov

    # Configure Project Kestrel's provider policy to use CUDAExecutionProvider on RTX 4060
    kml.GPU_EP = "CUDAExecutionProvider"
    kprov.GPU_GRAPH_EPS = frozenset({"DmlExecutionProvider", "CUDAExecutionProvider"})
    kprov._probe_unsupported = True


def compute_masked_head_sharpness(
    img_rgb: np.ndarray,
    mask_bool: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
) -> Tuple[float, Tuple[int, int, int, int]]:
    """
    Use the SAM-HQ binary silhouette mask (ignoring background sky/branches!) to locate
    the upper/head region of the segmented animal and compute contrast-normalized sharpness.
    Returns (head_sharpness_raw, head_bbox_xyxy).
    """
    h_img, w_img = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w_img - 2, int(x1)))
    y1 = max(0, min(h_img - 2, int(y1)))
    x2 = max(x1 + 2, min(w_img, int(x2)))
    y2 = max(y1 + 2, min(h_img, int(y2)))

    sub_rgb = img_rgb[y1:y2, x1:x2]
    sub_mask = mask_bool[y1:y2, x1:x2]
    bh, bw = sub_rgb.shape[:2]
    if bh < 16 or bw < 16 or not np.any(sub_mask):
        return 0.0, (x1, y1, x2, y2)

    # Find centroid of the upper 45% of masked wildlife pixels (where head/eye almost always sits,
    # even on wide wingspan birds where wings stretch far left/right!)
    ys, xs = np.nonzero(sub_mask)
    y_cut = np.percentile(ys, 45)
    upper_sel = ys <= y_cut
    if np.any(upper_sel):
        head_cx = int(np.median(xs[upper_sel]))
        head_cy = int(np.median(ys[upper_sel]))
    else:
        head_cx = int(np.median(xs))
        head_cy = int(np.median(ys))

    # Build compact head box around that masked upper-body centroid
    half_side = max(24, int(round(min(bw, bh) * 0.28)))
    hx1 = max(0, head_cx - half_side)
    hy1 = max(0, head_cy - half_side)
    hx2 = min(bw, head_cx + half_side)
    hy2 = min(bh, head_cy + half_side)

    head_rgb = sub_rgb[hy1:hy2, hx1:hx2]
    head_m = sub_mask[hy1:hy2, hx1:hx2]
    hh, hw = head_rgb.shape[:2]
    if hh < 12 or hw < 12 or not np.any(head_m):
        head_rgb = sub_rgb
        head_m = sub_mask
        hx1, hy1, hx2, hy2 = 0, 0, bw, bh

    # Cap large close-ups at 640px for scale-consistent gradient measurement
    max_side = max(head_rgb.shape[:2])
    if max_side > 640:
        s = 640.0 / max_side
        head_rgb = cv2.resize(
            head_rgb, (int(round(head_rgb.shape[1] * s)), int(round(head_rgb.shape[0] * s))),
            interpolation=cv2.INTER_AREA,
        )
        head_m = cv2.resize(
            head_m.astype(np.uint8), (head_rgb.shape[1], head_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    gray = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_eval = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.45)
    gx = cv2.Sobel(gray_eval, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eval, cv2.CV_32F, 0, 1, ksize=3)
    ten = gx * gx + gy * gy
    lap = cv2.Laplacian(gray_eval, cv2.CV_32F, ksize=3)
    lap_sq = lap * lap

    # Evaluate strictly on masked wildlife pixels
    m_vals = gray[head_m]
    if m_vals.size < 16:
        return 0.0, (x1 + hx1, y1 + hy1, x1 + hx2, y1 + hy2)

    local_var = float(np.var(m_vals)) + 180.0
    ten_masked = ten[head_m]
    lap_masked = lap_sq[head_m]
    # Top 25% sharpest pixels on the head/upper body (eye, catchlight, crisp feather detail)
    ten_p75 = float(np.percentile(ten_masked, 75))
    lap_p75 = float(np.percentile(lap_masked, 75))
    sharp_raw = float(
        np.log1p((ten_p75 / local_var) * 6.0) * 0.65 + np.log1p((lap_p75 / local_var) * 10.0) * 0.35
    )

    abs_head_box = (x1 + hx1, y1 + hy1, x1 + hx2, y1 + hy2)
    return round(sharp_raw, 5), abs_head_box


def compute_global_landscape_metrics(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Compute global frame sharpness + tonal balance / non-clipped highlight ratio for Track B
    (Sun / Sunrise / Sunset / Landscape).
    """
    h, w = img_rgb.shape[:2]
    max_side = max(h, w)
    target = 1280
    if max_side > target:
        s = target / float(max_side)
        work = cv2.resize(img_rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    else:
        work = img_rgb

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_eval = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.5)
    gx = cv2.Sobel(gray_eval, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eval, cv2.CV_32F, 0, 1, ksize=3)
    ten = gx * gx + gy * gy

    # 80th percentile gradient normalized by global contrast variance
    g_var = float(np.var(gray)) + 200.0
    global_sharp_raw = float(np.log1p((np.percentile(ten, 80) / g_var) * 10.0))

    # Dynamic range (5th..95th percentile spread) & highlight clipping penalty
    p5, p95 = np.percentile(gray, [5, 95])
    dr_score = min(1.0, float(p95 - p5) / 170.0)
    blown_frac = float((gray >= 252.0).mean())
    crushed_frac = float((gray <= 5.0).mean())
    # Allow a small sun disk (~3% blown) without penalty; penalize large blown-out skies (>12%)
    clip_penalty = max(0.0, blown_frac - 0.06) * 2.5 + max(0.0, crushed_frac - 0.25) * 1.2
    tonal_balance = float(max(0.0, min(1.0, dr_score - clip_penalty)))

    return {
        "global_sharp_raw": round(global_sharp_raw, 5),
        "tonal_balance": round(tonal_balance, 4),
        "blown_frac": round(blown_frac, 4),
    }


def extract_crops_bgr(
    img_bgr: np.ndarray,
    subject_bbox: Optional[Tuple[int, int, int, int]],
    head_bbox: Optional[Tuple[int, int, int, int]],
    output_size: int = 384,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (gallery_detail_crop_bgr, siglip_head_crop_bgr):
      - If wildlife detected: 1:1 / adaptive zoomed head/subject crop + tight head crop for SigLIP 2
      - If no wildlife (Landscape/Other): 1:1 center-horizon detail crop (siglip_head_crop is None)
    """
    h_img, w_img = img_bgr.shape[:2]

    if subject_bbox is None or head_bbox is None:
        # Center-horizon 1:1 detail crop (slightly above vertical center where horizon/sun sits)
        side = min(output_size * 2, h_img, w_img)
        cx = w_img // 2
        cy = int(round(h_img * 0.45))
        x1 = max(0, min(w_img - side, cx - side // 2))
        y1 = max(0, min(h_img - side, cy - side // 2))
        crop = img_bgr[y1 : y1 + side, x1 : x1 + side]
        if crop.shape[0] != output_size or crop.shape[1] != output_size:
            crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
        return crop, None

    hx1, hy1, hx2, hy2 = head_bbox
    sx1, sy1, sx2, sy2 = subject_bbox
    hcx = (hx1 + hx2) // 2
    hcy = (hy1 + hy2) // 2
    hw = max(1, hx2 - hx1)
    hh = max(1, hy2 - hy1)
    sw = max(1, sx2 - sx1)
    sh = max(1, sy2 - sy1)

    # 1) Gallery 1:1 / adaptive head zoom crop
    desired_zoom = int(round(max(output_size, min(1100, max(hw, hh) * 1.15))))
    side_zoom = min(desired_zoom, h_img, w_img)
    zx1 = max(0, min(w_img - side_zoom, hcx - side_zoom // 2))
    zy1 = max(0, min(h_img - side_zoom, hcy - side_zoom // 2))
    zoom_crop = img_bgr[zy1 : zy1 + side_zoom, zx1 : zx1 + side_zoom]
    if zoom_crop.shape[0] != output_size or zoom_crop.shape[1] != output_size:
        interp = cv2.INTER_AREA if side_zoom > output_size else cv2.INTER_CUBIC
        zoom_crop = cv2.resize(zoom_crop, (output_size, output_size), interpolation=interp)

    # 2) Tight head/upper-body crop for SigLIP 2 eye-contact & catchlight scoring
    side_tight = int(round(max(max(hw, hh) * 1.35, min(sw, sh) * 0.65)))
    side_tight = max(min(output_size, min(h_img, w_img)), side_tight)
    side_tight = min(side_tight, min(h_img, w_img))
    tx1 = max(0, min(w_img - side_tight, hcx - side_tight // 2))
    ty1 = max(0, min(h_img - side_tight, hcy - side_tight // 2))
    tight_crop = img_bgr[ty1 : ty1 + side_tight, tx1 : tx1 + side_tight]
    if tight_crop.shape[0] != output_size or tight_crop.shape[1] != output_size:
        interp = cv2.INTER_AREA if side_tight > output_size else cv2.INTER_CUBIC
        tight_crop = cv2.resize(tight_crop, (output_size, output_size), interpolation=interp)

    return zoom_crop, tight_crop


class KestrelCoreRunner:
    """Wraps Project Kestrel's MegaDetector + SAM-HQ + BioCLIP 2 + AnimalGate + QualityClassifier."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        _init_kestrel_imports(cfg.kestrel_dir)

        from kestrel_analyzer.config import (
            MODELS_DIR,
            QUALITYCLASSIFIER_PATH,
            QUALITY_NORMALIZATION_DATA_PATH,
        )
        from kestrel_analyzer.ml.animal_gate import AnimalGate
        from kestrel_analyzer.ml.bioclip_species import BioClipSpeciesClassifier
        from kestrel_analyzer.ml.quality import QualityClassifier
        from kestrel_analyzer.ml.speciesnet_sam_hq import DetectorSAMHQWrapper

        logger.info("Initializing Project Kestrel ONNX GPU pipeline (%s)...", cfg.kestrel_detector)
        self.det_sam = DetectorSAMHQWrapper(
            max_bird_crops=5,
            use_gpu=True,
            detector_name=cfg.kestrel_detector,
        )
        self.det_sam.ensure_loaded()
        self.bio = BioClipSpeciesClassifier(MODELS_DIR, coord=self.det_sam.coord)
        self.gate = AnimalGate(MODELS_DIR / "bioclip")
        self.qual = QualityClassifier(
            str(QUALITYCLASSIFIER_PATH),
            str(QUALITY_NORMALIZATION_DATA_PATH),
            coord=self.det_sam.coord,
        )
        logger.info(
            "Project Kestrel ready | Detector=%s | SAM=%s | BioCLIP=%s | Quality=%s",
            self.det_sam.detector.device,
            self.det_sam.predictor.device,
            self.bio.session.get_providers()[0],
            self.qual.providers_used[0],
        )

    def analyze_image(
        self,
        img_rgb: np.ndarray,
        ascii_name: str,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Run MegaDetector + SAM-HQ + BioCLIP 2 + AnimalGate + Kestrel QualityClassifier
        on a single in-memory RGB uint8 numpy array.
        Returns (metadata_dict, bioclip_emb_768).
        """
        h, w = img_rgb.shape[:2]
        img_area = float(h * w)

        masks, pred_boxes, pred_class, pred_score = self.det_sam.get_prediction(
            img_rgb,
            ascii_name,
            wildlife_enabled=True,
            threshold=self.cfg.md_threshold,
        )

        best_subject: Optional[Dict[str, Any]] = None
        best_rank = -1.0
        verified_count = 0

        if masks is not None and len(masks) > 0:
            for i in range(len(masks)):
                mask = masks[i]
                q_crop, q_mask = self.det_sam.get_square_crop(mask, img_rgb, resize=True)
                emb = self.det_sam.run_with_resilience(lambda _c=q_crop: self.bio.embed(_c)[0])
                animal_score = float(self.gate.animal_score(emb))
                if not self.gate.keep_score(animal_score, self.cfg.md_threshold):
                    continue

                verified_count += 1
                route_info = self.bio.route(emb)
                if route_info["route"] == "wildlife":
                    species_label = route_info.get("wildlife_label") or "Wildlife"
                    species_conf = float(route_info.get("wildlife_confidence", 0.0))
                    family_label = route_info.get("wildlife_family") or "Wildlife"
                    animal_group = "wildlife"
                else:
                    cls_res = self.bio.classify_embedding(emb, top_k=3)
                    species_label = str(cls_res["top_species_labels"][0]) if len(cls_res["top_species_labels"]) else "Bird"
                    species_conf = float(cls_res["top_species_scores"][0]) if len(cls_res["top_species_scores"]) else 0.0
                    family_label = str(cls_res["top_family_labels"][0]) if cls_res["top_family_labels"] else "Bird"
                    animal_group = "bird"

                # Kestrel segmented sharpness percentile (0..1)
                kestrel_q = float(self.qual.classify(q_crop, q_mask))
                if kestrel_q < 0.0:
                    kestrel_q = 0.0

                (bx1, by1), (bx2, by2) = pred_boxes[i]
                bx1, by1 = max(0, int(round(bx1))), max(0, int(round(by1)))
                bx2, by2 = min(w, int(round(bx2))), min(h, int(round(by2)))
                area_frac = ((bx2 - bx1) * (by2 - by1)) / max(1.0, img_area)
                md_conf = float(pred_score[i])

                # Prioritize prominent, confident, crisp wildlife subjects
                rank_val = md_conf * (max(0.002, area_frac) ** 0.30) * (0.6 + 0.4 * animal_score)
                if rank_val > best_rank:
                    best_rank = rank_val
                    head_sharp_raw, head_bbox = compute_masked_head_sharpness(
                        img_rgb, mask, (bx1, by1, bx2, by2)
                    )
                    best_subject = {
                        "has_wildlife": True,
                        "animal_group": animal_group,
                        "species": species_label,
                        "species_conf": round(species_conf, 4),
                        "family": family_label,
                        "md_conf": round(md_conf, 4),
                        "animal_gate_score": round(animal_score, 4),
                        "kestrel_quality": round(kestrel_q, 4),
                        "head_sharp_raw": head_sharp_raw,
                        "bbox": [bx1, by1, bx2, by2],
                        "head_bbox": list(head_bbox),
                        "subject_area_frac": round(area_frac, 5),
                        "bioclip_emb": emb.astype(np.float16),
                    }

        global_metrics = compute_global_landscape_metrics(img_rgb)

        if best_subject is not None:
            best_subject["num_verified_animals"] = verified_count
            best_subject.update(global_metrics)
            emb_out = best_subject.pop("bioclip_emb")
            return best_subject, emb_out

        # No verified wildlife -> fallback BioCLIP embedding on center square for embeddings.npz
        side = min(h, w)
        cy, cx = h // 2, w // 2
        center_sq = img_rgb[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
        center_emb = self.det_sam.run_with_resilience(lambda _c=center_sq: self.bio.embed(_c)[0]).astype(np.float16)

        no_subj = {
            "has_wildlife": False,
            "animal_group": "none",
            "species": "None",
            "species_conf": 0.0,
            "family": "None",
            "md_conf": 0.0,
            "animal_gate_score": 0.0,
            "kestrel_quality": 0.0,
            "head_sharp_raw": 0.0,
            "bbox": [],
            "head_bbox": [],
            "subject_area_frac": 0.0,
            "num_verified_animals": 0,
        }
        no_subj.update(global_metrics)
        return no_subj, center_emb
