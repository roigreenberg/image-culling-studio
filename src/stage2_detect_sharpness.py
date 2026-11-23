"""Stage 2: GPU Wildlife Detection, Head/Eye Focus Localization & Sharpness Scoring."""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .config import ANIMAL_CLASS_IDS, PipelineConfig

logger = logging.getLogger(__name__)


def _saliency_fallback_bbox(img_bgr: np.ndarray) -> Tuple[int, int, int, int, float]:
    """
    Fallback subject bounding box when YOLO misses an occluded/extreme-closeup animal.
    Uses local edge/detail energy + spectral residual saliency with a mild center prior.
    Returns (x1, y1, x2, y2, pseudo_conf).
    """
    h, w = img_bgr.shape[:2]
    scale = 320.0 / max(h, w)
    sw = max(32, int(round(w * scale)))
    sh = max(32, int(round(h * scale)))
    small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Local gradient energy (high frequency foreground vs smooth bokeh)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_smooth = cv2.GaussianBlur(grad, (0, 0), sigmaX=max(3.0, min(sw, sh) * 0.06))

    # Color contrast relative to image border median
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]], axis=0)
    bg_med = np.median(border, axis=0)
    color_dist = np.linalg.norm(lab - bg_med, axis=2)
    color_smooth = cv2.GaussianBlur(color_dist, (0, 0), sigmaX=max(3.0, min(sw, sh) * 0.05))

    # Mild center prior
    yy, xx = np.mgrid[0:sh, 0:sw]
    cx, cy = sw * 0.5, sh * 0.5
    dist2 = ((xx - cx) / (sw * 0.55)) ** 2 + ((yy - cy) / (sh * 0.55)) ** 2
    center_prior = np.exp(-0.45 * dist2).astype(np.float32)

    g_norm = grad_smooth / (grad_smooth.max() + 1e-6)
    c_norm = color_smooth / (color_smooth.max() + 1e-6)
    sal = (0.65 * g_norm + 0.35 * c_norm) * (0.4 + 0.6 * center_prior)

    # Threshold top 12% saliency pixels
    thresh = np.percentile(sal, 88)
    mask = (sal >= thresh).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        # Center 40% box
        x1, y1 = int(w * 0.3), int(h * 0.3)
        x2, y2 = int(w * 0.7), int(h * 0.7)
        return x1, y1, x2, y2, 0.05

    # Pick component with highest integrated saliency
    best_label = 1
    best_score = -1.0
    for lbl in range(1, num_labels):
        score = float(sal[labels == lbl].sum())
        if score > best_score:
            best_score = score
            best_label = lbl

    bx = stats[best_label, cv2.CC_STAT_LEFT]
    by = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]

    # Expand slightly to include full subject head/body
    pad_x = int(bw * 0.25)
    pad_y = int(bh * 0.25)
    sx1 = max(0, bx - pad_x)
    sy1 = max(0, by - pad_y)
    sx2 = min(sw, bx + bw + pad_x)
    sy2 = min(sh, by + bh + pad_y)

    x1 = int(round(sx1 / scale))
    y1 = int(round(sy1 / scale))
    x2 = min(w, int(round(sx2 / scale)))
    y2 = min(h, int(round(sy2 / scale)))

    # Enforce reasonable min box size
    if (x2 - x1) < w * 0.08 or (y2 - y1) < h * 0.08:
        cx_full = (x1 + x2) // 2
        cy_full = (y1 + y2) // 2
        half_w = max(int(w * 0.12), (x2 - x1) // 2)
        half_h = max(int(h * 0.12), (y2 - y1) // 2)
        x1 = max(0, cx_full - half_w)
        y1 = max(0, cy_full - half_h)
        x2 = min(w, cx_full + half_w)
        y2 = min(h, cy_full + half_h)

    return x1, y1, x2, y2, 0.05


def locate_head_region(
    img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """
    Locate the upper/head (and likely eye) sub-region within a detected wildlife bounding box.
    Masks out bright sky/border background so sharp twigs against sky don't pull the head box
    away from the actual bird/animal head.
    """
    h_img, w_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w_img - 2, int(x1)))
    y1 = max(0, min(h_img - 2, int(y1)))
    x2 = max(x1 + 2, min(w_img, int(x2)))
    y2 = max(y1 + 2, min(h_img, int(y2)))

    bw = x2 - x1
    bh = y2 - y1
    if bw < 32 or bh < 32:
        return x1, y1, x2, y2

    crop = img_bgr[y1:y2, x1:x2]
    max_side = max(bw, bh)
    scale = min(1.0, 256.0 / max_side)
    cw = max(16, int(round(bw * scale)))
    ch = max(16, int(round(bh * scale)))
    small = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA) if scale < 1.0 else crop
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Mask out bright sky / background border colors so twigs against sky don't win over the head
    sky_mask = (gray > 215.0).astype(np.uint8)
    sky_dilated = cv2.dilate(sky_mask, np.ones((9, 9), np.uint8))
    fg_weight = np.where(sky_dilated > 0, 0.15, 1.0).astype(np.float32)

    # High-frequency detail energy (Tenengrad) + local eye/feather contrast
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy) * fg_weight

    mu = cv2.boxFilter(gray, cv2.CV_32F, (7, 7))
    mu2 = cv2.boxFilter(gray * gray, cv2.CV_32F, (7, 7))
    local_std = np.sqrt(np.maximum(0.0, mu2 - mu * mu)) * fg_weight

    # Candidate head zones (rx1, ry1, rx2, ry2, prior_weight)
    aspect = bw / float(bh)
    if aspect > 1.35:
        # Wide subject (wings spread or horizontal bird/mammal):
        # Head is very often top-center (spread wings) or upper-left / upper-right
        candidates = [
            (0.24, 0.02, 0.76, 0.56, 1.28),  # Top-center (spread wings / perched eagle)
            (0.05, 0.04, 0.52, 0.62, 1.02),  # Upper-left (facing left)
            (0.48, 0.04, 0.95, 0.62, 1.02),  # Upper-right (facing right)
            (0.06, 0.25, 0.50, 0.85, 0.88),  # Mid-left (grazing / low head)
            (0.50, 0.25, 0.94, 0.85, 0.88),  # Mid-right
        ]
    else:
        # Upright / square / close portrait wildlife
        candidates = [
            (0.14, 0.03, 0.86, 0.52, 1.25),  # Top-center
            (0.05, 0.06, 0.62, 0.60, 1.05),  # Upper-left
            (0.38, 0.06, 0.95, 0.60, 1.05),  # Upper-right
            (0.14, 0.22, 0.86, 0.76, 0.95),  # Center-face (close mammal/bird portrait)
        ]

    best_zone = candidates[0]
    best_score = -1.0
    for rx1, ry1, rx2, ry2, prior in candidates:
        px1, py1 = int(rx1 * cw), int(ry1 * ch)
        px2, py2 = max(px1 + 4, int(rx2 * cw)), max(py1 + 4, int(ry2 * ch))
        sub_g = grad[py1:py2, px1:px2]
        sub_s = local_std[py1:py2, px1:px2]
        sub_fg = fg_weight[py1:py2, px1:px2]
        if sub_g.size == 0:
            continue
        fg_coverage = float(sub_fg.mean())
        detail = float(np.percentile(sub_g, 80)) + 0.5 * float(np.percentile(sub_s, 80))
        score = detail * prior * (0.4 + 0.6 * fg_coverage)
        if score > best_score:
            best_score = score
            best_zone = (rx1, ry1, rx2, ry2, prior)

    rx1, ry1, rx2, ry2, _ = best_zone
    hx1 = x1 + int(round(rx1 * bw))
    hy1 = y1 + int(round(ry1 * bh))
    hx2 = x1 + int(round(rx2 * bw))
    hy2 = y1 + int(round(ry2 * bh))

    hx1 = max(x1, min(w_img - 2, hx1))
    hy1 = max(y1, min(h_img - 2, hy1))
    hx2 = max(hx1 + 2, min(x2, hx2))
    hy2 = max(hy1 + 2, min(y2, hy2))
    return hx1, hy1, hx2, hy2


def compute_subject_sharpness(
    img_bgr: np.ndarray,
    subject_bbox: Tuple[int, int, int, int],
    head_bbox: Tuple[int, int, int, int],
    iso: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute contrast- and exposure-normalized sharpness strictly on the head/subject crop.
    Uses uniform gentle smoothing across all ISOs (no ISO cliff) and excludes blown bright sky
    tiles so silhouette twigs against sky don't overpower true subject eye/feather sharpness.
    """
    h_img, w_img = img_bgr.shape[:2]

    def _measure_region(box: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        x1, y1, x2, y2 = box
        x1 = max(0, min(w_img - 2, x1))
        y1 = max(0, min(h_img - 2, y1))
        x2 = max(x1 + 2, min(w_img, x2))
        y2 = max(y1 + 2, min(h_img, y2))

        crop = img_bgr[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 16 or cw < 16:
            return 0.0, 0.0, 0.0

        # Cap huge close-up crops at 768px so scale-dependent gradients stay consistent
        max_side = max(ch, cw)
        if max_side > 768:
            s = 768.0 / max_side
            crop = cv2.resize(crop, (int(round(cw * s)), int(round(ch * s))), interpolation=cv2.INTER_AREA)
            ch, cw = crop.shape[:2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Uniform gentle noise suppression across all ISOs
        gray_eval = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.5)

        gx = cv2.Sobel(gray_eval, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_eval, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad_map = gx * gx + gy * gy

        lap = cv2.Laplacian(gray_eval, cv2.CV_32F, ksize=3)
        lap_sq = lap * lap

        tile = max(16, min(48, min(ch, cw) // 4))
        stride = max(8, tile // 2)

        # Vectorized C++ boxFilter computes exact tile stats for the entire crop in <0.5ms
        mean_g = cv2.boxFilter(gray, cv2.CV_32F, (tile, tile))
        mean_g2 = cv2.boxFilter(gray * gray, cv2.CV_32F, (tile, tile))
        var_g = np.maximum(0.0, mean_g2 - mean_g * mean_g)
        sky_frac_map = cv2.boxFilter((gray > 220.0).astype(np.float32), cv2.CV_32F, (tile, tile))
        mean_ten = cv2.boxFilter(tenengrad_map, cv2.CV_32F, (tile, tile))
        mean_lap = cv2.boxFilter(lap_sq, cv2.CV_32F, (tile, tile))

        # Sample interior grid points where full tile fits inside crop
        pad = tile // 2
        if ch > tile and cw > tile:
            ys = slice(pad, ch - pad, stride)
            xs = slice(pad, cw - pad, stride)
            s_sky = sky_frac_map[ys, xs].ravel()
            s_var = var_g[ys, xs].ravel()
            s_luma = mean_g[ys, xs].ravel()
            s_ten = mean_ten[ys, xs].ravel()
            s_lap = mean_lap[ys, xs].ravel()
            valid = s_sky <= 0.35
            if np.any(valid):
                norm_denom = s_var[valid] + 180.0
                ten_arr = s_ten[valid] / norm_denom
                lap_arr = s_lap[valid] / norm_denom
                luma_arr = s_luma[valid]
                k = max(1, int(round(len(ten_arr) * 0.25)))
                top_idx = np.argpartition(ten_arr, -k)[-k:]
                return (
                    float(np.mean(ten_arr[top_idx])),
                    float(np.mean(lap_arr[top_idx])),
                    float(np.mean(luma_arr[top_idx])),
                )

        p_std = float(gray.std())
        norm_denom = (p_std * p_std) + 180.0
        return (
            float(tenengrad_map.mean()) / norm_denom,
            float(lap_sq.mean()) / norm_denom,
            float(gray.mean()),
        )

    head_ten, head_lap, head_luma = _measure_region(head_bbox)
    subj_ten, subj_lap, _ = _measure_region(subject_bbox)

    exp_factor = 1.0
    if head_luma < 16.0:
        exp_factor = max(0.60, head_luma / 16.0)
    elif head_luma > 242.0:
        exp_factor = max(0.60, (255.0 - head_luma) / 13.0)

    head_comb = np.log1p(head_ten * 8.0) * 0.65 + np.log1p(head_lap * 14.0) * 0.35
    subj_comb = np.log1p(subj_ten * 8.0) * 0.65 + np.log1p(subj_lap * 14.0) * 0.35
    raw_sharpness = float((0.75 * head_comb + 0.25 * subj_comb) * exp_factor)

    return {
        "sharpness_raw": round(raw_sharpness, 5),
        "head_tenengrad": round(float(head_ten), 5),
        "head_laplacian": round(float(head_lap), 5),
        "head_luma": round(float(head_luma), 2),
    }


def extract_zoom_and_siglip_crop(
    img_bgr: np.ndarray,
    subject_bbox: Tuple[int, int, int, int],
    head_bbox: Tuple[int, int, int, int],
    output_size: int = 384,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      1. zoom_crop_bgr (output_size x output_size): High-magnification subject/head crop
         (true 1:1 native pixels for small/medium birds, and adaptive tight head zoom
         for huge frame-filling 45MP close-ups so the head & eye are always clearly visible).
      2. tight_head_bgr (output_size x output_size): Clean head+upper-body crop for
         Stage 3 SigLIP eye-contact / catchlight / branch-occlusion scoring.
    """
    h_img, w_img = img_bgr.shape[:2]
    hx1, hy1, hx2, hy2 = head_bbox
    sx1, sy1, sx2, sy2 = subject_bbox

    hcx = (hx1 + hx2) // 2
    hcy = (hy1 + hy2) // 2
    hw = max(1, hx2 - hx1)
    hh = max(1, hy2 - hy1)
    sw = max(1, sx2 - sx1)
    sh = max(1, sy2 - sy1)

    # --- 1) Gallery 1:1 / Head Zoom Crop ---
    # For small/medium birds: native 384px (true 1:1 pixels).
    # For large frame-filling animals (e.g., head > 700px on 11MP preview): crop ~65% of head_bbox
    # (capped at 1200px) so the actual eye + beak/face fits in the zoom preview instead of 5 feathers!
    desired_zoom = int(round(max(output_size, min(1150, max(hw, hh) * 0.65))))
    crop_side_zoom = min(desired_zoom, h_img, w_img)
    half = crop_side_zoom // 2
    zx1 = max(0, min(w_img - crop_side_zoom, hcx - half))
    zy1 = max(0, min(h_img - crop_side_zoom, hcy - half))
    zoom_crop = img_bgr[zy1 : zy1 + crop_side_zoom, zx1 : zx1 + crop_side_zoom]
    if zoom_crop.shape[0] != output_size or zoom_crop.shape[1] != output_size:
        interp = cv2.INTER_AREA if crop_side_zoom > output_size else cv2.INTER_CUBIC
        zoom_crop = cv2.resize(zoom_crop, (output_size, output_size), interpolation=interp)

    # --- 2) Tight bird/head crop for Stage 3 SigLIP ---
    # Generous head + upper-body framing so SigLIP reliably sees the eye, face, and foreground twigs
    side_tight = int(round(max(max(hw, hh) * 1.15, min(sw, sh) * 0.75)))
    side_tight = max(min(output_size, min(h_img, w_img)), side_tight)
    side_tight = min(side_tight, min(h_img, w_img))

    thalf = side_tight // 2
    tx1 = max(0, min(w_img - side_tight, hcx - thalf))
    ty1 = max(0, min(h_img - side_tight, hcy - thalf))
    tight_crop = img_bgr[ty1 : ty1 + side_tight, tx1 : tx1 + side_tight]
    if tight_crop.shape[0] != output_size or tight_crop.shape[1] != output_size:
        interp = cv2.INTER_AREA if side_tight > output_size else cv2.INTER_CUBIC
        tight_crop = cv2.resize(tight_crop, (output_size, output_size), interpolation=interp)

    return zoom_crop, tight_crop


class WildlifeDetector:
    """GPU-accelerated YOLOv11 wildlife detector for RTX 4060."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device.startswith("cuda")
        logger.info("Loading YOLO model %s on %s (fp16=%s)", cfg.yolo_model, self.device, self.use_half)
        logging.getLogger("ultralytics").setLevel(logging.ERROR)
        self.model = YOLO(cfg.yolo_model)
        # Warm up on GPU
        if self.use_half:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            with torch.no_grad():
                self.model.predict(
                    [dummy],
                    imgsz=cfg.yolo_imgsz,
                    conf=cfg.yolo_conf,
                    device=self.device,
                    half=True,
                    verbose=False,
                )

    def detect_batch(
        self,
        images_bgr: List[np.ndarray],
        pre_resized: Optional[List[Tuple[np.ndarray, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run batched GPU YOLO inference on a list of BGR images.
        Optionally accepts pre_resized = [(resized_bgr, inv_scale), ...] computed off-thread.
        """
        if not images_bgr:
            return []

        if pre_resized is not None:
            resized_inputs = [x[0] for x in pre_resized]
            scales = [x[1] for x in pre_resized]
        else:
            resized_inputs = []
            scales = []
            target_max = max(self.cfg.yolo_imgsz, 960)
            for img in images_bgr:
                h, w = img.shape[:2]
                longest = max(h, w)
                if longest > target_max:
                    s = target_max / float(longest)
                    resized = cv2.resize(
                        img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA
                    )
                    resized_inputs.append(resized)
                    scales.append(1.0 / s)
                else:
                    resized_inputs.append(img)
                    scales.append(1.0)

        with torch.no_grad():
            preds = self.model.predict(
                resized_inputs,
                imgsz=self.cfg.yolo_imgsz,
                conf=self.cfg.yolo_conf,
                classes=list(ANIMAL_CLASS_IDS.keys()),
                device=self.device,
                half=self.use_half,
                verbose=False,
            )

        outputs: List[Dict[str, Any]] = []
        for img, pred, inv_scale in zip(images_bgr, preds, scales):
            h, w = img.shape[:2]
            img_area = float(h * w)
            boxes = pred.boxes

            best_det = None
            best_rank_score = -1.0
            animal_count = 0

            if boxes is not None and len(boxes) > 0:
                xyxy_all = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                clss = boxes.cls.detach().cpu().numpy().astype(int)

                for xyxy, conf, cls_id in zip(xyxy_all, confs, clss):
                    if cls_id not in ANIMAL_CLASS_IDS:
                        continue
                    animal_count += 1
                    x1 = max(0, min(w - 2, int(round(xyxy[0] * inv_scale))))
                    y1 = max(0, min(h - 2, int(round(xyxy[1] * inv_scale))))
                    x2 = max(x1 + 2, min(w, int(round(xyxy[2] * inv_scale))))
                    y2 = max(y1 + 2, min(h, int(round(xyxy[3] * inv_scale))))

                    area_frac = ((x2 - x1) * (y2 - y1)) / max(1.0, img_area)
                    # Mild preference for birds & central/prominent subjects
                    cx = ((x1 + x2) * 0.5) / w
                    cy = ((y1 + y2) * 0.5) / h
                    center_dist = np.hypot(cx - 0.5, cy - 0.5)
                    center_w = max(0.65, 1.0 - 0.5 * center_dist)
                    bird_bonus = 1.15 if cls_id == 14 else 1.0
                    rank_score = float(conf) * (max(0.002, area_frac) ** 0.35) * center_w * bird_bonus

                    if rank_score > best_rank_score:
                        best_rank_score = rank_score
                        best_det = {
                            "bbox": [x1, y1, x2, y2],
                            "detected_class": ANIMAL_CLASS_IDS[cls_id],
                            "detection_conf": round(float(conf), 4),
                            "subject_area_frac": round(float(area_frac), 5),
                            "detection_source": "yolo",
                            "num_animals": animal_count,
                        }

            if best_det is not None:
                best_det["num_animals"] = animal_count
                outputs.append(best_det)
            else:
                fx1, fy1, fx2, fy2, fconf = _saliency_fallback_bbox(img)
                area_frac = ((fx2 - fx1) * (fy2 - fy1)) / max(1.0, img_area)
                outputs.append(
                    {
                        "bbox": [fx1, fy1, fx2, fy2],
                        "detected_class": "saliency_subject",
                        "detection_conf": round(float(fconf), 4),
                        "subject_area_frac": round(float(area_frac), 5),
                        "detection_source": "saliency_fallback",
                        "num_animals": 0,
                    }
                )

        return outputs

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
