"""Auto Smart Straighten (multi-horizon) & Bird-Aware Smart Crop engine.

Produces Lightroom-compatible normalized crop coordinates (0..1) and rotation
angles (degrees, positive = clockwise crop box / CCW image rotation), while
enforcing a minimum long-side pixel floor (default 2700px) and locked aspect
ratio by default.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rawpy
from PIL import Image

from .config import RAW_EXTENSIONS
from .io_utils import load_preview_bgr_and_meta

logger = logging.getLogger(__name__)

DEFAULT_MIN_LONG_PX = 2700


def _safe_is_file(path: Optional[Path]) -> bool:
    if not path:
        return False
    try:
        return path.is_file()
    except OSError:
        return False


def read_original_dimensions(
    source_path: Optional[Path],
    fallback_w: int = 0,
    fallback_h: int = 0,
) -> Tuple[int, int]:
    """
    Read the true full-resolution pixel dimensions (width, height) of source_path
    strictly read-only without full RAW demosaicing. Falls back to (fallback_w, fallback_h).
    """
    if _safe_is_file(source_path):
        assert source_path is not None
        ext = source_path.suffix.lower()
        try:
            if ext in RAW_EXTENSIONS:
                with open(source_path, "rb") as fh:
                    with rawpy.imread(fh) as raw:
                        w = int(getattr(raw.sizes, "iwidth", 0) or getattr(raw.sizes, "width", 0))
                        h = int(getattr(raw.sizes, "iheight", 0) or getattr(raw.sizes, "height", 0))
                        flip = int(getattr(raw.sizes, "flip", 0) or 0)
                        # LibRaw flip 5 (90 CCW) or 6 (90 CW) swaps width & height
                        if flip in (5, 6):
                            w, h = h, w
                        if w > 0 and h > 0:
                            return w, h
            else:
                with Image.open(source_path) as im:
                    w, h = im.size
                    try:
                        exif = im.getexif()
                        orient = exif.get(274, 1) if exif else 1
                        if orient in (5, 6, 7, 8):
                            w, h = h, w
                    except Exception:
                        pass
                    if w > 0 and h > 0:
                        return int(w), int(h)
        except Exception as exc:
            logger.debug("Could not read full dimensions for %s: %s", source_path, exc)

    return max(0, int(fallback_w or 0)), max(0, int(fallback_h or 0))


def clamp_crop_to_image(
    left: float,
    top: float,
    right: float,
    bottom: float,
    angle_deg: float,
    aspect_w_over_h: float,
    locked_ratio: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Ensure a normalized crop rectangle [left, top, right, bottom] rotated by
    `angle_deg` stays completely inside [0, 1] x [0, 1] (zero black corners).

    `aspect_w_over_h` is the image's pixel aspect ratio (orig_w / orig_h).
    If `locked_ratio` is provided (desired pixel W/H of the crop), the returned
    box preserves that exact pixel aspect ratio.
    """
    R = max(0.05, float(aspect_w_over_h or 1.5))
    left = max(0.0, min(1.0, float(left)))
    right = max(0.0, min(1.0, float(right)))
    top = max(0.0, min(1.0, float(top)))
    bottom = max(0.0, min(1.0, float(bottom)))
    if right <= left:
        left, right = 0.0, 1.0
    if bottom <= top:
        top, bottom = 0.0, 1.0

    w = right - left
    h = bottom - top
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)

    if locked_ratio and locked_ratio > 0:
        # Pixel ratio of crop is (w * R) / h == locked_ratio -> w = h * (locked_ratio / R)
        norm_ratio = float(locked_ratio) / R
        cur_area = w * h
        h = math.sqrt(max(1e-6, cur_area / max(1e-6, norm_ratio)))
        w = h * norm_ratio

    rad = math.radians(float(angle_deg or 0.0))
    c = abs(math.cos(rad))
    s = abs(math.sin(rad))

    # Axis-aligned half-extents of the rotated box in normalized image coords:
    # half_x = 0.5 * (w * c + (h / R) * s)
    # half_y = 0.5 * (w * R * s + h * c)
    half_x = 0.5 * (w * c + (h / R) * s)
    half_y = 0.5 * (w * R * s + h * c)

    # If the rotated box is larger than the image in X or Y, shrink uniformly to fit
    scale = 1.0
    if half_x > 0.5:
        scale = min(scale, 0.5 / half_x)
    if half_y > 0.5:
        scale = min(scale, 0.5 / half_y)
    if scale < 1.0:
        w *= scale
        h *= scale
        half_x *= scale
        half_y *= scale

    # Clamp center so all 4 rotated corners sit inside [0, 1] x [0, 1]
    cx = max(half_x, min(1.0 - half_x, cx))
    cy = max(half_y, min(1.0 - half_y, cy))

    n_left = max(0.0, cx - 0.5 * w)
    n_right = min(1.0, cx + 0.5 * w)
    n_top = max(0.0, cy - 0.5 * h)
    n_bottom = min(1.0, cy + 0.5 * h)
    return (
        round(n_left, 6),
        round(n_top, 6),
        round(n_right, 6),
        round(n_bottom, 6),
    )


def detect_horizons(
    bgr: np.ndarray,
    subject_bbox_norm: Optional[Sequence[float]] = None,
    max_angle_deg: float = 12.0,
    max_candidates: int = 5,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    Detect all plausible horizon / waterline candidates in an image.

    Returns (candidates, selected_idx):
    - `candidates`: list of dicts sorted by confidence desc:
        {id, angle, x1, y1, x2, y2, y_norm, confidence, span, source, label}
      where `angle` is in degrees (positive = horizon slopes down left-to-right,
      matching Lightroom `crs:CropAngle` sign convention).
    - `selected_idx`: index into `candidates` of the auto-selected best horizon,
      or None if no confident tilted horizon was found.
    """
    if bgr is None or bgr.size == 0:
        return [], None

    h_orig, w_orig = bgr.shape[:2]
    if h_orig < 32 or w_orig < 32:
        return [], None

    # Resize to ~800px wide for fast, consistent edge detection
    target_w = min(800, w_orig)
    scale = target_w / float(w_orig)
    target_h = max(32, int(round(h_orig * scale)))
    small = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Mask out the subject bbox (expanded slightly) so wings/beaks/perches aren't detected as horizons
    valid_mask = np.ones((target_h, target_w), dtype=np.uint8) * 255
    if subject_bbox_norm and len(subject_bbox_norm) == 4:
        sx1, sy1, sx2, sy2 = [float(v) for v in subject_bbox_norm]
        pad = 0.03
        px1 = max(0, int((sx1 - pad) * target_w))
        py1 = max(0, int((sy1 - pad) * target_h))
        px2 = min(target_w, int((sx2 + pad) * target_w))
        py2 = min(target_h, int((sy2 + pad) * target_h))
        # Only mask if the subject doesn't cover >85% of the frame
        if (px2 - px1) * (py2 - py1) < 0.85 * target_w * target_h:
            valid_mask[py1:py2, px1:px2] = 0

    raw_candidates: List[Dict[str, Any]] = []

    # --- Method A: Horizontal-edge Hough lines (waterlines, shores, distant ridges) ---
    # Emphasize vertical intensity transitions (horizontal lines) using Sobel dy
    dy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    dx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    horiz_resp = np.maximum(0.0, np.abs(dy) - 0.55 * np.abs(dx))
    max_val = float(np.percentile(horiz_resp, 99.5)) if horiz_resp.size else 0.0
    if max_val > 6.0:
        norm_h = np.clip((horiz_resp / max_val) * 255.0, 0, 255).astype(np.uint8)
        edges = cv2.Canny(norm_h, 40, 110)
        edges = cv2.bitwise_and(edges, valid_mask)
        # Ignore extreme 4% top/bottom border artifacts
        border_y = max(4, int(0.04 * target_h))
        edges[:border_y, :] = 0
        edges[target_h - border_y :, :] = 0

        min_len = max(40, int(0.14 * target_w))
        max_gap = max(12, int(0.045 * target_w))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360.0,  # 0.5 degree resolution
            threshold=max(35, int(0.08 * target_w)),
            minLineLength=min_len,
            maxLineGap=max_gap,
        )
        if lines is not None:
            segs = []
            for ln in lines.reshape(-1, 4):
                x1, y1, x2, y2 = [float(v) for v in ln]
                if x2 < x1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                dx_l = x2 - x1
                dy_l = y2 - y1
                if dx_l < min_len * 0.8:
                    continue
                ang = math.degrees(math.atan2(dy_l, dx_l))
                if abs(ang) > max_angle_deg:
                    continue
                length = math.hypot(dx_l, dy_l)
                # y at center x = target_w / 2
                slope = dy_l / dx_l
                y_mid = y1 + slope * (0.5 * target_w - x1)
                segs.append((x1, y1, x2, y2, ang, y_mid, length))

            # Cluster collinear segments (similar angle & y_mid)
            segs.sort(key=lambda s: s[6], reverse=True)
            clusters: List[List[Tuple[float, ...]]] = []
            for seg in segs:
                placed = False
                for cl in clusters:
                    ref = cl[0]
                    if abs(seg[4] - ref[4]) <= 1.2 and abs(seg[5] - ref[5]) <= 0.045 * target_h:
                        cl.append(seg)
                        placed = True
                        break
                if not placed:
                    clusters.append([seg])

            for cl in clusters:
                # Fit weighted line across all segment endpoints in cluster
                xs = []
                ys = []
                ws = []
                total_len = 0.0
                min_x = target_w
                max_x = 0.0
                for x1, y1, x2, y2, _, _, length in cl:
                    xs.extend([x1, x2])
                    ys.extend([y1, y2])
                    ws.extend([length, length])
                    total_len += length
                    min_x = min(min_x, x1)
                    max_x = max(max_x, x2)
                xs_arr = np.array(xs, dtype=np.float64)
                ys_arr = np.array(ys, dtype=np.float64)
                ws_arr = np.array(ws, dtype=np.float64)
                if np.ptp(xs_arr) < 0.14 * target_w:
                    continue
                slope, intercept = np.polyfit(xs_arr, ys_arr, 1, w=ws_arr)
                ang = math.degrees(math.atan(slope))
                if abs(ang) > max_angle_deg:
                    continue
                resid = float(np.mean(np.abs(ys_arr - (slope * xs_arr + intercept))))
                span_frac = min(1.0, (max_x - min_x) / float(target_w))
                coverage = min(1.0, total_len / float(target_w))

                # Evaluate mean vertical gradient strength along the line
                sample_xs = np.linspace(max(2, min_x), min(target_w - 3, max_x), 48)
                sample_ys = np.clip(slope * sample_xs + intercept, 2, target_h - 3)
                strengths = [
                    float(horiz_resp[int(round(sy)), int(round(sx))])
                    for sx, sy in zip(sample_xs, sample_ys)
                    if valid_mask[int(round(sy)), int(round(sx))] > 0
                ]
                edge_score = min(1.0, (float(np.mean(strengths)) / (max_val * 0.45))) if strengths else 0.0
                straight_score = max(0.0, 1.0 - resid / 6.0)
                angle_prior = max(0.25, 1.0 - (abs(ang) / max_angle_deg) * 0.6)

                conf = (
                    0.40 * span_frac
                    + 0.25 * coverage
                    + 0.20 * edge_score
                    + 0.15 * straight_score
                ) * angle_prior

                y_left = float(max(0.02, min(0.98, intercept / float(target_h))))
                y_right = float(max(0.02, min(0.98, (slope * target_w + intercept) / float(target_h))))
                y_center = float(0.5 * (y_left + y_right))

                raw_candidates.append(
                    {
                        "angle": round(float(ang), 2),
                        "x1": 0.0,
                        "y1": round(y_left, 4),
                        "x2": 1.0,
                        "y2": round(y_right, 4),
                        "y_norm": round(y_center, 4),
                        "confidence": round(float(conf), 3),
                        "span": round(float(span_frac), 3),
                        "source": "waterline",
                    }
                )

    # --- Method B: Column-wise strongest skyline/water-boundary RANSAC ---
    # Helpful on soft-bokeh horizons (telephoto wildlife) where Canny edges break up
    grad_y = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (9, 9), 0), cv2.CV_32F, 0, 1, ksize=5))
    border_y = max(6, int(0.06 * target_h))
    col_step = max(4, target_w // 80)
    pts_x = []
    pts_y = []
    pts_s = []
    for cx in range(col_step, target_w - col_step, col_step):
        col_mask = valid_mask[border_y : target_h - border_y, cx] > 0
        if not np.any(col_mask):
            continue
        col_g = grad_y[border_y : target_h - border_y, cx].copy()
        col_g[~col_mask] = 0.0
        idx = int(np.argmax(col_g))
        val = float(col_g[idx])
        if val > 18.0:
            pts_x.append(float(cx))
            pts_y.append(float(idx + border_y))
            pts_s.append(val)

    if len(pts_x) >= 14:
        px = np.array(pts_x, dtype=np.float64)
        py = np.array(pts_y, dtype=np.float64)
        best_inliers = None
        best_count = 0
        tol = max(4.0, 0.015 * target_h)
        # Deterministic pair sampling across evenly spaced points
        n_pts = len(px)
        step_i = max(1, n_pts // 16)
        for i in range(0, n_pts - 2, step_i):
            for j in range(i + 2, n_pts, step_i):
                if abs(px[j] - px[i]) < 0.2 * target_w:
                    continue
                sl = (py[j] - py[i]) / (px[j] - px[i])
                if abs(math.degrees(math.atan(sl))) > max_angle_deg:
                    continue
                ic = py[i] - sl * px[i]
                inl = np.abs(py - (sl * px + ic)) <= tol
                cnt = int(np.sum(inl))
                if cnt > best_count:
                    best_count = cnt
                    best_inliers = inl

        if best_inliers is not None and best_count >= max(10, int(0.32 * n_pts)):
            ix = px[best_inliers]
            iy = py[best_inliers]
            span_frac = float(np.ptp(ix) / float(target_w))
            if span_frac >= 0.35:
                slope, intercept = np.polyfit(ix, iy, 1)
                ang = math.degrees(math.atan(slope))
                if abs(ang) <= max_angle_deg:
                    inlier_ratio = best_count / float(n_pts)
                    angle_prior = max(0.25, 1.0 - (abs(ang) / max_angle_deg) * 0.6)
                    conf = (0.50 * inlier_ratio + 0.50 * span_frac) * 0.85 * angle_prior
                    y_left = float(max(0.02, min(0.98, intercept / float(target_h))))
                    y_right = float(max(0.02, min(0.98, (slope * target_w + intercept) / float(target_h))))
                    raw_candidates.append(
                        {
                            "angle": round(float(ang), 2),
                            "x1": 0.0,
                            "y1": round(y_left, 4),
                            "x2": 1.0,
                            "y2": round(y_right, 4),
                            "y_norm": round(float(0.5 * (y_left + y_right)), 4),
                            "confidence": round(float(conf), 3),
                            "span": round(float(span_frac), 3),
                            "source": "skyline",
                        }
                    )

    # Deduplicate similar candidates (keep highest confidence)
    raw_candidates.sort(key=lambda c: c["confidence"], reverse=True)
    deduped: List[Dict[str, Any]] = []
    for cand in raw_candidates:
        if cand["confidence"] < 0.16:
            continue
        is_dup = False
        for ex in deduped:
            if abs(cand["angle"] - ex["angle"]) < 0.45 and abs(cand["y_norm"] - ex["y_norm"]) < 0.06:
                is_dup = True
                break
        if not is_dup:
            cand_id = len(deduped)
            src_lbl = "Waterline" if cand["source"] == "waterline" else "Horizon"
            sign = "+" if cand["angle"] > 0 else ""
            cand["id"] = cand_id
            cand["label"] = f"{src_lbl} {cand_id + 1} ({sign}{cand['angle']:.1f}°)"
            deduped.append(cand)
            if len(deduped) >= max_candidates:
                break

    # Auto-select the strongest horizon only when confident and tilt is noticeable (0.25°..8.0°)
    selected_idx: Optional[int] = None
    if deduped:
        top = deduped[0]
        if top["confidence"] >= 0.40 and 0.25 <= abs(top["angle"]) <= 8.0:
            selected_idx = 0

    return deduped, selected_idx


def _compute_clutter_map(bgr: Optional[np.ndarray], subject_bbox_norm: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """Compute a fast normalized edge-energy map (background clutter outside subject)."""
    if bgr is None or bgr.size == 0:
        return None
    small = cv2.resize(bgr, (240, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    if subject_bbox_norm and len(subject_bbox_norm) == 4:
        sh, sw = mag.shape[:2]
        x1 = max(0, int(subject_bbox_norm[0] * sw))
        y1 = max(0, int(subject_bbox_norm[1] * sh))
        x2 = min(sw, int(subject_bbox_norm[2] * sw))
        y2 = min(sh, int(subject_bbox_norm[3] * sh))
        mag[y1:y2, x1:x2] = 0.0
    p95 = float(np.percentile(mag, 95)) if mag.size else 1.0
    return np.clip(mag / max(10.0, p95), 0.0, 1.0)


def _choose_best_wildlife_crop(
    bbox_norm: Sequence[float],
    head_norm: Optional[Sequence[float]],
    img_aspect: float,
    desired_aspect: float,
    orig_width: int,
    orig_height: int,
    min_long_px: int,
    angle_deg: float,
    aspect_locked: bool,
    bgr: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, str, float]:
    """
    Evaluate multiple real photography compositions (Rule of Thirds Left/Right,
    Golden Ratio, Upper-Third Center, Symmetric Center) across candidate scales
    and return the highest-scoring (left, top, right, bottom, rule_name, bird_size_frac).
    """
    bx1, by1, bx2, by2 = [float(v) for v in bbox_norm]
    bw = max(1e-4, bx2 - bx1)
    bh = max(1e-4, by2 - by1)
    scx = 0.5 * (bx1 + bx2)
    scy = 0.5 * (by1 + by2)
    bird_size_frac = max(bw, bh)
    subj_ar = (bw * img_aspect) / max(1e-4, bh)  # true pixel W/H of the bird bbox

    # Estimate true eye/head anchor & gaze direction
    if head_norm and len(head_norm) == 4:
        hx1, hy1, hx2, hy2 = [float(v) for v in head_norm]
        eye_x = 0.5 * (hx1 + hx2)
        # For upright/perched birds, eye sits in the upper ~30% of the head region;
        # for wide horizontal flying birds, eye sits near the vertical center of head_bbox.
        if subj_ar < 1.65:
            eye_y = min(hy1 + 0.25 * (hy2 - hy1), by1 + 0.10 * bh)
        else:
            eye_y = 0.5 * (hy1 + hy2)
    else:
        eye_x = scx
        eye_y = by1 + (0.20 * bh if subj_ar < 1.65 else 0.45 * bh)

    # Detect gaze direction (-1 = facing left, +1 = facing right, 0 = head-on/symmetric)
    gaze = 0
    if head_norm and subj_ar < 2.2:
        rel_head_offset = (eye_x - scx) / max(1e-4, bw)
        if rel_head_offset <= -0.045:
            gaze = -1
        elif rel_head_offset >= 0.045:
            gaze = 1

    orig_long = max(orig_width, orig_height)
    effective_min_long_px = min(int(min_long_px), orig_long)
    min_long_frac = effective_min_long_px / float(orig_long)

    # Minimum normalized crop size to hold the entire bird without clipping wings/tail
    # Use gentler padding when the bird already spans most of the frame
    pad_x = 0.03 if bw >= 0.80 else max(0.08 * bw, 0.035)
    pad_y = 0.03 if bh >= 0.80 else max(0.10 * bh, 0.04)
    req_w = min(1.0, bw + 2.0 * pad_x)
    req_h = min(1.0, bh + 2.0 * pad_y)

    # Convert a target long-side fraction to (crop_w, crop_h) honoring desired_aspect
    def _dims_from_long(long_frac: float) -> Tuple[float, float]:
        lf = max(min_long_frac, min(1.0, long_frac))
        if orig_width >= orig_height:
            if desired_aspect >= 1.0:
                cw = lf
                ch = (cw * img_aspect) / desired_aspect
            else:
                ch = min(1.0, lf * img_aspect)
                cw = (ch * desired_aspect) / img_aspect
        else:
            if desired_aspect <= 1.0:
                ch = lf
                cw = (ch * desired_aspect) / img_aspect
            else:
                cw = min(1.0, lf / img_aspect)
                ch = (cw * img_aspect) / desired_aspect
        if cw > 1.0:
            ch /= cw
            cw = 1.0
        if ch > 1.0:
            cw /= ch
            ch = 1.0
        return cw, ch

    # Compute the smallest long-side fraction that encloses (req_w, req_h) cleanly
    if orig_width >= orig_height and desired_aspect >= 1.0:
        min_fit_long = max(req_w, req_h * (desired_aspect / img_aspect))
    else:
        min_fit_long = max(req_w, req_h)
    base_long = max(min_long_frac, min(1.0, min_fit_long))

    # Build 3 candidate scales: tight, balanced, and wide (all >= 2700px & enclosing bird)
    if bird_size_frac >= 0.78:
        scale_candidates = [1.0]
    elif bird_size_frac >= 0.55:
        scale_candidates = sorted({round(min(1.0, max(base_long, s)), 3) for s in (base_long, base_long * 1.12, 1.0)})
    elif bird_size_frac >= 0.28:
        scale_candidates = sorted(
            {round(min(1.0, max(base_long, s)), 3) for s in (bird_size_frac / 0.56, bird_size_frac / 0.45, base_long * 1.18)}
        )
    else:
        scale_candidates = sorted(
            {round(min(1.0, max(base_long, s)), 3) for s in (min_long_frac, bird_size_frac / 0.40, bird_size_frac / 0.30)}
        )

    # Candidate target anchor points (tx, ty) for the bird's eye inside the cropped frame
    # plus human-readable composition rule name
    anchor_specs = [
        ("Rule of Thirds (Left)", 1.0 / 3.0, 1.0 / 3.0),
        ("Rule of Thirds (Right)", 2.0 / 3.0, 1.0 / 3.0),
        ("Golden Ratio (Left)", 0.382, 0.382),
        ("Golden Ratio (Right)", 0.618, 0.382),
        ("Upper-Third Center", 0.50, 0.36),
        ("Centered Subject", 0.50, 0.46),
    ]

    clutter_map = _compute_clutter_map(bgr, bbox_norm)
    locked_r = desired_aspect if aspect_locked else None

    best_score = -1e9
    best_result = (0.0, 0.0, 1.0, 1.0, "Full Frame")

    for lf in scale_candidates:
        cw, ch = _dims_from_long(lf)
        # If crop is essentially full-frame, just evaluate full frame once
        if cw >= 0.98 and ch >= 0.98:
            l, t, r, b = clamp_crop_to_image(0.0, 0.0, 1.0, 1.0, angle_deg, img_aspect, locked_ratio=locked_r)
            candidates_for_scale = [("Full Frame", l, t, r, b)]
        else:
            candidates_for_scale = []
            for rule_name, tx, ty in anchor_specs:
                if rule_name == "Centered Subject":
                    # Center the bird body symmetrically (slightly above exact middle for natural weight)
                    l = scx - 0.5 * cw
                    t = scy - 0.47 * ch
                else:
                    # Place the eye at (tx, ty) inside the crop
                    l = eye_x - tx * cw
                    t = eye_y - ty * ch
                r = l + cw
                b = t + ch
                l, t, r, b = clamp_crop_to_image(l, t, r, b, angle_deg, img_aspect, locked_ratio=locked_r)
                candidates_for_scale.append((rule_name, l, t, r, b))

        for rule_name, l, t, r, b in candidates_for_scale:
            w = max(1e-4, r - l)
            h = max(1e-4, b - t)

            # 1) Strict subject containment & breathing margin
            margin_l = bx1 - l
            margin_r = r - bx2
            margin_t = by1 - t
            margin_b = b - by2
            min_margin = min(margin_l, margin_r, margin_t, margin_b)
            if min_margin < -0.005:
                # Never accept a crop that slices off wings/tail/head
                score = -500.0 + min_margin * 1000.0
            else:
                score = 0.0
                # Reward comfortable breathing room around edges of subject
                if min_margin < 0.025:
                    score -= (0.025 - min_margin) * 400.0

                # Actual eye & body positions inside this clamped crop [0..1]
                rel_eye_x = (eye_x - l) / w
                rel_eye_y = (eye_y - t) / h
                rel_scx = (scx - l) / w
                rel_scy = (scy - t) / h

                # 2) Ideal photographic composition alignment (Rule of Thirds vs Centered Symmetric)
                is_wide_symmetric = (subj_ar >= 1.9) or (gaze == 0)
                if gaze == -1:
                    # Facing left -> best on RIGHT third (x ~ 0.64) and upper third (y ~ 0.34)
                    dist_target = math.hypot((rel_eye_x - 0.64) * 1.25, (rel_eye_y - 0.34) * 1.1)
                    score += max(0.0, 42.0 - dist_target * 95.0)
                    # Lead room bonus: more open space on left than right
                    lead_diff = (margin_l - margin_r) / w
                    score += max(-18.0, min(18.0, lead_diff * 55.0))
                elif gaze == 1:
                    # Facing right -> best on LEFT third (x ~ 0.36) and upper third (y ~ 0.34)
                    dist_target = math.hypot((rel_eye_x - 0.36) * 1.25, (rel_eye_y - 0.34) * 1.1)
                    score += max(0.0, 42.0 - dist_target * 95.0)
                    lead_diff = (margin_r - margin_l) / w
                    score += max(-18.0, min(18.0, lead_diff * 55.0))
                else:
                    # Head-on / symmetric bird (e.g. soaring vulture or front-facing eagle):
                    # Reward horizontal symmetry (body centered) + eye at upper-third/center (0.34..0.46)
                    if subj_ar >= 1.9:
                        # Wide soaring bird: centered horizontally & vertically near 0.45-0.48
                        dist_sym = math.hypot((rel_scx - 0.50) * 1.4, (rel_scy - 0.47) * 1.3)
                        score += max(0.0, 46.0 - dist_sym * 110.0)
                    else:
                        # Upright front-facing portrait (e.g. Steppe Eagle):
                        # Centered horizontally, eye near upper-third (0.32..0.38)
                        dist_eye = math.hypot((rel_eye_x - 0.50) * 1.3, (rel_eye_y - 0.34) * 1.2)
                        score += max(0.0, 44.0 - dist_eye * 105.0)
                    # Penalize horizontal lopsidedness on symmetric shots
                    horiz_asym = abs(margin_l - margin_r) / w
                    score -= horiz_asym * 35.0

                # 3) Vertical balance penalty (avoid huge empty bottom or top when not intentional)
                if rel_scy < 0.28 or rel_scy > 0.72:
                    score -= abs(rel_scy - 0.50) * 40.0

                # 4) Ideal subject fill ratio inside the cropped frame (~42%..70% on constraining axis)
                fill_x = bw / w
                fill_y = bh / h
                fill_max = max(fill_x, fill_y)
                ideal_fill = 0.62 if is_wide_symmetric else 0.50
                score += max(0.0, 18.0 - abs(fill_max - ideal_fill) * 42.0)

                # 5) Clean negative space & border distraction avoidance (image-driven!)
                if clutter_map is not None:
                    ch_m, cw_m = clutter_map.shape[:2]
                    ix1 = max(0, min(cw_m - 1, int(l * cw_m)))
                    ix2 = max(ix1 + 1, min(cw_m, int(r * cw_m)))
                    iy1 = max(0, min(ch_m - 1, int(t * ch_m)))
                    iy2 = max(iy1 + 1, min(ch_m, int(b * ch_m)))
                    crop_clutter = clutter_map[iy1:iy2, ix1:ix2]
                    if crop_clutter.size > 0:
                        # Penalize busy edges / distracting objects cut at crop borders
                        border_ring = np.concatenate(
                            [crop_clutter[0, :], crop_clutter[-1, :], crop_clutter[:, 0], crop_clutter[:, -1]]
                        )
                        score -= float(np.mean(border_ring)) * 12.0
                        score -= float(np.mean(crop_clutter)) * 6.0

            if score > best_score:
                best_score = score
                best_result = (l, t, r, b, rule_name)

    return best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], bird_size_frac


def compute_smart_crop(
    source_path: Optional[Path] = None,
    preview_path: Optional[Path] = None,
    preview_bgr: Optional[np.ndarray] = None,
    bbox: Optional[Sequence[float]] = None,
    head_bbox: Optional[Sequence[float]] = None,
    preview_width: int = 0,
    preview_height: int = 0,
    orig_width: int = 0,
    orig_height: int = 0,
    track: str = "",
    min_long_px: int = DEFAULT_MIN_LONG_PX,
    aspect_locked: bool = True,
    target_aspect: Optional[float] = None,
    selected_horizon_idx: Optional[int] = None,
    override_angle: Optional[float] = None,
    existing_horizons: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute a smart crop & straighten suggestion for a single image.

    - Calculates crop size dynamically from the bird's size in the frame.
    - Never crops smaller than `min_long_px` (default 2700px) on the long side
      (clamped to the original image's long side if the image itself is smaller).
    - Locks aspect ratio to the original image's ratio by default.
    - Detects all optional horizon candidates and auto-selects the best one.
    """
    # 1. Load preview if not provided (prefer fast cached preview JPEG if available)
    bgr = preview_bgr
    if bgr is None and _safe_is_file(preview_path):
        assert preview_path is not None
        try:
            raw_bytes = np.frombuffer(preview_path.read_bytes(), dtype=np.uint8)
            bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.debug("Smart crop cached preview load failed for %s: %s", preview_path, exc)
    if bgr is None and _safe_is_file(source_path):
        assert source_path is not None
        try:
            bgr, _ = load_preview_bgr_and_meta(source_path)
        except Exception as exc:
            logger.debug("Smart crop preview load failed for %s: %s", source_path, exc)

    # 2. Resolve original full-resolution dimensions & preview dimensions
    if orig_width <= 0 or orig_height <= 0:
        ow, oh = read_original_dimensions(source_path, fallback_w=preview_width, fallback_h=preview_height)
        orig_width = ow or preview_width
        orig_height = oh or preview_height

    if (orig_width <= 0 or orig_height <= 0) and bgr is not None:
        orig_height, orig_width = bgr.shape[:2]

    if preview_width <= 0 or preview_height <= 0:
        if bgr is not None:
            preview_height, preview_width = bgr.shape[:2]
        else:
            preview_width, preview_height = orig_width, orig_height

    orig_width = max(1, int(orig_width or 1))
    orig_height = max(1, int(orig_height or 1))
    preview_width = max(1, int(preview_width or orig_width))
    preview_height = max(1, int(preview_height or orig_height))

    img_aspect = orig_width / float(orig_height)
    desired_aspect = float(target_aspect) if (target_aspect and target_aspect > 0) else img_aspect

    # 3. Normalize subject bbox and head_bbox to [0, 1]
    bbox_norm: Optional[List[float]] = None
    if bbox and len(bbox) == 4:
        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        # Support both pixel coords (in preview space) and already-normalized coords
        if max(bx1, by1, bx2, by2) > 1.5:
            bx1 /= float(preview_width)
            bx2 /= float(preview_width)
            by1 /= float(preview_height)
            by2 /= float(preview_height)
        bx1, bx2 = sorted((max(0.0, min(1.0, bx1)), max(0.0, min(1.0, bx2))))
        by1, by2 = sorted((max(0.0, min(1.0, by1)), max(0.0, min(1.0, by2))))
        if (bx2 - bx1) > 0.01 and (by2 - by1) > 0.01:
            bbox_norm = [bx1, by1, bx2, by2]

    head_norm: Optional[List[float]] = None
    if head_bbox and len(head_bbox) == 4:
        hx1, hy1, hx2, hy2 = [float(v) for v in head_bbox]
        if max(hx1, hy1, hx2, hy2) > 1.5:
            hx1 /= float(preview_width)
            hx2 /= float(preview_width)
            hy1 /= float(preview_height)
            hy2 /= float(preview_height)
        hx1, hx2 = sorted((max(0.0, min(1.0, hx1)), max(0.0, min(1.0, hx2))))
        hy1, hy2 = sorted((max(0.0, min(1.0, hy1)), max(0.0, min(1.0, hy2))))
        if (hx2 - hx1) > 0.005 and (hy2 - hy1) > 0.005:
            head_norm = [hx1, hy1, hx2, hy2]

    # 4. Detect all optional horizons (or reuse existing candidates when re-selecting)
    if existing_horizons is not None:
        horizons = existing_horizons
        auto_sel_idx = selected_horizon_idx
    else:
        horizons, auto_sel_idx = detect_horizons(bgr, subject_bbox_norm=bbox_norm) if bgr is not None else ([], None)

    if override_angle is not None:
        angle_deg = max(-45.0, min(45.0, float(override_angle)))
        chosen_h_idx = selected_horizon_idx
    elif selected_horizon_idx is not None and 0 <= selected_horizon_idx < len(horizons):
        chosen_h_idx = int(selected_horizon_idx)
        angle_deg = float(horizons[chosen_h_idx]["angle"])
    elif auto_sel_idx is not None and 0 <= auto_sel_idx < len(horizons):
        chosen_h_idx = int(auto_sel_idx)
        angle_deg = float(horizons[chosen_h_idx]["angle"])
    else:
        chosen_h_idx = None
        angle_deg = 0.0

    # 5. Evaluate real photography compositions & pick the highest-scoring crop
    bird_size_frac = 0.0
    composition_rule = "Full Frame"
    locked_r = desired_aspect if aspect_locked else None
    if bbox_norm and track.lower() != "landscape":
        left, top, right, bottom, composition_rule, bird_size_frac = _choose_best_wildlife_crop(
            bbox_norm=bbox_norm,
            head_norm=head_norm,
            img_aspect=img_aspect,
            desired_aspect=desired_aspect,
            orig_width=orig_width,
            orig_height=orig_height,
            min_long_px=int(min_long_px),
            angle_deg=angle_deg,
            aspect_locked=aspect_locked,
            bgr=bgr,
        )
    else:
        # Landscape / no bird bbox: start with full frame, inset only for straighten angle
        left, top, right, bottom = clamp_crop_to_image(
            0.0, 0.0, 1.0, 1.0, angle_deg=angle_deg, aspect_w_over_h=img_aspect, locked_ratio=locked_r
        )
        composition_rule = "Horizon Straightened" if abs(angle_deg) >= 0.25 else "Full Frame"

    crop_w_px = int(round((right - left) * orig_width))
    crop_h_px = int(round((bottom - top) * orig_height))

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "angle": round(angle_deg, 2),
        "aspect_locked": bool(aspect_locked),
        "aspect_ratio": round(desired_aspect, 4),
        "orig_width": orig_width,
        "orig_height": orig_height,
        "crop_width_px": crop_w_px,
        "crop_height_px": crop_h_px,
        "min_long_px": int(min_long_px),
        "bird_size_frac": round(bird_size_frac, 4),
        "composition_rule": composition_rule,
        "selected_horizon_idx": chosen_h_idx,
        "horizons": horizons,
        "source": "auto",
        "applied": False,
    }
