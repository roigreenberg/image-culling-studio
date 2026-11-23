"""Stage 1 & Layer 1 Selection: Burst Grouping and Subject-Sharpness Culling (~15-20%)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def assign_bursts(records: List[Dict[str, Any]], cfg: PipelineConfig) -> List[Dict[str, Any]]:
    """
    Group consecutive shots within each subfolder into bursts if:
      - time delta <= cfg.burst_gap_seconds (2.0s)
      - HSV framing histogram cosine similarity >= cfg.burst_hist_min_sim
    Adds 'burst_id', 'burst_size', and 'burst_index' to every record in-place.
    """
    if not records:
        return records

    # Sort deterministically by (subfolder, timestamp, filename)
    records.sort(key=lambda r: (r["subfolder"], float(r["timestamp"]), r["filename"]))

    burst_counter = 0
    prev_rec = None
    burst_members: Dict[str, List[Dict[str, Any]]] = {}

    for rec in records:
        new_burst = True
        if prev_rec is not None and rec["subfolder"] == prev_rec["subfolder"]:
            dt_sec = float(rec["timestamp"]) - float(prev_rec["timestamp"])
            if 0.0 <= dt_sec <= cfg.burst_gap_seconds:
                h1 = np.asarray(prev_rec.get("_framing_hist", []), dtype=np.float32)
                h2 = np.asarray(rec.get("_framing_hist", []), dtype=np.float32)
                if h1.size > 0 and h1.size == h2.size:
                    sim = float(np.dot(h1, h2))
                else:
                    sim = 1.0
                if sim >= cfg.burst_hist_min_sim:
                    new_burst = False

        if new_burst:
            burst_counter += 1
        bid = f"B{burst_counter:04d}"
        rec["burst_id"] = bid
        burst_members.setdefault(bid, []).append(rec)
        prev_rec = rec

    for bid, members in burst_members.items():
        bsize = len(members)
        for idx, rec in enumerate(members, start=1):
            rec["burst_size"] = bsize
            rec["burst_index"] = idx

    num_bursts = sum(1 for m in burst_members.values() if len(m) > 1)
    num_standalones = sum(1 for m in burst_members.values() if len(m) == 1)
    logger.info(
        "Burst grouping complete: %d multi-frame bursts, %d standalone shots across %d total images.",
        num_bursts,
        num_standalones,
        len(records),
    )
    return records


def _robust_percentile_score(values: np.ndarray) -> np.ndarray:
    """Map raw scores to an intuitive 0..100 scale using robust 2nd..98th percentile scaling."""
    if values.size == 0:
        return values
    if values.size == 1:
        return np.array([75.0], dtype=np.float32)
    p2, p98 = np.percentile(values, [2, 98])
    if p98 - p2 < 1e-6:
        return np.full_like(values, 50.0, dtype=np.float32)
    scaled = (values - p2) / (p98 - p2)
    return np.clip(scaled * 100.0, 0.0, 100.0).astype(np.float32)


def _select_distinct_top_frames(
    members: List[Dict[str, Any]], max_picks: int, min_time_sep: float = 0.28
) -> List[Dict[str, Any]]:
    """
    Pick up to `max_picks` sharpest frames from a burst, preferring temporal/pose diversity
    for secondary picks (e.g., wing spread or head turn during a 20fps R5 burst).
    """
    ranked = sorted(members, key=lambda r: r["layer1_priority"], reverse=True)
    chosen: List[Dict[str, Any]] = []
    for cand in ranked:
        if len(chosen) >= max_picks:
            break
        if not chosen:
            chosen.append(cand)
            continue
        # Check temporal separation OR bbox aspect/position change (action shift!)
        is_distinct = True
        for prev in chosen:
            dt = abs(float(cand["timestamp"]) - float(prev["timestamp"]))
            b1 = cand["bbox"]
            b2 = prev["bbox"]
            w1, h1 = max(1, b1[2] - b1[0]), max(1, b1[3] - b1[1])
            w2, h2 = max(1, b2[2] - b2[0]), max(1, b2[3] - b2[1])
            aspect_diff = abs((w1 / h1) - (w2 / h2))
            if dt < min_time_sep and aspect_diff < 0.25:
                is_distinct = False
                break
        if is_distinct:
            chosen.append(cand)

    # If temporal diversity skipped too many in a super-short burst, top up by sharpness
    if len(chosen) < max_picks:
        chosen_ids = {id(c) for c in chosen}
        for cand in ranked:
            if len(chosen) >= max_picks:
                break
            if id(cand) not in chosen_ids:
                chosen.append(cand)
                chosen_ids.add(id(cand))
    return chosen


def select_layer1_keepers(records: List[Dict[str, Any]], cfg: PipelineConfig) -> List[Dict[str, Any]]:
    """
    Compute normalized sharpness_score (0..100) and select ~15-20% (target 17.5%)
    for Layer 1 ("Keep") using burst-aware ranking + high-scoring standalone shots.
    """
    if not records:
        return records

    # Normalize sharpness_score per subfolder (when a subfolder has >= 50 images) so
    # different shoots / lighting conditions (e.g., midday sun vs backlit dawn) compete fairly.
    by_subfolder: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_subfolder.setdefault(rec["subfolder"], []).append(rec)

    use_per_subfolder = len(by_subfolder) > 1 and all(len(v) >= 50 for v in by_subfolder.values())

    if use_per_subfolder:
        for sf, sf_recs in by_subfolder.items():
            raw_vals = np.array([float(r["sharpness_raw"]) for r in sf_recs], dtype=np.float32)
            norm_scores = _robust_percentile_score(raw_vals)
            for rec, ns in zip(sf_recs, norm_scores):
                rec["sharpness_score"] = round(float(ns), 2)
    else:
        raw_vals = np.array([float(r["sharpness_raw"]) for r in records], dtype=np.float32)
        norm_scores = _robust_percentile_score(raw_vals)
        for rec, ns in zip(records, norm_scores):
            rec["sharpness_score"] = round(float(ns), 2)

    for rec in records:
        ns = float(rec["sharpness_score"])
        det_bonus = 10.0 + 8.0 * float(rec["detection_conf"]) if rec["detection_source"] == "yolo" else -8.0
        area = float(rec.get("subject_area_frac", 0.05))
        size_adj = -10.0 if (rec["detection_source"] == "yolo" and area < 0.0025) else 0.0
        rec["layer1_priority"] = ns + det_bonus + size_adj
        rec["layer"] = "rejected"
        rec["burst_rank"] = 0

    total = len(records)

    def _select_within_group(group_records: List[Dict[str, Any]], target_k: int) -> None:
        bursts: Dict[str, List[Dict[str, Any]]] = {}
        for rec in group_records:
            bursts.setdefault(rec["burst_id"], []).append(rec)

        burst_primary_pool: List[Dict[str, Any]] = []
        secondary_pool: List[Dict[str, Any]] = []

        for bid, members in bursts.items():
            members_ranked = sorted(members, key=lambda r: r["layer1_priority"], reverse=True)
            for rank_i, m in enumerate(members_ranked, start=1):
                m["burst_rank"] = rank_i

            bsize = len(members)
            if bsize == 1:
                m = members_ranked[0]
                m["_l1_selection_score"] = m["layer1_priority"] + 2.0
                secondary_pool.append(m)
            else:
                max_burst_picks = 1 if bsize < 6 else (2 if bsize < 16 else 3)
                picks = _select_distinct_top_frames(members, max_burst_picks)
                picks[0]["_l1_selection_score"] = picks[0]["layer1_priority"] + 12.0
                burst_primary_pool.append(picks[0])
                for extra_idx, extra in enumerate(picks[1:], start=2):
                    extra["_l1_selection_score"] = extra["layer1_priority"] - (4.0 * (extra_idx - 1))
                    secondary_pool.append(extra)

                picked_ids = {id(p) for p in picks}
                for m in members_ranked:
                    if id(m) not in picked_ids:
                        m["_l1_selection_score"] = m["layer1_priority"] - 25.0
                        secondary_pool.append(m)

        all_candidates = burst_primary_pool + secondary_pool
        all_candidates.sort(key=lambda r: r["_l1_selection_score"], reverse=True)
        for rec in all_candidates[:target_k]:
            rec["layer"] = "layer1_keep"
        for rec in group_records:
            rec.pop("_l1_selection_score", None)

    if use_per_subfolder:
        for sf, sf_recs in sorted(by_subfolder.items()):
            sf_k = max(1, int(round(len(sf_recs) * cfg.layer1_target_ratio)))
            _select_within_group(sf_recs, sf_k)
            logger.info("  Subfolder %s Layer 1 budget: %d / %d (%.1f%%)", sf, sf_k, len(sf_recs), 100.0 * sf_k / len(sf_recs))
    else:
        target_k = max(1, int(round(total * cfg.layer1_target_ratio)))
        if total <= 25:
            target_k = max(3, min(total, int(round(total * 0.30))))
        _select_within_group(records, target_k)

    l1_count = sum(1 for r in records if r["layer"] == "layer1_keep")
    logger.info(
        "Layer 1 selection complete: kept %d / %d images (%.1f%%).",
        l1_count,
        total,
        100.0 * l1_count / max(1, total),
    )
    return records
