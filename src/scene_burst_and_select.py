"""Step 2A.4, Step 2C & Step 2D: Scenario/Zoom Grouping, Two-Track Routing & Quality-Driven Selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _robust_percentile_0_100(values: np.ndarray, low_p: float = 2.0, high_p: float = 98.0) -> np.ndarray:
    if values.size == 0:
        return values
    if values.size == 1:
        return np.array([75.0], dtype=np.float32)
    lo, hi = np.percentile(values, [low_p, high_p])
    if hi - lo < 1e-6:
        return np.full_like(values, 50.0, dtype=np.float32)
    return np.clip(((values - lo) / (hi - lo)) * 100.0, 0.0, 100.0).astype(np.float32)


def assign_scenes_and_bursts(
    records: List[Dict[str, Any]],
    siglip_embs: np.ndarray,
    cfg: PipelineConfig,
) -> List[Dict[str, Any]]:
    """
    Group consecutive shots within each subfolder into scenarios/bursts using:
      - EXIF focal length: >= 10% zoom change ALWAYS splits into a new scenario
      - EXIF capture time gap & SigLIP 2 cosine similarity
    """
    if not records:
        return records

    for idx, r in enumerate(records):
        r["_emb_idx"] = idx

    ordered = sorted(records, key=lambda r: (r["subfolder"], float(r["timestamp"]), r["filename"]))

    scene_counter = 0
    prev = None
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for rec in ordered:
        new_scene = True
        if prev is not None and rec["subfolder"] == prev["subfolder"]:
            # 1) Zoom split check (>= 10% focal length change -> different scenario)
            f1 = prev.get("focal_length")
            f2 = rec.get("focal_length")
            zoom_changed = False
            if f1 and f2 and float(f1) > 0 and float(f2) > 0:
                rel_diff = abs(float(f2) - float(f1)) / max(float(f1), float(f2))
                if rel_diff >= cfg.zoom_split_ratio:
                    zoom_changed = True

            if not zoom_changed:
                dt_sec = float(rec["timestamp"]) - float(prev["timestamp"])
                if 0.0 <= dt_sec <= cfg.scene_gap_seconds:
                    e1 = siglip_embs[prev["_emb_idx"]].astype(np.float32)
                    e2 = siglip_embs[rec["_emb_idx"]].astype(np.float32)
                    sem_sim = float(np.dot(e1, e2))

                    if dt_sec <= cfg.burst_gap_seconds and sem_sim >= 0.72:
                        new_scene = False
                    elif sem_sim >= cfg.scene_sim_threshold:
                        new_scene = False

        if new_scene:
            scene_counter += 1
        sid = f"S{scene_counter:04d}"
        rec["burst_id"] = sid
        groups.setdefault(sid, []).append(rec)
        prev = rec

    for sid, members in groups.items():
        bsize = len(members)
        for i, m in enumerate(members, start=1):
            m["burst_size"] = bsize
            m["burst_index"] = i

    multi = sum(1 for m in groups.values() if len(m) > 1)
    logger.info(
        "Scenario/Burst grouping (10%% zoom split): %d multi-frame scenarios, %d singletons (%d total).",
        multi,
        len(groups) - multi,
        len(ordered),
    )
    return ordered


def _apply_personal_ranker_if_available(
    records: List[Dict[str, Any]],
    bioclip_embs: Optional[np.ndarray],
    siglip_embs: Optional[np.ndarray],
    cfg: PipelineConfig,
) -> None:
    """If ./models/personal_ranker.pkl exists, blend learned user preference into overall_score."""
    ranker_path: Path = cfg.personal_ranker_path
    if not ranker_path.is_file() or bioclip_embs is None or siglip_embs is None:
        return
    try:
        import joblib

        bundle = joblib.load(ranker_path)
        model = bundle["model"]
        # Build feature matrix matching train_ranker.py
        tech = np.array(
            [
                [
                    float(r.get("sharpness_score", 0.0)),
                    float(r.get("eye_contact_score", 0.0)),
                    float(r.get("action_score", 0.0)),
                    float(r.get("scenic_score", 0.0)),
                    float(r.get("tonal_balance", 0.0)) * 100.0,
                    1.0 if r.get("track") == "Wildlife" else 0.0,
                ]
                for r in records
            ],
            dtype=np.float32,
        )
        emb_idx = [int(r["_emb_idx"]) for r in records]
        X = np.hstack([siglip_embs[emb_idx].astype(np.float32), bioclip_embs[emb_idx].astype(np.float32), tech])
        probs = model.predict_proba(X)[:, 1] * 100.0
        w = float(cfg.personal_blend_weight)
        for r, p in zip(records, probs):
            if r["track"] != "Other":
                r["personal_score"] = round(float(p), 2)
                r["overall_score"] = round(float(np.clip((1.0 - w) * r["overall_score"] + w * p, 0.0, 100.0)), 2)
        logger.info("Applied personal ranker (%s) with %.0f%% blend weight.", ranker_path, w * 100)
    except Exception as exc:
        logger.warning("Could not apply personal ranker %s: %s", ranker_path, exc)


def route_and_score_all(
    records: List[Dict[str, Any]],
    cfg: PipelineConfig,
    bioclip_embs: Optional[np.ndarray] = None,
    siglip_embs: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Step 2C: Two-Track Routing across 100% of scored images.
    """
    if not records:
        return records

    # --- 1) Track A: Wildlife / Birds ---
    w_indices = [i for i, r in enumerate(records) if r.get("has_wildlife")]
    if w_indices:
        head_raw = np.array([float(records[i]["head_sharp_raw"]) for i in w_indices], dtype=np.float32)
        kest_q = np.array([float(records[i]["kestrel_quality"]) * 100.0 for i in w_indices], dtype=np.float32)

        sf_groups: Dict[str, List[int]] = {}
        for pos, rec_idx in enumerate(w_indices):
            sf_groups.setdefault(records[rec_idx]["subfolder"], []).append(pos)

        head_norm = np.zeros(len(w_indices), dtype=np.float32)
        if len(sf_groups) > 1 and all(len(v) >= 25 for v in sf_groups.values()):
            for sf, pos_list in sf_groups.items():
                head_norm[pos_list] = _robust_percentile_0_100(head_raw[pos_list])
        else:
            head_norm = _robust_percentile_0_100(head_raw)

        w_sharp = 0.55 * head_norm + 0.45 * kest_q

        eye_raw = np.array([float(records[i]["siglip_eye_raw"]) for i in w_indices], dtype=np.float32)
        act_raw = np.array([float(records[i]["siglip_action_raw"]) for i in w_indices], dtype=np.float32)
        w_eye = _robust_percentile_0_100(eye_raw)
        w_act = _robust_percentile_0_100(act_raw)

        for pos, rec_idx in enumerate(w_indices):
            rec = records[rec_idx]
            rec["track"] = "Wildlife"
            rec["sharpness_score"] = round(float(w_sharp[pos]), 2)
            rec["eye_contact_score"] = round(float(w_eye[pos]), 2)
            rec["action_score"] = round(float(w_act[pos]), 2)
            rec["scenic_score"] = round(float(rec.get("siglip_light_prob", 0.0) * 100.0), 2)

            base_score = (
                cfg.wildlife_w_sharp * float(w_sharp[pos])
                + cfg.wildlife_w_eye * float(w_eye[pos])
                + cfg.wildlife_w_action * float(w_act[pos])
            )

            is_hybrid = bool(
                float(rec.get("siglip_light_prob", 0.0)) >= 0.05
                and float(rec.get("siglip_scenic_raw", -1.0)) > 0.005
            )
            rec["hybrid_bonus"] = is_hybrid
            if is_hybrid:
                base_score *= cfg.hybrid_light_bonus

            rec["overall_score"] = round(float(np.clip(base_score, 0.0, 100.0)), 2)

    # --- 2) Track B & Unclassified: Non-Wildlife frames ---
    nw_indices = [i for i, r in enumerate(records) if not r.get("has_wildlife")]
    if nw_indices:
        g_sharp_raw = np.array([float(records[i]["global_sharp_raw"]) for i in nw_indices], dtype=np.float32)
        tonal = np.array([float(records[i]["tonal_balance"]) * 100.0 for i in nw_indices], dtype=np.float32)
        sc_raw = np.array([float(records[i]["siglip_scenic_raw"]) for i in nw_indices], dtype=np.float32)

        g_sharp_norm = _robust_percentile_0_100(g_sharp_raw)
        sc_norm = _robust_percentile_0_100(sc_raw)

        for pos, rec_idx in enumerate(nw_indices):
            rec = records[rec_idx]
            tech_score = 0.55 * float(g_sharp_norm[pos]) + 0.45 * float(tonal[pos])
            scenic_score = float(sc_norm[pos])

            is_scenic = bool(
                float(rec.get("siglip_scenic_raw", -1.0)) > -0.005
                and float(rec.get("blown_frac", 0.0)) < 0.35
                and float(rec.get("tonal_balance", 0.0)) >= 0.20
            )
            rec["sharpness_score"] = round(float(g_sharp_norm[pos]), 2)
            rec["eye_contact_score"] = 0.0
            rec["action_score"] = 0.0
            rec["scenic_score"] = round(scenic_score, 2)
            rec["hybrid_bonus"] = False

            if is_scenic:
                rec["track"] = "Landscape"
                comb = cfg.landscape_w_scenic * scenic_score + cfg.landscape_w_tech * tech_score
                rec["overall_score"] = round(float(np.clip(comb, 0.0, 100.0)), 2)
            else:
                rec["track"] = "Other"
                rec["overall_score"] = round(float(np.clip(0.25 * tech_score, 0.0, 35.0)), 2)

    _apply_personal_ranker_if_available(records, bioclip_embs, siglip_embs, cfg)

    w_cnt = sum(1 for r in records if r["track"] == "Wildlife")
    l_cnt = sum(1 for r in records if r["track"] == "Landscape")
    o_cnt = sum(1 for r in records if r["track"] == "Other")
    logger.info("Two-track routing: Wildlife=%d | Landscape/Sun=%d | Other=%d", w_cnt, l_cnt, o_cnt)
    return records


def _is_not_bad(rec: Dict[str, Any], cfg: PipelineConfig) -> bool:
    tr = rec.get("track", "Other")
    ov = float(rec.get("overall_score", 0.0))
    sh = float(rec.get("sharpness_score", 0.0))
    if tr == "Wildlife":
        if rec.get("hybrid_bonus") and float(rec.get("scenic_score", 0.0)) >= 50.0:
            return ov >= cfg.keep_min_overall_wildlife
        return ov >= cfg.keep_min_overall_wildlife and sh >= cfg.keep_min_sharp_wildlife
    if tr == "Landscape":
        return ov >= cfg.keep_min_overall_landscape and float(rec.get("tonal_balance", 0.0)) >= 0.22
    return False


def _is_good(rec: Dict[str, Any], cfg: PipelineConfig) -> bool:
    tr = rec.get("track", "Other")
    ov = float(rec.get("overall_score", 0.0))
    sh = float(rec.get("sharpness_score", 0.0))
    if tr == "Wildlife":
        if rec.get("hybrid_bonus") and float(rec.get("scenic_score", 0.0)) >= 55.0:
            return ov >= (cfg.edit_min_overall_wildlife - 4.0)
        return ov >= cfg.edit_min_overall_wildlife and sh >= cfg.edit_min_sharp_wildlife
    if tr == "Landscape":
        return (
            ov >= cfg.edit_min_overall_landscape
            and float(rec.get("scenic_score", 0.0)) >= cfg.edit_min_scenic_landscape
        )
    return False


def _is_long_action_sequence(
    members: List[Dict[str, Any]],
    siglip_embs: Optional[np.ndarray] = None,
) -> bool:
    """
    Detect genuine dynamic action events worth keeping as a full burst — e.g.,
    a bird taking off, landing, vigorously flapping, diving, or undergoing a
    dramatic pose/shape transition.

    Deliberately excludes steady flat glides (e.g., soaring birds with static
    wings) and perched birds holding a static pose across many frames.
    """
    if len(members) < 6:
        return False
    w_members = [m for m in members if m.get("track") == "Wildlife"]
    if len(w_members) < 5:
        return False

    w_sorted = sorted(w_members, key=lambda m: float(m.get("timestamp") or 0.0))
    act_scores = np.array([float(m.get("action_score", 0.0)) for m in w_sorted], dtype=np.float64)
    mean_act = float(np.mean(act_scores))
    max_act = float(np.max(act_scores))
    act_range = float(np.percentile(act_scores, 95) - np.percentile(act_scores, 5))

    # Must have meaningful flight / action energy somewhere in the burst
    if mean_act < 58.0 and not (max_act >= 78.0 and act_range >= 32.0):
        return False

    # 1) Measure silhouette / wing-shape evolution from subject bbox aspect ratios
    ars: List[float] = []
    for m in w_sorted:
        bb = m.get("bbox") or []
        if len(bb) == 4 and (bb[2] > bb[0]) and (bb[3] > bb[1]):
            bw = float(bb[2] - bb[0])
            bh = float(bb[3] - bb[1])
            ars.append(bw / max(1.0, bh))

    has_bbox = len(ars) >= 4
    ar_cv = 0.0
    ar_ratio = 1.0
    if has_bbox:
        ar_arr = np.array(ars, dtype=np.float64)
        ar_cv = float(np.std(ar_arr) / max(0.05, float(np.mean(ar_arr))))
        ar_ratio = float(np.percentile(ar_arr, 90) / max(0.05, float(np.percentile(ar_arr, 10))))

    # 2) Measure frame-to-frame visual / pose evolution from SigLIP embeddings
    has_embs = False
    pose_spread = 0.0
    consec_change = 0.0
    if siglip_embs is not None:
        idxs = [
            int(m["_emb_idx"])
            for m in w_sorted
            if "_emb_idx" in m and 0 <= int(m["_emb_idx"]) < len(siglip_embs)
        ]
        if len(idxs) >= 4:
            embs = siglip_embs[idxs].astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.maximum(norms, 1e-8)
            centroid = embs.mean(axis=0)
            centroid /= max(1e-8, float(np.linalg.norm(centroid)))
            pose_spread = float(1.0 - np.mean(embs @ centroid))
            consec_change = float(1.0 - np.mean(np.sum(embs[:-1] * embs[1:], axis=1)))
            has_embs = True

    # Criterion A: Strong wingbeat / takeoff / landing shape transformation
    if has_bbox and ar_cv >= 0.35 and ar_ratio >= 2.0:
        return True

    # Criterion B: Perched-to-takeoff (or landing) surge in action score + shape shift
    if has_bbox and act_range >= 35.0 and max_act >= 78.0 and ar_cv >= 0.28:
        return True

    # Criterion C: Strong frame-to-frame visual/pose evolution (e.g., dive, strike, bank)
    if has_embs and pose_spread >= 0.020 and consec_change >= 0.015:
        return True

    # Fallback ONLY when neither bbox nor embeddings exist (minimal synthetic inputs)
    if not has_bbox and not has_embs and mean_act >= 80.0:
        return True

    return False


def select_layers_final(
    records: List[Dict[str, Any]],
    cfg: PipelineConfig,
    siglip_embs: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Step 2D: Scenario-driven selection (no hardcoded count/percentage):
      - Single picture: Keep if not bad; Edit if good.
      - Static/Repeated scenario: Keep top 2-3 not-bad frames; Edit best 1-2 good frames.
      - Long action sequence (flying bird): If most (>= 50%) are not bad, keep ALL frames
        (even the few bad ones); Edit up to 1 frame per distinct pose/phase.
    """
    if not records:
        return records

    for rec in records:
        rec["layer"] = "rejected"
        rec["burst_rank"] = 0
        rec["scenario_type"] = "single"

    scenes: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        scenes.setdefault(rec["burst_id"], []).append(rec)

    action_seq_count = 0
    for sid, members in scenes.items():
        ranked = sorted(members, key=lambda r: float(r["overall_score"]), reverse=True)
        for idx, m in enumerate(ranked, start=1):
            m["burst_rank"] = idx

        n = len(members)
        not_bad_members = [m for m in ranked if _is_not_bad(m, cfg)]
        good_members = [m for m in ranked if _is_good(m, cfg)]

        if n == 1:
            m = ranked[0]
            m["scenario_type"] = "single"
            if good_members:
                m["layer"] = "layer2_to_edit"
            elif not_bad_members:
                m["layer"] = "layer1_keep"
            continue

        if _is_long_action_sequence(members, siglip_embs=siglip_embs):
            action_seq_count += 1
            for m in members:
                m["scenario_type"] = "action_sequence"

            # Rule 1: If most (>= 60%) of the sequence is good/not-bad, keep ALL frames in Layer 1
            if len(not_bad_members) >= max(3, int(np.ceil(n * 0.60))):
                for m in members:
                    m["layer"] = "layer1_keep"
            else:
                for m in not_bad_members:
                    m["layer"] = "layer1_keep"

            # Rule 2: Edit up to 1 frame per distinct pose/phase
            chosen_l2: List[Dict[str, Any]] = []
            for cand in good_members:
                if not chosen_l2:
                    chosen_l2.append(cand)
                    continue
                is_new_pose = True
                for prev in chosen_l2:
                    dt = abs(float(cand["timestamp"]) - float(prev["timestamp"]))
                    if siglip_embs is not None and "_emb_idx" in cand and "_emb_idx" in prev:
                        e1 = siglip_embs[prev["_emb_idx"]].astype(np.float32)
                        e2 = siglip_embs[cand["_emb_idx"]].astype(np.float32)
                        sim = float(np.dot(e1, e2))
                        if sim >= cfg.action_pose_sim_threshold and dt < 0.75:
                            is_new_pose = False
                            break
                    elif dt < 0.75:
                        is_new_pose = False
                        break
                if is_new_pose:
                    chosen_l2.append(cand)

            for m in chosen_l2:
                m["layer"] = "layer2_to_edit"
            continue

        # Static / Repeated scenario: keep a few (2-3), edit best 1-2
        for m in members:
            m["scenario_type"] = "static_group"

        keep_cap = 2 if n <= 6 else 3
        for m in not_bad_members[:keep_cap]:
            m["layer"] = "layer1_keep"

        if good_members:
            good_members[0]["layer"] = "layer2_to_edit"
            # Promote 2nd frame to Edit only if it's also good and temporally/visually distinct
            if len(good_members) >= 2:
                best1 = good_members[0]
                for cand in good_members[1:]:
                    dt = abs(float(cand["timestamp"]) - float(best1["timestamp"]))
                    if dt >= 0.45 or float(best1["overall_score"]) - float(cand["overall_score"]) <= 4.0:
                        cand["layer"] = "layer2_to_edit"
                        break

    # Split rejected into two tiers:
    #   - 'rejected'  : decent/sharp frames simply not picked (e.g. burst duplicates, or artistic slow-shutter blur)
    #   - 'to_delete' : genuinely bad (blurry, out-of-focus, no subject / misfire, severely blown)
    from .io_utils import parse_exposure_seconds

    to_del_count = 0
    artistic_protected = 0
    for rec in records:
        if rec["layer"] != "rejected":
            continue
        track = rec.get("track", "")
        sharp = float(rec.get("sharpness_score") or 0.0)
        ov = float(rec.get("overall_score") or 0.0)
        act = float(rec.get("action_score") or 0.0)
        scenic = float(rec.get("scenic_score") or 0.0)
        kq = float(rec.get("kestrel_quality") or 0.0)
        tb = float(rec.get("tonal_balance") or 0.0)
        hyb = bool(rec.get("hybrid_bonus"))
        exp_sec = rec.get("exposure_seconds")
        if exp_sec is None:
            exp_sec = parse_exposure_seconds(rec.get("exposure_time", "")) or 0.0

        # Protect intentional artistic slow-shutter / panning motion blur
        has_artistic_potential = (
            (exp_sec >= 0.008 and (act >= 48.0 or scenic >= 45.0 or hyb or tb >= 0.50))
            or (exp_sec >= 0.015 and ov >= 32.0)
            or (scenic >= 55.0)
        )

        is_bad = False
        if track == "Other":
            if scenic < 42.0:
                is_bad = True
        elif track == "Wildlife":
            if sharp < 35.0 or (kq > 0.0 and kq < 0.35) or ov < 35.0:
                is_bad = True
        elif track == "Landscape":
            if ov < 32.0 and scenic < 38.0:
                is_bad = True

        if is_bad:
            if has_artistic_potential:
                artistic_protected += 1
            else:
                rec["layer"] = "to_delete"
                to_del_count += 1

    # Guarantee every multi-frame burst has at least 1 Keep image unless all frames are bad
    for sid, members in scenes.items():
        if len(members) < 2:
            continue
        if any(m["layer"] in ("layer1_keep", "layer2_to_edit") for m in members):
            continue
        non_bad = [
            m
            for m in members
            if m["layer"] != "to_delete" and m.get("track") in ("Wildlife", "Landscape")
        ]
        if non_bad:
            best = max(
                non_bad,
                key=lambda r: (float(r.get("overall_score") or 0.0), float(r.get("sharpness_score") or 0.0)),
            )
            best["layer"] = "layer1_keep"

    # Assign global overall_rank
    layer_order = {"layer2_to_edit": 0, "layer1_keep": 1, "rejected": 2, "to_delete": 3}
    all_ranked = sorted(
        records,
        key=lambda r: (layer_order.get(r["layer"], 3), -float(r["overall_score"]), -float(r["sharpness_score"])),
    )
    for rk, rec in enumerate(all_ranked, start=1):
        rec["overall_rank"] = rk

    total = len(records)
    l2_c = sum(1 for r in records if r["layer"] == "layer2_to_edit")
    l1_c = sum(1 for r in records if r["layer"] in ("layer2_to_edit", "layer1_keep"))
    rej_c = sum(1 for r in records if r["layer"] == "rejected")
    logger.info(
        "Quality-driven selection (%d action sequences): Keep=%d (%.2f%%) | Edit=%d (%.2f%%) | Rejected=%d | ToDelete=%d (artistic blur protected=%d)",
        action_seq_count,
        l1_c,
        100.0 * l1_c / max(1, total),
        l2_c,
        100.0 * l2_c / max(1, total),
        rej_c,
        to_del_count,
        artistic_protected,
    )
    return records
