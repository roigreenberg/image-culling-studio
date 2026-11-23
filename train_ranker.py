"""Train a personalized LightGBM ranker from exported my_picks.json feedback."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_ranker")


def train_from_picks(
    picks_path: Path,
    results_json: Path,
    embeddings_npz: Path,
    model_out: Path,
    history_jsonl: Path,
) -> None:
    if not picks_path.is_file():
        raise FileNotFoundError(f"Picks file not found: {picks_path}")

    picks_data = json.loads(picks_path.read_text(encoding="utf-8"))
    picks_list: List[Dict] = picks_data.get("picks", [])
    if not picks_list:
        raise ValueError("No picks found inside my_picks.json.")

    records = json.loads(results_json.read_text(encoding="utf-8"))
    rec_by_id = {r["image_id"]: r for r in records}

    npz = np.load(embeddings_npz)
    npz_ids = [str(x) for x in npz["image_ids"]]
    idx_by_id = {img_id: i for i, img_id in enumerate(npz_ids)}
    siglip_embs = npz["siglip_embs"].astype(np.float32)
    bioclip_embs = npz["bioclip_embs"].astype(np.float32)

    X_rows = []
    y_rows = []
    weights = []

    # Append new feedback to persistent history so learning accumulates across shoots
    history_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(history_jsonl, "a", encoding="utf-8") as hfh:
        for p in picks_list:
            img_id = p.get("image_id")
            dec = p.get("user_decision")
            if img_id not in idx_by_id or img_id not in rec_by_id:
                continue
            if dec not in ("layer2_edit", "layer1_keep", "reject"):
                continue

            i = idx_by_id[img_id]
            r = rec_by_id[img_id]
            tech = np.array(
                [
                    float(r.get("sharpness_score", 0.0)),
                    float(r.get("eye_contact_score", 0.0)),
                    float(r.get("action_score", 0.0)),
                    float(r.get("scenic_score", 0.0)),
                    float(r.get("tonal_balance", 0.0)) * 100.0,
                    1.0 if r.get("track") == "Wildlife" else 0.0,
                ],
                dtype=np.float32,
            )
            feat = np.concatenate([siglip_embs[i], bioclip_embs[i], tech])
            label = 1 if dec in ("layer2_edit", "layer1_keep") else 0
            weight = 2.0 if dec == "layer2_edit" else 1.0

            X_rows.append(feat)
            y_rows.append(label)
            weights.append(weight)
            hfh.write(json.dumps({"image_id": img_id, "label": label, "weight": weight, "feat": feat.tolist()}) + "\n")

    # Load full accumulated history if available
    all_X, all_y, all_w = [], [], []
    if history_jsonl.is_file():
        with open(history_jsonl, "r", encoding="utf-8") as hfh:
            dedup = {}
            for line in hfh:
                if not line.strip():
                    continue
                row = json.loads(line)
                dedup[row["image_id"]] = row
        for row in dedup.values():
            all_X.append(np.array(row["feat"], dtype=np.float32))
            all_y.append(int(row["label"]))
            all_w.append(float(row["weight"]))
    else:
        all_X, all_y, all_w = X_rows, y_rows, weights

    X = np.vstack(all_X)
    y = np.array(all_y, dtype=np.int32)
    w = np.array(all_w, dtype=np.float32)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    logger.info("Training LightGBM personal ranker on %d samples (%d Keep/Edit, %d Reject).", len(y), n_pos, n_neg)
    if n_pos < 3 or n_neg < 3:
        raise ValueError("Need at least 3 Keep/Edit and 3 Reject picks in my_picks.json to train.")

    clf = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.85,
        colsample_bytree=0.5,
        min_child_samples=max(2, min(10, len(y) // 10)),
        random_state=42,
        verbose=-1,
    )

    if min(n_pos, n_neg) >= 5:
        cv = StratifiedKFold(n_splits=min(5, min(n_pos, n_neg)), shuffle=True, random_state=42)
        aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        logger.info("Cross-validated ROC-AUC: %.3f (+/- %.3f)", float(aucs.mean()), float(aucs.std()))

    clf.fit(X, y, sample_weight=w)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "n_samples": len(y), "n_pos": n_pos, "n_neg": n_neg}, model_out)
    logger.info("Saved personal ranker to %s. Future runs of run_cull.py will apply it automatically!", model_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train personalized LightGBM ranker from my_picks.json")
    parser.add_argument("--picks", type=str, required=True, help="Path to exported my_picks.json")
    parser.add_argument("--results", type=str, default="./cull_output/results.json")
    parser.add_argument("--embeddings", type=str, default="./cull_output/embeddings.npz")
    parser.add_argument("--out", type=str, default="./models/personal_ranker.pkl")
    parser.add_argument("--history", type=str, default="./models/feedback_history.jsonl")
    args = parser.parse_args()

    train_from_picks(
        Path(args.picks),
        Path(args.results),
        Path(args.embeddings),
        Path(args.out),
        Path(args.history),
    )


if __name__ == "__main__":
    main()
