"""Deliverables generator: 2048px JPEG exports, CSV/JSON/NPZ, and interactive report.html."""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .io_utils import load_preview_bgr_and_meta, save_jpeg_bgr

logger = logging.getLogger(__name__)


def _safe_label_slug(rec: Dict[str, Any]) -> str:
    if rec.get("track") == "Landscape":
        return "SunriseLandscape"
    sp = str(rec.get("species") or "Wildlife").strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "", sp.title())
    return slug[:28] or "Wildlife"


def _export_single_preview(
    source_path: Path, out_paths: List[Path], max_dim: int, quality: int
) -> bool:
    img_bgr, _ = load_preview_bgr_and_meta(source_path)
    if img_bgr is None:
        return False
    for out_p in out_paths:
        save_jpeg_bgr(img_bgr, out_p, quality=quality, max_dim=max_dim)
    return True


def export_preview_folders(records: List[Dict[str, Any]], cfg: PipelineConfig) -> None:
    """
    Generate 2048px JPEG previews inside:
      - ./cull_output/layer2_to_edit/<subfolder>/<rank>_score<score>_<SpeciesOrLandscape>_<stem>.jpg
      - ./cull_output/layer1_keep/<subfolder>/<rank>_score<score>_<SpeciesOrLandscape>_<stem>.jpg
    """
    l2_dir = cfg.output_dir / "layer2_to_edit"
    l1_dir = cfg.output_dir / "layer1_keep"
    if l2_dir.exists():
        shutil.rmtree(l2_dir)
    if l1_dir.exists():
        shutil.rmtree(l1_dir)
    l2_dir.mkdir(parents=True, exist_ok=True)
    l1_dir.mkdir(parents=True, exist_ok=True)

    l2_recs = sorted([r for r in records if r["layer"] == "layer2_to_edit"], key=lambda r: r["overall_rank"])
    l1_all = sorted(
        [r for r in records if r["layer"] in ("layer2_to_edit", "layer1_keep")],
        key=lambda r: r["overall_rank"],
    )

    l2_sub_rank: Dict[str, int] = {}
    l2_path_map: Dict[str, Path] = {}
    for rec in l2_recs:
        sf = rec["subfolder"]
        l2_sub_rank[sf] = l2_sub_rank.get(sf, 0) + 1
        sc = int(round(float(rec["overall_score"])))
        slug = _safe_label_slug(rec)
        fname = f"{l2_sub_rank[sf]:03d}_score{sc:02d}_{slug}_{rec['stem']}.jpg"
        p = l2_dir / sf / fname
        l2_path_map[rec["image_id"]] = p
        rec["export_layer2_path"] = str(p.relative_to(cfg.output_dir)).replace("\\", "/")

    l1_sub_rank: Dict[str, int] = {}
    tasks = []
    for rec in l1_all:
        sf = rec["subfolder"]
        l1_sub_rank[sf] = l1_sub_rank.get(sf, 0) + 1
        sc = int(round(float(rec["overall_score"])))
        slug = _safe_label_slug(rec)
        fname = f"{l1_sub_rank[sf]:04d}_score{sc:02d}_{slug}_{rec['stem']}.jpg"
        p1 = l1_dir / sf / fname
        rec["export_layer1_path"] = str(p1.relative_to(cfg.output_dir)).replace("\\", "/")
        outs = [p1]
        if rec["image_id"] in l2_path_map:
            outs.append(l2_path_map[rec["image_id"]])
        tasks.append((Path(rec["source_path"]), outs))

    failed = 0
    with ThreadPoolExecutor(max_workers=cfg.io_workers) as pool:
        futs = [
            pool.submit(_export_single_preview, src, outs, cfg.export_max_dim, cfg.export_quality)
            for src, outs in tasks
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Exporting 2048px JPEGs", unit="img"):
            try:
                if not fut.result():
                    failed += 1
            except Exception as exc:
                logger.warning("Export error: %s", exc)
                failed += 1
    if failed:
        logger.warning("Failed to export %d preview JPEGs.", failed)


def write_structured_data_and_embeddings(
    records: List[Dict[str, Any]],
    bioclip_embs: np.ndarray,
    siglip_embs: np.ndarray,
    cfg: PipelineConfig,
) -> None:
    """Save results.json, results.csv, and embeddings.npz (aligned by image_id)."""
    ordered = sorted(records, key=lambda r: int(r.get("overall_rank", 999999)))
    clean_records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in ordered]

    json_path = cfg.output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(clean_records, fh, ensure_ascii=False, indent=2)

    csv_fields = [
        "overall_rank",
        "layer",
        "track",
        "species",
        "species_conf",
        "family",
        "animal_group",
        "hybrid_bonus",
        "subfolder",
        "filename",
        "stem",
        "overall_score",
        "sharpness_score",
        "eye_contact_score",
        "action_score",
        "scenic_score",
        "kestrel_quality",
        "head_sharp_raw",
        "global_sharp_raw",
        "tonal_balance",
        "md_conf",
        "animal_gate_score",
        "burst_id",
        "burst_size",
        "burst_rank",
        "timestamp_str",
        "focal_length",
        "f_number",
        "exposure_time",
        "iso",
        "bbox",
        "head_bbox",
        "source_path",
        "export_layer2_path",
        "export_layer1_path",
        "thumb_rel_path",
        "crop_rel_path",
    ]
    csv_path = cfg.output_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for rec in clean_records:
            row = dict(rec)
            row["bbox"] = json.dumps(rec.get("bbox", []))
            row["head_bbox"] = json.dumps(rec.get("head_bbox", []))
            writer.writerow(row)

    # Save aligned embeddings.npz ordered by overall_rank
    emb_indices = [int(r["_emb_idx"]) for r in ordered]
    npz_path = cfg.output_dir / "embeddings.npz"
    np.savez_compressed(
        npz_path,
        image_ids=np.array([r["image_id"] for r in ordered]),
        filenames=np.array([r["filename"] for r in ordered]),
        tracks=np.array([r["track"] for r in ordered]),
        layers=np.array([r["layer"] for r in ordered]),
        overall_scores=np.array([float(r["overall_score"]) for r in ordered], dtype=np.float32),
        bioclip_embs=bioclip_embs[emb_indices].astype(np.float16),
        siglip_embs=siglip_embs[emb_indices].astype(np.float16),
    )
    logger.info("Saved %s, %s, and %s", csv_path, json_path, npz_path)


def generate_html_report(
    records: List[Dict[str, Any]],
    cfg: PipelineConfig,
    summary_stats: Dict[str, Any],
) -> Path:
    """
    Generate interactive review gallery ./cull_output/report.html with:
      - Tabs: Layer 2 (To Edit), Layer 1 (Keep), All Scenes/Bursts, Rejected
      - Filters: Track (All / Wildlife / Landscape), Subfolder, Detected Species, Search
      - Sort: Overall Rank, Date (Oldest/Newest), Sharpness, Eye/Scenic, Scene/Burst
      - Infinite Scroll (load on reaching end)
      - Copy Lightroom Filter String buttons (Layer 2, Layer 1, Wildlife Only, Landscapes Only, My Picks)
      - Layer 2 (Edit) / Layer 1 (Keep) / Reject toggles + Export my_picks.json
    """
    ordered = sorted(records, key=lambda r: int(r.get("overall_rank", 999999)))

    js_items = []
    for r in ordered:
        js_items.append(
            {
                "id": r["image_id"],
                "fn": r["filename"],
                "st": r["stem"],
                "sf": r["subfolder"],
                "ly": r["layer"],
                "tr": r["track"],
                "sp_name": r.get("species", "None"),
                "sp_conf": r.get("species_conf", 0.0),
                "hyb": bool(r.get("hybrid_bonus", False)),
                "rk": r["overall_rank"],
                "os": r["overall_score"],
                "sh": r["sharpness_score"],
                "ey": r["eye_contact_score"],
                "ac": r["action_score"],
                "sc": r.get("scenic_score", 0.0),
                "bi": r["burst_id"],
                "bs": r["burst_size"],
                "br": r["burst_rank"],
                "bx": r.get("burst_index", 1),
                "ts": r["timestamp_str"],
                "tm": round(float(r.get("timestamp") or 0.0), 3),
                "fl": r.get("focal_length"),
                "ap": r.get("f_number"),
                "ex": r.get("exposure_time", ""),
                "is": r.get("iso"),
                "th": r["thumb_rel_path"],
                "cr": r["crop_rel_path"],
                "sp": r["source_path"],
            }
        )

    subfolders = sorted({r["subfolder"] for r in ordered})
    species_counts: Dict[str, int] = {}
    for r in ordered:
        if r.get("track") == "Wildlife":
            spn = str(r.get("species") or "Wildlife")
            species_counts[spn] = species_counts.get(spn, 0) + 1
    top_species = [k for k, _ in sorted(species_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    data_json = json.dumps(js_items, ensure_ascii=False, separators=(",", ":"))
    meta_json = json.dumps(
        {"subfolders": subfolders, "species": top_species, "stats": summary_stats},
        ensure_ascii=False,
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Project Kestrel + SigLIP 2 — Wildlife &amp; Landscape Culling Gallery</title>
<style>
  :root {{
    --bg: #0f1115;
    --surface: #171a21;
    --surface2: #1f242e;
    --border: #2b313f;
    --text: #e6ebf5;
    --muted: #929cb0;
    --gold: #f5b942;
    --green: #34d399;
    --blue: #60a5fa;
    --orange: #fb923c;
    --red: #f87171;
    --accent: #3b82f6;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.4;
  }}
  header {{
    position: sticky;
    top: 0;
    z-index: 50;
    background: rgba(15, 17, 21, 0.96);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
  }}
  .header-top {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
  }}
  .title-block h1 {{ margin: 0; font-size: 1.15rem; font-weight: 700; }}
  .title-block .subtitle {{ font-size: 0.8rem; color: var(--muted); margin-top: 2px; }}
  .actions-bar {{ display: flex; flex-wrap: wrap; gap: 7px; align-items: center; }}
  button, select, input {{
    font-family: inherit;
    font-size: 0.8rem;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--text);
    padding: 6px 10px;
    cursor: pointer;
    transition: all 0.14s ease;
  }}
  button:hover {{ background: #293040; border-color: #3d475c; }}
  button.primary {{ background: #2563eb; border-color: #3b82f6; color: #fff; font-weight: 600; }}
  button.gold-btn {{ background: rgba(245,185,66,0.16); border-color: rgba(245,185,66,0.55); color: #fcd34d; font-weight: 600; }}
  button.green-btn {{ background: rgba(52,211,153,0.14); border-color: rgba(52,211,153,0.45); color: #6ee7b7; font-weight: 600; }}
  .controls-row {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; }}
  .tabs {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .tab-btn {{ padding: 7px 12px; font-weight: 600; background: var(--surface); color: var(--muted); }}
  .tab-btn.active {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  .tab-btn .cnt {{
    display: inline-block; margin-left: 6px; padding: 1px 6px;
    border-radius: 999px; font-size: 0.72rem; background: rgba(255,255,255,0.14);
  }}
  .filters-right {{ display: flex; flex-wrap: wrap; gap: 7px; align-items: center; }}
  input[type="search"] {{ cursor: text; width: 175px; }}
  .status-strip {{
    padding: 8px 20px; font-size: 0.8rem; color: var(--muted);
    display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
    gap: 8px; border-bottom: 1px solid #1b1f28;
  }}
  main {{ padding: 16px 20px 60px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 14px; }}
  @media (max-width: 560px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    overflow: hidden; display: flex; flex-direction: column;
  }}
  .card.user-l2 {{ border-color: var(--gold); box-shadow: 0 0 0 1px rgba(245,185,66,0.45); }}
  .card.user-l1 {{ border-color: var(--green); box-shadow: 0 0 0 1px rgba(52,211,153,0.35); }}
  .card.user-reject {{ border-color: var(--red); opacity: 0.58; }}
  .card-images {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 2px; background: #08090c; height: 245px; }}
  .img-slot {{ position: relative; overflow: hidden; cursor: zoom-in; background: #0b0d11; display: flex; align-items: center; justify-content: center; }}
  .img-slot img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .img-slot.full-slot img {{ object-fit: contain; background: #0a0c10; }}
  .img-label {{
    position: absolute; bottom: 6px; left: 6px; background: rgba(0,0,0,0.72);
    color: #dce3f0; font-size: 0.68rem; padding: 2px 6px; border-radius: 4px; pointer-events: none;
  }}
  .rank-pill {{
    position: absolute; top: 6px; left: 6px; background: rgba(15,17,21,0.85);
    border: 1px solid rgba(255,255,255,0.16); color: #fff; font-weight: 700;
    font-size: 0.74rem; padding: 2px 7px; border-radius: 5px;
  }}
  .layer-pill {{
    position: absolute; top: 6px; right: 6px; font-size: 0.68rem; font-weight: 700;
    padding: 2px 7px; border-radius: 5px; text-transform: uppercase;
  }}
  .layer-layer2_to_edit {{ background: rgba(245,185,66,0.92); color: #1a1200; }}
  .layer-layer1_keep {{ background: rgba(52,211,153,0.90); color: #042217; }}
  .layer-rejected {{ background: rgba(148,163,184,0.82); color: #0f172a; }}
  .card-body {{ padding: 10px 12px; display: flex; flex-direction: column; gap: 7px; }}
  .card-header-line {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
  .fname {{ font-weight: 700; font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .meta-tags {{ display: flex; flex-wrap: wrap; gap: 5px; font-size: 0.72rem; color: var(--muted); }}
  .tag {{ background: var(--surface2); border: 1px solid var(--border); padding: 1px 6px; border-radius: 4px; }}
  .tag-wildlife {{ background: rgba(59,130,246,0.16); border-color: rgba(59,130,246,0.45); color: #93c5fd; font-weight: 600; }}
  .tag-landscape {{ background: rgba(251,146,60,0.18); border-color: rgba(251,146,60,0.5); color: #fdba74; font-weight: 600; }}
  .tag-hybrid {{ background: rgba(245,185,66,0.2); border-color: rgba(245,185,66,0.5); color: #fcd34d; font-weight: 600; }}
  .scores-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }}
  .score-box {{ background: var(--surface2); border-radius: 6px; padding: 5px 7px; text-align: center; }}
  .score-box .lbl {{ font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }}
  .score-box .val {{ font-size: 0.9rem; font-weight: 700; margin-top: 1px; }}
  .val-high {{ color: var(--green); }}
  .val-mid {{ color: var(--gold); }}
  .val-low {{ color: var(--muted); }}
  .card-footer {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; padding-top: 4px; border-top: 1px solid #232834; }}
  .exif-line {{ font-size: 0.71rem; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .pick-btns {{ display: flex; gap: 5px; flex-shrink: 0; }}
  .pick-btn {{ padding: 4px 8px; font-size: 0.74rem; font-weight: 600; }}
  .pick-btn.active-l2 {{ background: var(--gold); border-color: var(--gold); color: #1a1200; }}
  .pick-btn.active-l1 {{ background: var(--green); border-color: var(--green); color: #032217; }}
  .pick-btn.active-reject {{ background: var(--red); border-color: var(--red); color: #2c0606; }}
  .load-more-bar {{ margin-top: 22px; display: flex; justify-content: center; align-items: center; gap: 10px; }}
  .modal {{
    display: none; position: fixed; inset: 0; z-index: 200; background: rgba(5,7,10,0.94);
    padding: 18px; flex-direction: column; gap: 12px;
  }}
  .modal.open {{ display: flex; }}
  .modal-top {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
  .modal-images {{ flex: 1; min-height: 0; display: grid; grid-template-columns: 1.55fr 1fr; gap: 12px; }}
  .modal-pane {{ background: #0c0e13; border: 1px solid var(--border); border-radius: 8px; display: flex; flex-direction: column; overflow: hidden; min-height: 0; }}
  .modal-pane-hdr {{ padding: 6px 10px; font-size: 0.78rem; color: var(--muted); border-bottom: 1px solid var(--border); }}
  .modal-pane img {{ flex: 1; min-height: 0; width: 100%; height: 100%; object-fit: contain; }}
  .toast {{
    position: fixed; bottom: 22px; right: 22px; z-index: 300; background: #1e293b;
    border: 1px solid #475569; color: #f8fafc; padding: 10px 16px; border-radius: 8px;
    font-size: 0.85rem; box-shadow: 0 10px 25px rgba(0,0,0,0.45); opacity: 0;
    transform: translateY(10px); transition: all 0.2s ease; pointer-events: none;
  }}
  .toast.show {{ opacity: 1; transform: translateY(0); }}
</style>
</head>
<body>
<header>
  <div class="header-top">
    <div class="title-block">
      <h1>Project Kestrel + SigLIP 2 — Wildlife &amp; Landscape Culling Gallery</h1>
      <div class="subtitle" id="summarySubtitle">Loading summary...</div>
    </div>
    <div class="actions-bar">
      <button class="gold-btn" id="btnCopyL2">Copy LR: Layer 2</button>
      <button class="green-btn" id="btnCopyL1">Copy LR: Layer 1</button>
      <button id="btnCopyWildlife">Copy LR: Wildlife</button>
      <button id="btnCopyLandscapes">Copy LR: Landscapes</button>
      <button id="btnCopyPicks">Copy LR: My Picks</button>
      <button class="primary" id="btnExportPicks">Export my_picks.json</button>
    </div>
  </div>
  <div class="controls-row">
    <div class="tabs" id="tabsBar">
      <button class="tab-btn active" data-tab="layer2">Layer 2 (To Edit)<span class="cnt" id="cnt-layer2">0</span></button>
      <button class="tab-btn" data-tab="layer1">Layer 1 (Keep)<span class="cnt" id="cnt-layer1">0</span></button>
      <button class="tab-btn" data-tab="bursts">All Scenes/Bursts<span class="cnt" id="cnt-bursts">0</span></button>
      <button class="tab-btn" data-tab="rejected">Rejected<span class="cnt" id="cnt-rejected">0</span></button>
    </div>
    <div class="filters-right">
      <select id="trackFilter" title="Filter by track">
        <option value="ALL">All Tracks</option>
        <option value="Wildlife">Wildlife / Birds</option>
        <option value="Landscape">Sun / Sunrise / Landscape</option>
      </select>
      <select id="subfolderFilter" title="Filter by subfolder">
        <option value="ALL">All Subfolders</option>
      </select>
      <select id="speciesFilter" title="Filter by BioCLIP 2 species">
        <option value="ALL">All Species</option>
      </select>
      <select id="sortSelect" title="Sort order">
        <option value="rank">Sort: Overall Rank</option>
        <option value="date_asc">Sort: Date (Oldest &rarr; Newest)</option>
        <option value="date_desc">Sort: Date (Newest &rarr; Oldest)</option>
        <option value="sharpness">Sort: Sharpness</option>
        <option value="eye">Sort: Eye Contact</option>
        <option value="action">Sort: Action / Scenic</option>
        <option value="burst">Sort: Scene / Burst</option>
      </select>
      <input type="search" id="searchInput" placeholder="Filter filename / species..." />
    </div>
  </div>
</header>

<div class="status-strip">
  <div id="viewStatus">Showing 0 items</div>
  <div id="picksStatus">My picks: 0 Edit, 0 Keep, 0 Reject</div>
</div>

<main>
  <div class="grid" id="galleryGrid"></div>
  <div class="load-more-bar" id="loadMoreBar"></div>
  <div id="scrollSentinel" style="height: 24px;"></div>
</main>

<div class="modal" id="lightboxModal">
  <div class="modal-top">
    <div id="modalTitle" style="font-weight:700;font-size:0.95rem;"></div>
    <div style="display:flex;gap:8px;align-items:center;">
      <button id="modalL2Btn" class="pick-btn">&#9733; Edit (E)</button>
      <button id="modalL1Btn" class="pick-btn">&#10003; Keep (P)</button>
      <button id="modalRejectBtn" class="pick-btn">&#10007; Reject (X)</button>
      <button id="modalSplitBtn" class="pick-btn" title="Make this image #1 of a new burst">&#9986; Split Burst (S)</button>
      <button id="modalPrevBtn">&larr; Prev</button>
      <button id="modalNextBtn">Next &rarr;</button>
      <button id="modalCloseBtn">Close (Esc)</button>
    </div>
  </div>
  <div class="modal-images">
    <div class="modal-pane">
      <div class="modal-pane-hdr">Full Frame Preview</div>
      <img id="modalFullImg" alt="Full frame" />
    </div>
    <div class="modal-pane">
      <div class="modal-pane-hdr" id="modalCropHdr">1:1 Detail Crop</div>
      <img id="modalCropImg" alt="1:1 Crop" />
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
const ITEMS = {data_json};
const META = {meta_json};
const STORAGE_KEY = "kestrel_siglip2_picks_v1";
const SPLIT_STORAGE_KEY = "kestrel_siglip2_splits_v1";
const BATCH_SIZE = 60;

// Preserve original pipeline burst IDs before applying user splits
ITEMS.forEach(it => {{ it.base_bi = it.bi; }});

let state = {{
  tab: "layer2",
  track: "ALL",
  subfolder: "ALL",
  species: "ALL",
  sort: "rank",
  search: "",
  renderedCount: 0,
  picks: loadPicks(),
  splits: loadSplits(),
  filtered: [],
  modalIndex: -1
}};

function loadPicks() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
  catch {{ return {{}}; }}
}}

function loadSplits() {{
  try {{ return JSON.parse(localStorage.getItem(SPLIT_STORAGE_KEY) || "{{}}"); }}
  catch {{ return {{}}; }}
}}

function savePicks() {{
  try {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(state.picks)); }} catch {{}}
  updatePicksStatus();
}}

function saveSplits() {{
  try {{ localStorage.setItem(SPLIT_STORAGE_KEY, JSON.stringify(state.splits)); }} catch {{}}
}}

function recomputeBurstsFromSplits() {{
  // Group chronologically by (subfolder, timestamp, filename) and split whenever state.splits[id] is true
  const chrono = [...ITEMS].sort((a, b) => a.sf.localeCompare(b.sf) || (a.tm - b.tm) || a.fn.localeCompare(b.fn));
  let counter = 0;
  let prev = null;
  const groups = new Map();
  for (const it of chrono) {{
    let isNew = false;
    if (!prev || prev.sf !== it.sf || prev.base_bi !== it.base_bi || state.splits[it.id]) {{
      isNew = true;
    }}
    if (isNew) counter++;
    const newId = `S${{String(counter).padStart(4, "0")}}`;
    it.bi = newId;
    if (!groups.has(newId)) groups.set(newId, []);
    groups.get(newId).push(it);
    prev = it;
  }}
  for (const members of groups.values()) {{
    const bsize = members.length;
    members.forEach((m, idx) => {{
      m.bs = bsize;
      m.bx = idx + 1;
    }});
  }}
}}

function toggleSplitBurst(id) {{
  if (state.splits[id]) {{
    delete state.splits[id];
    showToast("Merged burst split back.");
  }} else {{
    state.splits[id] = true;
    showToast("Split burst here — this image is now #1 of its burst!");
  }}
  saveSplits();
  recomputeBurstsFromSplits();
  applyFilterAndRender();
  if (state.modalIndex >= 0 && document.getElementById("lightboxModal").classList.contains("open")) {{
    const cur = state.filtered.findIndex(x => x.id === id);
    if (cur >= 0) openModal(cur);
  }}
}}

function showToast(msg) {{
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2600);
}}

function init() {{
  const st = META.stats || {{}};
  document.getElementById("summarySubtitle").textContent =
    `Total: ${{st.total_processed || ITEMS.length}} | Wildlife: ${{st.wildlife_count || 0}} | Sun/Landscape: ${{st.landscape_count || 0}} | Layer 2 (To Edit): ${{st.layer2_count || 0}} (${{st.layer2_pct || 0}}%) | Layer 1 (Keep): ${{st.layer1_total_count || 0}} (${{st.layer1_pct || 0}}%) | Runtime: ${{st.runtime_str || "-"}}`;

  const sfSel = document.getElementById("subfolderFilter");
  (META.subfolders || []).forEach(sf => {{
    const opt = document.createElement("option");
    opt.value = sf; opt.textContent = sf;
    sfSel.appendChild(opt);
  }});

  const spSel = document.getElementById("speciesFilter");
  (META.species || []).forEach(sp => {{
    const opt = document.createElement("option");
    opt.value = sp; opt.textContent = sp;
    spSel.appendChild(opt);
  }});

  document.querySelectorAll(".tab-btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.tab = btn.dataset.tab;
      if (state.tab === "bursts") {{
        state.sort = "burst";
        document.getElementById("sortSelect").value = "burst";
      }}
      applyFilterAndRender();
    }});
  }});

  document.getElementById("trackFilter").addEventListener("change", e => {{ state.track = e.target.value; applyFilterAndRender(); }});
  sfSel.addEventListener("change", e => {{ state.subfolder = e.target.value; applyFilterAndRender(); }});
  spSel.addEventListener("change", e => {{ state.species = e.target.value; applyFilterAndRender(); }});
  document.getElementById("sortSelect").addEventListener("change", e => {{ state.sort = e.target.value; applyFilterAndRender(); }});
  document.getElementById("searchInput").addEventListener("input", e => {{ state.search = e.target.value.trim().toLowerCase(); applyFilterAndRender(); }});

  document.getElementById("btnCopyL2").addEventListener("click", () => copyLightroomFilter("layer2"));
  document.getElementById("btnCopyL1").addEventListener("click", () => copyLightroomFilter("layer1"));
  document.getElementById("btnCopyWildlife").addEventListener("click", () => copyLightroomFilter("wildlife"));
  document.getElementById("btnCopyLandscapes").addEventListener("click", () => copyLightroomFilter("landscape"));
  document.getElementById("btnCopyPicks").addEventListener("click", () => copyLightroomFilter("picks"));
  document.getElementById("btnExportPicks").addEventListener("click", exportMyPicks);

  document.getElementById("modalCloseBtn").addEventListener("click", closeModal);
  document.getElementById("modalPrevBtn").addEventListener("click", () => stepModal(-1));
  document.getElementById("modalNextBtn").addEventListener("click", () => stepModal(1));
  document.getElementById("modalL2Btn").addEventListener("click", () => toggleModalPick("layer2_edit"));
  document.getElementById("modalL1Btn").addEventListener("click", () => toggleModalPick("layer1_keep"));
  document.getElementById("modalRejectBtn").addEventListener("click", () => toggleModalPick("reject"));
  document.getElementById("modalSplitBtn").addEventListener("click", () => {{
    if (state.modalIndex >= 0 && state.modalIndex < state.filtered.length) {{
      toggleSplitBurst(state.filtered[state.modalIndex].id);
    }}
  }});

  document.addEventListener("keydown", e => {{
    const modal = document.getElementById("lightboxModal");
    if (!modal.classList.contains("open")) return;
    if (e.key === "Escape") closeModal();
    else if (e.key === "ArrowLeft") stepModal(-1);
    else if (e.key === "ArrowRight") stepModal(1);
    else if (e.key.toLowerCase() === "e") toggleModalPick("layer2_edit");
    else if (e.key.toLowerCase() === "p") toggleModalPick("layer1_keep");
    else if (e.key.toLowerCase() === "x") toggleModalPick("reject");
    else if (e.key.toLowerCase() === "s") toggleSplitBurst(state.filtered[state.modalIndex].id);
  }});

  const sentinel = document.getElementById("scrollSentinel");
  if ("IntersectionObserver" in window && sentinel) {{
    const observer = new IntersectionObserver(entries => {{
      if (entries.some(e => e.isIntersecting) && state.renderedCount < state.filtered.length) {{
        appendNextBatch();
      }}
    }}, {{ rootMargin: "600px" }});
    observer.observe(sentinel);
  }}

  recomputeBurstsFromSplits();
  updatePicksStatus();
  applyFilterAndRender();
}}

function matchesBaseFilters(it) {{
  if (state.subfolder !== "ALL" && it.sf !== state.subfolder) return false;
  if (state.track !== "ALL" && it.tr !== state.track) return false;
  if (state.species !== "ALL" && it.sp_name !== state.species) return false;
  return true;
}}

function updateTabCounts() {{
  let cL2 = 0, cL1 = 0, cAll = 0, cRej = 0;
  for (const it of ITEMS) {{
    if (!matchesBaseFilters(it)) continue;
    cAll++;
    if (it.ly === "layer2_to_edit") {{ cL2++; cL1++; }}
    else if (it.ly === "layer1_keep") {{ cL1++; }}
    else {{ cRej++; }}
  }}
  document.getElementById("cnt-layer2").textContent = cL2;
  document.getElementById("cnt-layer1").textContent = cL1;
  document.getElementById("cnt-bursts").textContent = cAll;
  document.getElementById("cnt-rejected").textContent = cRej;
}}

function updatePicksStatus() {{
  let l2 = 0, l1 = 0, rej = 0;
  for (const v of Object.values(state.picks)) {{
    if (v === "layer2_edit") l2++;
    else if (v === "layer1_keep") l1++;
    else if (v === "reject") rej++;
  }}
  document.getElementById("picksStatus").textContent =
    `My picks: ${{l2}} Edit, ${{l1}} Keep, ${{rej}} Reject (saved in browser)`;
}}

function scoreClass(v) {{
  if (v >= 72) return "val-high";
  if (v >= 45) return "val-mid";
  return "val-low";
}}

function layerLabel(ly) {{
  if (ly === "layer2_to_edit") return "Layer 2 · To Edit";
  if (ly === "layer1_keep") return "Layer 1 · Keep";
  return "Rejected";
}}

function applyFilterAndRender() {{
  updateTabCounts();
  const q = state.search;
  const list = ITEMS.filter(it => {{
    if (!matchesBaseFilters(it)) return false;
    if (state.tab === "layer2" && it.ly !== "layer2_to_edit") return false;
    if (state.tab === "layer1" && it.ly !== "layer2_to_edit" && it.ly !== "layer1_keep") return false;
    if (state.tab === "rejected" && it.ly !== "rejected") return false;
    if (q) {{
      const hay = `${{it.fn}} ${{it.bi}} ${{it.sf}} ${{it.sp_name}} ${{it.tr}}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }}
    return true;
  }});

  if (state.sort === "rank") list.sort((a, b) => a.rk - b.rk);
  else if (state.sort === "date_asc") list.sort((a, b) => (a.tm - b.tm) || a.fn.localeCompare(b.fn));
  else if (state.sort === "date_desc") list.sort((a, b) => (b.tm - a.tm) || b.fn.localeCompare(a.fn));
  else if (state.sort === "sharpness") list.sort((a, b) => b.sh - a.sh || a.rk - b.rk);
  else if (state.sort === "eye") list.sort((a, b) => b.ey - a.ey || a.rk - b.rk);
  else if (state.sort === "action") list.sort((a, b) => Math.max(b.ac, b.sc) - Math.max(a.ac, a.sc) || a.rk - b.rk);
  else if (state.sort === "burst") list.sort((a, b) => a.bi.localeCompare(b.bi) || (a.tm - b.tm) || a.fn.localeCompare(b.fn));

  state.filtered = list;
  document.getElementById("galleryGrid").innerHTML = "";
  state.renderedCount = 0;
  appendNextBatch();
}}

function refreshCardPickState(id) {{
  const grid = document.getElementById("galleryGrid");
  for (const card of grid.children) {{
    if (card.dataset.id !== id) continue;
    const pick = state.picks[id] || "";
    card.className = "card" + (pick === "layer2_edit" ? " user-l2" : pick === "layer1_keep" ? " user-l1" : pick === "reject" ? " user-reject" : "");
    card.querySelectorAll(".pick-btn").forEach(btn => {{
      const act = btn.dataset.action;
      btn.classList.toggle("active-l2", act === "layer2_edit" && pick === "layer2_edit");
      btn.classList.toggle("active-l1", act === "layer1_keep" && pick === "layer1_keep");
      btn.classList.toggle("active-reject", act === "reject" && pick === "reject");
    }});
    break;
  }}
}}

function appendNextBatch() {{
  const grid = document.getElementById("galleryGrid");
  const total = state.filtered.length;
  const startIdx = state.renderedCount;
  const endIdx = Math.min(total, startIdx + BATCH_SIZE);
  const batchItems = state.filtered.slice(startIdx, endIdx);

  const frag = document.createDocumentFragment();
  batchItems.forEach((it, localIdx) => {{
    const globalFilteredIdx = startIdx + localIdx;
    const pick = state.picks[it.id] || "";
    const card = document.createElement("div");
    card.className = "card" + (pick === "layer2_edit" ? " user-l2" : pick === "layer1_keep" ? " user-l1" : pick === "reject" ? " user-reject" : "");
    card.dataset.id = it.id;

    const exifParts = [];
    if (it.fl) exifParts.push(`${{it.fl}}mm`);
    if (it.ap) exifParts.push(`f/${{it.ap}}`);
    if (it.ex) exifParts.push(it.ex);
    if (it.is) exifParts.push(`ISO ${{it.is}}`);
    const exifText = exifParts.join(" · ") || "";

    const isWild = it.tr === "Wildlife";
    const trackBadge = isWild
      ? `<span class="tag tag-wildlife">Wildlife</span>`
      : it.tr === "Landscape"
        ? `<span class="tag tag-landscape">Sunrise/Landscape</span>`
        : `<span class="tag">Other</span>`;
    const speciesTag = isWild ? `<span class="tag">${{it.sp_name}}</span>` : "";
    const hybridTag = it.hyb ? `<span class="tag tag-hybrid">+15% Light</span>` : "";

    const thirdLbl = isWild ? "Eye Contact" : "Scenic/Light";
    const thirdVal = isWild ? it.ey.toFixed(1) : it.sc.toFixed(1);
    const thirdCls = isWild ? scoreClass(it.ey) : scoreClass(it.sc);
    const fourthLbl = isWild ? "Action/Pose" : "Golden Light";
    const fourthVal = isWild ? it.ac.toFixed(1) : it.sc.toFixed(1);
    const fourthCls = isWild ? scoreClass(it.ac) : scoreClass(it.sc);

    card.innerHTML = `
      <div class="card-images">
        <div class="img-slot full-slot" data-idx="${{globalFilteredIdx}}">
          <img loading="lazy" src="${{it.th}}" alt="${{it.fn}}" />
          <span class="rank-pill">#${{it.rk}}</span>
          <span class="img-label">Full Frame</span>
        </div>
        <div class="img-slot" data-idx="${{globalFilteredIdx}}">
          <img loading="lazy" src="${{it.cr}}" alt="Detail Crop" />
          <span class="layer-pill layer-${{it.ly}}">${{layerLabel(it.ly)}}</span>
          <span class="img-label">${{isWild ? "1:1 Head/Subject Crop" : "1:1 Center-Horizon Crop"}}</span>
        </div>
      </div>
      <div class="card-body">
        <div class="card-header-line">
          <div class="fname" title="${{it.sp}}">${{it.fn}}</div>
          <div class="meta-tags">
            ${{trackBadge}}
            ${{speciesTag}}
            ${{hybridTag}}
            <span class="tag">${{it.sf}}</span>
            <span class="tag">${{it.bi}} (${{it.bx}}/${{it.bs}})</span>
          </div>
        </div>
        <div class="scores-row">
          <div class="score-box"><div class="lbl">Combined</div><div class="val ${{scoreClass(it.os)}}">${{it.os.toFixed(1)}}</div></div>
          <div class="score-box"><div class="lbl">Sharpness</div><div class="val ${{scoreClass(it.sh)}}">${{it.sh.toFixed(1)}}</div></div>
          <div class="score-box"><div class="lbl">${{thirdLbl}}</div><div class="val ${{thirdCls}}">${{thirdVal}}</div></div>
          <div class="score-box"><div class="lbl">${{fourthLbl}}</div><div class="val ${{fourthCls}}">${{fourthVal}}</div></div>
        </div>
        <div class="card-footer">
          <div class="exif-line" title="${{it.ts}}">${{exifText}} · ${{it.ts}}</div>
          <div class="pick-btns">
            ${{(it.bs > 1 && it.bx > 1) || state.splits[it.id] ? `<button class="pick-btn split-btn" data-split="${{it.id}}" title="Make this image #1 of a new burst">${{state.splits[it.id] ? "&#8634; Unsplit" : "&#9986; Split"}}</button>` : ""}}
            <button class="pick-btn ${{pick === "layer2_edit" ? "active-l2" : ""}}" data-action="layer2_edit" data-id="${{it.id}}">&#9733; Edit</button>
            <button class="pick-btn ${{pick === "layer1_keep" ? "active-l1" : ""}}" data-action="layer1_keep" data-id="${{it.id}}">&#10003; Keep</button>
            <button class="pick-btn ${{pick === "reject" ? "active-reject" : ""}}" data-action="reject" data-id="${{it.id}}">&#10007; Reject</button>
          </div>
        </div>
      </div>
    `;

    card.querySelectorAll(".img-slot").forEach(slot => {{
      slot.addEventListener("click", () => openModal(globalFilteredIdx));
    }});
    card.querySelectorAll(".pick-btn[data-action]").forEach(btn => {{
      btn.addEventListener("click", () => {{
        const id = btn.dataset.id;
        const act = btn.dataset.action;
        if (state.picks[id] === act) delete state.picks[id];
        else state.picks[id] = act;
        savePicks();
        refreshCardPickState(id);
      }});
    }});
    const splitBtn = card.querySelector(".split-btn");
    if (splitBtn) {{
      splitBtn.addEventListener("click", () => toggleSplitBurst(splitBtn.dataset.split));
    }}

    frag.appendChild(card);
  }});

  grid.appendChild(frag);
  state.renderedCount = endIdx;
  document.getElementById("viewStatus").textContent =
    total === 0 ? "No images match the current filter." : `Showing ${{state.renderedCount}} of ${{total}} images`;

  const bar = document.getElementById("loadMoreBar");
  bar.innerHTML = "";
  if (total > 0 && state.renderedCount < total) {{
    const rem = total - state.renderedCount;
    const btn = document.createElement("button");
    btn.textContent = `Load more (${{Math.min(BATCH_SIZE, rem)}} of ${{rem}} remaining)`;
    btn.onclick = () => appendNextBatch();
    bar.appendChild(btn);
  }} else if (total > 0) {{
    const done = document.createElement("span");
    done.style.fontSize = "0.82rem"; done.style.color = "var(--muted)";
    done.textContent = `All ${{total}} images loaded.`;
    bar.appendChild(done);
  }}
}}

function openModal(idx) {{
  if (idx < 0 || idx >= state.filtered.length) return;
  state.modalIndex = idx;
  const it = state.filtered[idx];
  document.getElementById("modalTitle").textContent =
    `#${{it.rk}} · ${{it.tr}} (${{it.sp_name}}) · ${{it.sf}}/${{it.fn}} · ${{it.bi}} (${{it.br}}/${{it.bs}}) · Score ${{it.os.toFixed(1)}}`;
  document.getElementById("modalFullImg").src = it.th;
  document.getElementById("modalCropImg").src = it.cr;
  document.getElementById("modalCropHdr").textContent =
    it.tr === "Wildlife" ? "1:1 Zoomed Subject / Head Crop" : "1:1 Center-Horizon Detail Crop";
  updateModalPickButtons(it.id);
  document.getElementById("lightboxModal").classList.add("open");
}}

function updateModalPickButtons(id) {{
  const p = state.picks[id] || "";
  document.getElementById("modalL2Btn").className = "pick-btn" + (p === "layer2_edit" ? " active-l2" : "");
  document.getElementById("modalL1Btn").className = "pick-btn" + (p === "layer1_keep" ? " active-l1" : "");
  document.getElementById("modalRejectBtn").className = "pick-btn" + (p === "reject" ? " active-reject" : "");
}}

function toggleModalPick(action) {{
  if (state.modalIndex < 0 || state.modalIndex >= state.filtered.length) return;
  const it = state.filtered[state.modalIndex];
  if (state.picks[it.id] === action) delete state.picks[it.id];
  else state.picks[it.id] = action;
  savePicks();
  updateModalPickButtons(it.id);
  refreshCardPickState(it.id);
}}

function stepModal(delta) {{
  const nxt = state.modalIndex + delta;
  if (nxt >= 0 && nxt < state.filtered.length) openModal(nxt);
}}

function closeModal() {{ document.getElementById("lightboxModal").classList.remove("open"); }}

async function copyLightroomFilter(mode) {{
  let subset = [];
  if (mode === "layer2") subset = ITEMS.filter(it => matchesBaseFilters(it) && it.ly === "layer2_to_edit");
  else if (mode === "layer1") subset = ITEMS.filter(it => matchesBaseFilters(it) && (it.ly === "layer2_to_edit" || it.ly === "layer1_keep"));
  else if (mode === "wildlife") subset = ITEMS.filter(it => matchesBaseFilters(it) && (it.ly === "layer2_to_edit" || it.ly === "layer1_keep") && it.tr === "Wildlife");
  else if (mode === "landscape") subset = ITEMS.filter(it => matchesBaseFilters(it) && (it.ly === "layer2_to_edit" || it.ly === "layer1_keep") && it.tr === "Landscape");
  else if (mode === "picks") subset = ITEMS.filter(it => matchesBaseFilters(it) && (state.picks[it.id] === "layer2_edit" || state.picks[it.id] === "layer1_keep"));

  if (subset.length === 0) {{ showToast("No images in that selection to copy."); return; }}
  const filterStr = subset.map(it => it.st).join(", ");
  try {{ await navigator.clipboard.writeText(filterStr); }}
  catch {{
    const ta = document.createElement("textarea");
    ta.value = filterStr; document.body.appendChild(ta); ta.select();
    document.execCommand("copy"); document.body.removeChild(ta);
  }}
  showToast(`Copied ${{subset.length}} filenames for Lightroom Classic!`);
}}

function exportMyPicks() {{
  const payload = {{
    exported_at: new Date().toISOString(),
    burst_splits: Object.keys(state.splits),
    picks: ITEMS.filter(it => state.picks[it.id]).map(it => ({{
      image_id: it.id,
      filename: it.fn,
      stem: it.st,
      subfolder: it.sf,
      source_path: it.sp,
      user_decision: state.picks[it.id],
      pipeline_layer: it.ly,
      track: it.tr,
      species: it.sp_name,
      overall_score: it.os,
      sharpness_score: it.sh,
      eye_contact_score: it.ey,
      action_score: it.ac,
      scenic_score: it.sc,
      burst_id: it.bi,
    }})),
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: "application/json" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "my_picks.json"; a.click();
  URL.revokeObjectURL(url);
  showToast(`Exported ${{payload.picks.length}} decisions to my_picks.json!`);
}}

init();
</script>
</body>
</html>
"""
    report_path = cfg.output_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)
    logger.info("Interactive review gallery generated at %s", report_path.resolve())
    return report_path
