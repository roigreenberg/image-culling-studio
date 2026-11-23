"""FastAPI Backend Server for ImageCulling Studio."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from webapp.db import CullDatabase, make_dir_id
from webapp.pipeline_runner import TaskManager
from webapp.xmp_sync import normalize_label, restore_xmp_sidecar, write_xmp_sidecar

logger = logging.getLogger("webapp.server")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "cull_output"
STUDIO_DATA_DIR = WORKSPACE_ROOT / "studio_data"
STUDIO_DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = STUDIO_DATA_DIR / "cull_studio.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="ImageCulling Studio", version="1.0.0")
db = CullDatabase(DB_PATH)

# WebSocket connection manager (thread-safe broadcast from background threads)
_ws_clients: Set[WebSocket] = set()
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def broadcast_event(payload: Dict[str, Any]) -> None:
    """Thread-safe broadcast of JSON events to all connected WebSocket clients."""
    if _main_loop is None or not _ws_clients:
        return
    for ws in list(_ws_clients):
        try:
            asyncio.run_coroutine_threadsafe(ws.send_json(payload), _main_loop)
        except Exception:
            pass


tasks = TaskManager(db=db, broadcast=broadcast_event)


@app.on_event("startup")
async def _on_startup() -> None:
    global _main_loop
    _main_loop = asyncio.get_running_loop()

    # Automatically import previous CLI session results.json if present
    results_json = DEFAULT_OUTPUT_DIR / "results.json"
    if results_json.exists():
        imported = db.import_existing_results_json(results_json, DEFAULT_OUTPUT_DIR)
        if imported > 0:
            logger.info("Auto-imported %d images from previous session results.json", imported)

    # Auto-sync XMP sidecars, in-camera ratings & Lens EXIF on startup if not synced yet
    dirs = db.list_directories()
    if dirs:
        d = dirs[0]
        src_path = Path(d["path"])
        if src_path.exists():
            facets = db.get_filter_facets(d["dir_id"])
            if not facets.get("lenses") and (d.get("stats", {}).get("rated", 0) == 0):
                logger.info("Auto-starting background EXIF/Camera/XMP scan for %s", src_path)
                tasks.start_directory_scan(d["dir_id"])


# ------------------------------------------------------------------
# Request Models
# ------------------------------------------------------------------
class OpenDirectoryRequest(BaseModel):
    path: str
    scan_now: bool = True


class MetadataUpdateRequest(BaseModel):
    rating: Optional[int] = None
    label: Optional[str] = None
    user_flag: Optional[int] = None


class BatchMetadataUpdateRequest(BaseModel):
    image_ids: List[str]
    rating: Optional[int] = None
    label: Optional[str] = None
    user_flag: Optional[int] = None


class PipelineStartRequest(BaseModel):
    dir_id: str
    mode: str = "unculled"  # "full", "unculled", "selected"
    subfolder: str = ""
    selected_ids: List[str] = []


class ReselectRequest(BaseModel):
    dir_id: str


class TrainRankerRequest(BaseModel):
    auto_reselect: bool = True


class PerfEventsRequest(BaseModel):
    events: List[Dict[str, Any]]


class PreviewGenRequest(BaseModel):
    dir_id: str
    subfolder: str = ""
    only_layers: Optional[List[str]] = None


class ApplyAiXmpRequest(BaseModel):
    dir_id: str
    subfolder: str = ""
    image_ids: Optional[List[str]] = None
    dry_run: bool = False


class SessionStateRequest(BaseModel):
    state: Dict[str, Any]


# ------------------------------------------------------------------
# Static & Root
# ------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>ImageCulling Studio UI not found</h1>", status_code=404)
    return HTMLResponse(idx.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# WebSocket Live Events
# ------------------------------------------------------------------
@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        # Send immediate current task status on connect
        await websocket.send_json({"type": "task_progress", "data": tasks.get_status()})
        while True:
            # Keep-alive ping/pong
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(websocket)


# ------------------------------------------------------------------
# Directories & Native Folder Picker
# ------------------------------------------------------------------
@app.get("/api/directories")
async def list_directories() -> Dict[str, Any]:
    # Auto-refresh if the CLI culling session produced a newer results.json
    results_json = DEFAULT_OUTPUT_DIR / "results.json"
    if results_json.exists():
        imported = db.import_existing_results_json(results_json, DEFAULT_OUTPUT_DIR)
        if imported > 0:
            _thumb_path_cache.clear()
            _crop_path_cache.clear()
    dirs = db.list_directories()
    session = db.get_session_state()
    return {"directories": dirs, "session": session}


@app.post("/api/directories/browse")
async def browse_native_folder() -> Dict[str, Any]:
    """Open native Windows folder picker dialog on the local machine."""
    def _pick() -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory(title="Select Photo Directory to Cull")
            root.destroy()
            return folder or ""
        except Exception as exc:
            logger.warning("Native folder picker failed: %s", exc)
            return ""

    chosen = await asyncio.to_thread(_pick)
    return {"path": chosen}


@app.post("/api/directories/open")
async def open_directory(req: OpenDirectoryRequest) -> Dict[str, Any]:
    p = Path(req.path.strip().strip('"'))
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist or is not accessible: {p}")

    dir_id = make_dir_id(str(p))
    existing = db.get_directory(dir_id)
    if existing:
        out_dir = Path(existing["output_dir"])
        is_legacy = bool(existing.get("is_legacy_default"))
    else:
        out_dir = DEFAULT_OUTPUT_DIR / "projects" / dir_id
        out_dir.mkdir(parents=True, exist_ok=True)
        is_legacy = False

    d = db.register_directory(p, out_dir, is_legacy_default=is_legacy)
    db.add_event_log("INFO", "system", f"Opened directory: {p}", {"dir_id": d["dir_id"]})
    scan_started = False
    if req.scan_now:
        ok, _ = tasks.start_directory_scan(d["dir_id"])
        scan_started = ok
    return {"directory": d, "scan_started": scan_started}


@app.get("/api/directories/{dir_id}")
async def get_directory(dir_id: str) -> Dict[str, Any]:
    d = db.get_directory(dir_id)
    if not d:
        raise HTTPException(status_code=404, detail="Directory not found")
    return d


@app.post("/api/directories/{dir_id}/scan")
async def scan_directory(dir_id: str, subfolder: str = "") -> Dict[str, Any]:
    ok, msg = tasks.start_directory_scan(dir_id, subfolder=subfolder)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "message": msg}


@app.get("/api/directories/{dir_id}/facets")
async def get_facets(dir_id: str, subfolder: str = "") -> Dict[str, Any]:
    return db.get_filter_facets(dir_id, subfolder=subfolder)


# ------------------------------------------------------------------
# Images Query & Details
# ------------------------------------------------------------------
@app.get("/api/images")
async def query_images(
    dir_id: str,
    subfolder: str = "",
    ai_layer: str = "",
    use_ai_layer: bool = False,
    culled_filter: str = "",
    min_rating: Optional[int] = None,
    exact_rating: Optional[int] = None,
    rating_val: Optional[int] = None,
    rating_op: str = "gte",
    label: str = "",
    speed_tier: str = "",
    camera: str = "",
    lens: str = "",
    track: str = "",
    species: str = "",
    scenario_type: str = "",
    hybrid_only: bool = False,
    burst_id: str = "",
    search: str = "",
    only_burst_best: bool = False,
    sort_by: str = "overall_rank",
    sort_dir: str = "asc",
    offset: int = 0,
    limit: int = Query(default=200, le=2000),
) -> Dict[str, Any]:
    res = db.query_images(
        dir_id=dir_id,
        subfolder=subfolder,
        ai_layer=ai_layer,
        use_ai_layer=use_ai_layer,
        culled_filter=culled_filter,
        min_rating=min_rating,
        exact_rating=exact_rating,
        rating_val=rating_val,
        rating_op=rating_op,
        label=label,
        speed_tier=speed_tier,
        camera=camera,
        lens=lens,
        track=track,
        species=species,
        scenario_type=scenario_type,
        hybrid_only=hybrid_only,
        burst_id=burst_id,
        search=search,
        only_burst_best=only_burst_best,
        sort_by=sort_by,
        sort_dir=sort_dir,
        offset=offset,
        limit=limit,
    )
    _warm_media_caches(res["items"])
    return res


@app.get("/api/images/{image_id}")
async def get_image_detail(image_id: str) -> Dict[str, Any]:
    img = db.get_image(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    # On-the-fly Canon AF focus-point extraction (fast header-only, cached in DB)
    if img.get("af_points") is None:
        src = Path(img["source_path"])
        if src.exists():
            from src.io_utils import read_fast_exif_and_camera_meta
            meta = await asyncio.to_thread(read_fast_exif_and_camera_meta, src)
            af_pts = meta.get("af_points") or []
            db.set_af_points(image_id, af_pts)
            img["af_points"] = af_pts
        else:
            img["af_points"] = []
    siblings = db.get_burst_siblings(img["dir_id"], img.get("burst_id", ""))
    return {"image": img, "burst_siblings": siblings}


# ------------------------------------------------------------------
# Fast Media Endpoints (Thumb, Head Crop, HQ JPG Preview)
# ------------------------------------------------------------------
# In-memory path caches so thumbnail/crop requests do ZERO SQLite queries
_dir_output_cache: Dict[str, Path] = {}
_thumb_path_cache: Dict[str, Path] = {}
_crop_path_cache: Dict[str, Path] = {}


def _resolve_dir_output(image: Dict[str, Any]) -> Path:
    did = image["dir_id"]
    if did not in _dir_output_cache:
        d = db.get_directory(did)
        _dir_output_cache[did] = Path(d["output_dir"]) if d else DEFAULT_OUTPUT_DIR
    return _dir_output_cache[did]


def _warm_media_caches(items: List[Dict[str, Any]]) -> None:
    for img in items:
        iid = img["image_id"]
        if iid not in _thumb_path_cache and img.get("thumb_rel_path"):
            out_dir = _resolve_dir_output(img)
            _thumb_path_cache[iid] = out_dir / img["thumb_rel_path"]
        if iid not in _crop_path_cache and img.get("crop_rel_path"):
            out_dir = _resolve_dir_output(img)
            _crop_path_cache[iid] = out_dir / img["crop_rel_path"]


@app.get("/api/media/thumb/{image_id}")
async def serve_thumb(image_id: str) -> FileResponse:
    cached = _thumb_path_cache.get(image_id)
    if cached and cached.exists():
        return FileResponse(cached, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800, immutable"})
    img = db.get_image(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    out_dir = _resolve_dir_output(img)
    rel = img.get("thumb_rel_path")
    if rel and (out_dir / rel).exists():
        p = out_dir / rel
        _thumb_path_cache[image_id] = p
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800, immutable"})
    # Fallback: generate HQ/thumb on the fly if source is accessible
    p = await asyncio.to_thread(tasks.ensure_single_hq_preview, img, out_dir)
    if p and p.exists():
        _thumb_path_cache[image_id] = p
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800, immutable"})
    raise HTTPException(status_code=404, detail="Thumbnail not available")


@app.get("/api/media/crop/{image_id}")
async def serve_crop(image_id: str) -> FileResponse:
    cached = _crop_path_cache.get(image_id)
    if cached and cached.exists():
        return FileResponse(cached, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800, immutable"})
    img = db.get_image(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    out_dir = _resolve_dir_output(img)
    rel = img.get("crop_rel_path")
    if rel and (out_dir / rel).exists():
        p = out_dir / rel
        _crop_path_cache[image_id] = p
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800, immutable"})
    # Fallback to HQ preview
    return await serve_hq(image_id)


@app.get("/api/media/hq/{image_id}")
async def serve_hq(image_id: str) -> FileResponse:
    img = db.get_image(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    out_dir = _resolve_dir_output(img)
    p = await asyncio.to_thread(tasks.ensure_single_hq_preview, img, out_dir)
    if p and p.exists():
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
    raise HTTPException(status_code=404, detail="HQ preview not available")


@app.get("/api/media/full/{image_id}")
async def serve_full(image_id: str, raw: int = 0) -> FileResponse:
    img = db.get_image(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    out_dir = _resolve_dir_output(img)
    p = await asyncio.to_thread(tasks.ensure_full_res_jpeg, img, out_dir, 500, bool(raw))
    if p and p.exists():
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
    raise HTTPException(status_code=404, detail="Full-resolution image not available")


# ------------------------------------------------------------------
# Real-Time Rating, Color Label & XMP Sync
# ------------------------------------------------------------------
@app.post("/api/images/{image_id}/metadata")
async def update_image_metadata(image_id: str, req: MetadataUpdateRequest) -> Dict[str, Any]:
    img = db.get_image(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    settings = db.get_settings()
    realtime_xmp = bool(settings.get("realtime_xmp_write", True))
    xmp_synced = False
    prev_xml: Optional[str] = None
    xmp_warning: Optional[str] = None

    new_rating = req.rating if req.rating is not None else img["effective_rating"]
    new_label = normalize_label(req.label) if req.label is not None else img["effective_label"]

    if realtime_xmp and (req.rating is not None or req.label is not None):
        ok, prev_xml, _, err = write_xmp_sidecar(
            Path(img["source_path"]),
            rating=new_rating,
            label=new_label,
        )
        if ok:
            xmp_synced = True
        else:
            xmp_warning = err
            db.add_event_log("WARNING", "xmp", f"XMP write failed for {img['filename']}: {err}")

    updated = db.record_metadata_change(
        image_id=image_id,
        rating=req.rating,
        label=req.label,
        user_flag=req.user_flag,
        source="user",
        action_type="edit",
        xmp_synced=xmp_synced,
        prev_xmp_xml=prev_xml,
    )
    return {"image": updated, "xmp_synced": xmp_synced, "xmp_warning": xmp_warning}


@app.post("/api/images/batch-metadata")
async def batch_update_metadata(req: BatchMetadataUpdateRequest) -> Dict[str, Any]:
    settings = db.get_settings()
    realtime_xmp = bool(settings.get("realtime_xmp_write", True))
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    updated_count = 0
    xmp_ok = 0
    xmp_fail = 0

    for iid in req.image_ids:
        img = db.get_image(iid)
        if not img:
            continue
        xmp_synced = False
        prev_xml = None
        new_rating = req.rating if req.rating is not None else img["effective_rating"]
        new_label = normalize_label(req.label) if req.label is not None else img["effective_label"]

        if realtime_xmp and (req.rating is not None or req.label is not None):
            ok, prev_xml, _, _ = write_xmp_sidecar(
                Path(img["source_path"]), rating=new_rating, label=new_label
            )
            if ok:
                xmp_synced = True
                xmp_ok += 1
            else:
                xmp_fail += 1

        db.record_metadata_change(
            image_id=iid,
            rating=req.rating,
            label=req.label,
            user_flag=req.user_flag,
            source="user",
            action_type="batch_edit",
            xmp_synced=xmp_synced,
            prev_xmp_xml=prev_xml,
            batch_id=batch_id,
        )
        updated_count += 1

    db.add_event_log(
        "INFO",
        "xmp",
        f"Batch updated {updated_count} images (XMP synced: {xmp_ok}, failed: {xmp_fail})",
        {"batch_id": batch_id, "rating": req.rating, "label": req.label},
    )
    return {"updated": updated_count, "xmp_synced": xmp_ok, "xmp_failed": xmp_fail, "batch_id": batch_id}


# ------------------------------------------------------------------
# Change History & Restore / Undo
# ------------------------------------------------------------------
@app.get("/api/changes")
async def list_changes(limit: int = 200, dir_id: str = "") -> Dict[str, Any]:
    return {"changes": db.get_change_log(limit=limit, dir_id=dir_id)}


def _restore_single_change(entry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    img = db.get_image(entry["image_id"])
    if not img:
        return False, "Image not found"
    xmp_err = None
    if entry.get("xmp_synced"):
        ok, err = restore_xmp_sidecar(Path(img["source_path"]), entry.get("prev_xmp_xml"))
        if not ok:
            xmp_err = err
    db.mark_change_reverted(entry["id"], entry["field"], entry["old_value"], entry["image_id"])
    return True, xmp_err


@app.post("/api/changes/{change_id}/restore")
async def restore_change(change_id: int) -> Dict[str, Any]:
    entry = db.get_change_entry(change_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Change entry not found")
    if entry["reverted"]:
        return {"ok": True, "message": "Already reverted"}

    # If part of a batch, restore just this one or the batch? Here restore this single entry
    ok, xmp_err = _restore_single_change(entry)
    db.add_event_log(
        "INFO",
        "system",
        f"Restored {entry['filename']} ({entry['field']}: {entry['new_value']} -> {entry['old_value']})",
    )
    return {"ok": ok, "image": db.get_image(entry["image_id"]), "xmp_warning": xmp_err}


@app.post("/api/changes/undo-last")
async def undo_last_change(dir_id: str = "") -> Dict[str, Any]:
    entry = db.get_latest_unreverted_change(dir_id=dir_id)
    if not entry:
        return {"ok": False, "message": "Nothing to undo"}

    restored_images = []
    if entry.get("batch_id"):
        batch_entries = db.get_batch_changes(entry["batch_id"])
        for be in batch_entries:
            _restore_single_change(be)
            restored_images.append(be["image_id"])
        msg = f"Undid batch of {len(batch_entries)} changes ({entry['batch_id']})"
    else:
        _restore_single_change(entry)
        restored_images.append(entry["image_id"])
        msg = f"Undid {entry['filename']} ({entry['field']}: {entry['new_value']} -> {entry['old_value']})"

    db.add_event_log("INFO", "system", msg)
    return {"ok": True, "message": msg, "image_ids": restored_images}


# ------------------------------------------------------------------
# AI-to-XMP Sync
# ------------------------------------------------------------------
@app.post("/api/xmp/apply-ai-mapping")
async def apply_ai_xmp(req: ApplyAiXmpRequest) -> Dict[str, Any]:
    result = await asyncio.to_thread(
        tasks.apply_ai_mapping_to_xmp,
        req.dir_id,
        req.subfolder,
        req.image_ids,
        req.dry_run,
    )
    return result


# ------------------------------------------------------------------
# Pipeline & Preview Generation Controls
# ------------------------------------------------------------------
@app.post("/api/pipeline/start")
async def start_pipeline(req: PipelineStartRequest) -> Dict[str, Any]:
    ok, msg = tasks.start_culling(
        dir_id=req.dir_id,
        mode=req.mode,
        subfolder=req.subfolder,
        selected_ids=req.selected_ids,
    )
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "run_id": msg}


@app.post("/api/pipeline/reselect")
async def reselect_pipeline(req: ReselectRequest) -> Dict[str, Any]:
    ok, msg = tasks.start_reselect(dir_id=req.dir_id)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "run_id": msg}


@app.post("/api/directories/{dir_id}/sync-results")
async def sync_cli_results(dir_id: str) -> Dict[str, Any]:
    d = db.get_directory(dir_id)
    if not d:
        raise HTTPException(status_code=404, detail="Directory not found")
    out_dir = Path(d["output_dir"])
    res_json = out_dir / "results.json"
    if not res_json.is_file():
        raise HTTPException(status_code=404, detail=f"results.json not found in {out_dir}")
    count = await asyncio.to_thread(db.import_existing_results_json, res_json, out_dir, True)
    _thumb_path_cache.clear()
    _crop_path_cache.clear()
    return {"ok": True, "imported": count, "has_embeddings": (out_dir / "embeddings.npz").is_file()}


@app.post("/api/pipeline/cancel")
async def cancel_pipeline() -> Dict[str, Any]:
    ok = tasks.cancel_current()
    return {"ok": ok}


@app.get("/api/pipeline/status")
async def pipeline_status() -> Dict[str, Any]:
    return tasks.get_status()


@app.get("/api/pipeline/runs")
async def pipeline_runs(limit: int = 30) -> Dict[str, Any]:
    return {"runs": db.list_pipeline_runs(limit=limit)}


@app.post("/api/previews/generate")
async def generate_previews(req: PreviewGenRequest) -> Dict[str, Any]:
    ok, msg = tasks.start_hq_preview_batch(
        dir_id=req.dir_id,
        subfolder=req.subfolder,
        only_layers=req.only_layers,
    )
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "message": msg}


# ------------------------------------------------------------------
# Logs, Learning Insights, Settings & Session State
# ------------------------------------------------------------------
@app.get("/api/logs")
async def get_logs(limit: int = 200, level: str = "", category: str = "", run_id: str = "") -> Dict[str, Any]:
    return {"logs": db.get_event_logs(limit=limit, level=level, category=category, run_id=run_id)}


@app.post("/api/perf/log")
async def post_perf_logs(req: PerfEventsRequest) -> Dict[str, Any]:
    logged = 0
    for ev in req.events[:50]:
        total_ms = float(ev.get("total_ms") or 0.0)
        lvl = "WARNING" if total_ms >= 250.0 else "INFO"
        msg = str(ev.get("message") or f"perf {ev.get('event', '')} {total_ms:.0f}ms")
        entry = db.add_event_log(lvl, "perf", msg, details=ev)
        broadcast_event({"type": "log", "data": entry})
        logged += 1
    return {"ok": True, "logged": logged}


@app.get("/api/learning/{dir_id}")
async def get_learning(dir_id: str) -> Dict[str, Any]:
    stats = db.get_learning_stats(dir_id)
    picks = db.get_training_picks(dir_id)
    n_keep = sum(1 for p in picks if p["user_decision"] in ("layer2_edit", "layer1_keep"))
    n_rej = sum(1 for p in picks if p["user_decision"] == "reject")
    model_path = WORKSPACE_ROOT / "models" / "personal_ranker.pkl"
    stats["training_ready"] = {
        "total_picks": len(picks),
        "keep_edit": n_keep,
        "reject": n_rej,
        "can_train": n_keep >= 3 and n_rej >= 3,
        "model_exists": model_path.is_file(),
    }
    return stats


@app.post("/api/learning/{dir_id}/train")
async def train_learning_ranker(dir_id: str, req: Optional[TrainRankerRequest] = None) -> Dict[str, Any]:
    try:
        res = await asyncio.to_thread(tasks.train_personal_ranker, dir_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Personal ranker training failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if req is None or req.auto_reselect:
        tasks.start_reselect(dir_id)
        res["reselect_started"] = True
    return res


@app.get("/api/settings")
async def get_settings() -> Dict[str, Any]:
    return db.get_settings()


@app.put("/api/settings")
async def put_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    saved = db.update_settings(updates)
    db.add_event_log("INFO", "system", "Updated studio & XMP sync settings")
    return saved


@app.get("/api/session")
async def get_session() -> Dict[str, Any]:
    return db.get_session_state()


@app.put("/api/session")
async def put_session(req: SessionStateRequest) -> Dict[str, Any]:
    db.save_session_state(req.state)
    return {"ok": True}
