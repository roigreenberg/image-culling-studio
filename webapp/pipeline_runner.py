"""Background GPU Culling Runner, Fast Directory EXIF/XMP Scanner, and HQ Preview Generator."""

from __future__ import annotations

import collections
import gc
import json
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import PipelineConfig
from src.io_utils import (
    discover_images,
    load_preview_bgr_and_meta,
    read_fast_exif_and_camera_meta,
    save_jpeg_bgr,
)
from webapp.db import CullDatabase, make_image_id, utc_now_iso
from webapp.xmp_sync import normalize_label, read_xmp_sidecar, write_xmp_sidecar

logger = logging.getLogger(__name__)

BroadcastFn = Callable[[Dict[str, Any]], None]


class TaskManager:
    """Coordinates background GPU culling, directory metadata scanning, and HQ JPG preview generation."""

    def __init__(self, db: CullDatabase, broadcast: BroadcastFn):
        self.db = db
        self.broadcast = broadcast
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.state: Dict[str, Any] = {
            "active": False,
            "task_type": "",  # "cull", "scan", "previews"
            "run_id": "",
            "dir_id": "",
            "mode": "",
            "stage": "idle",
            "message": "Ready",
            "current": 0,
            "total": 0,
            "percent": 0.0,
            "speed_ips": 0.0,
            "eta_seconds": 0.0,
            "current_file": "",
            "started_at": "",
            "errors": 0,
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.state)

    def _emit_status(self, **updates: Any) -> None:
        with self._lock:
            self.state.update(updates)
            snap = dict(self.state)
        self.broadcast({"type": "task_progress", "data": snap})

    def _log(self, level: str, category: str, message: str, details: Optional[Dict[str, Any]] = None, run_id: str = "") -> None:
        entry = self.db.add_event_log(level, category, message, details=details, run_id=run_id)
        self.broadcast({"type": "log", "data": entry})

    def cancel_current(self) -> bool:
        with self._lock:
            if not self.state["active"]:
                return False
            self._cancel_event.set()
            self.state["message"] = "Cancelling..."
        self._log("WARNING", "system", "Cancellation requested by user.", run_id=self.state.get("run_id", ""))
        self._emit_status(message="Cancelling...")
        return True

    # ------------------------------------------------------------------
    # Fast Directory Scan (Discover Files + EXIF + Camera Rating + XMP)
    # ------------------------------------------------------------------
    def start_directory_scan(self, dir_id: str, subfolder: str = "") -> Tuple[bool, str]:
        with self._lock:
            if self.state["active"]:
                return False, f"Another task ({self.state['task_type']}) is already running."
            self._cancel_event.clear()
            self.state.update(
                {
                    "active": True,
                    "task_type": "scan",
                    "run_id": f"scan_{uuid.uuid4().hex[:8]}",
                    "dir_id": dir_id,
                    "mode": subfolder or "all",
                    "stage": "scanning",
                    "message": "Discovering files...",
                    "current": 0,
                    "total": 0,
                    "percent": 0.0,
                    "speed_ips": 0.0,
                    "eta_seconds": 0.0,
                    "current_file": "",
                    "started_at": utc_now_iso(),
                    "errors": 0,
                }
            )
        t = threading.Thread(target=self._run_scan, args=(dir_id, subfolder), daemon=True)
        self._worker_thread = t
        t.start()
        return True, "Scan started"

    def _run_scan(self, dir_id: str, subfolder: str) -> None:
        run_id = self.state["run_id"]
        d = self.db.get_directory(dir_id)
        if not d:
            self._emit_status(active=False, stage="error", message="Directory not found")
            return
        source_dir = Path(d["path"])
        is_legacy = bool(d.get("is_legacy_default"))
        t0 = time.perf_counter()

        try:
            if not source_dir.exists():
                msg = f"Source directory is not accessible: {source_dir}"
                self._log("ERROR", "scan", msg, run_id=run_id)
                self._emit_status(active=False, stage="error", message=msg)
                return

            discovered = discover_images(source_dir)
            if subfolder and subfolder != "__all__":
                discovered = [(p, sf) for p, sf in discovered if sf == subfolder]

            total = len(discovered)
            self._log("INFO", "scan", f"Scanning EXIF, camera ratings & XMP for {total} files in {source_dir.name}...", run_id=run_id)
            self._emit_status(total=total, message=f"Scanning {total} files (EXIF + XMP)...")

            def _scan_one(item: Tuple[Path, str]) -> Dict[str, Any]:
                p, sf = item
                meta = read_fast_exif_and_camera_meta(p)
                xmp = read_xmp_sidecar(p)
                img_id = make_image_id(dir_id, sf, p.name, legacy=is_legacy)
                return {
                    "image_id": img_id,
                    "source_path": str(p),
                    "subfolder": sf,
                    "filename": p.name,
                    "stem": p.stem,
                    "file_ext": p.suffix.lower(),
                    "file_size": meta.get("file_size", 0),
                    "timestamp": meta.get("timestamp") or 0.0,
                    "timestamp_str": meta.get("timestamp_str", ""),
                    "focal_length": meta.get("focal_length"),
                    "f_number": meta.get("f_number"),
                    "exposure_time": meta.get("exposure_time", ""),
                    "exposure_seconds": meta.get("exposure_seconds"),
                    "iso": meta.get("iso"),
                    "camera_model": meta.get("camera_model", ""),
                    "lens_model": meta.get("lens_model", ""),
                    "camera_rating": meta.get("camera_rating", 0),
                    "camera_label": meta.get("camera_label", ""),
                    "af_points": meta.get("af_points", []),
                    "xmp_exists": xmp["exists"],
                    "xmp_rating": xmp["rating"],
                    "xmp_label": xmp["label"],
                }

            batch: List[Dict[str, Any]] = []
            done = 0
            total_ins = 0
            total_upd = 0
            with ThreadPoolExecutor(max_workers=12) as pool:
                futures = [pool.submit(_scan_one, item) for item in discovered]
                for fut in as_completed(futures):
                    if self._cancel_event.is_set():
                        break
                    try:
                        res = fut.result()
                        batch.append(res)
                    except Exception as exc:
                        self._log("WARNING", "scan", f"Scan error: {exc}", run_id=run_id)
                    done += 1
                    if len(batch) >= 200:
                        ins, upd = self.db.upsert_scanned_images(dir_id, batch)
                        total_ins += ins
                        total_upd += upd
                        batch.clear()
                        elapsed = max(0.01, time.perf_counter() - t0)
                        ips = done / elapsed
                        eta = (total - done) / max(0.1, ips)
                        self._emit_status(
                            current=done,
                            percent=round(100.0 * done / max(1, total), 1),
                            speed_ips=round(ips, 1),
                            eta_seconds=round(eta, 1),
                            current_file=res["filename"] if "res" in locals() else "",
                            message=f"Scanned {done}/{total} ({total_ins} new, {total_upd} synced)",
                        )
            if batch:
                ins, upd = self.db.upsert_scanned_images(dir_id, batch)
                total_ins += ins
                total_upd += upd

            elapsed = time.perf_counter() - t0
            status_msg = f"Scan complete: {total_ins} new files, {total_upd} EXIF/XMP synced in {elapsed:.1f}s"
            self._log("INFO", "scan", status_msg, {"inserted": total_ins, "updated": total_upd}, run_id=run_id)
            self._emit_status(active=False, stage="done", current=done, percent=100.0, message=status_msg)
            self.broadcast({"type": "refresh_directory", "dir_id": dir_id})
        except Exception as exc:
            logger.exception("Directory scan failed")
            self._log("ERROR", "scan", f"Directory scan failed: {exc}", run_id=run_id)
            self._emit_status(active=False, stage="error", message=f"Scan failed: {exc}")

    # ------------------------------------------------------------------
    # Fast HQ JPG Preview Generator (on-demand + batch)
    # ------------------------------------------------------------------
    def ensure_single_hq_preview(self, image: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """
        Return path to the highest-quality fast JPG preview for `image`:
          1) Existing 2048px export_layer2_path or export_layer1_path
          2) Existing cached 1600px hq_preview_rel_path
          3) On-the-fly extract from CR3/JPG source -> save 1600px Q86 JPG
          4) Fallback to existing 720px thumb_rel_path
        """
        for key in ("export_layer2_path", "export_layer1_path", "hq_preview_rel_path"):
            rel = image.get(key)
            if rel:
                p = output_dir / rel
                if p.exists():
                    return p

        # Try on-demand generation from source CR3/JPG
        src = Path(image["source_path"])
        if src.exists():
            settings = self.db.get_settings()
            max_dim = int(settings.get("hq_preview_max_dim", 1600))
            quality = int(settings.get("hq_preview_quality", 86))
            rel = f"gallery_assets/previews/{image['image_id']}.jpg"
            out_p = output_dir / rel
            if not out_p.exists():
                img_bgr, _ = load_preview_bgr_and_meta(src)
                if img_bgr is not None:
                    save_jpeg_bgr(img_bgr, out_p, quality=quality, max_dim=max_dim)
                    self.db.set_hq_preview_path(image["image_id"], rel)
                    return out_p
            else:
                self.db.set_hq_preview_path(image["image_id"], rel)
                return out_p

        # Fallback to thumbnail
        thumb_rel = image.get("thumb_rel_path")
        if thumb_rel:
            tp = output_dir / thumb_rel
            if tp.exists():
                return tp
        return None

    def ensure_full_res_jpeg(
        self,
        image: Dict[str, Any],
        output_dir: Path,
        max_cache_files: int = 500,
        force_raw_full: bool = False,
    ) -> Optional[Path]:
        """
        Extract and cache the crisp Zoom Preview JPEG at the user's configured resolution
        (default 3072px Q88 ~1.2MB — 7x lighter in browser GPU RAM and 12x faster to decode
        than 45MP). When force_raw_full=True (Hold-Q peek), extracts the true 45MP (8192px) JPEG.
        """
        src = Path(image["source_path"])
        if not src.exists():
            return self.ensure_single_hq_preview(image, output_dir)

        settings = self.db.get_settings()
        target_dim = 8192 if force_raw_full else int(settings.get("zoom_preview_max_dim", 3072) or 3072)
        target_q = 94 if force_raw_full else int(settings.get("zoom_preview_quality", 88) or 88)
        if force_raw_full:
            max_cache_files = 40

        # If source is already a JPEG and target_dim >= 8000, serve directly
        if src.suffix.lower() in (".jpg", ".jpeg") and target_dim >= 8000:
            return src

        full_dir = output_dir / "gallery_assets" / f"zoom_{target_dim}"
        out_p = full_dir / f"{image['image_id']}.jpg"
        if out_p.exists():
            try:
                out_p.touch()
            except Exception:
                pass
            return out_p

        try:
            import io
            import os
            import rawpy
            from PIL import Image
            from src.config import RAW_EXTENSIONS
            from src.io_utils import apply_orientation_bgr, parse_cr3_exif_from_stream, parse_raw_exif_from_stream

            full_dir.mkdir(parents=True, exist_ok=True)
            orient = 1
            suffix = src.suffix.lower()
            if suffix in RAW_EXTENSIONS:
                with open(src, "rb") as fh:
                    meta = parse_cr3_exif_from_stream(fh) if suffix == ".cr3" else parse_raw_exif_from_stream(fh)
                    orient = int(meta.get("orientation", 1) or 1)
                    fh.seek(0)
                    with rawpy.imread(fh) as raw:
                        thumb = raw.extract_thumb()
                if thumb.format != rawpy.ThumbFormat.JPEG:
                    return self.ensure_single_hq_preview(image, output_dir)
                data = thumb.data
            else:
                with open(src, "rb") as fh:
                    data = fh.read()

            tmp_p = out_p.with_suffix(".jpg.tmp")
            # Check embedded JPEG EXIF orientation too, in case CR3 CMT1 differs
            if orient == 1:
                try:
                    with Image.open(io.BytesIO(data)) as im_chk:
                        j_orient = int(im_chk.getexif().get(0x0112, 1) or 1)
                        if j_orient in (3, 6, 8):
                            orient = j_orient
                except Exception:
                    pass

            if target_dim >= 8000:
                # Full 45MP path — if embedded JPEG already carries EXIF orientation (or is landscape orient=1),
                # write raw JPEG bytes directly (zero re-encode, browser auto-orients upright).
                # Only decode+rotate if CR3 had orient in (3,6,8) AND the JPEG itself lacked that EXIF tag.
                j_orient = 1
                try:
                    with Image.open(io.BytesIO(data)) as im_chk:
                        j_orient = int(im_chk.getexif().get(0x0112, 1) or 1)
                except Exception:
                    pass

                if j_orient == orient or orient == 1:
                    with open(tmp_p, "wb") as out_fh:
                        out_fh.write(data)
                else:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None:
                        return self.ensure_single_hq_preview(image, output_dir)
                    h, w = bgr.shape[:2]
                    # Only rotate if cv2.imdecode did not already auto-orient it upright
                    if orient == 3 or (orient in (6, 8) and w > h):
                        bgr = apply_orientation_bgr(bgr, orient)
                    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), target_q])
                    if not ok:
                        return self.ensure_single_hq_preview(image, output_dir)
                    with open(tmp_p, "wb") as out_fh:
                        out_fh.write(buf.tobytes())
            else:
                # Fast hardware 1/2 DCT decode (4096x2732) -> optional resize to target_dim (e.g. 3072px)
                arr = np.frombuffer(data, dtype=np.uint8)
                flag = cv2.IMREAD_REDUCED_COLOR_2 if target_dim <= 4096 else cv2.IMREAD_COLOR
                bgr = cv2.imdecode(arr, flag)
                if bgr is None:
                    return self.ensure_single_hq_preview(image, output_dir)
                h, w = bgr.shape[:2]
                if orient == 3 or (orient in (6, 8) and w > h):
                    bgr = apply_orientation_bgr(bgr, orient)
                    h, w = bgr.shape[:2]
                elif w > h and int(image.get("preview_height") or 0) > int(image.get("preview_width") or 0):
                    bgr = apply_orientation_bgr(bgr, orient if orient in (6, 8) else 6)
                    h, w = bgr.shape[:2]
                longest = max(h, w)
                if longest > target_dim:
                    s = target_dim / float(longest)
                    bgr = cv2.resize(
                        bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), target_q])
                if not ok:
                    return self.ensure_single_hq_preview(image, output_dir)
                with open(tmp_p, "wb") as out_fh:
                    out_fh.write(buf.tobytes())

            os.replace(tmp_p, out_p)

            # Lightweight LRU eviction so cache stays bounded on disk
            try:
                cached_files = list(full_dir.glob("*.jpg"))
                if len(cached_files) > max_cache_files:
                    cached_files.sort(key=lambda p: p.stat().st_mtime)
                    for old_p in cached_files[: len(cached_files) - max_cache_files]:
                        old_p.unlink(missing_ok=True)
            except Exception:
                pass
            return out_p
        except Exception as exc:
            logger.warning("Zoom preview extract failed for %s: %s", src.name, exc)

        return self.ensure_single_hq_preview(image, output_dir)

    def start_hq_preview_batch(
        self,
        dir_id: str,
        subfolder: str = "",
        only_layers: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        with self._lock:
            if self.state["active"]:
                return False, f"Another task ({self.state['task_type']}) is already running."
            self._cancel_event.clear()
            self.state.update(
                {
                    "active": True,
                    "task_type": "previews",
                    "run_id": f"prev_{uuid.uuid4().hex[:8]}",
                    "dir_id": dir_id,
                    "mode": subfolder or "all",
                    "stage": "generating_previews",
                    "message": "Preparing fast HQ JPG previews...",
                    "current": 0,
                    "total": 0,
                    "percent": 0.0,
                    "speed_ips": 0.0,
                    "eta_seconds": 0.0,
                    "current_file": "",
                    "started_at": utc_now_iso(),
                    "errors": 0,
                }
            )
        t = threading.Thread(
            target=self._run_hq_preview_batch, args=(dir_id, subfolder, only_layers), daemon=True
        )
        self._worker_thread = t
        t.start()
        return True, "HQ preview generation started"

    def _run_hq_preview_batch(
        self, dir_id: str, subfolder: str, only_layers: Optional[List[str]]
    ) -> None:
        run_id = self.state["run_id"]
        d = self.db.get_directory(dir_id)
        if not d:
            self._emit_status(active=False, stage="error", message="Directory not found")
            return
        output_dir = Path(d["output_dir"])
        settings = self.db.get_settings()
        max_dim = int(settings.get("hq_preview_max_dim", 1600))
        quality = int(settings.get("hq_preview_quality", 86))

        res = self.db.query_images(dir_id=dir_id, subfolder=subfolder, limit=100000, offset=0)
        items = res["items"]
        if only_layers:
            items = [i for i in items if i["ai_layer"] in only_layers]

        # Skip ones that already have 2048px export or cached HQ preview
        todo = []
        for img in items:
            has_hq = False
            for key in ("export_layer2_path", "export_layer1_path", "hq_preview_rel_path"):
                rel = img.get(key)
                if rel and (output_dir / rel).exists():
                    has_hq = True
                    break
            if not has_hq:
                todo.append(img)

        total = len(todo)
        if total == 0:
            msg = "All images in this scope already have fast HQ JPG previews!"
            self._log("INFO", "preview", msg, run_id=run_id)
            self._emit_status(active=False, stage="done", percent=100.0, message=msg)
            return

        self._log("INFO", "preview", f"Generating {max_dim}px Q{quality} JPG previews for {total} images...", run_id=run_id)
        self._emit_status(total=total, message=f"Converting {total} images to fast HQ JPGs...")
        t0 = time.perf_counter()
        done = 0
        errors = 0

        def _gen_one(img: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
            src = Path(img["source_path"])
            if not src.exists():
                return img["image_id"], None, "Source missing"
            img_bgr, _ = load_preview_bgr_and_meta(src)
            if img_bgr is None:
                return img["image_id"], None, "Decode failed"
            rel = f"gallery_assets/previews/{img['image_id']}.jpg"
            save_jpeg_bgr(img_bgr, output_dir / rel, quality=quality, max_dim=max_dim)
            # Also ensure thumb exists if missing
            if not img.get("thumb_rel_path") or not (output_dir / img["thumb_rel_path"]).exists():
                trel = f"gallery_assets/thumbs/{img['image_id']}.jpg"
                save_jpeg_bgr(img_bgr, output_dir / trel, quality=80, max_dim=720)
            return img["image_id"], rel, None

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_gen_one, img): img["filename"] for img in todo}
            for fut in as_completed(futures):
                if self._cancel_event.is_set():
                    break
                fname = futures[fut]
                try:
                    iid, rel, err = fut.result()
                    if rel:
                        self.db.set_hq_preview_path(iid, rel)
                    else:
                        errors += 1
                except Exception:
                    errors += 1
                done += 1
                if done % 15 == 0 or done == total:
                    elapsed = max(0.01, time.perf_counter() - t0)
                    ips = done / elapsed
                    eta = (total - done) / max(0.1, ips)
                    self._emit_status(
                        current=done,
                        percent=round(100.0 * done / max(1, total), 1),
                        speed_ips=round(ips, 1),
                        eta_seconds=round(eta, 1),
                        current_file=fname,
                        errors=errors,
                        message=f"HQ JPGs: {done}/{total} ({ips:.1f} img/s)",
                    )

        elapsed = time.perf_counter() - t0
        msg = f"HQ JPG conversion done: {done - errors}/{total} ready in {elapsed:.1f}s (errors: {errors})"
        self._log("INFO", "preview", msg, run_id=run_id)
        self._emit_status(active=False, stage="done", current=done, percent=100.0, message=msg, errors=errors)
        self.broadcast({"type": "refresh_directory", "dir_id": dir_id})

    # ------------------------------------------------------------------
    # GPU AI Culling Pipeline Runner (full / unculled / selected)
    # ------------------------------------------------------------------
    def start_culling(
        self,
        dir_id: str,
        mode: str = "unculled",  # "full", "unculled", "selected"
        subfolder: str = "",
        selected_ids: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        with self._lock:
            if self.state["active"]:
                return False, f"Another task ({self.state['task_type']}) is already running."
            self._cancel_event.clear()
            run_id = f"cull_{uuid.uuid4().hex[:8]}"
            self.state.update(
                {
                    "active": True,
                    "task_type": "cull",
                    "run_id": run_id,
                    "dir_id": dir_id,
                    "mode": mode,
                    "stage": "initializing",
                    "message": "Preparing GPU culling pipeline...",
                    "current": 0,
                    "total": 0,
                    "percent": 0.0,
                    "speed_ips": 0.0,
                    "eta_seconds": 0.0,
                    "current_file": "",
                    "started_at": utc_now_iso(),
                    "errors": 0,
                }
            )
        t = threading.Thread(
            target=self._run_culling,
            args=(run_id, dir_id, mode, subfolder, selected_ids or []),
            daemon=True,
        )
        self._worker_thread = t
        t.start()
        return True, run_id

    def _run_culling(
        self,
        run_id: str,
        dir_id: str,
        mode: str,
        subfolder: str,
        selected_ids: List[str],
    ) -> None:
        d = self.db.get_directory(dir_id)
        if not d:
            self._emit_status(active=False, stage="error", message="Directory not found")
            return

        source_dir = Path(d["path"])
        output_dir = Path(d["output_dir"])
        is_legacy = bool(d.get("is_legacy_default"))
        t0 = time.perf_counter()

        try:
            import torch
            from src.deliverables import write_structured_data_and_embeddings
            from src.kestrel_bridge import KestrelCoreRunner, extract_crops_bgr
            from src.scene_burst_and_select import (
                assign_scenes_and_bursts,
                route_and_score_all,
                select_layers_final,
            )
            from src.siglip2_scorer import SigLIP2Scorer

            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if has_cuda:
                device_label = "CUDA GPU"
                props = torch.cuda.get_device_properties(0)
                self._log(
                    "INFO",
                    "pipeline",
                    f"AI culling device: {props.name} ({props.total_memory / (1024**3):.1f} GB VRAM)",
                    run_id=run_id,
                )
            elif has_mps:
                device_label = "Apple Silicon MPS"
                self._log("INFO", "pipeline", "AI culling device: Apple Silicon MPS", run_id=run_id)
            else:
                device_label = "CPU"
                self._log(
                    "WARNING",
                    "pipeline",
                    "No CUDA/MPS GPU detected — running AI culling on CPU (slower).",
                    run_id=run_id,
                )

            # Determine target images
            target_items: List[Tuple[Path, str, str]] = []
            if mode == "selected" and selected_ids:
                for iid in selected_ids:
                    img = self.db.get_image(iid)
                    if img:
                        target_items.append((Path(img["source_path"]), img["subfolder"], img["image_id"]))
            else:
                if source_dir.exists():
                    discovered = discover_images(source_dir)
                    if subfolder and subfolder != "__all__":
                        discovered = [(p, sf) for p, sf in discovered if sf == subfolder]
                    new_entries = []
                    for p, sf in discovered:
                        iid = make_image_id(dir_id, sf, p.name, legacy=is_legacy)
                        existing = self.db.get_image(iid)
                        if not existing:
                            new_entries.append(
                                {
                                    "image_id": iid,
                                    "source_path": str(p),
                                    "subfolder": sf,
                                    "filename": p.name,
                                    "stem": p.stem,
                                    "file_ext": p.suffix.lower(),
                                }
                            )
                        if mode == "full" or not existing or not existing.get("is_culled"):
                            target_items.append((p, sf, iid))
                    if new_entries:
                        self.db.upsert_scanned_images(dir_id, new_entries)
                else:
                    raise RuntimeError(f"Source directory is not accessible: {source_dir}")

            total = len(target_items)
            if total == 0:
                msg = "No images to cull (all images in this scope are already culled!)."
                self._log("INFO", "pipeline", msg, run_id=run_id)
                self._emit_status(active=False, stage="done", percent=100.0, message=msg)
                return

            self.db.create_pipeline_run(run_id, dir_id, mode, total)
            self._log(
                "INFO",
                "pipeline",
                f"Starting Project Kestrel + SigLIP 2 culling ({mode}) on {total} images in {source_dir.name}",
                {"mode": mode, "subfolder": subfolder, "total": total},
                run_id=run_id,
            )

            cfg = PipelineConfig(source_dir=source_dir, output_dir=output_dir, resume=(mode != "full"))
            cfg.ensure_output_dirs()
            bio_emb_dir = output_dir / "gallery_assets" / "bioclip_embs"
            bio_emb_dir.mkdir(parents=True, exist_ok=True)
            preview_dir = output_dir / "gallery_assets" / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)

            settings = self.db.get_settings()
            hq_dim = int(settings.get("hq_preview_max_dim", 1600))
            hq_q = int(settings.get("hq_preview_quality", 86))

            # Step 2A: Project Kestrel (MegaDetector + SAM-HQ + BioCLIP 2 + AnimalGate + QualityClassifier)
            self._emit_status(stage="step2a_kestrel", total=total, message=f"Loading Project Kestrel (MegaDetector + SAM-HQ + BioCLIP 2) on {device_label}...")
            kestrel = KestrelCoreRunner(cfg)
            skipped_errors = 0
            records: List[Dict[str, Any]] = []
            bio_emb_list: List[np.ndarray] = []

            max_inflight = 12
            work_q: queue.Queue = queue.Queue(maxsize=12)
            sentinel = object()

            def _load_one(item: Tuple[Path, str, str]) -> Optional[Dict[str, Any]]:
                if self._cancel_event.is_set():
                    return None
                path, sf, iid = item
                img_bgr, meta = load_preview_bgr_and_meta(path)
                if img_bgr is None:
                    return None
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                return {
                    "path": path,
                    "subfolder": sf,
                    "image_id": iid,
                    "img_bgr": img_bgr,
                    "img_rgb": img_rgb,
                    "meta": meta,
                }

            def _producer() -> None:
                nonlocal skipped_errors
                inflight: collections.deque = collections.deque()
                with ThreadPoolExecutor(max_workers=cfg.io_workers) as pool:
                    it = iter(target_items)
                    for _ in range(max_inflight):
                        try:
                            inflight.append(pool.submit(_load_one, next(it)))
                        except StopIteration:
                            break
                    while inflight:
                        if self._cancel_event.is_set():
                            break
                        fut = inflight.popleft()
                        try:
                            inflight.append(pool.submit(_load_one, next(it)))
                        except StopIteration:
                            pass
                        try:
                            res = fut.result()
                        except Exception as exc:
                            self._log("WARNING", "pipeline", f"Read error: {exc}", run_id=run_id)
                            res = None
                        if res is None:
                            if not self._cancel_event.is_set():
                                skipped_errors += 1
                            continue
                        work_q.put(res)
                work_q.put(sentinel)

            prod = threading.Thread(target=_producer, daemon=True)
            prod.start()
            save_pool = ThreadPoolExecutor(max_workers=4)

            done = 0
            while True:
                item = work_q.get()
                if item is sentinel:
                    break
                if self._cancel_event.is_set():
                    continue

                iid = item["image_id"]
                img_bgr = item["img_bgr"]
                img_rgb = item["img_rgb"]
                meta = item["meta"]
                path: Path = item["path"]
                sf: str = item["subfolder"]

                kres, bio_emb = kestrel.analyze_image(img_rgb, ascii_name=f"{iid}.jpg")
                subj_box = tuple(kres["bbox"]) if kres.get("has_wildlife") else None
                head_box = tuple(kres["head_bbox"]) if kres.get("has_wildlife") else None
                zoom_crop, tight_crop = extract_crops_bgr(
                    img_bgr, subj_box, head_box, output_size=cfg.crop_output_size
                )

                thumb_rel = f"gallery_assets/thumbs/{iid}.jpg"
                crop_rel = f"gallery_assets/crops/{iid}.jpg"
                scrop_rel = f"gallery_assets/siglip_crops/{iid}.jpg" if tight_crop is not None else ""
                prev_rel = f"gallery_assets/previews/{iid}.jpg"

                np.save(bio_emb_dir / f"{iid}.npy", bio_emb)
                bio_emb_list.append(bio_emb.astype(np.float16))

                save_pool.submit(save_jpeg_bgr, img_bgr, output_dir / thumb_rel, cfg.gallery_thumb_quality, cfg.gallery_thumb_max_dim)
                save_pool.submit(save_jpeg_bgr, zoom_crop, output_dir / crop_rel, cfg.crop_quality, None)
                save_pool.submit(save_jpeg_bgr, img_bgr, output_dir / prev_rel, hq_q, hq_dim)
                if tight_crop is not None:
                    save_pool.submit(save_jpeg_bgr, tight_crop, output_dir / scrop_rel, cfg.crop_quality, None)

                rec: Dict[str, Any] = {
                    "image_id": iid,
                    "source_path": str(path),
                    "subfolder": sf,
                    "filename": path.name,
                    "stem": path.stem,
                    "timestamp": float(meta["timestamp"]),
                    "timestamp_str": meta["timestamp_str"],
                    "focal_length": meta.get("focal_length"),
                    "f_number": meta.get("f_number"),
                    "exposure_time": meta.get("exposure_time", ""),
                    "exposure_seconds": meta.get("exposure_seconds"),
                    "iso": meta.get("iso"),
                    "camera_model": meta.get("camera_model", ""),
                    "lens_model": meta.get("lens_model", ""),
                    "preview_width": meta.get("preview_width", 0),
                    "preview_height": meta.get("preview_height", 0),
                    "thumb_rel_path": thumb_rel,
                    "crop_rel_path": crop_rel,
                    "hq_preview_rel_path": prev_rel,
                    "_siglip_crop_rel_path": scrop_rel,
                }
                rec.update(kres)
                rec["detected_class"] = rec.get("species") or rec.get("animal_group") or ""
                rec["detection_conf"] = rec.get("species_conf") or rec.get("md_conf") or 0.0
                records.append(rec)

                item["img_bgr"] = None
                item["img_rgb"] = None
                done += 1
                if done % 8 == 0 or done == total:
                    elapsed = max(0.01, time.perf_counter() - t0)
                    ips = done / elapsed
                    eta = (total - done) / max(0.1, ips)
                    self._emit_status(
                        current=done,
                        percent=round(68.0 * done / max(1, total), 1),
                        speed_ips=round(ips, 1),
                        eta_seconds=round(eta, 1),
                        current_file=path.name,
                        errors=skipped_errors,
                        message=f"Step 2A Kestrel (MD+SAM+BioCLIP2): {done}/{total} ({ips:.1f} img/s)",
                    )
                if done % 200 == 0:
                    gc.collect()

            save_pool.shutdown(wait=True)
            prod.join()
            del kestrel
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self._cancel_event.is_set():
                self.db.finish_pipeline_run(run_id, "cancelled", done, skipped_errors, {"cancelled": True})
                self._log("WARNING", "pipeline", f"Culling run cancelled after {done}/{total} images.", run_id=run_id)
                self._emit_status(active=False, stage="cancelled", message="Cancelled by user.")
                return

            if not records:
                raise RuntimeError("No images could be decoded successfully.")

            bio_embs = np.stack(bio_emb_list).astype(np.float16)

            # Step 2B: SigLIP 2 (CUDA FP16 / Apple Silicon MPS / CPU fallback)
            self._emit_status(stage="step2b_siglip2", percent=72.0, message=f"Running SigLIP 2 scoring on {device_label}...")
            siglip2 = SigLIP2Scorer(cfg)
            siglip_embs = siglip2.score_all_images(records, output_dir)
            siglip2.close()

            # Step 2A.4 + 2C + 2D: Scenario/Zoom Grouping, Two-Track Routing & Selection
            self._emit_status(stage="select", percent=88.0, message="Grouping scenarios (>=10% zoom split), routing & selecting...")
            records = assign_scenes_and_bursts(records, siglip_embs, cfg)
            records = route_and_score_all(records, cfg, bioclip_embs=bio_embs, siglip_embs=siglip_embs)
            records = select_layers_final(records, cfg, siglip_embs=siglip_embs)

            # Save/merge embeddings.npz & results.json on every culling run so
            # Fast Reselect & Personal Ranker work right after "Run Unculled Only" too!
            self._emit_status(stage="saving", percent=93.0, message="Saving results.json & embeddings.npz...")
            is_full_dir = mode == "full" and (not subfolder or subfolder == "__all__")
            res_path = output_dir / "results.json"
            npz_path = output_dir / "embeddings.npz"
            if is_full_dir or not (res_path.is_file() and npz_path.is_file()):
                write_structured_data_and_embeddings(records, bio_embs, siglip_embs, cfg)
            else:
                try:
                    prev_records = json.loads(res_path.read_text(encoding="utf-8"))
                    prev_npz = np.load(npz_path)
                    prev_ids = [str(x) for x in prev_npz["image_ids"]]
                    prev_bio = prev_npz["bioclip_embs"].astype(np.float16)
                    prev_sig = prev_npz["siglip_embs"].astype(np.float16)
                    prev_idx_by_id = {iid: idx for idx, iid in enumerate(prev_ids)}

                    new_by_id = {r["image_id"]: r for r in records}
                    merged_records: List[Dict[str, Any]] = []
                    merged_bio_list: List[np.ndarray] = []
                    merged_sig_list: List[np.ndarray] = []

                    # Keep previously cached records not replaced in this run
                    for pr in prev_records:
                        pid = pr.get("image_id")
                        if not pid or pid in new_by_id:
                            continue
                        pidx = prev_idx_by_id.get(pid)
                        if pidx is None:
                            continue
                        pr_copy = dict(pr)
                        pr_copy["_emb_idx"] = len(merged_records)
                        merged_records.append(pr_copy)
                        merged_bio_list.append(prev_bio[pidx])
                        merged_sig_list.append(prev_sig[pidx])

                    # Append newly processed records
                    for nr in records:
                        nidx = int(nr["_emb_idx"])
                        nr_copy = dict(nr)
                        nr_copy["_emb_idx"] = len(merged_records)
                        merged_records.append(nr_copy)
                        merged_bio_list.append(bio_embs[nidx])
                        merged_sig_list.append(siglip_embs[nidx])

                    if merged_records:
                        merged_bio = np.stack(merged_bio_list).astype(np.float16)
                        merged_sig = np.stack(merged_sig_list).astype(np.float16)
                        write_structured_data_and_embeddings(merged_records, merged_bio, merged_sig, cfg)
                    else:
                        write_structured_data_and_embeddings(records, bio_embs, siglip_embs, cfg)
                except Exception as merge_exc:
                    logger.warning("Failed to merge cached embeddings; writing current run only: %s", merge_exc)
                    write_structured_data_and_embeddings(records, bio_embs, siglip_embs, cfg)

            # Save to SQLite (preserving user ratings & XMPs strictly separately!)
            self._emit_status(stage="saving", percent=96.0, message="Saving Project Kestrel results to database...")
            self.db.update_ai_results(dir_id, run_id, records)
            for rec in records:
                if rec.get("hq_preview_rel_path"):
                    self.db.set_hq_preview_path(rec["image_id"], rec["hq_preview_rel_path"])

            if settings.get("ai_xmp_auto_apply"):
                self._emit_status(stage="xmp_sync", percent=98.0, message="Applying AI mapping to XMP sidecars...")
                self.apply_ai_mapping_to_xmp(dir_id, image_ids=[r["image_id"] for r in records])

            elapsed = time.perf_counter() - t0
            l2 = sum(1 for r in records if r["layer"] == "layer2_to_edit")
            l1 = sum(1 for r in records if r["layer"] == "layer1_keep")
            rej = sum(1 for r in records if r["layer"] == "rejected")
            w_cnt = sum(1 for r in records if r.get("track") == "Wildlife")
            l_cnt = sum(1 for r in records if r.get("track") == "Landscape")
            summary = {
                "total_processed": len(records),
                "skipped_errors": skipped_errors,
                "wildlife_count": w_cnt,
                "landscape_count": l_cnt,
                "layer2_count": l2,
                "layer1_only_count": l1,
                "rejected_count": rej,
                "runtime_seconds": round(elapsed, 2),
            }
            self.db.finish_pipeline_run(run_id, "completed", len(records), skipped_errors, summary)
            msg = f"Kestrel + SigLIP 2 complete: {len(records)} processed ({l2} Edit, {l1} Keep, {rej} Reject) in {elapsed:.1f}s"
            self._log("INFO", "pipeline", msg, summary, run_id=run_id)
            self._emit_status(active=False, stage="done", current=total, percent=100.0, message=msg)
            self.broadcast({"type": "refresh_directory", "dir_id": dir_id})

        except Exception as exc:
            logger.exception("GPU culling run failed")
            self.db.finish_pipeline_run(run_id, "failed", 0, 0, {"error": str(exc)})
            self._log("ERROR", "pipeline", f"Culling failed: {exc}", run_id=run_id)
            self._emit_status(active=False, stage="error", message=f"Culling failed: {exc}")

    # ------------------------------------------------------------------
    # Fast Reselect (from cached results.json + embeddings.npz + personal_ranker.pkl)
    # ------------------------------------------------------------------
    def start_reselect(self, dir_id: str) -> Tuple[bool, str]:
        with self._lock:
            if self.state["active"]:
                return False, f"Another task ({self.state['task_type']}) is already running."
            self._cancel_event.clear()
            run_id = f"resel_{uuid.uuid4().hex[:8]}"
            self.state.update(
                {
                    "active": True,
                    "task_type": "reselect",
                    "run_id": run_id,
                    "dir_id": dir_id,
                    "mode": "reselect",
                    "stage": "reselecting",
                    "message": "Re-running scenario grouping, routing & personal ranker...",
                    "current": 0,
                    "total": 0,
                    "percent": 15.0,
                    "speed_ips": 0.0,
                    "eta_seconds": 0.0,
                    "current_file": "",
                    "started_at": utc_now_iso(),
                    "errors": 0,
                }
            )
        t = threading.Thread(target=self._run_reselect, args=(run_id, dir_id), daemon=True)
        self._worker_thread = t
        t.start()
        return True, run_id

    def _run_reselect(self, run_id: str, dir_id: str) -> None:
        d = self.db.get_directory(dir_id)
        if not d:
            self._emit_status(active=False, stage="error", message="Directory not found")
            return
        output_dir = Path(d["output_dir"])
        source_dir = Path(d["path"])
        t0 = time.perf_counter()
        try:
            from src.deliverables import write_structured_data_and_embeddings
            from src.scene_burst_and_select import (
                assign_scenes_and_bursts,
                route_and_score_all,
                select_layers_final,
            )

            res_path = output_dir / "results.json"
            npz_path = output_dir / "embeddings.npz"
            if not res_path.is_file() or not npz_path.is_file():
                raise FileNotFoundError(
                    f"Fast reselect requires results.json and embeddings.npz in {output_dir}"
                )

            self._emit_status(percent=25.0, message="Loading cached scores & embeddings.npz...")
            records = json.loads(res_path.read_text(encoding="utf-8"))
            npz = np.load(npz_path)
            bio_embs = npz["bioclip_embs"]
            siglip_embs = npz["siglip_embs"]

            cfg = PipelineConfig(source_dir=source_dir, output_dir=output_dir)
            self.db.create_pipeline_run(run_id, dir_id, "reselect", len(records))

            self._emit_status(percent=50.0, message="Grouping scenarios (>=10% zoom split) & routing tracks...")
            records = assign_scenes_and_bursts(records, siglip_embs, cfg)
            records = route_and_score_all(records, cfg, bioclip_embs=bio_embs, siglip_embs=siglip_embs)
            records = select_layers_final(records, cfg, siglip_embs=siglip_embs)

            self._emit_status(percent=80.0, message="Saving updated selection & refreshing database...")
            write_structured_data_and_embeddings(records, bio_embs, siglip_embs, cfg)
            self.db.import_existing_results_json(res_path, output_dir, force=True)

            elapsed = time.perf_counter() - t0
            l2 = sum(1 for r in records if r["layer"] == "layer2_to_edit")
            l1 = sum(1 for r in records if r["layer"] == "layer1_keep")
            has_pers = any(r.get("personal_score") is not None for r in records)
            summary = {
                "total_processed": len(records),
                "layer2_count": l2,
                "layer1_only_count": l1,
                "personal_ranker_active": has_pers,
                "runtime_seconds": round(elapsed, 2),
            }
            self.db.finish_pipeline_run(run_id, "completed", len(records), 0, summary)
            msg = (
                f"Fast reselect done in {elapsed:.1f}s: {l2} Edit, {l1} Keep"
                + (" (Personal Ranker applied!)" if has_pers else "")
            )
            self._log("INFO", "pipeline", msg, summary, run_id=run_id)
            self._emit_status(active=False, stage="done", current=len(records), total=len(records), percent=100.0, message=msg)
            self.broadcast({"type": "refresh_directory", "dir_id": dir_id})
        except Exception as exc:
            logger.exception("Fast reselect failed")
            self.db.finish_pipeline_run(run_id, "failed", 0, 0, {"error": str(exc)})
            self._log("ERROR", "pipeline", f"Fast reselect failed: {exc}", run_id=run_id)
            self._emit_status(active=False, stage="error", message=f"Reselect failed: {exc}")

    # ------------------------------------------------------------------
    # Train Personalized LightGBM Ranker from User Ratings/Overrides
    # ------------------------------------------------------------------
    def train_personal_ranker(self, dir_id: str) -> Dict[str, Any]:
        d = self.db.get_directory(dir_id)
        if not d:
            raise ValueError("Directory not found")
        output_dir = Path(d["output_dir"])
        res_path = output_dir / "results.json"
        npz_path = output_dir / "embeddings.npz"
        if not res_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(
                f"Training requires results.json and embeddings.npz in {output_dir}"
            )

        picks = self.db.get_training_picks(dir_id)
        n_keep = sum(1 for p in picks if p["user_decision"] in ("layer2_edit", "layer1_keep"))
        n_rej = sum(1 for p in picks if p["user_decision"] == "reject")
        if n_keep < 3 or n_rej < 3:
            raise ValueError(
                f"Need at least 3 Keep/Edit (4-5★/2-3★/Green/Yellow) and 3 Reject (1★/Red/X) picks to train "
                f"(currently {n_keep} Keep/Edit, {n_rej} Reject)."
            )

        workspace_root = Path(__file__).resolve().parent.parent
        picks_path = workspace_root / "studio_data" / "my_picks.json"
        picks_path.parent.mkdir(parents=True, exist_ok=True)
        manual_splits = self.db.list_burst_splits(dir_id)
        picks_path.write_text(
            json.dumps({"picks": picks, "manual_burst_splits": manual_splits}, indent=2),
            encoding="utf-8",
        )

        model_out = workspace_root / "models" / "personal_ranker.pkl"
        history_jsonl = workspace_root / "models" / "feedback_history.jsonl"

        from train_ranker import train_from_picks

        train_from_picks(
            picks_path=picks_path,
            results_json=res_path,
            embeddings_npz=npz_path,
            model_out=model_out,
            history_jsonl=history_jsonl,
        )
        msg = f"Trained LightGBM personal ranker on {len(picks)} picks ({n_keep} Keep/Edit, {n_rej} Reject) -> {model_out.name}"
        self._log("INFO", "pipeline", msg, {"picks": len(picks), "keep": n_keep, "reject": n_rej})
        return {
            "ok": True,
            "picks_exported": len(picks),
            "keep_edit_count": n_keep,
            "reject_count": n_rej,
            "model_path": str(model_out),
            "message": msg,
        }

    # ------------------------------------------------------------------
    # Apply AI Culling Mapping -> XMP Sidecars (Separate & Configurable!)
    # ------------------------------------------------------------------
    def apply_ai_mapping_to_xmp(
        self,
        dir_id: str,
        subfolder: str = "",
        image_ids: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        settings = self.db.get_settings()
        mapping = settings.get("ai_xmp_mapping", {})
        only_unrated = bool(settings.get("ai_xmp_only_if_unrated", True))

        if image_ids:
            images = [self.db.get_image(iid) for iid in image_ids]
            images = [i for i in images if i is not None and i["is_culled"]]
        else:
            res = self.db.query_images(dir_id=dir_id, subfolder=subfolder, culled_filter="culled", limit=100000, offset=0)
            images = res["items"]

        batch_id = f"ai_sync_{uuid.uuid4().hex[:8]}"
        would_update = 0
        skipped_rated = 0
        synced = 0
        failed = 0

        for img in images:
            layer = img["ai_layer"]
            rule = mapping.get(layer)
            if not rule or not rule.get("enabled"):
                continue
            if only_unrated and (img["effective_rating"] > 0 or img["effective_label"]):
                skipped_rated += 1
                continue

            target_rating = rule.get("rating")
            target_label = normalize_label(rule.get("label", ""))
            would_update += 1

            if dry_run:
                continue

            ok, prev_xml, _, err = write_xmp_sidecar(
                Path(img["source_path"]),
                rating=target_rating,
                label=target_label,
            )
            if ok:
                self.db.record_metadata_change(
                    image_id=img["image_id"],
                    rating=target_rating,
                    label=target_label,
                    source="ai_sync",
                    action_type="ai_xmp_sync",
                    xmp_synced=True,
                    prev_xmp_xml=prev_xml,
                    batch_id=batch_id,
                )
                synced += 1
            else:
                failed += 1

        if not dry_run:
            self._log(
                "INFO",
                "xmp",
                f"Applied AI culling mapping to XMP: {synced} updated, {skipped_rated} skipped (already rated), {failed} failed (batch {batch_id})",
                {"batch_id": batch_id, "synced": synced, "skipped": skipped_rated, "failed": failed},
            )
        return {
            "dry_run": dry_run,
            "batch_id": "" if dry_run else batch_id,
            "would_update": would_update,
            "synced": synced,
            "skipped_rated": skipped_rated,
            "failed": failed,
        }
