"""Persistent SQLite database for ImageCulling Studio."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.io_utils import parse_exposure_seconds
from webapp.xmp_sync import normalize_label

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS: Dict[str, Any] = {
    "realtime_xmp_write": True,
    "ai_xmp_auto_apply": False,
    "ai_xmp_only_if_unrated": True,
    "ai_xmp_mapping": {
        "layer2_to_edit": {"enabled": True, "rating": None, "label": "Green"},
        "layer1_keep": {"enabled": True, "rating": None, "label": "Yellow"},
        "rejected": {"enabled": False, "rating": None, "label": ""},
        "to_delete": {"enabled": True, "rating": None, "label": "Red"},
    },
    "auto_advance_on_rate": False,
    "hq_preview_max_dim": 1600,
    "hq_preview_quality": 86,
    "zoom_preview_max_dim": 8192,
    "zoom_preview_quality": 94,
    "show_bbox_overlay": False,
    "grid_card_size": 240,
}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def make_dir_id(path_str: str) -> str:
    norm = str(Path(path_str)).strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def make_image_id(dir_id: str, subfolder: str, filename: str, legacy: bool = False) -> str:
    safe_sf = "".join(c if c.isalnum() or c in "-_" else "_" for c in subfolder)
    safe_fn = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)
    if legacy:
        return f"{safe_sf}__{safe_fn}"
    return f"{dir_id}__{safe_sf}__{safe_fn}"


class CullDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS directories (
                    dir_id TEXT PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    is_legacy_default INTEGER DEFAULT 0,
                    last_opened_at TEXT,
                    last_culled_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    dir_id TEXT NOT NULL,
                    source_path TEXT UNIQUE NOT NULL,
                    subfolder TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    stem TEXT NOT NULL,
                    file_ext TEXT DEFAULT '',
                    file_size INTEGER DEFAULT 0,

                    -- EXIF
                    timestamp REAL DEFAULT 0,
                    timestamp_str TEXT DEFAULT '',
                    focal_length REAL,
                    f_number REAL,
                    exposure_time TEXT DEFAULT '',
                    exposure_seconds REAL,
                    iso INTEGER,
                    camera_model TEXT DEFAULT '',
                    lens_model TEXT DEFAULT '',
                    preview_width INTEGER DEFAULT 0,
                    preview_height INTEGER DEFAULT 0,

                    -- AI Culling (kept strictly separate from user/XMP ratings)
                    is_culled INTEGER DEFAULT 0,
                    culled_at TEXT,
                    cull_run_id TEXT,
                    ai_layer TEXT DEFAULT 'unculled',
                    burst_id TEXT DEFAULT '',
                    burst_size INTEGER DEFAULT 1,
                    burst_index INTEGER DEFAULT 1,
                    burst_rank INTEGER DEFAULT 1,
                    bbox_json TEXT DEFAULT '[]',
                    head_bbox_json TEXT DEFAULT '[]',
                    af_points_json TEXT DEFAULT 'null',
                    detected_class TEXT DEFAULT '',
                    detection_conf REAL DEFAULT 0,
                    detection_source TEXT DEFAULT '',
                    subject_area_frac REAL DEFAULT 0,
                    num_animals INTEGER DEFAULT 0,
                    sharpness_raw REAL DEFAULT 0,
                    sharpness_score REAL DEFAULT 0,
                    head_tenengrad REAL DEFAULT 0,
                    head_laplacian REAL DEFAULT 0,
                    head_luma REAL DEFAULT 0,
                    layer1_priority REAL DEFAULT 0,
                    eye_contact_score REAL DEFAULT 0,
                    action_score REAL DEFAULT 0,
                    scenic_score REAL DEFAULT 0,
                    personal_score REAL,
                    overall_score REAL DEFAULT 0,
                    overall_rank INTEGER DEFAULT 999999,
                    no_bird_prob REAL DEFAULT 0,
                    track TEXT DEFAULT '',
                    species TEXT DEFAULT '',
                    species_conf REAL DEFAULT 0,
                    scenario_type TEXT DEFAULT '',
                    hybrid_bonus INTEGER DEFAULT 0,
                    tonal_balance REAL DEFAULT 0,
                    kestrel_quality REAL DEFAULT 0,

                    -- Previews & Exports (relative to directory output_dir)
                    thumb_rel_path TEXT DEFAULT '',
                    crop_rel_path TEXT DEFAULT '',
                    hq_preview_rel_path TEXT DEFAULT '',
                    export_layer1_path TEXT DEFAULT '',
                    export_layer2_path TEXT DEFAULT '',

                    -- Camera & XMP / User Ratings & Color Labels
                    camera_rating INTEGER DEFAULT 0,
                    camera_label TEXT DEFAULT '',
                    xmp_exists INTEGER DEFAULT 0,
                    xmp_rating INTEGER,
                    xmp_label TEXT,
                    effective_rating INTEGER DEFAULT 0,
                    effective_label TEXT DEFAULT '',
                    user_flag INTEGER DEFAULT 0,
                    rating_source TEXT DEFAULT 'none',
                    user_modified_at TEXT,

                    FOREIGN KEY(dir_id) REFERENCES directories(dir_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_images_dir_sub ON images(dir_id, subfolder);
                CREATE INDEX IF NOT EXISTS idx_images_dir_layer ON images(dir_id, ai_layer);
                CREATE INDEX IF NOT EXISTS idx_images_dir_rating ON images(dir_id, effective_rating);
                CREATE INDEX IF NOT EXISTS idx_images_dir_label ON images(dir_id, effective_label);
                CREATE INDEX IF NOT EXISTS idx_images_dir_burst ON images(dir_id, burst_id);
                CREATE INDEX IF NOT EXISTS idx_images_dir_culled ON images(dir_id, is_culled);
                CREATE INDEX IF NOT EXISTS idx_images_dir_rank ON images(dir_id, overall_rank);
                CREATE INDEX IF NOT EXISTS idx_images_dir_ts ON images(dir_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_images_dir_exp ON images(dir_id, exposure_seconds);

                CREATE TABLE IF NOT EXISTS change_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dir_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    field TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    prev_xmp_xml TEXT,
                    xmp_synced INTEGER DEFAULT 0,
                    reverted INTEGER DEFAULT 0,
                    batch_id TEXT DEFAULT '',
                    ai_layer_at_time TEXT DEFAULT '',
                    ai_score_at_time REAL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_changelog_ts ON change_log(id DESC);
                CREATE INDEX IF NOT EXISTS idx_changelog_batch ON change_log(batch_id);

                CREATE TABLE IF NOT EXISTS event_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT DEFAULT '{}',
                    run_id TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_eventlogs_id ON event_logs(id DESC);

                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    dir_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    total_images INTEGER DEFAULT 0,
                    processed_images INTEGER DEFAULT 0,
                    skipped_errors INTEGER DEFAULT 0,
                    summary_json TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS burst_splits (
                    image_id TEXT PRIMARY KEY,
                    dir_id TEXT NOT NULL,
                    orig_burst_id TEXT NOT NULL,
                    new_burst_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_burst_splits_dir ON burst_splits(dir_id);
                """
            )
            # Safe schema migration for existing DBs
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
            migrations = [
                ("af_points_json", "TEXT DEFAULT 'null'"),
                ("scenic_score", "REAL DEFAULT 0"),
                ("personal_score", "REAL"),
                ("track", "TEXT DEFAULT ''"),
                ("species", "TEXT DEFAULT ''"),
                ("species_conf", "REAL DEFAULT 0"),
                ("scenario_type", "TEXT DEFAULT ''"),
                ("hybrid_bonus", "INTEGER DEFAULT 0"),
                ("tonal_balance", "REAL DEFAULT 0"),
                ("kestrel_quality", "REAL DEFAULT 0"),
            ]
            for col_name, col_def in migrations:
                if col_name not in cols:
                    conn.execute(f"ALTER TABLE images ADD COLUMN {col_name} {col_def}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_dir_track ON images(dir_id, track)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_dir_species ON images(dir_id, species)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_images_dir_scenario ON images(dir_id, scenario_type)")

    # ------------------------------------------------------------------
    # Settings & Session Resume
    # ------------------------------------------------------------------
    def get_settings(self) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT key, value_json FROM settings").fetchall()
        merged = json.loads(json.dumps(DEFAULT_SETTINGS))
        for r in rows:
            if r["key"] == "last_session_state":
                continue
            try:
                merged[r["key"]] = json.loads(r["value_json"])
            except Exception:
                pass
        return merged

    def update_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            for k, v in updates.items():
                if k == "last_session_state":
                    continue
                conn.execute(
                    "INSERT INTO settings(key, value_json) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                    (k, json.dumps(v, ensure_ascii=False)),
                )
        return self.get_settings()

    def get_session_state(self) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM settings WHERE key = 'last_session_state'"
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["value_json"])
        except Exception:
            return {}

    def save_session_state(self, state: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value_json) VALUES ('last_session_state', ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (json.dumps(state, ensure_ascii=False),),
            )

    # ------------------------------------------------------------------
    # Event Logging
    # ------------------------------------------------------------------
    def add_event_log(
        self,
        level: str,
        category: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        run_id: str = "",
    ) -> Dict[str, Any]:
        ts = utc_now_iso()
        det_str = json.dumps(details or {}, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO event_logs(timestamp, level, category, message, details_json, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, level.upper(), category, message, det_str, run_id),
            )
            log_id = cur.lastrowid
            # Keep event_logs bounded so frequent perf telemetry never grows SQLite indefinitely
            if log_id and log_id % 100 == 0:
                conn.execute(
                    """
                    DELETE FROM event_logs
                    WHERE category = 'perf'
                      AND id NOT IN (
                          SELECT id FROM event_logs WHERE category = 'perf' ORDER BY id DESC LIMIT 500
                      )
                    """
                )
                conn.execute(
                    """
                    DELETE FROM event_logs
                    WHERE id NOT IN (
                        SELECT id FROM event_logs ORDER BY id DESC LIMIT 5000
                    )
                    """
                )
        return {
            "id": log_id,
            "timestamp": ts,
            "level": level.upper(),
            "category": category,
            "message": message,
            "details": details or {},
            "run_id": run_id,
        }

    def get_event_logs(
        self,
        limit: int = 200,
        level: str = "",
        category: str = "",
        run_id: str = "",
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if level:
            clauses.append("level = ?")
            params.append(level.upper())
        if category:
            clauses.append("category = ?")
            params.append(category)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM event_logs {where} ORDER BY id DESC LIMIT ?", params
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["details"] = json.loads(d.pop("details_json") or "{}")
            except Exception:
                d["details"] = {}
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Directories & Subfolders
    # ------------------------------------------------------------------
    def register_directory(
        self,
        path: Path,
        output_dir: Path,
        is_legacy_default: bool = False,
    ) -> Dict[str, Any]:
        path_str = str(path)
        dir_id = make_dir_id(path_str)
        name = path.name or path_str
        now = utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO directories(dir_id, path, name, output_dir, is_legacy_default, last_opened_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dir_id) DO UPDATE SET
                    last_opened_at = excluded.last_opened_at,
                    output_dir = excluded.output_dir
                """,
                (dir_id, path_str, name, str(output_dir), 1 if is_legacy_default else 0, now, now),
            )
        return self.get_directory(dir_id)  # type: ignore[return-value]

    def get_directory(self, dir_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM directories WHERE dir_id = ?", (dir_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN is_culled = 1 THEN 1 ELSE 0 END) AS culled,
                    SUM(CASE WHEN is_culled = 0 THEN 1 ELSE 0 END) AS unculled,
                    SUM(CASE WHEN ai_layer = 'layer2_to_edit' THEN 1 ELSE 0 END) AS layer2,
                    SUM(CASE WHEN ai_layer = 'layer1_keep' THEN 1 ELSE 0 END) AS layer1,
                    SUM(CASE WHEN ai_layer = 'rejected' THEN 1 ELSE 0 END) AS rejected,
                    SUM(CASE WHEN ai_layer = 'to_delete' THEN 1 ELSE 0 END) AS to_delete,
                    SUM(CASE WHEN effective_label = 'Green' THEN 1 ELSE 0 END) AS color_layer2,
                    SUM(CASE WHEN effective_label = 'Yellow' THEN 1 ELSE 0 END) AS color_layer1,
                    SUM(CASE WHEN is_culled = 1 AND effective_label NOT IN ('Green', 'Yellow', 'Red') THEN 1 ELSE 0 END) AS color_rejected,
                    SUM(CASE WHEN effective_label = 'Red' THEN 1 ELSE 0 END) AS color_to_delete,
                    SUM(CASE WHEN effective_rating > 0 THEN 1 ELSE 0 END) AS rated,
                    SUM(CASE WHEN effective_label != '' THEN 1 ELSE 0 END) AS labeled
                FROM images WHERE dir_id = ?
                """,
                (dir_id,),
            ).fetchone()
            d["stats"] = {k: int(counts[k] or 0) for k in counts.keys()} if counts else {}
            sub_rows = conn.execute(
                """
                SELECT
                    subfolder,
                    COUNT(*) AS total,
                    SUM(CASE WHEN is_culled = 1 THEN 1 ELSE 0 END) AS culled,
                    SUM(CASE WHEN is_culled = 0 THEN 1 ELSE 0 END) AS unculled,
                    SUM(CASE WHEN ai_layer = 'layer2_to_edit' THEN 1 ELSE 0 END) AS layer2,
                    SUM(CASE WHEN ai_layer = 'layer1_keep' THEN 1 ELSE 0 END) AS layer1,
                    SUM(CASE WHEN ai_layer = 'rejected' THEN 1 ELSE 0 END) AS rejected,
                    SUM(CASE WHEN ai_layer = 'to_delete' THEN 1 ELSE 0 END) AS to_delete,
                    SUM(CASE WHEN effective_label = 'Green' THEN 1 ELSE 0 END) AS color_layer2,
                    SUM(CASE WHEN effective_label = 'Yellow' THEN 1 ELSE 0 END) AS color_layer1,
                    SUM(CASE WHEN is_culled = 1 AND effective_label NOT IN ('Green', 'Yellow', 'Red') THEN 1 ELSE 0 END) AS color_rejected,
                    SUM(CASE WHEN effective_label = 'Red' THEN 1 ELSE 0 END) AS color_to_delete,
                    SUM(CASE WHEN effective_rating > 0 THEN 1 ELSE 0 END) AS rated
                FROM images
                WHERE dir_id = ?
                GROUP BY subfolder
                ORDER BY subfolder ASC
                """,
                (dir_id,),
            ).fetchall()
            d["subfolders"] = [
                {k: (int(r[k] or 0) if k != "subfolder" else r[k]) for k in r.keys()}
                for r in sub_rows
            ]
        return d

    def list_directories(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT dir_id FROM directories ORDER BY last_opened_at DESC, created_at DESC"
            ).fetchall()
        out = []
        for r in rows:
            d = self.get_directory(r["dir_id"])
            if d:
                out.append(d)
        return out

    # ------------------------------------------------------------------
    # Import Previous Session (cull_output/results.json)
    # ------------------------------------------------------------------
    def import_existing_results_json(
        self, results_json_path: Path, output_dir: Path, force: bool = False
    ) -> int:
        """Import or refresh CLI run results.json into SQLite whenever results.json is new or updated."""
        if not results_json_path.exists():
            return 0
        file_mtime = float(results_json_path.stat().st_mtime)
        if not force:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT value_json FROM settings WHERE key = 'last_results_json_mtime'"
                ).fetchone()
                prev_mtime = float(json.loads(row["value_json"])) if row else 0.0
                total_existing = conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()["c"]
                has_tracks = conn.execute(
                    "SELECT COUNT(*) AS c FROM images WHERE track != ''"
                ).fetchone()["c"]
                if total_existing > 0 and has_tracks > 0 and abs(file_mtime - prev_mtime) < 1.0:
                    return 0

        try:
            with open(results_json_path, "r", encoding="utf-8") as fh:
                records = json.load(fh)
        except Exception as exc:
            logger.error("Failed to read %s: %s", results_json_path, exc)
            return 0
        if not records:
            return 0

        # Infer source root from first record
        first_src = Path(records[0]["source_path"])
        first_sub = records[0].get("subfolder", "root")
        if first_sub != "root" and first_src.parent.name == first_sub:
            source_root = first_src.parent.parent
        else:
            source_root = first_src.parent

        dir_info = self.register_directory(source_root, output_dir.resolve(), is_legacy_default=True)
        dir_id = dir_info["dir_id"]

        with self._lock, self._connect() as conn:
            now = utc_now_iso()
            rows_to_insert = []
            for rec in records:
                exp_str = rec.get("exposure_time", "") or ""
                exp_sec = parse_exposure_seconds(exp_str)
                src_p = rec["source_path"]
                ext = Path(src_p).suffix.lower()
                det_cls = rec.get("species") or rec.get("detected_class") or rec.get("animal_group") or ""
                det_conf = rec.get("species_conf") or rec.get("detection_conf") or rec.get("md_conf") or 0.0
                det_src = rec.get("track") or rec.get("detection_source") or ""
                num_anim = rec.get("num_verified_animals") if rec.get("num_verified_animals") is not None else rec.get("num_animals", 0)
                sharp_raw = rec.get("head_sharp_raw") if rec.get("head_sharp_raw") is not None else rec.get("sharpness_raw", 0.0)
                rows_to_insert.append(
                    (
                        rec["image_id"],
                        dir_id,
                        src_p,
                        rec.get("subfolder", "root"),
                        rec["filename"],
                        rec.get("stem", Path(rec["filename"]).stem),
                        ext,
                        0,
                        float(rec.get("timestamp") or 0.0),
                        rec.get("timestamp_str", "") or "",
                        rec.get("focal_length"),
                        rec.get("f_number"),
                        exp_str,
                        exp_sec,
                        rec.get("iso"),
                        rec.get("camera_model", "") or "",
                        rec.get("lens_model", "") or "",
                        int(rec.get("preview_width") or 0),
                        int(rec.get("preview_height") or 0),
                        1,
                        now,
                        "cli_session",
                        rec.get("layer", "rejected"),
                        rec.get("burst_id", "") or "",
                        int(rec.get("burst_size") or 1),
                        int(rec.get("burst_index") or 1),
                        int(rec.get("burst_rank") or 1),
                        json.dumps(rec.get("bbox") or []),
                        json.dumps(rec.get("head_bbox") or []),
                        det_cls,
                        float(det_conf or 0.0),
                        det_src,
                        float(rec.get("subject_area_frac") or 0.0),
                        int(num_anim or 0),
                        float(sharp_raw or 0.0),
                        float(rec.get("sharpness_score") or 0.0),
                        float(rec.get("head_tenengrad") or 0.0),
                        float(rec.get("head_laplacian") or 0.0),
                        float(rec.get("head_luma") or 0.0),
                        float(rec.get("layer1_priority") or 0.0),
                        float(rec.get("eye_contact_score") or 0.0),
                        float(rec.get("action_score") or 0.0),
                        float(rec.get("scenic_score") or 0.0),
                        float(rec["personal_score"]) if rec.get("personal_score") is not None else None,
                        float(rec.get("overall_score") or 0.0),
                        int(rec.get("overall_rank") or 999999),
                        float(rec.get("no_bird_prob") or 0.0),
                        str(rec.get("track") or ""),
                        str(rec.get("species") or ""),
                        float(rec.get("species_conf") or 0.0),
                        str(rec.get("scenario_type") or ""),
                        1 if rec.get("hybrid_bonus") else 0,
                        float(rec.get("tonal_balance") or 0.0),
                        float(rec.get("kestrel_quality") or 0.0),
                        rec.get("thumb_rel_path", "") or "",
                        rec.get("crop_rel_path", "") or "",
                        "",
                        rec.get("export_layer1_path", "") or "",
                        rec.get("export_layer2_path", "") or "",
                    )
                )

            conn.executemany(
                """
                INSERT INTO images (
                    image_id, dir_id, source_path, subfolder, filename, stem, file_ext, file_size,
                    timestamp, timestamp_str, focal_length, f_number, exposure_time, exposure_seconds,
                    iso, camera_model, lens_model, preview_width, preview_height,
                    is_culled, culled_at, cull_run_id, ai_layer, burst_id, burst_size, burst_index, burst_rank,
                    bbox_json, head_bbox_json, detected_class, detection_conf, detection_source,
                    subject_area_frac, num_animals, sharpness_raw, sharpness_score,
                    head_tenengrad, head_laplacian, head_luma, layer1_priority,
                    eye_contact_score, action_score, scenic_score, personal_score,
                    overall_score, overall_rank, no_bird_prob,
                    track, species, species_conf, scenario_type, hybrid_bonus,
                    tonal_balance, kestrel_quality,
                    thumb_rel_path, crop_rel_path, hq_preview_rel_path, export_layer1_path, export_layer2_path
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                ON CONFLICT(image_id) DO UPDATE SET
                    is_culled = 1,
                    ai_layer = excluded.ai_layer,
                    burst_id = excluded.burst_id,
                    burst_size = excluded.burst_size,
                    burst_index = excluded.burst_index,
                    burst_rank = excluded.burst_rank,
                    bbox_json = excluded.bbox_json,
                    head_bbox_json = excluded.head_bbox_json,
                    detected_class = excluded.detected_class,
                    detection_conf = excluded.detection_conf,
                    detection_source = excluded.detection_source,
                    sharpness_score = excluded.sharpness_score,
                    eye_contact_score = excluded.eye_contact_score,
                    action_score = excluded.action_score,
                    scenic_score = excluded.scenic_score,
                    personal_score = excluded.personal_score,
                    overall_score = excluded.overall_score,
                    overall_rank = excluded.overall_rank,
                    track = excluded.track,
                    species = excluded.species,
                    species_conf = excluded.species_conf,
                    scenario_type = excluded.scenario_type,
                    hybrid_bonus = excluded.hybrid_bonus,
                    tonal_balance = excluded.tonal_balance,
                    kestrel_quality = excluded.kestrel_quality,
                    thumb_rel_path = excluded.thumb_rel_path,
                    crop_rel_path = excluded.crop_rel_path,
                    export_layer1_path = excluded.export_layer1_path,
                    export_layer2_path = excluded.export_layer2_path
                """,
                rows_to_insert,
            )
            conn.execute(
                "UPDATE directories SET last_culled_at = ? WHERE dir_id = ?", (now, dir_id)
            )
            conn.execute(
                "INSERT INTO settings(key, value_json) VALUES ('last_results_json_mtime', ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (json.dumps(file_mtime),),
            )

        self.add_event_log(
            "INFO",
            "system",
            f"Imported culling session ({len(rows_to_insert)} images across {len(self.get_directory(dir_id)['subfolders'])} folders) from results.json",
            {"dir_id": dir_id, "count": len(rows_to_insert)},
        )
        # Re-apply any saved manual burst splits so user splits survive reselect/re-import
        self.reapply_manual_burst_splits(dir_id)
        return len(rows_to_insert)

    # ------------------------------------------------------------------
    # Image Queries & Filters
    # ------------------------------------------------------------------
    def _row_to_image(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["bbox"] = json.loads(d.pop("bbox_json") or "[]")
        except Exception:
            d["bbox"] = []
        try:
            d["head_bbox"] = json.loads(d.pop("head_bbox_json") or "[]")
        except Exception:
            d["head_bbox"] = []
        raw_af = d.pop("af_points_json", None)
        if raw_af is None or raw_af == "null":
            d["af_points"] = None  # None = not yet extracted; [] = extracted, no AF points
        else:
            try:
                d["af_points"] = json.loads(raw_af)
            except Exception:
                d["af_points"] = []
        return d

    def query_images(
        self,
        dir_id: str,
        subfolder: str = "",
        ai_layer: str = "",
        use_ai_layer: bool = False,
        culled_filter: str = "",  # "", "culled", "unculled"
        min_rating: Optional[int] = None,
        exact_rating: Optional[int] = None,
        rating_val: Optional[int] = None,
        rating_op: str = "gte",  # "gte", "eq", "lte"
        label: str = "",
        speed_tier: str = "",  # "", "slow", "medium", "fast", "super_fast"
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
        limit: int = 200,
    ) -> Dict[str, Any]:
        clauses = ["dir_id = ?"]
        params: List[Any] = [dir_id]

        if subfolder and subfolder != "__all__":
            clauses.append("subfolder = ?")
            params.append(subfolder)

        if ai_layer:
            if use_ai_layer:
                if ai_layer == "layer1_all":
                    clauses.append("ai_layer IN ('layer2_to_edit', 'layer1_keep')")
                else:
                    clauses.append("ai_layer = ?")
                    params.append(ai_layer)
            else:
                # Current state based on color labels (Keep includes Edit!)
                if ai_layer == "layer2_to_edit":
                    clauses.append("effective_label = 'Green'")
                elif ai_layer == "layer1_all":
                    clauses.append("effective_label IN ('Green', 'Yellow')")
                elif ai_layer == "layer1_keep":
                    clauses.append("effective_label = 'Yellow'")
                elif ai_layer == "rejected":
                    clauses.append("is_culled = 1 AND effective_label NOT IN ('Green', 'Yellow', 'Red')")
                elif ai_layer == "to_delete":
                    clauses.append("effective_label = 'Red'")
                elif ai_layer == "unculled":
                    clauses.append("is_culled = 0")
                else:
                    clauses.append("ai_layer = ?")
                    params.append(ai_layer)

        if culled_filter == "culled":
            clauses.append("is_culled = 1")
        elif culled_filter == "unculled":
            clauses.append("is_culled = 0")

        if rating_val is not None:
            rv = int(rating_val)
            if rating_op == "eq":
                clauses.append("effective_rating = ?")
            elif rating_op == "lte":
                clauses.append("effective_rating <= ?")
            else:
                clauses.append("effective_rating >= ?")
            params.append(rv)
        elif exact_rating is not None:
            clauses.append("effective_rating = ?")
            params.append(int(exact_rating))
        elif min_rating is not None and min_rating > 0:
            clauses.append("effective_rating >= ?")
            params.append(int(min_rating))

        if label:
            parts = [p.strip() for p in str(label).split(",") if p.strip()]
            if len(parts) == 1 and parts[0] == "__any__":
                clauses.append("effective_label != ''")
            elif len(parts) == 1 and parts[0] == "__none__":
                clauses.append("effective_label = ''")
            elif parts:
                norm_vals = []
                for p in parts:
                    if p == "__none__":
                        norm_vals.append("")
                    else:
                        norm_vals.append(normalize_label(p))
                placeholders = ",".join("?" for _ in norm_vals)
                clauses.append(f"effective_label IN ({placeholders})")
                params.extend(norm_vals)

        if speed_tier:
            # slow: > 1/125s (0.008)
            # medium: 1/500s (0.002) < t <= 1/125s (0.008)
            # fast: 1/1600s (0.000625) < t <= 1/500s (0.002)
            # super_fast: t <= 1/1600s (0.000625)
            if speed_tier == "slow":
                clauses.append("exposure_seconds IS NOT NULL AND exposure_seconds > 0.008")
            elif speed_tier == "medium":
                clauses.append("exposure_seconds IS NOT NULL AND exposure_seconds > 0.002 AND exposure_seconds <= 0.008")
            elif speed_tier == "fast":
                clauses.append("exposure_seconds IS NOT NULL AND exposure_seconds > 0.000625 AND exposure_seconds <= 0.002")
            elif speed_tier == "super_fast":
                clauses.append("exposure_seconds IS NOT NULL AND exposure_seconds <= 0.000625")

        if camera:
            clauses.append("camera_model = ?")
            params.append(camera)
        if lens:
            clauses.append("lens_model = ?")
            params.append(lens)
        if track:
            clauses.append("track = ?")
            params.append(track)
        if species:
            clauses.append("species = ?")
            params.append(species)
        if scenario_type:
            clauses.append("scenario_type = ?")
            params.append(scenario_type)
        if hybrid_only:
            clauses.append("hybrid_bonus = 1")
        if burst_id:
            clauses.append("burst_id = ?")
            params.append(burst_id)
        if only_burst_best:
            clauses.append("burst_rank = 1")
        if search:
            clauses.append("(filename LIKE ? OR burst_id LIKE ? OR detected_class LIKE ? OR species LIKE ? OR lens_model LIKE ?)")
            like = f"%{search.strip()}%"
            params.extend([like, like, like, like, like])

        where = "WHERE " + " AND ".join(clauses)

        sort_map = {
            "overall_rank": "overall_rank",
            "overall_score": "overall_score",
            "sharpness": "sharpness_score",
            "eye_contact": "eye_contact_score",
            "action": "action_score",
            "scenic": "scenic_score",
            "personal": "COALESCE(personal_score, -1)",
            "timestamp": "timestamp",
            "filename": "filename",
            "rating": "effective_rating",
            "shutter_speed": "COALESCE(exposure_seconds, 999)",
            "focal_length": "COALESCE(focal_length, 0)",
            "iso": "COALESCE(iso, 0)",
        }
        col = sort_map.get(sort_by, "overall_rank")
        direction = "DESC" if sort_dir.lower() == "desc" else "ASC"
        order_sql = f"ORDER BY {col} {direction}, timestamp ASC, filename ASC"

        with self._lock, self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) AS c FROM images {where}", params).fetchone()["c"]
            rows = conn.execute(
                f"SELECT * FROM images {where} {order_sql} LIMIT ? OFFSET ?",
                params + [int(limit), int(offset)],
            ).fetchall()

        return {
            "total": int(total),
            "offset": int(offset),
            "limit": int(limit),
            "items": [self._row_to_image(r) for r in rows],
        }

    def get_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM images WHERE image_id = ?", (image_id,)).fetchone()
        return self._row_to_image(row) if row else None

    def get_burst_siblings(self, dir_id: str, burst_id: str) -> List[Dict[str, Any]]:
        if not burst_id:
            return []
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM images WHERE dir_id = ? AND burst_id = ? ORDER BY burst_index ASC, timestamp ASC",
                (dir_id, burst_id),
            ).fetchall()
        return [self._row_to_image(r) for r in rows]

    def get_burst_position_info(self, dir_id: str, burst_id: str, subfolder: str = "") -> Dict[str, int]:
        """Return 1-based burst position, total bursts, first frame's 1-based image position, and total images."""
        if not burst_id:
            return {"burst_pos": 0, "total_bursts": 0, "first_image_pos": 0, "total_images": 0}
        with self._lock, self._connect() as conn:
            sub_clause = " AND subfolder = ?" if subfolder else ""
            sub_params: List[Any] = [dir_id] + ([subfolder] if subfolder else [])

            burst_rows = conn.execute(
                f"SELECT burst_id FROM images WHERE dir_id = ? AND burst_id != ''{sub_clause} "
                f"GROUP BY burst_id ORDER BY MIN(timestamp) ASC, burst_id ASC",
                sub_params,
            ).fetchall()
            total_bursts = len(burst_rows)
            burst_pos = 0
            for idx, r in enumerate(burst_rows, start=1):
                if r["burst_id"] == burst_id:
                    burst_pos = idx
                    break

            total_images = int(
                conn.execute(
                    f"SELECT COUNT(*) AS c FROM images WHERE dir_id = ?{sub_clause}",
                    sub_params,
                ).fetchone()["c"]
            )
            first_row = conn.execute(
                f"SELECT timestamp, filename FROM images WHERE dir_id = ? AND burst_id = ?{sub_clause} "
                f"ORDER BY burst_index ASC, timestamp ASC, filename ASC LIMIT 1",
                [dir_id, burst_id] + ([subfolder] if subfolder else []),
            ).fetchone()
            first_image_pos = 0
            if first_row:
                ts = float(first_row["timestamp"] or 0.0)
                fn = first_row["filename"] or ""
                before = int(
                    conn.execute(
                        f"SELECT COUNT(*) AS c FROM images WHERE dir_id = ?{sub_clause} "
                        f"AND (COALESCE(timestamp, 0) < ? OR (COALESCE(timestamp, 0) = ? AND filename < ?))",
                        sub_params + [ts, ts, fn],
                    ).fetchone()["c"]
                )
                first_image_pos = before + 1

        return {
            "burst_pos": burst_pos,
            "total_bursts": total_bursts,
            "first_image_pos": first_image_pos,
            "total_images": total_images,
        }

    # ------------------------------------------------------------------
    # Manual Burst Split / Unsplit (Persisted across Reselect & Re-import)
    # ------------------------------------------------------------------
    def _renumber_burst_group(self, conn: sqlite3.Connection, burst_id: str, rows: List[sqlite3.Row]) -> None:
        size = len(rows)
        if size == 0:
            return
        # Compute burst_rank by overall_score DESC, sharpness_score DESC
        ranked = sorted(
            rows,
            key=lambda r: (float(r["overall_score"] or 0.0), float(r["sharpness_score"] or 0.0)),
            reverse=True,
        )
        rank_by_id = {r["image_id"]: rk for rk, r in enumerate(ranked, start=1)}
        updates = []
        for idx, r in enumerate(rows, start=1):
            updates.append((burst_id, size, idx, rank_by_id[r["image_id"]], r["image_id"]))
        conn.executemany(
            "UPDATE images SET burst_id = ?, burst_size = ?, burst_index = ?, burst_rank = ? WHERE image_id = ?",
            updates,
        )

    def split_burst_at(self, image_id: str, record_split: bool = True) -> Dict[str, Any]:
        """Split a burst at `image_id` so `image_id` becomes frame 1/M of a new burst."""
        with self._lock, self._connect() as conn:
            target = conn.execute("SELECT * FROM images WHERE image_id = ?", (image_id,)).fetchone()
            if not target:
                raise ValueError("Image not found")
            dir_id = target["dir_id"]
            orig_bid = target["burst_id"]
            if not orig_bid:
                raise ValueError("Image does not belong to a burst")

            members = conn.execute(
                "SELECT * FROM images WHERE dir_id = ? AND burst_id = ? ORDER BY burst_index ASC, timestamp ASC, filename ASC",
                (dir_id, orig_bid),
            ).fetchall()
            split_idx = next((i for i, r in enumerate(members) if r["image_id"] == image_id), -1)
            if split_idx <= 0:
                raise ValueError("Cannot split at the first frame of a burst (must be position > 1)")

            # Generate unique child burst_id (e.g. S0034 -> S0034_2, S0034_3...)
            base_bid = orig_bid.split("_")[0]
            existing_bids = {
                r["burst_id"]
                for r in conn.execute(
                    "SELECT DISTINCT burst_id FROM images WHERE dir_id = ? AND burst_id LIKE ?",
                    (dir_id, f"{base_bid}%"),
                ).fetchall()
            }
            suffix = 2
            while f"{base_bid}_{suffix}" in existing_bids:
                suffix += 1
            new_bid = f"{base_bid}_{suffix}"

            left = members[:split_idx]
            right = members[split_idx:]
            self._renumber_burst_group(conn, orig_bid, left)
            self._renumber_burst_group(conn, new_bid, right)

            if record_split:
                conn.execute(
                    """
                    INSERT INTO burst_splits(image_id, dir_id, orig_burst_id, new_burst_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(image_id) DO UPDATE SET
                        orig_burst_id = excluded.orig_burst_id,
                        new_burst_id = excluded.new_burst_id,
                        created_at = excluded.created_at
                    """,
                    (image_id, dir_id, orig_bid, new_bid, utc_now_iso()),
                )

        if record_split:
            self.add_event_log(
                "INFO",
                "system",
                f"Split burst {orig_bid} at {target['filename']} → {orig_bid} ({len(left)}) + {new_bid} ({len(right)})",
                {"image_id": image_id, "orig_burst_id": orig_bid, "new_burst_id": new_bid},
            )
        return {
            "ok": True,
            "image_id": image_id,
            "orig_burst_id": orig_bid,
            "new_burst_id": new_bid,
            "left_count": len(left),
            "right_count": len(right),
        }

    def unsplit_burst(self, image_id: str) -> Dict[str, Any]:
        """Merge a previously split burst back into its original burst."""
        with self._lock, self._connect() as conn:
            sp = conn.execute("SELECT * FROM burst_splits WHERE image_id = ?", (image_id,)).fetchone()
            if not sp:
                raise ValueError("No manual split found at this image")
            dir_id = sp["dir_id"]
            orig_bid = sp["orig_burst_id"]
            new_bid = sp["new_burst_id"]

            combined = conn.execute(
                "SELECT * FROM images WHERE dir_id = ? AND burst_id IN (?, ?) ORDER BY timestamp ASC, filename ASC",
                (dir_id, orig_bid, new_bid),
            ).fetchall()
            if not combined:
                raise ValueError("Burst frames not found")

            self._renumber_burst_group(conn, orig_bid, combined)
            conn.execute("DELETE FROM burst_splits WHERE image_id = ?", (image_id,))

        self.add_event_log(
            "INFO",
            "system",
            f"Merged split burst {new_bid} back into {orig_bid} ({len(combined)} frames)",
            {"image_id": image_id, "orig_burst_id": orig_bid, "new_burst_id": new_bid},
        )
        return {
            "ok": True,
            "image_id": image_id,
            "merged_burst_id": orig_bid,
            "total_count": len(combined),
        }

    def list_burst_splits(self, dir_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM burst_splits WHERE dir_id = ? ORDER BY created_at ASC",
                (dir_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def reapply_manual_burst_splits(self, dir_id: str) -> int:
        """Re-apply all saved manual burst split points after a pipeline reselect or re-import."""
        splits = self.list_burst_splits(dir_id)
        applied = 0
        for sp in splits:
            try:
                self.split_burst_at(sp["image_id"], record_split=False)
                applied += 1
            except Exception:
                continue
        return applied

    def query_bursts(
        self,
        dir_id: str,
        subfolder: str = "",
        track: str = "",
        species: str = "",
        scenario_type: str = "",
        search: str = "",
        burst_filter: str = "",  # "", "with_picks", "no_picks", "multi_only"
        use_ai_layer: bool = False,
        sort_by: str = "timestamp",  # "timestamp", "score", "size"
        offset: int = 0,
        limit: int = 40,
    ) -> Dict[str, Any]:
        """Return paginated scenario/burst groups with selected items and top runner-ups."""
        clauses = ["dir_id = ?", "burst_id != ''"]
        params: List[Any] = [dir_id]

        if subfolder and subfolder != "__all__":
            clauses.append("subfolder = ?")
            params.append(subfolder)
        if track:
            clauses.append("track = ?")
            params.append(track)
        if species:
            clauses.append("species = ?")
            params.append(species)
        if scenario_type:
            clauses.append("scenario_type = ?")
            params.append(scenario_type)
        if search:
            clauses.append("(filename LIKE ? OR burst_id LIKE ? OR species LIKE ? OR detected_class LIKE ?)")
            like = f"%{search.strip()}%"
            params.extend([like, like, like, like])

        where = "WHERE " + " AND ".join(clauses)

        if use_ai_layer:
            edit_expr = "SUM(CASE WHEN ai_layer = 'layer2_to_edit' THEN 1 ELSE 0 END)"
            keep_expr = "SUM(CASE WHEN ai_layer = 'layer1_keep' THEN 1 ELSE 0 END)"
            del_expr = "SUM(CASE WHEN ai_layer = 'to_delete' THEN 1 ELSE 0 END)"
        else:
            edit_expr = "SUM(CASE WHEN effective_label = 'Green' THEN 1 ELSE 0 END)"
            keep_expr = "SUM(CASE WHEN effective_label = 'Yellow' THEN 1 ELSE 0 END)"
            del_expr = "SUM(CASE WHEN effective_label = 'Red' THEN 1 ELSE 0 END)"

        having = ""
        if burst_filter == "with_picks":
            having = f"HAVING ({edit_expr} + {keep_expr}) > 0"
        elif burst_filter == "no_picks":
            having = f"HAVING ({edit_expr} + {keep_expr}) = 0"
        elif burst_filter == "multi_only":
            having = "HAVING COUNT(*) > 1"

        order_map = {
            "timestamp": "MIN(timestamp) ASC",
            "score": "MAX(overall_score) DESC, MIN(timestamp) ASC",
            "size": "COUNT(*) DESC, MAX(overall_score) DESC",
        }
        order_sql = order_map.get(sort_by, "MIN(timestamp) ASC")

        with self._lock, self._connect() as conn:
            total_sql = f"""
                SELECT COUNT(*) AS c FROM (
                    SELECT burst_id FROM images {where} GROUP BY burst_id {having}
                )
            """
            total_bursts = conn.execute(total_sql, params).fetchone()["c"]

            group_sql = f"""
                SELECT
                    burst_id,
                    MIN(subfolder) AS subfolder,
                    COUNT(*) AS count,
                    MIN(timestamp) AS start_time,
                    MIN(timestamp_str) AS start_time_str,
                    MAX(overall_score) AS max_score,
                    MAX(sharpness_score) AS max_sharpness,
                    MAX(track) AS track,
                    MAX(species) AS species,
                    MAX(scenario_type) AS scenario_type,
                    {edit_expr} AS edit_count,
                    {keep_expr} AS keep_count,
                    {del_expr} AS delete_count
                FROM images
                {where}
                GROUP BY burst_id
                {having}
                ORDER BY {order_sql}
                LIMIT ? OFFSET ?
            """
            group_rows = conn.execute(group_sql, params + [int(limit), int(offset)]).fetchall()

            burst_ids = [r["burst_id"] for r in group_rows]
            items_by_burst: Dict[str, List[Dict[str, Any]]] = {bid: [] for bid in burst_ids}
            if burst_ids:
                ph = ",".join("?" for _ in burst_ids)
                img_rows = conn.execute(
                    f"SELECT * FROM images WHERE dir_id = ? AND burst_id IN ({ph}) ORDER BY burst_index ASC, timestamp ASC",
                    [dir_id] + burst_ids,
                ).fetchall()
                for ir in img_rows:
                    img = self._row_to_image(ir)
                    items_by_burst[img["burst_id"]].append(img)

        bursts_out: List[Dict[str, Any]] = []
        for gr in group_rows:
            bid = gr["burst_id"]
            all_imgs = items_by_burst.get(bid, [])
            if use_ai_layer:
                sel = [x for x in all_imgs if x["ai_layer"] in ("layer2_to_edit", "layer1_keep")]
                unsel = [x for x in all_imgs if x["ai_layer"] not in ("layer2_to_edit", "layer1_keep")]
            else:
                sel = [x for x in all_imgs if x["effective_label"] in ("Green", "Yellow")]
                unsel = [x for x in all_imgs if x["effective_label"] not in ("Green", "Yellow")]
            # Top 3 unselected runner-ups by overall_score/sharpness (useful when a burst has 0 picks)
            preview_others = sorted(
                unsel,
                key=lambda x: (float(x.get("overall_score") or 0.0), float(x.get("sharpness_score") or 0.0)),
                reverse=True,
            )[:3]
            bursts_out.append(
                {
                    "burst_id": bid,
                    "subfolder": gr["subfolder"],
                    "count": int(gr["count"]),
                    "start_time": float(gr["start_time"] or 0.0),
                    "start_time_str": gr["start_time_str"] or "",
                    "max_score": round(float(gr["max_score"] or 0.0), 1),
                    "max_sharpness": round(float(gr["max_sharpness"] or 0.0), 1),
                    "track": gr["track"] or "",
                    "species": gr["species"] or "",
                    "scenario_type": gr["scenario_type"] or "",
                    "edit_count": int(gr["edit_count"] or 0),
                    "keep_count": int(gr["keep_count"] or 0),
                    "delete_count": int(gr["delete_count"] or 0),
                    "selected_items": sel,
                    "preview_others": preview_others,
                    "unselected_count": len(unsel),
                }
            )
        return {
            "total": int(total_bursts),
            "offset": int(offset),
            "limit": int(limit),
            "bursts": bursts_out,
        }

    def get_filter_facets(self, dir_id: str, subfolder: str = "") -> Dict[str, Any]:
        clauses = ["dir_id = ?"]
        params: List[Any] = [dir_id]
        if subfolder and subfolder != "__all__":
            clauses.append("subfolder = ?")
            params.append(subfolder)
        where = "WHERE " + " AND ".join(clauses)
        with self._lock, self._connect() as conn:
            cams = [
                r["camera_model"]
                for r in conn.execute(
                    f"SELECT DISTINCT camera_model FROM images {where} AND camera_model != '' ORDER BY camera_model",
                    params,
                ).fetchall()
            ]
            lenses = [
                r["lens_model"]
                for r in conn.execute(
                    f"SELECT DISTINCT lens_model FROM images {where} AND lens_model != '' ORDER BY lens_model",
                    params,
                ).fetchall()
            ]
            speeds = conn.execute(
                f"""
                SELECT
                    SUM(CASE WHEN exposure_seconds IS NOT NULL AND exposure_seconds > 0.008 THEN 1 ELSE 0 END) AS slow,
                    SUM(CASE WHEN exposure_seconds IS NOT NULL AND exposure_seconds > 0.002 AND exposure_seconds <= 0.008 THEN 1 ELSE 0 END) AS medium,
                    SUM(CASE WHEN exposure_seconds IS NOT NULL AND exposure_seconds > 0.000625 AND exposure_seconds <= 0.002 THEN 1 ELSE 0 END) AS fast,
                    SUM(CASE WHEN exposure_seconds IS NOT NULL AND exposure_seconds <= 0.000625 THEN 1 ELSE 0 END) AS super_fast
                FROM images {where}
                """,
                params,
            ).fetchone()
            track_rows = conn.execute(
                f"SELECT track, COUNT(*) AS c FROM images {where} AND track != '' GROUP BY track ORDER BY c DESC",
                params,
            ).fetchall()
            species_rows = conn.execute(
                f"SELECT species, COUNT(*) AS c FROM images {where} AND species != '' GROUP BY species ORDER BY c DESC LIMIT 100",
                params,
            ).fetchall()
            scenario_rows = conn.execute(
                f"SELECT scenario_type, COUNT(*) AS c FROM images {where} AND scenario_type != '' GROUP BY scenario_type ORDER BY c DESC",
                params,
            ).fetchall()
            hybrid_cnt = conn.execute(
                f"SELECT COUNT(*) AS c FROM images {where} AND hybrid_bonus = 1",
                params,
            ).fetchone()["c"]
        return {
            "cameras": cams,
            "lenses": lenses,
            "speed_counts": {k: int(speeds[k] or 0) for k in speeds.keys()} if speeds else {},
            "tracks": {r["track"]: int(r["c"]) for r in track_rows},
            "species": [{"name": r["species"], "count": int(r["c"])} for r in species_rows],
            "scenarios": {r["scenario_type"]: int(r["c"]) for r in scenario_rows},
            "hybrid_count": int(hybrid_cnt or 0),
        }

    # ------------------------------------------------------------------
    # Upserts from Scanner / Pipeline
    # ------------------------------------------------------------------
    def upsert_scanned_images(self, dir_id: str, items: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Insert newly discovered images or update their EXIF/camera/XMP metadata without losing AI scores."""
        inserted = 0
        updated = 0
        with self._lock, self._connect() as conn:
            for item in items:
                existing = conn.execute(
                    "SELECT image_id, user_modified_at, effective_rating, effective_label FROM images WHERE source_path = ?",
                    (item["source_path"],),
                ).fetchone()
                cam_r = int(item.get("camera_rating") or 0)
                cam_l = normalize_label(item.get("camera_label") or "")
                xmp_exists = 1 if item.get("xmp_exists") else 0
                xmp_r = item.get("xmp_rating")
                xmp_l = normalize_label(item.get("xmp_label")) if item.get("xmp_label") is not None else None

                # Effective rating precedence: XMP sidecar > camera embedded > 0
                if xmp_r is not None:
                    eff_r = int(xmp_r)
                    src_r = "xmp"
                elif cam_r > 0:
                    eff_r = cam_r
                    src_r = "camera"
                else:
                    eff_r = 0
                    src_r = "none"

                if xmp_l is not None and xmp_l != "":
                    eff_l = xmp_l
                elif cam_l:
                    eff_l = cam_l
                else:
                    eff_l = ""

                if existing:
                    # Preserve user's in-app edits if they haven't been superseded by external XMP
                    conn.execute(
                        """
                        UPDATE images SET
                            file_size = CASE WHEN ? > 0 THEN ? ELSE file_size END,
                            lens_model = CASE WHEN ? != '' THEN ? ELSE lens_model END,
                            camera_model = CASE WHEN ? != '' THEN ? ELSE camera_model END,
                            exposure_seconds = COALESCE(?, exposure_seconds),
                            camera_rating = ?,
                            camera_label = ?,
                            af_points_json = CASE WHEN ? IS NOT NULL THEN ? ELSE af_points_json END,
                            xmp_exists = ?,
                            xmp_rating = ?,
                            xmp_label = ?,
                            effective_rating = CASE WHEN user_modified_at IS NULL THEN ? ELSE effective_rating END,
                            effective_label = CASE WHEN user_modified_at IS NULL THEN ? ELSE effective_label END,
                            rating_source = CASE WHEN user_modified_at IS NULL THEN ? ELSE rating_source END
                        WHERE image_id = ?
                        """,
                        (
                            int(item.get("file_size") or 0),
                            int(item.get("file_size") or 0),
                            item.get("lens_model", "") or "",
                            item.get("lens_model", "") or "",
                            item.get("camera_model", "") or "",
                            item.get("camera_model", "") or "",
                            item.get("exposure_seconds"),
                            cam_r,
                            cam_l,
                            json.dumps(item["af_points"]) if item.get("af_points") is not None else None,
                            json.dumps(item["af_points"]) if item.get("af_points") is not None else None,
                            xmp_exists,
                            xmp_r,
                            xmp_l,
                            eff_r,
                            eff_l,
                            src_r,
                            existing["image_id"],
                        ),
                    )
                    updated += 1
                else:
                    conn.execute(
                        """
                        INSERT INTO images (
                            image_id, dir_id, source_path, subfolder, filename, stem, file_ext, file_size,
                            timestamp, timestamp_str, focal_length, f_number, exposure_time, exposure_seconds,
                            iso, camera_model, lens_model,
                            is_culled, ai_layer,
                            camera_rating, camera_label, af_points_json, xmp_exists, xmp_rating, xmp_label,
                            effective_rating, effective_label, rating_source
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'unculled', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item["image_id"],
                            dir_id,
                            item["source_path"],
                            item["subfolder"],
                            item["filename"],
                            item["stem"],
                            item.get("file_ext", ""),
                            int(item.get("file_size") or 0),
                            float(item.get("timestamp") or 0.0),
                            item.get("timestamp_str", "") or "",
                            item.get("focal_length"),
                            item.get("f_number"),
                            item.get("exposure_time", "") or "",
                            item.get("exposure_seconds"),
                            item.get("iso"),
                            item.get("camera_model", "") or "",
                            item.get("lens_model", "") or "",
                            cam_r,
                            cam_l,
                            json.dumps(item["af_points"]) if item.get("af_points") is not None else "null",
                            xmp_exists,
                            xmp_r,
                            xmp_l,
                            eff_r,
                            eff_l,
                            src_r,
                        ),
                    )
                    inserted += 1
        return inserted, updated

    def update_ai_results(self, dir_id: str, run_id: str, records: List[Dict[str, Any]]) -> None:
        now = utc_now_iso()
        with self._lock, self._connect() as conn:
            for rec in records:
                exp_str = rec.get("exposure_time", "") or ""
                exp_sec = rec.get("exposure_seconds") or parse_exposure_seconds(exp_str)
                conn.execute(
                    """
                    UPDATE images SET
                        timestamp = COALESCE(?, timestamp),
                        timestamp_str = CASE WHEN ? != '' THEN ? ELSE timestamp_str END,
                        focal_length = COALESCE(?, focal_length),
                        f_number = COALESCE(?, f_number),
                        exposure_time = CASE WHEN ? != '' THEN ? ELSE exposure_time END,
                        exposure_seconds = COALESCE(?, exposure_seconds),
                        iso = COALESCE(?, iso),
                        camera_model = CASE WHEN ? != '' THEN ? ELSE camera_model END,
                        lens_model = CASE WHEN ? != '' THEN ? ELSE lens_model END,
                        preview_width = COALESCE(?, preview_width),
                        preview_height = COALESCE(?, preview_height),
                        is_culled = 1,
                        culled_at = ?,
                        cull_run_id = ?,
                        ai_layer = ?,
                        burst_id = ?,
                        burst_size = ?,
                        burst_index = ?,
                        burst_rank = ?,
                        bbox_json = ?,
                        head_bbox_json = ?,
                        detected_class = ?,
                        detection_conf = ?,
                        detection_source = ?,
                        subject_area_frac = ?,
                        num_animals = ?,
                        sharpness_raw = ?,
                        sharpness_score = ?,
                        head_tenengrad = ?,
                        head_laplacian = ?,
                        head_luma = ?,
                        layer1_priority = ?,
                        eye_contact_score = ?,
                        action_score = ?,
                        scenic_score = ?,
                        personal_score = ?,
                        overall_score = ?,
                        overall_rank = ?,
                        no_bird_prob = ?,
                        track = ?,
                        species = ?,
                        species_conf = ?,
                        scenario_type = ?,
                        hybrid_bonus = ?,
                        tonal_balance = ?,
                        kestrel_quality = ?,
                        thumb_rel_path = CASE WHEN ? != '' THEN ? ELSE thumb_rel_path END,
                        crop_rel_path = CASE WHEN ? != '' THEN ? ELSE crop_rel_path END,
                        export_layer1_path = CASE WHEN ? != '' THEN ? ELSE export_layer1_path END,
                        export_layer2_path = CASE WHEN ? != '' THEN ? ELSE export_layer2_path END
                    WHERE image_id = ? OR source_path = ?
                    """,
                    (
                        rec.get("timestamp"),
                        rec.get("timestamp_str", "") or "",
                        rec.get("timestamp_str", "") or "",
                        rec.get("focal_length"),
                        rec.get("f_number"),
                        exp_str,
                        exp_str,
                        exp_sec,
                        rec.get("iso"),
                        rec.get("camera_model", "") or "",
                        rec.get("camera_model", "") or "",
                        rec.get("lens_model", "") or "",
                        rec.get("lens_model", "") or "",
                        rec.get("preview_width"),
                        rec.get("preview_height"),
                        now,
                        run_id,
                        rec.get("layer", "rejected"),
                        rec.get("burst_id", "") or "",
                        int(rec.get("burst_size") or 1),
                        int(rec.get("burst_index") or 1),
                        int(rec.get("burst_rank") or 1),
                        json.dumps(rec.get("bbox") or []),
                        json.dumps(rec.get("head_bbox") or []),
                        rec.get("detected_class", "") or "",
                        float(rec.get("detection_conf") or 0.0),
                        rec.get("detection_source", "") or "",
                        float(rec.get("subject_area_frac") or 0.0),
                        int(rec.get("num_animals") or 0),
                        float(rec.get("sharpness_raw") or 0.0),
                        float(rec.get("sharpness_score") or 0.0),
                        float(rec.get("head_tenengrad") or 0.0),
                        float(rec.get("head_laplacian") or 0.0),
                        float(rec.get("head_luma") or 0.0),
                        float(rec.get("layer1_priority") or 0.0),
                        float(rec.get("eye_contact_score") or 0.0),
                        float(rec.get("action_score") or 0.0),
                        float(rec.get("scenic_score") or 0.0),
                        float(rec["personal_score"]) if rec.get("personal_score") is not None else None,
                        float(rec.get("overall_score") or 0.0),
                        int(rec.get("overall_rank") or 999999),
                        float(rec.get("no_bird_prob") or 0.0),
                        str(rec.get("track") or ""),
                        str(rec.get("species") or ""),
                        float(rec.get("species_conf") or 0.0),
                        str(rec.get("scenario_type") or ""),
                        1 if rec.get("hybrid_bonus") else 0,
                        float(rec.get("tonal_balance") or 0.0),
                        float(rec.get("kestrel_quality") or 0.0),
                        rec.get("thumb_rel_path", "") or "",
                        rec.get("thumb_rel_path", "") or "",
                        rec.get("crop_rel_path", "") or "",
                        rec.get("crop_rel_path", "") or "",
                        rec.get("export_layer1_path", "") or "",
                        rec.get("export_layer1_path", "") or "",
                        rec.get("export_layer2_path", "") or "",
                        rec.get("export_layer2_path", "") or "",
                        rec["image_id"],
                        rec["source_path"],
                    ),
                )
            conn.execute(
                "UPDATE directories SET last_culled_at = ? WHERE dir_id = ?", (now, dir_id)
            )

    def set_hq_preview_path(self, image_id: str, rel_path: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE images SET hq_preview_rel_path = ? WHERE image_id = ?",
                (rel_path, image_id),
            )

    def set_af_points(self, image_id: str, af_points: List[Dict[str, Any]]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE images SET af_points_json = ? WHERE image_id = ?",
                (json.dumps(af_points, ensure_ascii=False), image_id),
            )

    # ------------------------------------------------------------------
    # Metadata Edits, Change Log & Undo/Restore
    # ------------------------------------------------------------------
    def record_metadata_change(
        self,
        image_id: str,
        rating: Optional[int] = None,
        label: Optional[str] = None,
        user_flag: Optional[int] = None,
        source: str = "user",
        action_type: str = "edit",
        xmp_synced: bool = False,
        prev_xmp_xml: Optional[str] = None,
        batch_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        img = self.get_image(image_id)
        if not img:
            return None
        now = utc_now_iso()
        new_rating = img["effective_rating"] if rating is None else max(0, min(5, int(rating)))
        new_label = img["effective_label"] if label is None else normalize_label(label)
        new_flag = img["user_flag"] if user_flag is None else int(user_flag)

        changes = []
        if rating is not None and new_rating != img["effective_rating"]:
            changes.append(("rating", str(img["effective_rating"]), str(new_rating)))
        if label is not None and new_label != img["effective_label"]:
            changes.append(("label", img["effective_label"], new_label))
        if user_flag is not None and new_flag != img["user_flag"]:
            changes.append(("flag", str(img["user_flag"]), str(new_flag)))

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE images SET
                    effective_rating = ?,
                    effective_label = ?,
                    user_flag = ?,
                    xmp_exists = CASE WHEN ? = 1 THEN 1 ELSE xmp_exists END,
                    xmp_rating = CASE WHEN ? = 1 THEN ? ELSE xmp_rating END,
                    xmp_label = CASE WHEN ? = 1 THEN ? ELSE xmp_label END,
                    rating_source = ?,
                    user_modified_at = ?
                WHERE image_id = ?
                """,
                (
                    new_rating,
                    new_label,
                    new_flag,
                    1 if xmp_synced else 0,
                    1 if xmp_synced else 0,
                    new_rating,
                    1 if xmp_synced else 0,
                    new_label,
                    source,
                    now,
                    image_id,
                ),
            )
            for fld, old_v, new_v in changes:
                row_xmp_synced = 1 if (xmp_synced and fld in ("rating", "label")) else 0
                conn.execute(
                    """
                    INSERT INTO change_log(
                        timestamp, dir_id, image_id, filename, action_type, field,
                        old_value, new_value, prev_xmp_xml, xmp_synced, reverted,
                        batch_id, ai_layer_at_time, ai_score_at_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                    """,
                    (
                        now,
                        img["dir_id"],
                        image_id,
                        img["filename"],
                        action_type,
                        fld,
                        old_v,
                        new_v,
                        prev_xmp_xml if row_xmp_synced else None,
                        row_xmp_synced,
                        batch_id,
                        img["ai_layer"],
                        float(img.get("overall_score") or 0.0),
                    ),
                )
        return self.get_image(image_id)

    def get_change_log(self, limit: int = 200, dir_id: str = "") -> List[Dict[str, Any]]:
        where = "WHERE dir_id = ?" if dir_id else ""
        params: List[Any] = [dir_id, int(limit)] if dir_id else [int(limit)]
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, timestamp, dir_id, image_id, filename, action_type, field,
                       old_value, new_value, xmp_synced, reverted, batch_id,
                       ai_layer_at_time, ai_score_at_time
                FROM change_log {where}
                ORDER BY id DESC LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_change_entry(self, change_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM change_log WHERE id = ?", (change_id,)).fetchone()
        return dict(row) if row else None

    def get_latest_unreverted_change(self, dir_id: str = "") -> Optional[Dict[str, Any]]:
        where = "WHERE reverted = 0" + (" AND dir_id = ?" if dir_id else "")
        params = [dir_id] if dir_id else []
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM change_log {where} ORDER BY id DESC LIMIT 1", params
            ).fetchone()
        return dict(row) if row else None

    def get_batch_changes(self, batch_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM change_log WHERE batch_id = ? AND reverted = 0 ORDER BY id DESC",
                (batch_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_sibling_unreverted_changes(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return unreverted sibling change_log rows from the exact same edit on the same image."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM change_log
                WHERE image_id = ? AND timestamp = ? AND action_type = ?
                  AND batch_id = ? AND reverted = 0 AND id != ?
                ORDER BY id DESC
                """,
                (
                    entry["image_id"],
                    entry["timestamp"],
                    entry["action_type"],
                    entry.get("batch_id") or "",
                    entry["id"],
                ),
            ).fetchall()
        return [dict(r) for r in rows]

    def mark_change_reverted(self, change_id: int, field: str, old_value: str, image_id: str) -> None:
        with self._lock, self._connect() as conn:
            if field == "rating":
                conn.execute(
                    "UPDATE images SET effective_rating = ?, user_modified_at = ? WHERE image_id = ?",
                    (int(old_value or 0), utc_now_iso(), image_id),
                )
            elif field == "label":
                conn.execute(
                    "UPDATE images SET effective_label = ?, user_modified_at = ? WHERE image_id = ?",
                    (old_value or "", utc_now_iso(), image_id),
                )
            elif field == "flag":
                conn.execute(
                    "UPDATE images SET user_flag = ?, user_modified_at = ? WHERE image_id = ?",
                    (int(old_value or 0), utc_now_iso(), image_id),
                )
            conn.execute("UPDATE change_log SET reverted = 1 WHERE id = ?", (change_id,))

    # ------------------------------------------------------------------
    # Pipeline Runs
    # ------------------------------------------------------------------
    def create_pipeline_run(self, run_id: str, dir_id: str, mode: str, total_images: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs(run_id, dir_id, mode, status, started_at, total_images)
                VALUES (?, ?, ?, 'running', ?, ?)
                """,
                (run_id, dir_id, mode, utc_now_iso(), int(total_images)),
            )

    def finish_pipeline_run(
        self,
        run_id: str,
        status: str,
        processed_images: int,
        skipped_errors: int,
        summary: Dict[str, Any],
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE pipeline_runs SET
                    status = ?,
                    finished_at = ?,
                    processed_images = ?,
                    skipped_errors = ?,
                    summary_json = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    utc_now_iso(),
                    int(processed_images),
                    int(skipped_errors),
                    json.dumps(summary, ensure_ascii=False),
                    run_id,
                ),
            )

    def list_pipeline_runs(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["summary"] = json.loads(d.pop("summary_json") or "{}")
            except Exception:
                d["summary"] = {}
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Learning & Insights (AI vs User Ratings)
    # ------------------------------------------------------------------
    def get_learning_stats(self, dir_id: str) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            by_rating = conn.execute(
                """
                SELECT
                    effective_rating AS rating,
                    COUNT(*) AS count,
                    ROUND(AVG(overall_score), 1) AS avg_overall,
                    ROUND(AVG(sharpness_score), 1) AS avg_sharpness,
                    ROUND(AVG(eye_contact_score), 1) AS avg_eye,
                    ROUND(AVG(action_score), 1) AS avg_action
                FROM images
                WHERE dir_id = ? AND is_culled = 1 AND effective_rating > 0
                GROUP BY effective_rating
                ORDER BY effective_rating DESC
                """,
                (dir_id,),
            ).fetchall()

            matrix = conn.execute(
                """
                SELECT
                    ai_layer,
                    SUM(CASE WHEN effective_rating >= 4 THEN 1 ELSE 0 END) AS high_4_5,
                    SUM(CASE WHEN effective_rating IN (2, 3) THEN 1 ELSE 0 END) AS mid_2_3,
                    SUM(CASE WHEN effective_rating = 1 OR user_flag = -1 THEN 1 ELSE 0 END) AS low_1_rej,
                    COUNT(*) AS total
                FROM images
                WHERE dir_id = ? AND is_culled = 1 AND (effective_rating > 0 OR user_flag != 0)
                GROUP BY ai_layer
                """,
                (dir_id,),
            ).fetchall()

            overrides = conn.execute(
                """
                SELECT image_id, filename, subfolder, ai_layer, overall_score, sharpness_score,
                       eye_contact_score, effective_rating, effective_label
                FROM images
                WHERE dir_id = ? AND is_culled = 1 AND (
                    (ai_layer = 'rejected' AND effective_rating >= 4) OR
                    (ai_layer = 'layer2_to_edit' AND (effective_rating = 1 OR user_flag = -1))
                )
                ORDER BY user_modified_at DESC
                LIMIT 50
                """,
                (dir_id,),
            ).fetchall()

        return {
            "by_rating": [dict(r) for r in by_rating],
            "matrix": [dict(r) for r in matrix],
            "notable_overrides": [dict(r) for r in overrides],
        }

    def get_training_picks(self, dir_id: str) -> List[Dict[str, Any]]:
        """
        Export user feedback (ratings, color labels, pick/reject flags) in the exact format
        expected by train_ranker.train_from_picks:
          - 4..5★ or Green or Pick flag -> 'layer2_edit'
          - 2..3★ or Yellow             -> 'layer1_keep'
          - 1★ or Red or Reject flag    -> 'reject'
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT image_id, filename, effective_rating, effective_label, user_flag, ai_layer
                FROM images
                WHERE dir_id = ? AND is_culled = 1 AND (
                    effective_rating > 0 OR user_flag != 0 OR effective_label IN ('Green', 'Yellow', 'Red')
                )
                """,
                (dir_id,),
            ).fetchall()
        picks: List[Dict[str, Any]] = []
        for r in rows:
            rating = int(r["effective_rating"] or 0)
            label = str(r["effective_label"] or "")
            flag = int(r["user_flag"] or 0)
            dec = ""
            if flag == -1 or rating == 1 or (rating == 0 and label == "Red"):
                dec = "reject"
            elif rating >= 4 or flag == 1 or (rating == 0 and label == "Green"):
                dec = "layer2_edit"
            elif rating in (2, 3) or (rating == 0 and label == "Yellow"):
                dec = "layer1_keep"
            if dec:
                picks.append({"image_id": r["image_id"], "user_decision": dec})
        return picks
