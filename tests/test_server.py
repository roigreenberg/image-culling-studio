import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlencode

import numpy as np

import webapp.server as srv
from src.io_utils import save_jpeg_bgr
from webapp.db import CullDatabase, make_image_id
from webapp.pipeline_runner import TaskManager


class _AsgiResponse:
    def __init__(self, status_code: int, headers: dict, body: bytes):
        self.status_code = status_code
        self.headers = headers
        self.content = body

    def json(self):
        return json.loads(self.content.decode("utf-8"))

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class _AsgiClient:
    """Socket-free in-memory ASGI client for FastAPI endpoints."""

    def __init__(self, app):
        self.app = app

    def request(self, method: str, path: str, params=None, json_body=None) -> _AsgiResponse:
        qs = urlencode(params, doseq=True).encode("utf-8") if params else b""
        body = b""
        headers = []
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers.append((b"content-type", b"application/json"))
            headers.append((b"content-length", str(len(body)).encode("ascii")))

        sent = False
        msgs = []

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        async def send(msg):
            msgs.append(msg)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method.upper(),
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": qs,
            "headers": headers,
            "scheme": "http",
            "server": ("testserver", 80),
        }

        coro = self.app(scope, receive, send)
        try:
            while True:
                coro.send(None)
        except StopIteration:
            pass

        status = 500
        resp_headers = {}
        chunks = []
        for m in msgs:
            if m["type"] == "http.response.start":
                status = m["status"]
                resp_headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in m.get("headers", [])}
            elif m["type"] == "http.response.body":
                chunks.append(m.get("body", b""))
        return _AsgiResponse(status, resp_headers, b"".join(chunks))

    def get(self, path: str, params=None) -> _AsgiResponse:
        return self.request("GET", path, params=params)

    def post(self, path: str, params=None, json=None) -> _AsgiResponse:
        return self.request("POST", path, params=params, json_body=json)

    def put(self, path: str, params=None, json=None) -> _AsgiResponse:
        return self.request("PUT", path, params=params, json_body=json)


class TestWebappServer(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self._tmp.name)
        self.test_db = CullDatabase(self.root / "test_studio.db")
        self.test_tm = TaskManager(self.test_db, broadcast=lambda _p: None)

        self.src_dir = self.root / "photos"
        self.out_dir = self.root / "out"
        (self.src_dir / "day1").mkdir(parents=True)
        self.out_dir.mkdir(parents=True)

        img = np.full((120, 160, 3), 110, dtype=np.uint8)
        save_jpeg_bgr(img, self.src_dir / "day1" / "IMG_001.jpg")
        save_jpeg_bgr(img, self.src_dir / "day1" / "IMG_002.jpg")

        self.dir_rec = self.test_db.register_directory(self.src_dir, self.out_dir)
        self.dir_id = self.dir_rec["dir_id"]

        self.img1_id = make_image_id(self.dir_id, "day1", "IMG_001.jpg")
        self.img2_id = make_image_id(self.dir_id, "day1", "IMG_002.jpg")
        self.test_db.upsert_scanned_images(
            self.dir_id,
            [
                {
                    "image_id": self.img1_id,
                    "source_path": str(self.src_dir / "day1" / "IMG_001.jpg"),
                    "subfolder": "day1",
                    "filename": "IMG_001.jpg",
                    "stem": "IMG_001",
                },
                {
                    "image_id": self.img2_id,
                    "source_path": str(self.src_dir / "day1" / "IMG_002.jpg"),
                    "subfolder": "day1",
                    "filename": "IMG_002.jpg",
                    "stem": "IMG_002",
                },
            ],
        )
        self.test_db.update_ai_results(
            self.dir_id,
            "run_1",
            [
                {
                    "image_id": self.img1_id,
                    "source_path": str(self.src_dir / "day1" / "IMG_001.jpg"),
                    "layer": "layer2_to_edit",
                    "track": "Wildlife",
                    "burst_id": "S0001",
                    "burst_size": 2,
                    "burst_index": 1,
                    "burst_rank": 1,
                    "overall_score": 88.0,
                    "overall_rank": 1,
                },
                {
                    "image_id": self.img2_id,
                    "source_path": str(self.src_dir / "day1" / "IMG_002.jpg"),
                    "layer": "layer1_keep",
                    "track": "Wildlife",
                    "burst_id": "S0001",
                    "burst_size": 2,
                    "burst_index": 2,
                    "burst_rank": 2,
                    "overall_score": 70.0,
                    "overall_rank": 2,
                },
            ],
        )

        async def _sync_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        async def _sync_anyio_run_sync(fn, *args, **_kwargs):
            return fn(*args)

        async def _sync_file_response(self_resp, _scope, _receive, send):
            data = Path(self_resp.path).read_bytes()
            await send({
                "type": "http.response.start",
                "status": self_resp.status_code,
                "headers": [(b"content-type", (self_resp.media_type or "application/octet-stream").encode("latin1"))],
            })
            await send({"type": "http.response.body", "body": data})

        self._patches = [
            patch.object(srv, "db", self.test_db),
            patch.object(srv, "tasks", self.test_tm),
            patch.object(srv, "DEFAULT_OUTPUT_DIR", self.out_dir),
            patch("webapp.server.asyncio.to_thread", side_effect=_sync_to_thread),
            patch("anyio.to_thread.run_sync", side_effect=_sync_anyio_run_sync),
            patch("starlette.responses.FileResponse.__call__", new=_sync_file_response),
        ]
        for p in self._patches:
            p.start()
        srv._thumb_path_cache.clear()
        srv._crop_path_cache.clear()
        self.client = _AsgiClient(srv.app)

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def test_index_settings_session_and_directories(self) -> None:
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)

        r_set = self.client.get("/api/settings")
        self.assertEqual(r_set.status_code, 200)
        self.assertIn("realtime_xmp_write", r_set.json())

        r_put = self.client.put("/api/settings", json={"grid_card_size": 280})
        self.assertEqual(r_put.status_code, 200)
        self.assertEqual(r_put.json()["grid_card_size"], 280)

        r_sess = self.client.put("/api/session", json={"state": {"tab": "layer1"}})
        self.assertEqual(r_sess.status_code, 200)
        self.assertEqual(self.client.get("/api/session").json()["tab"], "layer1")

        r_dirs = self.client.get("/api/directories")
        self.assertEqual(r_dirs.status_code, 200)
        self.assertEqual(len(r_dirs.json()["directories"]), 1)

        r_dir = self.client.get(f"/api/directories/{self.dir_id}")
        self.assertEqual(r_dir.status_code, 200)
        self.assertEqual(r_dir.json()["dir_id"], self.dir_id)

        r_open = self.client.post("/api/directories/open", json={"path": str(self.src_dir), "scan_now": False})
        self.assertEqual(r_open.status_code, 200)
        self.assertFalse(r_open.json()["scan_started"])

        r_open_bad = self.client.post("/api/directories/open", json={"path": str(self.root / "missing"), "scan_now": False})
        self.assertEqual(r_open_bad.status_code, 400)

    def test_images_metadata_undo_bursts(self) -> None:
        r_imgs = self.client.get("/api/images", params={"dir_id": self.dir_id})
        self.assertEqual(r_imgs.status_code, 200)
        self.assertEqual(r_imgs.json()["total"], 2)

        r_one = self.client.get(f"/api/images/{self.img1_id}")
        self.assertEqual(r_one.status_code, 200)
        self.assertEqual(r_one.json()["image"]["filename"], "IMG_001.jpg")

        # Update metadata + undo
        r_upd = self.client.post(f"/api/images/{self.img1_id}/metadata", json={"rating": 5, "label": "Green"})
        self.assertEqual(r_upd.status_code, 200)
        self.assertEqual(r_upd.json()["image"]["effective_rating"], 5)

        r_undo = self.client.post("/api/changes/undo-last", params={"dir_id": self.dir_id})
        self.assertEqual(r_undo.status_code, 200)
        self.assertTrue(r_undo.json()["ok"])

        # Batch metadata
        r_batch = self.client.post(
            "/api/images/batch-metadata",
            json={"image_ids": [self.img1_id, self.img2_id], "rating": 4, "label": "Yellow"},
        )
        self.assertEqual(r_batch.status_code, 200)
        self.assertEqual(r_batch.json()["updated"], 2)

        # Burst split & unsplit
        r_split = self.client.post("/api/bursts/split", json={"image_id": self.img2_id})
        self.assertEqual(r_split.status_code, 200)
        self.assertTrue(r_split.json()["ok"])

        r_unsplit = self.client.post("/api/bursts/unsplit", json={"image_id": self.img2_id})
        self.assertEqual(r_unsplit.status_code, 200)
        self.assertTrue(r_unsplit.json()["ok"])

    def test_pipeline_status_logs_learning_and_xmp_mapping(self) -> None:
        self.assertEqual(self.client.get("/api/pipeline/status").status_code, 200)
        self.assertEqual(self.client.get("/api/pipeline/runs").status_code, 200)
        self.assertEqual(self.client.get("/api/logs").status_code, 200)
        self.assertEqual(self.client.post("/api/perf/log", json={"events": [{"name": "render", "ms": 12.3}]}).status_code, 200)
        self.assertEqual(self.client.get(f"/api/learning/{self.dir_id}").status_code, 200)

        r_map = self.client.post(
            "/api/xmp/apply-ai-mapping",
            json={"dir_id": self.dir_id, "dry_run": True},
        )
        self.assertEqual(r_map.status_code, 200)
        self.assertEqual(r_map.json()["would_update"], 2)

    def test_remaining_endpoints_bursts_changes_media_and_pipeline(self) -> None:
        # Facets & browse & scan
        self.assertEqual(self.client.get(f"/api/directories/{self.dir_id}/facets").status_code, 200)
        with patch("webapp.server.subprocess.run", return_value=SimpleNamespace(stdout=str(self.src_dir))):
            r_br = self.client.post("/api/directories/browse")
            self.assertEqual(r_br.status_code, 200)
            self.assertEqual(r_br.json()["path"], str(self.src_dir))

        with patch.object(self.test_tm, "start_directory_scan", return_value=(True, "Scan started")):
            r_sc = self.client.post(f"/api/directories/{self.dir_id}/scan")
            self.assertEqual(r_sc.status_code, 200)
            self.assertTrue(r_sc.json()["ok"])

        # Bursts GET endpoints
        r_b = self.client.get("/api/bursts", params={"dir_id": self.dir_id})
        self.assertEqual(r_b.status_code, 200)
        self.assertEqual(r_b.json()["total"], 1)
        r_b1 = self.client.get(f"/api/bursts/{self.dir_id}/S0001")
        self.assertEqual(r_b1.status_code, 200)
        self.assertEqual(len(r_b1.json()["items"]), 2)
        self.assertEqual(self.client.get(f"/api/bursts/splits/{self.dir_id}").status_code, 200)

        # Changes GET & restore
        self.client.post(f"/api/images/{self.img1_id}/metadata", json={"rating": 3, "label": "Blue"})
        r_ch = self.client.get("/api/changes", params={"dir_id": self.dir_id})
        self.assertEqual(r_ch.status_code, 200)
        chg_id = r_ch.json()["changes"][0]["id"]
        r_rest = self.client.post(f"/api/changes/{chg_id}/restore")
        self.assertEqual(r_rest.status_code, 200)
        self.assertTrue(r_rest.json()["ok"])

        # Media endpoints (create thumb/crop files + HQ/full on-demand)
        thumb_p = self.out_dir / "gallery_assets" / "thumbs" / "t1.jpg"
        crop_p = self.out_dir / "gallery_assets" / "crops" / "c1.jpg"
        img = np.full((64, 64, 3), 120, dtype=np.uint8)
        save_jpeg_bgr(img, thumb_p)
        save_jpeg_bgr(img, crop_p)
        with self.test_db._lock, self.test_db._connect() as conn:
            conn.execute(
                "UPDATE images SET thumb_rel_path = ?, crop_rel_path = ? WHERE image_id = ?",
                ("gallery_assets/thumbs/t1.jpg", "gallery_assets/crops/c1.jpg", self.img1_id),
            )
        self.assertEqual(self.client.get(f"/api/media/thumb/{self.img1_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/media/crop/{self.img1_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/media/hq/{self.img1_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/media/full/{self.img1_id}").status_code, 200)

        # Pipeline control, sync-results, previews/generate, learning/train
        with patch.object(self.test_tm, "start_culling", return_value=(True, "Started")):
            self.assertEqual(
                self.client.post("/api/pipeline/start", json={"dir_id": self.dir_id, "mode": "full"}).status_code,
                200,
            )
        with patch.object(self.test_tm, "start_reselect", return_value=(True, "Started")):
            self.assertEqual(self.client.post("/api/pipeline/reselect", json={"dir_id": self.dir_id}).status_code, 200)
        with patch.object(self.test_tm, "cancel_current", return_value=True):
            self.assertEqual(self.client.post("/api/pipeline/cancel").status_code, 200)
        with patch.object(self.test_tm, "start_hq_preview_batch", return_value=(True, "Started")):
            self.assertEqual(self.client.post("/api/previews/generate", json={"dir_id": self.dir_id}).status_code, 200)
        (self.out_dir / "results.json").write_text(
            json.dumps([{"image_id": self.img1_id, "source_path": str(self.src_dir / "day1" / "IMG_001.jpg"), "subfolder": "day1", "filename": "IMG_001.jpg", "stem": "IMG_001", "layer": "layer2_to_edit", "track": "Wildlife", "overall_score": 88.0}]),
            encoding="utf-8",
        )
        self.assertEqual(self.client.post(f"/api/directories/{self.dir_id}/sync-results").status_code, 200)
        with patch.object(self.test_tm, "train_personal_ranker", return_value={"ok": True, "picks_exported": 6}):
            r_tr = self.client.post(f"/api/learning/{self.dir_id}/train", json={"auto_reselect": False})
            self.assertEqual(r_tr.status_code, 200)
            self.assertTrue(r_tr.json()["ok"])

    def test_websocket_events_and_broadcast(self) -> None:
        from fastapi import WebSocketDisconnect

        accepted = []
        sent_msgs = []
        recv_calls = 0

        class _MockWebSocket:
            async def accept(self):
                accepted.append(True)

            async def send_json(self, data):
                sent_msgs.append(data)

            async def receive_text(self):
                nonlocal recv_calls
                recv_calls += 1
                if recv_calls == 1:
                    return "ping"
                raise WebSocketDisconnect(code=1000)

        mock_ws = _MockWebSocket()
        srv._ws_clients.clear()
        coro = srv.ws_events(mock_ws)
        try:
            while True:
                coro.send(None)
        except StopIteration:
            pass

        self.assertEqual(accepted, [True])
        self.assertEqual(len(sent_msgs), 1)
        self.assertEqual(sent_msgs[0]["type"], "task_progress")
        self.assertNotIn(mock_ws, srv._ws_clients)

        # Test broadcast_event with mocked _main_loop and connected client
        srv._ws_clients.add(mock_ws)
        scheduled = []

        def _fake_run_coro(c, loop):
            scheduled.append((c, loop))
            c.close()

        try:
            with (
                patch.object(srv, "_main_loop", object()),
                patch("webapp.server.asyncio.run_coroutine_threadsafe", side_effect=_fake_run_coro),
            ):
                srv.broadcast_event({"type": "log", "data": {"message": "hello"}})
            self.assertEqual(len(scheduled), 1)
        finally:
            srv._ws_clients.clear()


if __name__ == "__main__":
    unittest.main()
