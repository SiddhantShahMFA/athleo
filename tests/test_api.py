from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from athleo.api.app import create_app
from athleo.jobs import JobManager
from athleo.models import ArtifactType
from athleo.storage import JobStorage


class FakeBackend:
    def __init__(self) -> None:
        self.process_calls = 0
        self.export_calls = 0

    def process_job(self, paths, options):
        self.process_calls += 1
        tracks = {
            "frames": {
                "0": [
                    {
                        "frame_idx": 0,
                        "track_id": 7,
                        "x1": 10.0,
                        "y1": 20.0,
                        "x2": 30.0,
                        "y2": 60.0,
                    }
                ]
            }
        }
        poses = {
            "frames": {
                "0": [
                    {
                        "frame_idx": 0,
                        "track_id": 7,
                        "keypoints": [],
                    }
                ]
            }
        }
        balls = {
            "frames": {
                "0": [
                    {
                        "frame_idx": 0,
                        "track_id": 99,
                        "x1": 100.0,
                        "y1": 110.0,
                        "x2": 120.0,
                        "y2": 130.0,
                    }
                ]
            }
        }
        paths.track_cache_path.write_text(json.dumps(tracks), encoding="utf-8")
        paths.pose_cache_path.write_text(json.dumps(poses), encoding="utf-8")
        paths.ball_cache_path.write_text(json.dumps(balls), encoding="utf-8")
        stat = paths.video_path.stat()
        return {
            "video_info": {
                "path": str(paths.video_path),
                "width": 1280,
                "height": 720,
                "fps": 30.0,
                "total_frames": 2,
                "file_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        }

    def export_job(self, paths, options, export_request, export_id, video_info):
        self.export_calls += 1
        output_path = paths.export_dir / f"{export_id}_{export_request.mode.value}.mp4"
        output_path.write_bytes(b"fake-export")
        return output_path


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.backend = FakeBackend()
        self.manager = JobManager(
            storage=JobStorage(Path(self.temp_dir.name)),
            backend=self.backend,
            max_workers=1,
        )
        self.client = TestClient(create_app(job_manager=self.manager))

    def tearDown(self) -> None:
        self.client.close()
        self.manager.close()
        self.temp_dir.cleanup()

    def wait_for_job(self, job_id: str) -> dict:
        for _ in range(60):
            response = self.client.get(f"/api/v1/jobs/{job_id}")
            payload = response.json()
            if payload["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.05)
        self.fail(f"Timed out waiting for job {job_id}")

    def wait_for_export(self, job_id: str, export_id: str) -> dict:
        for _ in range(60):
            response = self.client.get(f"/api/v1/jobs/{job_id}/exports/{export_id}")
            payload = response.json()
            if payload["status"] in {"completed", "failed"}:
                return payload
            time.sleep(0.05)
        self.fail(f"Timed out waiting for export {export_id}")

    def test_health_and_ready(self) -> None:
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/readyz").json(), {"status": "ok"})

    def test_job_lifecycle_artifacts_and_cache_reuse(self) -> None:
        first_response = self.client.post(
            "/api/v1/jobs",
            files={"video": ("clip.mp4", io.BytesIO(b"same-video"), "video/mp4")},
        )
        self.assertEqual(first_response.status_code, 202)
        first_job_id = first_response.json()["job_id"]
        first_job = self.wait_for_job(first_job_id)
        self.assertEqual(first_job["status"], "completed")
        self.assertEqual(self.backend.process_calls, 1)

        frame_response = self.client.get(f"/api/v1/jobs/{first_job_id}/frames/0")
        self.assertEqual(frame_response.status_code, 200)
        self.assertEqual(frame_response.json()["tracks"][0]["track_id"], 7)

        artifacts_response = self.client.get(f"/api/v1/jobs/{first_job_id}/artifacts")
        self.assertEqual(artifacts_response.status_code, 200)
        artifact_ids = {artifact["artifact_id"] for artifact in artifacts_response.json()["artifacts"]}
        self.assertTrue({"input_video", "job_manifest", "track_cache", "pose_cache", "ball_cache"}.issubset(artifact_ids))

        download_response = self.client.get(f"/api/v1/jobs/{first_job_id}/artifacts/input_video")
        self.assertEqual(download_response.status_code, 200)
        self.assertEqual(download_response.content, b"same-video")

        second_response = self.client.post(
            "/api/v1/jobs",
            files={"video": ("clip.mp4", io.BytesIO(b"same-video"), "video/mp4")},
        )
        self.assertEqual(second_response.status_code, 202)
        second_job_id = second_response.json()["job_id"]
        second_job = self.wait_for_job(second_job_id)
        self.assertEqual(second_job["status"], "completed")
        self.assertEqual(self.backend.process_calls, 1)

    def test_export_validation_and_download(self) -> None:
        response = self.client.post(
            "/api/v1/jobs",
            files={"video": ("clip.mp4", io.BytesIO(b"export-video"), "video/mp4")},
        )
        job_id = response.json()["job_id"]
        self.wait_for_job(job_id)

        invalid_export = self.client.post(
            f"/api/v1/jobs/{job_id}/exports",
            json={"mode": "spotlight"},
        )
        self.assertEqual(invalid_export.status_code, 422)

        export_response = self.client.post(
            f"/api/v1/jobs/{job_id}/exports",
            json={"mode": "spotlight", "selected_track_id": 7},
        )
        self.assertEqual(export_response.status_code, 202)
        export_id = export_response.json()["export_id"]

        export_status = self.wait_for_export(job_id, export_id)
        self.assertEqual(export_status["status"], "completed")
        self.assertEqual(self.backend.export_calls, 1)

        download_response = self.client.get(f"/api/v1/jobs/{job_id}/exports/{export_id}/download")
        self.assertEqual(download_response.status_code, 200)
        self.assertEqual(download_response.content, b"fake-export")


class StorageTests(unittest.TestCase):
    def test_storage_paths_and_manifest_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = JobStorage(Path(temp_dir))
            paths = storage.build_paths("job123", "My Match.mp4")
            storage.initialize_job(paths)
            video_hash = storage.save_upload(io.BytesIO(b"video"), paths.video_path)
            manifest = {"job_id": "job123", "input_sha256": video_hash}
            storage.save_manifest(paths, manifest)

            loaded_manifest = storage.load_manifest(paths)
            artifact = storage.artifact_record(paths, "input_video", ArtifactType.INPUT_VIDEO, paths.video_path)

            self.assertEqual(paths.video_path.name, "My_Match.mp4")
            self.assertEqual(loaded_manifest["input_sha256"], video_hash)
            self.assertEqual(artifact["relative_path"], "input/My_Match.mp4")


if __name__ == "__main__":
    unittest.main()
