from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, BinaryIO
from uuid import uuid4

from .backend import RuntimeBackend
from .models import ArtifactType, ExportRequest, JobStatus, OverlayMode, ProcessingOptions
from .storage import JobPaths, JobStorage


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(
        self,
        storage: JobStorage,
        backend: RuntimeBackend | None = None,
        max_workers: int = 2,
    ) -> None:
        self.storage = storage
        self.backend = backend or RuntimeBackend()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="athleo-jobs")
        self._lock = threading.RLock()

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)

    def create_job(
        self,
        upload: BinaryIO,
        filename: str,
        options: ProcessingOptions,
    ) -> dict[str, Any]:
        job_id = uuid4().hex
        paths = self.storage.build_paths(job_id, filename)
        self.storage.initialize_job(paths)
        input_sha256 = self.storage.save_upload(upload, paths.video_path)

        created_at = utc_now()
        manifest = {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
            "stored_video_name": paths.video_path.name,
            "input_sha256": input_sha256,
            "options": options.to_dict(),
            "video_info": None,
            "artifacts": {},
            "exports": {},
        }
        self.storage.save_manifest(paths, manifest)
        manifest["artifacts"] = self._collect_base_artifacts(paths)
        self.storage.save_manifest(paths, manifest)
        if self._try_reuse_processed_artifacts(paths, manifest):
            return manifest
        self.executor.submit(self._run_processing_job, job_id)
        return manifest

    def get_job(self, job_id: str) -> dict[str, Any]:
        paths, manifest = self._load_paths_and_manifest(job_id)
        manifest["artifacts"] = self._collect_artifacts(paths, manifest)
        return manifest

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        paths, manifest = self._load_paths_and_manifest(job_id)
        artifacts = self._collect_artifacts(paths, manifest)
        return list(artifacts.values())

    def get_frame(self, job_id: str, frame_idx: int) -> dict[str, Any]:
        paths, manifest = self._load_paths_and_manifest(job_id)
        if manifest["status"] != JobStatus.COMPLETED.value:
            raise RuntimeError("Frame metadata is only available after processing completes.")
        video_info = manifest.get("video_info") or {}
        total_frames = int(video_info.get("total_frames", 0))
        if frame_idx < 0 or (total_frames and frame_idx >= total_frames):
            raise ValueError(f"Frame {frame_idx} is outside the processed video range.")
        return {
            "job_id": job_id,
            "frame_idx": frame_idx,
            "tracks": self.storage.load_frame_entries(paths.track_cache_path, frame_idx),
            "pose_entries": self.storage.load_frame_entries(paths.pose_cache_path, frame_idx),
            "ball_tracks": self.storage.load_frame_entries(paths.ball_cache_path, frame_idx),
        }

    def create_export(self, job_id: str, export_request: ExportRequest) -> dict[str, Any]:
        with self._lock:
            paths, manifest = self._load_paths_and_manifest(job_id)
            if manifest["status"] != JobStatus.COMPLETED.value:
                raise RuntimeError("Exports can only be created after a job completes.")

            export_id = uuid4().hex[:12]
            created_at = utc_now()
            export_record = {
                "export_id": export_id,
                "status": JobStatus.QUEUED.value,
                "created_at": created_at,
                "updated_at": created_at,
                "error": None,
                **export_request.to_dict(),
                "artifact": None,
            }
            manifest.setdefault("exports", {})[export_id] = export_record
            self.storage.save_manifest(paths, manifest)

        self.executor.submit(self._run_export_job, job_id, export_id)
        return export_record

    def get_export(self, job_id: str, export_id: str) -> dict[str, Any]:
        paths, manifest = self._load_paths_and_manifest(job_id)
        export_record = self._require_export(manifest, export_id)
        artifact = export_record.get("artifact")
        if artifact:
            export_record = dict(export_record)
            export_record["artifact"] = self.storage.artifact_record(
                paths,
                artifact["artifact_id"],
                ArtifactType.EXPORT_VIDEO,
                self.storage.resolve_artifact_path(paths, artifact["relative_path"]),
            )
        return export_record

    def get_artifact_path(self, job_id: str, artifact_id: str) -> Path:
        paths, manifest = self._load_paths_and_manifest(job_id)
        artifacts = self._collect_artifacts(paths, manifest)
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            raise KeyError(f"Artifact {artifact_id} was not found.")
        return self.storage.resolve_artifact_path(paths, artifact["relative_path"])

    def get_export_artifact_path(self, job_id: str, export_id: str) -> Path:
        paths, manifest = self._load_paths_and_manifest(job_id)
        export_record = self._require_export(manifest, export_id)
        artifact = export_record.get("artifact")
        if artifact is None:
            raise RuntimeError("Export output is not available yet.")
        return self.storage.resolve_artifact_path(paths, artifact["relative_path"])

    def _run_processing_job(self, job_id: str) -> None:
        try:
            with self._lock:
                paths, manifest = self._load_paths_and_manifest(job_id)
                manifest["status"] = JobStatus.RUNNING.value
                manifest["updated_at"] = utc_now()
                self.storage.save_manifest(paths, manifest)

            options = ProcessingOptions.from_dict(manifest["options"])
            result = self.backend.process_job(paths, options)

            with self._lock:
                paths, manifest = self._load_paths_and_manifest(job_id)
                manifest["status"] = JobStatus.COMPLETED.value
                manifest["updated_at"] = utc_now()
                manifest["video_info"] = result["video_info"]
                manifest["error"] = None
                manifest["artifacts"] = self._collect_base_artifacts(paths)
                manifest["artifacts"].update(self._collect_processing_artifacts(paths))
                self.storage.save_manifest(paths, manifest)
        except Exception as exc:
            self._mark_job_failed(job_id, exc)

    def _run_export_job(self, job_id: str, export_id: str) -> None:
        try:
            with self._lock:
                paths, manifest = self._load_paths_and_manifest(job_id)
                export_record = self._require_export(manifest, export_id)
                export_record["status"] = JobStatus.RUNNING.value
                export_record["updated_at"] = utc_now()
                self.storage.save_manifest(paths, manifest)

            options = ProcessingOptions.from_dict(manifest["options"])
            export_request = ExportRequest.from_dict(export_record)
            output_path = self.backend.export_job(
                paths=paths,
                options=options,
                export_request=export_request,
                export_id=export_id,
                video_info=manifest["video_info"],
            )

            with self._lock:
                paths, manifest = self._load_paths_and_manifest(job_id)
                export_record = self._require_export(manifest, export_id)
                export_record["status"] = JobStatus.COMPLETED.value
                export_record["updated_at"] = utc_now()
                export_record["artifact"] = self.storage.artifact_record(
                    paths,
                    f"export_{export_id}",
                    ArtifactType.EXPORT_VIDEO,
                    output_path,
                )
                self.storage.save_manifest(paths, manifest)
        except Exception as exc:
            with self._lock:
                paths, manifest = self._load_paths_and_manifest(job_id)
                export_record = self._require_export(manifest, export_id)
                export_record["status"] = JobStatus.FAILED.value
                export_record["updated_at"] = utc_now()
                export_record["error"] = str(exc)
                self.storage.save_manifest(paths, manifest)

    def _mark_job_failed(self, job_id: str, exc: Exception) -> None:
        with self._lock:
            paths, manifest = self._load_paths_and_manifest(job_id)
            manifest["status"] = JobStatus.FAILED.value
            manifest["updated_at"] = utc_now()
            manifest["error"] = str(exc)
            self.storage.save_manifest(paths, manifest)

    def _collect_base_artifacts(self, paths: JobPaths) -> dict[str, dict[str, Any]]:
        artifacts = {
            "input_video": self.storage.artifact_record(
                paths,
                "input_video",
                ArtifactType.INPUT_VIDEO,
                paths.video_path,
            ),
        }
        if paths.manifest_path.exists():
            artifacts["job_manifest"] = self.storage.artifact_record(
                paths,
                "job_manifest",
                ArtifactType.JOB_MANIFEST,
                paths.manifest_path,
            )
        return artifacts

    def _collect_processing_artifacts(self, paths: JobPaths) -> dict[str, dict[str, Any]]:
        artifacts: dict[str, dict[str, Any]] = {}
        if paths.track_cache_path.exists():
            artifacts["track_cache"] = self.storage.artifact_record(
                paths,
                "track_cache",
                ArtifactType.TRACK_CACHE,
                paths.track_cache_path,
            )
        if paths.pose_cache_path.exists():
            artifacts["pose_cache"] = self.storage.artifact_record(
                paths,
                "pose_cache",
                ArtifactType.POSE_CACHE,
                paths.pose_cache_path,
            )
        if paths.ball_cache_path.exists():
            artifacts["ball_cache"] = self.storage.artifact_record(
                paths,
                "ball_cache",
                ArtifactType.BALL_CACHE,
                paths.ball_cache_path,
            )
        return artifacts

    def _collect_artifacts(
        self,
        paths: JobPaths,
        manifest: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        artifacts = dict(manifest.get("artifacts", {}))
        for export_id, export_record in manifest.get("exports", {}).items():
            artifact = export_record.get("artifact")
            if artifact is None:
                continue
            artifact_path = self.storage.resolve_artifact_path(paths, artifact["relative_path"])
            artifacts[f"export_{export_id}"] = self.storage.artifact_record(
                paths,
                artifact["artifact_id"],
                ArtifactType.EXPORT_VIDEO,
                artifact_path,
            )
        return artifacts

    def _try_reuse_processed_artifacts(self, paths: JobPaths, manifest: dict[str, Any]) -> bool:
        reusable = self._find_reusable_job(
            job_id=manifest["job_id"],
            input_sha256=str(manifest.get("input_sha256", "")),
            options=manifest["options"],
        )
        if reusable is None:
            return False

        source_paths, source_manifest = reusable
        for source_path, destination_path in (
            (source_paths.track_cache_path, paths.track_cache_path),
            (source_paths.pose_cache_path, paths.pose_cache_path),
            (source_paths.ball_cache_path, paths.ball_cache_path),
        ):
            if source_path.exists():
                shutil.copy2(source_path, destination_path)

        manifest["status"] = JobStatus.COMPLETED.value
        manifest["updated_at"] = utc_now()
        manifest["error"] = None
        manifest["video_info"] = source_manifest.get("video_info")
        manifest["artifacts"] = self._collect_base_artifacts(paths)
        manifest["artifacts"].update(self._collect_processing_artifacts(paths))
        self.storage.save_manifest(paths, manifest)
        return True

    def _find_reusable_job(
        self,
        job_id: str,
        input_sha256: str,
        options: dict[str, Any],
    ) -> tuple[JobPaths, dict[str, Any]] | None:
        if not input_sha256:
            return None
        for job_dir in self.storage.root_dir.iterdir():
            if not job_dir.is_dir() or job_dir.name == job_id:
                continue
            manifest_path = job_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            with manifest_path.open("r", encoding="utf-8") as manifest_file:
                candidate = json.load(manifest_file)
            if candidate.get("status") != JobStatus.COMPLETED.value:
                continue
            if candidate.get("input_sha256") != input_sha256:
                continue
            if candidate.get("options") != options:
                continue
            if candidate.get("video_info") is None:
                continue
            source_paths = self.storage.build_paths(
                str(candidate["job_id"]),
                str(candidate.get("stored_video_name", "input.mp4")),
            )
            if not (
                source_paths.track_cache_path.exists()
                and source_paths.pose_cache_path.exists()
                and source_paths.ball_cache_path.exists()
            ):
                continue
            return source_paths, candidate
        return None

    def _load_paths_and_manifest(self, job_id: str) -> tuple[JobPaths, dict[str, Any]]:
        root_dir = self.storage.root_dir / job_id
        if not root_dir.exists():
            raise KeyError(f"Job {job_id} was not found.")
        manifest_path = root_dir / "manifest.json"
        if not manifest_path.exists():
            raise KeyError(f"Job {job_id} manifest is missing.")
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        stored_video_name = str(manifest.get("stored_video_name", "input.mp4"))
        paths = self.storage.build_paths(job_id, stored_video_name)
        return paths, manifest

    def _require_export(self, manifest: dict[str, Any], export_id: str) -> dict[str, Any]:
        export_record = manifest.get("exports", {}).get(export_id)
        if export_record is None:
            raise KeyError(f"Export {export_id} was not found.")
        return export_record


def validate_processing_options(options: ProcessingOptions) -> None:
    if not 0.0 <= options.conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be between 0.0 and 1.0.")
    if not 0.0 < options.center_alpha <= 1.0:
        raise ValueError("center_alpha must be in the range (0.0, 1.0].")
    if not 0.0 < options.size_alpha <= 1.0:
        raise ValueError("size_alpha must be in the range (0.0, 1.0].")
    if options.lost_buffer < 0:
        raise ValueError("lost_buffer must be 0 or greater.")


def validate_export_request(export_request: ExportRequest) -> None:
    if export_request.mode in {OverlayMode.SINGLE, OverlayMode.SPOTLIGHT, OverlayMode.POSE, OverlayMode.ZOOM}:
        if export_request.selected_track_id is None:
            raise ValueError(f"{export_request.mode.value} exports require selected_track_id.")
    if export_request.mode in {OverlayMode.MULTI_LINK, OverlayMode.AREA}:
        minimum = 2 if export_request.mode == OverlayMode.MULTI_LINK else 3
        if len(export_request.multi_track_ids) < minimum:
            raise ValueError(f"{export_request.mode.value} exports require {minimum}+ multi_track_ids.")
    if export_request.mode == OverlayMode.TEXT and not export_request.text_labels:
        raise ValueError("player_text exports require at least one text label.")
    if export_request.mode == OverlayMode.BALL and export_request.selected_ball_track_id is None:
        raise ValueError("ball_fire exports require selected_ball_track_id.")
