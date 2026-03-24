from __future__ import annotations

import json
import mimetypes
import re
import shutil
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .models import ArtifactType


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    root_dir: Path
    input_dir: Path
    cache_dir: Path
    export_dir: Path
    manifest_path: Path
    video_path: Path
    track_cache_path: Path
    pose_cache_path: Path
    ball_cache_path: Path


class JobStorage:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def build_paths(self, job_id: str, source_filename: str) -> JobPaths:
        safe_name = self._sanitize_filename(source_filename or "input.mp4")
        root_dir = self.root_dir / job_id
        input_dir = root_dir / "input"
        cache_dir = root_dir / "cache"
        export_dir = root_dir / "exports"
        return JobPaths(
            job_id=job_id,
            root_dir=root_dir,
            input_dir=input_dir,
            cache_dir=cache_dir,
            export_dir=export_dir,
            manifest_path=root_dir / "manifest.json",
            video_path=input_dir / safe_name,
            track_cache_path=cache_dir / "tracks.json",
            pose_cache_path=cache_dir / "pose.json",
            ball_cache_path=cache_dir / "balls.json",
        )

    def initialize_job(self, paths: JobPaths) -> None:
        paths.input_dir.mkdir(parents=True, exist_ok=True)
        paths.cache_dir.mkdir(parents=True, exist_ok=True)
        paths.export_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, source: BinaryIO, destination: Path) -> str:
        digest = hashlib.sha256()
        with destination.open("wb") as output_file:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
                digest.update(chunk)
        return digest.hexdigest()

    def load_manifest(self, paths: JobPaths) -> dict[str, Any]:
        with paths.manifest_path.open("r", encoding="utf-8") as manifest_file:
            return json.load(manifest_file)

    def save_manifest(self, paths: JobPaths, manifest: dict[str, Any]) -> None:
        temp_path = paths.manifest_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        temp_path.replace(paths.manifest_path)

    def load_frame_entries(self, cache_path: Path, frame_idx: int) -> list[dict[str, Any]]:
        if not cache_path.exists():
            return []
        with cache_path.open("r", encoding="utf-8") as cache_file:
            cache_data = json.load(cache_file)
        raw_frames = cache_data.get("frames", {})
        return list(raw_frames.get(str(frame_idx), raw_frames.get(frame_idx, [])))

    def artifact_record(
        self,
        paths: JobPaths,
        artifact_id: str,
        artifact_type: ArtifactType,
        file_path: Path,
    ) -> dict[str, Any]:
        resolved_root = paths.root_dir.resolve()
        resolved_file = file_path.resolve()
        content_type = mimetypes.guess_type(file_path.name)[0] or self._default_content_type(artifact_type)
        return {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type.value,
            "file_name": file_path.name,
            "content_type": content_type,
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "relative_path": str(resolved_file.relative_to(resolved_root)),
        }

    def resolve_artifact_path(self, paths: JobPaths, relative_path: str) -> Path:
        return (paths.root_dir / relative_path).resolve()

    def _default_content_type(self, artifact_type: ArtifactType) -> str:
        if artifact_type == ArtifactType.EXPORT_VIDEO or artifact_type == ArtifactType.INPUT_VIDEO:
            return "video/mp4"
        return "application/json"

    def _sanitize_filename(self, filename: str) -> str:
        cleaned = Path(filename).name.strip() or "input.mp4"
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
        if "." not in cleaned:
            cleaned = f"{cleaned}.mp4"
        return cleaned
