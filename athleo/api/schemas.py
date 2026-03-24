from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from athleo.models import JobStatus, OverlayMode


class ProcessingOptionsModel(BaseModel):
    model_path: str = "yolo26x.pt"
    pose_model_path: str = "yolo26x-pose.pt"
    conf_threshold: float = 0.35
    center_alpha: float = 0.25
    size_alpha: float = 0.15
    lost_buffer: int = 10


class ArtifactModel(BaseModel):
    artifact_id: str
    artifact_type: str
    file_name: str
    content_type: str
    size_bytes: int
    relative_path: str


class ExportRequestModel(BaseModel):
    mode: OverlayMode
    selected_track_id: int | None = None
    multi_track_ids: list[int] = Field(default_factory=list)
    text_labels: dict[int, str] = Field(default_factory=dict)
    selected_ball_track_id: int | None = None


class ExportStatusModel(BaseModel):
    export_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    error: str | None = None
    mode: OverlayMode
    selected_track_id: int | None = None
    multi_track_ids: list[int] = Field(default_factory=list)
    text_labels: dict[str, str] = Field(default_factory=dict)
    selected_ball_track_id: int | None = None
    artifact: ArtifactModel | None = None


class JobStatusModel(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str
    error: str | None = None
    options: ProcessingOptionsModel
    video_info: dict[str, Any] | None = None
    artifacts: dict[str, ArtifactModel] = Field(default_factory=dict)
    exports: dict[str, ExportStatusModel] = Field(default_factory=dict)


class JobCreatedModel(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    updated_at: str


class ArtifactListModel(BaseModel):
    artifacts: list[ArtifactModel]


class FrameMetadataModel(BaseModel):
    job_id: str
    frame_idx: int
    tracks: list[dict[str, Any]] = Field(default_factory=list)
    pose_entries: list[dict[str, Any]] = Field(default_factory=list)
    ball_tracks: list[dict[str, Any]] = Field(default_factory=list)
