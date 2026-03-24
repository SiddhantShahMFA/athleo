from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .desktop import (
    MODE_ALL,
    MODE_AREA,
    MODE_BALL,
    MODE_PAIR,
    MODE_POSE,
    MODE_SINGLE,
    MODE_SPOTLIGHT,
    MODE_TEXT,
    MODE_ZOOM,
)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactType(str, Enum):
    INPUT_VIDEO = "input_video"
    TRACK_CACHE = "track_cache"
    POSE_CACHE = "pose_cache"
    BALL_CACHE = "ball_cache"
    EXPORT_VIDEO = "export_video"
    JOB_MANIFEST = "job_manifest"


class OverlayMode(str, Enum):
    ALL = MODE_ALL
    SINGLE = MODE_SINGLE
    SPOTLIGHT = MODE_SPOTLIGHT
    MULTI_LINK = MODE_PAIR
    TEXT = MODE_TEXT
    AREA = MODE_AREA
    POSE = MODE_POSE
    ZOOM = MODE_ZOOM
    BALL = MODE_BALL


@dataclass(frozen=True)
class ProcessingOptions:
    model_path: str = "yolo26x.pt"
    pose_model_path: str = "yolo26x-pose.pt"
    conf_threshold: float = 0.35
    center_alpha: float = 0.25
    size_alpha: float = 0.15
    lost_buffer: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingOptions":
        return cls(
            model_path=str(data.get("model_path", "yolo26x.pt")),
            pose_model_path=str(data.get("pose_model_path", "yolo26x-pose.pt")),
            conf_threshold=float(data.get("conf_threshold", 0.35)),
            center_alpha=float(data.get("center_alpha", 0.25)),
            size_alpha=float(data.get("size_alpha", 0.15)),
            lost_buffer=int(data.get("lost_buffer", 10)),
        )


@dataclass(frozen=True)
class ExportRequest:
    mode: OverlayMode
    selected_track_id: int | None = None
    multi_track_ids: list[int] = field(default_factory=list)
    text_labels: dict[int, str] = field(default_factory=dict)
    selected_ball_track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["text_labels"] = {str(track_id): label for track_id, label in self.text_labels.items()}
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportRequest":
        text_labels = {
            int(track_id): str(label)
            for track_id, label in dict(data.get("text_labels", {})).items()
        }
        return cls(
            mode=OverlayMode(str(data["mode"])),
            selected_track_id=(
                int(data["selected_track_id"])
                if data.get("selected_track_id") is not None
                else None
            ),
            multi_track_ids=[int(track_id) for track_id in data.get("multi_track_ids", [])],
            text_labels=text_labels,
            selected_ball_track_id=(
                int(data["selected_ball_track_id"])
                if data.get("selected_ball_track_id") is not None
                else None
            ),
        )
