from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


WINDOW_NAME = "Athleo Interactive Player"
TRACKER_NAME = "bytetrack.yaml"
CACHE_VERSION = 2
POSE_CACHE_VERSION = 1
BALL_CACHE_VERSION = 1
DEFAULT_FPS = 30.0
LEFT_ARROW_KEYS = {81, 2424832, 65361}
RIGHT_ARROW_KEYS = {83, 2555904, 65363}
SPACE_KEYS = {32}
ESC_KEYS = {27}
MODE_ALL = "all"
MODE_SINGLE = "single_box"
MODE_SPOTLIGHT = "spotlight"
MODE_PAIR = "pair_link"
MODE_TEXT = "player_text"
MODE_AREA = "polygon_area"
MODE_POSE = "pose_skeleton"
MODE_ZOOM = "zoom_follow"
MODE_BALL = "ball_fire"
SPOTLIGHT_COLOR = (255, 255, 0)
PAIR_LINK_COLOR = (30, 30, 210)
PAIR_SELECTED_BOX_COLOR = (0, 235, 255)
PAIR_RING_COLOR = (40, 40, 220)
PAIR_RING_ACCENT_COLOR = (90, 120, 255)
PAIR_RING_HIGHLIGHT_COLOR = (210, 235, 255)
PAIR_RING_SHADOW_COLOR = (20, 20, 120)
AREA_FILL_COLOR = (40, 70, 220)
AREA_OUTLINE_COLOR = (100, 150, 255)
POSE_SKELETON_COLOR = (80, 255, 120)
POSE_JOINT_COLOR = (20, 220, 255)
BALL_MARKER_COLOR = (0, 215, 255)
BALL_MARKER_SELECTED_COLOR = (0, 140, 255)
FIREBALL_CORE_COLOR = (235, 250, 255)
FIREBALL_HOT_COLOR = (170, 245, 255)
FIREBALL_WARM_COLOR = (60, 210, 255)
FIREBALL_TRAIL_COLOR = (0, 150, 255)
FIREBALL_OUTER_COLOR = (40, 90, 255)
POSE_SKELETON_EDGES = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)
POSE_KEYPOINT_CONFIDENCE_THRESHOLD = 0.2
ZOOM_HISTORY_LENGTH = 8
ZOOM_SCALE_PADDING = 2.8
ZOOM_MIN_WIDTH_RATIO = 0.18
ZOOM_MIN_HEIGHT_RATIO = 0.34
BALL_TRAIL_HISTORY = 10


@dataclass(frozen=True)
class AppConfig:
    video_path: Path
    model_path: Path
    pose_model_path: Path
    conf_threshold: float
    center_alpha: float
    size_alpha: float
    lost_buffer: int
    cache_path: Path
    pose_cache_path: Path
    ball_cache_path: Path
    all_output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track soccer players once with YOLO + ByteTrack, cache the smoothed metadata, "
            "and open an interactive OpenCV playback window."
        )
    )
    parser.add_argument("--video",default="Trial.mp4",help="Input video path.")
    parser.add_argument(
        "--model",
        default="yolo26x.pt",
        help="Ultralytics model weights path. Defaults to ./yolo26x.pt",
    )
    parser.add_argument(
        "--pose-model",
        default="yolo26x-pose.pt",
        help="Ultralytics pose model weights path. Defaults to ./yolo26x-pose.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for player/person detections.",
    )
    parser.add_argument(
        "--center-alpha",
        type=float,
        default=0.25,
        help="EMA alpha for box center smoothing.",
    )
    parser.add_argument(
        "--size-alpha",
        type=float,
        default=0.15,
        help="EMA alpha for box size smoothing.",
    )
    parser.add_argument(
        "--lost-buffer",
        type=int,
        default=10,
        help="Number of missing frames to keep the last smoothed box alive.",
    )
    parser.add_argument(
        "--cache",
        help="Optional JSON cache path. Defaults to <input_stem>.tracks.json beside the video.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> AppConfig:
    video_text = args.video.strip() if args.video else input("Enter input video path: ").strip()
    if not video_text:
        raise ValueError("An input video path is required.")

    video_path = Path(video_text).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file was not found: {video_path}")

    model_path = Path(args.model).expanduser()
    pose_model_path = Path(args.pose_model).expanduser()
    if args.conf < 0.0 or args.conf > 1.0:
        raise ValueError("--conf must be between 0.0 and 1.0.")
    if args.center_alpha <= 0.0 or args.center_alpha > 1.0:
        raise ValueError("--center-alpha must be in the range (0.0, 1.0].")
    if args.size_alpha <= 0.0 or args.size_alpha > 1.0:
        raise ValueError("--size-alpha must be in the range (0.0, 1.0].")
    if args.lost_buffer < 0:
        raise ValueError("--lost-buffer must be 0 or greater.")

    cache_path = Path(args.cache).expanduser() if args.cache else video_path.with_name(
        f"{video_path.stem}.tracks.json"
    )
    pose_cache_path = video_path.with_name(f"{video_path.stem}.pose.json")
    ball_cache_path = video_path.with_name(f"{video_path.stem}.balls.json")

    return AppConfig(
        video_path=video_path.resolve(),
        model_path=model_path.resolve(),
        pose_model_path=pose_model_path.resolve(),
        conf_threshold=float(args.conf),
        center_alpha=float(args.center_alpha),
        size_alpha=float(args.size_alpha),
        lost_buffer=int(args.lost_buffer),
        cache_path=cache_path.resolve(),
        pose_cache_path=pose_cache_path.resolve(),
        ball_cache_path=ball_cache_path.resolve(),
        all_output_path=video_path.resolve().with_name("annotated_all.mp4"),
    )


def selected_output_path_for(video_path: Path, track_id: int) -> Path:
    return video_path.with_name(f"annotated_selected_{track_id}.mp4")


def spotlight_output_path_for(video_path: Path, track_id: int) -> Path:
    return video_path.with_name(f"annotated_spotlight_{track_id}.mp4")


def selection_suffix(track_ids: list[int]) -> str:
    return "_".join(str(track_id) for track_id in track_ids)


def pair_output_path_for(video_path: Path, track_ids: list[int]) -> Path:
    return video_path.with_name(f"annotated_pair_{selection_suffix(track_ids)}.mp4")


def text_output_path_for(video_path: Path) -> Path:
    return video_path.with_name("annotated_text.mp4")


def area_output_path_for(video_path: Path, track_ids: list[int]) -> Path:
    return video_path.with_name(f"annotated_area_{selection_suffix(track_ids)}.mp4")


def pose_output_path_for(video_path: Path, track_id: int) -> Path:
    return video_path.with_name(f"annotated_pose_{track_id}.mp4")


def zoom_output_path_for(video_path: Path, track_id: int) -> Path:
    return video_path.with_name(f"annotated_zoom_{track_id}.mp4")


def fireball_output_path_for(video_path: Path, track_id: int) -> Path:
    return video_path.with_name(f"annotated_fireball_{track_id}.mp4")


def load_video_info(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Failed to read video dimensions from: {video_path}")
    if total_frames <= 0:
        raise RuntimeError(f"Failed to read frame count from: {video_path}")
    if fps <= 0:
        print(f"Video FPS metadata was invalid for {video_path.name}; falling back to {DEFAULT_FPS}.")
        fps = DEFAULT_FPS

    stat = video_path.stat()
    return {
        "path": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def video_meta_matches(
    cache_data: dict[str, Any],
    video_path: Path,
    video_info: dict[str, Any],
) -> bool:
    video_meta = cache_data.get("video", {})
    return (
        video_meta.get("path") == str(video_path)
        and int(video_meta.get("width", -1)) == int(video_info["width"])
        and int(video_meta.get("height", -1)) == int(video_info["height"])
        and int(video_meta.get("total_frames", -1)) == int(video_info["total_frames"])
        and int(video_meta.get("file_size", -1)) == int(video_info["file_size"])
        and int(video_meta.get("mtime_ns", -1)) == int(video_info["mtime_ns"])
        and round(float(video_meta.get("fps", -1.0)), 3) == round(float(video_info["fps"]), 3)
    )


def cache_matches(cache_data: dict[str, Any], config: AppConfig, video_info: dict[str, Any]) -> bool:
    if int(cache_data.get("cache_version", -1)) != CACHE_VERSION:
        return False

    tracking_meta = cache_data.get("tracking", {})
    expected_model_name = config.model_path.name

    return (
        video_meta_matches(cache_data, config.video_path, video_info)
        and tracking_meta.get("model_path") == str(config.model_path)
        and tracking_meta.get("model_name") == expected_model_name
        and tracking_meta.get("tracker") == TRACKER_NAME
        and float(tracking_meta.get("conf_threshold", -1.0)) == config.conf_threshold
        and float(tracking_meta.get("center_alpha", -1.0)) == config.center_alpha
        and float(tracking_meta.get("size_alpha", -1.0)) == config.size_alpha
        and int(tracking_meta.get("lost_buffer", -1)) == config.lost_buffer
    )


def ball_cache_matches(cache_data: dict[str, Any], config: AppConfig, video_info: dict[str, Any]) -> bool:
    if int(cache_data.get("cache_version", -1)) != BALL_CACHE_VERSION:
        return False

    tracking_meta = cache_data.get("tracking", {})
    return (
        video_meta_matches(cache_data, config.video_path, video_info)
        and tracking_meta.get("model_path") == str(config.model_path)
        and tracking_meta.get("model_name") == config.model_path.name
        and tracking_meta.get("tracker") == TRACKER_NAME
        and list(tracking_meta.get("classes", [])) == [32]
        and float(tracking_meta.get("conf_threshold", -1.0)) == config.conf_threshold
        and float(tracking_meta.get("center_alpha", -1.0)) == config.center_alpha
        and float(tracking_meta.get("size_alpha", -1.0)) == config.size_alpha
        and int(tracking_meta.get("lost_buffer", -1)) == config.lost_buffer
    )


def pose_cache_matches(cache_data: dict[str, Any], config: AppConfig, video_info: dict[str, Any]) -> bool:
    if int(cache_data.get("cache_version", -1)) != POSE_CACHE_VERSION:
        return False

    pose_meta = cache_data.get("pose", {})
    return (
        video_meta_matches(cache_data, config.video_path, video_info)
        and pose_meta.get("model_path") == str(config.pose_model_path)
        and pose_meta.get("model_name") == config.pose_model_path.name
        and float(pose_meta.get("conf_threshold", -1.0)) == config.conf_threshold
        and int(pose_meta.get("keypoint_count", -1)) == 17
    )


def load_frame_cache(
    cache_path: Path,
    label: str,
    matcher: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            cache_data = json.load(cache_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable cache file {cache_path}: {exc}")
        return None

    if not matcher(cache_data):
        print(f"Existing cache does not match current settings, rebuilding: {cache_path}")
        return None

    loaded_frames: dict[int, list[dict[str, Any]]] = {}
    raw_frames = cache_data.get("frames", {})
    for frame_idx_text, frame_tracks in raw_frames.items():
        try:
            frame_idx = int(frame_idx_text)
        except (TypeError, ValueError):
            continue
        loaded_frames[frame_idx] = list(frame_tracks)

    cache_data["frames"] = loaded_frames
    print(f"Loaded {label} cache from {cache_path}")
    return cache_data


def load_track_cache(config: AppConfig, video_info: dict[str, Any]) -> dict[str, Any] | None:
    return load_frame_cache(
        cache_path=config.cache_path,
        label="tracking",
        matcher=lambda cache_data: cache_matches(cache_data, config, video_info),
    )


def load_ball_cache(config: AppConfig, video_info: dict[str, Any]) -> dict[str, Any] | None:
    return load_frame_cache(
        cache_path=config.ball_cache_path,
        label="ball tracking",
        matcher=lambda cache_data: ball_cache_matches(cache_data, config, video_info),
    )


def load_pose_cache(config: AppConfig, video_info: dict[str, Any]) -> dict[str, Any] | None:
    return load_frame_cache(
        cache_path=config.pose_cache_path,
        label="pose",
        matcher=lambda cache_data: pose_cache_matches(cache_data, config, video_info),
    )


def save_frame_cache(cache_data: dict[str, Any], cache_path: Path) -> None:
    serializable_cache = dict(cache_data)
    serializable_cache["frames"] = {
        str(frame_idx): frame_tracks
        for frame_idx, frame_tracks in cache_data["frames"].items()
    }

    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(serializable_cache, cache_file, indent=2)


def save_track_cache(cache_data: dict[str, Any], cache_path: Path) -> None:
    save_frame_cache(cache_data, cache_path)


def build_frame_index(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    frame_index: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        frame_idx = int(record["frame_idx"])
        frame_index.setdefault(frame_idx, []).append(record)
    return frame_index


def smooth_box(
    previous_box: dict[str, float] | None,
    raw_box: dict[str, Any],
    center_alpha: float,
    size_alpha: float,
) -> dict[str, float]:
    x1 = float(raw_box["x1"])
    y1 = float(raw_box["y1"])
    x2 = float(raw_box["x2"])
    y2 = float(raw_box["y2"])

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    if previous_box is None:
        smoothed_cx = cx
        smoothed_cy = cy
        smoothed_width = width
        smoothed_height = height
    else:
        smoothed_cx = (center_alpha * cx) + ((1.0 - center_alpha) * previous_box["cx"])
        smoothed_cy = (center_alpha * cy) + ((1.0 - center_alpha) * previous_box["cy"])
        smoothed_width = (size_alpha * width) + ((1.0 - size_alpha) * previous_box["w"])
        smoothed_height = (size_alpha * height) + ((1.0 - size_alpha) * previous_box["h"])

    return {
        "cx": smoothed_cx,
        "cy": smoothed_cy,
        "w": smoothed_width,
        "h": smoothed_height,
        "x1": smoothed_cx - (smoothed_width / 2.0),
        "y1": smoothed_cy - (smoothed_height / 2.0),
        "x2": smoothed_cx + (smoothed_width / 2.0),
        "y2": smoothed_cy + (smoothed_height / 2.0),
    }


def smooth_tracks(
    raw_frame_index: dict[int, list[dict[str, Any]]],
    frame_count: int,
    center_alpha: float,
    size_alpha: float,
    lost_buffer: int,
) -> dict[int, list[dict[str, Any]]]:
    smoothed_by_frame: dict[int, list[dict[str, Any]]] = {}
    track_states: dict[int, dict[str, Any]] = {}

    for frame_idx in range(frame_count):
        frame_tracks: list[dict[str, Any]] = []
        detections = raw_frame_index.get(frame_idx, [])
        seen_track_ids: set[int] = set()

        for record in detections:
            track_id = int(record["track_id"])
            previous_box = None
            if track_id in track_states:
                previous_box = track_states[track_id]["smoothed_box"]

            smoothed = smooth_box(previous_box, record, center_alpha, size_alpha)
            track_states[track_id] = {
                "smoothed_box": smoothed,
                "last_seen_frame": frame_idx,
                "class_id": int(record["class_id"]),
                "confidence": float(record["confidence"]),
            }
            seen_track_ids.add(track_id)
            frame_tracks.append(
                {
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "class_id": int(record["class_id"]),
                    "confidence": float(record["confidence"]),
                    "x1": round(smoothed["x1"], 2),
                    "y1": round(smoothed["y1"], 2),
                    "x2": round(smoothed["x2"], 2),
                    "y2": round(smoothed["y2"], 2),
                    "is_lost_buffer": False,
                }
            )

        for track_id, state in list(track_states.items()):
            if track_id in seen_track_ids:
                continue

            missing_frames = frame_idx - int(state["last_seen_frame"])
            if missing_frames <= 0:
                continue
            if missing_frames > lost_buffer:
                del track_states[track_id]
                continue

            smoothed = state["smoothed_box"]
            frame_tracks.append(
                {
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "class_id": int(state["class_id"]),
                    "confidence": float(state["confidence"]),
                    "x1": round(smoothed["x1"], 2),
                    "y1": round(smoothed["y1"], 2),
                    "x2": round(smoothed["x2"], 2),
                    "y2": round(smoothed["y2"], 2),
                    "is_lost_buffer": True,
                }
            )

        if frame_tracks:
            smoothed_by_frame[frame_idx] = sorted(frame_tracks, key=lambda item: item["track_id"])

    return smoothed_by_frame


def box_iou(first_box: dict[str, Any], second_box: dict[str, Any]) -> float:
    ax1 = float(first_box["x1"])
    ay1 = float(first_box["y1"])
    ax2 = float(first_box["x2"])
    ay2 = float(first_box["y2"])
    bx1 = float(second_box["x1"])
    by1 = float(second_box["y1"])
    bx2 = float(second_box["x2"])
    by2 = float(second_box["y2"])

    overlap_x1 = max(ax1, bx1)
    overlap_y1 = max(ay1, by1)
    overlap_x2 = min(ax2, bx2)
    overlap_y2 = min(ay2, by2)
    overlap_width = max(0.0, overlap_x2 - overlap_x1)
    overlap_height = max(0.0, overlap_y2 - overlap_y1)
    intersection = overlap_width * overlap_height
    if intersection <= 0.0:
        return 0.0

    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def find_best_matching_track_id(
    detection_box: dict[str, Any],
    frame_tracks: list[dict[str, Any]],
    used_track_ids: set[int],
) -> int | None:
    best_track_id = None
    best_iou = 0.0
    for track in frame_tracks:
        track_id = int(track["track_id"])
        if track_id in used_track_ids:
            continue
        score = box_iou(detection_box, track)
        if score <= best_iou:
            continue
        best_iou = score
        best_track_id = track_id

    if best_iou < 0.1:
        return None
    return best_track_id


def load_yolo_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights file was not found: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install requirements.txt before running a new pass."
        ) from exc

    return YOLO(str(model_path))


def run_tracking(config: AppConfig, video_info: dict[str, Any]) -> dict[str, Any]:
    cached_data = load_track_cache(config, video_info)
    if cached_data is not None:
        return cached_data

    model = load_yolo_model(config.model_path)
    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for tracking: {config.video_path}")

    records: list[dict[str, Any]] = []
    total_frames = int(video_info["total_frames"])
    frame_idx = 0
    started_at = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=TRACKER_NAME,
                    conf=config.conf_threshold,
                    classes=[0],
                    verbose=False,
                )
            except ModuleNotFoundError as exc:
                if exc.name == "lap":
                    raise RuntimeError(
                        "ByteTrack requires the 'lap' package. Install requirements.txt in the "
                        "active environment before running tracking."
                    ) from exc
                raise

            result = results[0] if results else None
            if result is not None and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = (
                    boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(boxes))
                )
                class_ids = (
                    boxes.cls.cpu().numpy().astype(int)
                    if boxes.cls is not None
                    else np.zeros(len(boxes), dtype=int)
                )
                track_ids = (
                    boxes.id.cpu().numpy().astype(int).tolist()
                    if boxes.id is not None
                    else [None] * len(boxes)
                )

                for index, raw_box in enumerate(xyxy):
                    track_id = track_ids[index]
                    class_id = int(class_ids[index])
                    confidence = float(confidences[index])
                    if class_id != 0 or confidence < config.conf_threshold or track_id is None:
                        continue

                    x1, y1, x2, y2 = [float(value) for value in raw_box]
                    records.append(
                        {
                            "frame_idx": frame_idx,
                            "track_id": int(track_id),
                            "class_id": class_id,
                            "confidence": round(confidence, 6),
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        }
                    )

            if frame_idx % 30 == 0 or frame_idx + 1 == total_frames:
                elapsed = max(time.time() - started_at, 0.001)
                processed_frames = frame_idx + 1
                fps = processed_frames / elapsed
                percent = (processed_frames / total_frames) * 100.0
                print(
                    f"Tracking progress: {processed_frames}/{total_frames} "
                    f"frames ({percent:.1f}%) at {fps:.2f} fps"
                )

            frame_idx += 1
    finally:
        capture.release()

    raw_frame_index = build_frame_index(records)
    smoothed_frames = smooth_tracks(
        raw_frame_index=raw_frame_index,
        frame_count=total_frames,
        center_alpha=config.center_alpha,
        size_alpha=config.size_alpha,
        lost_buffer=config.lost_buffer,
    )

    cache_data = {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": str(config.video_path),
            "width": int(video_info["width"]),
            "height": int(video_info["height"]),
            "fps": float(video_info["fps"]),
            "total_frames": int(video_info["total_frames"]),
            "file_size": int(video_info["file_size"]),
            "mtime_ns": int(video_info["mtime_ns"]),
        },
        "tracking": {
            "model_name": config.model_path.name,
            "model_path": str(config.model_path),
            "tracker": TRACKER_NAME,
            "conf_threshold": config.conf_threshold,
            "center_alpha": config.center_alpha,
            "size_alpha": config.size_alpha,
            "lost_buffer": config.lost_buffer,
        },
        "frames": smoothed_frames,
    }
    save_track_cache(cache_data, config.cache_path)
    print(f"Saved tracking cache to {config.cache_path}")
    return cache_data


def run_ball_tracking(config: AppConfig, video_info: dict[str, Any]) -> dict[str, Any]:
    cached_data = load_ball_cache(config, video_info)
    if cached_data is not None:
        return cached_data

    model = load_yolo_model(config.model_path)
    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for ball tracking: {config.video_path}")

    records: list[dict[str, Any]] = []
    total_frames = int(video_info["total_frames"])
    frame_idx = 0
    started_at = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=TRACKER_NAME,
                    conf=config.conf_threshold,
                    classes=[32],
                    verbose=False,
                )
            except ModuleNotFoundError as exc:
                if exc.name == "lap":
                    raise RuntimeError(
                        "ByteTrack requires the 'lap' package. Install requirements.txt in the "
                        "active environment before running ball tracking."
                    ) from exc
                raise

            result = results[0] if results else None
            if result is not None and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = (
                    boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(boxes))
                )
                class_ids = (
                    boxes.cls.cpu().numpy().astype(int)
                    if boxes.cls is not None
                    else np.zeros(len(boxes), dtype=int)
                )
                track_ids = (
                    boxes.id.cpu().numpy().astype(int).tolist()
                    if boxes.id is not None
                    else [None] * len(boxes)
                )

                for index, raw_box in enumerate(xyxy):
                    track_id = track_ids[index]
                    class_id = int(class_ids[index])
                    confidence = float(confidences[index])
                    if class_id != 32 or confidence < config.conf_threshold or track_id is None:
                        continue

                    x1, y1, x2, y2 = [float(value) for value in raw_box]
                    records.append(
                        {
                            "frame_idx": frame_idx,
                            "track_id": int(track_id),
                            "class_id": class_id,
                            "confidence": round(confidence, 6),
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        }
                    )

            if frame_idx % 30 == 0 or frame_idx + 1 == total_frames:
                elapsed = max(time.time() - started_at, 0.001)
                processed_frames = frame_idx + 1
                fps = processed_frames / elapsed
                percent = (processed_frames / total_frames) * 100.0
                print(
                    f"Ball tracking progress: {processed_frames}/{total_frames} "
                    f"frames ({percent:.1f}%) at {fps:.2f} fps"
                )

            frame_idx += 1
    finally:
        capture.release()

    raw_frame_index = build_frame_index(records)
    smoothed_frames = smooth_tracks(
        raw_frame_index=raw_frame_index,
        frame_count=total_frames,
        center_alpha=config.center_alpha,
        size_alpha=config.size_alpha,
        lost_buffer=config.lost_buffer,
    )

    cache_data = {
        "cache_version": BALL_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": str(config.video_path),
            "width": int(video_info["width"]),
            "height": int(video_info["height"]),
            "fps": float(video_info["fps"]),
            "total_frames": int(video_info["total_frames"]),
            "file_size": int(video_info["file_size"]),
            "mtime_ns": int(video_info["mtime_ns"]),
        },
        "tracking": {
            "model_name": config.model_path.name,
            "model_path": str(config.model_path),
            "tracker": TRACKER_NAME,
            "classes": [32],
            "conf_threshold": config.conf_threshold,
            "center_alpha": config.center_alpha,
            "size_alpha": config.size_alpha,
            "lost_buffer": config.lost_buffer,
        },
        "frames": smoothed_frames,
    }
    save_frame_cache(cache_data, config.ball_cache_path)
    print(f"Saved ball tracking cache to {config.ball_cache_path}")
    return cache_data


def run_pose_estimation(
    config: AppConfig,
    video_info: dict[str, Any],
    player_frames_index: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    cached_data = load_pose_cache(config, video_info)
    if cached_data is not None:
        return cached_data

    model = load_yolo_model(config.pose_model_path)
    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for pose estimation: {config.video_path}")

    total_frames = int(video_info["total_frames"])
    pose_frames: dict[int, list[dict[str, Any]]] = {}
    frame_idx = 0
    started_at = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.predict(frame, conf=config.conf_threshold, verbose=False)
            result = results[0] if results else None
            frame_entries: list[dict[str, Any]] = []

            if (
                result is not None
                and result.boxes is not None
                and result.keypoints is not None
                and len(result.boxes) > 0
            ):
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = (
                    boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(boxes))
                )
                keypoints_xy = result.keypoints.xy.cpu().numpy()
                keypoints_conf = (
                    result.keypoints.conf.cpu().numpy()
                    if result.keypoints.conf is not None
                    else np.ones((len(boxes), keypoints_xy.shape[1]), dtype=float)
                )
                matched_tracks = player_frames_index.get(frame_idx, [])
                used_track_ids: set[int] = set()

                for index, raw_box in enumerate(xyxy):
                    confidence = float(confidences[index])
                    if confidence < config.conf_threshold:
                        continue

                    x1, y1, x2, y2 = [float(value) for value in raw_box]
                    pose_box = {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    }
                    matched_track_id = find_best_matching_track_id(
                        pose_box,
                        matched_tracks,
                        used_track_ids,
                    )
                    if matched_track_id is not None:
                        used_track_ids.add(matched_track_id)

                    points = [
                        [round(float(point[0]), 2), round(float(point[1]), 2)]
                        for point in keypoints_xy[index]
                    ]
                    point_confidences = [
                        round(float(point_confidence), 4)
                        for point_confidence in keypoints_conf[index]
                    ]
                    frame_entries.append(
                        {
                            "frame_idx": frame_idx,
                            "confidence": round(confidence, 6),
                            "matched_track_id": matched_track_id,
                            "x1": pose_box["x1"],
                            "y1": pose_box["y1"],
                            "x2": pose_box["x2"],
                            "y2": pose_box["y2"],
                            "keypoints": points,
                            "keypoint_confidences": point_confidences,
                        }
                    )

            if frame_entries:
                pose_frames[frame_idx] = frame_entries

            if frame_idx % 30 == 0 or frame_idx + 1 == total_frames:
                elapsed = max(time.time() - started_at, 0.001)
                processed_frames = frame_idx + 1
                fps = processed_frames / elapsed
                percent = (processed_frames / total_frames) * 100.0
                print(
                    f"Pose progress: {processed_frames}/{total_frames} "
                    f"frames ({percent:.1f}%) at {fps:.2f} fps"
                )

            frame_idx += 1
    finally:
        capture.release()

    cache_data = {
        "cache_version": POSE_CACHE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": str(config.video_path),
            "width": int(video_info["width"]),
            "height": int(video_info["height"]),
            "fps": float(video_info["fps"]),
            "total_frames": int(video_info["total_frames"]),
            "file_size": int(video_info["file_size"]),
            "mtime_ns": int(video_info["mtime_ns"]),
        },
        "pose": {
            "model_name": config.pose_model_path.name,
            "model_path": str(config.pose_model_path),
            "conf_threshold": config.conf_threshold,
            "keypoint_count": 17,
        },
        "frames": pose_frames,
    }
    save_frame_cache(cache_data, config.pose_cache_path)
    print(f"Saved pose cache to {config.pose_cache_path}")
    return cache_data


def color_for_track(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    red, green, blue = [int(channel) for channel in rng.integers(64, 256, size=3)]
    return blue, green, red


def clip_box_to_frame(
    frame_shape: tuple[int, int, int],
    box: dict[str, Any],
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1 = max(0, min(width - 1, int(round(float(box["x1"])))))
    y1 = max(0, min(height - 1, int(round(float(box["y1"])))))
    x2 = max(0, min(width - 1, int(round(float(box["x2"])))))
    y2 = max(0, min(height - 1, int(round(float(box["y2"])))))
    return x1, y1, x2, y2


def find_track_by_id(tracks: list[dict[str, Any]], track_id: int) -> dict[str, Any] | None:
    for track in tracks:
        if int(track["track_id"]) == track_id:
            return track
    return None


def draw_track_box(
    annotated: np.ndarray,
    track: dict[str, Any],
    color: tuple[int, int, int],
    thickness: int,
    show_track_label: bool = True,
) -> None:
    x1, y1, x2, y2 = clip_box_to_frame(annotated.shape, track)
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    if not show_track_label:
        return

    label = f"ID {int(track['track_id'])}"
    if bool(track.get("is_lost_buffer")):
        label += " (hold)"

    text_origin = (x1, max(25, y1 - 10))
    cv2.putText(
        annotated,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_spotlight_overlay(
    frame: np.ndarray,
    track: dict[str, Any],
) -> np.ndarray:
    annotated = frame.copy()
    x1, _y1, x2, y2 = clip_box_to_frame(annotated.shape, track)
    if x2 <= x1:
        return annotated

    center_x = int(round((x1 + x2) / 2))
    overlay = annotated.copy()
    beam_points = np.array(
        [
            [center_x, 0],
            [x1, y2],
            [x2, y2],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(overlay, beam_points, SPOTLIGHT_COLOR)
    cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0.0, annotated)
    cv2.polylines(annotated, [beam_points], True, SPOTLIGHT_COLOR, 2, cv2.LINE_AA)
    return annotated


def bottom_center_for_track(
    frame_shape: tuple[int, int, int],
    track: dict[str, Any],
) -> tuple[int, int]:
    x1, _y1, x2, y2 = clip_box_to_frame(frame_shape, track)
    return int(round((x1 + x2) / 2)), y2


def pair_ring_geometry(
    frame_shape: tuple[int, int, int],
    track: dict[str, Any],
) -> tuple[tuple[int, int], tuple[int, int]]:
    height, _width = frame_shape[:2]
    x1, y1, x2, y2 = clip_box_to_frame(frame_shape, track)
    if x2 <= x1:
        return (0, 0), (0, 0)

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    axes_x = max(18, int(round(box_width * 0.95)))
    axes_y = max(6, int(round(box_height * 0.08)))
    center_x = int(round((x1 + x2) / 2))
    center_y = min(height - axes_y - 1, y2 + max(4, axes_y // 2))
    return (center_x, center_y), (axes_x, axes_y)


def draw_pair_link(
    annotated: np.ndarray,
    first_track: dict[str, Any],
    second_track: dict[str, Any],
) -> None:
    first_point, first_axes = pair_ring_geometry(annotated.shape, first_track)
    second_point, second_axes = pair_ring_geometry(annotated.shape, second_track)
    if first_axes == (0, 0) or second_axes == (0, 0):
        return
    cv2.line(annotated, first_point, second_point, PAIR_LINK_COLOR, 5, cv2.LINE_AA)


def visible_selected_tracks(
    tracks: list[dict[str, Any]],
    selected_track_ids: list[int],
) -> list[dict[str, Any]]:
    visible_tracks: list[dict[str, Any]] = []
    for track_id in selected_track_ids:
        track = find_track_by_id(tracks, track_id)
        if track is None or bool(track.get("is_lost_buffer")):
            continue
        visible_tracks.append(track)
    return visible_tracks


def draw_multi_link(
    annotated: np.ndarray,
    selected_tracks: list[dict[str, Any]],
) -> None:
    for first_track, second_track in zip(selected_tracks, selected_tracks[1:]):
        draw_pair_link(annotated, first_track, second_track)


def draw_pair_ground_ring(
    annotated: np.ndarray,
    track: dict[str, Any],
) -> None:
    center, axes = pair_ring_geometry(annotated.shape, track)
    if axes == (0, 0):
        return

    shadow_center = (center[0], min(annotated.shape[0] - 1, center[1] + 2))
    inner_axes = (max(axes[0] - 6, 8), max(axes[1] - 2, 4))
    highlight_axes = (max(axes[0] - 9, 6), max(axes[1] - 3, 3))

    cv2.ellipse(
        annotated,
        shadow_center,
        axes,
        0,
        0,
        360,
        PAIR_RING_SHADOW_COLOR,
        9,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        annotated,
        center,
        axes,
        0,
        0,
        360,
        PAIR_RING_COLOR,
        7,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        annotated,
        center,
        inner_axes,
        0,
        0,
        360,
        PAIR_RING_ACCENT_COLOR,
        3,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        annotated,
        (center[0], max(0, center[1] - 1)),
        highlight_axes,
        0,
        200,
        340,
        PAIR_RING_HIGHLIGHT_COLOR,
        2,
        cv2.LINE_AA,
    )


def center_for_track(
    frame_shape: tuple[int, int, int],
    track: dict[str, Any],
) -> tuple[int, int]:
    x1, y1, x2, y2 = clip_box_to_frame(frame_shape, track)
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


def draw_area_polygon(
    annotated: np.ndarray,
    selected_tracks: list[dict[str, Any]],
) -> None:
    polygon_points: list[tuple[int, int]] = []
    for track in selected_tracks:
        point, axes = pair_ring_geometry(annotated.shape, track)
        if axes == (0, 0):
            continue
        polygon_points.append(point)

    if len(polygon_points) < 3:
        return

    polygon = np.array(polygon_points, dtype=np.int32)
    overlay = annotated.copy()
    cv2.fillPoly(overlay, [polygon], AREA_FILL_COLOR)
    cv2.addWeighted(overlay, 0.24, annotated, 0.76, 0.0, annotated)
    cv2.polylines(annotated, [polygon], True, AREA_OUTLINE_COLOR, 4, cv2.LINE_AA)


def draw_track_text_label(
    annotated: np.ndarray,
    track: dict[str, Any],
    text: str,
) -> None:
    if not text:
        return

    frame_height, frame_width = annotated.shape[:2]
    x1, y1, _x2, _y2 = clip_box_to_frame(annotated.shape, track)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2
    padding_x = 6
    padding_y = 5
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    box_width = text_width + (padding_x * 2)
    box_height = text_height + baseline + (padding_y * 2)

    label_x1 = max(0, min(frame_width - box_width - 1, x1))
    label_y2 = y1 - 6
    label_y1 = label_y2 - box_height
    if label_y1 < 0:
        label_y1 = min(frame_height - box_height - 1, y1 + 6)
        label_y2 = label_y1 + box_height

    text_origin = (
        label_x1 + padding_x,
        label_y2 - baseline - padding_y,
    )
    cv2.rectangle(
        annotated,
        (label_x1, label_y1),
        (label_x1 + box_width, label_y2),
        (12, 18, 28),
        -1,
    )
    cv2.rectangle(
        annotated,
        (label_x1, label_y1),
        (label_x1 + box_width, label_y2),
        PAIR_SELECTED_BOX_COLOR,
        2,
    )
    cv2.putText(
        annotated,
        text,
        text_origin,
        font,
        font_scale,
        PAIR_SELECTED_BOX_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def find_pose_by_track_id(
    pose_entries: list[dict[str, Any]],
    track_id: int,
) -> dict[str, Any] | None:
    for pose_entry in pose_entries:
        if pose_entry.get("matched_track_id") == track_id:
            return pose_entry
    return None


def draw_pose_overlay(
    frame: np.ndarray,
    pose_entry: dict[str, Any],
) -> np.ndarray:
    annotated = frame.copy()
    overlay = annotated.copy()
    points = pose_entry.get("keypoints", [])
    confidences = pose_entry.get("keypoint_confidences", [])

    for start_index, end_index in POSE_SKELETON_EDGES:
        if start_index >= len(points) or end_index >= len(points):
            continue
        if (
            start_index >= len(confidences)
            or end_index >= len(confidences)
            or float(confidences[start_index]) < POSE_KEYPOINT_CONFIDENCE_THRESHOLD
            or float(confidences[end_index]) < POSE_KEYPOINT_CONFIDENCE_THRESHOLD
        ):
            continue

        start_point = (int(round(points[start_index][0])), int(round(points[start_index][1])))
        end_point = (int(round(points[end_index][0])), int(round(points[end_index][1])))
        cv2.line(overlay, start_point, end_point, POSE_SKELETON_COLOR, 4, cv2.LINE_AA)

    for index, point in enumerate(points):
        if index >= len(confidences) or float(confidences[index]) < POSE_KEYPOINT_CONFIDENCE_THRESHOLD:
            continue
        point_xy = (int(round(point[0])), int(round(point[1])))
        cv2.circle(overlay, point_xy, 6, POSE_JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(overlay, point_xy, 9, POSE_SKELETON_COLOR, 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0.0, annotated)
    return annotated


def smoothed_track_window(
    frame_shape: tuple[int, int, int],
    frames_index: dict[int, list[dict[str, Any]]],
    frame_idx: int,
    selected_track_id: int,
    history_length: int,
) -> tuple[float, float, float, float] | None:
    weighted_center_x = 0.0
    weighted_center_y = 0.0
    weighted_width = 0.0
    weighted_height = 0.0
    total_weight = 0.0
    weight = 1.0

    for past_frame_idx in range(frame_idx, max(-1, frame_idx - history_length), -1):
        track = find_track_by_id(frames_index.get(past_frame_idx, []), selected_track_id)
        if track is None:
            continue

        center_x, center_y = center_for_track(frame_shape, track)
        x1, y1, x2, y2 = clip_box_to_frame(frame_shape, track)
        track_width = max(1.0, float(x2 - x1))
        track_height = max(1.0, float(y2 - y1))

        weighted_center_x += center_x * weight
        weighted_center_y += center_y * weight
        weighted_width += track_width * weight
        weighted_height += track_height * weight
        total_weight += weight
        weight *= 0.82

    if total_weight <= 0.0:
        return None

    return (
        weighted_center_x / total_weight,
        weighted_center_y / total_weight,
        weighted_width / total_weight,
        weighted_height / total_weight,
    )


def zoom_crop_bounds(
    frame_shape: tuple[int, int, int],
    frames_index: dict[int, list[dict[str, Any]]],
    frame_idx: int,
    selected_track_id: int,
) -> tuple[int, int, int, int] | None:
    history_window = smoothed_track_window(
        frame_shape,
        frames_index,
        frame_idx,
        selected_track_id,
        ZOOM_HISTORY_LENGTH,
    )
    if history_window is None:
        return None

    frame_height, frame_width = frame_shape[:2]
    center_x, center_y, track_width, track_height = history_window
    aspect_ratio = frame_width / max(frame_height, 1)

    crop_width = max(frame_width * ZOOM_MIN_WIDTH_RATIO, track_width * ZOOM_SCALE_PADDING)
    crop_height = max(frame_height * ZOOM_MIN_HEIGHT_RATIO, track_height * ZOOM_SCALE_PADDING)

    if crop_width / crop_height < aspect_ratio:
        crop_width = crop_height * aspect_ratio
    else:
        crop_height = crop_width / aspect_ratio

    crop_width = min(float(frame_width), crop_width)
    crop_height = min(float(frame_height), crop_height)

    half_width = crop_width / 2.0
    half_height = crop_height / 2.0
    left = max(0.0, min(frame_width - crop_width, center_x - half_width))
    top = max(0.0, min(frame_height - crop_height, center_y - half_height))
    right = left + crop_width
    bottom = top + crop_height
    return (
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    )


def draw_zoom_overlay(
    frame: np.ndarray,
    current_tracks: list[dict[str, Any]],
    frames_index: dict[int, list[dict[str, Any]]],
    frame_idx: int,
    selected_track_id: int,
) -> np.ndarray:
    crop_bounds = zoom_crop_bounds(frame.shape, frames_index, frame_idx, selected_track_id)
    if crop_bounds is None:
        return frame.copy()

    annotated = frame.copy()
    current_track = find_track_by_id(current_tracks, selected_track_id)
    if current_track is not None:
        draw_track_box(annotated, current_track, PAIR_SELECTED_BOX_COLOR, 4)

    left, top, right, bottom = crop_bounds
    crop = annotated[top:bottom, left:right]
    if crop.size == 0:
        return annotated

    zoomed = cv2.resize(
        crop,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return zoomed


def draw_ball_marker(
    annotated: np.ndarray,
    track: dict[str, Any],
    selected: bool,
) -> None:
    center_x, center_y = center_for_track(annotated.shape, track)
    x1, y1, x2, y2 = clip_box_to_frame(annotated.shape, track)
    radius = max(8, int(round(max(x2 - x1, y2 - y1) * 0.85)))
    color = BALL_MARKER_SELECTED_COLOR if selected else BALL_MARKER_COLOR
    cv2.circle(annotated, (center_x, center_y), radius, color, 2, cv2.LINE_AA)
    cv2.circle(annotated, (center_x, center_y), max(4, radius // 2), color, -1, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"Ball {int(track['track_id'])}",
        (center_x + radius + 6, max(24, center_y - radius - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def recent_track_centers(
    frame_shape: tuple[int, int, int],
    frames_index: dict[int, list[dict[str, Any]]],
    frame_idx: int,
    track_id: int,
    history_length: int,
) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for past_frame_idx in range(max(0, frame_idx - history_length + 1), frame_idx + 1):
        track = find_track_by_id(frames_index.get(past_frame_idx, []), track_id)
        if track is None:
            continue
        centers.append(center_for_track(frame_shape, track))
    return centers


def draw_fireball_overlay(
    frame: np.ndarray,
    current_track: dict[str, Any],
    trail_points: list[tuple[int, int]],
) -> np.ndarray:
    annotated = frame.copy()
    overlay = annotated.copy()
    x1, y1, x2, y2 = clip_box_to_frame(annotated.shape, current_track)
    current_center = center_for_track(annotated.shape, current_track)
    radius = max(12, int(round(max(x2 - x1, y2 - y1) * 1.4)))

    if len(trail_points) >= 2:
        previous_point = trail_points[-2]
        direction = np.array(
            [current_center[0] - previous_point[0], current_center[1] - previous_point[1]],
            dtype=float,
        )
        norm = float(np.linalg.norm(direction))
        if norm > 1.0:
            direction /= norm
            perpendicular = np.array([-direction[1], direction[0]], dtype=float)
            tail_length = max(radius * 3.5, 48.0)
            tail_width = max(radius * 0.8, 12.0)
            tail_tip = np.array(current_center, dtype=float) - (direction * tail_length)
            left_point = np.array(current_center, dtype=float) + (perpendicular * tail_width)
            right_point = np.array(current_center, dtype=float) - (perpendicular * tail_width)
            flame_points = np.array(
                [left_point, right_point, tail_tip],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(overlay, flame_points, FIREBALL_OUTER_COLOR)

    if trail_points:
        for index, point in enumerate(trail_points[:-1]):
            blend = (index + 1) / max(1, len(trail_points))
            point_radius = max(6, int(round(radius * (0.35 + (blend * 0.55)))))
            color = FIREBALL_TRAIL_COLOR if blend < 0.6 else FIREBALL_WARM_COLOR
            cv2.circle(overlay, point, point_radius, color, -1, cv2.LINE_AA)

    cv2.circle(overlay, current_center, radius + 10, FIREBALL_OUTER_COLOR, -1, cv2.LINE_AA)
    cv2.circle(overlay, current_center, radius + 4, FIREBALL_WARM_COLOR, -1, cv2.LINE_AA)
    cv2.circle(overlay, current_center, radius, FIREBALL_HOT_COLOR, -1, cv2.LINE_AA)
    cv2.circle(overlay, current_center, max(6, radius // 2), FIREBALL_CORE_COLOR, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0.0, annotated)
    return annotated


def render_overlay(
    frame: np.ndarray,
    tracks: list[dict[str, Any]],
    mode: str,
    frame_idx: int,
    frames_index: dict[int, list[dict[str, Any]]],
    selected_track_id: int | None = None,
    multi_track_ids: list[int] | None = None,
    text_labels: dict[int, str] | None = None,
    pose_entries: list[dict[str, Any]] | None = None,
    ball_tracks: list[dict[str, Any]] | None = None,
    ball_frames_index: dict[int, list[dict[str, Any]]] | None = None,
    selected_ball_track_id: int | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    multi_track_ids = list(multi_track_ids or [])
    text_labels = dict(text_labels or {})
    pose_entries = list(pose_entries or [])
    ball_tracks = list(ball_tracks or [])
    ball_frames_index = ball_frames_index or {}

    if mode == MODE_SPOTLIGHT and selected_track_id is not None:
        selected_track = find_track_by_id(tracks, selected_track_id)
        if selected_track is not None:
            return draw_spotlight_overlay(annotated, selected_track)
        return annotated

    if mode == MODE_POSE:
        if selected_track_id is None:
            pass
        else:
            selected_pose = find_pose_by_track_id(pose_entries, selected_track_id)
            if selected_pose is None:
                return annotated
            return draw_pose_overlay(annotated, selected_pose)

    if mode == MODE_ZOOM and selected_track_id is not None:
        return draw_zoom_overlay(
            frame=annotated,
            current_tracks=tracks,
            frames_index=frames_index,
            frame_idx=frame_idx,
            selected_track_id=selected_track_id,
        )

    if mode == MODE_BALL:
        if selected_ball_track_id is not None:
            selected_ball = find_track_by_id(ball_tracks, selected_ball_track_id)
            if selected_ball is None:
                return annotated
            trail_points = recent_track_centers(
                frame_shape=annotated.shape,
                frames_index=ball_frames_index,
                frame_idx=frame_idx,
                track_id=selected_ball_track_id,
                history_length=BALL_TRAIL_HISTORY,
            )
            return draw_fireball_overlay(annotated, selected_ball, trail_points)

        for track in ball_tracks:
            draw_ball_marker(annotated, track, selected=False)
        return annotated

    highlighted_multi_ids = set(multi_track_ids)
    highlighted_text_ids = set(text_labels)
    for track in tracks:
        track_id = int(track["track_id"])
        if mode in {MODE_SINGLE, MODE_POSE, MODE_ZOOM} and selected_track_id is not None and track_id != selected_track_id:
            continue

        color = color_for_track(track_id)
        thickness = 2
        show_track_label = True
        if mode in {MODE_SINGLE, MODE_POSE, MODE_ZOOM} and selected_track_id is not None and track_id == selected_track_id:
            thickness = 4
        elif mode in {MODE_PAIR, MODE_AREA} and track_id in highlighted_multi_ids:
            thickness = 4
            color = PAIR_SELECTED_BOX_COLOR
        elif (
            mode == MODE_TEXT
            and track_id in highlighted_text_ids
            and not bool(track.get("is_lost_buffer"))
        ):
            thickness = 4
            color = PAIR_SELECTED_BOX_COLOR
            show_track_label = False

        draw_track_box(
            annotated,
            track,
            color,
            thickness,
            show_track_label=show_track_label,
        )

    if mode == MODE_PAIR:
        selected_tracks = visible_selected_tracks(tracks, multi_track_ids)
        if len(selected_tracks) >= 2:
            draw_multi_link(annotated, selected_tracks)
        for track in selected_tracks:
            draw_pair_ground_ring(annotated, track)

    if mode == MODE_AREA:
        selected_tracks = visible_selected_tracks(tracks, multi_track_ids)
        draw_area_polygon(annotated, selected_tracks)
        for track in selected_tracks:
            draw_pair_ground_ring(annotated, track)

    if mode == MODE_TEXT:
        for track_id, label in text_labels.items():
            track = find_track_by_id(tracks, track_id)
            if track is None or bool(track.get("is_lost_buffer")):
                continue
            draw_track_text_label(annotated, track, label)

    return annotated


def pick_track_from_click(x: int, y: int, tracks: list[dict[str, Any]]) -> int | None:
    smallest_area = None
    selected_track_id = None

    for track in tracks:
        x1 = float(track["x1"])
        y1 = float(track["y1"])
        x2 = float(track["x2"])
        y2 = float(track["y2"])
        if x < x1 or x > x2 or y < y1 or y > y2:
            continue

        area = max(1.0, (x2 - x1) * (y2 - y1))
        if smallest_area is None or area < smallest_area:
            smallest_area = area
            selected_track_id = int(track["track_id"])

    return selected_track_id


def create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter | None:
    for codec in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            print(f"Writing {output_path.name} with codec {codec}")
            return writer
        writer.release()

    print(
        f"Failed to create an MP4 writer for {output_path}. "
        "Tried codecs: mp4v, avc1."
    )
    return None


def export_video(
    video_path: Path,
    video_info: dict[str, Any],
    frames_index: dict[int, list[dict[str, Any]]],
    pose_frames_index: dict[int, list[dict[str, Any]]],
    ball_frames_index: dict[int, list[dict[str, Any]]],
    output_path: Path,
    mode: str,
    selected_track_id: int | None = None,
    multi_track_ids: list[int] | None = None,
    text_labels: dict[int, str] | None = None,
    selected_ball_track_id: int | None = None,
) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Could not open video for export: {video_path}")
        return False

    writer = create_video_writer(
        output_path=output_path,
        fps=float(video_info["fps"]),
        frame_size=(int(video_info["width"]), int(video_info["height"])),
    )
    if writer is None:
        capture.release()
        return False

    total_frames = int(video_info["total_frames"])
    frame_idx = 0
    started_at = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_tracks = frames_index.get(frame_idx, [])
            frame_pose_entries = pose_frames_index.get(frame_idx, [])
            frame_ball_tracks = ball_frames_index.get(frame_idx, [])
            annotated = render_overlay(
                frame,
                frame_tracks,
                mode=mode,
                frame_idx=frame_idx,
                frames_index=frames_index,
                selected_track_id=selected_track_id,
                multi_track_ids=multi_track_ids,
                text_labels=text_labels,
                pose_entries=frame_pose_entries,
                ball_tracks=frame_ball_tracks,
                ball_frames_index=ball_frames_index,
                selected_ball_track_id=selected_ball_track_id,
            )

            writer.write(annotated)

            if frame_idx % 30 == 0 or frame_idx + 1 == total_frames:
                elapsed = max(time.time() - started_at, 0.001)
                rendered_frames = frame_idx + 1
                fps = rendered_frames / elapsed
                percent = (rendered_frames / total_frames) * 100.0
                print(
                    f"Export progress: {rendered_frames}/{total_frames} "
                    f"frames ({percent:.1f}%) at {fps:.2f} fps"
                )

            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    print(f"Saved output video to {output_path}")
    return True


def overlay_player_status(
    frame: np.ndarray,
    frame_idx: int,
    frame_count: int,
    paused: bool,
    mode: str,
    selected_track_id: int | None,
    multi_track_ids: list[int],
    text_labels: dict[int, str],
    selected_ball_track_id: int | None,
) -> np.ndarray:
    annotated = frame.copy()
    if mode == MODE_SINGLE and selected_track_id is not None:
        mode_label = f"Track {selected_track_id}"
    elif mode == MODE_SPOTLIGHT:
        mode_label = (
            f"Spotlight track {selected_track_id}"
            if selected_track_id is not None
            else "Spotlight mode (click a player)"
        )
    elif mode == MODE_PAIR:
        if len(multi_track_ids) >= 2:
            mode_label = f"Multi link {' -> '.join(str(track_id) for track_id in multi_track_ids)}"
        elif len(multi_track_ids) == 1:
            mode_label = f"Multi link ({multi_track_ids[0]} selected, pick more)"
        else:
            mode_label = "Multi link mode (pick 2+ players)"
    elif mode == MODE_TEXT:
        if text_labels:
            mode_label = f"Text labels {', '.join(str(track_id) for track_id in text_labels)}"
        else:
            mode_label = "Text mode (click a player to label)"
    elif mode == MODE_AREA:
        if len(multi_track_ids) >= 3:
            mode_label = f"Area {' -> '.join(str(track_id) for track_id in multi_track_ids)}"
        elif multi_track_ids:
            mode_label = f"Area mode ({len(multi_track_ids)} selected, pick more)"
        else:
            mode_label = "Area mode (pick 3+ players)"
    elif mode == MODE_POSE:
        mode_label = (
            f"Pose skeleton {selected_track_id}"
            if selected_track_id is not None
            else "Pose mode (click a player)"
        )
    elif mode == MODE_ZOOM:
        mode_label = (
            f"Zoom follow {selected_track_id}"
            if selected_track_id is not None
            else "Zoom mode (click a player)"
        )
    elif mode == MODE_BALL:
        mode_label = (
            f"Fireball {selected_ball_track_id}"
            if selected_ball_track_id is not None
            else "Ball mode (click a ball)"
        )
    else:
        mode_label = "All players"

    play_state = "Paused" if paused else "Playing"
    lines = [
        f"{play_state} | Frame {frame_idx + 1}/{frame_count}",
        f"Mode: {mode_label}",
        "Space pause/resume | Left/Right step when paused",
        "Left click select | Right click/C clear | A all | Esc exit special mode",
        "S spotlight | P multi link | T text | G area | K pose | Z zoom | B ball | J jump | E export | Q quit",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return annotated


class InteractivePlayer:
    def __init__(
        self,
        config: AppConfig,
        video_info: dict[str, Any],
        frames_index: dict[int, list[dict[str, Any]]],
        pose_frames_index: dict[int, list[dict[str, Any]]],
        ball_frames_index: dict[int, list[dict[str, Any]]],
    ) -> None:
        self.config = config
        self.video_info = video_info
        self.frames_index = frames_index
        self.pose_frames_index = pose_frames_index
        self.ball_frames_index = ball_frames_index
        self.capture: cv2.VideoCapture | None = None
        self.current_frame_idx = 0
        self.paused = False
        self.mode = MODE_ALL
        self.selected_track_id: int | None = None
        self.selected_ball_track_id: int | None = None
        self.multi_track_ids: list[int] = []
        self.text_track_labels: dict[int, str] = {}
        self.current_frame: np.ndarray | None = None
        self.current_tracks: list[dict[str, Any]] = []
        self.current_pose_entries: list[dict[str, Any]] = []
        self.current_ball_tracks: list[dict[str, Any]] = []
        self.frame_duration = 1.0 / max(float(video_info["fps"]), 0.001)
        self.next_frame_deadline = 0.0

    def run(self) -> int:
        self.open()
        try:
            while True:
                if self.current_frame is None:
                    print("No video frame is available for playback.")
                    return 1

                display_frame = render_overlay(
                    self.current_frame,
                    self.current_tracks,
                    mode=self.mode,
                    frame_idx=self.current_frame_idx,
                    frames_index=self.frames_index,
                    selected_track_id=self.selected_track_id,
                    multi_track_ids=self.multi_track_ids,
                    text_labels=self.text_track_labels,
                    pose_entries=self.current_pose_entries,
                    ball_tracks=self.current_ball_tracks,
                    ball_frames_index=self.ball_frames_index,
                    selected_ball_track_id=self.selected_ball_track_id,
                )
                display_frame = overlay_player_status(
                    display_frame,
                    self.current_frame_idx,
                    int(self.video_info["total_frames"]),
                    self.paused,
                    self.mode,
                    self.selected_track_id,
                    self.multi_track_ids,
                    self.text_track_labels,
                    self.selected_ball_track_id,
                )
                cv2.imshow(WINDOW_NAME, display_frame)

                wait_ms = 30 if self.paused else max(
                    1,
                    int((self.next_frame_deadline - time.perf_counter()) * 1000),
                )
                key = cv2.waitKeyEx(wait_ms)
                if key != -1 and not self.handle_key(key):
                    return 0

                if self.paused:
                    continue

                now = time.perf_counter()
                if now < self.next_frame_deadline:
                    continue

                if not self.load_frame(self.current_frame_idx + 1, prefer_sequential=True):
                    if self.current_frame_idx >= int(self.video_info["total_frames"]) - 1:
                        self.paused = True
                        print("Reached the end of the video. Playback paused on the last frame.")
                    else:
                        print("Failed to decode the next frame. Playback paused.")
                        self.paused = True
                    self.next_frame_deadline = time.perf_counter() + self.frame_duration
                    continue

                self.next_frame_deadline += self.frame_duration
                if now - self.next_frame_deadline > self.frame_duration:
                    self.next_frame_deadline = time.perf_counter() + self.frame_duration
        finally:
            self.close()

    def open(self) -> None:
        self.capture = cv2.VideoCapture(str(self.config.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video for playback: {self.config.video_path}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        if not self.load_frame(0):
            raise RuntimeError(f"Could not load the first frame from {self.config.video_path}")

        # Once the tracking metadata is cached, switching overlays only changes which box we draw.
        # The player keeps reading the original frames while the selected track ID changes instantly.
        self.next_frame_deadline = time.perf_counter() + self.frame_duration
        print("Interactive playback started.")
        print(
            "Controls: Space pause/resume, left click select, right click/C clear, "
            "A all, S spotlight, P multi link, T text, G area, K pose, Z zoom, B ball, "
            "Esc exit special mode, E export, Q quit"
        )

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            cv2.destroyAllWindows()

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == MODE_BALL:
                picked_ball_id = pick_track_from_click(x, y, self.current_ball_tracks)
                if picked_ball_id is None:
                    print("Click landed outside all tracked balls.")
                    return
                self.selected_ball_track_id = picked_ball_id
                print(f"Fireball track ID: {picked_ball_id}")
                return

            picked_track_id = pick_track_from_click(x, y, self.current_tracks)
            if picked_track_id is None:
                print("Click landed outside all tracked player boxes.")
                return

            if self.mode == MODE_SPOTLIGHT:
                self.selected_track_id = picked_track_id
                print(f"Spotlight track ID: {picked_track_id}")
                return

            if self.mode == MODE_PAIR:
                self.update_multi_selection(picked_track_id, "Multi-link")
                return

            if self.mode == MODE_TEXT:
                self.prompt_for_text_label(picked_track_id)
                return

            if self.mode == MODE_AREA:
                self.update_multi_selection(picked_track_id, "Area")
                return

            if self.mode == MODE_POSE:
                self.selected_track_id = picked_track_id
                print(f"Pose skeleton track ID: {picked_track_id}")
                return

            if self.mode == MODE_ZOOM:
                self.selected_track_id = picked_track_id
                print(f"Zoom follow track ID: {picked_track_id}")
                return

            self.mode = MODE_SINGLE
            self.selected_track_id = picked_track_id
            self.selected_ball_track_id = None
            self.multi_track_ids = []
            self.text_track_labels = {}
            print(f"Selected track ID: {picked_track_id}")
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self.clear_current_selection()

    def reset_to_all_mode(self) -> None:
        if (
            self.mode == MODE_ALL
            and self.selected_track_id is None
            and self.selected_ball_track_id is None
            and not self.multi_track_ids
            and not self.text_track_labels
        ):
            return
        self.mode = MODE_ALL
        self.selected_track_id = None
        self.selected_ball_track_id = None
        self.multi_track_ids = []
        self.text_track_labels = {}
        print("Selection cleared. Showing all tracked players.")

    def activate_special_mode(self, mode: str) -> None:
        self.mode = mode
        self.selected_track_id = None
        self.selected_ball_track_id = None
        self.multi_track_ids = []
        self.text_track_labels = {}
        if mode == MODE_SPOTLIGHT:
            print("Spotlight mode active. Left click a player to draw the sky spotlight.")
        elif mode == MODE_PAIR:
            print("Multi-link mode active. Left click players to build an ordered link chain.")
        elif mode == MODE_TEXT:
            print("Text mode active. Left click a player to add or update a moving label.")
        elif mode == MODE_AREA:
            print("Area mode active. Left click players to build a transparent polygon area.")
        elif mode == MODE_POSE:
            print("Pose mode active. Left click a player to draw the skeleton overlay.")
        elif mode == MODE_ZOOM:
            print("Zoom mode active. Left click a player to reframe around that player.")
        elif mode == MODE_BALL:
            print("Ball mode active. Left click a tracked ball to apply the fireball effect.")

    def clear_current_selection(self) -> None:
        if self.mode in {MODE_SPOTLIGHT, MODE_POSE, MODE_ZOOM}:
            if self.selected_track_id is None:
                return
            self.selected_track_id = None
            if self.mode == MODE_SPOTLIGHT:
                print("Spotlight selection cleared. Spotlight mode is still active.")
            elif self.mode == MODE_POSE:
                print("Pose selection cleared. Pose mode is still active.")
            else:
                print("Zoom selection cleared. Zoom mode is still active.")
            return

        if self.mode in {MODE_PAIR, MODE_AREA}:
            if not self.multi_track_ids:
                return
            self.multi_track_ids = []
            if self.mode == MODE_PAIR:
                print("Multi-link selection cleared. Multi-link mode is still active.")
            else:
                print("Area selection cleared. Area mode is still active.")
            return

        if self.mode == MODE_TEXT:
            if not self.text_track_labels:
                return
            self.text_track_labels = {}
            print("Text labels cleared. Text mode is still active.")
            return

        if self.mode == MODE_BALL:
            if self.selected_ball_track_id is None:
                return
            self.selected_ball_track_id = None
            print("Ball selection cleared. Ball mode is still active.")
            return

        self.reset_to_all_mode()

    def update_multi_selection(self, track_id: int, mode_label: str) -> None:
        if track_id in self.multi_track_ids:
            print(f"Track ID {track_id} is already selected for {mode_label.lower()}.")
            return

        self.multi_track_ids.append(track_id)
        print(
            f"{mode_label} selection: {' -> '.join(str(selected_id) for selected_id in self.multi_track_ids)}"
        )

    def prompt_for_text_label(self, track_id: int) -> None:
        previous_paused = self.paused
        self.paused = True
        current_label = self.text_track_labels.get(track_id, "")
        prompt = f"Enter label for track {track_id}"
        if current_label:
            prompt += f" [{current_label}]"
        prompt += ": "

        try:
            response = input(prompt)
        except EOFError:
            response = ""

        label = response.strip()
        if not label:
            print(f"Label update cancelled for track {track_id}.")
        else:
            self.text_track_labels[track_id] = label
            print(f"Text label set for track {track_id}: {label}")

        self.paused = previous_paused
        self.next_frame_deadline = time.perf_counter() + self.frame_duration

    def handle_key(self, key: int) -> bool:
        if key in SPACE_KEYS:
            self.paused = not self.paused
            state = "paused" if self.paused else "resumed"
            print(f"Playback {state}.")
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return True

        key_char = key & 0xFF
        if key_char in (ord("q"), ord("Q")):
            print("Quitting interactive player.")
            return False
        if key_char in (ord("a"), ord("A")):
            self.reset_to_all_mode()
            return True
        if key_char in (ord("c"), ord("C")):
            self.clear_current_selection()
            return True
        if key_char in (ord("s"), ord("S")):
            self.activate_special_mode(MODE_SPOTLIGHT)
            return True
        if key_char in (ord("p"), ord("P")):
            self.activate_special_mode(MODE_PAIR)
            return True
        if key_char in (ord("t"), ord("T")):
            self.activate_special_mode(MODE_TEXT)
            return True
        if key_char in (ord("g"), ord("G")):
            self.activate_special_mode(MODE_AREA)
            return True
        if key_char in (ord("k"), ord("K")):
            self.activate_special_mode(MODE_POSE)
            return True
        if key_char in (ord("z"), ord("Z")):
            self.activate_special_mode(MODE_ZOOM)
            return True
        if key_char in (ord("b"), ord("B")):
            self.activate_special_mode(MODE_BALL)
            return True
        if key_char in (ord("j"), ord("J")):
            self.jump_to_frame()
            return True
        if key_char in (ord("e"), ord("E")):
            self.export_current_mode()
            return True
        if key in ESC_KEYS and self.mode in {
            MODE_SPOTLIGHT,
            MODE_PAIR,
            MODE_TEXT,
            MODE_AREA,
            MODE_POSE,
            MODE_ZOOM,
            MODE_BALL,
        }:
            self.reset_to_all_mode()
            return True

        if self.paused and key in LEFT_ARROW_KEYS:
            self.step_frame(-1)
            return True
        if self.paused and key in RIGHT_ARROW_KEYS:
            self.step_frame(1)
            return True

        return True

    def step_frame(self, delta: int) -> None:
        target_frame = self.current_frame_idx + delta
        if self.load_frame(target_frame):
            print(f"Stepped to frame {self.current_frame_idx}.")
        else:
            print(f"Could not step to frame {target_frame}.")

    def jump_to_frame(self) -> None:
        previous_paused = self.paused
        self.paused = True
        try:
            response = input(
                f"Enter frame index (0 to {int(self.video_info['total_frames']) - 1}): "
            ).strip()
        except EOFError:
            response = ""

        if not response:
            self.paused = previous_paused
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return

        try:
            requested_frame = int(response)
        except ValueError:
            print("Frame index must be an integer.")
            self.paused = previous_paused
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return

        if self.load_frame(requested_frame):
            print(f"Jumped to frame {self.current_frame_idx}.")
        else:
            print(f"Could not jump to frame {requested_frame}.")

        self.paused = previous_paused
        self.next_frame_deadline = time.perf_counter() + self.frame_duration

    def export_current_mode(self) -> None:
        output_path: Path
        export_selected_id = self.selected_track_id
        export_multi_ids = list(self.multi_track_ids)
        export_text_labels = dict(self.text_track_labels)
        export_ball_id = self.selected_ball_track_id

        if self.mode == MODE_SPOTLIGHT and export_selected_id is None:
            print("Spotlight mode needs a selected player before export.")
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return

        if self.mode == MODE_PAIR and len(export_multi_ids) < 2:
            print("Multi-link mode needs at least two selected players before export.")
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return

        if self.mode == MODE_TEXT and not export_text_labels:
            print("Text mode needs at least one labeled player before export.")
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return

        if self.mode == MODE_AREA and len(export_multi_ids) < 3:
            print("Area mode needs at least three selected players before export.")
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return

        if self.mode == MODE_POSE and export_selected_id is None:
            print("Pose mode needs a selected player before export.")
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return

        if self.mode == MODE_ZOOM and export_selected_id is None:
            print("Zoom mode needs a selected player before export.")
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return

        if self.mode == MODE_BALL and export_ball_id is None:
            print("Ball mode needs a selected ball before export.")
            self.next_frame_deadline = time.perf_counter() + self.frame_duration
            return

        if self.mode == MODE_ALL:
            output_path = self.config.all_output_path
            print("Exporting all-player view from cached tracks.")
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_ALL,
                selected_track_id=None,
                multi_track_ids=[],
                text_labels={},
                selected_ball_track_id=None,
            )
        elif self.mode == MODE_SINGLE and export_selected_id is not None:
            output_path = selected_output_path_for(self.config.video_path, export_selected_id)
            print(f"Exporting selected-player view for track {export_selected_id}.")
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_SINGLE,
                selected_track_id=export_selected_id,
                multi_track_ids=[],
                text_labels={},
                selected_ball_track_id=None,
            )
        elif self.mode == MODE_SPOTLIGHT and export_selected_id is not None:
            output_path = spotlight_output_path_for(self.config.video_path, export_selected_id)
            print(f"Exporting spotlight view for track {export_selected_id}.")
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_SPOTLIGHT,
                selected_track_id=export_selected_id,
                multi_track_ids=[],
                text_labels={},
                selected_ball_track_id=None,
            )
        elif self.mode == MODE_PAIR and len(export_multi_ids) >= 2:
            output_path = pair_output_path_for(self.config.video_path, export_multi_ids)
            print(
                "Exporting multi-link view for tracks "
                f"{' -> '.join(str(track_id) for track_id in export_multi_ids)}."
            )
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_PAIR,
                selected_track_id=None,
                multi_track_ids=export_multi_ids,
                text_labels={},
                selected_ball_track_id=None,
            )
        elif self.mode == MODE_TEXT and export_text_labels:
            output_path = text_output_path_for(self.config.video_path)
            print(
                "Exporting text-label view for tracks "
                f"{', '.join(str(track_id) for track_id in export_text_labels)}."
            )
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_TEXT,
                selected_track_id=None,
                multi_track_ids=[],
                text_labels=export_text_labels,
                selected_ball_track_id=None,
            )
        elif self.mode == MODE_AREA and len(export_multi_ids) >= 3:
            output_path = area_output_path_for(self.config.video_path, export_multi_ids)
            print(
                "Exporting area view for tracks "
                f"{' -> '.join(str(track_id) for track_id in export_multi_ids)}."
            )
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_AREA,
                selected_track_id=None,
                multi_track_ids=export_multi_ids,
                text_labels={},
                selected_ball_track_id=None,
            )
        elif self.mode == MODE_POSE and export_selected_id is not None:
            output_path = pose_output_path_for(self.config.video_path, export_selected_id)
            print(f"Exporting pose view for track {export_selected_id}.")
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_POSE,
                selected_track_id=export_selected_id,
                multi_track_ids=[],
                text_labels={},
                selected_ball_track_id=None,
            )
        elif self.mode == MODE_ZOOM and export_selected_id is not None:
            output_path = zoom_output_path_for(self.config.video_path, export_selected_id)
            print(f"Exporting zoom-follow view for track {export_selected_id}.")
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_ZOOM,
                selected_track_id=export_selected_id,
                multi_track_ids=[],
                text_labels={},
                selected_ball_track_id=None,
            )
        elif self.mode == MODE_BALL and export_ball_id is not None:
            output_path = fireball_output_path_for(self.config.video_path, export_ball_id)
            print(f"Exporting fireball view for ball track {export_ball_id}.")
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_BALL,
                selected_track_id=None,
                multi_track_ids=[],
                text_labels={},
                selected_ball_track_id=export_ball_id,
            )
        else:
            output_path = self.config.all_output_path
            print("Nothing is selected, so export fell back to the all-player view.")
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                pose_frames_index=self.pose_frames_index,
                ball_frames_index=self.ball_frames_index,
                output_path=output_path,
                mode=MODE_ALL,
                selected_track_id=None,
                multi_track_ids=[],
                text_labels={},
                selected_ball_track_id=None,
            )

        if success:
            print(f"Export completed: {output_path}")
        else:
            print("Export failed.")

        self.next_frame_deadline = time.perf_counter() + self.frame_duration

    def load_frame(self, frame_idx: int, prefer_sequential: bool = False) -> bool:
        if self.capture is None:
            return False

        max_frame_idx = int(self.video_info["total_frames"]) - 1
        if frame_idx < 0 or frame_idx > max_frame_idx:
            return False

        clamped_frame_idx = frame_idx

        if prefer_sequential and clamped_frame_idx == self.current_frame_idx + 1:
            ok, frame = self.capture.read()
            if ok:
                self.current_frame = frame
                self.current_frame_idx = clamped_frame_idx
                self.current_tracks = self.frames_index.get(clamped_frame_idx, [])
                self.current_pose_entries = self.pose_frames_index.get(clamped_frame_idx, [])
                self.current_ball_tracks = self.ball_frames_index.get(clamped_frame_idx, [])
                return True

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, clamped_frame_idx)
        ok, frame = self.capture.read()
        if not ok:
            return False

        self.current_frame = frame
        self.current_frame_idx = clamped_frame_idx
        self.current_tracks = self.frames_index.get(clamped_frame_idx, [])
        self.current_pose_entries = self.pose_frames_index.get(clamped_frame_idx, [])
        self.current_ball_tracks = self.ball_frames_index.get(clamped_frame_idx, [])
        return True


def main() -> int:
    args = parse_args()

    try:
        config = resolve_config(args)
        video_info = load_video_info(config.video_path)
        track_cache_data = run_tracking(config, video_info)
        pose_cache_data = run_pose_estimation(
            config,
            video_info,
            track_cache_data.get("frames", {}),
        )
        ball_cache_data = run_ball_tracking(config, video_info)
    except Exception as exc:
        print(f"Startup failed: {exc}")
        return 1

    frames_index = track_cache_data.get("frames", {})
    pose_frames_index = pose_cache_data.get("frames", {})
    ball_frames_index = ball_cache_data.get("frames", {})
    if not frames_index:
        print(
            "Tracking cache is valid but contains no player boxes. "
            "The video will still open without overlays."
        )

    # Future improvements:
    # - Add a BoT-SORT tracker option alongside ByteTrack.
    # - Add ReID support to recover identities after longer occlusions.
    # - Replace OpenCV HighGUI with a PyQt-based desktop UI.
    # - Use FFmpeg or PyAV for better seeking and export behavior.
    # - Classify team colors for per-team filtering.
    # - Track referee, goalkeeper, and ball classes separately.
    # - Add pitch-mask filtering to suppress off-field detections.
    player = InteractivePlayer(
        config=config,
        video_info=video_info,
        frames_index=frames_index,
        pose_frames_index=pose_frames_index,
        ball_frames_index=ball_frames_index,
    )

    try:
        return player.run()
    except Exception as exc:
        print(f"Interactive playback failed: {exc}")
        cv2.destroyAllWindows()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
