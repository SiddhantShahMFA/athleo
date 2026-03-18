from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WINDOW_NAME = "Athleo Interactive Player"
TRACKER_NAME = "bytetrack.yaml"
CACHE_VERSION = 2
DEFAULT_FPS = 30.0
LEFT_ARROW_KEYS = {81, 2424832, 65361}
RIGHT_ARROW_KEYS = {83, 2555904, 65363}
SPACE_KEYS = {32}


@dataclass(frozen=True)
class AppConfig:
    video_path: Path
    model_path: Path
    conf_threshold: float
    center_alpha: float
    size_alpha: float
    lost_buffer: int
    cache_path: Path
    all_output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track soccer players once with YOLO + ByteTrack, cache the smoothed metadata, "
            "and open an interactive OpenCV playback window."
        )
    )
    parser.add_argument("--video", help="Input video path.")
    parser.add_argument(
        "--model",
        default="yolo26x.pt",
        help="Ultralytics model weights path. Defaults to ./yolo26x.pt",
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

    return AppConfig(
        video_path=video_path.resolve(),
        model_path=model_path.resolve(),
        conf_threshold=float(args.conf),
        center_alpha=float(args.center_alpha),
        size_alpha=float(args.size_alpha),
        lost_buffer=int(args.lost_buffer),
        cache_path=cache_path.resolve(),
        all_output_path=video_path.resolve().with_name("annotated_all.mp4"),
    )


def selected_output_path_for(video_path: Path, track_id: int) -> Path:
    return video_path.with_name(f"annotated_selected_{track_id}.mp4")


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


def cache_matches(cache_data: dict[str, Any], config: AppConfig, video_info: dict[str, Any]) -> bool:
    if int(cache_data.get("cache_version", -1)) != CACHE_VERSION:
        return False

    video_meta = cache_data.get("video", {})
    tracking_meta = cache_data.get("tracking", {})
    expected_model_name = config.model_path.name

    return (
        video_meta.get("path") == str(config.video_path)
        and int(video_meta.get("width", -1)) == int(video_info["width"])
        and int(video_meta.get("height", -1)) == int(video_info["height"])
        and int(video_meta.get("total_frames", -1)) == int(video_info["total_frames"])
        and int(video_meta.get("file_size", -1)) == int(video_info["file_size"])
        and int(video_meta.get("mtime_ns", -1)) == int(video_info["mtime_ns"])
        and round(float(video_meta.get("fps", -1.0)), 3) == round(float(video_info["fps"]), 3)
        and tracking_meta.get("model_path") == str(config.model_path)
        and tracking_meta.get("model_name") == expected_model_name
        and tracking_meta.get("tracker") == TRACKER_NAME
        and float(tracking_meta.get("conf_threshold", -1.0)) == config.conf_threshold
        and float(tracking_meta.get("center_alpha", -1.0)) == config.center_alpha
        and float(tracking_meta.get("size_alpha", -1.0)) == config.size_alpha
        and int(tracking_meta.get("lost_buffer", -1)) == config.lost_buffer
    )


def load_track_cache(config: AppConfig, video_info: dict[str, Any]) -> dict[str, Any] | None:
    if not config.cache_path.exists():
        return None

    try:
        with config.cache_path.open("r", encoding="utf-8") as cache_file:
            cache_data = json.load(cache_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable cache file {config.cache_path}: {exc}")
        return None

    if not cache_matches(cache_data, config, video_info):
        print(f"Existing cache does not match current settings, rebuilding: {config.cache_path}")
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
    print(f"Loaded tracking cache from {config.cache_path}")
    return cache_data


def save_track_cache(cache_data: dict[str, Any], cache_path: Path) -> None:
    serializable_cache = dict(cache_data)
    serializable_cache["frames"] = {
        str(frame_idx): frame_tracks
        for frame_idx, frame_tracks in cache_data["frames"].items()
    }

    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(serializable_cache, cache_file, indent=2)


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


def run_tracking(config: AppConfig, video_info: dict[str, Any]) -> dict[str, Any]:
    cached_data = load_track_cache(config, video_info)
    if cached_data is not None:
        return cached_data

    if not config.model_path.exists():
        raise FileNotFoundError(
            f"Model weights file was not found and no valid cache is available: {config.model_path}"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install requirements.txt before running a new tracking pass."
        ) from exc

    model = YOLO(str(config.model_path))
    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for tracking: {config.video_path}")

    records: list[dict[str, Any]] = []
    total_frames = int(video_info["total_frames"])
    frame_idx = 0
    started_at = time.time()

    # The offline pass exists so live playback only has to decode frames and draw cached overlays.
    # That keeps clicks instantaneous and avoids rerunning detection after every interaction.
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

                # Instant isolation depends on stable tracker IDs. Raw detector boxes alone can
                # jitter and reorder across frames, so selection would not stay attached to a player.
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

    # Tracker output is still noisy at the box level. Smoothing the center and size per track_id
    # makes the live overlay steadier while preserving the original video pixels underneath.
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


def draw_tracks_on_frame(
    frame: np.ndarray,
    tracks: list[dict[str, Any]],
    selected_track_id: int | None = None,
    show_all_tracks: bool = True,
    dim_unselected: bool = False,
) -> np.ndarray:
    annotated = frame.copy()

    for track in tracks:
        track_id = int(track["track_id"])
        is_selected = selected_track_id is not None and track_id == selected_track_id
        if not show_all_tracks and selected_track_id is not None and not is_selected:
            continue

        x1, y1, x2, y2 = clip_box_to_frame(annotated.shape, track)
        if x2 <= x1 or y2 <= y1:
            continue

        color = color_for_track(track_id)
        if dim_unselected and selected_track_id is not None and not is_selected:
            color = tuple(int(channel * 0.45) for channel in color)

        thickness = 4 if is_selected else 2
        label = f"ID {track_id}"
        if bool(track.get("is_lost_buffer")):
            label += " (hold)"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
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
    output_path: Path,
    selected_track_id: int | None = None,
    show_all_tracks: bool = True,
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
    drawn_boxes = 0
    frame_idx = 0
    started_at = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_tracks = frames_index.get(frame_idx, [])
            annotated = draw_tracks_on_frame(
                frame,
                frame_tracks,
                selected_track_id=selected_track_id,
                show_all_tracks=show_all_tracks,
            )
            if show_all_tracks or selected_track_id is None:
                drawn_boxes += len(frame_tracks)
            else:
                drawn_boxes += sum(
                    1 for track in frame_tracks if int(track["track_id"]) == selected_track_id
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

    if drawn_boxes == 0:
        print(f"No boxes were drawn into {output_path.name}. The video was still written.")
    else:
        print(f"Drew {drawn_boxes} boxes into {output_path.name}.")

    print(f"Saved output video to {output_path}")
    return True


def overlay_player_status(
    frame: np.ndarray,
    frame_idx: int,
    frame_count: int,
    paused: bool,
    selected_track_id: int | None,
    show_all_tracks: bool,
) -> np.ndarray:
    annotated = frame.copy()
    mode_label = "All players" if show_all_tracks or selected_track_id is None else f"Track {selected_track_id}"
    play_state = "Paused" if paused else "Playing"
    lines = [
        f"{play_state} | Frame {frame_idx + 1}/{frame_count}",
        f"Mode: {mode_label}",
        "Space pause/resume | Left/Right step when paused",
        "Left click select | Right click clear | A all | C clear",
        "J jump | E export current mode | Q quit",
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
    ) -> None:
        self.config = config
        self.video_info = video_info
        self.frames_index = frames_index
        self.capture: cv2.VideoCapture | None = None
        self.current_frame_idx = 0
        self.paused = False
        self.selected_track_id: int | None = None
        self.show_all_tracks = True
        self.current_frame: np.ndarray | None = None
        self.current_tracks: list[dict[str, Any]] = []
        self.frame_duration = 1.0 / max(float(video_info["fps"]), 0.001)
        self.next_frame_deadline = 0.0

    def run(self) -> int:
        self.open()
        try:
            while True:
                if self.current_frame is None:
                    print("No video frame is available for playback.")
                    return 1

                display_frame = draw_tracks_on_frame(
                    self.current_frame,
                    self.current_tracks,
                    selected_track_id=self.selected_track_id,
                    show_all_tracks=self.show_all_tracks,
                )
                display_frame = overlay_player_status(
                    display_frame,
                    self.current_frame_idx,
                    int(self.video_info["total_frames"]),
                    self.paused,
                    self.selected_track_id,
                    self.show_all_tracks,
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
        print("Controls: Space pause/resume, left click select, right click clear, E export, Q quit")

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
            selected_track_id = pick_track_from_click(x, y, self.current_tracks)
            if selected_track_id is None:
                print("Click landed outside all tracked player boxes.")
                return

            self.selected_track_id = selected_track_id
            self.show_all_tracks = False
            print(f"Selected track ID: {selected_track_id}")
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self.clear_selection()

    def clear_selection(self) -> None:
        if self.selected_track_id is None and self.show_all_tracks:
            return
        self.selected_track_id = None
        self.show_all_tracks = True
        print("Selection cleared. Showing all tracked players.")

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
            self.clear_selection()
            return True
        if key_char in (ord("c"), ord("C")):
            self.clear_selection()
            return True
        if key_char in (ord("j"), ord("J")):
            self.jump_to_frame()
            return True
        if key_char in (ord("e"), ord("E")):
            self.export_current_mode()
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
        export_selected_id = None if self.show_all_tracks else self.selected_track_id
        if export_selected_id is None:
            output_path = self.config.all_output_path
            print("Exporting all-player view from cached tracks.")
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                output_path=output_path,
                selected_track_id=None,
                show_all_tracks=True,
            )
        else:
            output_path = selected_output_path_for(self.config.video_path, export_selected_id)
            print(f"Exporting selected-player view for track {export_selected_id}.")
            success = export_video(
                video_path=self.config.video_path,
                video_info=self.video_info,
                frames_index=self.frames_index,
                output_path=output_path,
                selected_track_id=export_selected_id,
                show_all_tracks=False,
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
                return True

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, clamped_frame_idx)
        ok, frame = self.capture.read()
        if not ok:
            return False

        self.current_frame = frame
        self.current_frame_idx = clamped_frame_idx
        self.current_tracks = self.frames_index.get(clamped_frame_idx, [])
        return True


def main() -> int:
    args = parse_args()

    try:
        config = resolve_config(args)
        video_info = load_video_info(config.video_path)
        cache_data = run_tracking(config, video_info)
    except Exception as exc:
        print(f"Startup failed: {exc}")
        return 1

    frames_index = cache_data.get("frames", {})
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
    )

    try:
        return player.run()
    except Exception as exc:
        print(f"Interactive playback failed: {exc}")
        cv2.destroyAllWindows()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
