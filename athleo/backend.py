from __future__ import annotations

from pathlib import Path
from typing import Any

from .desktop import (
    AppConfig,
    export_video,
    load_ball_cache,
    load_pose_cache,
    load_track_cache,
    load_video_info,
    run_ball_tracking,
    run_pose_estimation,
    run_tracking,
)
from .models import ExportRequest, ProcessingOptions
from .storage import JobPaths


class RuntimeBackend:
    def build_config(self, paths: JobPaths, options: ProcessingOptions) -> AppConfig:
        return AppConfig(
            video_path=paths.video_path.resolve(),
            model_path=Path(options.model_path).expanduser().resolve(),
            pose_model_path=Path(options.pose_model_path).expanduser().resolve(),
            conf_threshold=float(options.conf_threshold),
            center_alpha=float(options.center_alpha),
            size_alpha=float(options.size_alpha),
            lost_buffer=int(options.lost_buffer),
            cache_path=paths.track_cache_path.resolve(),
            pose_cache_path=paths.pose_cache_path.resolve(),
            ball_cache_path=paths.ball_cache_path.resolve(),
            all_output_path=(paths.export_dir / "annotated_all.mp4").resolve(),
        )

    def process_job(self, paths: JobPaths, options: ProcessingOptions) -> dict[str, Any]:
        config = self.build_config(paths, options)
        video_info = load_video_info(config.video_path)
        track_cache_data = run_tracking(config, video_info)
        pose_cache_data = run_pose_estimation(config, video_info, track_cache_data.get("frames", {}))
        ball_cache_data = run_ball_tracking(config, video_info)
        return {
            "video_info": video_info,
            "track_frames": len(track_cache_data.get("frames", {})),
            "pose_frames": len(pose_cache_data.get("frames", {})),
            "ball_frames": len(ball_cache_data.get("frames", {})),
        }

    def export_job(
        self,
        paths: JobPaths,
        options: ProcessingOptions,
        export_request: ExportRequest,
        export_id: str,
        video_info: dict[str, Any],
    ) -> Path:
        config = self.build_config(paths, options)
        track_cache_data = load_track_cache(config, video_info)
        pose_cache_data = load_pose_cache(config, video_info)
        ball_cache_data = load_ball_cache(config, video_info)
        if track_cache_data is None or pose_cache_data is None or ball_cache_data is None:
            raise RuntimeError("Cached tracking artifacts are missing or no longer match the job settings.")

        output_path = paths.export_dir / f"{export_id}_{export_request.mode.value}.mp4"
        success = export_video(
            video_path=paths.video_path,
            video_info=video_info,
            frames_index=track_cache_data.get("frames", {}),
            pose_frames_index=pose_cache_data.get("frames", {}),
            ball_frames_index=ball_cache_data.get("frames", {}),
            output_path=output_path,
            mode=export_request.mode.value,
            selected_track_id=export_request.selected_track_id,
            multi_track_ids=export_request.multi_track_ids,
            text_labels=export_request.text_labels,
            selected_ball_track_id=export_request.selected_ball_track_id,
        )
        if not success:
            raise RuntimeError(f"Export failed for mode {export_request.mode.value}.")
        return output_path
