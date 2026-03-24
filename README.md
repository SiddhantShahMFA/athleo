# Athleo

Athleo now ships as a package-based video-processing backend with a FastAPI service for async jobs, while keeping the original OpenCV desktop player as a compatibility entrypoint on top of the same tracking runtime.

## Requirements

- Python 3.11+
- Local model weights at `./yolo26x.pt` and `./yolo26x-pose.pt` by default
- OpenCV window support if you want the desktop player

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run The FastAPI Backend

Start the API:

```bash
uvicorn athleo.api.app:app --reload
```

The backend stores uploaded videos, caches, manifests, and exports under `.athleo_jobs/` by default. Override that root with `ATHLEO_STORAGE_ROOT=/path/to/storage`.

### Main Endpoints

- `POST /api/v1/jobs`: upload a video and create a processing job
- `GET /api/v1/jobs/{job_id}`: poll job status, progress state, and artifact metadata
- `GET /api/v1/jobs/{job_id}/artifacts`: list job artifacts
- `GET /api/v1/jobs/{job_id}/artifacts/{artifact_id}`: download an artifact
- `GET /api/v1/jobs/{job_id}/frames/{frame_idx}`: read cached frame metadata
- `POST /api/v1/jobs/{job_id}/exports`: create an overlay export job
- `GET /api/v1/jobs/{job_id}/exports/{export_id}`: poll export status
- `GET /api/v1/jobs/{job_id}/exports/{export_id}/download`: download a rendered export
- `GET /healthz`: liveness check
- `GET /readyz`: readiness check

### Example Job Flow

Create a job:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs \
  -F "video=@Trial.mp4" \
  -F "model_path=yolo26x.pt" \
  -F "pose_model_path=yolo26x-pose.pt" \
  -F "conf_threshold=0.35" \
  -F "center_alpha=0.25" \
  -F "size_alpha=0.15" \
  -F "lost_buffer=10"
```

Create an export after processing completes:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs/<job_id>/exports \
  -H "Content-Type: application/json" \
  -d '{"mode":"spotlight","selected_track_id":7}'
```

The backend keeps the long-running work asynchronous and reuses existing caches when the same uploaded video and processing options are submitted again.

## Run The Desktop Player

The desktop workflow still works through the slim `main.py` compatibility entrypoint:

```bash
python3 main.py
```

Run with explicit arguments:

```bash
python3 main.py \
  --video "path/to/video.mp4" \
  --model yolo26x.pt \
  --pose-model yolo26x-pose.pt \
  --conf 0.35 \
  --center-alpha 0.25 \
  --size-alpha 0.15 \
  --lost-buffer 10
```

## Shared Processing Behavior

For both the API and desktop flows:

1. Athleo runs player tracking with `yolo26x.pt`, pose estimation with `yolo26x-pose.pt`, and ball tracking with `yolo26x.pt` class `32`.
2. Tracking data is cached as track, pose, and ball metadata instead of rerunning detection for every interaction.
3. Overlay exports reuse the cached metadata and never rerun tracking.

## Overlay Modes

The backend and desktop player share the same overlay/export modes:

- `all`
- `single_box`
- `spotlight`
- `pair_link`
- `player_text`
- `polygon_area`
- `pose_skeleton`
- `zoom_follow`
- `ball_fire`

Mode-specific requests require the same IDs the desktop app uses internally:

- `selected_track_id` for single, spotlight, pose, and zoom
- `multi_track_ids` for multi-link and area
- `text_labels` for player text
- `selected_ball_track_id` for fireball exports

## Desktop Controls

- Left click: select the clicked player on the current frame
- Right click: clear the current selection; in spotlight, multi-link, text, area, pose, zoom, or ball mode it stays in that mode
- `Space`: pause or resume playback
- Left arrow: step backward one frame when paused
- Right arrow: step forward one frame when paused
- `A`: force all-player mode
- `C`: clear current selection
- `S`: enter spotlight mode
- `P`: enter multi-link mode
- `T`: enter text mode
- `G`: enter area mode
- `K`: enter pose mode
- `Z`: enter zoom mode
- `B`: enter ball mode
- `Esc`: leave the current special mode
- `J`: jump to a frame by entering the frame number in the terminal
- `E`: export the current overlay mode to MP4
- `Q`: quit
