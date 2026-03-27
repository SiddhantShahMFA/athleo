# Athleo Project Handover

## 1. Project Overview

### Project name

- Athleo

### Purpose and business context

- Athleo is a soccer video analysis tool that processes a source clip once, caches tracking metadata, and then reuses that metadata for interactive playback and export workflows.
- The repository currently supports two product surfaces built on the same runtime:
- A FastAPI backend for asynchronous job processing and export generation.
- A local OpenCV desktop player kept as a compatibility entrypoint.
- The business value is in reducing repeated inference cost. Expensive tracking, pose estimation, and ball tracking are cached so users can inspect footage and generate multiple overlays without rerunning the AI pipeline every time.

### Key features and functionality

- Upload-driven backend job creation for video processing.
- Async export generation after a job completes.
- Cached player tracking, pose estimation, and ball tracking.
- Artifact download endpoints for input videos, manifests, caches, and rendered exports.
- Per-frame metadata retrieval through the API.
- Local interactive desktop player with click-based selection, keyboard-driven modes, and MP4 export.
- Overlay modes shared between backend exports and desktop playback:
- `all`
- `single_box`
- `spotlight`
- `pair_link`
- `player_text`
- `polygon_area`
- `pose_skeleton`
- `zoom_follow`
- `ball_fire`

## 2. Architecture & Tech Stack

### High-level architecture explanation

- The core architecture is a shared runtime model:
- `athleo/desktop.py` contains the main processing logic for video inspection, cache validation, tracking, pose estimation, overlay rendering, exporting, and the desktop interaction loop.
- `athleo/backend.py` wraps that runtime in a backend-friendly adapter so the API can reuse the exact same processing and export functions.
- `athleo/jobs.py` manages async job and export orchestration with a `ThreadPoolExecutor`, manifest persistence, and cache reuse across matching uploads.
- `athleo/storage.py` provides file-based workspace management for jobs, artifacts, and manifests under a root storage directory.
- `athleo/api/*` exposes the orchestration layer through FastAPI.
- There is no separate service boundary, queue broker, or database. Concurrency is in-process only.

### Technologies, frameworks, and libraries used

- Python 3.11+
- FastAPI for the HTTP API
- Pydantic for request/response schemas
- Uvicorn for local API serving
- OpenCV (`opencv-python`) for video decode, rendering, export, and desktop UI
- Ultralytics YOLO for detection, tracking, and pose estimation
- ByteTrack via `tracker="bytetrack.yaml"` for tracked detections
- `lap` as the ByteTrack support dependency
- `httpx` and `fastapi.testclient` in tests
- Standard library concurrency via `ThreadPoolExecutor`
- Standard library JSON and filesystem APIs for persistence

### Folder structure explanation

- `main.py`: compatibility CLI entrypoint for the desktop player.
- `athleo/desktop.py`: main runtime and desktop application logic.
- `athleo/backend.py`: adapter layer that turns the desktop runtime into backend processing/export operations.
- `athleo/jobs.py`: job lifecycle, export lifecycle, manifest updates, and cache reuse.
- `athleo/storage.py`: file storage abstraction and artifact metadata helpers.
- `athleo/models.py`: enums and dataclasses shared across API and backend layers.
- `athleo/api/app.py`: FastAPI app factory, lifespan handling, readiness and health endpoints.
- `athleo/api/routes/jobs.py`: versioned REST endpoints under `/api/v1/jobs`.
- `athleo/api/schemas.py`: Pydantic API contracts.
- `tests/test_api.py`: API and storage tests with a fake backend.
- `docs/desktop-player-guide.md`: detailed runtime guide for the desktop player.
- `docs/releases.md`: release log required by the repo workflow.

## 3. Setup & Installation

### Prerequisites

- Python 3.11 or newer
- Local model weight files:
- `yolo26x.pt`
- `yolo26x-pose.pt`
- OpenCV-compatible environment if the desktop player will be used
- A machine capable of running Ultralytics inference locally

### Step-by-step setup instructions

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Place the required model weights in the project root, or be ready to pass explicit paths.
4. Start either the API or the desktop player.

Example:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

- `ATHLEO_STORAGE_ROOT`
- Optional.
- Used only by the FastAPI backend.
- Controls where uploaded videos, cache files, manifests, and exports are stored.
- Defaults to `<repo>/.athleo_jobs` when unset.

### Build and run commands

- Run the API:

```bash
uvicorn athleo.api.app:app --reload
```

- Run the desktop player with defaults:

```bash
python3 main.py
```

- Run the desktop player with explicit paths and tuning:

```bash
python3 main.py \
  --video path/to/video.mp4 \
  --model yolo26x.pt \
  --pose-model yolo26x-pose.pt \
  --conf 0.35 \
  --center-alpha 0.25 \
  --size-alpha 0.15 \
  --lost-buffer 10
```

## 4. Code Structure & Key Components

### Explanation of major modules, services, or components

- `athleo/desktop.py`
- Defines `AppConfig`, cache formats, tracking and pose processing, overlay rendering, MP4 export, HUD rendering, and `InteractivePlayer`.
- This is effectively the domain/runtime layer for the project.

- `athleo/backend.py`
- Defines `RuntimeBackend`.
- Converts `JobPaths` plus `ProcessingOptions` into `AppConfig`.
- Reuses desktop runtime functions for processing and export instead of duplicating logic.

- `athleo/jobs.py`
- Defines `JobManager`.
- Handles job submission, async execution, manifest persistence, artifact listing, export creation, export polling, and cross-job cache reuse based on input hash and matching options.

- `athleo/storage.py`
- Defines `JobPaths` and `JobStorage`.
- Responsible for directory layout, upload persistence, manifest read/write, artifact metadata generation, and frame-level cache reads.

- `athleo/models.py`
- Centralizes enums for job status, artifact types, and overlay modes.
- Defines `ProcessingOptions` and `ExportRequest` dataclasses used across layers.

- `athleo/api/*`
- Provides the HTTP interface and schema validation around `JobManager`.

### Important classes/functions and what they do

- `create_app()` in `athleo/api/app.py`
- Builds the FastAPI app, injects a `JobManager`, and wires health endpoints.

- `JobManager.create_job()`
- Creates a job workspace, stores the upload, writes an initial manifest, and either reuses matching artifacts or submits background processing.

- `JobManager.create_export()`
- Validates that the source job is complete, writes an export record, and submits background rendering.

- `validate_processing_options()` and `validate_export_request()`
- Reject invalid tuning values and incomplete export payloads before work starts.

- `RuntimeBackend.process_job()`
- Loads video metadata, runs tracking, runs pose estimation, and runs ball tracking.

- `RuntimeBackend.export_job()`
- Reloads caches and renders an export without rerunning inference.

- `run_tracking()`
- Performs player detection/tracking with YOLO + ByteTrack, smooths boxes, and writes `tracks.json`.

- `run_pose_estimation()`
- Runs pose inference and matches pose detections back to tracked players via IoU.

- `run_ball_tracking()`
- Tracks class `32` from the detection model and writes `balls.json`.

- `render_overlay()`
- Applies the selected overlay mode to a frame using cached metadata.

- `export_video()`
- Replays the source video, renders overlays frame by frame, and writes MP4 output.

- `InteractivePlayer`
- Handles OpenCV window lifecycle, mouse/keyboard interactions, frame loading, and mode-aware desktop exports.

### Data flow within the system

#### Backend flow

1. Client uploads a video to `POST /api/v1/jobs`.
2. The API builds `ProcessingOptions` and calls `JobManager.create_job()`.
3. `JobStorage` creates a job workspace and stores the uploaded video.
4. `JobManager` writes `manifest.json`.
5. If another completed job has the same input SHA-256 and identical processing options, caches are copied into the new job and the job is marked complete.
6. Otherwise a background thread runs `RuntimeBackend.process_job()`.
7. The runtime reads video metadata, then generates track, pose, and ball caches.
8. Clients poll job status and fetch artifacts or frame metadata.
9. When an export is requested, another background thread reloads caches and calls `export_video()`.

#### Desktop flow

1. `main.py` calls `athleo.desktop.main()`.
2. CLI arguments are resolved into `AppConfig`.
3. Video metadata is inspected.
4. Track, pose, and ball caches are reused or regenerated.
5. `InteractivePlayer` opens the source clip and overlays cached metadata during playback.
6. When the user exports, the same cached metadata drives `export_video()`.

## 5. APIs & Integrations

### Internal and external APIs used

- Internal API:
- REST API under `/api/v1/jobs`

- External libraries and integrations:
- Ultralytics YOLO local model inference
- ByteTrack configuration through Ultralytics tracking
- OpenCV video decode, rendering, MP4 writing, and desktop UI

- External SaaS services:
- None found in the repository

### Endpoints with purpose and sample requests/responses

#### `POST /api/v1/jobs`

- Purpose: upload a video and start processing.
- Input type: `multipart/form-data`

Example request:

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

Example response shape:

```json
{
  "job_id": "5f5f0c4a1d6f4c22b9381f4e7f5b9c7d",
  "status": "queued",
  "created_at": "2026-03-27T10:00:00+00:00",
  "updated_at": "2026-03-27T10:00:00+00:00"
}
```

#### `GET /api/v1/jobs/{job_id}`

- Purpose: poll overall job status and inspect artifacts/exports.

Example response shape:

```json
{
  "job_id": "5f5f0c4a1d6f4c22b9381f4e7f5b9c7d",
  "status": "completed",
  "created_at": "2026-03-27T10:00:00+00:00",
  "updated_at": "2026-03-27T10:05:00+00:00",
  "error": null,
  "options": {
    "model_path": "yolo26x.pt",
    "pose_model_path": "yolo26x-pose.pt",
    "conf_threshold": 0.35,
    "center_alpha": 0.25,
    "size_alpha": 0.15,
    "lost_buffer": 10
  },
  "video_info": {
    "path": "/abs/path/to/video.mp4",
    "width": 1280,
    "height": 720,
    "fps": 30.0,
    "total_frames": 1000,
    "file_size": 12345678,
    "mtime_ns": 1234567890123456789
  },
  "artifacts": {},
  "exports": {}
}
```

#### `GET /api/v1/jobs/{job_id}/artifacts`

- Purpose: list available artifacts for a processed job.

Example response shape:

```json
{
  "artifacts": [
    {
      "artifact_id": "track_cache",
      "artifact_type": "track_cache",
      "file_name": "tracks.json",
      "content_type": "application/json",
      "size_bytes": 12345,
      "relative_path": "cache/tracks.json"
    }
  ]
}
```

#### `GET /api/v1/jobs/{job_id}/artifacts/{artifact_id}`

- Purpose: download an input video, manifest, cache, or export artifact.
- Response type: file download via `FileResponse`

#### `GET /api/v1/jobs/{job_id}/frames/{frame_idx}`

- Purpose: inspect cached metadata for one frame after processing completes.

Example response shape:

```json
{
  "job_id": "5f5f0c4a1d6f4c22b9381f4e7f5b9c7d",
  "frame_idx": 0,
  "tracks": [],
  "pose_entries": [],
  "ball_tracks": []
}
```

#### `POST /api/v1/jobs/{job_id}/exports`

- Purpose: create an async export from cached metadata.
- Input type: JSON

Example request:

```json
{
  "mode": "spotlight",
  "selected_track_id": 7
}
```

Example response shape:

```json
{
  "export_id": "43d8b0a5c1a2",
  "status": "queued",
  "created_at": "2026-03-27T10:06:00+00:00",
  "updated_at": "2026-03-27T10:06:00+00:00",
  "error": null,
  "mode": "spotlight",
  "selected_track_id": 7,
  "multi_track_ids": [],
  "text_labels": {},
  "selected_ball_track_id": null,
  "artifact": null
}
```

#### `GET /api/v1/jobs/{job_id}/exports/{export_id}`

- Purpose: poll export status and output artifact metadata.

#### `GET /api/v1/jobs/{job_id}/exports/{export_id}/download`

- Purpose: download the rendered MP4 once the export completes.

#### `GET /healthz`

- Purpose: liveness check.

#### `GET /readyz`

- Purpose: readiness check based on whether the configured storage root exists and is a directory.

### Third-party services and how they are configured

- Ultralytics models are configured by filesystem path, not by remote service configuration.
- Player tracking uses the detection model with `classes=[0]`.
- Ball tracking uses the same detection model with `classes=[32]`.
- Pose estimation uses a separate pose model file.
- ByteTrack is configured inline through Ultralytics with `tracker="bytetrack.yaml"`.
- No credentials, API keys, or managed external services are defined in the codebase.

## 6. Database & Data Models

### Database type and structure

- No traditional database is present.
- The system uses filesystem-based persistence rooted at `ATHLEO_STORAGE_ROOT` or `.athleo_jobs/`.
- Each job gets its own directory containing:
- `input/`
- `cache/`
- `exports/`
- `manifest.json`

### Key tables/collections and relationships

- Not applicable in the relational or document-database sense.
- The equivalent persisted records are:
- Job manifest
- Processing caches
- Export artifacts

Job storage layout:

```text
<storage-root>/
  <job_id>/
    manifest.json
    input/
      <uploaded-video>.mp4
    cache/
      tracks.json
      pose.json
      balls.json
    exports/
      <export_id>_<mode>.mp4
```

### Important persisted models

- `manifest.json`
- Stores job status, timestamps, input SHA-256, processing options, `video_info`, base artifacts, and export records.

- `tracks.json`
- Stores cache version, video metadata, tracking metadata, and a frame-indexed dictionary of smoothed player boxes.

- `pose.json`
- Stores pose cache metadata and frame-indexed pose detections with `matched_track_id`.

- `balls.json`
- Stores ball tracking metadata and frame-indexed tracked ball boxes.

### Migration or seeding instructions

- None. There are no migrations, seeds, or schema management tools in this repository.

## 7. Deployment & DevOps

### Deployment process

- No production deployment manifests, container definitions, or infrastructure-as-code files were found.
- The documented local deployment path is:
- Install dependencies
- Set `ATHLEO_STORAGE_ROOT` if needed
- Run `uvicorn athleo.api.app:app --reload`

### CI/CD pipelines

- No CI/CD configuration files were found in the repository.
- Assumption: deployment and pipeline automation have either not been set up yet or live outside this repo.

### Environments

- The code distinguishes only between local/default storage and an optional custom storage root.
- No explicit `dev`, `staging`, or `production` configs were found.

## 8. Testing

### Testing strategy

- Tests focus on API orchestration and storage behavior rather than real model inference.
- `tests/test_api.py` uses a `FakeBackend` so API lifecycle tests do not depend on OpenCV or Ultralytics inference.
- Coverage currently includes:
- Health and readiness endpoints
- Job creation and polling
- Artifact listing and download
- Frame metadata retrieval
- Export validation, polling, and download
- Storage path sanitization and manifest persistence

### How to run tests

```bash
python3 -m unittest discover -s tests
```

### Known gaps in testing

- No tests for actual YOLO inference behavior.
- No tests for `athleo/desktop.py` interactive player behavior.
- No tests for overlay rendering correctness across all modes.
- No tests for OpenCV export codec compatibility.
- No tests for concurrent access, job cancellation, or manifest corruption recovery.

## 9. Known Issues & Technical Debt

- The desktop runtime is large and multi-responsibility. `athleo/desktop.py` contains CLI parsing, processing, rendering, exporting, and UI code in one module.
- Job execution is in-process only. If the API process restarts, queued/running jobs do not have a recovery mechanism beyond persisted artifacts already written to disk.
- Readiness is shallow. `/readyz` only checks storage directory existence, not model availability, disk capacity, or executor health.
- Artifact path resolution trusts relative paths stored in the manifest and does not explicitly enforce that the resolved path stays under the job root.
- There is no authentication or authorization on the API endpoints.
- There is no deletion, retention, or cleanup workflow for old jobs and exports.
- Export and playback depend on OpenCV codec availability, which can vary by machine.
- The pose pipeline links detections to tracked players using IoU per frame only, so identity matching can degrade under occlusion or detection drift.

## 10. Security Considerations

### Authentication/authorization mechanisms

- None are implemented in the API.
- Any client with network access to the service can create jobs, inspect jobs, and download artifacts.

### Sensitive data handling

- Uploaded videos are written to disk in the job workspace.
- SHA-256 hashes are recorded for cache reuse matching.
- No encryption-at-rest, signed URLs, or access controls are present in the codebase.
- Filenames are sanitized before being written under job storage.
- No secrets or environment-based credentials are defined in the repository.

### Additional security notes

- Because artifacts are downloadable by job and artifact ID only, a production deployment would need network-layer protection or application-layer auth before exposure outside a trusted environment.
- If this backend will process customer footage, retention and deletion rules should be added before production use.

## 11. Troubleshooting Guide

- `Startup failed: Model weights file was not found`
- Ensure `yolo26x.pt` and `yolo26x-pose.pt` exist at the configured paths.

- `Ultralytics is not installed`
- Install dependencies from `requirements.txt` in the active environment.

- `ByteTrack requires the 'lap' package`
- Ensure `lap` installed successfully in the same environment as the running process.

- `/readyz` returns `not_ready`
- Confirm that `ATHLEO_STORAGE_ROOT` points to a writable directory and that the directory exists.

- Export creation returns `422`
- Check that the request includes the required selection fields for the chosen mode.
- Examples:
- `spotlight`, `pose_skeleton`, `zoom_follow`, and `single_box` require `selected_track_id`
- `pair_link` needs at least 2 `multi_track_ids`
- `polygon_area` needs at least 3 `multi_track_ids`
- `player_text` needs non-empty `text_labels`
- `ball_fire` needs `selected_ball_track_id`

- Frame metadata endpoint returns `409`
- The job has not completed yet. Poll the job until status is `completed`.

- Playback opens but has no overlays
- The caches may be valid but empty. The code explicitly allows opening the video even if `frames_index` is empty.

- MP4 export fails
- OpenCV may not have a working MP4 codec on the current machine. The code tries `mp4v` and `avc1` only.

## 12. Future Improvements

- Split `athleo/desktop.py` into smaller modules for processing, rendering, exporting, and desktop UI.
- Add API authentication and artifact access controls.
- Add cleanup and retention management for job storage.
- Add structured logging and richer job progress reporting.
- Improve readiness and health checks to cover model availability and writable storage.
- Add resumable or recoverable background job execution beyond an in-process thread pool.
- Add stronger path-safety validation when resolving artifact download paths.
- Expand tests to cover rendering and runtime edge cases.
- Existing code comments in `athleo.desktop.main()` also suggest:
- BoT-SORT support
- ReID for identity recovery after occlusions
- PyQt desktop UI instead of HighGUI
- FFmpeg or PyAV for better seeking/export behavior
- Team-color classification and richer sports semantics

## 13. Ownership & Notes

### Important context not obvious from code

- The FastAPI backend is not a separate implementation. It is a wrapper around the same runtime used by the desktop player.
- Cache reuse happens at two levels:
- Inside the desktop/runtime layer, caches are reused when local cache metadata matches the current video and settings.
- Inside `JobManager`, previously completed backend jobs can seed a new job if the uploaded bytes hash and processing options match exactly.
- Overlay exports intentionally reuse cached metadata and never rerun tracking.

### Assumptions and decisions made during this handover

- No database, CI/CD, deployment manifests, staging environments, or auth systems were documented because none were found in the repository.
- Sample API payloads above are representative shapes derived from the Pydantic schemas, README examples, and tests rather than captured live responses from a running server.
- The repository appears intended for local or trusted-network use in its current form; production hardening concerns are called out explicitly because the code does not yet address them.
