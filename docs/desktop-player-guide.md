# Athleo Desktop Player Guide

This document explains how the desktop player in [athleo/desktop.py](/Users/siddhantshah/Desktop/Projects/athleo/athleo/desktop.py) works from two perspectives:

- a high-level view for prospects, clients, and stakeholders
- a lower-level implementation view for developers

## Executive Summary

Athleo's desktop player is a local video analysis workflow for soccer footage. It opens a source video, runs AI analysis to detect and track players, optionally estimates player pose, optionally tracks the ball, stores those results as reusable JSON caches, and then launches an interactive OpenCV player where a user can click subjects, switch overlay modes, and export annotated videos.

The most important product behavior is that the expensive AI work happens up front and is cached. After that, playback interactions and exports reuse cached metadata instead of rerunning detection every time. That makes the experience feel immediate once preprocessing is complete.

## High-Level Story For Prospects And Stakeholders

### What problem it solves

Sports video review often breaks down into two painful steps:

1. identifying the people or objects of interest frame by frame
2. recreating multiple presentation views from the same clip

The desktop player turns that into a single workflow:

1. load the match clip once
2. let Athleo analyze and track the subjects
3. interactively choose how to present the action
4. export multiple visual treatments from the same cached analysis

### What the user experiences

At a high level, the flow is:

1. The operator launches the desktop app through [main.py](/Users/siddhantshah/Desktop/Projects/athleo/main.py).
2. Athleo validates the input video and model paths.
3. Athleo reads basic video metadata such as frame count, resolution, and FPS.
4. Athleo checks whether matching caches already exist for player tracking, pose estimation, and ball tracking.
5. If the caches are missing or stale, Athleo runs AI analysis and writes new JSON cache files beside the video.
6. Athleo opens an interactive playback window.
7. The user clicks tracked players or the ball and changes modes with keyboard shortcuts.
8. Athleo renders overlays in real time using the cached metadata.
9. The user exports the current view to MP4 without rerunning the heavy analysis step.

### Why this matters commercially

- Faster review cycles: the same analyzed clip can be reused across many exports.
- Better storytelling: the same underlying tracking data can drive multiple visual styles.
- Lower operator effort: users click to select subjects instead of redrawing overlays manually.
- Clear upgrade path: the desktop player already separates analysis, metadata, rendering, and export, which makes the product easier to evolve into service and API-based workflows.

### What the different overlay modes communicate

- `all`: show every tracked player box
- `single_box`: isolate one player visually
- `spotlight`: create a broadcast-style spotlight beam on a selected player
- `pair_link`: connect selected players in an ordered chain
- `player_text`: attach moving labels to selected players
- `polygon_area`: create a shaded area between selected players
- `pose_skeleton`: draw a 17-keypoint pose skeleton for one selected player
- `zoom_follow`: crop and enlarge the frame around a selected player
- `ball_fire`: either show tracked balls or apply a stylized fireball effect to a selected ball

### Practical value for clients

This makes Athleo useful in several client conversations:

- coaching and tactical review
- player spotlight packages
- social content generation
- analyst-assisted presentation workflows
- prototype-to-product demonstrations where the same engine can later power backend exports

## End-To-End Runtime Flow

### 1. Entry point

The desktop flow starts in [main.py](/Users/siddhantshah/Desktop/Projects/athleo/main.py), which simply imports and calls `athleo.desktop.main()`.

### 2. Argument parsing and configuration

`parse_args()` and `resolve_config()` collect runtime inputs:

- video path
- player model path
- pose model path
- confidence threshold
- smoothing values for box center and size
- lost-buffer window
- optional track cache path

`resolve_config()` also derives related paths automatically:

- `<video>.tracks.json`
- `<video>.pose.json`
- `<video>.balls.json`
- `annotated_all.mp4`

Mode-specific exports derive their own names later, such as:

- `annotated_selected_<track_id>.mp4`
- `annotated_spotlight_<track_id>.mp4`
- `annotated_pair_<ids>.mp4`
- `annotated_pose_<track_id>.mp4`
- `annotated_zoom_<track_id>.mp4`
- `annotated_fireball_<track_id>.mp4`

### 3. Video metadata inspection

`load_video_info()` opens the video with OpenCV and reads:

- width
- height
- frames per second
- total frame count
- file size
- modification timestamp

That metadata is used for both playback/export setup and cache validation. If FPS metadata is missing or invalid, the code falls back to `30.0`.

### 4. Cache reuse decision

Before running new inference, the app checks whether previously written caches still match the current run. Matching is strict:

- same video path
- same video dimensions
- same frame count
- same file size
- same file modification timestamp
- same FPS
- same model path and model file name
- same confidence and smoothing settings
- same tracker configuration

This is handled by:

- `video_meta_matches()`
- `cache_matches()`
- `pose_cache_matches()`
- `ball_cache_matches()`
- `load_track_cache()`
- `load_pose_cache()`
- `load_ball_cache()`

If the cache is unreadable or no longer matches, Athleo prints a message and rebuilds it.

### 5. Player tracking pass

`run_tracking()` performs person tracking with Ultralytics YOLO using:

- `model.track(...)`
- `tracker="bytetrack.yaml"`
- `classes=[0]` for the person class

For each frame, Athleo collects:

- `frame_idx`
- `track_id`
- `class_id`
- `confidence`
- `x1`, `y1`, `x2`, `y2`

The raw detections are then smoothed by `smooth_tracks()`, which applies exponential moving averages to:

- box center using `center_alpha`
- box width and height using `size_alpha`

It also keeps a track alive for a short time using `lost_buffer`, so a recently missing player can still be shown as a held box with `is_lost_buffer=True`.

The output is stored as a frame-indexed JSON structure.

### 6. Pose estimation pass

`run_pose_estimation()` uses the separate pose model with `model.predict(...)`. It does not independently track pose identities over time. Instead, for each pose detection in a frame, it finds the best-matching already-tracked player box using IoU:

- `box_iou()`
- `find_best_matching_track_id()`

That means the player-tracking cache acts as the identity backbone, and the pose pass attaches 17-keypoint skeleton data to those identities when it can match them confidently enough.

Each pose entry stores:

- the pose bounding box
- overall pose confidence
- `matched_track_id`
- `keypoints`
- `keypoint_confidences`

### 7. Ball tracking pass

`run_ball_tracking()` is structurally similar to player tracking, but it uses:

- the same detection model
- `classes=[32]` for the ball class

It also writes a separate smoothed cache so ball overlays and exports can be generated later without rerunning inference.

### 8. Interactive playback

After caches are ready, `InteractivePlayer` opens the source video and starts an OpenCV HighGUI loop:

- `cv2.namedWindow(...)`
- `cv2.setMouseCallback(...)`
- `cv2.imshow(...)`
- `cv2.waitKeyEx(...)`

The player keeps two things in sync:

- the current decoded video frame
- the cached metadata for that frame

On each display iteration, Athleo:

1. gets the current frame's player, pose, and ball metadata
2. renders the selected overlay mode
3. draws the status/help text
4. waits for keyboard input or mouse clicks
5. advances to the next frame according to the original FPS

### 9. Selection and interaction model

Mouse interactions:

- left click selects a player or ball depending on the active mode
- right click clears the current selection while staying in the active special mode when appropriate

Keyboard interactions:

- `Space`: pause or resume
- `Left` / `Right`: step frames when paused
- `A`: reset to all-player mode
- `C`: clear current selection
- `S`: spotlight mode
- `P`: multi-link mode
- `T`: text mode
- `G`: area mode
- `K`: pose mode
- `Z`: zoom mode
- `B`: ball mode
- `J`: jump to a frame by typing a frame number in the terminal
- `E`: export current mode
- `Q`: quit
- `Esc`: leave special mode and return to all-player mode

### 10. Export pipeline

When the user exports, `export_current_mode()` validates that the necessary selection state exists for that mode. For example:

- spotlight needs one selected player
- pair link needs at least two selected players
- area needs at least three selected players
- ball fire needs one selected ball

It then calls `export_video()`, which:

1. reopens the original source video
2. creates an MP4 writer
3. reads every frame in sequence
4. re-renders the selected overlay using cached metadata
5. writes the rendered frames to a new MP4

This is a key design point: export replays rendering over the original footage, but it does not redo tracking, pose estimation, or ball detection.

## Developer-Level Architecture

### Core modules inside `desktop.py`

The file has four main responsibilities:

1. configuration and cache management
2. AI analysis passes
3. overlay rendering
4. interactive player state management

### Configuration and path conventions

`AppConfig` is the central immutable runtime object. It carries the resolved paths and runtime knobs needed by every downstream stage.

This is helpful for future refactors because analysis and playback already depend on one shared config shape rather than ad hoc arguments.

### Cache shape and semantics

All caches are JSON files with a common pattern:

- `cache_version`
- `created_at`
- `video`
- analysis-specific metadata such as `tracking` or `pose`
- `frames`

`frames` is a dictionary keyed by frame index. Each value is a list of records for that frame.

The use of frame-indexed JSON makes the cache easy to inspect, simple to serialize, and friendly for reuse in both desktop and backend flows. The tradeoff is that JSON is not the most compact or highest-throughput storage format for very large workloads.

### Tracking smoothing behavior

The smoothing layer is a notable implementation choice. Raw detector output can jitter noticeably frame to frame, especially in sports footage. `smooth_box()` and `smooth_tracks()` stabilize the visual overlays by smoothing:

- the box center independently
- the box dimensions independently

This matters because the overlay system depends on box coordinates for nearly every mode. Better-smoothed boxes improve:

- selected-player highlighting
- spotlight geometry
- pair-link anchors
- polygon stability
- zoom framing

The `lost_buffer` feature also improves perceived continuity by holding the last smoothed box for a configurable number of missing frames.

### Pose-to-track matching

Pose estimation is intentionally decoupled from identity tracking. The code does not run a dedicated pose tracker. Instead, it reuses the player-tracking cache as the identity source and matches pose boxes to tracked players per frame by IoU.

Benefits:

- simpler implementation
- consistent player IDs across overlays
- easier reuse of the existing tracking cache

Limitations:

- if the pose box and player box diverge too much, matching can fail
- pose identity depends on the quality of the tracked player boxes in that frame

### Overlay renderer

`render_overlay()` is the central compositor. It takes:

- the raw frame
- current frame track metadata
- current mode
- current selection state
- optional pose and ball metadata

It then routes to the correct renderer for the mode.

Important helper functions include:

- `draw_track_box()`
- `draw_spotlight_overlay()`
- `draw_multi_link()`
- `draw_pair_ground_ring()`
- `draw_area_polygon()`
- `draw_track_text_label()`
- `draw_pose_overlay()`
- `draw_zoom_overlay()`
- `draw_ball_marker()`
- `draw_fireball_overlay()`

This structure is clean enough that new overlay modes can usually be added by:

1. creating a renderer helper
2. wiring a new mode constant
3. extending `render_overlay()`
4. extending input handling and export-path selection

### Zoom-follow implementation

The zoom mode does not simply crop the current frame around the current box. It calculates a smoothed view window across recent frames using:

- `smoothed_track_window()`
- `zoom_crop_bounds()`

This reduces abrupt camera jumps and keeps the zoomed crop more watchable.

### Ball-fire implementation

Ball mode supports two states:

- no ball selected: show tracked ball markers
- ball selected: render a stylized fireball effect with a recent movement trail

The fireball look is based on recent center points from the ball cache and layered circles and polygons blended over the current frame. It is not physics-based; it is a purely visual effect driven by tracked positions.

### Interactive player state machine

`InteractivePlayer` is effectively a small state machine. The key mutable state is:

- `current_frame_idx`
- `paused`
- `mode`
- `selected_track_id`
- `selected_ball_track_id`
- `multi_track_ids`
- `text_track_labels`
- current frame-local metadata lists

The class divides responsibilities across:

- startup and teardown: `open()`, `close()`
- event handling: `on_mouse()`, `handle_key()`
- state transitions: `reset_to_all_mode()`, `activate_special_mode()`, `clear_current_selection()`
- mode-specific selection helpers: `update_multi_selection()`, `prompt_for_text_label()`
- playback navigation: `step_frame()`, `jump_to_frame()`, `load_frame()`
- exporting: `export_current_mode()`

### Playback timing

Playback pacing uses `frame_duration` derived from source FPS and a `next_frame_deadline` timer. This is a lightweight approach that keeps the player close to source timing without introducing a more complex media framework.

### Error handling and failure modes

The code is defensive in several useful places:

- it validates file existence and numeric arguments early
- it falls back to default FPS if metadata is bad
- it ignores unreadable or stale caches and rebuilds them
- it reports missing dependencies like `ultralytics` or `lap`
- it handles first-frame and seek failures explicitly
- it pauses playback rather than crashing on decode issues in the main loop

### Coupling with the backend

Even though this document focuses on the desktop player, the file already serves as shared processing infrastructure for the backend path too. The functions:

- `run_tracking()`
- `run_pose_estimation()`
- `run_ball_tracking()`
- `export_video()`

are reused by [athleo/backend.py](/Users/siddhantshah/Desktop/Projects/athleo/athleo/backend.py). That is important for product planning because the desktop experience and the API workflow are aligned around the same cache-and-render model rather than separate engines.

## Important Implementation Notes For Developers

### Dependencies

The desktop flow relies on:

- OpenCV (`cv2`) for video I/O, rendering, UI, and export
- NumPy for geometry helpers and deterministic colors
- Ultralytics YOLO for detection, tracking, and pose inference
- ByteTrack through `bytetrack.yaml`

### Class usage assumptions

- player tracking uses class `0`
- ball tracking uses class `32`
- pose estimation assumes 17 keypoints

If the model taxonomy changes, those values must stay aligned with the trained weights.

### Output naming assumptions

Export file naming is deterministic and mode-dependent. If product requirements change around job IDs, user IDs, or storage roots, the helper functions near the top of the file are the first place to update.

### UX limitations to be aware of

- The desktop player depends on OpenCV window support.
- Label entry and frame jump prompts use terminal `input()`, not an in-window form.
- Export writes MP4 through OpenCV codecs only, so behavior can vary across environments.
- Very large videos may make JSON caches heavy.
- The desktop UI is functional but intentionally lightweight rather than polished.

### Natural extension points

The code itself already points to likely next steps:

- alternate tracker support such as BoT-SORT
- ReID for identity recovery after longer occlusions
- PyQt or another richer desktop UI instead of HighGUI
- FFmpeg or PyAV for improved seeking and export behavior
- team-color classification
- role-aware tracking for referees, goalkeepers, and other classes
- pitch masking and field-aware filtering

## Suggested Talking Track For External Sharing

If this needs to be shown to a prospect, the safest concise narrative is:

1. Athleo analyzes a sports clip once and builds reusable tracking metadata.
2. That metadata powers an interactive review experience where a user can instantly switch between player- and ball-focused storytelling modes.
3. Exports reuse the cached analysis, so the same clip can produce multiple presentation outputs without repeating the heavy AI work.
4. The same underlying runtime is already aligned with Athleo's backend architecture, which supports a clean path from desktop demo to scalable product workflow.

## File References

- Desktop implementation: [athleo/desktop.py](/Users/siddhantshah/Desktop/Projects/athleo/athleo/desktop.py)
- Desktop entrypoint: [main.py](/Users/siddhantshah/Desktop/Projects/athleo/main.py)
- Shared backend reuse: [athleo/backend.py](/Users/siddhantshah/Desktop/Projects/athleo/athleo/backend.py)
- Existing product overview: [README.md](/Users/siddhantshah/Desktop/Projects/athleo/README.md)
