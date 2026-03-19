# Athleo

Local Python desktop app for offline soccer player tracking plus live interactive playback with cached overlays.

## Requirements

- Python 3.11+
- Local model weights at `./yolo26x.pt` and `./yolo26x-pose.pt` by default
- OpenCV window support on the machine running the app

Install `requirements.txt` before running a fresh tracking pass. Ultralytics uses ByteTrack here, so `lap` is included as a required dependency.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Prompt for the video path:

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

## Workflow

1. Open a local video.
2. On the first run, Athleo performs a full player tracking pass with `yolo26x.pt`, a pose pass with `yolo26x-pose.pt`, and a ball tracking pass with `yolo26x.pt` class `32` (`sports ball`).
3. The app smooths the tracked player and ball boxes offline and writes 3 caches beside the video:
   - `<input_stem>.tracks.json`
   - `<input_stem>.pose.json`
   - `<input_stem>.balls.json`
4. On later runs, if those caches still match the video and model settings, Athleo loads them directly and skips the expensive preprocessing.
5. The original video then starts playing in an OpenCV window at source FPS.
6. Overlays are drawn live from cached player, pose, and ball metadata only.
7. Clicking a player or ball switches the current mode instantly without rerunning detection.
8. Special overlay modes can isolate a player box, draw a spotlight, connect two players, draw a matched pose skeleton, reframe the shot around one player, or turn a selected tracked ball into a procedural fireball.

## Interactive Controls

- Left click: select the clicked player on the current frame
- Right click: clear the current selection; in spotlight, pair-link, pose, zoom, or ball mode it stays in that mode
- `Space`: pause or resume playback
- Left arrow: step backward one frame when paused
- Right arrow: step forward one frame when paused
- `A`: force all-player mode
- `C`: clear current selection
- `S`: enter spotlight mode, then click one player to draw a triangular sky spotlight instead of a box
- `P`: enter pair-link mode, then click two players to highlight them with yellow boxes, red ground rings, and a red connecting line
- `K`: enter pose mode, then click one player to draw the cached `yolo26x-pose.pt` skeleton for the matched tracked player
- `Z`: enter zoom mode, then click one player to crop-follow that player and scale the crop back to full-frame output
- `B`: enter ball mode, then click a tracked ball to replace it with a procedural fireball effect
- `Esc`: leave spotlight, pair-link, pose, zoom, or ball mode and return to all-player mode
- `J`: jump to a frame by entering the frame number in the terminal
- `E`: export the current overlay mode to MP4
- `Q`: quit

If no player is selected, the player shows all tracked boxes. In the default selection flow, clicking a player hides the rest and keeps only that player box visible. In spotlight mode, selecting a player hides the boxes and draws only a transparent triangular beam from the top of the frame to that player. In pair-link mode, Athleo keeps the other tracked boxes unchanged, highlights the two selected players with yellow boxes, places broadcast-style red ground rings at their feet, and connects those ground anchors with a strong red line. In pose mode, the player click uses the existing tracked `track_id` and then looks up the cached matched pose for that player on each frame. In zoom mode, the player click switches the output to a smoothed crop-follow camera centered on the selected player. In ball mode, the player shows tracked ball markers until you pick one, then swaps the ball for a stylized fireball with a glowing head and motion trail.

## Outputs

Outputs are written beside the input video:

- `<input_stem>.tracks.json`
- `<input_stem>.pose.json`
- `<input_stem>.balls.json`
- `annotated_all.mp4`
- `annotated_selected_<track_id>.mp4`
- `annotated_spotlight_<track_id>.mp4`
- `annotated_pair_<track_id_a>_<track_id_b>.mp4`
- `annotated_pose_<track_id>.mp4`
- `annotated_zoom_<track_id>.mp4`
- `annotated_fireball_<ball_track_id>.mp4`

Export always uses the original video plus cached tracking data. It does not rerun tracking.

## Notes

- The tracking pass is intentionally separate from live playback so the UI only has to decode frames and draw overlays.
- Cached `track_id` values are the source of truth for instant player isolation, pose matching, zoom follow, and ball/fireball selection.
- `detect.py` remains in the repo as the older one-off detection script; `main.py` is the interactive desktop workflow entrypoint.
