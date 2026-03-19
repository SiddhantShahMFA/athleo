# Athleo

Local Python desktop app for offline soccer player tracking plus live interactive playback with cached overlays.

## Requirements

- Python 3.11+
- Local model weights at `./yolo26x.pt` by default
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
  --conf 0.35 \
  --center-alpha 0.25 \
  --size-alpha 0.15 \
  --lost-buffer 10
```

## Workflow

1. Open a local video.
2. On the first run, Athleo performs one full Ultralytics tracking pass with `persist=True` and `tracker="bytetrack.yaml"`.
3. The app keeps only `person` detections, smooths each `track_id` offline, and writes a cache beside the video as `<input_stem>.tracks.json`.
4. On later runs, if the cache still matches the video and tracking settings, Athleo loads it directly and skips tracking.
5. The original video then starts playing in an OpenCV window at source FPS.
6. Overlays are drawn live from the cached tracking metadata only.
7. Clicking a player instantly switches the overlay mode to that `track_id` without rerunning tracking or re-rendering the video.
8. Special overlay modes can swap the selected-player box for a spotlight or connect two selected players with a line.

## Interactive Controls

- Left click: select the clicked player on the current frame
- Right click: clear the current selection; in spotlight or pair-link mode it stays in that mode
- `Space`: pause or resume playback
- Left arrow: step backward one frame when paused
- Right arrow: step forward one frame when paused
- `A`: force all-player mode
- `C`: clear current selection
- `S`: enter spotlight mode, then click one player to draw a triangular sky spotlight instead of a box
- `P`: enter pair-link mode, then click two players to highlight them with yellow boxes, red ground rings, and a red connecting line
- `Esc`: leave spotlight or pair-link mode and return to all-player mode
- `J`: jump to a frame by entering the frame number in the terminal
- `E`: export the current overlay mode to MP4
- `Q`: quit

If no player is selected, the player shows all tracked boxes. In the default selection flow, clicking a player hides the rest and keeps only that player box visible. In spotlight mode, selecting a player hides the boxes and draws only a transparent triangular beam from the top of the frame to that player. In pair-link mode, Athleo keeps the other tracked boxes unchanged, highlights the two selected players with yellow boxes, places broadcast-style red ground rings at their feet, and connects those ground anchors with a strong red line.

## Outputs

Outputs are written beside the input video:

- `<input_stem>.tracks.json`
- `annotated_all.mp4`
- `annotated_selected_<track_id>.mp4`
- `annotated_spotlight_<track_id>.mp4`
- `annotated_pair_<track_id_a>_<track_id_b>.mp4`

Export always uses the original video plus cached tracking data. It does not rerun tracking.

## Notes

- The tracking pass is intentionally separate from live playback so the UI only has to decode frames and draw overlays.
- Cached `track_id` values are the source of truth for instant player isolation.
- `detect.py` remains in the repo as the older one-off detection script; `main.py` is the interactive desktop workflow entrypoint.
