# Monocular Visual Localization — Areg.ai Test Task

Sequential monocular visual odometry on OpenCV (KLT tracking + Essential
init + PnP tracking against a local 3D map + keyframe relocalization). See
`APPROACH.md` for the write-up.

## Video links

- Source video: https://drive.google.com/file/d/1V99JtQDdRgFUuPsN_GS1L7GYwlODocBQ/view
- Result (trajectory, 89% tracked): https://drive.google.com/file/d/1exC7ZfPoWWgmmQwAEMDU3DpD2jq8M_uQ/view?usp=sharing

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 ugv_visual_navigation.py --video path/to/video.mp4 --calib config.yaml --headless --save-video
```

Outputs: `video_output/*.mp4` (visualization), `trajectory_output/*.csv`
(`frame,x,y,z`). Drop `--headless` for a live `cv2.imshow` window.

Useful flags: `--scale 0.5`, `--skip 1`, `--max-features 500`,
`--set key=value` (overrides `CFG` in the code).

## Files

- `ugv_visual_navigation.py` — main script
- `config.yaml` — camera calibration (fill in fx, fy, cx, cy, dist)
- `requirements.txt` — dependencies
- `APPROACH.md` — method, decisions, limitations

At the end of a run the script prints tracking-state percentages and three
FPS figures (VO-only / pipeline / wall-clock) with a real-time verdict
against the camera's actual frame rate.
