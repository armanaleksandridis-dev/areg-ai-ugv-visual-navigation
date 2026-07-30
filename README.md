# Visual Localization — Areg.ai Test Task

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
python3 ugv_visual_navigation.py --video rosbag2_2026_07_22-15_40_59__rgb_image.mp4 --scale 0.5 --headless --skip 1 --log-every 1 --save-video --set max_map_points=8000 --set lost_grace=8 --set reinit_after=20 --set max_jump=5.0 --set init_parallax=2.5 --max-features=2000
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
