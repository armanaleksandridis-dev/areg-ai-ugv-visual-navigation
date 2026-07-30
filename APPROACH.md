# Approach — Visual Localization

## Pipeline

1. **Undistort** — remove lens distortion using the provided calibration.
2. **Initialization** — detect Shi-Tomasi corners, track them with KLT
   optical flow; once there's enough parallax, recover relative pose via the
   Essential matrix (RANSAC) and triangulate the first 3D map points.
3. **Tracking** — new frames track existing map points via KLT, pose is
   estimated with `solvePnPRansac` against the 3D map (not just chaining
   frame-to-frame poses — less drift, and gives something to relocalize
   against later).
4. **Keyframes & map growth** — new keyframes are added when tracked points
   drop or the camera moves enough; new map points are triangulated between
   the current frame and a recent keyframe.
5. **Loss & relocalization** — if tracking degrades, state becomes `LOST`;
   relocalization matches ORB descriptors against recent keyframes and
   verifies the pose with PnP.
6. **Re-init after prolonged loss** — if relocalization doesn't recover in
   time, falls back to two-view re-initialization, glued onto the last known
   pose.
7. **Visualization** — tracked points + status banner on the frame, plus a
   top-down 2D plot of the accumulated trajectory.

## Key decisions

- **PnP tracking against a local 3D map** instead of chaining frame-to-frame
  poses — less accumulated drift, and gives a concrete target for
  relocalization.
- **KLT for tracking, ORB only for relocalization** — KLT is cheap for
  same-point frame-to-frame motion; ORB is used only where wide-baseline
  matching against older keyframes is needed.
- **Sky/ceiling cut mask** — the upper part of the frame is excluded from
  feature detection, since it's low-texture and destabilizes pose
  estimation in this scene.
- **No absolute scale** — allowed by the task; scale is unobservable from a
  single camera without extra sensors.
- **Honest FPS reporting** — three separate FPS figures (VO-only, full
  pipeline, wall-clock), compared directly against the camera's real frame
  rate for a real-time / not-real-time verdict.

## Known limitations

  VO, not a full SLAM back-end; drift is never globally corrected.
- Re-initialization after long tracking loss is a visual stitch, not a
  geometrically verified loop closure — drift during the loss gets baked in.
- Large flat, low-texture surfaces (solar panels) are the main cause of
  tracking loss and failed relocalization in this dataset.
- Monocular scale can drift slowly even during nominally good tracking,
  especially on low-parallax (straight-line) segments.
- Single-threaded, CPU-only — no GPU acceleration.

## Result

On the provided video, the pipeline stayed in `TRACKING`/`RELOCALIZED` for
about 89% of processed frames (see `trajectory_output/*.csv` and
`video_output/*.mp4`).
