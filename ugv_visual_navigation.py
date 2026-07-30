from collections import deque
from enum import Enum
import numpy as np
import argparse
import time
import cv2
import os

def load_calibration(path):
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # our own config format
    if "rgb_camera_matrix" in data:
        K = np.array(data["rgb_camera_matrix"], dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.array(data.get("rgb_dist_coeffs", [0] * 5),
                               dtype=np.float64).ravel()
        return K, dist_coeffs

    # ROS camera_info format
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        dist_data = data.get("distortion_coefficients", {}).get("data", [0] * 5)
        return K, np.array(dist_data, dtype=np.float64).ravel()

    # flat fx/fy/cx/cy format
    if "fx" in data:
        K = np.array([[data["fx"], 0, data["cx"]],
                      [0, data["fy"], data["cy"]],
                      [0, 0, 1]], dtype=np.float64)
        return K, np.array(data.get("dist", [0] * 5), dtype=np.float64).ravel()

    raise ValueError(f"Unrecognized calibration format: {path}")


class State(Enum):
    INITIALIZING = "INITIALIZING"   # no map yet, accumulating parallax to start
    TRACKING = "TRACKING"           # normal tracking against the 3D map
    LOST = "LOST"                   # tracking lost, trying to relocalize
    RELOCALIZED = "RELOCALIZED"     # just recovered (flagged for a few frames)


class Keyframe:
    """Reference frame: camera pose plus everything needed to relocalize by it."""
    __slots__ = ("frame_idx", "gray", "R", "t",
                 "descriptors", "keypoints", "map_point_by_kp")

    def __init__(self, frame_idx, gray, R, t, keypoints, descriptors,
                 map_point_by_kp):
        self.frame_idx = frame_idx              # frame index this kf was created on
        self.gray = gray                        # image (used for re-triangulation)
        self.R = R                              # world->camera rotation at this kf
        self.t = t                              # world->camera translation at this kf
        self.keypoints = keypoints              # ORB keypoints
        self.descriptors = descriptors          # ORB descriptors
        # mapping: keypoint index -> 3D map point id
        self.map_point_by_kp = map_point_by_kp

    def camera_center(self):
        """Camera center in world coordinates."""
        return (-self.R.T @ self.t).ravel()


class MonoVO:
    def __init__(self, K, cfg):
        self.K = K                              # camera intrinsic matrix
        self.cfg = cfg
        self.state = State.INITIALIZING
        self.frame_idx = -1

        # current camera pose (world -> camera)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

        # local 3D map and the projections currently being tracked
        self.map_points = np.empty((0, 3), np.float64)    # 3D map points
        self.track_pixels = np.empty((0, 2), np.float32)  # their pixels in current frame
        self.track_ids = np.empty((0,), np.int32)         # ids of the matching 3D points

        # keyframes and bookkeeping counters
        self.keyframes = []
        self.last_keyframe_frame = -10 ** 9
        self.last_inlier_count = 0

        self.init_reference = None              # (gray, corners) reference for two-view init
        self.prev_gray = None                   # previous frame for KLT

        self.trajectory = []                    # history of (frame, camera_center, state)
        self.lost_at_frame = None               # frame index where tracking was lost
        self.reloc_display_left = 0             # frames left to keep the RELOCALIZED flag
        self.consecutive_bad_frames = 0         # length of the current bad-frame streak

        # pose at the moment of loss - the "glue" for reinitialization after LOST
        self.carry_rotation = None
        self.carry_translation = None

        self.orb = cv2.ORB_create(nfeatures=cfg["orb_features"])
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def camera_center(self):
        """Camera center in world coordinates."""
        return (-self.R.T @ self.t).ravel()

    def _mask(self, gray):
        """Region-of-interest mask: cut off the top of the frame (sky/ceiling),

        where features are unstable and hurt tracking.
        """
        h, w = gray.shape
        mask = np.zeros((h, w), np.uint8)
        mask[int(h * self.cfg["sky_cut"]):, :] = 255
        return mask

    def _detect(self, gray, existing_px=None):
        """Detect Shi-Tomasi corners inside the region of interest.

        If already-occupied pixels are passed, zero out the mask around them so
        new points do not clump onto existing ones.
        """
        mask = self._mask(gray)
        if existing_px is not None and len(existing_px):
            for x, y in existing_px:
                cv2.circle(mask, (int(x), int(y)),
                           self.cfg["min_feat_dist"], 0, -1)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.cfg["max_features"],
            qualityLevel=self.cfg["quality"],
            minDistance=self.cfg["min_feat_dist"], mask=mask, blockSize=7)
        if corners is None:
            return np.empty((0, 2), np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

    def _klt(self, prev_gray, gray, points_prev):
        """Propagate points from the previous frame to the current one via flow.

        Returns the new point positions and a boolean mask of valid tracks. To
        reject bad matches we run a forward-backward check: track the flow there
        and back, keeping only points that return close to their origin.
        """
        if len(points_prev) == 0:
            return np.empty((0, 2), np.float32), np.zeros((0,), bool)

        lk_params = dict(
            winSize=(self.cfg["klt_win"], self.cfg["klt_win"]),
            maxLevel=self.cfg["klt_levels"],
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      self.cfg["klt_iters"], 0.03))

        points_next, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, points_prev, None, **lk_params)

        if self.cfg["klt_fb_check"]:
            points_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, points_next, None, **lk_params)
            fb_error = np.linalg.norm(points_prev - points_back, axis=1)
            valid = ((status_fwd.ravel() == 1) & (status_bwd.ravel() == 1)
                     & (fb_error < self.cfg["fb_thresh"]))
        else:
            valid = status_fwd.ravel() == 1

        # drop points that drifted out of the frame
        h, w = gray.shape
        valid &= ((points_next[:, 0] > 0) & (points_next[:, 0] < w)
                  & (points_next[:, 1] > 0) & (points_next[:, 1] < h))
        return points_next, valid

    def _triangulate(self, R1, t1, R2, t2, pts_a, pts_b):
        """Triangulate 3D points from two views and filter obvious garbage.

        Keep only points that are: positive-depth in both cameras, closer than
        max_depth, and reproject with small error into both frames.
        """
        proj1 = self.K @ np.hstack([R1, t1])
        proj2 = self.K @ np.hstack([R2, t2])

        points_homog = cv2.triangulatePoints(proj1, proj2, pts_a.T, pts_b.T)
        w = points_homog[3]
        valid = np.abs(w) > 1e-9
        w = np.where(valid, w, 1.0)
        points_3d = (points_homog[:3] / w).T

        # point coordinates in each camera frame (needed for the depth check)
        coords_cam1 = (R1 @ points_3d.T + t1).T
        coords_cam2 = (R2 @ points_3d.T + t2).T
        valid &= (coords_cam1[:, 2] > 0) & (coords_cam2[:, 2] > 0)
        valid &= ((coords_cam1[:, 2] < self.cfg["max_depth"])
                  & (coords_cam2[:, 2] < self.cfg["max_depth"]))

        # reprojection error into both frames
        for proj, pixels in ((proj1, pts_a), (proj2, pts_b)):
            points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
            projected = (proj @ points_3d_h.T).T
            depths = projected[:, 2]
            depth_ok = np.abs(depths) > 1e-9
            reprojected = np.zeros((len(points_3d), 2))
            reprojected[depth_ok] = projected[depth_ok, :2] / depths[depth_ok, None]
            valid &= depth_ok & (np.linalg.norm(reprojected - pixels, axis=1)
                                 < self.cfg["tri_reproj_px"])
        return points_3d, valid

    def _try_init(self, gray):
        """Try to build the initial map from two views.

        Wait for enough parallax between the reference frame and the current one,
        then recover the relative pose via the Essential matrix and triangulate
        the first map points.
        """
        ref_gray, ref_corners = self.init_reference
        cur_corners, valid = self._klt(ref_gray, gray, ref_corners)

        # too few tracks - take the current frame as the new reference
        if valid.sum() < max(20, self.cfg["min_init_pts"] // 3):
            self.init_reference = (gray.copy(), self._detect(gray))
            return False

        pts_ref, pts_cur = ref_corners[valid], cur_corners[valid]

        # without parallax the two-view geometry cannot be recovered
        parallax = np.median(np.linalg.norm(pts_cur - pts_ref, axis=1))
        if parallax < self.cfg["init_parallax"]:
            return False

        essential, ess_mask = cv2.findEssentialMat(
            pts_ref, pts_cur, self.K, method=cv2.RANSAC,
            prob=0.999, threshold=self.cfg["ransac_px"])
        if essential is None or essential.shape != (3, 3):
            return False

        num_pose_inliers, rel_R, rel_t, pose_mask = cv2.recoverPose(
            essential, pts_ref, pts_cur, self.K, mask=ess_mask)
        if num_pose_inliers < self.cfg["min_init_pts"]:
            return False

        inlier = pose_mask.ravel() > 0
        pts_ref_in, pts_cur_in = pts_ref[inlier], pts_cur[inlier]

        points_3d, good = self._triangulate(
            np.eye(3), np.zeros((3, 1)), rel_R, rel_t, pts_ref_in, pts_cur_in)
        if good.sum() < self.cfg["min_init_pts"]:
            return False

        self.map_points = points_3d[good]
        self.track_pixels = pts_cur_in[good]
        self.track_ids = np.arange(len(self.map_points), dtype=np.int32)

        # Carry-over: if this is a reinitialization after LOST, glue the new
        # segment onto the last known pose so the trajectory does not jump back
        # to (0,0). A monocular system cannot know the real displacement during
        # the loss, so this is a visual stitch, honestly noted as such in the
        # report.
        if self.carry_rotation is not None:
            prev_R = self.carry_rotation
            prev_t = self.carry_translation
            self.R = rel_R @ prev_R
            self.t = rel_t + rel_R @ prev_t
            R_world_from_prev = prev_R.T
            self.map_points = (R_world_from_prev @ self.map_points.T
                               - R_world_from_prev @ prev_t).T
            self.carry_rotation = None
            self.carry_translation = None
        else:
            self.R, self.t = rel_R, rel_t

        self._enforce_track_limit()
        self._add_keyframe(gray)
        self.state = State.TRACKING
        return True

    def _add_keyframe(self, gray):
        """Create a keyframe: compute ORB and link its points to the 3D map."""
        keypoints, descriptors = self.orb.detectAndCompute(gray, self._mask(gray))

        map_point_by_kp = {}
        if keypoints and len(self.track_pixels):
            # for each ORB point find the nearest tracked map point
            kp_xy = np.array([k.pt for k in keypoints], dtype=np.float32)
            dist_sq = ((kp_xy[:, None, :] - self.track_pixels[None, :, :]) ** 2).sum(-1)
            nearest_track = dist_sq.argmin(1)
            nearest_dist_sq = dist_sq[np.arange(len(kp_xy)), nearest_track]
            # link only if the point is within 3px (9 = 3^2)
            for kp_idx in np.nonzero(nearest_dist_sq < 9.0)[0]:
                map_point_by_kp[int(kp_idx)] = int(self.track_ids[nearest_track[kp_idx]])

        self.keyframes.append(Keyframe(
            self.frame_idx, gray.copy(), self.R.copy(), self.t.copy(),
            keypoints, descriptors, map_point_by_kp))
        self.last_keyframe_frame = self.frame_idx

        # cap the keyframe window so memory does not grow without bound
        if len(self.keyframes) > self.cfg["max_keyframes"]:
            self.keyframes.pop(0)

    def _extend_map(self, gray):
        """Triangulate new map points between a recent keyframe and the current frame."""
        if not self.keyframes:
            return

        cam_center = self.camera_center()

        # find a keyframe with enough baseline (otherwise triangulation degenerates)
        ref_kf = None
        for candidate in reversed(self.keyframes):
            baseline = np.linalg.norm(cam_center - candidate.camera_center())
            if baseline >= self.cfg["min_tri_baseline"]:
                ref_kf = candidate
                break
        if ref_kf is None:
            return

        new_corners = self._detect(ref_kf.gray, existing_px=None)
        if len(new_corners) == 0:
            return

        tracked_corners, valid = self._klt(ref_kf.gray, gray, new_corners)
        if valid.sum() < 20:
            return

        pts_ref, pts_cur = new_corners[valid], tracked_corners[valid]

        essential, ess_mask = cv2.findEssentialMat(
            pts_ref, pts_cur, self.K, method=cv2.RANSAC,
            prob=0.999, threshold=self.cfg["ransac_px"])
        if essential is None or ess_mask is None:
            return

        inlier = ess_mask.ravel() > 0
        if inlier.sum() < 20:
            return
        pts_ref, pts_cur = pts_ref[inlier], pts_cur[inlier]

        points_3d, good = self._triangulate(
            ref_kf.R, ref_kf.t, self.R, self.t, pts_ref, pts_cur)
        num_new = int(good.sum())
        if num_new == 0:
            return

        first_new_id = len(self.map_points)
        self.map_points = np.vstack([self.map_points, points_3d[good]])
        self.track_pixels = np.vstack([self.track_pixels, pts_cur[good]])
        self.track_ids = np.concatenate([
            self.track_ids,
            np.arange(first_new_id, first_new_id + num_new, dtype=np.int32)])

        self._prune_map()
        self._enforce_track_limit()

    def _prune_map(self):
        """Keep the map size within the limit: retain the most recent points

        and every point that is currently being tracked.
        """
        limit = self.cfg["max_map_points"]
        if len(self.map_points) <= limit:
            return

        keep = np.zeros(len(self.map_points), bool)
        keep[-limit:] = True              # most recently added
        keep[self.track_ids] = True       # actively tracked points must survive

        kept_idx = np.nonzero(keep)[0]
        old_to_new = -np.ones(len(self.map_points), np.int64)
        old_to_new[kept_idx] = np.arange(len(kept_idx))

        self.map_points = self.map_points[kept_idx]
        self.track_ids = old_to_new[self.track_ids].astype(np.int32)

    def _relocalize(self, gray):
        """Try to recover the pose via ORB matches against recent keyframes.

        For each keyframe: match descriptors (Lowe ratio), gather 3D-2D
        correspondences and verify the pose with PnP RANSAC. The first keyframe
        that yields enough inliers is treated as a match.
        """
        keypoints, descriptors = self.orb.detectAndCompute(gray, self._mask(gray))
        if descriptors is None or len(keypoints) < 30:
            return False

        for kf in reversed(self.keyframes[-self.cfg["reloc_search"]:]):
            if kf.descriptors is None or not kf.map_point_by_kp:
                continue

            knn = self.matcher.knnMatch(descriptors, kf.descriptors, k=2)
            good_matches = [m for pair in knn if len(pair) == 2
                            for m, n in [pair]
                            if m.distance < self.cfg["lowe_ratio"] * n.distance]
            if len(good_matches) < self.cfg["min_reloc_matches"]:
                continue

            # build 3D-2D correspondences for PnP
            object_pts_3d, image_pts_2d, point_ids = [], [], []
            for match in good_matches:
                point_id = kf.map_point_by_kp.get(match.trainIdx)
                if point_id is None or point_id >= len(self.map_points):
                    continue
                object_pts_3d.append(self.map_points[point_id])
                image_pts_2d.append(keypoints[match.queryIdx].pt)
                point_ids.append(point_id)
            if len(object_pts_3d) < self.cfg["min_reloc_matches"]:
                continue

            object_pts_3d = np.array(object_pts_3d, np.float64)
            image_pts_2d = np.array(image_pts_2d, np.float64)

            ok, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
                object_pts_3d, image_pts_2d, self.K, None,
                iterationsCount=200, reprojectionError=self.cfg["pnp_reproj_px"],
                confidence=0.99, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inlier_idx is None \
                    or len(inlier_idx) < self.cfg["min_reloc_inliers"]:
                continue

            R_est, _ = cv2.Rodrigues(rvec)
            t_est = tvec.reshape(3, 1)
            cam_center_est = (-R_est.T @ t_est).ravel()
            if not np.all(np.isfinite(cam_center_est)):
                continue

            self.R = R_est
            self.t = t_est
            inlier_idx = inlier_idx.ravel()
            self.track_pixels = image_pts_2d[inlier_idx].astype(np.float32)
            self.track_ids = np.array(point_ids, np.int32)[inlier_idx]
            self._enforce_track_limit()
            return True
        return False

    def _enforce_track_limit(self):
        """Hard-cap the number of tracked points for a stable FPS.

        When the cap is exceeded, take a uniform sample over indices so the
        points stay spread across the frame instead of being chopped in a block.
        """
        max_points = self.cfg["max_features"]
        if len(self.track_pixels) > max_points:
            sample_idx = np.linspace(0, len(self.track_pixels) - 1,
                                     max_points, dtype=int)
            self.track_pixels = self.track_pixels[sample_idx]
            self.track_ids = self.track_ids[sample_idx]

    def process(self, gray):
        """Process one frame: update the state machine and the camera pose."""
        self.frame_idx += 1

        # --- INITIALIZING: accumulate parallax and try to build the first map ---
        if self.state == State.INITIALIZING:
            if self.init_reference is None:
                self.init_reference = (gray.copy(), self._detect(gray))
            else:
                self._try_init(gray)
            self.prev_gray = gray
            self._log()
            return

        if self.state == State.LOST:
            if self._relocalize(gray):
                self.state = State.RELOCALIZED
                self.reloc_display_left = self.cfg["reloc_display_frames"]
                self._add_keyframe(gray)
            elif self.frame_idx - self.lost_at_frame > self.cfg["reinit_after"]:
                self.state = State.INITIALIZING
                self.init_reference = (gray.copy(), self._detect(gray))
                # remember the pose at loss time for carry-over on reinit
                self.carry_rotation = self.R.copy()
                self.carry_translation = self.t.copy()
            self.prev_gray = gray
            self._log()
            return

        # 1) propagate map points with KLT optical flow
        tracked_px, valid = self._klt(self.prev_gray, gray, self.track_pixels)
        self.track_pixels = tracked_px[valid]
        self.track_ids = self.track_ids[valid]

        # 2) estimate the pose with PnP from the 3D-2D correspondences
        inlier_count = 0
        if len(self.track_pixels) >= self.cfg["min_pnp_pts"]:
            object_pts = self.map_points[self.track_ids]
            image_pts = self.track_pixels.astype(np.float64)
            rvec_init, _ = cv2.Rodrigues(self.R)

            success, rvec, tvec, inlier_idx = cv2.solvePnPRansac(
                object_pts, image_pts, self.K, None,
                rvec=rvec_init, tvec=self.t.copy(), useExtrinsicGuess=True,
                iterationsCount=200, reprojectionError=self.cfg["pnp_reproj_px"],
                confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

            if success and inlier_idx is not None:
                inlier_count = len(inlier_idx)
                # refine the pose with a nonlinear step over the inliers
                cv2.solvePnPRefineLM(object_pts[inlier_idx.ravel()],
                                     image_pts[inlier_idx.ravel()],
                                     self.K, None, rvec, tvec)
                R_est, _ = cv2.Rodrigues(rvec)
                t_est = tvec.reshape(3, 1)
                cam_center_prev = self.camera_center()
                cam_center_est = (-R_est.T @ t_est).ravel()

                # reject a degenerate pose (NaN/inf) and a non-physical jump
                jump = np.linalg.norm(cam_center_est - cam_center_prev)
                if not np.all(np.isfinite(cam_center_est)) \
                        or jump > self.cfg["max_jump"]:
                    inlier_count = 0
                else:
                    self.R, self.t = R_est, t_est
                    keep = inlier_idx.ravel()
                    self.track_pixels = self.track_pixels[keep]
                    self.track_ids = self.track_ids[keep]

        if inlier_count < self.cfg["min_inliers"]:
            # A single bad frame (shake, glare) must not drop us into LOST.
            # Hold TRACKING for lost_grace more frames on the previous pose -
            # the trajectory usually breaks exactly on such single glitches.
            self.consecutive_bad_frames += 1
            if (self.consecutive_bad_frames <= self.cfg["lost_grace"]
                    and len(self.track_pixels) >= 4):
                self.last_inlier_count = inlier_count
                if self.state != State.RELOCALIZED:
                    self.state = State.TRACKING
                self.prev_gray = gray
                self._log()
                return
            # the bad-frame streak is exhausted - go to LOST
            self.last_inlier_count = inlier_count
            self.state = State.LOST
            self.lost_at_frame = self.frame_idx
            self.prev_gray = gray
            self._log()
            return
        self.consecutive_bad_frames = 0

        # 4) successful tracking: clear the RELOCALIZED flag on a timer
        if self.state == State.RELOCALIZED:
            self.reloc_display_left -= 1
            if self.reloc_display_left <= 0:
                self.state = State.TRACKING
        else:
            self.state = State.TRACKING
        self.last_inlier_count = inlier_count

        # 5) add a keyframe and extend the map if needed
        enough_gap = (self.frame_idx - self.last_keyframe_frame) >= self.cfg["min_kf_gap"]
        need_new_keyframe = False
        if self.keyframes:
            dist_since_last_kf = np.linalg.norm(
                self.camera_center() - self.keyframes[-1].camera_center())
            need_new_keyframe = enough_gap and (
                dist_since_last_kf > self.cfg["kf_min_translation"]
                or len(self.track_pixels) < self.cfg["kf_min_points"])
        if need_new_keyframe:
            self._add_keyframe(gray)
            self._extend_map(gray)

        self.prev_gray = gray
        self._log()

    def _log(self):
        """Append a trajectory sample (camera center + state) for this frame."""
        self.trajectory.append(
            (self.frame_idx, self.camera_center().copy(), self.state))


# per-state color in BGR (used for the overlay and the trajectory view)
STATE_COLOR = {
    State.INITIALIZING: (0, 200, 200),
    State.TRACKING: (0, 220, 0),
    State.LOST: (0, 0, 255),
    State.RELOCALIZED: (0, 165, 255),
}


def draw_overlay(frame, vo, fps):
    """Draw tracked points, status and telemetry on the frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # tracked map points
    for x, y in vo.track_pixels:
        cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 0), -1)

    state_color = STATE_COLOR[vo.state]

    # semi-transparent banner behind the text
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    cv2.putText(canvas, f"SYSTEM STATUS: {vo.state.value}", (15, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, state_color, 2, cv2.LINE_AA)

    telemetry = (f"Frame: {vo.frame_idx} | "
                 f"Trk: {len(vo.track_pixels)}/{vo.cfg['max_features']} | "
                 f"Inl: {vo.last_inlier_count} | Map: {len(vo.map_points)} | "
                 f"KF: {len(vo.keyframes)} | FPS: {fps:.1f}")
    cv2.putText(canvas, telemetry, (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def draw_trajectory(vo, size=520):
    """Draw a top-down (X-Z plane) view of the camera trajectory."""
    canvas = np.zeros((size, size, 3), np.uint8)
    if len(vo.trajectory) < 2:
        return canvas

    positions = np.array([center for _, center, _ in vo.trajectory])
    xz = positions[:, [0, 2]]                       # take the horizontal plane
    finite_xz = xz[np.all(np.isfinite(xz), axis=1)]
    if len(finite_xz) < 2:
        return canvas

    # auto-scale to the trajectory bounds
    bounds_min, bounds_max = finite_xz.min(0), finite_xz.max(0)
    span = max((bounds_max - bounds_min).max(), 1e-6) * 1.25
    center = (bounds_max + bounds_min) / 2

    def to_pixel(point):
        norm = (point - center) / span + 0.5
        return int(norm[0] * size), int((1 - norm[1]) * size)

    # grid
    for g in range(0, size, size // 10):
        cv2.line(canvas, (g, 0), (g, size), (35, 35, 35), 1)
        cv2.line(canvas, (0, g), (size, g), (35, 35, 35), 1)

    # the trajectory itself (LOST as dots, everything else as lines)
    for i in range(1, len(vo.trajectory)):
        if not (np.all(np.isfinite(xz[i - 1])) and np.all(np.isfinite(xz[i]))):
            continue
        state = vo.trajectory[i][2]
        if state == State.LOST:
            cv2.circle(canvas, to_pixel(xz[i]), 2, STATE_COLOR[State.LOST], -1)
        else:
            cv2.line(canvas, to_pixel(xz[i - 1]), to_pixel(xz[i]),
                     STATE_COLOR[state], 2)

    # start (white) and current position (yellow)
    if np.all(np.isfinite(xz[0])):
        cv2.circle(canvas, to_pixel(xz[0]), 5, (255, 255, 255), -1)
    cv2.circle(canvas, to_pixel(finite_xz[-1]), 6, (0, 255, 255), -1)
    return canvas


# Parameters tuned for this footage (slow forward motion, narrow FOV).
# Any key can be overridden from the command line via --set KEY=VAL.
CFG = dict(
    max_features=800, orb_features=700, quality=0.01, min_feat_dist=10,
    sky_cut=0.30,
    klt_win=21, klt_levels=4, klt_iters=30, klt_fb_check=True, fb_thresh=1.5,
    init_parallax=4.0, min_init_pts=40,
    min_pnp_pts=10, min_inliers=10, pnp_reproj_px=5.0, ransac_px=1.5,
    max_depth=100.0, max_jump=1.5,
    lost_grace=4,                    # frames to hold before dropping to LOST
    tri_reproj_px=3.0, min_tri_baseline=0.08,
    max_map_points=4000,
    kf_min_points=120, kf_min_translation=0.15, max_keyframes=400, min_kf_gap=3,
    reloc_search=20, lowe_ratio=0.8, min_reloc_matches=15,
    min_reloc_inliers=10, reloc_display_frames=15, reinit_after=20,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--calib", default="config.yaml",
                        help="Path to the YAML calibration file")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=800,
                        help="Max number of tracked points (caps the FPS drop).")
    parser.add_argument("--save", default=None)
    parser.add_argument("--save-video", action="store_true",
                        help="save video with an auto name based on the tracking "
                             "percentage to video_output/ugv_output_video_XX_pct.mp4")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    args = parser.parse_args()

    # --- calibration ---
    try:
        K, dist_coeffs = load_calibration(args.calib)
        print(f"[i] Calibration loaded from: {args.calib}")
    except FileNotFoundError:
        raise SystemExit(f"[!] Error: calibration file '{args.calib}' not found.")
    except Exception as e:
        raise SystemExit(f"[!] Error reading calibration: {e}")

    # --- config with CLI overrides ---
    cfg = dict(CFG)
    cfg["max_features"] = args.max_features
    for item in args.set:
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in cfg:
            raise SystemExit(f"unknown parameter: {key}")
        default_value = cfg[key]
        if isinstance(default_value, bool):
            cfg[key] = raw_value.lower() in ("1", "true", "yes")
        elif isinstance(default_value, int):
            cfg[key] = int(float(raw_value))
        else:
            cfg[key] = float(raw_value)
        print(f"[i] override {key} = {cfg[key]}")

    # --- video source ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 16.2
    print(f"[i] video {frame_w}x{frame_h} @ {source_fps:.1f} fps")

    # scale the image and the camera matrix together
    if args.scale != 1.0:
        K = K.copy()
        K[0, :] *= args.scale
        K[1, :] *= args.scale
        frame_w, frame_h = int(frame_w * args.scale), int(frame_h * args.scale)

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (frame_w, frame_h), 0)
    undistort_map1, undistort_map2 = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, new_K, (frame_w, frame_h), cv2.CV_16SC2)

    vo = MonoVO(new_K, cfg)

    save_writer = None
    frames_buffer = [] if args.save_video else None   # for the auto name by % at the end
    out_size = None

    # two independent FPS measurements: pure VO and the useful pipeline
    vo_fps_hist = deque(maxlen=30)          # vo.process only
    pipeline_fps_hist = deque(maxlen=30)    # undistort + grayscale + VO

    read_count = 0                          # frames read from the video
    processed_count = 0                     # frames actually processed
    wall_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        read_count += 1
        # frame decimation (--skip N: process every N+1-th frame)
        if args.skip and (read_count % (args.skip + 1) != 0):
            continue

        pipeline_start = time.perf_counter()   # start timing the useful pipeline
        if args.scale != 1.0:
            frame = cv2.resize(frame, (frame_w, frame_h),
                               interpolation=cv2.INTER_AREA)
        frame = cv2.remap(frame, undistort_map1, undistort_map2, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        vo_start = time.perf_counter()
        vo.process(gray)
        vo_fps_hist.append(1.0 / max(time.perf_counter() - vo_start, 1e-6))
        pipeline_fps_hist.append(1.0 / max(time.perf_counter() - pipeline_start, 1e-6))
        fps = float(np.mean(vo_fps_hist))
        processed_count += 1

        # visualization: frame + top-down view
        vis = np.hstack([draw_overlay(frame, vo, fps),
                         draw_trajectory(vo, size=frame.shape[0])])
        out_size = (vis.shape[1], vis.shape[0])

        if args.save:
            if save_writer is None:
                save_writer = cv2.VideoWriter(
                    args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, out_size)
            save_writer.write(vis)
        if args.save_video:
            frames_buffer.append(vis)

        if not args.headless:
            cv2.imshow("Mono VO", vis)
            if cv2.waitKey(1) & 0xFF == 27:     # ESC
                break

        if args.log_every and vo.frame_idx % args.log_every == 0:
            print(f"f{vo.frame_idx:5d}  {vo.state.value:12s}  {fps:5.1f}fps  "
                  f"trk {len(vo.track_pixels):4d}  map {len(vo.map_points):5d}  "
                  f"kf {len(vo.keyframes):3d}", flush=True)

    loop_end = time.perf_counter()   # end of processing (before writing the video)
    cap.release()
    if save_writer:
        save_writer.release()
    cv2.destroyAllWindows()

    # --- state summary ---
    states = [s for _, _, s in vo.trajectory]
    total = len(states)
    tracked_pct = 0.0
    if total:
        print("\n=== summary ===")
        print(f"frames processed : {total}")
        for state in State:
            count = states.count(state)
            pct = 100.0 * count / total
            if state in (State.TRACKING, State.RELOCALIZED):
                tracked_pct += pct
            print(f"{state.value:14s} : {count:5d}  ({pct:5.1f}%)")
        print(f"tracked (TRK+RELOC): {tracked_pct:5.1f}%")
        print(f"keyframes        : {len(vo.keyframes)}")
        print(f"map points       : {len(vo.map_points)}")

        # --- three honest FPS figures ---
        wall_elapsed = loop_end - wall_start
        effective_stream_fps = source_fps / (args.skip + 1)   # stream after decimation
        vo_only_fps = float(np.mean(vo_fps_hist)) if vo_fps_hist else 0.0
        pipeline_fps = float(np.mean(pipeline_fps_hist)) if pipeline_fps_hist else 0.0
        wall_fps = processed_count / wall_elapsed if wall_elapsed > 0 else 0.0

        print("\n=== FPS (performance) ===")
        print(f"VO only          : {vo_only_fps:5.1f} FPS   (vo.process only, algorithm potential)")
        print(f"pipeline         : {pipeline_fps:5.1f} FPS   (undistort+grayscale+VO, what runs on the robot)")
        print(f"wall-clock loop  : {wall_fps:5.1f} FPS   (processing loop, without video writing)")
        print("--- real-time threshold ---")
        print(f"camera captures  : {source_fps:5.1f} fps")
        print(f"after skip={args.skip}      : {effective_stream_fps:5.1f} fps  <- must keep up with this")
        # verdict by the pipeline - that is what actually runs on the robot
        keeps_up = "YES (real-time)" if pipeline_fps >= effective_stream_fps \
            else "no (falling behind the stream)"
        margin = pipeline_fps / effective_stream_fps if effective_stream_fps > 0 else 0
        print(f"pipeline keeps up: {keeps_up}   (margin x{margin:.1f})")


    if args.save_video and frames_buffer and out_size:
        os.makedirs("video_output", exist_ok=True)
        out_name = f"video_output/ugv_output_video_{int(tracked_pct)}_pct.mp4"
        writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"mp4v"),
                                 20.0, out_size)
        for vis in frames_buffer:
            writer.write(vis)
        writer.release()
        print(f"[i] video saved -> {out_name}")

    # --- save the trajectory to CSV ---
    if vo.trajectory:
        os.makedirs("trajectory_output", exist_ok=True)
        traj_name = f"trajectory_output/ugv_output_trajectory_{int(tracked_pct)}_pct.csv"
        traj_rows = np.array([[idx, *center] for idx, center, _ in vo.trajectory])
        np.savetxt(traj_name, traj_rows, delimiter=",",
                   header="frame,x,y,z", comments="")
        print(f"[i] trajectory saved -> {traj_name}")


if __name__ == "__main__":
    main()
