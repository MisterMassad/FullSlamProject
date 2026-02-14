import os
import glob
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np


# ============================================================
# Helpers (from your VO project)
# ============================================================

def build_k_from_image_size(width: int, height: int) -> np.ndarray:
    f = float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=np.float64)


def to_homogeneous_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t.reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def load_image_paths(folder: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths.sort()
    return paths


# ============================================================
# Error metrics (required by new project)
# ============================================================

def epipolar_errors(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Symmetric point-to-epipolar-line distance (simple version).
    pts1, pts2: Nx2
    Returns: N residuals (pixels-ish)
    """
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    # l2 = F * x1, l1 = F^T * x2
    l2 = (F @ pts1_h.T).T
    l1 = (F.T @ pts2_h.T).T

    # distance point to line: |a x + b y + c| / sqrt(a^2+b^2)
    num2 = np.abs(np.sum(l2 * pts2_h, axis=1))
    den2 = np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2) + 1e-12
    d2 = num2 / den2

    num1 = np.abs(np.sum(l1 * pts1_h, axis=1))
    den1 = np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2) + 1e-12
    d1 = num1 / den1

    return 0.5 * (d1 + d2)


def project_points(K: np.ndarray, T_world_cam: np.ndarray, Pw: np.ndarray) -> np.ndarray:
    """
    Project world 3D points Pw (Nx3) into the camera image using world_T_cam.
    Returns: Nx2 pixel coordinates.
    """
    # cam_T_world
    T_cam_world = invert_T(T_world_cam)
    R = T_cam_world[:3, :3]
    t = T_cam_world[:3, 3]

    Pc = (R @ Pw.T).T + t.reshape(1, 3)
    z = Pc[:, 2:3]
    z = np.where(z == 0, 1e-9, z)
    x = Pc[:, 0:1] / z
    y = Pc[:, 1:2] / z

    uv = (K @ np.hstack([x, y, np.ones_like(x)]).T).T
    return uv[:, :2]


def reprojection_error(K: np.ndarray, T_world_cam: np.ndarray, Pw: np.ndarray, uv_obs: np.ndarray) -> float:
    uv_hat = project_points(K, T_world_cam, Pw)
    err = np.linalg.norm(uv_hat - uv_obs, axis=1)
    return float(np.mean(err)) if len(err) > 0 else float("inf")


# ============================================================
# Data structures required by assignment (Frame, Point, Map)
# ============================================================

@dataclass
class Frame:
    frame_id: int
    image_bgr: np.ndarray
    timestamp: float
    keypoints: Optional[List[cv2.KeyPoint]] = None
    descriptors: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None  # world_T_cam
    is_keyframe: bool = False

    def extract_features(self, detector) -> None:
        gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        kps, desc = detector.detectAndCompute(gray, None)
        self.keypoints = kps
        self.descriptors = desc


@dataclass
class MapPoint:
    point_id: int
    xyz_world: np.ndarray  # (3,)
    # Store a "reference descriptor" for matching (MVP approach)
    descriptor: Optional[np.ndarray] = None
    # Observations: frame_id -> keypoint index
    observations: Dict[int, int] = field(default_factory=dict)


@dataclass
class Map3D:
    points: List[MapPoint] = field(default_factory=list)
    keyframes: List[Frame] = field(default_factory=list)


# ============================================================
# Tracking (your VO, adapted into a component)
# ============================================================

class Tracker:
    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)
        self.world_T_cam = np.eye(4, dtype=np.float64)
        self.prev_frame: Optional[Frame] = None

        self.detector = cv2.ORB_create(
            nfeatures=1500,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Tunables
        self.ratio_thresh = 0.7
        self.min_matches = 12
        self.min_inliers = 10
        self.ransac_threshold = 1.0
        self.ransac_prob = 0.999

    def process_frame(self, frame: Frame) -> Tuple[bool, dict, Optional[np.ndarray], Optional[np.ndarray], Optional[List[cv2.DMatch]]]:
        """
        Returns:
          ok, debug, pts_prev, pts_curr, good_matches
        """
        debug = {"frame_id": frame.frame_id, "status": "init",
                 "num_keypoints": 0, "num_matches": 0, "num_inliers": 0,
                 "epi_err_before": None, "epi_err_after": None}

        frame.extract_features(self.detector)
        debug["num_keypoints"] = 0 if frame.keypoints is None else len(frame.keypoints)

        if self.prev_frame is None:
            frame.pose = self.world_T_cam.copy()
            self.prev_frame = frame
            debug["status"] = "first_frame"
            return True, debug, None, None, None

        if self.prev_frame.descriptors is None or frame.descriptors is None:
            self.prev_frame = frame
            debug["status"] = "no_descriptors"
            return False, debug, None, None, None

        # Matches before filtering (for requirement 1 visualization)
        knn = self.matcher.knnMatch(self.prev_frame.descriptors, frame.descriptors, k=2)

        # Ratio filter
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        debug["num_matches"] = len(good)
        if len(good) < self.min_matches:
            self.prev_frame = frame
            debug["status"] = "too_few_matches"
            return False, debug, None, None, good

        pts_prev = np.float64([self.prev_frame.keypoints[m.queryIdx].pt for m in good])
        pts_curr = np.float64([frame.keypoints[m.trainIdx].pt for m in good])

        # For epipolar error, estimate a FUNDAMENTAL matrix to compute line error
        # This is only for reporting/debugging, so it must NEVER crash the pipeline.
        F_all, maskF_all = None, None
        if len(pts_prev) >= 8:
            try:
                F_all, maskF_all = cv2.findFundamentalMat(
                    pts_prev, pts_curr,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=1.0,
                    confidence=0.999
                )
            except cv2.error:
                # fallback: plain RANSAC (more stable, less strict)
                try:
                    F_all, maskF_all = cv2.findFundamentalMat(
                        pts_prev, pts_curr,
                        method=cv2.FM_RANSAC,
                        ransacReprojThreshold=1.0,
                        confidence=0.999
                    )
                except cv2.error:
                    F_all, maskF_all = None, None


        if F_all is not None and F_all.shape == (3, 3) and np.isfinite(F_all).all():
            errs_all = epipolar_errors(F_all, pts_prev, pts_curr)
            debug["epi_err_before"] = float(np.median(errs_all))


        # Essential matrix (your approach)
        E, maskE = cv2.findEssentialMat(
            pts_prev, pts_curr,
            self.K,
            method=cv2.USAC_MAGSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_threshold
        )

        if E is None or maskE is None:
            self.prev_frame = frame
            debug["status"] = "E_failed"
            return False, debug, pts_prev, pts_curr, good

        inliers = int(maskE.sum())
        debug["num_inliers"] = inliers
        if inliers < self.min_inliers:
            self.prev_frame = frame
            debug["status"] = "too_few_inliers"
            return False, debug, pts_prev, pts_curr, good

        # Epipolar error after geometry: use inliers only
        if F_all is not None and F_all.shape == (3, 3):
            inl = maskE.ravel().astype(bool)
            if np.count_nonzero(inl) >= 8:
                errs_inl = epipolar_errors(F_all, pts_prev[inl], pts_curr[inl])
                debug["epi_err_after"] = float(np.median(errs_inl))

        # recoverPose
        try:
            cheir_inliers, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, self.K, mask=maskE)
        except cv2.error:
            self.prev_frame = frame
            debug["status"] = "recoverPose_failed"
            return False, debug, pts_prev, pts_curr, good

        if R is None or t is None or (not np.isfinite(R).all()) or (not np.isfinite(t).all()):
            self.prev_frame = frame
            debug["status"] = "bad_pose"
            return False, debug, pts_prev, pts_curr, good

        detR = float(np.linalg.det(R))
        if abs(detR - 1.0) > 0.1:
            self.prev_frame = frame
            debug["status"] = "bad_detR"
            return False, debug, pts_prev, pts_curr, good

        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
        if angle > 60.0:
            self.prev_frame = frame
            debug["status"] = "flip_guard"
            return False, debug, pts_prev, pts_curr, good

        T_prev_to_curr = to_homogeneous_transform(R, t)

        # Accumulate pose
        self.world_T_cam = self.world_T_cam @ T_prev_to_curr
        frame.pose = self.world_T_cam.copy()

        debug["detR"] = detR
        debug["rot_angle_deg"] = angle
        debug["t_norm"] = float(np.linalg.norm(t))
        debug["status"] = "ok"

        self.prev_frame = frame
        return True, debug, pts_prev, pts_curr, good


# ============================================================
# Mapping: keyframes + triangulation (MVP)
# ============================================================

class Mapper:
    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)
        self.next_point_id = 0

    def make_keyframe(self, frame: Frame, slam_map: Map3D) -> None:
        frame.is_keyframe = True
        slam_map.keyframes.append(frame)

    def triangulate_between_keyframes(
        self,
        kf1: Frame,
        kf2: Frame,
        matches: List[cv2.DMatch],
        slam_map: Map3D,
        max_new_points: int = 800
    ) -> int:
        """
        MVP triangulation:
        - triangulate matched keypoints between two keyframes with known poses
        - create MapPoints with descriptor copied from kf2 keypoint (simple)
        """
        if kf1.pose is None or kf2.pose is None:
            return 0
        if kf1.keypoints is None or kf2.keypoints is None:
            return 0
        if kf1.descriptors is None or kf2.descriptors is None:
            return 0
        if len(matches) < 8:
            return 0

        # Build projection matrices P = K [R|t] where R,t are cam_T_world
        T1_cw = invert_T(kf1.pose)
        T2_cw = invert_T(kf2.pose)
        P1 = self.K @ T1_cw[:3, :]
        P2 = self.K @ T2_cw[:3, :]

        pts1 = np.float64([kf1.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float64([kf2.keypoints[m.trainIdx].pt for m in matches])

        X_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).T  # Nx4
        X = X_h[:, :3] / (X_h[:, 3:4] + 1e-12)

        # Basic filtering: keep points with finite values
        good = np.isfinite(X).all(axis=1)

        # Also keep points with positive depth in both cameras
        Xg = X[good]
        if len(Xg) == 0:
            return 0

        def depth_in_cam(T_world_cam: np.ndarray, Pw: np.ndarray) -> np.ndarray:
            T_cw = invert_T(T_world_cam)
            Pc = (T_cw[:3, :3] @ Pw.T).T + T_cw[:3, 3].reshape(1, 3)
            return Pc[:, 2]

        z1 = depth_in_cam(kf1.pose, Xg)
        z2 = depth_in_cam(kf2.pose, Xg)
        ok_depth = (z1 > 0.1) & (z2 > 0.1)

        Xkeep = Xg[ok_depth]
        idx_keep = np.where(good)[0][ok_depth]

        added = 0
        for j, mi in enumerate(idx_keep[:max_new_points]):
            m = matches[mi]

            mp = MapPoint(
                point_id=self.next_point_id,
                xyz_world=Xkeep[j].astype(np.float64),
                descriptor=kf2.descriptors[m.trainIdx].copy(),
                observations={kf1.frame_id: m.queryIdx, kf2.frame_id: m.trainIdx}
            )
            slam_map.points.append(mp)
            self.next_point_id += 1
            added += 1

        return added


# ============================================================
# Relocalization: PnP every N frames (MVP)
# ============================================================

class RelocalizerPnP:
    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def localize(self, frame: Frame, slam_map: Map3D, ratio: float = 0.75) -> Tuple[bool, dict]:
        """
        MVP:
        - match frame descriptors to MapPoint descriptors
        - build 2D-3D correspondences
        - run solvePnPRansac
        """
        dbg = {"pnp_num_corr": 0, "pnp_inliers": 0, "pnp_reproj_before": None, "pnp_reproj_after": None, "status": "init"}

        if frame.descriptors is None or frame.keypoints is None:
            dbg["status"] = "no_features"
            return False, dbg
        if len(slam_map.points) < 30:
            dbg["status"] = "map_too_small"
            return False, dbg

        # Build descriptor matrix for map points
        mp_desc = []
        mp_xyz = []
        mp_ids = []
        for mp in slam_map.points:
            if mp.descriptor is None:
                continue
            mp_desc.append(mp.descriptor)
            mp_xyz.append(mp.xyz_world)
            mp_ids.append(mp.point_id)

        if len(mp_desc) < 30:
            dbg["status"] = "no_map_desc"
            return False, dbg

        mp_desc = np.vstack(mp_desc)
        mp_xyz = np.vstack(mp_xyz)

        knn = self.bf.knnMatch(mp_desc, frame.descriptors, k=2)

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < 12:
            dbg["status"] = "too_few_corr"
            return False, dbg

        # Each match maps: queryIdx -> map point index, trainIdx -> frame keypoint index
        Pw = np.float64([mp_xyz[m.queryIdx] for m in good])
        uv = np.float64([frame.keypoints[m.trainIdx].pt for m in good])

        dbg["pnp_num_corr"] = len(good)

        # Initial reprojection error (requires current pose guess)
        if frame.pose is not None and np.isfinite(frame.pose).all():
            dbg["pnp_reproj_before"] = reprojection_error(self.K, frame.pose, Pw, uv)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=Pw,
            imagePoints=uv,
            cameraMatrix=self.K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=4.0,
            confidence=0.999,
            iterationsCount=200
        )

        if not ok or inliers is None or len(inliers) < 10:
            dbg["status"] = "pnp_failed"
            return False, dbg

        dbg["pnp_inliers"] = int(len(inliers))

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3,)

        # solvePnP returns cam_T_world? Actually it returns rvec,tvec such that:
        # x_cam = R * X_world + t
        # That's T_cam_world. We store world_T_cam, so invert.
        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = R
        T_cam_world[:3, 3] = t
        T_world_cam = invert_T(T_cam_world)

        frame.pose = T_world_cam

        dbg["pnp_reproj_after"] = reprojection_error(self.K, frame.pose, Pw, uv)
        dbg["status"] = "ok"

        return True, dbg


# ============================================================
# Loop closure (stub for MVP)
# ============================================================

class LoopClosure:
    def detect(self, current_kf: Frame, slam_map: Map3D) -> Optional[int]:
        """
        MVP stub:
        Return frame_id of matched past keyframe if loop detected, else None.
        We'll implement this after MVP mapping+PnP is stable.
        """
        return None


# Helper Functions

def match_descriptors_ratio(matcher, desc1, desc2, ratio=0.75, min_matches=12):
    if desc1 is None or desc2 is None:
        return []
    knn = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good if len(good) >= min_matches else []


# ============================================================
# Main: MVP SLAM loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Minimal Monocular SLAM MVP (based on VO)")
    parser.add_argument("--data_dir", type=str, default="VO_Dataset", help="Folder containing images")
    parser.add_argument("--keyframe_stride", type=int, default=10, help="Make a keyframe every N frames")
    parser.add_argument("--pnp_stride", type=int, default=10, help="Run PnP relocalization every N frames")
    args = parser.parse_args()

    image_paths = load_image_paths(args.data_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in: {args.data_dir}")

    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read: {image_paths[0]}")
    h, w = first.shape[:2]
    K = build_k_from_image_size(w, h)

    tracker = Tracker(K)
    mapper = Mapper(K)
    relocalizer = RelocalizerPnP(K)
    slam_map = Map3D()

    kf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    cv2.namedWindow("keypoints", cv2.WINDOW_NORMAL)

    last_keyframe: Optional[Frame] = None

    for i, p in enumerate(image_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read {p}")
            continue

        try:
            timestamp = float(os.path.splitext(os.path.basename(p))[0])
        except ValueError:
            timestamp = float(i)

        frame = Frame(frame_id=i, image_bgr=img, timestamp=timestamp)

        ok_vo, dbg, pts_prev, pts_curr, good_matches = tracker.process_frame(frame)

        # Decide keyframe (MVP: fixed stride, only if pose exists)
        if frame.pose is not None and (i % args.keyframe_stride == 0):
            mapper.make_keyframe(frame, slam_map)

            # Triangulate with previous keyframe (if exists)
            if last_keyframe is not None:
                matches_kf = match_descriptors_ratio(
                    kf_matcher,
                    last_keyframe.descriptors,
                    frame.descriptors,
                    ratio=0.75,
                    min_matches=12
                )

                if matches_kf:
                    added = mapper.triangulate_between_keyframes(last_keyframe, frame, matches_kf, slam_map)
                else:
                    added = 0

                dbg["triangulated_new_points"] = int(added)
            else:
                dbg["triangulated_new_points"] = 0

            last_keyframe = frame

        # PnP relocalization every N frames (MVP)
        pnp_dbg = None
        if frame.pose is not None and (i % args.pnp_stride == 0):
            ok_pnp, pnp_dbg = relocalizer.localize(frame, slam_map)
            dbg["pnp_status"] = pnp_dbg["status"]
            dbg["pnp_corr"] = pnp_dbg["pnp_num_corr"]
            dbg["pnp_inliers"] = pnp_dbg["pnp_inliers"]
            dbg["reproj_before"] = pnp_dbg["pnp_reproj_before"]
            dbg["reproj_after"] = pnp_dbg["pnp_reproj_after"]

        # --- Visualization (image + keypoints)
        if frame.keypoints is not None:
            vis = cv2.drawKeypoints(
                img, frame.keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        else:
            vis = img

        text1 = (f"id={i} kp={dbg.get('num_keypoints',0)} "
                 f"m={dbg.get('num_matches',0)} inl={dbg.get('num_inliers',0)} "
                 f"epi_med_before={dbg.get('epi_err_before',None)} epi_med_after={dbg.get('epi_err_after',None)} "
                 f"st={dbg.get('status','')}")
        cv2.putText(vis, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

        text2 = (f"map_points={len(slam_map.points)} keyframes={len(slam_map.keyframes)} "
                 f"new_pts={dbg.get('triangulated_new_points',0)} "
                 f"pnp={dbg.get('pnp_status','-')} reproj={dbg.get('reproj_after','-')}")
        cv2.putText(vis, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2)

        cv2.imshow("keypoints", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()
    print("[DONE] MVP run finished.")
    print(f"Keyframes: {len(slam_map.keyframes)} | Map points: {len(slam_map.points)}")


if __name__ == "__main__":
    main()
