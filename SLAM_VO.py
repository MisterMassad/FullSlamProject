import os
import glob
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

# ------------ 3D Visualization using VisPy ------------
try:
    from vispy import scene
    from vispy.scene import visuals
    VISPY_OK = True
except Exception:
    VISPY_OK = False


# ============================================================
# Helpers
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


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion (x, y, z, w).
    Stable enough for our usage.
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


# ============================================================
# Error Metrics
# ============================================================

def epipolar_errors(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    l2 = (F @ pts1_h.T).T
    l1 = (F.T @ pts2_h.T).T

    num2 = np.abs(np.sum(l2 * pts2_h, axis=1))
    den2 = np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2) + 1e-12
    d2 = num2 / den2

    num1 = np.abs(np.sum(l1 * pts1_h, axis=1))
    den1 = np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2) + 1e-12
    d1 = num1 / den1

    return 0.5 * (d1 + d2)


def project_points(K: np.ndarray, T_world_cam: np.ndarray, Pw: np.ndarray) -> np.ndarray:
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
# SE(3) utilities + Pose-only Gauss-Newton
# ============================================================

def _skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=np.float64)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64).reshape(6,)
    w = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(w)

    T = np.eye(4, dtype=np.float64)
    if theta < 1e-12:
        T[:3, 3] = v
        return T

    W = _skew(w / theta)
    R = np.eye(3) + np.sin(theta) * W + (1.0 - np.cos(theta)) * (W @ W)
    V = np.eye(3) + (1.0 - np.cos(theta)) / theta * W + (theta - np.sin(theta)) / theta * (W @ W)

    T[:3, :3] = R
    T[:3, 3] = (V @ v)
    return T


def huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    w = np.ones_like(r)
    mask = a > delta
    w[mask] = delta / (a[mask] + 1e-12)
    return w


class PoseOptimizerGN:
    def __init__(self, K: np.ndarray, huber_delta_px: float = 3.0, eps: float = 1e-4):
        self.K = K.astype(np.float64)
        self.huber_delta_px = float(huber_delta_px)
        self.eps = float(eps)

    def _residuals(self, T_world_cam: np.ndarray, Pw: np.ndarray, uv: np.ndarray) -> np.ndarray:
        uv_hat = project_points(self.K, T_world_cam, Pw)
        return (uv_hat - uv).reshape(-1)  # (2N,)

    def refine(self, T_init: np.ndarray, Pw: np.ndarray, uv: np.ndarray,
               max_iters: int = 8, min_step_norm: float = 1e-6) -> Tuple[np.ndarray, Dict]:

        dbg = {"gn_iters": 0, "gn_reproj_before": None, "gn_reproj_after": None, "gn_status": "init"}

        if Pw is None or uv is None or len(Pw) < 6:
            dbg["gn_status"] = "too_few_corr"
            return T_init, dbg

        T = T_init.copy()
        dbg["gn_reproj_before"] = reprojection_error(self.K, T, Pw, uv)

        for it in range(max_iters):
            r = self._residuals(T, Pw, uv)
            w = huber_weights(r, self.huber_delta_px)
            rw = w * r

            J = np.zeros((r.size, 6), dtype=np.float64)
            for j in range(6):
                dxi = np.zeros(6, dtype=np.float64)
                dxi[j] = self.eps
                T_pert = se3_exp(dxi) @ T
                rj = self._residuals(T_pert, Pw, uv)
                J[:, j] = (rj - r) / self.eps

            Jw = (w[:, None] * J)
            A = Jw.T @ Jw
            b = -Jw.T @ rw

            lam = 1e-6
            A = A + lam * np.eye(6)

            try:
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dbg["gn_status"] = "singular"
                break

            if float(np.linalg.norm(dx)) < min_step_norm:
                dbg["gn_status"] = "converged"
                dbg["gn_iters"] = it
                break

            T = se3_exp(dx) @ T
            dbg["gn_iters"] = it + 1

        dbg["gn_reproj_after"] = reprojection_error(self.K, T, Pw, uv)
        if dbg["gn_status"] == "init":
            dbg["gn_status"] = "ok"
        return T, dbg


# ============================================================
# Data Structures (Frame / Point / Map)
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

    @property
    def RotationMatrix(self) -> Optional[np.ndarray]:
        return None if self.pose is None else self.pose[:3, :3]

    @property
    def TranslationVector(self) -> Optional[np.ndarray]:
        return None if self.pose is None else self.pose[:3, 3]

    def extract_features(self, detector) -> None:
        gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        kps, desc = detector.detectAndCompute(gray, None)
        self.keypoints = kps
        self.descriptors = desc


@dataclass
class Point:
    point_id: int
    xyz_world: np.ndarray
    descriptor: Optional[np.ndarray] = None
    observations: Dict[int, int] = field(default_factory=dict)
    num_obs: int = 0


@dataclass
class Map:
    points: List[Point] = field(default_factory=list)
    keyframes: List[Frame] = field(default_factory=list)
    voxel_hash: Dict[Tuple[int, int, int], int] = field(default_factory=dict)


# ============================================================
# Tracking (VO) + matches before/after
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

        self.ratio_thresh = 0.7
        self.min_matches = 12
        self.min_inliers = 10
        self.ransac_threshold = 1.0
        self.ransac_prob = 0.999

    def process_frame(self, frame: Frame) -> Tuple[bool, dict]:
        debug = {
            "frame_id": frame.frame_id,
            "status": "init",
            "num_keypoints": 0,
            "num_matches_raw": 0,
            "num_matches_good": 0,
            "num_inliers": 0,
            "epi_err_before": None,
            "epi_err_after": None,
            "raw_matches": None,
            "good_matches": None,
            "inlier_mask": None,
            "detR": None,
            "rot_angle_deg": None,
            "t_norm": None,
        }

        frame.extract_features(self.detector)
        debug["num_keypoints"] = 0 if frame.keypoints is None else len(frame.keypoints)

        if self.prev_frame is None:
            frame.pose = self.world_T_cam.copy()
            self.prev_frame = frame
            debug["status"] = "first_frame"
            return True, debug

        if self.prev_frame.descriptors is None or frame.descriptors is None:
            self.prev_frame = frame
            debug["status"] = "no_descriptors"
            return False, debug

        knn = self.matcher.knnMatch(self.prev_frame.descriptors, frame.descriptors, k=2)

        raw = []
        for pair in knn:
            if len(pair) >= 1:
                raw.append(pair[0])
        debug["num_matches_raw"] = len(raw)

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)

        debug["num_matches_good"] = len(good)
        debug["raw_matches"] = raw
        debug["good_matches"] = good

        if len(good) < self.min_matches:
            self.prev_frame = frame
            debug["status"] = "too_few_matches"
            return False, debug

        pts_prev = np.float64([self.prev_frame.keypoints[m.queryIdx].pt for m in good])
        pts_curr = np.float64([frame.keypoints[m.trainIdx].pt for m in good])

        # Fundamental matrix for epipolar error reporting
        F_all = None
        if len(pts_prev) >= 8:
            try:
                F_all, _ = cv2.findFundamentalMat(
                    pts_prev, pts_curr,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=1.0,
                    confidence=0.999
                )
            except cv2.error:
                try:
                    F_all, _ = cv2.findFundamentalMat(
                        pts_prev, pts_curr,
                        method=cv2.FM_RANSAC,
                        ransacReprojThreshold=1.0,
                        confidence=0.999
                    )
                except cv2.error:
                    F_all = None

        if F_all is not None and F_all.shape == (3, 3) and np.isfinite(F_all).all():
            debug["epi_err_before"] = float(np.median(epipolar_errors(F_all, pts_prev, pts_curr)))

        # Essential -> motion
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
            return False, debug

        inliers = int(maskE.sum())
        debug["num_inliers"] = inliers
        if inliers < self.min_inliers:
            self.prev_frame = frame
            debug["status"] = "too_few_inliers"
            return False, debug

        inl = maskE.ravel().astype(bool)
        debug["inlier_mask"] = inl

        if F_all is not None and F_all.shape == (3, 3) and np.isfinite(F_all).all():
            if np.count_nonzero(inl) >= 8:
                debug["epi_err_after"] = float(np.median(epipolar_errors(F_all, pts_prev[inl], pts_curr[inl])))

        try:
            _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, self.K, mask=maskE)
        except cv2.error:
            self.prev_frame = frame
            debug["status"] = "recoverPose_failed"
            return False, debug

        if R is None or t is None or (not np.isfinite(R).all()) or (not np.isfinite(t).all()):
            self.prev_frame = frame
            debug["status"] = "bad_pose"
            return False, debug

        detR = float(np.linalg.det(R))
        if abs(detR - 1.0) > 0.1:
            self.prev_frame = frame
            debug["status"] = "bad_detR"
            return False, debug

        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
        if angle > 60.0:
            self.prev_frame = frame
            debug["status"] = "flip_guard"
            return False, debug

        T_prev_to_curr = to_homogeneous_transform(R, t)

        self.world_T_cam = self.world_T_cam @ T_prev_to_curr
        frame.pose = self.world_T_cam.copy()

        debug["detR"] = detR
        debug["rot_angle_deg"] = angle
        debug["t_norm"] = float(np.linalg.norm(t))
        debug["status"] = "ok"

        self.prev_frame = frame
        return True, debug


# ============================================================
# Triangulation and Mapping
# ============================================================

class Mapper:
    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)
        self.Kinv = np.linalg.inv(self.K)
        self.next_point_id = 0

        self.max_new_points = 600
        self.min_baseline = 0.02
        self.min_parallax_deg = 1.0
        self.max_reproj_px = 3.5
        self.min_depth = 0.1
        self.max_depth = 200.0
        self.voxel_size = 0.05

        self.kf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.kf_ratio = 0.75
        self.kf_min_matches = 20

    def make_keyframe(self, frame: Frame, slam_map: Map) -> None:
        frame.is_keyframe = True
        slam_map.keyframes.append(frame)

    def _match_keyframes(self, kf1: Frame, kf2: Frame) -> List[cv2.DMatch]:
        if kf1.descriptors is None or kf2.descriptors is None:
            return []
        knn = self.kf_matcher.knnMatch(kf1.descriptors, kf2.descriptors, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.kf_ratio * n.distance:
                good.append(m)
        return good if len(good) >= self.kf_min_matches else []

    def _baseline(self, kf1: Frame, kf2: Frame) -> float:
        c1 = kf1.pose[:3, 3]
        c2 = kf2.pose[:3, 3]
        return float(np.linalg.norm(c2 - c1))

    def _parallax_deg(self, kf1: Frame, kf2: Frame, pts1: np.ndarray, pts2: np.ndarray, max_samples: int = 200) -> float:
        if len(pts1) == 0:
            return 0.0

        idx = np.arange(len(pts1))
        if len(idx) > max_samples:
            idx = np.random.choice(idx, size=max_samples, replace=False)

        pts1s = pts1[idx]
        pts2s = pts2[idx]

        x1 = np.hstack([pts1s, np.ones((len(pts1s), 1))]).T
        x2 = np.hstack([pts2s, np.ones((len(pts2s), 1))]).T

        r1_cam = (self.Kinv @ x1).T
        r2_cam = (self.Kinv @ x2).T

        r1_cam /= (np.linalg.norm(r1_cam, axis=1, keepdims=True) + 1e-12)
        r2_cam /= (np.linalg.norm(r2_cam, axis=1, keepdims=True) + 1e-12)

        R_w_c1 = kf1.pose[:3, :3]
        R_w_c2 = kf2.pose[:3, :3]
        r1_w = (R_w_c1 @ r1_cam.T).T
        r2_w = (R_w_c2 @ r2_cam.T).T

        dot = np.sum(r1_w * r2_w, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        ang = np.degrees(np.arccos(dot))
        return float(np.mean(ang))

    def _voxel_key(self, Xw: np.ndarray) -> Tuple[int, int, int]:
        v = self.voxel_size
        return (int(np.floor(Xw[0] / v)), int(np.floor(Xw[1] / v)), int(np.floor(Xw[2] / v)))

    def triangulate_from_keyframes(self, kf1: Frame, kf2: Frame, slam_map: Map) -> Tuple[int, Dict]:
        dbg = {
            "kf_matches_raw": 0,
            "kf_inliers_F": 0,
            "kf_baseline": 0.0,
            "kf_parallax_deg": 0.0,
            "kf_added": 0,
            "kf_status": "init"
        }

        if kf1.pose is None or kf2.pose is None:
            dbg["kf_status"] = "no_pose"
            return 0, dbg
        if kf1.keypoints is None or kf2.keypoints is None:
            dbg["kf_status"] = "no_kp"
            return 0, dbg
        if kf1.descriptors is None or kf2.descriptors is None:
            dbg["kf_status"] = "no_desc"
            return 0, dbg

        matches = self._match_keyframes(kf1, kf2)
        dbg["kf_matches_raw"] = len(matches)
        if len(matches) < 8:
            dbg["kf_status"] = "too_few_matches"
            return 0, dbg

        pts1 = np.float64([kf1.keypoints[m.queryIdx].pt for m in matches])
        pts2 = np.float64([kf2.keypoints[m.trainIdx].pt for m in matches])

        F, maskF = None, None
        try:
            F, maskF = cv2.findFundamentalMat(
                pts1, pts2,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=1.0,
                confidence=0.999
            )
        except cv2.error:
            try:
                F, maskF = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=1.0,
                    confidence=0.999
                )
            except cv2.error:
                F, maskF = None, None

        if F is None or maskF is None or F.shape != (3, 3) or (not np.isfinite(F).all()):
            dbg["kf_status"] = "F_failed"
            return 0, dbg

        inl = maskF.ravel().astype(bool)
        dbg["kf_inliers_F"] = int(np.count_nonzero(inl))
        if dbg["kf_inliers_F"] < 8:
            dbg["kf_status"] = "too_few_inliers_F"
            return 0, dbg

        pts1_in = pts1[inl]
        pts2_in = pts2[inl]
        matches_in = [m for m, keep in zip(matches, inl) if keep]

        b = self._baseline(kf1, kf2)
        dbg["kf_baseline"] = b
        if b < self.min_baseline:
            dbg["kf_status"] = "baseline_too_small"
            return 0, dbg

        par = self._parallax_deg(kf1, kf2, pts1_in, pts2_in)
        dbg["kf_parallax_deg"] = par
        if par < self.min_parallax_deg:
            dbg["kf_status"] = "parallax_too_small"
            return 0, dbg

        T1_cw = invert_T(kf1.pose)
        T2_cw = invert_T(kf2.pose)
        P1 = self.K @ T1_cw[:3, :]
        P2 = self.K @ T2_cw[:3, :]

        X_h = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T).T
        X = X_h[:, :3] / (X_h[:, 3:4] + 1e-12)

        good = np.isfinite(X).all(axis=1)
        X = X[good]
        pts1_in = pts1_in[good]
        pts2_in = pts2_in[good]
        matches_in = [m for m, keep in zip(matches_in, good) if keep]

        if len(X) == 0:
            dbg["kf_status"] = "no_finite_points"
            return 0, dbg

        def depth_in_cam(T_world_cam: np.ndarray, Pw: np.ndarray) -> np.ndarray:
            T_cw = invert_T(T_world_cam)
            Pc = (T_cw[:3, :3] @ Pw.T).T + T_cw[:3, 3].reshape(1, 3)
            return Pc[:, 2]

        z1 = depth_in_cam(kf1.pose, X)
        z2 = depth_in_cam(kf2.pose, X)
        ok_depth = (z1 > self.min_depth) & (z2 > self.min_depth) & (z1 < self.max_depth) & (z2 < self.max_depth)

        X = X[ok_depth]
        pts1_in = pts1_in[ok_depth]
        pts2_in = pts2_in[ok_depth]
        matches_in = [m for m, keep in zip(matches_in, ok_depth) if keep]

        if len(X) == 0:
            dbg["kf_status"] = "depth_rejected"
            return 0, dbg

        uv1_hat = project_points(self.K, kf1.pose, X)
        uv2_hat = project_points(self.K, kf2.pose, X)
        e1 = np.linalg.norm(uv1_hat - pts1_in, axis=1)
        e2 = np.linalg.norm(uv2_hat - pts2_in, axis=1)
        ok_rep = (e1 < self.max_reproj_px) & (e2 < self.max_reproj_px)

        X = X[ok_rep]
        matches_in = [m for m, keep in zip(matches_in, ok_rep) if keep]

        if len(X) == 0:
            dbg["kf_status"] = "reproj_rejected"
            return 0, dbg

        added = 0
        for j in range(min(len(X), self.max_new_points)):
            Xw = X[j].astype(np.float64)
            key = self._voxel_key(Xw)
            if key in slam_map.voxel_hash:
                continue

            m = matches_in[j]
            mp = Point(
                point_id=self.next_point_id,
                xyz_world=Xw,
                descriptor=kf2.descriptors[m.trainIdx].copy(),
                observations={kf1.frame_id: m.queryIdx, kf2.frame_id: m.trainIdx},
                num_obs=2
            )
            slam_map.points.append(mp)
            slam_map.voxel_hash[key] = mp.point_id
            self.next_point_id += 1
            added += 1

        dbg["kf_added"] = added
        dbg["kf_status"] = "ok"
        return added, dbg


# ============================================================
# Relocalization (PnP)
# ============================================================

class RelocalizerPnP:
    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def localize(self, frame: Frame, slam_map: Map, ratio: float = 0.75) -> Tuple[bool, dict, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        dbg = {
            "pnp_num_corr": 0,
            "pnp_inliers": 0,
            "pnp_reproj_before": None,
            "pnp_reproj_after": None,
            "status": "init"
        }

        if frame.descriptors is None or frame.keypoints is None:
            dbg["status"] = "no_features"
            return False, dbg, None, None, None
        if len(slam_map.points) < 30:
            dbg["status"] = "map_too_small"
            return False, dbg, None, None, None

        mp_desc = []
        mp_xyz = []
        for mp in slam_map.points:
            if mp.descriptor is None:
                continue
            mp_desc.append(mp.descriptor)
            mp_xyz.append(mp.xyz_world)

        if len(mp_desc) < 30:
            dbg["status"] = "no_map_desc"
            return False, dbg, None, None, None

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
            return False, dbg, None, None, None

        Pw = np.float64([mp_xyz[m.queryIdx] for m in good])
        uv = np.float64([frame.keypoints[m.trainIdx].pt for m in good])

        dbg["pnp_num_corr"] = len(good)

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
            return False, dbg, Pw, uv, None

        inliers_idx = inliers.reshape(-1)
        dbg["pnp_inliers"] = int(len(inliers_idx))

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3,)

        T_cam_world = np.eye(4, dtype=np.float64)
        T_cam_world[:3, :3] = R
        T_cam_world[:3, 3] = t
        T_world_cam = invert_T(T_cam_world)

        frame.pose = T_world_cam
        dbg["pnp_reproj_after"] = reprojection_error(self.K, frame.pose, Pw[inliers_idx], uv[inliers_idx])
        dbg["status"] = "ok"

        return True, dbg, Pw, uv, inliers_idx


# ============================================================
# Loop Closure (stub, keep for PDF structure)
# ============================================================

class LoopClosure:
    def detect(self, current_kf: Frame, slam_map: Map) -> Optional[int]:
        return None


# ============================================================
# VisPy Viewer (Trajectory + Map points)
# ============================================================

class VispyMapViewer:
    def __init__(self, title="Trajectory + 3D Map (VisPy)", size=(1100, 800)):
        if not VISPY_OK:
            raise RuntimeError("VisPy is not available. Install: pip install vispy")

        self.canvas = scene.SceneCanvas(keys="interactive", title=title, size=size, show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=60, distance=3.0)
        self.axes = visuals.XYZAxis(parent=self.view.scene)

        self.traj_line = visuals.Line(pos=np.zeros((1, 3), dtype=np.float32), parent=self.view.scene, method="gl")
        self.cur_marker = visuals.Markers(parent=self.view.scene)
        self.cur_marker.set_data(np.zeros((1, 3), dtype=np.float32), size=10)

        self.map_markers = visuals.Markers(parent=self.view.scene)
        self._last_map_count = 0

    def update(self, traj_xyz: List[Tuple[float, float, float]], map_points_xyz: np.ndarray):
        if traj_xyz is not None and len(traj_xyz) > 0:
            pts = np.asarray(traj_xyz, dtype=np.float32)
            self.traj_line.set_data(pos=pts)
            self.cur_marker.set_data(pts[-1:].copy(), size=10)

        if map_points_xyz is not None and len(map_points_xyz) > 0:
            if map_points_xyz.shape[0] != self._last_map_count:
                self._last_map_count = map_points_xyz.shape[0]
                max_show = 20000
                X = map_points_xyz
                if X.shape[0] > max_show:
                    idx = np.random.choice(X.shape[0], size=max_show, replace=False)
                    X = X[idx]
                self.map_markers.set_data(X.astype(np.float32), size=2)

        self.canvas.update()

    def process_events(self):
        self.canvas.app.process_events()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Monocular SLAM MVP (VO + Mapping + PnP + GN), VisPy + traj_est save")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to rgb folder with images")
    parser.add_argument("--keyframe_stride", type=int, default=10, help="Make a keyframe every N frames")
    parser.add_argument("--pnp_stride", type=int, default=10, help="Run PnP every N frames")
    parser.add_argument("--gn_iters", type=int, default=8, help="Gauss-Newton iterations after PnP")
    parser.add_argument("--show_every", type=int, default=1, help="Update visualization every N frames")
    parser.add_argument("--traj_out", type=str, default="traj_est.txt", help="Trajectory output file (TUM format)")
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
    optimizer = PoseOptimizerGN(K, huber_delta_px=3.0, eps=1e-4)
    slam_map = Map()
    loop = LoopClosure()  # stub (kept for PDF structure)

    # PDF windows
    cv2.namedWindow("image_keypoints", cv2.WINDOW_NORMAL)
    cv2.namedWindow("matches_raw", cv2.WINDOW_NORMAL)
    cv2.namedWindow("matches_filtered", cv2.WINDOW_NORMAL)

    viewer3d = None
    if VISPY_OK:
        viewer3d = VispyMapViewer()
    else:
        print("[WARN] VisPy not available -> viewer disabled.")

    # For match visualization, we store previous image+keypoints ourselves (cleanly).
    prev_img_for_matches = None
    prev_kps_for_matches = None

    last_keyframe: Optional[Frame] = None
    traj_xyz: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    traj_rows: List[Tuple[float, float, float, float, float, float, float, float]] = []  # (ts, tx,ty,tz, qx,qy,qz,qw)

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

        ok_vo, dbg = tracker.process_frame(frame)

        # --- trajectory + file rows
        if frame.pose is not None:
            t = frame.pose[:3, 3].astype(np.float64)
            R = frame.pose[:3, :3].astype(np.float64)
            qxyzw = rotmat_to_quat_xyzw(R)
            traj_xyz.append((float(t[0]), float(t[1]), float(t[2])))
            traj_rows.append((float(timestamp), float(t[0]), float(t[1]), float(t[2]),
                              float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2]), float(qxyzw[3])))

        # ===== Mapping
        if frame.pose is not None and (i % args.keyframe_stride == 0):
            mapper.make_keyframe(frame, slam_map)

            if last_keyframe is not None:
                added, map_dbg = mapper.triangulate_from_keyframes(last_keyframe, frame, slam_map)
                dbg.update(map_dbg)
                dbg["triangulated_new_points"] = int(added)
            else:
                dbg["triangulated_new_points"] = 0
                dbg["kf_status"] = "first_keyframe"

            last_keyframe = frame

        # ===== Relocalization + Optimization
        if frame.pose is not None and (i % args.pnp_stride == 0):
            ok_pnp, pnp_dbg, Pw, uv, inl_idx = relocalizer.localize(frame, slam_map)
            dbg["pnp_status"] = pnp_dbg["status"]
            dbg["pnp_corr"] = pnp_dbg["pnp_num_corr"]
            dbg["pnp_inliers"] = pnp_dbg["pnp_inliers"]
            dbg["reproj_before"] = pnp_dbg["pnp_reproj_before"]
            dbg["reproj_after_pnp"] = pnp_dbg["pnp_reproj_after"]

            if ok_pnp and inl_idx is not None and len(inl_idx) >= 10:
                max_use = 200
                idx_use = inl_idx
                if len(idx_use) > max_use:
                    idx_use = np.random.choice(idx_use, size=max_use, replace=False)

                T_refined, gn_dbg = optimizer.refine(frame.pose, Pw[idx_use], uv[idx_use], max_iters=args.gn_iters)
                frame.pose = T_refined

                dbg["gn_status"] = gn_dbg["gn_status"]
                dbg["gn_iters"] = gn_dbg["gn_iters"]
                dbg["reproj_after_gn"] = gn_dbg["gn_reproj_after"]
            else:
                dbg["gn_status"] = "skip"

        # ============================================================
        # OpenCV visualization windows
        # ============================================================

        vis = img.copy()
        if frame.keypoints is not None:
            vis = cv2.drawKeypoints(vis, frame.keypoints, None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        text1 = (f"id={i} kp={dbg.get('num_keypoints',0)} "
                 f"rawM={dbg.get('num_matches_raw',0)} goodM={dbg.get('num_matches_good',0)} "
                 f"inl={dbg.get('num_inliers',0)} st={dbg.get('status','')}")
        cv2.putText(vis, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 2)

        text2 = (f"map={len(slam_map.points)} kf={len(slam_map.keyframes)} "
                 f"new={dbg.get('triangulated_new_points',0)} kfSt={dbg.get('kf_status','-')} "
                 f"pnp={dbg.get('pnp_status','-')} rpP={dbg.get('reproj_after_pnp','-')} "
                 f"rpGN={dbg.get('reproj_after_gn','-')}")
        cv2.putText(vis, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 200, 255), 2)

        cv2.imshow("image_keypoints", vis)

        # Matches: raw vs filtered (use the stored previous image/kps)
        if prev_img_for_matches is not None and prev_kps_for_matches is not None and frame.keypoints is not None:
            raw = dbg.get("raw_matches", None) or []
            good = dbg.get("good_matches", None) or []
            inl_mask = dbg.get("inlier_mask", None)

            raw_show = raw[:80]
            if len(raw_show) > 0:
                raw_img = cv2.drawMatches(
                    prev_img_for_matches, prev_kps_for_matches,
                    frame.image_bgr, frame.keypoints,
                    raw_show, None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
            else:
                raw_img = np.zeros((200, 600, 3), dtype=np.uint8)

            good_show = good[:80]
            if len(good_show) > 0:
                if inl_mask is not None and len(inl_mask) == len(good):
                    subset_mask = [1 if bool(x) else 0 for x in inl_mask[:len(good_show)]]
                else:
                    subset_mask = None

                try:
                    filt_img = cv2.drawMatches(
                        prev_img_for_matches, prev_kps_for_matches,
                        frame.image_bgr, frame.keypoints,
                        good_show, None,
                        matchColor=None,
                        singlePointColor=None,
                        matchesMask=subset_mask,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                        matchesThickness=1
                    )
                except TypeError:
                    filt_img = cv2.drawMatches(
                        prev_img_for_matches, prev_kps_for_matches,
                        frame.image_bgr, frame.keypoints,
                        good_show, None,
                        matchColor=None,
                        singlePointColor=None,
                        matchesMask=subset_mask,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
            else:
                filt_img = np.zeros((200, 600, 3), dtype=np.uint8)

            cv2.imshow("matches_raw", raw_img)
            cv2.imshow("matches_filtered", filt_img)

        # Update stored previous for next iteration
        prev_img_for_matches = frame.image_bgr.copy()
        prev_kps_for_matches = frame.keypoints

        # ============================================================
        # VisPy window
        # ============================================================

        if viewer3d is not None and (i % args.show_every == 0):
            if len(slam_map.points) > 0:
                Xmap = np.vstack([mp.xyz_world.reshape(1, 3) for mp in slam_map.points]).astype(np.float32)
            else:
                Xmap = np.zeros((0, 3), dtype=np.float32)
            viewer3d.update(traj_xyz, Xmap)
            viewer3d.process_events()

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()

    # ============================================================
    # Save trajectory (TUM format)
    # ============================================================
    try:
        with open(args.traj_out, "w", encoding="utf-8") as f:
            for row in traj_rows:
                ts, tx, ty, tz, qx, qy, qz, qw = row
                f.write(f"{ts:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
        print(f"[SAVE] Trajectory written: {args.traj_out} ({len(traj_rows)} poses)")
    except Exception as e:
        print(f"[WARN] Failed to write traj file: {e}")

    print("[DONE] run finished.")
    print(f"Keyframes: {len(slam_map.keyframes)} | Map points: {len(slam_map.points)}")


if __name__ == "__main__":
    main()
