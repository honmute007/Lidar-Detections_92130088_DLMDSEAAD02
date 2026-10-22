"""LiDAR dynamic object tracking."""
import os
import glob
import time
import threading
from collections import deque, Counter

import numpy as np
import pandas as pd
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial.distance import cdist

CLASS_ORDER = ["Pedestrian", "Cyclist", "Car", "Big Vehicle", "Object"]
CLASS_COLORS = {
    "Pedestrian": [0.00, 0.55, 0.15],
    "Cyclist": [0.00, 0.42, 0.75],
    "Car": [0.80, 0.05, 0.05],
    "Big Vehicle": [0.85, 0.42, 0.00],
    "Object": [0.55, 0.10, 0.75],
}
BACKGROUND_COLOR = [1.0, 1.0, 1.0, 1.0]
MAX_LUMINANCE = 0.30


def build_lut(size: int = 256, max_luminance: float = MAX_LUMINANCE) -> np.ndarray:
    t = np.linspace(0.0, 1.0, size)
    lut = np.clip(np.stack([
        1.5 - np.abs(4.0 * t - 3.0),
        1.5 - np.abs(4.0 * t - 2.0),
        1.5 - np.abs(4.0 * t - 1.0),
    ], axis=1), 0.0, 1.0)

    if max_luminance is None:
        return lut

    lum = 0.2126 * lut[:, 0] + 0.7152 * lut[:, 1] + 0.0722 * lut[:, 2]
    factor = np.minimum(1.0, max_luminance / np.maximum(lum, 1e-6))
    return np.clip(lut * factor[:, None], 0.0, 1.0)


POINT_LUT = build_lut()


# --------------------------------------------------------------------------- #
# 1. Data Loader
# --------------------------------------------------------------------------- #
class LidarDataLoader:
    """Discovers and parses CSV frames."""

    COLUMNS = ["X", "Y", "Z"]

    def __init__(self, parent_dir: str, min_range_m: float = 0.3, max_range_m: float = 60.0):
        self.parent_dir = parent_dir
        self.min_range_m = min_range_m
        self.max_range_m = max_range_m

    def get_data_files(self) -> list:
        """All CSV frames under the parent directory, sorted."""
        files = sorted(glob.glob(os.path.join(self.parent_dir, "**", "*.csv"), recursive=True))
        print(f"Found {len(files)} frame(s) across all subfolders." if files
              else f"No .csv files found under directory tree: {self.parent_dir}")
        return files

    def read_csv(self, filepath: str):
        """Parse one CSV, apply the range gate, return (points, raw_row_count)."""
        df = pd.read_csv(filepath, sep=";", header=0, usecols=self.COLUMNS, engine="c")
        loc = df.to_numpy(dtype=np.float64)
        radius = np.linalg.norm(loc, axis=1)
        keep = (radius > self.min_range_m) & (radius < self.max_range_m)
        return loc[keep], len(df)


# --------------------------------------------------------------------------- #
# 2. Ground Removal (RANSAC)
# --------------------------------------------------------------------------- #
class RansacGroundRemover:
    """Segments and strips horizontal planar surfaces (road/ground)."""

    def __init__(self, distance_threshold: float = 0.18, ransac_n: int = 3,
                 num_iterations: int = 500, min_points_for_fit: int = 200,
                 min_vertical_normal: float = 0.85):
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.min_points_for_fit = min_points_for_fit
        self.min_vertical_normal = min_vertical_normal

    def remove_ground(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Fit a plane and drop its inliers, unless the fit is not horizontal."""
        if len(pcd.points) < self.min_points_for_fit:
            return pcd

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )
        if not len(inliers):
            return pcd

        normal = np.asarray(plane_model[:3], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm == 0 or abs(normal[2] / norm) < self.min_vertical_normal:
            return pcd

        return pcd.select_by_index(inliers, invert=True)


# --------------------------------------------------------------------------- #
# 3. Static Background Filter (Voxel Persistence)
# --------------------------------------------------------------------------- #
class VoxelStaticFilter:
    """Strips persistent static background targets (walls, poles, fences)."""

    _SHIFT = 1 << 20

    def __init__(self, voxel_size: float = 0.35, persistence_threshold: int = 4,
                 warmup_frames: int = 2, forget_after: int = 60):
        self.voxel_size = voxel_size
        self.persistence_threshold = persistence_threshold
        self.warmup_frames = warmup_frames
        self.forget_after = forget_after
        self.voxel_hits = {}
        self.frame_idx = 0

    @classmethod
    def _pack(cls, v: np.ndarray) -> np.ndarray:
        """Pack a 3D voxel index into one int64 key."""
        s = cls._SHIFT
        return ((v[:, 0] + s) << 42) | ((v[:, 1] + s) << 21) | (v[:, 2] + s)

    def filter_static_environment(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Drop points in voxels seen on at least persistence_threshold frames."""
        points = np.asarray(pcd.points)
        if not len(points):
            return pcd

        self.frame_idx += 1
        frame = self.frame_idx

        keys = self._pack(np.floor(points / self.voxel_size).astype(np.int64))
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)

        counts = np.empty(len(unique_keys), dtype=np.int32)
        hits = self.voxel_hits

        for i, k in enumerate(unique_keys.tolist()):
            entry = hits.get(k)
            if entry is None:
                hits[k] = [1, frame]
                counts[i] = 1
            else:
                entry[0] += 1
                entry[1] = frame
                counts[i] = entry[0]

        if frame % 30 == 0:
            cutoff = frame - self.forget_after
            self.voxel_hits = {k: v for k, v in hits.items() if v[1] >= cutoff}

        if frame <= self.warmup_frames:
            return pcd

        dynamic = np.flatnonzero(~(counts >= self.persistence_threshold)[inverse])
        if dynamic.size == len(points):
            return pcd
        return pcd.select_by_index(dynamic.tolist())


# --------------------------------------------------------------------------- #
# 4. DBSCAN Clustering & Oriented Bounding Boxes
# --------------------------------------------------------------------------- #
class DbscanClusterer:
    """Clusters points and builds yaw-constrained OrientedBoundingBoxes."""

    def __init__(self, eps: float = 0.65, min_points: int = 12, voxel_size: float = 0.12,
                 min_volume: float = 0.05, max_volume: float = 100.0,
                 class_hysteresis_m: float = 0.35, degenerate_ratio: float = 0.85,
                 outlier_radius: float = 0.0, outlier_neighbours: int = 4):
        self.eps = eps
        self.min_points = min_points
        self.voxel_size = voxel_size
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.class_hysteresis_m = class_hysteresis_m
        self.degenerate_ratio = degenerate_ratio
        self.outlier_radius = outlier_radius
        self.outlier_neighbours = outlier_neighbours

    def _footprint(self, points: np.ndarray):
        """Yaw-only OBB parameters from a 2D PCA of the ground-plane footprint."""
        if len(points) < 3:
            return None

        xy = points[:, :2]
        centre = xy.mean(axis=0)
        centered = xy - centre

        cov = np.cov(centered, rowvar=False)
        if not np.isfinite(cov).all():
            return None

        evals, evecs = np.linalg.eigh(cov)

        for j in range(2):
            if evecs[int(np.argmax(np.abs(evecs[:, j]))), j] < 0:
                evecs[:, j] *= -1.0
        if np.linalg.det(evecs) < 0:
            evecs[:, 0] *= -1.0

        proj = centered @ evecs
        u_min, u_max = float(proj[:, 0].min()), float(proj[:, 0].max())
        v_min, v_max = float(proj[:, 1].min()), float(proj[:, 1].max())
        z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())

        lo, hi = float(abs(evals[0])), float(abs(evals[1]))
        degenerate = bool(hi > 0 and lo / hi >= self.degenerate_ratio)

        centre_xy = centre + evecs[:, 0] * (u_min + u_max) / 2 + evecs[:, 1] * (v_min + v_max) / 2
        rot = np.eye(3)
        rot[:2, :2] = evecs

        return {
            "centre3": np.array([centre_xy[0], centre_xy[1], (z_min + z_max) / 2]),
            "rot": rot,
            "extent": np.array([u_max - u_min, v_max - v_min, z_max - z_min]),
            "z_min": z_min,
            "z_max": z_max,
            "degenerate": degenerate,
        }

    @staticmethod
    def to_obb(det: dict, colour) -> o3d.geometry.OrientedBoundingBox:
        """Build a coloured OrientedBoundingBox from a detection dict."""
        obb = o3d.geometry.OrientedBoundingBox(det["centre3"], det["rot"], det["extent"])
        obb.color = colour
        return obb

    def classify(self, length: float, width: float, height: float, current: str = None):
        """Size to class, with boundaries relaxed in favour of `current`."""
        volume = length * width * height
        if volume < self.min_volume or volume > self.max_volume:
            return None, None

        m = self.class_hysteresis_m
        big = 5.0 - m if current == "Big Vehicle" else 5.0 + m
        big_alt = 3.8 - m if current == "Big Vehicle" else 3.8 + m
        car = 2.6 - m if current in ("Car", "Big Vehicle") else 2.6 + m
        cyc = 1.1 - m if current in ("Cyclist", "Car", "Big Vehicle") else 1.1 + m

        if length >= big or (length >= big_alt and height >= 2.3):
            label = "Big Vehicle"
        elif length >= car and 0.8 <= height <= 2.5:
            label = "Car"
        elif length >= cyc and 0.7 <= height <= 2.0:
            label = "Cyclist"
        elif length < cyc and 0.5 <= height <= 2.1:
            label = "Pedestrian"
        else:
            label = "Object"

        return label, CLASS_COLORS[label]

    def extract_clusters(self, pcd: o3d.geometry.PointCloud) -> list:
        """Cluster and return unlabelled detections; the tracker classifies."""
        if len(pcd.points) < self.min_points:
            return []

        # Downsampling equalises range-dependent density so one eps works scene-wide.
        work = pcd.voxel_down_sample(self.voxel_size) if self.voxel_size > 0 else pcd

        if self.outlier_radius > 0 and len(work.points) > self.outlier_neighbours:
            work, _ = work.remove_radius_outlier(nb_points=self.outlier_neighbours,
                                                 radius=self.outlier_radius)

        if len(work.points) < self.min_points:
            return []

        labels = np.asarray(work.cluster_dbscan(self.eps, self.min_points, print_progress=False))
        if labels.size == 0 or labels.max() < 0:
            return []

        points = np.asarray(work.points)
        detections = []

        for cid in range(labels.max() + 1):
            member = labels == cid
            if int(member.sum()) < self.min_points:
                continue

            cluster = points[member]
            fp = self._footprint(cluster)
            if fp is None:
                continue

            width, length, height = (float(v) for v in fp["extent"])
            if min(length, width, height) <= 0:
                continue

            volume = length * width * height
            if volume < self.min_volume or volume > self.max_volume:
                continue

            centroid = cluster.mean(axis=0)
            fp.update({
                "points": int(member.sum()),
                "length_m": round(length, 3),
                "width_m": round(width, 3),
                "height_m": round(height, 3),
                "volume_m3": round(volume, 3),
                "centre_x_m": round(float(centroid[0]), 3),
                "centre_y_m": round(float(centroid[1]), 3),
                "z_min_m": round(fp["z_min"], 3),
                "z_max_m": round(fp["z_max"], 3),
                "range_m": round(float(np.linalg.norm(centroid[:2])), 3),
            })
            detections.append(fp)

        return detections


# --------------------------------------------------------------------------- #
# 5. Object Tracker with Hysteresis
# --------------------------------------------------------------------------- #
class Track:
    """Centroid history, smoothed velocity, latched motion state, voted class."""

    def __init__(self, track_id: int, detection: dict, frame_idx: int, fps: float,
                 classifier: DbscanClusterer, class_vote_window: int = 9,
                 dim_decay: float = 0.97, vote_margin: int = 2):
        self.track_id = track_id
        self.classifier = classifier
        self.dim_decay = dim_decay
        self.vote_margin = vote_margin

        self.centroid = np.array([detection["centre_x_m"], detection["centre_y_m"],
                                  detection["z_min_m"]])
        self.velocity = np.zeros(3)
        self.speed_mps = 0.0
        self.last_frame = frame_idx
        self.hits = 1
        self.fps = fps
        self.detection = detection
        self.is_moving = False
        self.slow_frames = 0

        self.len_f = float(detection["length_m"])
        self.wid_f = float(detection["width_m"])
        self.hgt_f = float(detection["height_m"])

        self.class_history = deque(maxlen=class_vote_window)
        label, color = classifier.classify(self.len_f, self.wid_f, self.hgt_f, None)
        if label is None:
            label, color = "Object", CLASS_COLORS["Object"]
        self.stable_class = label
        self.raw_class = label
        self.color = color
        self.class_history.append(label)

    def _maybe_swap_axes(self, det: dict) -> dict:
        """Undo an arbitrary length/width swap on a near-square footprint."""
        if not det.get("degenerate"):
            return det

        L, W = float(det["length_m"]), float(det["width_m"])
        if abs(W - self.len_f) + abs(L - self.wid_f) >= abs(L - self.len_f) + abs(W - self.wid_f):
            return det

        d = dict(det)
        d["length_m"], d["width_m"] = det["width_m"], det["length_m"]
        rot = det["rot"].copy()
        rot[:, [0, 1]] = rot[:, [1, 0]]
        rot[:, 0] *= -1.0
        d["rot"] = rot
        d["extent"] = det["extent"][[1, 0, 2]]
        return d

    def _update_dims(self, det: dict):
        """Grow immediately, shrink slowly: occlusion under-measures, never over."""
        for key, attr in (("length_m", "len_f"), ("width_m", "wid_f"), ("height_m", "hgt_f")):
            new, old = float(det[key]), getattr(self, attr)
            setattr(self, attr, new if new >= old else max(new, old * self.dim_decay))

    def _vote_class(self):
        """Switch class only on a clear, sustained majority against the incumbent."""
        top, top_n = Counter(self.class_history).most_common(1)[0]
        if top == self.stable_class:
            return
        incumbent = sum(1 for c in self.class_history if c == self.stable_class)
        if top_n >= incumbent + self.vote_margin and top_n * 2 > len(self.class_history):
            self.stable_class = top
            self.color = CLASS_COLORS[top]

    def update(self, detection: dict, frame_idx: int):
        """Associate a detection: refresh kinematics, dimensions and class."""
        detection = self._maybe_swap_axes(detection)
        new_centroid = np.array([detection["centre_x_m"], detection["centre_y_m"],
                                 detection["z_min_m"]])
        dt = max((frame_idx - self.last_frame) / self.fps, 1e-3)

        # Smoothed: the centroid shifts as the visible face changes.
        self.velocity = 0.4 * self.velocity + 0.6 * (new_centroid - self.centroid) / dt
        self.speed_mps = float(np.linalg.norm(self.velocity[:2]))

        self.centroid = new_centroid
        self.last_frame = frame_idx
        self.hits += 1
        self.detection = detection

        self._update_dims(detection)
        label, _ = self.classifier.classify(self.len_f, self.wid_f, self.hgt_f, self.stable_class)
        if label is not None:
            self.raw_class = label
            self.class_history.append(label)
            self._vote_class()

    def update_motion_state(self, enter_speed, exit_speed, release_frames):
        """Latch moving/stationary with separate enter and exit thresholds."""
        if not self.is_moving:
            if self.speed_mps >= enter_speed:
                self.is_moving = True
                self.slow_frames = 0
        elif self.speed_mps >= exit_speed:
            self.slow_frames = 0
        else:
            self.slow_frames += 1
            if self.slow_frames >= release_frames:
                self.is_moving = False


class ObjectTracker:
    """Tracks objects across frames, classifies them, isolates dynamic targets."""

    def __init__(self, classifier: DbscanClusterer, max_distance_m: float = 2.5,
                 min_hits: int = 2, min_speed_mps: float = 0.6, fps: float = 10.0,
                 max_age: int = 3, release_ratio: float = 0.55, release_frames: int = 5,
                 coast_frames: int = 2, class_vote_window: int = 9,
                 dim_decay: float = 0.97, vote_margin: int = 2):
        self.classifier = classifier
        self.max_distance_m = max_distance_m
        self.min_hits = min_hits
        self.min_speed_mps = min_speed_mps
        self.exit_speed_mps = min_speed_mps * release_ratio
        self.release_frames = release_frames
        self.coast_frames = coast_frames
        self.fps = fps
        self.max_age = max_age
        self.class_vote_window = class_vote_window
        self.dim_decay = dim_decay
        self.vote_margin = vote_margin
        self.tracks = []
        self.next_id = 1

    def update(self, detections: list, frame_idx: int) -> list:
        """Associate, age out, and return the tracks that should be drawn."""
        matched = set()

        if detections:
            det_c = np.array([[d["centre_x_m"], d["centre_y_m"], d["z_min_m"]] for d in detections])

            if self.tracks:
                dists = cdist(np.array([t.centroid for t in self.tracks]), det_c)
                # Greedy: repeatedly take the globally closest surviving pair.
                for _ in range(min(len(self.tracks), len(detections))):
                    if dists.min() > self.max_distance_m:
                        break
                    t_idx, d_idx = np.unravel_index(dists.argmin(), dists.shape)
                    self.tracks[t_idx].update(detections[d_idx], frame_idx)
                    matched.add(int(d_idx))
                    dists[t_idx, :] = 1e6
                    dists[:, d_idx] = 1e6

            for d_idx, det in enumerate(detections):
                if d_idx not in matched:
                    self.tracks.append(Track(self.next_id, det, frame_idx, self.fps,
                                             self.classifier, self.class_vote_window,
                                             self.dim_decay, self.vote_margin))
                    self.next_id += 1

        self.tracks = [t for t in self.tracks if frame_idx - t.last_frame <= self.max_age]
        for t in self.tracks:
            t.update_motion_state(self.min_speed_mps, self.exit_speed_mps, self.release_frames)

        return [t for t in self.tracks
                if t.is_moving and t.hits >= self.min_hits
                and frame_idx - t.last_frame <= self.coast_frames]


# --------------------------------------------------------------------------- #
# 6. Excel Reporting Logger
# --------------------------------------------------------------------------- #
class DetectionLogger:
    """Exports per-frame and per-detection metrics to Excel."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.frame_rows = []
        self.detection_rows = []

    def log_frame(self, frame_idx, filepath, raw_points, valid_points, object_points, tracks):
        """Record one frame and each of its drawn tracks."""
        counts = {c: 0 for c in CLASS_ORDER}      # seeded, so columns never go ragged
        for t in tracks:
            counts[t.stable_class] = counts.get(t.stable_class, 0) + 1

        row = {"Frame": frame_idx, "File": os.path.basename(filepath),
               "Raw Points": raw_points, "Valid Points": valid_points,
               "Filtered Points": object_points}
        row.update(counts)
        row["Moving Objects"] = len(tracks)
        self.frame_rows.append(row)

        for t in tracks:
            d = t.detection
            self.detection_rows.append({
                "Frame": frame_idx, "Track ID": t.track_id,
                "Class": t.stable_class, "Raw Class": t.raw_class,
                "Class Stable": "Yes" if t.stable_class == t.raw_class else "No",
                "Speed (m/s)": round(t.speed_mps, 2),
                "Speed (km/h)": round(t.speed_mps * 3.6, 2),
                "Coasting": "Yes" if t.last_frame != frame_idx else "No",
                "Length (m)": d["length_m"], "Width (m)": d["width_m"],
                "Height (m)": d["height_m"],
                "Length filt (m)": round(t.len_f, 3),
                "Width filt (m)": round(t.wid_f, 3),
                "Height filt (m)": round(t.hgt_f, 3),
                "Axes Degenerate": "Yes" if d.get("degenerate") else "No",
                "Volume (m3)": d["volume_m3"],
                "Centre X (m)": d["centre_x_m"], "Centre Y (m)": d["centre_y_m"],
                "Range (m)": d["range_m"],
            })

    def write(self):
        """Write both sheets to disk."""
        if not self.frame_rows:
            return
        detections = (pd.DataFrame(self.detection_rows) if self.detection_rows
                      else pd.DataFrame(columns=["Frame", "Track ID", "Class"]))
        try:
            with pd.ExcelWriter(self.output_path, engine="openpyxl") as w:
                pd.DataFrame(self.frame_rows).to_excel(w, sheet_name="Per Frame", index=False)
                detections.to_excel(w, sheet_name="Moving Detections", index=False)
            print(f"Excel report generated: {self.output_path}")
        except PermissionError:
            print(f"Could not write {self.output_path} -- close it in Excel and re-run.")
        except Exception as e:
            print(f"Could not write Excel report: {e}")


# --------------------------------------------------------------------------- #
# 7. Application
# --------------------------------------------------------------------------- #
class LidarApp:
    """O3DVisualizer front end; processing runs on a worker thread."""

    def __init__(self, fps: float = 10.0, eye=(0.0, -35.0, 14.0), centre=(0.0, 15.0, 0.0),
                 point_size: int = 3, window_size=(1280, 720), show_ground: bool = True,
                 max_colour_range_m: float = 40.0, print_every: int = 25,
                 speed: float = 1.0, start_paused: bool = False):
        self.fps = fps
        self.eye = np.asarray(eye, dtype=np.float64)
        self.centre = np.asarray(centre, dtype=np.float64)
        self.point_size = point_size
        self.window_size = window_size
        self.show_ground = show_ground
        self.max_colour_range_m = max_colour_range_m
        self.print_every = print_every

        self._running = True
        self._camera_set = False
        self._label_colour_ok = True

        # Playback state, driven by the toolbar buttons.
        self.SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
        self._speed_idx = min(range(len(self.SPEEDS)),
                              key=lambda i: abs(self.SPEEDS[i] - speed))
        self._paused = start_paused
        self._step_once = False

        self._pcd_mat = rendering.MaterialRecord()
        self._pcd_mat.shader = "defaultUnlit"
        self._pcd_mat.point_size = float(point_size)

        self._line_mat = rendering.MaterialRecord()
        self._line_mat.shader = "unlitLine"
        self._line_mat.line_width = 2.5

    @property
    def speed(self) -> float:
        """Current playback multiplier."""
        return self.SPEEDS[self._speed_idx]

    def _slower(self, _vis=None):
        """Step down one speed preset."""
        self._speed_idx = max(0, self._speed_idx - 1)
        print(f"Playback speed: {self.speed:g}x ({self.fps * self.speed:g} fps)")

    def _faster(self, _vis=None):
        """Step up one speed preset."""
        self._speed_idx = min(len(self.SPEEDS) - 1, self._speed_idx + 1)
        print(f"Playback speed: {self.speed:g}x ({self.fps * self.speed:g} fps)")

    def _toggle_pause(self, _vis=None):
        """Pause or resume the frame loop."""
        self._paused = not self._paused
        print("Paused." if self._paused else f"Playing at {self.speed:g}x.")

    def _step(self, _vis=None):
        """Advance exactly one frame while paused."""
        self._step_once = True
        if not self._paused:
            self._paused = True
            print("Paused.")

    def _colours(self, points: np.ndarray) -> np.ndarray:
        """Per-point colour by radial distance."""
        if not len(points):
            return np.zeros((0, 3))
        radius = np.linalg.norm(points, axis=1)
        idx = (np.clip(radius / self.max_colour_range_m, 0, 1) * 255).astype(np.int32)
        return POINT_LUT[idx]

    @staticmethod
    def _merge_boxes(tracks) -> o3d.geometry.LineSet:
        """Merge every track's wireframe into a single LineSet."""
        merged = o3d.geometry.LineSet()
        for t in tracks:
            merged += o3d.geometry.LineSet.create_from_oriented_bounding_box(
                DbscanClusterer.to_obb(t.detection, t.color))
        return merged

    def _apply_frame(self, vis, points: np.ndarray, tracks: list, frame_idx: int):
        """Push one frame's geometry and labels to the scene (GUI thread only)."""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(self._colours(points))

            for name in ("points", "boxes"):
                if vis.scene.has_geometry(name):
                    vis.remove_geometry(name)

            vis.add_geometry("points", pcd, self._pcd_mat)
            if tracks:
                vis.add_geometry("boxes", self._merge_boxes(tracks), self._line_mat)

            vis.clear_3d_labels()
            for t in tracks:
                d = t.detection
                label = vis.add_3d_label(
                    (d["centre_x_m"], d["centre_y_m"], d["z_max_m"] + 0.4),
                    f"ID:{t.track_id} {t.stable_class} {t.speed_mps * 3.6:.0f}km/h")

                # add_3d_label returns a handle on some builds and None on others.
                if label is not None and self._label_colour_ok:
                    try:
                        label.color = gui.Color(*t.color)
                    except Exception:
                        self._label_colour_ok = False
                        print("Label colouring unavailable on this Open3D build; "
                              "labels will use the default colour.")

            if not self._camera_set:
                vis.setup_camera(60.0, self.centre, self.eye, [0.0, 0.0, 1.0])
                self._camera_set = True

        except Exception as exc:
            # An escaping exception would unwind the GUI loop and kill the window.
            print(f"Frame {frame_idx} display failed: {exc}")

    def _worker(self, vis, loader, files, ground, static, clusterer, tracker, logger):
        """Frame loop: all heavy work, off the GUI thread."""
        app = gui.Application.instance

        for frame_idx, path in enumerate(files):
            if not self._running:
                break

            # Block here while paused, polling so Pause/Step stay responsive.
            while self._paused and self._running and not self._step_once:
                time.sleep(0.05)
            self._step_once = False
            if not self._running:
                break

            start = time.time()

            loc, n_raw = loader.read_csv(path)
            if not len(loc):
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(loc)
            pcd = ground.remove_ground(pcd)
            pcd = static.filter_static_environment(pcd)

            detections = clusterer.extract_clusters(pcd)
            moving = tracker.update(detections, frame_idx)

            points = np.asarray(pcd.points).copy()
            if logger is not None:
                logger.log_frame(frame_idx, path, n_raw, len(loc), len(points), moving)

            if self.print_every and frame_idx % self.print_every == 0:
                print(f"[{frame_idx:4d}] {len(points):5d} obstacle pts | "
                      f"detections: {len(detections):2d} | moving: {len(moving)}")

            app.post_to_main_thread(
                vis, lambda p=points, m=list(moving), i=frame_idx: self._apply_frame(vis, p, m, i))


            budget = 1.0 / (self.fps * self.speed)
            while self._running:
                remaining = budget - (time.time() - start)
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.05))

        if logger is not None:
            logger.write()

    def run(self, loader, files, ground, static, clusterer, tracker, logger):
        """Open the window and start the frame loop."""
        if not files:
            print("No files to process.")
            return

        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer("LiDAR Dynamic Tracking",
                                              self.window_size[0], self.window_size[1])
        vis.show_settings = False
        vis.show_skybox(False)
        vis.set_background(BACKGROUND_COLOR, None)
        vis.show_ground = self.show_ground
        vis.ground_plane = rendering.Scene.GroundPlane.XY
        vis.point_size = self.point_size

        def on_close():
            self._running = False
            return True

        vis.set_on_close(on_close)

        vis.add_action("<< Slower", self._slower)
        vis.add_action("Pause / Play", self._toggle_pause)
        vis.add_action("Faster >>", self._faster)
        vis.add_action("Step +1", self._step)

        app.add_window(vis)

        print(f"Playback: {self.speed:g}x ({self.fps * self.speed:g} fps). "
              f"Use the toolbar buttons to change speed, pause or step.")
        print("View: scroll to zoom, left-drag to rotate, shift+left-drag to pan.")

        threading.Thread(
            target=self._worker,
            args=(vis, loader, files, ground, static, clusterer, tracker, logger),
            daemon=True).start()

        app.run()
        self._running = False


# --------------------------------------------------------------------------- #
# Execution Entry Point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    root_folder = r"C:\Users\pravin\Desktop\PythonProject1\LiDAR data"
    report_path = os.path.join(root_folder, "moving_objects_report.xlsx")

    loader = LidarDataLoader(parent_dir=root_folder, min_range_m=0.3, max_range_m=50.0)

    ground_remover = RansacGroundRemover(distance_threshold=0.18, num_iterations=500)
    static_filter = VoxelStaticFilter(voxel_size=0.35, persistence_threshold=4, forget_after=60)

    clusterer = DbscanClusterer(
        eps=0.65,
        min_points=12,
        voxel_size=0.12,
        class_hysteresis_m=0.35,
        degenerate_ratio=0.85,
        outlier_radius=0.0,
        outlier_neighbours=4,
    )

    tracker = ObjectTracker(
        classifier=clusterer,
        max_distance_m=2.5,
        min_hits=2,
        min_speed_mps=0.6,
        release_ratio=0.55,
        release_frames=5,
        coast_frames=2,
        fps=10.0,
        class_vote_window=9,
        vote_margin=2,
        dim_decay=0.97,
    )

    logger = DetectionLogger(output_path=report_path)
    files = loader.get_data_files()

    app = LidarApp(
        fps=10.0,
        eye=(0.0, -35.0, 14.0),
        centre=(0.0, 15.0, 0.0),
        point_size=3,
        window_size=(1280, 720),
        show_ground=True,
        max_colour_range_m=40.0,
        print_every=25,
        speed=1.0,
        start_paused=False,
    )

    app.run(
        loader=loader,
        files=files,
        ground=ground_remover,
        static=static_filter,
        clusterer=clusterer,
        tracker=tracker,
        logger=logger,
    )