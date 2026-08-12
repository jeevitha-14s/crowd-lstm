"""Per-frame spatial-grid feature extraction for crowd risk forecasting."""

from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
from scipy.stats import entropy as shannon_entropy

N_FEATURES_PER_ZONE = 4
N_DIRECTION_BINS = 8


class FeatureExtractor:
    """Turns per-frame person detections into a 64-dim spatial-grid feature vector."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        grid_rows: int = 4,
        grid_cols: int = 4,
        history_len: int = 5,
        stop_threshold: float = 0.5,
    ) -> None:
        """stop_threshold is a fraction of the clip's running median track
        speed (NOT an absolute px/frame value): a track counts as "stopped"
        if its speed is below stop_threshold * running_median_speed. This is
        relative rather than absolute because typical walking speed varies
        4x+ between UMN's scenes (different camera distances), so no single
        px/frame cutoff is well-calibrated across all clips.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.history_len = history_len
        self.stop_threshold = stop_threshold

        self.track_history: Dict[int, Deque[Tuple[float, float]]] = {}
        self.running_max_count: float = 1.0
        self.running_max_speed: float = 1.0
        # Median, not max: a single mistracked/teleported detection inflates a
        # running max for the rest of the clip, but barely moves a median.
        self.speed_history: List[float] = []
        self.running_median_speed: float = 1.0

    def reset(self) -> None:
        """Clear all per-scene state (track history and running maxima).

        Must be called at every clip/scene boundary: track IDs are only unique
        within a scene, and track_history is never purged of departed tracks
        on its own (it is a plain dict of every ID ever seen since the last
        reset). All internal lookups index it by the current frame's
        detection IDs only, never by iterating its keys, so stale entries are
        harmless at this dataset's scale -- but callers should not rely on
        that and should reset() between scenes instead.
        """
        self.track_history.clear()
        self.running_max_count = 1.0
        self.running_max_speed = 1.0
        self.speed_history.clear()
        self.running_median_speed = 1.0

    def extract(self, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Compute the 64-dim feature vector for one frame of detections.

        Note: new track IDs contribute to normalized_count immediately but stay
        at zero for flow/entropy/stop_ratio until they have >=2 history points
        (a single observation has no velocity). This is intentional, not a bug:
        the first history_len-ish frames after any reset() will show degenerate
        motion features for freshly-seen tracks.
        """
        self._update_track_history(detections)
        zones = self._assign_to_zones(detections)

        features = np.zeros(self.grid_rows * self.grid_cols * N_FEATURES_PER_ZONE, dtype=np.float32)
        frame_max_speed = 0.0
        frame_speeds: List[float] = []
        for zone_idx, zone_dets in enumerate(zones):
            speeds, directions = self._zone_motion(zone_dets)
            base = zone_idx * N_FEATURES_PER_ZONE
            features[base + 0] = self._normalized_count(len(zone_dets))
            features[base + 1] = self._flow_magnitude(speeds)
            features[base + 2] = self._direction_entropy(directions)
            features[base + 3] = self._stop_ratio(speeds)
            if speeds:
                frame_max_speed = max(frame_max_speed, float(np.mean(speeds)))
                frame_speeds.extend(speeds)

        # Update BOTH running stats after dividing, not before, for the same
        # reason: a same-frame spike (panic onset, or a single mistracked ID
        # teleporting across the frame) must not be able to inflate its own
        # normalizing reference before anything gets measured against it.
        self.running_max_speed = max(self.running_max_speed, frame_max_speed)
        if frame_speeds:
            self.speed_history.extend(frame_speeds)
            self.running_median_speed = float(np.median(self.speed_history))
        return features

    def _update_track_history(self, detections: List[Dict[str, Any]]) -> None:
        for det in detections:
            track_id = det["track_id"]
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.history_len)
            self.track_history[track_id].append((det["cx"], det["cy"]))

    def _assign_to_zones(self, detections: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        zones: List[List[Dict[str, Any]]] = [[] for _ in range(self.grid_rows * self.grid_cols)]
        zone_w = self.frame_width / self.grid_cols
        zone_h = self.frame_height / self.grid_rows
        for det in detections:
            col = min(max(int(det["cx"] // zone_w), 0), self.grid_cols - 1)
            row = min(max(int(det["cy"] // zone_h), 0), self.grid_rows - 1)
            zones[row * self.grid_cols + col].append(det)
        return zones

    def _zone_motion(self, zone_dets: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """Return (speeds, angles) for tracks in this zone with >=2 history points."""
        speeds: List[float] = []
        angles: List[float] = []
        for det in zone_dets:
            hist = self.track_history[det["track_id"]]
            if len(hist) < 2:
                continue
            first = np.array(hist[0], dtype=np.float64)
            last = np.array(hist[-1], dtype=np.float64)
            dx, dy = last[0] - first[0], last[1] - first[1]
            dist = float(np.hypot(dx, dy))
            speeds.append(dist / (len(hist) - 1))
            if dist >= 1.0:
                angles.append(float(np.arctan2(dy, dx)))
        return speeds, angles

    def _normalized_count(self, zone_count: int) -> float:
        self.running_max_count = max(self.running_max_count, float(zone_count))
        return float(np.clip(zone_count / self.running_max_count, 0.0, 1.0))

    def _flow_magnitude(self, speeds: List[float]) -> float:
        # Divides by the ceiling as of the END of the previous frame; extract()
        # updates running_max_speed only after all zones have been divided.
        if not speeds:
            return 0.0
        mean_speed = float(np.mean(speeds))
        return float(np.clip(mean_speed / self.running_max_speed, 0.0, 1.0))

    def _direction_entropy(self, angles: List[float]) -> float:
        if len(angles) < 2:
            return 0.0
        bin_edges = np.linspace(-np.pi, np.pi, N_DIRECTION_BINS + 1)
        counts, _ = np.histogram(angles, bins=bin_edges)
        return float(shannon_entropy(counts) / np.log(N_DIRECTION_BINS))

    def _stop_ratio(self, speeds: List[float]) -> float:
        # Compares against the median as of the END of the previous frame (see
        # the lag comment in extract()); "stopped" means moving slower than
        # stop_threshold times this clip's typical pace so far.
        if not speeds:
            return 0.0
        threshold = self.stop_threshold * self.running_median_speed
        stopped = sum(1 for s in speeds if s < threshold)
        return stopped / len(speeds)
