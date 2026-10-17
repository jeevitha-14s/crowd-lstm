"""Macroscopic (optical-flow) counterpart to process_umn.py, for a controlled
test of the paper's central representational claim.

Every baseline in the ladder (rule, logistic regression, random forest,
gradient boosting, zone attention) consumes the SAME tracked features as the
proposed model, so they vary the classifier and hold the representation fixed.
None of them tests the claim the paper actually rests on: that tracking
individual pedestrians beats treating the crowd as an undifferentiated field,
which is what [2] and [5] do. The DeepROD comparison does change the
representation, but simultaneously changes the architecture, the supervision
and the training protocol, so it cannot isolate the cause.

This module produces a 64-dim descriptor identical in shape, grid and
semantics to FeatureExtractor's, computed from dense Farneback optical flow
instead of from tracks. Feeding it to the same LSTM under the same
leave-one-clip-out folds isolates the representation as the single
independent variable.

Each 4x4 zone yields four features chosen to mirror the tracked ones concept
for concept, so the comparison is between *what a unit of motion is* (a
tracked person vs a pixel) rather than between different quantities:

  motion_density    <- normalized_count    fraction of zone pixels in motion
  flow_magnitude    <- flow_magnitude      mean magnitude of moving pixels
  direction_entropy <- direction_entropy   entropy over 8 direction bins
  stop_ratio        <- stop_ratio          moving pixels below half the median

"In motion" is defined relative to the running median magnitude over the whole
frame rather than by an absolute cutoff. Farneback assigns small non-zero flow
to static background, and that noise floor varies with scene texture and
camera distance; the frame median tracks it because background pixels
outnumber moving ones. This mirrors FeatureExtractor's own choice to define
"stopped" relative to a running median rather than in absolute px/frame.

Two properties are preserved exactly from the tracked pipeline, because
breaking either would make the comparison unfair rather than merely different:

  1. CAUSAL NORMALISATION. Running maxima and medians are updated only AFTER
     the current frame has been normalised against them, so a spike at panic
     onset cannot inflate its own reference. FeatureExtractor does the same.
  2. PER-CLIP RESET. All running statistics reset at clip boundaries, which
     are read from the same meta CSV process_umn.py produced, so both
     representations see identical clip segmentation.

Farneback runs with the conventional parameter set (pyr_scale 0.5, 3 levels,
winsize 15, 3 iterations, poly_n 5, poly_sigma 1.2) and is not tuned. The
tracked pipeline's detector was fine-tuned on CrowdHuman, so any tuning here
would need a comparable budget to be fair; leaving the standard values is the
honest option and is stated as a limitation rather than hidden.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

VIDEO_PATH = Path("data/raw/umn/Crowd-Activity-All.avi")
META_IN = Path("data/features/umn_meta.csv")          # reuse the tracked run's clip boundaries
FEATURES_OUT = Path("data/features/umn_flow_features.npy")
META_OUT = Path("data/features/umn_flow_meta.csv")

GRID_ROWS = GRID_COLS = 4
N_FEATURES_PER_ZONE = 4
N_DIRECTION_BINS = 8
STOP_THRESHOLD = 0.5          # same constant as FeatureExtractor
FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)


class FlowFeatureExtractor:
    """Mirror of FeatureExtractor over a dense flow field instead of tracks."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w, self.h = frame_width, frame_height
        self.reset()

    def reset(self) -> None:
        self.running_max_density = 1e-6
        self.running_max_speed = 1e-6
        self.magnitude_history: List[float] = []
        self.running_median_mag = 1e-6      # frame-level noise floor
        self.running_median_moving = 1e-6   # median over moving pixels only

    def _zone_slices(self) -> List[Tuple[slice, slice]]:
        zh, zw = self.h / GRID_ROWS, self.w / GRID_COLS
        out = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                out.append((slice(int(r * zh), int((r + 1) * zh)),
                            slice(int(c * zw), int((c + 1) * zw))))
        return out

    def extract(self, flow: np.ndarray) -> np.ndarray:
        """flow: (H, W, 2). Returns the 64-dim descriptor for this frame."""
        u, v = flow[..., 0], flow[..., 1]
        mag = np.sqrt(u * u + v * v)
        ang = np.arctan2(v, u)

        # Thresholds come from the END of the previous frame -- never this one.
        noise_floor = self.running_median_mag
        stop_cut = STOP_THRESHOLD * self.running_median_moving

        feats = np.zeros(GRID_ROWS * GRID_COLS * N_FEATURES_PER_ZONE, dtype=np.float32)
        frame_max_speed = 0.0
        frame_max_density = 0.0
        moving_mags_all: List[np.ndarray] = []

        for zi, (rs, cs) in enumerate(self._zone_slices()):
            zm, za = mag[rs, cs], ang[rs, cs]
            moving = zm > noise_floor
            n_moving = int(moving.sum())
            base = zi * N_FEATURES_PER_ZONE

            density = n_moving / zm.size if zm.size else 0.0
            feats[base + 0] = float(np.clip(density / self.running_max_density, 0.0, 1.0))
            frame_max_density = max(frame_max_density, density)

            if n_moving:
                mm = zm[moving]
                mean_speed = float(mm.mean())
                feats[base + 1] = float(np.clip(mean_speed / self.running_max_speed, 0.0, 1.0))
                frame_max_speed = max(frame_max_speed, mean_speed)
                moving_mags_all.append(mm)

                if n_moving >= 2:
                    # magnitude-weighted so faster pixels count more, matching the
                    # tracked version's use of per-track (not per-pixel) directions
                    hist, _ = np.histogram(za[moving], bins=N_DIRECTION_BINS,
                                           range=(-np.pi, np.pi), weights=mm)
                    tot = hist.sum()
                    if tot > 0:
                        p = hist / tot
                        p = p[p > 0]
                        feats[base + 2] = float(-(p * np.log(p)).sum() / np.log(N_DIRECTION_BINS))

                feats[base + 3] = float((mm < stop_cut).sum() / n_moving)

        # Causal update: only now, after every zone has been divided.
        self.running_max_density = max(self.running_max_density, frame_max_density)
        self.running_max_speed = max(self.running_max_speed, frame_max_speed)
        self.magnitude_history.append(float(np.median(mag)))
        self.running_median_mag = float(np.median(self.magnitude_history))
        if moving_mags_all:
            self.running_median_moving = float(np.median(np.concatenate(moving_mags_all)))
        return feats


def load_clip_ids() -> np.ndarray:
    with open(META_IN, newline="") as f:
        return np.array([int(r["clip_id"]) for r in csv.DictReader(f)], dtype=np.int64)


def main() -> None:
    clip_ids = load_clip_ids()
    boundaries = {i for i in range(1, len(clip_ids)) if clip_ids[i] != clip_ids[i - 1]}
    print(f"{len(set(clip_ids.tolist()))} clips, {len(boundaries)} boundaries "
          f"(read from {META_IN}, identical to the tracked run)")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert n == len(clip_ids), (n, len(clip_ids))

    fe = FlowFeatureExtractor(w, h)
    feats = np.zeros((n, GRID_ROWS * GRID_COLS * N_FEATURES_PER_ZONE), dtype=np.float32)
    rows: List[Tuple[int, int]] = []

    prev_gray = None
    idx = 0
    with tqdm(total=n, desc="dense flow + features") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if idx in boundaries:
                fe.reset()
                prev_gray = None       # no flow across a clip boundary
            if prev_gray is None:
                flow = np.zeros((h, w, 2), dtype=np.float32)
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FARNEBACK)
            feats[idx] = fe.extract(flow)
            rows.append((idx, int(clip_ids[idx])))
            prev_gray = gray
            idx += 1
            pbar.update(1)
    cap.release()

    feats = feats[:idx]
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(FEATURES_OUT, feats)
    with open(META_OUT, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "clip_id"])
        wr.writerows(rows)
    print(f"\nSaved {FEATURES_OUT} {feats.shape}")
    print(f"Saved {META_OUT}")
    nz = (feats != 0).mean()
    print(f"non-zero fraction: {nz:.3f}  (tracked features for comparison: "
          f"{(np.load('data/features/umn_features.npy') != 0).mean():.3f})")


if __name__ == "__main__":
    main()
