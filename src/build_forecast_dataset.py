"""Build fixed-length history windows with multi-horizon abnormal labels from
per-frame UMN features, for LSTM-based crowd risk *forecasting*.
"""

import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

FEATURES_PATH = Path("data/features/umn_features.npy")
META_PATH = Path("data/features/umn_meta.csv")
GROUNDTRUTH_PATH = Path("data/umn_groundtruth.csv")
OUT_PATH = Path("data/forecast_dataset.npz")

SEQ_LEN = 30
STRIDE = 5
HORIZONS = [0, 30, 60, 90]
MIN_POSITIVE_RATE_WARN = 0.05


def load_clip_ids(meta_path: Path) -> np.ndarray:
    clip_ids = []
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            clip_ids.append(int(row["clip_id"]))
    return np.array(clip_ids, dtype=np.int64)


def load_labels(groundtruth_path: Path) -> np.ndarray:
    labels = []
    with open(groundtruth_path, newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["abnormal"]))
    return np.array(labels, dtype=np.int64)


def compute_clip_ranges(clip_ids: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """Return {clip_id: (start_frame, end_frame_inclusive)}."""
    ranges = {}
    for cid in np.unique(clip_ids):
        idx = np.where(clip_ids == cid)[0]
        ranges[int(cid)] = (int(idx.min()), int(idx.max()))
    return ranges


def build_dataset() -> None:
    features = np.load(FEATURES_PATH)
    clip_ids_per_frame = load_clip_ids(META_PATH)
    labels = load_labels(GROUNDTRUTH_PATH)
    total_frames = features.shape[0]
    assert clip_ids_per_frame.shape[0] == total_frames
    assert labels.shape[0] == total_frames

    ranges = compute_clip_ranges(clip_ids_per_frame)
    history_span = (SEQ_LEN - 1) * STRIDE
    max_horizon = max(HORIZONS)

    X_list = []
    y_lists = {k: [] for k in HORIZONS}
    frame_ids = []
    clip_ids_out = []

    global_candidates = 0
    for t in range(total_frames):
        # Would this t have enough history/horizon room in one continuous stream?
        if t - history_span < 0 or t + max_horizon >= total_frames:
            continue
        global_candidates += 1

        cid = int(clip_ids_per_frame[t])
        clip_start, clip_end = ranges[cid]
        if t - history_span < clip_start:
            continue  # history would reach into the previous clip
        if t + max_horizon > clip_end:
            continue  # largest horizon would reach into the next clip; horizons
            # are checked via max_horizon only since HORIZONS is monotonic, so
            # if t+max(HORIZONS) fits, every smaller horizon fits too.

        X = features[t - history_span : t + 1 : STRIDE]
        X_list.append(X)
        for k in HORIZONS:
            y_lists[k].append(labels[t + k])
        frame_ids.append(t)
        clip_ids_out.append(cid)

    X_arr = np.stack(X_list).astype(np.float32)
    frame_ids_arr = np.array(frame_ids, dtype=np.int64)
    clip_ids_arr = np.array(clip_ids_out, dtype=np.int64)

    save_kwargs = {"X": X_arr, "frame_ids": frame_ids_arr, "clip_ids": clip_ids_arr}
    for k in HORIZONS:
        save_kwargs[f"y_{k}"] = np.array(y_lists[k], dtype=np.int64)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **save_kwargs)

    n_samples = X_arr.shape[0]
    dropped_by_boundary = global_candidates - n_samples

    print("=== Forecast dataset summary ===")
    print(f"X shape: {X_arr.shape}")
    print(f"Total samples: {n_samples}")
    print(
        f"Samples dropped due to clip boundary constraints: {dropped_by_boundary} "
        f"(of {global_candidates} candidates that would qualify ignoring clip splits)"
    )
    print()
    any_low = False
    for k in HORIZONS:
        y = save_kwargs[f"y_{k}"]
        pos = int(y.sum())
        rate = pos / len(y) if len(y) else 0.0
        flag = ""
        if rate < MIN_POSITIVE_RATE_WARN:
            flag = "  <<< WARNING: under 5% positives, severe class imbalance"
            any_low = True
        print(f"horizon {k:>3}: total={len(y)} positive={pos} rate={rate:.4f}{flag}")

    if any_low:
        print(
            "\nWARNING: at least one horizon has under 5% positives. "
            "Consider adjusting sampling/weighting before training."
        )


if __name__ == "__main__":
    build_dataset()
