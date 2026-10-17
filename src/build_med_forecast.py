"""Build multi-horizon forecast windows from MED per-frame features.

Mirrors build_forecast_dataset.py (same SEQ_LEN=30, STRIDE=5, 145-frame history
span, HORIZONS=[0,30,60,90]) so the resulting tensors are drop-in compatible
with train_forecast_model.py and with the deployed UMN checkpoint. Windows
never cross a video boundary, for the same reason UMN windows never cross a
clip boundary: track IDs and the running normalisers are scene-scoped.

One constraint MED adds that UMN does not. MED leaves trailing frames
unlabelled (behaviour label 0) rather than labelling them Neutral, so a window
whose *target* frame t+k is unlabelled has no ground truth at that horizon.
Such windows are dropped, and because a window must be usable at every horizon
to enter the tensor, the criterion is that t+k is labelled for all k. Treating
unlabelled frames as normal would silently invent negatives -- typically 39-384
frames per video, and always at the end, i.e. immediately after the panic
segment where the distinction matters most.

Note on what this enables. UMN's four horizons are not all evaluable on every
clip: four clips contain no positive k=0 sample because their anomaly begins
too close to the start for a post-warmup window to fall inside it. Every MED
panic video has its onset at frame 518 or later, well past the 145-frame
warmup, so all 11 are expected to yield positives at every horizon.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

FEATURES_PATH = Path("data/features/med_features.npy")
META_PATH = Path("data/features/med_meta.csv")
OUT_PATH = Path("data/med_forecast_dataset.npz")

SEQ_LEN = 30
STRIDE = 5
HORIZONS = [0, 30, 60, 90]


def load_meta() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(video_id, abnormal, valid) per global frame index."""
    vid, ab, va = [], [], []
    with open(META_PATH, newline="") as f:
        for row in csv.DictReader(f):
            vid.append(int(row["video_id"]))
            ab.append(int(row["abnormal"]))
            va.append(int(row["valid"]))
    return (np.array(vid, dtype=np.int64),
            np.array(ab, dtype=np.int64),
            np.array(va, dtype=np.int64))


def video_ranges(video_ids: np.ndarray) -> Dict[int, Tuple[int, int]]:
    out = {}
    for v in np.unique(video_ids):
        idx = np.where(video_ids == v)[0]
        out[int(v)] = (int(idx.min()), int(idx.max()))
    return out


def main() -> None:
    features = np.load(FEATURES_PATH)
    video_ids, abnormal, valid = load_meta()
    n = features.shape[0]
    assert video_ids.shape[0] == n == abnormal.shape[0] == valid.shape[0], (
        features.shape, video_ids.shape, abnormal.shape, valid.shape)

    ranges = video_ranges(video_ids)
    span = (SEQ_LEN - 1) * STRIDE
    max_h = max(HORIZONS)

    X_list: List[np.ndarray] = []
    y_lists: Dict[int, List[int]] = {k: [] for k in HORIZONS}
    frame_ids: List[int] = []
    vid_out: List[int] = []

    candidates = dropped_boundary = dropped_unlabelled = 0
    for t in range(n):
        if t - span < 0 or t + max_h >= n:
            continue
        candidates += 1
        v = int(video_ids[t])
        v_start, v_end = ranges[v]
        if t - span < v_start or t + max_h > v_end:
            dropped_boundary += 1
            continue
        if not all(valid[t + k] for k in HORIZONS):
            dropped_unlabelled += 1
            continue

        X_list.append(features[t - span : t + 1 : STRIDE])
        for k in HORIZONS:
            y_lists[k].append(int(abnormal[t + k]))
        frame_ids.append(t)
        vid_out.append(v)

    X = np.stack(X_list).astype(np.float32)
    save = {
        "X": X,
        "frame_ids": np.array(frame_ids, dtype=np.int64),
        # named clip_ids for drop-in compatibility with the UMN LOCO scripts
        "clip_ids": np.array(vid_out, dtype=np.int64),
    }
    for k in HORIZONS:
        save[f"y_{k}"] = np.array(y_lists[k], dtype=np.int64)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **save)

    print("=== MED forecast dataset ===")
    print(f"X shape: {X.shape}   videos: {sorted(set(vid_out))}")
    print(f"candidates {candidates}  dropped(boundary) {dropped_boundary}  "
          f"dropped(unlabelled target) {dropped_unlabelled}  kept {len(X_list)}")
    print()
    for k in HORIZONS:
        y = save[f"y_{k}"]
        print(f"horizon {k:>3}: total={len(y)} positive={int(y.sum())} "
              f"rate={y.mean():.4f}")
    print("\nper-video positives (all four horizons must be non-zero for a "
          "video to be evaluable as a LOCO fold):")
    cid = save["clip_ids"]
    print(f"{'video':>6}" + "".join(f"{'k=' + str(k):>8}" for k in HORIZONS) + "  evaluable")
    n_all = 0
    for v in sorted(set(vid_out)):
        m = cid == v
        counts = [int(save[f"y_{k}"][m].sum()) for k in HORIZONS]
        ok = all(c > 0 for c in counts)
        n_all += ok
        print(f"{v + 1:>6}" + "".join(f"{c:>8}" for c in counts) + f"  {'yes' if ok else 'NO'}")
    print(f"\nvideos evaluable at all four horizons: {n_all}/{len(set(vid_out))}")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
