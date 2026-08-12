"""Build fixed-length history windows with multi-horizon abnormal labels from
per-frame ShanghaiTech features, for LSTM-based crowd risk *forecasting*.

Mirrors build_forecast_dataset.py (UMN), but each video is an independent
unit -- no clip stitching, no history/horizon window ever crosses a video
boundary. Parameters are matched to UMN in SECONDS, not frames, since
ShanghaiTech is 24fps vs UMN's 30fps.

Also stratifies every sample by where its history window sits relative to
the video's anomalous segment (window_class), because ShanghaiTech's flat
positive rate across horizons -- unlike UMN's rising one -- means a model
could reach a high AUROC by recognizing "this video is near an event"
rather than genuinely forecasting one. See CLEAN_PRE below.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

FEATURES_DIR = Path("data/features/shanghaitech")
META_CSV_PATH = Path("data/features/shanghaitech_meta.csv")
MANIFEST_PATH = Path("data/shanghaitech_meta.json")
OUT_PATH = Path("data/shanghaitech_forecast.npz")

SEQ_LEN = 30
STRIDE = 3
HORIZONS = [0, 24, 48, 72]
MIN_POSITIVE_RATE_WARN = 0.02
CLEAN_PRE_WARN_THRESHOLD = 200

NEGATIVE, CLEAN_PRE, OVERLAP, POST = 0, 1, 2, 3
WINDOW_CLASS_NAMES = {NEGATIVE: "NEGATIVE", CLEAN_PRE: "CLEAN_PRE", OVERLAP: "OVERLAP", POST: "POST"}
WINDOW_CLASS_README = (
    "window_class[i, j] classifies sample i's history window relative to its "
    "video's anomalous segment, for horizon HORIZONS[j] (columns match "
    "window_class_horizons). Window = frames [t-87, t] (the sample's full "
    "observed history, i.e. history_span=(SEQ_LEN-1)*STRIDE frames ending at "
    "the current frame t). Encoding: "
    "0=NEGATIVE (no anomaly in video, or window+target both clear of it), "
    "1=CLEAN_PRE (window entirely before onset AND target frame t+k falls "
    "inside the event -- genuine forecasting: model sees only normal footage "
    "and must predict an event it has not observed), "
    "2=OVERLAP (window intersects the anomalous segment -- model can already "
    "see the anomaly, so a positive label here is partly detection, not "
    "forecasting), "
    "3=POST (window entirely after the event ended -- always a negative "
    "target, included so training/eval can separate 'recently saw an event' "
    "proximity effects from true negatives)."
)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def load_scene_map(path: Path) -> Dict[str, str]:
    scene_map = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            scene_map[row["vid"]] = row["scene"]
    return scene_map


def build_labels(n_frames: int, anom: int, start: int, end: int) -> np.ndarray:
    labels = np.zeros(n_frames, dtype=np.int64)
    if anom:
        labels[start:end] = 1
    return labels


def window_position(window_start: int, window_end: int, start: int, end: int) -> int:
    """Classify a window's geometric relation to event [start, end), ignoring
    the target frame -- CLEAN_PRE vs NEGATIVE for a pre-onset window is
    resolved later, per horizon, since it depends on the target frame."""
    overlaps = window_start < end and window_end >= start
    if overlaps:
        return OVERLAP
    if window_end < start:
        return CLEAN_PRE  # "pre" placeholder; refined per-horizon below
    if window_start >= end:
        return POST
    raise AssertionError("window neither overlaps, precedes, nor follows the event")


def build_dataset() -> None:
    manifest = load_manifest(MANIFEST_PATH)
    scene_map = load_scene_map(META_CSV_PATH)

    history_span = (SEQ_LEN - 1) * STRIDE
    max_horizon = max(HORIZONS)

    X_list = []
    y_lists: Dict[int, List[int]] = {k: [] for k in HORIZONS}
    window_class_lists: Dict[int, List[int]] = {k: [] for k in HORIZONS}
    frame_ids: List[int] = []
    video_ids: List[str] = []
    scene_ids: List[str] = []

    videos_per_scene: Dict[str, int] = {}
    total_candidates = 0
    total_valid = 0

    for entry in manifest:
        vid = entry["vid"]
        scene = scene_map[vid]
        videos_per_scene[scene] = videos_per_scene.get(scene, 0) + 1

        features = np.load(FEATURES_DIR / f"{vid}.npy")
        n_frames = features.shape[0]
        assert n_frames == entry["n"], (
            f"{vid}: feature array has {n_frames} frames, manifest says {entry['n']}"
        )
        anom, start, end = entry["anom"], entry["start"], entry["end"]
        labels = build_labels(n_frames, anom, start, end)

        total_candidates += n_frames
        for t in range(n_frames):
            if t - history_span < 0 or t + max_horizon >= n_frames:
                continue
            total_valid += 1
            X_list.append(features[t - history_span : t + 1 : STRIDE])

            position: Optional[int] = (
                window_position(t - history_span, t, start, end) if anom else None
            )
            for k in HORIZONS:
                y_val = int(labels[t + k])
                y_lists[k].append(y_val)
                if position is None:
                    wc = NEGATIVE
                elif position == CLEAN_PRE:
                    wc = CLEAN_PRE if y_val == 1 else NEGATIVE
                else:
                    wc = position  # OVERLAP or POST, horizon-independent
                window_class_lists[k].append(wc)

            frame_ids.append(t)
            video_ids.append(vid)
            scene_ids.append(scene)

    X_arr = np.stack(X_list).astype(np.float32)
    frame_ids_arr = np.array(frame_ids, dtype=np.int64)
    video_ids_arr = np.array(video_ids)
    scene_ids_arr = np.array(scene_ids)
    window_class_arr = np.stack(
        [np.array(window_class_lists[k], dtype=np.int64) for k in HORIZONS], axis=1
    )

    save_kwargs = {
        "X": X_arr,
        "frame_ids": frame_ids_arr,
        "video_ids": video_ids_arr,
        "scene_ids": scene_ids_arr,
        "window_class": window_class_arr,
        "window_class_horizons": np.array(HORIZONS, dtype=np.int64),
        "window_class_readme": np.array(WINDOW_CLASS_README),
    }
    for k in HORIZONS:
        save_kwargs[f"y_{k}"] = np.array(y_lists[k], dtype=np.int64)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **save_kwargs)

    n_samples = X_arr.shape[0]
    dropped = total_candidates - total_valid

    print("=== ShanghaiTech forecast dataset summary ===")
    print(f"X shape: {X_arr.shape}")
    print(f"Videos: {len(manifest)}  Total samples: {n_samples}")
    print(
        f"Frame positions dropped for insufficient history or horizon overrun: "
        f"{dropped} (of {total_candidates} total frame positions across all videos)"
    )

    print("\n--- Per horizon ---")
    any_low = False
    for k in HORIZONS:
        y = save_kwargs[f"y_{k}"]
        pos = int(y.sum())
        rate = pos / len(y) if len(y) else 0.0
        flag = ""
        if rate < MIN_POSITIVE_RATE_WARN:
            flag = "  <<< WARNING: under 2% positives, severe class imbalance"
            any_low = True
        print(f"horizon {k:>3}: total={len(y)} positive={pos} rate={rate:.4f}{flag}")

    print("\n--- Per scene ---")
    header = f"{'scene':>6} {'videos':>7} {'samples':>8}" + "".join(
        f"  pos@{k:<3}" for k in HORIZONS
    )
    print(header)
    for scene in sorted(videos_per_scene):
        mask = scene_ids_arr == scene
        n_scene_samples = int(mask.sum())
        row = f"{scene:>6} {videos_per_scene[scene]:>7} {n_scene_samples:>8}"
        for k in HORIZONS:
            pos_k = int(save_kwargs[f"y_{k}"][mask].sum())
            row += f"  {pos_k:>6}"
        print(row)

    if any_low:
        print(
            "\nWARNING: at least one horizon has under 2% positives. "
            "Consider adjusting sampling/weighting before training."
        )

    print("\n--- Window-position stratification (per horizon) ---")
    for j, k in enumerate(HORIZONS):
        wc = window_class_arr[:, j]
        y = save_kwargs[f"y_{k}"]
        print(f"horizon {k}:")
        for cls_id in (NEGATIVE, CLEAN_PRE, OVERLAP, POST):
            mask = wc == cls_id
            total = int(mask.sum())
            pos = int(y[mask].sum())
            print(f"    {WINDOW_CLASS_NAMES[cls_id]:<9} total={total:>7}  positive={pos:>6}")

    print("\n--- CLEAN_PRE positives per horizon (forecasting-defensibility check) ---")
    any_thin = False
    for j, k in enumerate(HORIZONS):
        wc = window_class_arr[:, j]
        clean_pre_pos = int((wc == CLEAN_PRE).sum())
        flag = ""
        if clean_pre_pos < CLEAN_PRE_WARN_THRESHOLD:
            flag = (
                f"  <<< WARNING: under {CLEAN_PRE_WARN_THRESHOLD} CLEAN_PRE positives; "
                "this horizon may not support a defensible forecasting claim"
            )
            any_thin = True
        print(f"horizon {k:>3}: CLEAN_PRE positives = {clean_pre_pos}{flag}")

    if any_thin:
        print(
            "\nWARNING: at least one horizon has too few CLEAN_PRE positives to "
            "reliably measure true forecasting performance on this dataset."
        )


if __name__ == "__main__":
    build_dataset()
