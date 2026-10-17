"""DeepROD reimplementation evaluated on MED, under this project's
leave-one-video-out protocol.

Companion to src/deeprod_gridsearch.py, which runs the same comparison on UMN.
MED matters here for a specific reason: it is one of the five corpora DeepROD
itself evaluates (their Table 1), and they publish per-video results for it
(their Table 3, average AUC 0.917 under their per-video self-calibration
protocol). Running their method on MED under our protocol therefore gives a
*second* matched comparison against the base paper, on a dataset they chose,
rather than resting the entire comparison on UMN.

Everything downstream of feature extraction is imported from
deeprod_gridsearch: the same 504-configuration grid drawn from their Sect. 4.2,
the same 80/20 validation split, the same minimum-validation-MSE selection
rule, the same Box-Cox/Mahalanobis detection stage. Only the corpus differs.
Keeping one implementation avoids the two runs drifting apart, which would make
the UMN and MED rows incomparable.

MED-specific handling:
  - Each of the 11 panic videos is its own file and its own fold; GoM is
    computed per video and never spans a file boundary.
  - Frames MED leaves unlabelled (behaviour label 0, always a trailing run) are
    dropped rather than treated as normal, matching build_med_forecast.py. A
    GoM value is kept only when both frames of its pair are labelled.
  - MED is 854x480 against UMN's 320x240, so GoM magnitudes differ by roughly
    an order of magnitude between corpora. This is harmless because the
    standardisation in deeprod_gridsearch is fit per fold, but it is the reason
    the two datasets' raw GoM values should never be pooled.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd

from src.deeprod_gridsearch import (
    D_CHOICES,
    EPOCH_CHECKPOINTS,
    architecture_grid,
    compute_gom_sequence,
    run_loco,
)

MED_DIR = Path("(Copy) Motion_Emotion_Dataset")
META_PATH = Path("data/features/med_meta.csv")
GOM_CACHE = Path("data/features/med_gom.npz")
OUT_PATH = Path("outputs/reports/deeprod_med_loco.json")

# DeepROD's own reported MED figures, from their Table 3 (per-video online
# self-calibration). Recorded for context only -- not comparable to the LOCO
# row produced here, for the same protocol reasons set out in the paper.
DEEPROD_REPORTED_MED = {"auc": 0.917, "accuracy": 0.956, "pf": 0.04,
                        "precision": 0.76, "recall": 0.87}


def build_med_gom_cache() -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """{video_id: (gom_sequence, gom_labels)} over labelled frames only."""
    if GOM_CACHE.exists():
        blob = np.load(GOM_CACHE)
        vids = sorted({int(k.split("_")[1]) for k in blob.files})
        print(f"Loaded cached MED GoM for {len(vids)} videos from {GOM_CACHE}")
        return {v: (blob[f"gom_{v}"], blob[f"lab_{v}"]) for v in vids}

    meta = pd.read_csv(META_PATH)
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for vid, g in meta.groupby("video_id"):
        path = MED_DIR / f"{int(vid) + 1:03d}.mp4"
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()

        g = g.sort_values("local_frame")
        n = min(len(frames), len(g))
        frames = frames[:n]
        abnormal = g["abnormal"].to_numpy()[:n]
        valid = g["valid"].to_numpy()[:n]

        print(f"video {int(vid) + 1:03d}: GoM over {n} frames...", flush=True)
        gom = compute_gom_sequence(frames)  # length n-1; gom[i] spans frames i, i+1
        gom_labels = abnormal[1:]
        # Keep a GoM value only if both frames of its pair were annotated.
        gom_valid = (valid[:-1] & valid[1:]).astype(bool)
        out[int(vid)] = (gom[gom_valid], gom_labels[gom_valid])
        print(f"  kept {int(gom_valid.sum())}/{len(gom)} GoM values "
              f"({int(gom_labels[gom_valid].sum())} abnormal)")

    GOM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOM_CACHE,
                        **{f"gom_{v}": d[0] for v, d in out.items()},
                        **{f"lab_{v}": d[1] for v, d in out.items()})
    print(f"Cached MED GoM to {GOM_CACHE}")
    return out


def main() -> None:
    clip_data = build_med_gom_cache()
    n_cfg = len(architecture_grid()) * len(D_CHOICES) * len(EPOCH_CHECKPOINTS)
    print(f"\n{len(clip_data)} folds, {n_cfg} configurations each "
          f"(same grid as the UMN run)\n")

    result = run_loco(clip_data)

    payload = {
        "description": (
            "DeepROD (Ammar & Cherif 2021) reimplemented with their full published "
            "hyperparameter grid and evaluated on MED under leave-one-video-out. Second "
            "matched base-paper comparison, on a corpus DeepROD itself evaluates. "
            "Shares all modelling code with src/deeprod_gridsearch.py so the UMN and MED "
            "rows are measured identically."
        ),
        "corpus": "MED (11 panic videos)",
        "deeprod_reported_on_med_their_protocol": DEEPROD_REPORTED_MED,
        "protocol_note": (
            "DeepROD's reported MED figures use per-video online self-calibration "
            "(training on each test video's own opening 400-1900 frames, their Table 4). "
            "The row produced here holds each video out entirely and is not comparable "
            "to that figure."
        ),
        **result,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    print("\n=== DeepROD on MED, leave-one-video-out ===")
    for k, v in result["average_selected_by_val_mse"].items():
        print(f"  {k}: {v:.4f}")
    print(f"\n  (DeepROD report 0.917 AUC on MED under their own per-video protocol)")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
