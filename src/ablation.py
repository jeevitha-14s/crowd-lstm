"""Feature ablation: 5 LSTM variants (full model + one-feature-zeroed for
each of density/flow/entropy/stop_ratio), each trained and evaluated on the
same 11-fold LOCO splits as train_forecast_model.py, for a like-for-like
comparison against the full model.

Prediction under test (from the lead-window sweep / Section 3-4 of
PAPER_RESULTS.md): stop_ratio and flow_magnitude should be load-bearing
(large drop when zeroed, matching their robust per-onset sign consistency
-- 11/11 and 10/11 respectively at W=60); normalized_count should be
near-decorative (small drop, given only 3/11 onsets held a consistent sign
and ~79% of its naive gap was population-collapse artifact, not signal).

"Zeroed" means literally set to 0 across all 16 zones, in both train and
test data, for every timestep in the sequence -- NOT removed from the input
tensor. Every variant keeps input_size=64; only WHICH 16 of the 64 columns
are held at constant zero differs. This isolates the effect of removing a
feature's information while holding architecture and input dimensionality
fixed across variants, so any AUROC difference is attributable to the
feature itself, not a capacity change.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from sklearn.metrics import average_precision_score, roc_auc_score
from src.train_forecast_model import HORIZONS, train_one_fold

DATASET_PATH = "data/forecast_dataset.npz"
OUT_JSON_PATH = Path("outputs/reports/umn_ablation_results.json")
FEATURE_NAMES = ["normalized_count", "flow_magnitude", "direction_entropy", "stop_ratio"]
VARIANTS = ["full"] + [f"zero_{name}" for name in FEATURE_NAMES]


def zero_feature(X: np.ndarray, feature_idx: int) -> np.ndarray:
    X = X.copy()
    X[:, :, feature_idx::4] = 0.0
    return X


def _agg(fold_subset: List[Dict], k: int) -> Dict[str, float]:
    vals = [r["horizons"][str(k)] for r in fold_subset if not r["horizons"][str(k)]["excluded"]]
    auroc_arr = np.array([v["auroc"] for v in vals])
    auprc_arr = np.array([v["auprc"] for v in vals])
    return {
        "auroc_mean": float(auroc_arr.mean()) if len(auroc_arr) else float("nan"),
        "auroc_std": float(auroc_arr.std()) if len(auroc_arr) else float("nan"),
        "auprc_mean": float(auprc_arr.mean()) if len(auprc_arr) else float("nan"),
        "auprc_std": float(auprc_arr.std()) if len(auprc_arr) else float("nan"),
        "n_folds": len(vals),
    }


def run_variant(
    variant: str,
    X: np.ndarray,
    clip_ids: np.ndarray,
    y: Dict[int, np.ndarray],
    device: torch.device,
) -> List[Dict]:
    if variant == "full":
        X_variant = X
    else:
        feat_name = variant.replace("zero_", "")
        X_variant = zero_feature(X, FEATURE_NAMES.index(feat_name))

    unique_clips = sorted(np.unique(clip_ids).tolist())
    fold_records: List[Dict] = []
    for fold_clip in unique_clips:
        train_mask = clip_ids != fold_clip
        test_mask = clip_ids == fold_clip
        X_train, X_test = X_variant[train_mask], X_variant[test_mask]
        y_train = {k: y[k][train_mask] for k in HORIZONS}
        y_test = {k: y[k][test_mask] for k in HORIZONS}
        train_clip_ids = clip_ids[train_mask]

        probs, val_clips, best_epoch, stopped_epoch = train_one_fold(
            X_train, y_train, train_clip_ids, X_test, fold_clip, device
        )
        record = {
            "fold_clip": fold_clip,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "horizons": {},
        }
        line = f"  [{variant}] fold {fold_clip}: best_epoch={best_epoch}"
        for k in HORIZONS:
            yt = y_test[k]
            p = probs[k]
            n_pos = int(yt.sum())
            excluded = n_pos == 0 or n_pos == len(yt)
            if excluded:
                auroc, auprc = float("nan"), float("nan")
            else:
                auroc = roc_auc_score(yt, p)
                auprc = average_precision_score(yt, p)
            record["horizons"][str(k)] = {"n_pos": n_pos, "excluded": excluded, "auroc": auroc, "auprc": auprc}
            line += f"  k{k}={auroc:.3f}"
        fold_records.append(record)
        print(line)
    return fold_records


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(DATASET_PATH)
    X = data["X"]
    clip_ids = data["clip_ids"]
    y = {k: data[f"y_{k}"] for k in HORIZONS}
    print(f"Device: {device}")
    print(f"Variants: {VARIANTS}")

    all_results: Dict[str, Dict] = {}
    for variant in VARIANTS:
        print(f"\n=== variant: {variant} ===")
        fold_records = run_variant(variant, X, clip_ids, y, device)
        common_fold_clips = [
            r["fold_clip"] for r in fold_records if all(not r["horizons"][str(k)]["excluded"] for k in HORIZONS)
        ]
        common_fold_records = [r for r in fold_records if r["fold_clip"] in common_fold_clips]
        aggregate_full = {str(k): _agg(fold_records, k) for k in HORIZONS}
        aggregate_common = {str(k): _agg(common_fold_records, k) for k in HORIZONS}
        all_results[variant] = {
            "folds": fold_records,
            "common_fold_subset": common_fold_clips,
            "aggregate_full_set": aggregate_full,
            "aggregate_common_subset": aggregate_common,
        }
        print("  full-set AUROC:      " + "  ".join(f"k{k}={aggregate_full[str(k)]['auroc_mean']:.4f}" for k in HORIZONS))
        print("  common-subset AUROC: " + "  ".join(f"k{k}={aggregate_common[str(k)]['auroc_mean']:.4f}" for k in HORIZONS))

    print("\n=== Drop vs full model (common-subset AUROC mean) ===")
    full_agg = all_results["full"]["aggregate_common_subset"]
    header = f"{'variant':28s}" + "".join(f"{'k' + str(k):>16}" for k in HORIZONS)
    print(header)
    for variant in VARIANTS:
        agg = all_results[variant]["aggregate_common_subset"]
        row = f"{variant:28s}"
        for k in HORIZONS:
            v = agg[str(k)]["auroc_mean"]
            if variant == "full":
                row += f"{v:>16.4f}"
            else:
                drop = full_agg[str(k)]["auroc_mean"] - v
                row += f"{v:>9.4f}({drop:+.4f})"
        print(row)

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
