"""Logistic regression control: a LEARNED but non-recurrent baseline, trained
on the same 30-timestep x 64-dim window the LSTM sees (flattened to 1920
dims), on the same 11-fold LOCO splits and 4 horizons as
train_forecast_model.py.

baseline_rule.py already isolates "does any learning help at all" (a
no-learning current-frame rule vs. the LSTM). This isolates a narrower
question: given a model that IS learned and DOES see the full window (same
information access as the LSTM), does recurrence specifically add value, or
would a simple linear classifier over the flattened window do just as well?
The two baselines together form a ladder: no-learning current-frame rule <
learned non-recurrent full-window logistic regression < learned recurrent
LSTM. If logistic regression closes most of the gap at k=0 but falls
further behind as horizon grows, that isolates recurrence -- not just
learning -- as what buys the forecasting range.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "data/forecast_dataset.npz"
OUT_JSON_PATH = Path("outputs/reports/umn_logreg_control_results.json")
HORIZONS = [0, 30, 60, 90]
SEED = 42
MAX_ITER = 2000


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


def fit_and_score(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced", random_state=SEED)
    clf.fit(X_train_s, y_train)
    return clf.predict_proba(X_test_s)[:, 1]


def main() -> None:
    data = np.load(DATASET_PATH)
    X = data["X"]  # (N, seq_len=30, 64)
    clip_ids = data["clip_ids"]
    y = {k: data[f"y_{k}"] for k in HORIZONS}
    n = X.shape[0]
    X_flat = X.reshape(n, -1)  # (N, 1920) -- same info as the LSTM, order flattened

    unique_clips = sorted(np.unique(clip_ids).tolist())
    fold_records: List[Dict] = []

    print("=== Logistic regression control: learned, non-recurrent, full-window ===")
    header = f"{'fold(clip)':>10}" + "".join(f"{'AUROC_k' + str(k):>12}{'AUPRC_k' + str(k):>12}" for k in HORIZONS)
    print(header)

    for fold_clip in unique_clips:
        train_mask = clip_ids != fold_clip
        test_mask = clip_ids == fold_clip
        X_train, X_test = X_flat[train_mask], X_flat[test_mask]
        y_train = {k: y[k][train_mask] for k in HORIZONS}
        y_test = {k: y[k][test_mask] for k in HORIZONS}

        record = {"fold_clip": fold_clip, "horizons": {}}
        row = f"{fold_clip:>10}"
        for k in HORIZONS:
            yt_train, yt_test = y_train[k], y_test[k]
            n_pos_test = int(yt_test.sum())
            excluded = n_pos_test == 0 or n_pos_test == len(yt_test)
            train_degenerate = yt_train.sum() == 0 or yt_train.sum() == len(yt_train)
            if excluded or train_degenerate:
                auroc, auprc = float("nan"), float("nan")
                excluded = True  # can't fit or can't score either way
            else:
                probs = fit_and_score(X_train, yt_train, X_test)
                auroc = roc_auc_score(yt_test, probs)
                auprc = average_precision_score(yt_test, probs)
            record["horizons"][str(k)] = {
                "n_test": int(len(yt_test)),
                "n_pos": n_pos_test,
                "excluded": excluded,
                "auroc": auroc,
                "auprc": auprc,
            }
            row += f"{auroc:>12.4f}{auprc:>12.4f}"
        fold_records.append(record)
        print(row)

    aggregate_full = {str(k): _agg(fold_records, k) for k in HORIZONS}
    common_fold_clips = [
        r["fold_clip"] for r in fold_records if all(not r["horizons"][str(k)]["excluded"] for k in HORIZONS)
    ]
    common_fold_records = [r for r in fold_records if r["fold_clip"] in common_fold_clips]
    aggregate_common = {str(k): _agg(common_fold_records, k) for k in HORIZONS}

    print("\n=== Logistic regression control summary: FULL SET ===")
    for k in HORIZONS:
        a = aggregate_full[str(k)]
        print(
            f"horizon {k:>3}: AUROC = {a['auroc_mean']:.4f} +/- {a['auroc_std']:.4f}   "
            f"AUPRC = {a['auprc_mean']:.4f} +/- {a['auprc_std']:.4f}   (n_folds={a['n_folds']}/{len(unique_clips)})"
        )

    print(f"\n=== Logistic regression control summary: COMMON FOLD SUBSET {common_fold_clips} ===")
    for k in HORIZONS:
        a = aggregate_common[str(k)]
        print(
            f"horizon {k:>3}: AUROC = {a['auroc_mean']:.4f} +/- {a['auroc_std']:.4f}   "
            f"AUPRC = {a['auprc_mean']:.4f} +/- {a['auprc_std']:.4f}   (n_folds={a['n_folds']}/{len(common_fold_clips)})"
        )

    payload = {
        "description": (
            "learned, non-recurrent control: logistic regression (StandardScaler "
            "+ class_weight='balanced') on the flattened 30x64 window, identical "
            "information access to the LSTM, differing only in recurrence"
        ),
        "folds": fold_records,
        "common_fold_subset": common_fold_clips,
        "aggregate_full_set": aggregate_full,
        "aggregate_common_subset": aggregate_common,
    }
    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
