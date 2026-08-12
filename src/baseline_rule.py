"""No-learning rule-based baseline for UMN forecasting, evaluated on the
same 11-fold LOCO splits and 4 horizons as the LSTM (train_forecast_model.py)
for a like-for-like comparison.

THIS BASELINE HAS NO FORECASTING MECHANISM. It computes ONE risk score from
the CURRENT frame's 64-dim feature vector (the last timestep of each 30-step
window -- build_forecast_dataset.py's window always ends exactly at "now")
and evaluates that SAME score against every horizon's future label
(t+0/30/60/90). The score is horizon-agnostic by construction -- it cannot
"know" a horizon exists. That is precisely the point: it establishes what
purely reactive, non-temporal thresholding achieves, so any AUROC gap over
this baseline at k=30/60/90 is attributable to the LSTM's use of temporal
structure, not to better per-frame features.

Risk score = mean of 4 sign-corrected, z-scored feature-group means (density
/ flow / entropy / stop_ratio, each averaged over the 16 zones), combined
with EQUAL weights -- the simplest non-learned combination rule. Per-feature
sign (does this feature run higher or lower during panic) and z-score
statistics are fit ONLY on the fold's 10 training clips (sign from
correlation with the k=0 current-frame label, since that is the only label a
purely reactive rule could plausibly reference), then frozen and applied
unchanged to the held-out test clip. A classification threshold per horizon
is ALSO tuned on train-fold data only (Youden's J on the ROC curve),
reported alongside AUROC/AUPRC -- the threshold-independent ranking metrics
used for direct comparison with the LSTM's own reported numbers.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

DATASET_PATH = "data/forecast_dataset.npz"
OUT_JSON_PATH = Path("outputs/reports/umn_baseline_rule_results.json")
HORIZONS = [0, 30, 60, 90]
FEATURE_NAMES = ["normalized_count", "flow_magnitude", "direction_entropy", "stop_ratio"]


def group_means(X_last: np.ndarray) -> Dict[str, np.ndarray]:
    """X_last: (n_samples, 64), zone-major (index = zone*4 + feature). Returns
    {feature_name: (n_samples,)} averaged over the 16 zones."""
    return {name: X_last[:, i::4].mean(axis=1) for i, name in enumerate(FEATURE_NAMES)}


def fit_risk_score(train_groups: Dict[str, np.ndarray], y_train_k0: np.ndarray) -> Dict[str, Dict[str, float]]:
    params = {}
    for name, vals in train_groups.items():
        mean, std = float(vals.mean()), float(vals.std())
        std = std if std > 1e-8 else 1.0
        z = (vals - mean) / std
        corr = float(np.corrcoef(z, y_train_k0)[0, 1]) if y_train_k0.std() > 0 else 0.0
        sign = 1.0 if corr >= 0 else -1.0
        params[name] = {"mean": mean, "std": std, "sign": sign, "corr_with_k0_label": corr}
    return params


def apply_risk_score(groups: Dict[str, np.ndarray], params: Dict[str, Dict[str, float]]) -> np.ndarray:
    n = len(next(iter(groups.values())))
    score = np.zeros(n)
    for name in FEATURE_NAMES:
        p = params[name]
        score += p["sign"] * (groups[name] - p["mean"]) / p["std"]
    return score / len(FEATURE_NAMES)


def best_threshold(score_train: np.ndarray, y_train: np.ndarray) -> float:
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        return float(np.median(score_train))
    fpr, tpr, thresholds = roc_curve(y_train, score_train)
    return float(thresholds[np.argmax(tpr - fpr)])


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


def main() -> None:
    data = np.load(DATASET_PATH)
    X = data["X"]
    clip_ids = data["clip_ids"]
    y = {k: data[f"y_{k}"] for k in HORIZONS}
    X_last = X[:, -1, :]  # current-frame ("now") features -- see module docstring

    unique_clips = sorted(np.unique(clip_ids).tolist())
    fold_records: List[Dict] = []

    print("=== Rule-based baseline: no learning, current-frame risk score ===")
    header = f"{'fold(clip)':>10}" + "".join(f"{'AUROC_k' + str(k):>12}{'AUPRC_k' + str(k):>12}" for k in HORIZONS)
    print(header)

    for fold_clip in unique_clips:
        train_mask = clip_ids != fold_clip
        test_mask = clip_ids == fold_clip

        train_groups = group_means(X_last[train_mask])
        test_groups = group_means(X_last[test_mask])
        y_train = {k: y[k][train_mask] for k in HORIZONS}
        y_test = {k: y[k][test_mask] for k in HORIZONS}

        params = fit_risk_score(train_groups, y_train[0])
        score_train = apply_risk_score(train_groups, params)
        score_test = apply_risk_score(test_groups, params)

        record = {"fold_clip": fold_clip, "feature_params": params, "horizons": {}}
        row = f"{fold_clip:>10}"
        for k in HORIZONS:
            yt = y_test[k]
            n_pos = int(yt.sum())
            excluded = n_pos == 0 or n_pos == len(yt)
            threshold = best_threshold(score_train, y_train[k])
            if excluded:
                auroc, auprc = float("nan"), float("nan")
            else:
                auroc = roc_auc_score(yt, score_test)
                auprc = average_precision_score(yt, score_test)
            record["horizons"][str(k)] = {
                "n_test": int(len(yt)),
                "n_pos": n_pos,
                "excluded": excluded,
                "threshold": threshold,
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

    print("\n=== Rule-based baseline summary: FULL SET ===")
    for k in HORIZONS:
        a = aggregate_full[str(k)]
        print(
            f"horizon {k:>3}: AUROC = {a['auroc_mean']:.4f} +/- {a['auroc_std']:.4f}   "
            f"AUPRC = {a['auprc_mean']:.4f} +/- {a['auprc_std']:.4f}   (n_folds={a['n_folds']}/{len(unique_clips)})"
        )

    print(f"\n=== Rule-based baseline summary: COMMON FOLD SUBSET {common_fold_clips} ===")
    for k in HORIZONS:
        a = aggregate_common[str(k)]
        print(
            f"horizon {k:>3}: AUROC = {a['auroc_mean']:.4f} +/- {a['auroc_std']:.4f}   "
            f"AUPRC = {a['auprc_mean']:.4f} +/- {a['auprc_std']:.4f}   (n_folds={a['n_folds']}/{len(common_fold_clips)})"
        )

    payload = {
        "description": (
            "no-learning rule-based baseline: mean of 4 sign-corrected z-scored "
            "feature-group means, equal weights, current-frame score evaluated "
            "against every horizon's future label"
        ),
        "feature_names": FEATURE_NAMES,
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
