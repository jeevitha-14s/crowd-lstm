"""Tree-ensemble controls: random forest and gradient boosting, both LEARNED
and NON-RECURRENT, on exactly the same flattened 30x64 window, the same 11
LOCO folds and the same 4 horizons as train_forecast_model.py and
baseline_logreg.py.

This extends the existing baseline ladder by one rung. baseline_rule.py asks
"does any learning help?"; baseline_logreg.py asks "does recurrence beat a
LINEAR model with the same information?" -- and already answered awkwardly:
logistic regression matches or slightly beats the LSTM at k=30 and k=60.
These two ask the natural follow-up: does recurrence beat a strong NON-LINEAR
non-recurrent model? Linear separability was a real limitation of the logreg
control, so if trees also match the LSTM the recurrence story is weak on
non-linearity grounds too, not just linear ones.

The expected outcome is worth stating in advance so the result cannot be
read selectively either way: given logistic regression already wins at the
mid horizons, tree ensembles are likely to as well. That does NOT undercut
this project's contribution -- the contribution is the forecasting reframe
(panic is predictable k seconds ahead under honest held-out evaluation), and
several independent model families confirming the signal makes that claim
stronger, not weaker. What it would undercut is any claim that recurrence
specifically is necessary, which the logreg control already fails to support.

Trees need no feature scaling, so unlike baseline_logreg.py there is no
StandardScaler here -- that difference is deliberate, not an oversight.
Gradient boosting uses the histogram-based implementation, which is the
practical choice at 1920 features; it is still gradient boosting.
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

DATASET_PATH = "data/forecast_dataset.npz"
OUT_JSON_PATH = Path("outputs/reports/umn_tree_baselines_results.json")
HORIZONS = [0, 30, 60, 90]
SEED = 42

RF_N_ESTIMATORS = 300
RF_MIN_LEAF = 2
GB_MAX_ITER = 300
GB_LR = 0.05


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


def fit_and_score_rf(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, min_samples_leaf=RF_MIN_LEAF,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def fit_and_score_gb(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    clf = HistGradientBoostingClassifier(
        max_iter=GB_MAX_ITER, learning_rate=GB_LR,
        class_weight="balanced", random_state=SEED,
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def run_model(name: str, scorer: Callable, X_flat: np.ndarray, y: Dict[int, np.ndarray],
              clip_ids: np.ndarray, unique_clips: List[int]) -> Dict:
    print(f"\n=== {name}: learned, non-recurrent, full-window ===")
    header = f"{'fold(clip)':>10}" + "".join(
        f"{'AUROC_k' + str(k):>12}{'AUPRC_k' + str(k):>12}" for k in HORIZONS)
    print(header)

    fold_records: List[Dict] = []
    t0 = time.perf_counter()
    for fold_clip in unique_clips:
        train_mask = clip_ids != fold_clip
        test_mask = clip_ids == fold_clip
        X_train, X_test = X_flat[train_mask], X_flat[test_mask]

        record = {"fold_clip": int(fold_clip), "horizons": {}}
        row = f"{fold_clip:>10}"
        for k in HORIZONS:
            yt_train, yt_test = y[k][train_mask], y[k][test_mask]
            n_pos_test = int(yt_test.sum())
            excluded = n_pos_test == 0 or n_pos_test == len(yt_test)
            train_degenerate = yt_train.sum() == 0 or yt_train.sum() == len(yt_train)
            if excluded or train_degenerate:
                auroc, auprc = float("nan"), float("nan")
                excluded = True
            else:
                probs = scorer(X_train, yt_train, X_test)
                auroc = roc_auc_score(yt_test, probs)
                auprc = average_precision_score(yt_test, probs)
            record["horizons"][str(k)] = {
                "n_test": int(len(yt_test)), "n_pos": n_pos_test,
                "excluded": excluded, "auroc": auroc, "auprc": auprc,
            }
            row += f"{auroc:>12.4f}{auprc:>12.4f}"
        fold_records.append(record)
        print(row)
    elapsed = time.perf_counter() - t0

    aggregate_full = {str(k): _agg(fold_records, k) for k in HORIZONS}
    common_fold_clips = [
        r["fold_clip"] for r in fold_records
        if all(not r["horizons"][str(k)]["excluded"] for k in HORIZONS)
    ]
    common_records = [r for r in fold_records if r["fold_clip"] in common_fold_clips]
    aggregate_common = {str(k): _agg(common_records, k) for k in HORIZONS}

    print(f"\n--- {name}: COMMON FOLD SUBSET {common_fold_clips} ---")
    for k in HORIZONS:
        a = aggregate_common[str(k)]
        print(f"horizon {k:>3}: AUROC = {a['auroc_mean']:.4f} +/- {a['auroc_std']:.4f}   "
              f"AUPRC = {a['auprc_mean']:.4f} +/- {a['auprc_std']:.4f}   (n_folds={a['n_folds']})")
    print(f"  ({elapsed:.1f}s)")

    return {
        "folds": fold_records,
        "common_fold_subset": common_fold_clips,
        "aggregate_full_set": aggregate_full,
        "aggregate_common_subset": aggregate_common,
        "elapsed_s": elapsed,
    }


def print_ladder(results: Dict) -> None:
    """The whole point of this script is the comparison, so print it."""
    lstm = {"0": 0.963, "30": 0.882, "60": 0.791, "90": 0.734}
    rule = {"0": 0.686, "30": 0.571, "60": 0.592, "90": 0.583}
    logreg = {"0": 0.924, "30": 0.904, "60": 0.830, "90": 0.681}
    print("\n\n=== Baseline ladder, common-fold subset, AUROC ===")
    print(f"{'horizon':>8}{'rule':>10}{'logreg':>10}{'rand.for':>10}{'grad.boost':>12}{'LSTM':>10}")
    for k in HORIZONS:
        ks = str(k)
        rf = results["random_forest"]["aggregate_common_subset"][ks]["auroc_mean"]
        gb = results["gradient_boosting"]["aggregate_common_subset"][ks]["auroc_mean"]
        print(f"{k:>8}{rule[ks]:>10.3f}{logreg[ks]:>10.3f}{rf:>10.3f}{gb:>12.3f}{lstm[ks]:>10.3f}")
    print("\n  rule / logreg / LSTM columns are prior results, quoted from "
          "outputs/reports/ (umn_baseline_rule_results.json, "
          "umn_logreg_control_results.json, umn_loco_results.json).")


def main() -> None:
    data = np.load(DATASET_PATH)
    X = data["X"]
    clip_ids = data["clip_ids"]
    y = {k: data[f"y_{k}"] for k in HORIZONS}
    X_flat = X.reshape(X.shape[0], -1)
    unique_clips = sorted(np.unique(clip_ids).tolist())

    print(f"X {X.shape} -> flattened {X_flat.shape}, {len(unique_clips)} LOCO folds")

    results = {
        "random_forest": run_model("Random forest", fit_and_score_rf, X_flat, y, clip_ids, unique_clips),
        "gradient_boosting": run_model("Gradient boosting", fit_and_score_gb, X_flat, y, clip_ids, unique_clips),
    }
    print_ladder(results)

    payload = {
        "description": (
            "Tree-ensemble controls (random forest, histogram gradient boosting): learned, "
            "non-recurrent, trained on the flattened 30x64 window with identical information "
            "access to the LSTM and the identical 11-fold LOCO splits. Extends the "
            "rule < logistic-regression < LSTM ladder with a non-linear non-recurrent rung. "
            "No feature scaling (trees do not need it)."
        ),
        "model_config": {
            "random_forest": {"n_estimators": RF_N_ESTIMATORS, "min_samples_leaf": RF_MIN_LEAF,
                              "class_weight": "balanced", "seed": SEED},
            "gradient_boosting": {"implementation": "HistGradientBoostingClassifier",
                                  "max_iter": GB_MAX_ITER, "learning_rate": GB_LR,
                                  "class_weight": "balanced", "seed": SEED},
        },
        "prior_results_for_comparison": {
            "note": "quoted from other files in outputs/reports/, not recomputed here",
            "rule_baseline_common": {"0": 0.686, "30": 0.571, "60": 0.592, "90": 0.583},
            "logistic_regression_common": {"0": 0.924, "30": 0.904, "60": 0.830, "90": 0.681},
            "lstm_common": {"0": 0.963, "30": 0.882, "60": 0.791, "90": 0.734},
        },
        **results,
    }
    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
