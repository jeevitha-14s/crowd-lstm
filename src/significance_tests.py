"""Statistical significance testing over the UMN leave-one-clip-out results.

Closes the gap flagged as open item #6 in outputs/reports/PAPER_RESULTS.md
("Statistical significance testing is absent throughout"). Two questions the
existing tables state but never test:

  Q1. Is the mean AUROC at each horizon actually above chance (0.5)? This
      matters most at k=90, where the point estimate (0.734) sits well
      inside the reported +/- 0.240 spread.
  Q2. Does the LSTM genuinely beat each baseline, or is the gap inside the
      fold-to-fold noise? Relevant because logistic regression already
      matches or beats it at k=30/60 in the point estimates.

Why fold-level tests rather than DeLong on raw scores: DeLong compares two
ROC curves computed on the SAME sample, and would be the stronger test --
but per-frame prediction scores were never saved by train_forecast_model.py
(only aggregated per-fold AUROC/AUPRC), so they are not recoverable without
a full retrain of every method. What IS available is a paired design: every
method was evaluated on the identical 11 leave-one-clip-out folds, so each
fold contributes one AUROC per method and the folds pair up exactly. That
supports the standard cross-validation comparison machinery below.

Tests used, and why:
  - Percentile bootstrap over folds (10,000 resamples) for a CI on the mean
    AUROC. Resampling folds, not frames: the fold is the independent unit
    here (one held-out scene each), and frames within a clip are heavily
    autocorrelated, so a frame-level bootstrap would badly understate the
    variance.
  - Wilcoxon signed-rank for paired method comparisons. Non-parametric
    because n=7 or 11 folds is far too small to justify a normality
    assumption, and the fold AUROCs are visibly skewed (several near-1.0,
    a couple inverted below 0.5).
  - One-sided sign test for "is the mean above 0.5", reported alongside the
    bootstrap CI, since with n<=11 the bootstrap CI alone is coarse.
  - Holm-Bonferroni correction across the family of comparisons at each
    horizon. Without it, 4 horizons x 3 baselines = 12 tests would make a
    spurious "significant" result near-certain.

A blunt caveat that belongs in the paper, not just here: with 11 folds
(7 at the common subset) these tests have very low power. A non-significant
result here means "this experiment cannot tell", NOT "the methods are
equivalent". Reporting that honestly is the point of running them.
"""

import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats

LOCO_PATH = Path("outputs/reports/umn_loco_results.json")
RULE_PATH = Path("outputs/reports/umn_baseline_rule_results.json")
DEEPROD_PATH = Path("outputs/reports/deeprod_gridsearch_loco.json")
TREES_PATH = Path("outputs/reports/umn_tree_baselines_results.json")
OUT_PATH = Path("outputs/reports/significance_tests.json")

HORIZONS = ["0", "30", "60", "90"]
N_BOOTSTRAP = 10_000
SEED = 42
ALPHA = 0.05


def fold_aurocs(folds: List[Dict], horizon: str) -> Dict[int, float]:
    """{clip_id: auroc} for folds where this horizon is evaluable."""
    out: Dict[int, float] = {}
    for f in folds:
        h = f.get("horizons", {}).get(horizon)
        if h is None or h.get("excluded") or h.get("auroc") is None:
            continue
        if not np.isfinite(h["auroc"]):
            continue
        out[int(f["fold_clip"])] = float(h["auroc"])
    return out


def deeprod_fold_aurocs() -> Dict[int, float]:
    """DeepROD has no horizon concept -- one AUROC per fold, reused at every
    horizon it is compared against (see its module docstring: its lookahead
    is a single frame, so k=0 is the only horizon it is really answering)."""
    d = json.loads(DEEPROD_PATH.read_text())
    return {
        int(f["clip_id"]): float(f["selected_by_val_mse"]["auc"])
        for f in d["folds"]
        if not f.get("excluded") and np.isfinite(f["selected_by_val_mse"]["auc"])
    }


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int = N_BOOTSTRAP, alpha: float = ALPHA
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean, resampling folds with replacement."""
    rng = np.random.default_rng(SEED)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def sign_test_above(values: Sequence[float], threshold: float = 0.5) -> Dict:
    """One-sided binomial test: are more folds above `threshold` than chance
    would give? Ties (exactly at threshold) are dropped, per convention."""
    arr = np.asarray(values, dtype=float)
    above = int((arr > threshold).sum())
    below = int((arr < threshold).sum())
    n = above + below
    if n == 0:
        return {"n": 0, "n_above": 0, "p_value": float("nan")}
    p = sp_stats.binomtest(above, n, 0.5, alternative="greater").pvalue
    return {"n": n, "n_above": above, "p_value": float(p)}


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Two-sided Wilcoxon signed-rank on paired fold AUROCs (a vs b)."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    if np.allclose(diff, 0):
        return {"n_pairs": len(diff), "median_diff": 0.0, "p_value": 1.0, "note": "identical"}
    try:
        stat, p = sp_stats.wilcoxon(a_arr, b_arr)
    except ValueError as exc:  # e.g. all-zero differences after dropping ties
        return {"n_pairs": len(diff), "median_diff": float(np.median(diff)), "p_value": float("nan"), "note": str(exc)}
    return {
        "n_pairs": int(len(diff)),
        "median_diff": float(np.median(diff)),
        "mean_diff": float(diff.mean()),
        "n_favouring_a": int((diff > 0).sum()),
        "statistic": float(stat),
        "p_value": float(p),
    }


def holm_bonferroni(pvals: Dict[str, float], alpha: float = ALPHA) -> Dict[str, Dict]:
    """Holm-Bonferroni step-down correction over a family of tests."""
    finite = {k: v for k, v in pvals.items() if v is not None and np.isfinite(v)}
    ordered = sorted(finite.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: Dict[str, Dict] = {}
    rejected_so_far = True
    for i, (key, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        # Holm is step-down: once one test fails, every later (larger) p fails.
        reject = bool(rejected_so_far and p <= thresh)
        rejected_so_far = reject
        out[key] = {"p_raw": float(p), "holm_threshold": float(thresh), "significant": reject}
    for key, p in pvals.items():
        if key not in out:
            out[key] = {"p_raw": None if p is None else float(p), "holm_threshold": None, "significant": False}
    return out


def paired_on_common(a: Dict[int, float], b: Dict[int, float]) -> Tuple[List[float], List[float], List[int]]:
    """Restrict two {clip: auroc} maps to the clips both evaluated."""
    clips = sorted(set(a) & set(b))
    return [a[c] for c in clips], [b[c] for c in clips], clips


def main() -> None:
    loco = json.loads(LOCO_PATH.read_text())
    rule = json.loads(RULE_PATH.read_text())
    common = set(loco["common_fold_subset"])

    lstm_folds = loco["folds"]
    logreg_folds = loco["logistic_regression"]["folds"]
    zone_folds = loco["zone_attention_lstm"]["folds"]
    rule_folds = rule["folds"]
    deeprod = deeprod_fold_aurocs()
    trees = json.loads(TREES_PATH.read_text()) if TREES_PATH.exists() else None
    if trees is None:
        print(f"NOTE: {TREES_PATH} absent -- tree comparisons skipped. "
              f"Run `python3 -m src.baseline_trees` first to include them.")

    results: Dict[str, Dict] = {"per_horizon": {}}

    for h in HORIZONS:
        lstm = fold_aurocs(lstm_folds, h)
        methods = {
            "rule_based": fold_aurocs(rule_folds, h),
            "logistic_regression": fold_aurocs(logreg_folds, h),
            "zone_attention_lstm": fold_aurocs(zone_folds, h),
            "deeprod_reimpl": deeprod,
        }
        # Tree ensembles are the non-linear non-recurrent rung: logistic
        # regression tests whether recurrence beats a LINEAR model with the same
        # window, these test whether it beats a non-linear one. Optional so the
        # script still runs before baseline_trees.py has been executed.
        if trees is not None:
            methods["random_forest"] = fold_aurocs(trees["random_forest"]["folds"], h)
            methods["gradient_boosting"] = fold_aurocs(trees["gradient_boosting"]["folds"], h)

        entry: Dict[str, Dict] = {}

        # Q1 -- above chance? Reported on both the full evaluable set and the
        # common 7-fold subset, since the two disagree about which folds count.
        for label, clip_filter in (("full_set", None), ("common_subset", common)):
            vals = [v for c, v in sorted(lstm.items()) if clip_filter is None or c in clip_filter]
            if not vals:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals)
            entry[f"lstm_above_chance_{label}"] = {
                "n_folds": len(vals),
                "mean_auroc": mean,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "ci_excludes_0.5": bool(lo > 0.5),
                "sign_test": sign_test_above(vals, 0.5),
            }

        # Q2 -- paired comparisons against each baseline, on clips both saw.
        comparisons: Dict[str, Dict] = {}
        for name, other in methods.items():
            a, b, clips = paired_on_common(lstm, other)
            if len(clips) < 3:
                comparisons[name] = {"n_pairs": len(clips), "p_value": None, "note": "too few paired folds"}
                continue
            comparisons[name] = {**wilcoxon_paired(a, b), "clips": clips}
        entry["lstm_vs_baselines"] = comparisons
        results["per_horizon"][h] = entry

    # Holm correction across the whole family: 4 horizons x 4 baselines.
    family = {
        f"k={h}:{name}": cmp.get("p_value")
        for h, e in results["per_horizon"].items()
        for name, cmp in e["lstm_vs_baselines"].items()
    }
    results["holm_bonferroni_across_all_comparisons"] = holm_bonferroni(family)

    results["method_notes"] = {
        "unit_of_resampling": "fold (one held-out clip), not frame",
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "deeprod_horizon_note": (
            "DeepROD produces a single AUROC per fold with no horizon concept; the same "
            "value is compared at every horizon, so only the k=0 comparison is a like-for-like "
            "task match. The k=30/60/90 rows show what its single-frame lookahead scores "
            "against labels it was never designed to predict."
        ),
        "power_caveat": (
            "n=7-11 folds. These tests are severely underpowered: a non-significant result "
            "means the experiment cannot distinguish the methods, not that they are equivalent."
        ),
        "why_not_delong": (
            "Per-frame prediction scores were not saved by train_forecast_model.py, so a "
            "DeLong test on paired ROC curves is not computable without retraining every method."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))

    # ---- console summary ----
    print("=== Q1: is the LSTM above chance? (bootstrap CI over folds) ===")
    for h in HORIZONS:
        e = results["per_horizon"][h].get("lstm_above_chance_common_subset")
        if not e:
            continue
        st = e["sign_test"]
        print(
            f"  k={h:>2}: AUROC {e['mean_auroc']:.3f}  95% CI [{e['ci95_lo']:.3f}, {e['ci95_hi']:.3f}]"
            f"  {'ABOVE 0.5' if e['ci_excludes_0.5'] else 'CI includes 0.5'}"
            f"   sign test {st['n_above']}/{st['n']} p={st['p_value']:.4f}"
        )

    print("\n=== Q2: LSTM vs baselines (Wilcoxon, paired folds) ===")
    for h in HORIZONS:
        print(f"  k={h}:")
        for name, cmp in results["per_horizon"][h]["lstm_vs_baselines"].items():
            p = cmp.get("p_value")
            if p is None or not np.isfinite(p):
                print(f"    {name:<22} n/a ({cmp.get('note','')})")
                continue
            holm = results["holm_bonferroni_across_all_comparisons"].get(f"k={h}:{name}", {})
            sig = "SIG" if holm.get("significant") else "ns"
            print(
                f"    {name:<22} n={cmp['n_pairs']:>2} median diff {cmp['median_diff']:+.3f}"
                f"  wins {cmp.get('n_favouring_a','?')}/{cmp['n_pairs']}  p={p:.4f}  {sig} (Holm)"
            )
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
