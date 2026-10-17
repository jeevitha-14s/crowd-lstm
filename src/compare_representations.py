"""Tracked vs optical-flow vs concatenated representations on UMN.

Every baseline in the ladder (rule, logistic regression, random forest,
gradient boosting, zone attention) consumes the same tracked descriptor as the
proposed model, so they vary the classifier while holding the representation
fixed. None of them tests the claim the paper's second contribution rests on:
that tracking individual pedestrians beats treating the crowd as an
undifferentiated motion field, which is what [2] and [5] do. The DeepROD
comparison does vary the representation but changes the architecture, the
supervision and the protocol at the same time, so it cannot isolate the cause.

This script reads the three LOCO runs that differ ONLY in their input
features -- identical grid, LSTM, folds, horizons and labels -- and reports:

  1. mean AUROC per horizon on the fold subset the three share;
  2. paired Wilcoxon of each alternative against the tracked reference;
  3. how often each representation falls below chance, which turned out to
     matter more than the means;
  4. per-clip feature direction during the anomaly, which explains why.

Point 4 is the substantive finding. The tracked descriptor's person count
falls during every one of the eleven onsets without exception, while the flow
descriptor's motion-density rises in some clips and falls in others: it
measures how many pixels are moving, which increases when a crowd runs across
the frame and decreases when it runs out of frame. The tracked signature is
invariant to escape geometry; the flow signature is not. That is why the flow
model collapses on clips 4 and 7 while matching or beating the tracked model
elsewhere.

Reads only existing artifacts; writes outputs/reports/representation_comparison.json.
Regenerate the inputs with:
  python3 -m src.process_umn_opticalflow
  python3 -m src.build_forecast_dataset --features data/features/umn_flow_features.npy \\
      --meta data/features/umn_flow_meta.csv --out data/flow_forecast_dataset.npz
  python3 -m src.train_forecast_model --dataset data/flow_forecast_dataset.npz \\
      --out outputs/reports/umn_flow_loco_results.json
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

RUNS = {
    "tracked": Path("outputs/reports/umn_loco_results.json"),
    "flow": Path("outputs/reports/umn_flow_loco_results.json"),
    "combined": Path("outputs/reports/umn_combined_loco_results.json"),
}
FEATURES = {
    "tracked": Path("data/features/umn_features.npy"),
    "flow": Path("data/features/umn_flow_features.npy"),
}
META = Path("data/features/umn_meta.csv")
GROUNDTRUTH = Path("data/umn_groundtruth.csv")
OUT = Path("outputs/reports/representation_comparison.json")

HORIZONS = ["0", "30", "60", "90"]
REFERENCE = "tracked"
FEATURE_NAMES = ["normalized_count", "flow_magnitude", "direction_entropy", "stop_ratio"]


def fold_aurocs(path: Path, horizon: str) -> Dict[int, float]:
    """{clip_id: AUROC} for folds evaluable at this horizon."""
    data = json.loads(path.read_text())
    out: Dict[int, float] = {}
    for fold in data["folds"]:
        h = fold["horizons"][horizon]
        if h["excluded"] or h["auroc"] is None or not np.isfinite(h["auroc"]):
            continue
        out[int(fold["fold_clip"])] = float(h["auroc"])
    return out


def compare_horizon(horizon: str) -> Dict:
    per_run = {name: fold_aurocs(path, horizon) for name, path in RUNS.items()}
    shared = sorted(set.intersection(*(set(v) for v in per_run.values())))
    entry: Dict = {"n_folds": len(shared), "folds": shared, "mean_auroc": {}, "vs_reference": {}}
    for name, aurocs in per_run.items():
        entry["mean_auroc"][name] = float(np.mean([aurocs[c] for c in shared]))

    for name in RUNS:
        if name == REFERENCE:
            continue
        diff = np.array([per_run[name][c] - per_run[REFERENCE][c] for c in shared])
        # Wilcoxon is undefined when every pair is tied; that never occurs here
        # but guard anyway rather than raise mid-report.
        p = float(stats.wilcoxon(diff).pvalue) if not np.allclose(diff, 0) else 1.0
        entry["vs_reference"][name] = {
            "mean_difference": float(diff.mean()),
            "median_difference": float(np.median(diff)),
            "n_folds_favouring_reference": int((diff < 0).sum()),
            "wilcoxon_p": p,
        }
    return entry


def below_chance(horizon: str) -> Dict:
    """How often each representation is worse than guessing, on the folds all share."""
    per_run = {name: fold_aurocs(path, horizon) for name, path in RUNS.items()}
    shared = sorted(set.intersection(*(set(v) for v in per_run.values())))
    return {
        name: {
            "count": int(sum(per_run[name][c] < 0.5 for c in shared)),
            "of": len(shared),
            "clips": [c for c in shared if per_run[name][c] < 0.5],
        }
        for name in RUNS
    }


def feature_direction() -> Dict:
    """Per-clip feature shift during the anomaly, for both representations.

    Reported as (mean over anomalous frames) - (mean over normal frames) for
    each of the four feature groups, averaged across that representation's
    zones. The sign is the point: a representation whose sign flips between
    clips cannot have learned a transferable onset signature.
    """
    meta = pd.read_csv(META)
    gt = pd.read_csv(GROUNDTRUTH)
    df = meta.merge(gt, on="frame")
    arrays = {k: np.load(v) for k, v in FEATURES.items()}

    out: Dict[str, Dict] = {}
    for rep, F in arrays.items():
        per_clip: Dict[str, Dict[str, float]] = {}
        for clip, g in df.groupby("clip_id"):
            idx = g.index.values
            abnormal = g["abnormal"].values.astype(bool)
            if abnormal.sum() < 10 or (~abnormal).sum() < 10:
                continue
            A, N = F[idx][abnormal], F[idx][~abnormal]
            per_clip[str(int(clip))] = {
                name: float(A[:, i::4].mean() - N[:, i::4].mean())
                for i, name in enumerate(FEATURE_NAMES)
            }
        signs = {
            name: {
                "n_negative": sum(1 for v in per_clip.values() if v[name] < 0),
                "n_positive": sum(1 for v in per_clip.values() if v[name] > 0),
            }
            for name in FEATURE_NAMES
        }
        out[rep] = {"per_clip": per_clip, "sign_counts": signs, "n_clips": len(per_clip)}
    return out


def main() -> None:
    missing = [str(p) for p in list(RUNS.values()) + list(FEATURES.values()) if not p.exists()]
    if missing:
        raise SystemExit("missing inputs (see module docstring for how to regenerate):\n  "
                         + "\n  ".join(missing))

    horizons = {h: compare_horizon(h) for h in HORIZONS}
    chance = {h: below_chance(h) for h in HORIZONS}
    directions = feature_direction()

    print("=== Mean AUROC (folds shared by all three representations) ===")
    print(f"{'horizon':>8}" + "".join(f"{n:>11}" for n in RUNS) + f"{'n_folds':>9}")
    for h in HORIZONS:
        e = horizons[h]
        print(f"{h:>8}" + "".join(f"{e['mean_auroc'][n]:>11.3f}" for n in RUNS)
              + f"{e['n_folds']:>9}")

    print(f"\n=== Paired Wilcoxon vs '{REFERENCE}' ===")
    for h in HORIZONS:
        for name, v in horizons[h]["vs_reference"].items():
            print(f"  k={h:>2} {name:<9} mean diff {v['mean_difference']:+.3f}  "
                  f"median {v['median_difference']:+.3f}  "
                  f"{REFERENCE} wins {v['n_folds_favouring_reference']}/{horizons[h]['n_folds']}  "
                  f"p={v['wilcoxon_p']:.4f}")

    print("\n=== Folds scoring below chance ===")
    print(f"{'horizon':>8}" + "".join(f"{n:>13}" for n in RUNS))
    for h in HORIZONS:
        print(f"{h:>8}" + "".join(f"{chance[h][n]['count']:>7}/{chance[h][n]['of']:<5}" for n in RUNS))
    print(f"\n  at k=90: " + ",  ".join(
        f"{n} fails on clips {chance['90'][n]['clips']}" for n in RUNS))

    print("\n=== Feature direction during anomaly (negative = falls) ===")
    for rep, d in directions.items():
        counts = ", ".join(f"{n}: {v['n_negative']}neg/{v['n_positive']}pos"
                           for n, v in d["sign_counts"].items())
        print(f"  {rep:<9} over {d['n_clips']} clips -> {counts}")

    payload = {
        "description": (
            "Tracked vs optical-flow vs concatenated input representations under an otherwise "
            "identical pipeline: same 4x4 grid, same LSTM, same leave-one-clip-out folds, same "
            "horizons, same labels. Isolates the representation as the only independent variable."
        ),
        "runs": {k: str(v) for k, v in RUNS.items()},
        "reference": REFERENCE,
        "per_horizon": horizons,
        "below_chance": chance,
        "feature_direction_during_anomaly": directions,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
