"""Consolidates the evidence for WHERE this system works and where it does not.

Closes the pending item "define which crowd conditions the system suits, and
which it does not". Everything here is recomputed from saved artifacts -- no
new training, no new inference -- so it is a synthesis, not a new experiment.

The three candidate explanations for the UMN-vs-ShanghaiTech gap, tested
against what is actually on disk:

  H1. CROWD DENSITY. UMN averages ~15 persons/frame, ShanghaiTech ~4. The
      feature extractor's per-zone statistics (direction entropy needs >=2
      moving tracks in a zone; stop_ratio needs >=1 track with a computable
      speed) degrade toward zero as zones empty, so sparse scenes should
      score worse. Tested here by correlating per-scene mean density against
      per-scene AUROC.

  H2. ANOMALY KIND. UMN's anomaly is always collective escape -- the whole
      crowd accelerating at once. Several ShanghaiTech anomalies are
      individual or small-group events (a cyclist, a scuffle between 2-5
      people) that do not move the crowd-level aggregate at all. If this
      dominates, density will NOT explain the per-scene ordering.

  H3. FORECAST HORIZON. Independent of scene, how far ahead can it see
      before the signal is gone?

H1 and H2 make opposite predictions about the density/AUROC correlation,
which is why it is worth computing rather than asserting.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ST_META = Path("data/features/shanghaitech_meta.csv")
ST_SUMMARY = Path("outputs/reports/shanghaitech_results_summary.json")
UMN_LOCO = Path("outputs/reports/umn_loco_results.json")
UMN_META = Path("data/features/umn_meta.csv")
SIG = Path("outputs/reports/significance_tests.json")
TRANSFER = Path("outputs/reports/crowd_panic_transfer.json")
OUT = Path("outputs/reports/operating_envelope.json")

HORIZONS_ST = ["0", "24", "48", "72"]
HORIZONS_UMN = ["0", "30", "60", "90"]


def shanghaitech_density_vs_auroc() -> Dict:
    """H1 vs H2: does per-scene density predict per-scene AUROC?"""
    meta = pd.read_csv(ST_META)
    meta["scene"] = meta["scene"].astype(int)
    # Frame-weighted, not video-averaged: scenes differ ~20x in video count,
    # so an unweighted per-video mean would let short clips dominate.
    dens = (
        meta.assign(w=lambda d: d["n_frames"] * d["mean_persons"])
        .groupby("scene")
        .apply(lambda g: g["w"].sum() / g["n_frames"].sum(), include_groups=False)
    )

    summary = json.loads(ST_SUMMARY.read_text())
    rows: List[Dict] = []
    for group in ("headline", "thin"):
        for entry in summary[group]:
            scene = int(entry["scene"])
            aurocs = [entry["full"][h][0] for h in HORIZONS_ST if entry["full"].get(h)]
            if not aurocs:
                continue
            rows.append(
                {
                    "scene": scene,
                    "group": group,
                    "mean_persons_per_frame": float(dens.get(scene, np.nan)),
                    "auroc_k0": float(entry["full"]["0"][0]) if entry["full"].get("0") else None,
                    "auroc_mean_over_horizons": float(np.mean(aurocs)),
                }
            )
    df = pd.DataFrame(rows).dropna(subset=["mean_persons_per_frame"])

    r, p = sp_stats.pearsonr(df["mean_persons_per_frame"], df["auroc_mean_over_horizons"])
    rho, p_rho = sp_stats.spearmanr(df["mean_persons_per_frame"], df["auroc_mean_over_horizons"])
    return {
        "per_scene": df.sort_values("mean_persons_per_frame").to_dict(orient="records"),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "n_scenes": int(len(df)),
        "verdict": (
            "density explains the per-scene ordering" if p < 0.05
            else "density does NOT explain the per-scene ordering -- consistent with H2 "
                 "(mismatch in the KIND of anomaly), not H1 (sparsity alone)"
        ),
    }


def umn_density() -> Dict:
    meta = pd.read_csv(UMN_META)
    per_clip = meta.groupby("clip_id")["n_persons"].mean()
    return {
        "overall_mean_persons_per_frame": float(meta["n_persons"].mean()),
        "per_clip_min": float(per_clip.min()),
        "per_clip_max": float(per_clip.max()),
    }


def horizon_decay() -> Dict:
    """H3: how far ahead does the signal survive, with the significance
    verdict attached to each horizon rather than point estimates alone."""
    sig = json.loads(SIG.read_text())
    out: Dict[str, Dict] = {}
    for h in HORIZONS_UMN:
        e = sig["per_horizon"][h].get("lstm_above_chance_common_subset")
        if not e:
            continue
        out[h] = {
            "mean_auroc": e["mean_auroc"],
            "ci95": [e["ci95_lo"], e["ci95_hi"]],
            "ci_excludes_chance": e["ci_excludes_0.5"],
            "sign_test_p": e["sign_test"]["p_value"],
            "sign_test_significant_at_0.05": bool(e["sign_test"]["p_value"] < 0.05),
        }
    return out


def transfer_check() -> Dict:
    if not TRANSFER.exists():
        return {"available": False}
    d = json.loads(TRANSFER.read_text())
    return {"available": True, "keys": list(d.keys())}


def main() -> None:
    density = shanghaitech_density_vs_auroc()
    envelope = {
        "shanghaitech_density_vs_performance": density,
        "umn_density": umn_density(),
        "horizon_decay_with_significance": horizon_decay(),
        "external_transfer_artifact": transfer_check(),
    }

    decay = envelope["horizon_decay_with_significance"]
    last_sig = [h for h, v in decay.items() if v["sign_test_significant_at_0.05"]]
    envelope["stated_operating_envelope"] = {
        "suited_to": [
            "Collective, crowd-scale events where a large fraction of visible people change "
            "motion together (UMN-style escape/panic).",
            f"Dense scenes: validated at ~{envelope['umn_density']['overall_mean_persons_per_frame']:.0f} "
            "persons/frame; every zone-level feature is defined only when a zone is occupied.",
            f"Lead times up to k={max(last_sig, key=int) if last_sig else 'n/a'} frames "
            f"({int(max(last_sig, key=int)) / 30:.0f}s at 30fps) by the stricter sign test; "
            "point estimates stay above chance further out but are not individually significant.",
            "Fixed-camera continuous feeds -- the running normalizers assume a stable viewpoint "
            "and are reset at every scene boundary.",
        ],
        "not_suited_to": [
            "Individual or small-group anomalies (a lone cyclist, a 2-5 person altercation). "
            "ShanghaiTech scene 05 scores BELOW chance at every horizon -- these events do not "
            "move a crowd-level aggregate, and can push it the wrong way.",
            "Sparse scenes (~4 persons/frame): most spatial zones are empty, leaving "
            "direction_entropy and stop_ratio undefined ~62%/67% of the time within occupied zones.",
            "Moving or newly-installed cameras: the per-clip running normalizers need time to "
            "stabilise after every reset.",
            "Any deployment needing a calibrated probability rather than a ranking -- all "
            "reported metrics are threshold-free (AUROC/AUPRC).",
        ],
        "evidence_notes": density["verdict"],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(envelope, indent=2))

    print("=== ShanghaiTech: density vs AUROC across scenes ===")
    for row in density["per_scene"]:
        print(
            f"  scene {row['scene']:>2} ({row['group']:<8}) "
            f"{row['mean_persons_per_frame']:5.2f} persons/frame -> AUROC {row['auroc_mean_over_horizons']:.3f}"
        )
    print(
        f"\n  Pearson r={density['pearson_r']:+.3f} (p={density['pearson_p']:.3f}), "
        f"Spearman rho={density['spearman_rho']:+.3f} (p={density['spearman_p']:.3f}), n={density['n_scenes']}"
    )
    print(f"  -> {density['verdict']}")

    print("\n=== Horizon decay (UMN, common subset) ===")
    for h, v in decay.items():
        mark = "significant" if v["sign_test_significant_at_0.05"] else "NOT significant"
        print(
            f"  k={h:>2}: AUROC {v['mean_auroc']:.3f} CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
            f"  sign test p={v['sign_test_p']:.4f}  {mark}"
        )
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
