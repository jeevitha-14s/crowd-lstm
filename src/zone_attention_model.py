"""Spatial attention over zones: an architectural variant of CrowdRiskLSTM
that learns a per-timestep weighting over the 16 spatial-grid zones before
the LSTM backbone, evaluated on the identical 11-fold LOCO splits as
train_forecast_model.py for a like-for-like comparison.

Where ablation.py answers "which features matter" by zeroing feature groups
in a fixed architecture, this answers "which zones matter" by letting the
model learn that weighting itself, and adds one new ablation row: baseline
LSTM vs. LSTM + zone attention, same folds, same horizons.

Zone layout: each timestep's 64-dim vector is 16 zones x 4 features,
zone-major (index = zone*4 + feature) -- see feature_extractor.py's
`base = zone_idx * N_FEATURES_PER_ZONE`. ZoneAttention reshapes to
(zones, features), scores each zone with a single shared Linear(4, 1),
softmax-normalizes across the 16 zones, and reweights each zone's 4-dim
block before flattening back to 64-dim for the LSTM -- a lightweight,
per-timestep, input-dependent attention that adds negligible parameters
(5 total: 4 weights + 1 bias, shared across all zones and timesteps) so any
AUROC change is attributable to the reweighting mechanism itself, not added
capacity.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from src.train_forecast_model import (
    DROPOUT,
    HIDDEN_SIZE,
    HORIZONS,
    NUM_LAYERS,
    CrowdRiskLSTM,
    train_one_fold,
)

DATASET_PATH = "data/forecast_dataset.npz"
OUT_JSON_PATH = Path("outputs/reports/umn_zone_attention_results.json")
N_ZONES = 16
N_FEATURES_PER_ZONE = 4
VARIANTS = ["baseline_lstm", "zone_attention_lstm"]


class ZoneAttention(nn.Module):
    """Per-timestep learned softmax weighting over the 16 spatial zones."""

    def __init__(self, n_zones: int = N_ZONES, n_features_per_zone: int = N_FEATURES_PER_ZONE) -> None:
        super().__init__()
        self.n_zones = n_zones
        self.n_features_per_zone = n_features_per_zone
        self.score = nn.Linear(n_features_per_zone, 1)
        self.last_weights: torch.Tensor = None  # (B, T, n_zones), set on each forward for inspection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        zones = x.view(batch, steps, self.n_zones, self.n_features_per_zone)
        scores = self.score(zones).squeeze(-1)  # (B, T, n_zones)
        weights = torch.softmax(scores, dim=-1)
        self.last_weights = weights.detach()
        weighted = zones * weights.unsqueeze(-1)
        return weighted.view(batch, steps, self.n_zones * self.n_features_per_zone)


class CrowdRiskLSTMZoneAttention(nn.Module):
    """ZoneAttention -> shared LSTM backbone -> one Linear head per horizon.
    Same forward() contract as CrowdRiskLSTM (Dict[str, Tensor] of logits) so
    it is a drop-in model_cls for train_one_fold."""

    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        horizons: List[int] = HORIZONS,
    ) -> None:
        super().__init__()
        self.attention = ZoneAttention()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.shared_dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({str(k): nn.Linear(hidden_size, 1) for k in horizons})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        attended = self.attention(x)
        _, (hn, _) = self.lstm(attended)
        shared = self.shared_dropout(hn[-1])
        return {k: head(shared).squeeze(-1) for k, head in self.heads.items()}


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
    model_cls = CrowdRiskLSTM if variant == "baseline_lstm" else CrowdRiskLSTMZoneAttention

    unique_clips = sorted(np.unique(clip_ids).tolist())
    fold_records: List[Dict] = []
    for fold_clip in unique_clips:
        train_mask = clip_ids != fold_clip
        test_mask = clip_ids == fold_clip
        X_train, X_test = X[train_mask], X[test_mask]
        y_train = {k: y[k][train_mask] for k in HORIZONS}
        y_test = {k: y[k][test_mask] for k in HORIZONS}
        train_clip_ids = clip_ids[train_mask]

        probs, val_clips, best_epoch, stopped_epoch = train_one_fold(
            X_train, y_train, train_clip_ids, X_test, fold_clip, device, model_cls=model_cls
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

    print("\n=== Zone attention vs baseline LSTM (common-subset AUROC mean) ===")
    base_agg = all_results["baseline_lstm"]["aggregate_common_subset"]
    header = f"{'variant':24s}" + "".join(f"{'k' + str(k):>16}" for k in HORIZONS)
    print(header)
    for variant in VARIANTS:
        agg = all_results[variant]["aggregate_common_subset"]
        row = f"{variant:24s}"
        for k in HORIZONS:
            v = agg[str(k)]["auroc_mean"]
            if variant == "baseline_lstm":
                row += f"{v:>16.4f}"
            else:
                delta = v - base_agg[str(k)]["auroc_mean"]
                row += f"{v:>9.4f}({delta:+.4f})"
        print(row)

    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
