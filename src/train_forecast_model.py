"""Task 4: shared-backbone LSTM with one binary head per forecast horizon,
evaluated via leave-one-clip-out cross-validation (11 folds).

Split is always by clip_id, never by row: adjacent samples share up to 144 of
145 history frames (stride=5, seq_len=30), so a row-level split would leak
near-duplicate windows across train/test.

Each outer fold further splits its 10 training clips into ~8 inner-train +
2 inner-validation clips, used only for early stopping: with 11 independent
clips total (not 5154 independent rows), unconstrained training duration is
the primary overfitting risk, more so than model width. Stopping is decided
on a SMOOTHED (moving-average) validation AUROC, MEAN across all four heads
-- smoothed because a 2-clip validation set is itself high-variance and would
trigger on noise; mean-across-heads because selecting per-head would silently
turn this back into four independent models and lose the shared-backbone
rationale.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

DATASET_PATH = "data/forecast_dataset.npz"
OUT_JSON_PATH = Path("outputs/reports/umn_loco_results.json")
HORIZONS = [0, 30, 60, 90]
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
LR = 1e-3
BATCH_SIZE = 64
SEED = 42

INNER_VAL_CLIPS = 2
MAX_EPOCHS = 150
PATIENCE = 20
SMOOTH_WINDOW = 5


class CrowdRiskLSTM(nn.Module):
    """Single LSTM backbone -> one Linear head per forecast horizon."""

    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        horizons: List[int] = HORIZONS,
    ) -> None:
        super().__init__()
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
        _, (hn, _) = self.lstm(x)
        shared = self.shared_dropout(hn[-1])  # (B, hidden_size), final layer's hidden state
        return {k: head(shared).squeeze(-1) for k, head in self.heads.items()}


def compute_pos_weight(y: np.ndarray) -> float:
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def mean_val_auroc(model: CrowdRiskLSTM, X_val_t: torch.Tensor, y_val: Dict[int, np.ndarray]) -> float:
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_probs = {k: torch.sigmoid(val_preds[str(k)]).cpu().numpy() for k in HORIZONS}
    head_aurocs = []
    for k in HORIZONS:
        yv = y_val[k]
        if yv.sum() == 0 or yv.sum() == len(yv):
            continue  # this head undefined on this tiny validation set this epoch
        head_aurocs.append(roc_auc_score(yv, val_probs[k]))
    return float(np.mean(head_aurocs)) if head_aurocs else float("nan")


def train_one_fold(
    X_train_all: np.ndarray,
    y_train_all: Dict[int, np.ndarray],
    train_clip_ids: np.ndarray,
    X_test: np.ndarray,
    fold_clip: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], List[int], int, int]:
    rng = np.random.default_rng(SEED + fold_clip)
    train_clips_unique = sorted(np.unique(train_clip_ids).tolist())
    val_clips = sorted(rng.choice(train_clips_unique, size=INNER_VAL_CLIPS, replace=False).tolist())
    val_mask = np.isin(train_clip_ids, val_clips)
    inner_mask = ~val_mask

    X_inner = X_train_all[inner_mask]
    y_inner = {k: y_train_all[k][inner_mask] for k in HORIZONS}
    y_val = {k: y_train_all[k][val_mask] for k in HORIZONS}

    torch.manual_seed(SEED)
    model = CrowdRiskLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fns = {
        k: nn.BCEWithLogitsLoss(pos_weight=torch.tensor(compute_pos_weight(y_inner[k]), device=device))
        for k in HORIZONS
    }

    X_inner_t = torch.tensor(X_inner, dtype=torch.float32, device=device)
    y_inner_t = {k: torch.tensor(y_inner[k], dtype=torch.float32, device=device) for k in HORIZONS}
    X_val_t = torch.tensor(X_train_all[val_mask], dtype=torch.float32, device=device)

    n = X_inner_t.shape[0]
    val_history: List[float] = []
    best_smoothed = float("-inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    stopped_epoch = MAX_EPOCHS

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            preds = model(X_inner_t[idx])
            loss = sum(loss_fns[k](preds[str(k)], y_inner_t[k][idx]) for k in HORIZONS)
            loss.backward()
            optimizer.step()

        val_history.append(mean_val_auroc(model, X_val_t, y_val))
        window = [v for v in val_history[-SMOOTH_WINDOW:] if not np.isnan(v)]
        smoothed = float(np.mean(window)) if window else float("-inf")

        if smoothed > best_smoothed:
            best_smoothed = smoothed
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve > PATIENCE:
            stopped_epoch = epoch + 1
            break

    model.load_state_dict(best_state)
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(X_test_t)
        probs = {k: torch.sigmoid(preds[str(k)]).cpu().numpy() for k in HORIZONS}
    return probs, val_clips, best_epoch, stopped_epoch


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(DATASET_PATH)
    X = data["X"]
    clip_ids = data["clip_ids"]
    y = {k: data[f"y_{k}"] for k in HORIZONS}

    unique_clips = sorted(np.unique(clip_ids).tolist())
    print(f"Device: {device}")
    print(
        f"Model: hidden={HIDDEN_SIZE} layers={NUM_LAYERS} dropout={DROPOUT} weight_decay={WEIGHT_DECAY}, "
        f"{len(unique_clips)} outer folds, inner_val_clips={INNER_VAL_CLIPS}, "
        f"patience={PATIENCE} (on smoothed mean-of-4-heads val AUROC)"
    )
    print()

    header = f"{'fold(clip)':>10}" + "".join(f"{'AUROC_k' + str(k):>12}{'AUPRC_k' + str(k):>12}" for k in HORIZONS)
    print(header)

    fold_records: List[Dict] = []
    for fold_clip in unique_clips:
        train_mask = clip_ids != fold_clip
        test_mask = clip_ids == fold_clip

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = {k: y[k][train_mask] for k in HORIZONS}
        y_test = {k: y[k][test_mask] for k in HORIZONS}
        train_clip_ids = clip_ids[train_mask]

        probs, val_clips, best_epoch, stopped_epoch = train_one_fold(
            X_train, y_train, train_clip_ids, X_test, fold_clip, device
        )

        record = {
            "fold_clip": fold_clip,
            "scene": None,  # no clip-to-scene mapping exists in any saved artifact
            "val_clips": val_clips,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "test_n_frames": int(test_mask.sum()),
            "horizons": {},
        }
        row = f"{fold_clip:>10}"
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
            record["horizons"][str(k)] = {
                "n_test": int(len(yt)),
                "n_pos": n_pos,
                "excluded": excluded,
                "auroc": auroc,
                "auprc": auprc,
            }
            row += f"{auroc:>12.4f}{auprc:>12.4f}"
        fold_records.append(record)
        print(row)

    print()
    print("=== Early-stopping diagnostics per fold ===")
    print(f"{'fold(clip)':>10}{'val_clips':>14}{'best_epoch':>12}{'stopped_at':>12}")
    for r in fold_records:
        print(f"{r['fold_clip']:>10}{str(r['val_clips']):>14}{r['best_epoch']:>12}{r['stopped_epoch']:>12}")

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

    aggregate_full = {str(k): _agg(fold_records, k) for k in HORIZONS}
    excluded_by_horizon = {
        str(k): [r["fold_clip"] for r in fold_records if r["horizons"][str(k)]["excluded"]] for k in HORIZONS
    }

    common_fold_clips = [
        r["fold_clip"] for r in fold_records if all(not r["horizons"][str(k)]["excluded"] for k in HORIZONS)
    ]
    common_fold_records = [r for r in fold_records if r["fold_clip"] in common_fold_clips]
    aggregate_common = {str(k): _agg(common_fold_records, k) for k in HORIZONS}

    print()
    print("=== Leave-one-clip-out summary: FULL SET (mean +/- std, whichever folds are valid per horizon) ===")
    for k in HORIZONS:
        a = aggregate_full[str(k)]
        excl = excluded_by_horizon[str(k)]
        excl_note = f"  excluded folds: {excl}" if excl else ""
        print(
            f"horizon {k:>3}: AUROC = {a['auroc_mean']:.4f} +/- {a['auroc_std']:.4f}   "
            f"AUPRC = {a['auprc_mean']:.4f} +/- {a['auprc_std']:.4f}   "
            f"(n_folds={a['n_folds']}/{len(unique_clips)}){excl_note}"
        )

    print()
    print(f"=== Leave-one-clip-out summary: COMMON FOLD SUBSET (folds valid at ALL horizons: {common_fold_clips}) ===")
    for k in HORIZONS:
        a = aggregate_common[str(k)]
        print(
            f"horizon {k:>3}: AUROC = {a['auroc_mean']:.4f} +/- {a['auroc_std']:.4f}   "
            f"AUPRC = {a['auprc_mean']:.4f} +/- {a['auprc_std']:.4f}   "
            f"(n_folds={a['n_folds']}/{len(common_fold_clips)})"
        )

    payload = {
        "model_config": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "horizons": HORIZONS,
            "inner_val_clips": INNER_VAL_CLIPS,
            "patience": PATIENCE,
        },
        "note_scene": "No clip-to-scene (lawn/indoor/plaza) mapping exists in any saved artifact; 'scene' is null for every fold.",
        "folds": fold_records,
        "common_fold_subset": common_fold_clips,
        "aggregate_full_set": aggregate_full,
        "aggregate_full_set_excluded_folds": excluded_by_horizon,
        "aggregate_common_subset": aggregate_common,
    }
    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
