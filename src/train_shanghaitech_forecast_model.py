"""ShanghaiTech shared-backbone LSTM forecasting/nowcasting model, evaluated
with three separate result groups because the benchmark was built for
detection, not forecasting: only scenes {01,03,05,07} have enough clean
pre-onset (CLEAN_PRE) samples across enough videos to support a defensible
scene-level forecasting AUROC (see leakage_and_diagnostics / window_class in
build_shanghaitech_forecast.py for how CLEAN_PRE is derived).

  1. HEADLINE   -- LOSO over {01,03,05,07}: full-sample AND CLEAN_PRE+NEGATIVE
                   restricted AUROC/AUPRC, k=0/24/48/72. k=0 is reported as a
                   nowcasting baseline (comparable to published frame-level
                   VAD numbers) with its restricted column marked n/a --
                   CLEAN_PRE is structurally empty at k=0 on any dataset,
                   since the target frame IS the last frame of the window.
  2. THIN SCENES -- same protocol over {04,06,08,12}, reported separately,
                   never averaged into the headline result: these scenes have
                   only 1-2 contributing videos for CLEAN_PRE, so a fold here
                   is closer to a single-clip estimate than a scene estimate.
  3. SPECIFICITY -- the 4 headline models run over {02,09,10,11,13} (zero
                   CLEAN_PRE at every horizon) to check false-positive rate
                   on scenes structurally excluded from the forecasting
                   metric.

Training composition is NOT restricted to evaluation scenes: every fold
trains on all 199 videos minus its own held-out scene, explicitly including
the thin scenes and the non-evaluable scenes -- they are legitimate negative/
overlap-positive training signal, only excluded from being test folds for
specific structural reasons (see NON_EVALUABLE_SCENES below).

Inner-validation for early stopping is a video-level (never row-level --
adjacent windows share nearly all their history frames), scene-stratified
~15% holdout drawn from EVERY scene in the fold's training pool (see
INNER_VAL_FRACTION, select_inner_val_videos). This spends some CLEAN_PRE
positives on model selection, which is a deliberate trade: an earlier design
drew inner-val ONLY from the 5 non-evaluable scenes, on the reasoning that no
CLEAN_PRE positive should ever be spent on selection -- but 3 of those 5
scenes are anomaly-free, so the entire early-stopping signal came from ~530
OVERLAP positives in just 2 atypical scenes (02, 10). That signal was too
weak and too narrow to select a model on: best_epoch=0 on multiple headline
folds (the model never once improved on it across the full patience window).
A model selected on a broken signal produces no usable number at all, so
scene diversity in inner-val takes priority over hoarding CLEAN_PRE for
evaluation. Selection is still computed only over the k=24/48/72 heads
(k=0 is a different task, nowcasting, and must not steer forecasting
selection).
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

DATASET_PATH = "data/shanghaitech_forecast.npz"
OUT_DIR = Path("outputs/reports")
HORIZONS = [0, 24, 48, 72]
FORECASTING_HORIZONS = [24, 48, 72]  # k=0 excluded: nowcasting, not forecasting

HEADLINE_SCENES = ["01", "03", "05", "07"]
THIN_SCENES = ["04", "06", "08", "12"]
NON_EVALUABLE_SCENES = ["02", "09", "10", "11", "13"]

WC_NEGATIVE, WC_CLEAN_PRE, WC_OVERLAP, WC_POST = 0, 1, 2, 3

INPUT_SIZE = 16
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
LR = 1e-3
BATCH_SIZE = 64
SEED = 42

MAX_EPOCHS = 150
PATIENCE = 20
SMOOTH_WINDOW = 5
FPR_THRESHOLD = 0.5
INNER_VAL_FRACTION = 0.15  # ~15% of training videos per scene, held out for early stopping


class CrowdRiskLSTM(nn.Module):
    """Single LSTM backbone -> one Linear head per forecast horizon."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
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
        shared = self.shared_dropout(hn[-1])
        return {k: head(shared).squeeze(-1) for k, head in self.heads.items()}


def compute_pos_weight(y: np.ndarray) -> float:
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def safe_auroc_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan")
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


def mean_val_auroc(
    model: CrowdRiskLSTM, X_val_t: torch.Tensor, y_val: Dict[int, np.ndarray]
) -> float:
    """Early-stopping signal, averaged over FORECASTING_HORIZONS only (k=0 is
    nowcasting, a different task, and must not steer forecasting selection)."""
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_probs = {k: torch.sigmoid(val_preds[str(k)]).cpu().numpy() for k in HORIZONS}
    head_aurocs = []
    for k in FORECASTING_HORIZONS:
        yv = y_val[k]
        if yv.sum() == 0 or yv.sum() == len(yv):
            continue
        head_aurocs.append(roc_auc_score(yv, val_probs[k]))
    return float(np.mean(head_aurocs)) if head_aurocs else float("nan")


def clean_pre_counts_for_scene(
    scene: str, scene_ids: np.ndarray, window_class: np.ndarray
) -> Dict[int, int]:
    mask = scene_ids == scene
    counts = {}
    for j, k in enumerate(HORIZONS):
        if k == 0:
            continue
        counts[k] = int((window_class[mask, j] == WC_CLEAN_PRE).sum())
    return counts


def fold_json_path(group_slug: str, scene: str) -> Path:
    return OUT_DIR / f"shanghaitech_fold_{group_slug}_{scene}.json"


def fold_model_path(group_slug: str, scene: str) -> Path:
    return OUT_DIR / f"shanghaitech_model_{group_slug}_{scene}.pt"


def save_fold_result(
    group_slug: str,
    scene: str,
    model: CrowdRiskLSTM,
    full: Dict[int, Tuple[float, float]],
    restricted: Dict[int, Optional[Tuple[float, float]]],
    diag: Dict[str, int],
) -> None:
    """Write the model checkpoint BEFORE the JSON result file, so the JSON's
    existence can serve as the single "this fold is done" marker on resume --
    an interrupted run never leaves a JSON pointing at a missing checkpoint."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_model_path(group_slug, scene))
    payload = {
        "scene": scene,
        "full": {k: list(v) for k, v in full.items()},
        "restricted": {k: (list(v) if v is not None else None) for k, v in restricted.items()},
        "diag": diag,
    }
    with open(fold_json_path(group_slug, scene), "w") as f:
        json.dump(payload, f, indent=2)


def load_fold_result(
    group_slug: str, scene: str, device: torch.device
) -> Tuple[CrowdRiskLSTM, Dict[int, Tuple[float, float]], Dict[int, Optional[Tuple[float, float]]], Dict[str, int]]:
    json_path = fold_json_path(group_slug, scene)
    model_path = fold_model_path(group_slug, scene)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{json_path} exists but {model_path} does not -- inconsistent checkpoint state. "
            f"Delete {json_path} and rerun to retrain this fold."
        )
    with open(json_path) as f:
        payload = json.load(f)
    full = {int(k): tuple(v) for k, v in payload["full"].items()}
    restricted = {
        int(k): (tuple(v) if v is not None else None) for k, v in payload["restricted"].items()
    }
    model = CrowdRiskLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, full, restricted, payload["diag"]


def select_inner_val_videos(
    scene_ids_train_all: np.ndarray,
    video_ids_train_all: np.ndarray,
    frac: float,
    rng: np.random.Generator,
) -> set:
    """Video-level (never row-level -- adjacent windows share nearly all their
    history frames) stratified sample: from EVERY scene present in this fold's
    training pool, hold out ~frac of its videos for inner-val. The per-scene
    floor of 1 guarantees inner-val always spans every training scene, not
    just whichever scenes happen to have the most videos -- this is what
    fixes best_epoch=0: the prior design drew inner-val only from the 5
    non-evaluable scenes, so early-stopping signal came almost entirely from
    ~530 OVERLAP positives in just 2 of them (02, 10), too weak and atypical
    a signal to select a model on."""
    val_videos: set = set()
    for scene in np.unique(scene_ids_train_all):
        videos_in_scene = np.unique(video_ids_train_all[scene_ids_train_all == scene])
        n_val = max(1, round(frac * len(videos_in_scene)))
        n_val = min(n_val, len(videos_in_scene))
        chosen = rng.choice(videos_in_scene, size=n_val, replace=False)
        val_videos.update(chosen.tolist())
    return val_videos


def train_one_fold(
    X_train_all: np.ndarray,
    y_train_all: Dict[int, np.ndarray],
    scene_ids_train_all: np.ndarray,
    video_ids_train_all: np.ndarray,
    fold_scene: str,
    device: torch.device,
) -> Tuple[CrowdRiskLSTM, Dict[str, int]]:
    rng = np.random.default_rng(SEED + int(fold_scene))
    val_videos = select_inner_val_videos(
        scene_ids_train_all, video_ids_train_all, INNER_VAL_FRACTION, rng
    )
    inner_val_mask = np.isin(video_ids_train_all, list(val_videos))
    inner_train_mask = ~inner_val_mask
    val_scenes = sorted(set(np.unique(scene_ids_train_all[inner_val_mask]).tolist()))
    train_scenes_present = sorted(set(np.unique(scene_ids_train_all).tolist()))
    # Explicit, so this can't silently regress back to the single-scene-pool
    # bug: inner-val must span every scene in this fold's training pool.
    assert val_scenes == train_scenes_present, (
        f"inner-val scenes {val_scenes} != training-pool scenes {train_scenes_present}"
    )

    X_inner = X_train_all[inner_train_mask]
    y_inner = {k: y_train_all[k][inner_train_mask] for k in HORIZONS}
    X_val = X_train_all[inner_val_mask]
    y_val = {k: y_train_all[k][inner_val_mask] for k in HORIZONS}

    torch.manual_seed(SEED)
    model = CrowdRiskLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fns = {
        k: nn.BCEWithLogitsLoss(pos_weight=torch.tensor(compute_pos_weight(y_inner[k]), device=device))
        for k in HORIZONS
    }

    X_inner_t = torch.tensor(X_inner, dtype=torch.float32, device=device)
    y_inner_t = {k: torch.tensor(y_inner[k], dtype=torch.float32, device=device) for k in HORIZONS}
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

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
    diag = {
        "n_inner_train": int(n),
        "n_inner_val": int(X_val_t.shape[0]),
        "n_val_scenes": len(val_scenes),
        "n_val_videos": len(val_videos),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
    }
    return model, diag


def predict(model: CrowdRiskLSTM, X: np.ndarray, device: torch.device) -> Dict[int, np.ndarray]:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(X_t)
        return {k: torch.sigmoid(preds[str(k)]).cpu().numpy() for k in HORIZONS}


def evaluate_fold(
    model: CrowdRiskLSTM,
    X_test: np.ndarray,
    y_test: Dict[int, np.ndarray],
    window_class_test: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Optional[Tuple[float, float]]]]:
    probs = predict(model, X_test, device)
    full = {k: safe_auroc_auprc(y_test[k], probs[k]) for k in HORIZONS}
    restricted: Dict[int, Optional[Tuple[float, float]]] = {}
    for k in HORIZONS:
        if k == 0:
            restricted[k] = None  # structurally undefined: no CLEAN_PRE at k=0
            continue
        j = HORIZONS.index(k)
        mask = np.isin(window_class_test[:, j], [WC_NEGATIVE, WC_CLEAN_PRE])
        restricted[k] = safe_auroc_auprc(y_test[k][mask], probs[k][mask])
    return full, restricted


def run_loso_group(
    group_scenes: List[str],
    group_name: str,
    group_slug: str,
    X: np.ndarray,
    y: Dict[int, np.ndarray],
    scene_ids: np.ndarray,
    video_ids: np.ndarray,
    window_class: np.ndarray,
    device: torch.device,
) -> Tuple[List[Dict], Dict[str, CrowdRiskLSTM]]:
    print(f"\n{'=' * 78}\n{group_name}\n{'=' * 78}")
    fold_results = []
    models_by_scene: Dict[str, CrowdRiskLSTM] = {}

    for scene in group_scenes:
        test_mask = scene_ids == scene
        train_mask = ~test_mask

        cp_counts = clean_pre_counts_for_scene(scene, scene_ids, window_class)

        if fold_json_path(group_slug, scene).exists():
            print(
                f"\nfold: held-out scene {scene}  <<< cached result found, skipping training "
                f"({fold_json_path(group_slug, scene)})"
            )
            model, full, restricted, diag = load_fold_result(group_slug, scene, device)
            models_by_scene[scene] = model
            fold_results.append({"scene": scene, "full": full, "restricted": restricted, "diag": diag})
        else:
            train_scenes = set(np.unique(scene_ids[train_mask]).tolist())
            missing = set(NON_EVALUABLE_SCENES) - train_scenes
            assert not missing, f"non-evaluable scenes missing from training pool: {missing}"
            n_non_eval_train = int(np.isin(scene_ids[train_mask], NON_EVALUABLE_SCENES).sum())

            print(
                f"\nfold: held-out scene {scene}  train={int(train_mask.sum())} "
                f"(incl. {n_non_eval_train} from {NON_EVALUABLE_SCENES})  "
                f"test={int(test_mask.sum())}  CLEAN_PRE positives in this scene: {cp_counts}"
            )

            X_train_all = X[train_mask]
            y_train_all = {k: y[k][train_mask] for k in HORIZONS}
            scene_ids_train_all = scene_ids[train_mask]
            video_ids_train_all = video_ids[train_mask]

            model, diag = train_one_fold(
                X_train_all, y_train_all, scene_ids_train_all, video_ids_train_all, scene, device
            )
            models_by_scene[scene] = model

            X_test = X[test_mask]
            y_test = {k: y[k][test_mask] for k in HORIZONS}
            window_class_test = window_class[test_mask]
            full, restricted = evaluate_fold(model, X_test, y_test, window_class_test, device)
            fold_results.append({"scene": scene, "full": full, "restricted": restricted, "diag": diag})
            save_fold_result(group_slug, scene, model, full, restricted, diag)
            print(f"  saved checkpoint + result to {fold_json_path(group_slug, scene)}")

        print(
            f"  early stop: best_epoch={diag['best_epoch']} stopped_at={diag['stopped_epoch']} "
            f"inner_train={diag['n_inner_train']} inner_val={diag['n_inner_val']} "
            f"(from {diag['n_val_videos']} videos across {diag['n_val_scenes']} scenes)"
        )
        for k in HORIZONS:
            a_f, p_f = full[k]
            r = restricted[k]
            r_str = "AUROC=n/a    AUPRC=n/a" if r is None else f"AUROC={r[0]:.4f}  AUPRC={r[1]:.4f}"
            print(f"    k={k:>3}: full AUROC={a_f:.4f} AUPRC={p_f:.4f}  |  clean_pre+neg {r_str}")

    print(f"\n--- {group_name}: aggregate (mean +/- std across {len(group_scenes)} folds) ---")
    for k in HORIZONS:
        full_auroc = np.array([fr["full"][k][0] for fr in fold_results])
        full_auprc = np.array([fr["full"][k][1] for fr in fold_results])
        print(
            f"  k={k:>3}  full: AUROC={np.nanmean(full_auroc):.4f}+/-{np.nanstd(full_auroc):.4f}  "
            f"AUPRC={np.nanmean(full_auprc):.4f}+/-{np.nanstd(full_auprc):.4f}"
        )
        if k == 0:
            print("          clean_pre+neg: n/a (k=0 is nowcasting -- CLEAN_PRE is empty by construction)")
        else:
            r_auroc = np.array([fr["restricted"][k][0] for fr in fold_results])
            r_auprc = np.array([fr["restricted"][k][1] for fr in fold_results])
            print(
                f"          clean_pre+neg: AUROC={np.nanmean(r_auroc):.4f}+/-{np.nanstd(r_auroc):.4f}  "
                f"AUPRC={np.nanmean(r_auprc):.4f}+/-{np.nanstd(r_auprc):.4f}"
            )

    return fold_results, models_by_scene


def run_specificity_check(
    models_by_scene: Dict[str, CrowdRiskLSTM],
    X: np.ndarray,
    y: Dict[int, np.ndarray],
    scene_ids: np.ndarray,
    device: torch.device,
    threshold: float = FPR_THRESHOLD,
) -> Dict[int, List[float]]:
    print(f"\n{'=' * 78}\n3. SPECIFICITY CHECK: HEADLINE models over {NON_EVALUABLE_SCENES}\n{'=' * 78}")
    print(
        "Note: scenes 02 and 10 DO contain anomalous videos (02_0161; 10_0038, 10_0042).\n"
        "They are excluded from forecasting evaluation only because onset lands too early\n"
        "for 87 frames of clean pre-onset history -- not because they are anomaly-free.\n"
        "FPR below is computed only over frames with true label 0 at each horizon, so a\n"
        "flag on a genuinely negative frame still counts against specificity; a flag on the\n"
        "true positive frames of those two videos is correct behaviour, not a false positive."
    )

    mask = np.isin(scene_ids, NON_EVALUABLE_SCENES)
    X_spec = X[mask]
    y_spec = {k: y[k][mask] for k in HORIZONS}
    print(f"\nspecificity pool: {int(mask.sum())} samples across scenes {NON_EVALUABLE_SCENES}")

    header = f"{'fold(held-out)':>16}" + "".join(f"{'FPR_k' + str(k):>12}" for k in HORIZONS)
    print(header)
    per_fold_fpr: Dict[int, List[float]] = {k: [] for k in HORIZONS}
    for scene, model in models_by_scene.items():
        probs = predict(model, X_spec, device)
        row = f"{scene:>16}"
        for k in HORIZONS:
            neg_mask = y_spec[k] == 0
            n_neg = int(neg_mask.sum())
            fpr = float((probs[k][neg_mask] >= threshold).mean()) if n_neg > 0 else float("nan")
            per_fold_fpr[k].append(fpr)
            row += f"{fpr:>12.4f}"
        print(row)

    print(f"\n--- mean +/- std FPR across {len(models_by_scene)} headline-fold models (threshold={threshold}) ---")
    for k in HORIZONS:
        arr = np.array(per_fold_fpr[k])
        print(f"  k={k:>3}: FPR = {arr.mean():.4f} +/- {arr.std():.4f}")

    return per_fold_fpr


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(DATASET_PATH)
    X = data["X"]
    scene_ids = data["scene_ids"]
    video_ids = data["video_ids"]
    window_class = data["window_class"]
    window_class_horizons = data["window_class_horizons"].tolist()
    assert window_class_horizons == HORIZONS, (
        f"window_class column order {window_class_horizons} != HORIZONS {HORIZONS}"
    )
    y = {k: data[f"y_{k}"] for k in HORIZONS}

    print(f"Device: {device}")
    print(f"Model: input_size={INPUT_SIZE} hidden={HIDDEN_SIZE} layers={NUM_LAYERS} dropout={DROPOUT}")
    print(f"HEADLINE_SCENES={HEADLINE_SCENES}  THIN_SCENES={THIN_SCENES}  NON_EVALUABLE_SCENES={NON_EVALUABLE_SCENES}")
    print(
        "Every fold trains on all scenes except its own held-out scene, including all "
        "NON_EVALUABLE_SCENES and every other evaluable/thin scene -- training is never "
        f"restricted to the evaluation scenes. Inner-validation for early stopping holds out "
        f"~{INNER_VAL_FRACTION:.0%} of videos from EVERY scene in the fold's training pool "
        "(video-level, stratified by scene), not just the non-evaluable scenes -- narrowing "
        "the validation signal to 2-3 atypical scenes produced best_epoch=0 (no improvement "
        "ever recorded) on multiple headline folds."
    )

    headline_results, headline_models = run_loso_group(
        HEADLINE_SCENES,
        "1. HEADLINE: leave-one-scene-out over {01,03,05,07}",
        "headline",
        X, y, scene_ids, video_ids, window_class, device,
    )

    thin_results, _ = run_loso_group(
        THIN_SCENES,
        "2. THIN SCENES: leave-one-scene-out over {04,06,08,12} (separate table, not averaged into headline)",
        "thin",
        X, y, scene_ids, video_ids, window_class, device,
    )

    specificity_fpr = run_specificity_check(headline_models, X, y, scene_ids, device)

    summary_path = OUT_DIR / "shanghaitech_results_summary.json"
    summary = {
        "headline": [
            {"scene": fr["scene"], "full": {k: list(v) for k, v in fr["full"].items()},
             "restricted": {k: (list(v) if v is not None else None) for k, v in fr["restricted"].items()}}
            for fr in headline_results
        ],
        "thin": [
            {"scene": fr["scene"], "full": {k: list(v) for k, v in fr["full"].items()},
             "restricted": {k: (list(v) if v is not None else None) for k, v in fr["restricted"].items()}}
            for fr in thin_results
        ],
        "specificity_fpr_per_headline_model": specificity_fpr,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFinal combined summary written to {summary_path}")


if __name__ == "__main__":
    main()
