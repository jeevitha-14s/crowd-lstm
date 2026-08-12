# umn_forecast_lstm.pt -- deployment checkpoint, NOT an evaluation artifact

This model is trained on ALL 11 UMN clips, with no held-out fold. It exists
only for the latency benchmark (`src/benchmark_latency.py`) and the live
demo (`src/stream_processor.py`, `src/live_dashboard.py`).

**Every AUROC/AUPRC number reported in the paper comes from the 11-fold
leave-one-clip-out evaluation in `src/train_forecast_model.py`, never from
this checkpoint.** This model has seen every clip during training, so its
predictions on any UMN clip are not a valid held-out measurement of
anything. Do not compute or report any metric from this checkpoint.

## How the epoch count was chosen

Training on all 11 clips leaves no held-out validation signal for early
stopping. The epoch count below is the mean `best_epoch` across the 11 LOCO
folds from `train_forecast_model.py` (re-derived fresh by this script's
`derive_fixed_epoch_count()` every time it runs, never hardcoded), used as a
fixed, non-adaptive training budget:

| fold (held-out clip) | best_epoch |
|---|---|
| 0 | 18 |
| 1 | 6 |
| 2 | 13 |
| 3 | 11 |
| 4 | 31 |
| 5 | 17 |
| 6 | 18 |
| 7 | 33 |
| 8 | 21 |
| 9 | 54 |
| 10 | 19 |

mean best_epoch = 21.91 -> fixed_epochs used = 22

## Architecture

Shared LSTM backbone (hidden=64, 1 layer, dropout=0.2), 4 linear heads at
k=0/30/60/90 frames (0/1/2/3s at 30fps), input_size=64 (UMN's 4x4 spatial
grid FeatureExtractor output).
