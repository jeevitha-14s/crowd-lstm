"""Diagnostic: per-onset breakdown and lead-window sweep for the pre-onset
precursor check. Answers two questions on data already on disk (no rerun):
1. Is each feature's lead-up gap robust across all 11 onsets, or driven by
   a couple of outlier clips?
2. Does the gap grow as the lead window shrinks toward onset (a real ramp),
   giving an empirical answer for which forecast horizons are worth training.
"""

import numpy as np
import pandas as pd

FEATURES_PATH = "data/features/umn_features.npy"
META_PATH = "data/features/umn_meta.csv"
GROUNDTRUTH_PATH = "data/umn_groundtruth.csv"

ONSETS = [525, 1330, 1806, 2605, 3219, 3938, 4807, 5422, 6195, 6883, 7700]
LEAD_WINDOWS = [30, 60, 90, 120, 180]
BASELINE_END_OFFSET = 270  # frames before onset where the fixed baseline window ends
BASELINE_LEN = 60
FEATURE_NAMES = ["normalized_count", "flow_magnitude", "direction_entropy", "stop_ratio"]


def main() -> None:
    features = np.load(FEATURES_PATH)
    gt = pd.read_csv(GROUNDTRUTH_PATH)
    labels = gt["abnormal"].values
    meta = pd.read_csv(META_PATH)
    clip_ids = meta["clip_id"].values

    clip_ranges = []
    for cid in sorted(np.unique(clip_ids)):
        idx = np.where(clip_ids == cid)[0]
        clip_ranges.append((int(idx.min()), int(idx.max()) + 1))

    def feature_mean(frame_range, feat_idx):
        vals = features[list(frame_range), feat_idx::4]  # (n_frames, 16 zones)
        return float(vals.mean())

    # Fixed baseline window per onset, positioned safely before the largest
    # lead window tested (180) + a buffer, so it never overlaps any variant.
    baselines = []  # (cs, ce, baseline_start, baseline_end) per onset
    for onset, (cs, ce) in zip(ONSETS, clip_ranges):
        assert cs <= onset < ce
        baseline_end = onset - BASELINE_END_OFFSET
        baseline_start = baseline_end - BASELINE_LEN
        if baseline_start < cs:
            print(f"WARNING: baseline window for onset {onset} clipped at clip start")
            baseline_start = cs
        assert all(labels[f] == 0 for f in range(baseline_start, baseline_end))
        baselines.append((cs, ce, baseline_start, baseline_end))

    # ---- Check 1: per-onset sign breakdown at the original W=60 ----
    print("=== Per-onset sign of (lead_mean - baseline_mean), lead window=60 ===")
    header = f"{'onset':>6}" + "".join(f"{name:>19}" for name in FEATURE_NAMES)
    print(header)
    sign_counts = {name: {"+": 0, "-": 0} for name in FEATURE_NAMES}
    for onset, (cs, ce, bs, be) in zip(ONSETS, baselines):
        lead_start = max(onset - 60, cs)
        row = f"{onset:>6}"
        for i, name in enumerate(FEATURE_NAMES):
            baseline_mean = feature_mean(range(bs, be), i)
            lead_mean = feature_mean(range(lead_start, onset), i)
            gap = lead_mean - baseline_mean
            sign = "+" if gap > 0 else "-"
            sign_counts[name][sign] += 1
            row += f"{gap:>15.4f}{sign:>4}"
        print(row)

    print("\nSign counts across 11 onsets (W=60):")
    for name in FEATURE_NAMES:
        c = sign_counts[name]
        print(f"  {name:20s} +:{c['+']:>2}  -:{c['-']:>2}")

    # ---- Check 2: lead-window sweep ----
    print("\n=== Lead-window sweep: pooled gap and onset-level sign counts ===")
    print(f"{'feature':20s}{'window':>8}{'pooled_gap':>13}{'n_pos':>7}{'n_neg':>7}")
    for i, name in enumerate(FEATURE_NAMES):
        for W in LEAD_WINDOWS:
            lead_frames_all = []
            baseline_frames_all = []
            pos, neg = 0, 0
            for onset, (cs, ce, bs, be) in zip(ONSETS, baselines):
                lead_start = max(onset - W, cs)
                if lead_start < cs:
                    print(f"WARNING: lead window W={W} clipped at clip start for onset {onset}")
                lead_frames = list(range(lead_start, onset))
                baseline_frames = list(range(bs, be))
                lead_frames_all.extend(lead_frames)
                baseline_frames_all.extend(baseline_frames)

                onset_gap = feature_mean(lead_frames, i) - feature_mean(baseline_frames, i)
                if onset_gap > 0:
                    pos += 1
                else:
                    neg += 1

            pooled_gap = feature_mean(lead_frames_all, i) - feature_mean(baseline_frames_all, i)
            print(f"{name:20s}{W:>8}{pooled_gap:>13.4f}{pos:>7}{neg:>7}")


if __name__ == "__main__":
    main()
