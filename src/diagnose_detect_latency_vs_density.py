"""Checks whether YOLO detect() latency correlates with people-per-frame
(crowd density) rather than input resolution.

Motivated by benchmark_latency.py's CPU results showing NON-monotonic
detect times across resolutions (109 / 153 / 87 ms mean at 320x240 / 640x480
/ 856x480). That's expected in one sense -- imgsz is fixed at 640 regardless
of input frame size, so raw inference compute shouldn't scale with
resolution -- but NMS and ByteTrack association cost DOES scale with
detection count, which can vary between resolutions if resizing changes
which small/borderline detections clear the confidence threshold. This
reuses benchmark_latency.run_one_config() directly (same detect/track code,
no duplicated loop) across all three resolutions and correlates detect_ms
against n_persons pooled over all of them.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.stats import pearsonr

from src.benchmark_latency import RESOLUTIONS, WARMUP_FRAMES, run_one_config
from src.stream_processor import DEFAULT_LSTM_PATH, DEFAULT_YOLO_PATH

VIDEO_PATH = Path("data/raw/umn/Crowd-Activity-All.avi")
MEASURED_FRAMES = 200


def main() -> None:
    device = torch.device("cpu")
    all_detect_ms: List[float] = []
    all_n_persons: List[float] = []

    for width, height in RESOLUTIONS:
        timings = run_one_config(
            VIDEO_PATH, width, height, device, DEFAULT_YOLO_PATH, DEFAULT_LSTM_PATH,
            WARMUP_FRAMES + MEASURED_FRAMES,
        )
        detect_ms = timings["detect"]
        n_persons = timings["n_persons"]
        r_res, _ = pearsonr(n_persons, detect_ms) if len(set(n_persons)) > 1 else (float("nan"), None)
        print(
            f"{width}x{height}: mean detect={np.mean(detect_ms):.1f}ms  "
            f"mean n_persons={np.mean(n_persons):.2f}  "
            f"within-resolution r(detect_ms, n_persons)={r_res:.3f}"
        )
        all_detect_ms.extend(detect_ms)
        all_n_persons.extend(n_persons)

    r, p = pearsonr(all_n_persons, all_detect_ms)
    print(f"\nPooled across {len(RESOLUTIONS)} resolutions, n={len(all_detect_ms)} frames")
    print(f"Pearson r(detect_ms, n_persons) = {r:.3f}  (p={p:.2e})")

    n_arr = np.array(all_n_persons)
    d_arr = np.array(all_detect_ms)
    print("\nn_persons -> mean detect_ms (n frames) [pooled across all resolutions]")
    for count in sorted(set(n_arr.tolist())):
        mask = n_arr == count
        print(f"  {int(count):>3}: {d_arr[mask].mean():>8.2f} ms  (n={int(mask.sum())})")


if __name__ == "__main__":
    main()
