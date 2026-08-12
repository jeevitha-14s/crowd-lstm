"""Concentration diagnostic for ShanghaiTech CLEAN_PRE positives (the
genuine-forecasting subset from window_class in shanghaitech_forecast.npz).

CLEAN_PRE counts are small and grow with horizon (opposite of the raw
positive rate), so before committing to a leave-one-scene-out protocol we
need to know whether they're spread across videos/scenes or concentrated
in a handful -- concentration here is exactly what produced UMN's high
per-fold variance (see leakage_and_diagnostics / umn_task4_results memory).
"""

from pathlib import Path
from typing import List

import numpy as np

FORECAST_PATH = Path("data/shanghaitech_forecast.npz")
CLEAN_PRE = 1
FORECASTING_HORIZONS = [24, 48, 72]  # k=0 excluded: CLEAN_PRE is structurally 0 there


def report_horizon(
    k: int,
    j: int,
    window_class: np.ndarray,
    video_ids: np.ndarray,
    scene_ids: np.ndarray,
    all_scenes: List[str],
) -> None:
    mask = window_class[:, j] == CLEAN_PRE
    total = int(mask.sum())
    print(f"\n{'=' * 60}\nhorizon {k}  (CLEAN_PRE positives = {total})\n{'=' * 60}")
    if total == 0:
        print("  no CLEAN_PRE positives at this horizon")
        return

    scenes = scene_ids[mask]
    videos = video_ids[mask]

    print("\n1. Per scene:")
    scene_vals, scene_counts = np.unique(scenes, return_counts=True)
    order = np.argsort(-scene_counts)
    for i in order:
        print(f"    scene {scene_vals[i]}: {scene_counts[i]}")
    print(f"    scenes contributing: {len(scene_vals)} of {len(all_scenes)}")

    print("\n2. Per video (descending):")
    video_vals, video_counts = np.unique(videos, return_counts=True)
    vorder = np.argsort(-video_counts)
    for i in vorder:
        print(f"    {video_vals[i]}: {video_counts[i]}")
    print(f"    videos contributing: {len(video_vals)}")

    print("\n3. Concentration:")
    sorted_counts = video_counts[vorder]
    for top_n in (1, 3, 5):
        top_sum = int(sorted_counts[:top_n].sum())
        print(f"    top {top_n} video(s): {top_sum}/{total} = {top_sum / total:.1%}")
    top_scene_count = int(scene_counts[order[0]])
    print(
        f"    largest scene ({scene_vals[order[0]]}): {top_scene_count}/{total} "
        f"= {top_scene_count / total:.1%}"
    )

    print("\n4. Distinct contributing videos per scene:")
    for i in order:
        s = scene_vals[i]
        n_videos_in_scene = len(np.unique(videos[scenes == s]))
        print(f"    scene {s}: {n_videos_in_scene} video(s), {scene_counts[i]} positives")

    zero_scenes = sorted(set(all_scenes) - set(scene_vals.tolist()))
    print(f"\nScenes with ZERO CLEAN_PRE positives at horizon {k}: {zero_scenes}")
    print(
        f"  (these {len(zero_scenes)} scenes cannot be evaluable LOSO folds for the "
        "forecasting metric, even if they contain anomalies -- no positive class)"
    )


def main() -> None:
    d = np.load(FORECAST_PATH)
    window_class = d["window_class"]
    horizons = d["window_class_horizons"].tolist()
    video_ids = d["video_ids"]
    scene_ids = d["scene_ids"]
    all_scenes = sorted(set(scene_ids.tolist()))

    for k in FORECASTING_HORIZONS:
        j = horizons.index(k)
        report_horizon(k, j, window_class, video_ids, scene_ids, all_scenes)


if __name__ == "__main__":
    main()
