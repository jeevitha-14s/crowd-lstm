"""Why ShanghaiTech scene 05 scores below chance at every horizon.

The paper attributes the failure to a mismatch in the kind of event: frame
inspection identified scene 05's three anomalies as small-group physical
altercations of two to five people, and the stated reading was that such
events "do not perturb a crowd-level aggregate and can move it in the wrong
direction". This script tests the second half of that claim, and does not
support it -- the aggregate is perturbed, detectably, in two of the three
videos.

What it computes, per anomalous video in the scene:

  1. the shift in each feature group during the anomaly versus the rest of
     that same video, so the direction of the response is visible per video
     rather than only in the scene-level average;
  2. a naive AUROC from the mean of per-video z-scored features -- a trivial
     score with no model, no training and no temporal context;
  3. the same naive score pooled across the scene, which is what the
     scene-level metric aggregates over.

The result is that 05_0017 and 05_0019 are separable by the trivial score
(0.879 and 0.828) while 05_0023 is inverted (0.390), and 05_0023 contributes
the majority of the scene's positive frames because its anomaly spans 55% of
its duration. Pooled, the trivial score still reaches 0.631 -- well above the
model's reported 0.248 -- so the signal is not absent from the features. The
model's failure to exceed a feature average indicates a learned mapping that
does not transfer to this scene, which is a different and weaker claim than
the one the paper currently makes.

Caveat on the proxy: it z-scores within each video, whereas the model sees
features normalised by the running statistics applied during extraction. It
is indicative of what the features contain, not a stand-in for the model.

Reads only existing artifacts; writes outputs/reports/scene05_diagnostic.json.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

FEATURES_DIR = Path("data/features/shanghaitech")
MANIFEST = Path("data/shanghaitech_meta.json")
RESULTS = Path("outputs/reports/shanghaitech_results_summary.json")
OUT = Path("outputs/reports/scene05_diagnostic.json")

SCENE = "05"
FEATURE_NAMES = ["normalized_count", "flow_magnitude", "direction_entropy", "stop_ratio"]


def load_anomalous_videos(scene: str) -> List[Dict]:
    """Anomalous entries for one scene, recovered by regex.

    The manifest is a single JSON line that has been truncated at some point in
    its history, so json.load() on the whole file fails. Each record is
    self-contained and brace-balanced, so extracting them individually
    recovers every entry that is intact.
    """
    raw = MANIFEST.read_text()
    pattern = rf'\{{[^{{}}]*"vid"\s*:\s*"{scene}_[^"]*"[^{{}}]*\}}'
    entries = [json.loads(m) for m in re.findall(pattern, raw)]
    return [e for e in entries if e.get("anom")]


def reported_auroc(scene: str) -> Dict[str, float]:
    if not RESULTS.exists():
        return {}
    summary = json.loads(RESULTS.read_text())
    for group in ("headline", "thin"):
        for entry in summary.get(group, []):
            if entry["scene"] == scene:
                return {k: v[0] for k, v in entry["full"].items() if v}
    return {}


def main() -> None:
    videos = load_anomalous_videos(SCENE)
    if not videos:
        raise SystemExit(f"no anomalous videos recovered for scene {SCENE}")

    total_positive = sum(v["end"] - v["start"] for v in videos)
    per_video: Dict[str, Dict] = {}
    pooled_scores: List[np.ndarray] = []
    pooled_labels: List[np.ndarray] = []

    print(f"SCENE {SCENE}: {len(videos)} anomalous videos\n")
    print(f"{'video':>9}{'frames':>8}{'anom':>7}{'% of pos':>10}{'% of vid':>10}{'naive AUROC':>13}")

    for entry in sorted(videos, key=lambda e: e["vid"]):
        vid, n, s, e = entry["vid"], entry["n"], entry["start"], entry["end"]
        F = np.load(FEATURES_DIR / f"{vid}.npy")[:n]
        y = np.zeros(len(F), dtype=int)
        y[s:e] = 1

        abnormal, normal = F[y == 1], F[y == 0]
        deltas = {
            name: float(abnormal[:, i::4].mean() - normal[:, i::4].mean())
            for i, name in enumerate(FEATURE_NAMES)
        }
        z = (F - F.mean(0)) / (F.std(0) + 1e-8)
        score = z.mean(1)
        auc = float(roc_auc_score(y, score))
        pooled_scores.append(score)
        pooled_labels.append(y)

        per_video[vid] = {
            "n_frames": int(n), "anomaly_start": int(s), "anomaly_end": int(e),
            "n_anomalous": int(e - s),
            "share_of_scene_positives": float((e - s) / total_positive),
            "share_of_own_duration": float((e - s) / n),
            "feature_delta_during_anomaly": deltas,
            "naive_auroc": auc,
        }
        print(f"{vid:>9}{n:>8}{e - s:>7}{100 * (e - s) / total_positive:>9.1f}%"
              f"{100 * (e - s) / n:>9.1f}%{auc:>13.3f}")

    pooled = float(roc_auc_score(np.concatenate(pooled_labels), np.concatenate(pooled_scores)))
    reported = reported_auroc(SCENE)

    print(f"\npooled naive AUROC across the scene: {pooled:.3f}")
    if reported:
        print("reported model AUROC:            " +
              "  ".join(f"k={k}: {v:.3f}" for k, v in sorted(reported.items(), key=lambda t: int(t[0]))))

    print("\nfeature delta during anomaly (negative = falls):")
    print(f"{'video':>9}" + "".join(f"{n[:11]:>14}" for n in FEATURE_NAMES))
    for vid, d in per_video.items():
        print(f"{vid:>9}" + "".join(f"{d['feature_delta_during_anomaly'][n]:>+14.4f}"
                                     for n in FEATURE_NAMES))

    agree = sum(1 for d in per_video.values()
                if all(v > 0 for v in d["feature_delta_during_anomaly"].values()))
    print(f"\n{agree}/{len(per_video)} videos have all four features rising during the anomaly; "
          f"the remainder fall, so the scene's response is not internally consistent.")

    payload = {
        "description": (
            "Per-video diagnostic for ShanghaiTech scene 05, which scores below chance at every "
            "horizon. Tests whether the features respond to the anomaly at all, using a trivial "
            "z-scored feature average with no model. They do, in two of three videos; the third "
            "is inverted and holds the majority of the scene's positive frames."
        ),
        "scene": SCENE,
        "per_video": per_video,
        "pooled_naive_auroc": pooled,
        "reported_model_auroc": reported,
        "proxy_caveat": (
            "The naive score z-scores within each video, while the model sees features normalised "
            "by running statistics applied during extraction. It indicates what the features "
            "contain; it is not a substitute for the model."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
