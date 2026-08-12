"""Investigates a specific hypothesis for fold 0's inverted LOCO result
(AUROC 0.990 at k=0 vs 0.186 at k=60, on 70 positives -- the largest single
contributor to k=60's variance, and not obviously small-sample noise given
the magnitude and n).

HYPOTHESIS: FeatureExtractor's running normalizers (running_max_count,
running_max_speed, running_median_speed) reset to their floor values
(1.0/1.0/1.0) at every clip boundary and need frames to stabilize toward
that clip's real scale. build_forecast_dataset.py's first valid sample in
any clip has LOCAL offset 145 ((SEQ_LEN-1)*STRIDE) and its history window
reaches back to local offset 0 -- so every clip's early samples include the
cold-start period. If clip 0 specifically takes unusually long to stabilize
(e.g. because it opens with few/no tracked people, so running_max_count and
running_median_speed sit near their floor values for longer, inflating
normalized ratios computed against them), its pre-onset window would be
systematically miscalibrated relative to every other clip's, in a direction
that could plausibly explain a k=0-vs-k=60 sign inversion specific to this
fold.

This reprocesses only the first 200 frames of EACH of the 11 clips (seeking
directly to each clip's start rather than reprocessing the whole video) to
recover the running-normalizer trajectories, which are not saved anywhere
-- FeatureExtractor only exposes their current value as an instance
attribute during a live extract() call, and process_umn.py never logged
them.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.feature_extractor import FeatureExtractor

VIDEO_PATH = Path("data/raw/umn/Crowd-Activity-All.avi")
MODEL_PATH = Path("models/finetuned_yolov8s_cctv.pt")
OUT_JSON_PATH = Path("outputs/reports/umn_clip0_coldstart_diagnostic.json")

CLIP_STARTS = [0, 625, 1453, 1996, 2687, 3455, 4034, 4929, 5596, 6238, 6931]
ONSETS = [525, 1330, 1806, 2605, 3219, 3938, 4807, 5422, 6195, 6883, 7700]
N_FRAMES_TO_TRACK = 200
FIRST_VALID_SAMPLE_OFFSET = 145  # (SEQ_LEN-1)*STRIDE, matches build_forecast_dataset.py
STABILIZE_TOLERANCE = 0.05  # within 5% of this window's final value, and stays there


def stabilize_frame(series: np.ndarray) -> int:
    """First frame index after which `series` stays within STABILIZE_TOLERANCE
    of its own final value (within this 200-frame window) for the rest of the
    window. Returns len(series) if it never stabilizes within the window."""
    target = series[-1]
    if target == 0:
        return 0
    within = np.abs(series - target) <= STABILIZE_TOLERANCE * abs(target)
    for i in range(len(within)):
        if within[i:].all():
            return i
    return len(series)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Seek-accuracy sanity check: compare person count at the seeked first
    # frame of clip 1 against the already-known value in umn_meta.csv, so a
    # silent seek failure doesn't masquerade as a real finding.
    import csv as csv_mod

    known_counts = {}
    with open("data/features/umn_meta.csv", newline="") as f:
        for row in csv_mod.DictReader(f):
            known_counts[int(row["frame"])] = int(row["n_persons"])

    trajectories: Dict[int, List[List[float]]] = {}
    for clip_idx, start in enumerate(CLIP_STARTS):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        fe = FeatureExtractor(frame_width, frame_height)
        if hasattr(model.predictor, "trackers"):
            model.predictor.trackers[0].reset()

        traj: List[List[float]] = []
        for offset in range(N_FRAMES_TO_TRACK):
            ok, frame = cap.read()
            if not ok:
                break
            results = model.track(
                frame, imgsz=640, conf=0.25, classes=[0], tracker="bytetrack.yaml",
                persist=True, verbose=False, device=str(device),
            )
            detections: List[Dict[str, Any]] = []
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), tid in zip(xyxy, track_ids):
                    detections.append(
                        {
                            "track_id": int(tid),
                            "cx": float((x1 + x2) / 2),
                            "cy": float((y1 + y2) / 2),
                            "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        }
                    )
            if offset == 0:
                seeked_count = len(detections)
                known = known_counts.get(start)
                match = "OK" if known is not None and seeked_count == known else "MISMATCH"
                print(f"clip {clip_idx}: seek sanity check frame {start}: seeked_n={seeked_count} known_n={known} [{match}]")

            fe.extract(detections)
            traj.append([float(offset), fe.running_max_count, fe.running_max_speed, fe.running_median_speed])

        trajectories[clip_idx] = traj
        print(f"clip {clip_idx} (start={start}): captured {len(traj)} frames")

    print(f"\n{'clip':>5}{'onset_local':>12}{'@145_maxcnt':>13}{'@145_maxspd':>13}{'@145_medspd':>13}"
          f"{'final_maxcnt':>14}{'final_maxspd':>14}{'final_medspd':>14}")
    for clip_idx, traj in trajectories.items():
        arr = np.array(traj)
        onset_local = ONSETS[clip_idx] - CLIP_STARTS[clip_idx]
        if len(arr) > FIRST_VALID_SAMPLE_OFFSET:
            _, mc145, ms145, med145 = arr[FIRST_VALID_SAMPLE_OFFSET]
        else:
            mc145 = ms145 = med145 = float("nan")
        _, mcf, msf, medf = arr[-1]
        print(
            f"{clip_idx:>5}{onset_local:>12}{mc145:>13.3f}{ms145:>13.3f}{med145:>13.3f}"
            f"{mcf:>14.3f}{msf:>14.3f}{medf:>14.3f}"
        )

    print(f"\n{'clip':>5}{'stabilize_maxcnt':>18}{'stabilize_maxspd':>18}{'stabilize_medspd':>18}  vs frame 145")
    stabilize_summary = {}
    for clip_idx, traj in trajectories.items():
        arr = np.array(traj)
        stabs = [stabilize_frame(arr[:, col]) for col in (1, 2, 3)]
        stabilize_summary[clip_idx] = stabs
        past_145 = ["<<< PAST 145" if s > FIRST_VALID_SAMPLE_OFFSET else "" for s in stabs]
        print(f"{clip_idx:>5}{stabs[0]:>18}{stabs[1]:>18}{stabs[2]:>18}  {' '.join(p for p in past_145 if p)}")

    payload = {
        "first_valid_sample_offset": FIRST_VALID_SAMPLE_OFFSET,
        "onsets_local": {str(i): ONSETS[i] - CLIP_STARTS[i] for i in range(len(CLIP_STARTS))},
        "stabilize_frame": {str(k): v for k, v in stabilize_summary.items()},
        "trajectories": {str(k): v for k, v in trajectories.items()},
    }
    OUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
