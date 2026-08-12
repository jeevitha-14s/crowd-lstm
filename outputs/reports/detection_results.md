# Detection results — YOLOv8s on CrowdHuman

Test set: 1,937 images, 57,105 instances. Both rows evaluated on the
identical test split.

| model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| YOLOv8s, COCO-pretrained (no fine-tuning) | 0.632 | 0.398 | 0.432 | 0.215 |
| YOLOv8s + CrowdHuman fine-tuning | 0.8346 | 0.6016 | 0.6994 | 0.4312 |

Source: run in Colab, reported directly by the user (2026-08-08/09) — not
reproduced from a saved training/eval log in this repo. This file is the
first saved record of these numbers.

Fine-tuning improves every metric substantially: +0.20 precision, +0.20
recall, +0.27 mAP50, +0.22 mAP50-95 over the COCO-pretrained baseline on
identical data. This is the baseline row that was previously missing.

## Training run and the epoch 39-60 gap

The fine-tuned figures above are from the **completed 60-epoch run**. The
saved `outputs/yolo_training/crowdhuman_yolov8s/results.csv` only logs
through epoch 38 (see `PAPER_RESULTS.md` Section 1) because training resumed
in a fresh Colab session whose log never synced back to this Drive folder —
the run itself completed, only the epoch 39-60 portion of the CSV log did
not. Per the user: validation mAP50 moved from 0.682 (epoch 38, in the saved
log) to 0.683 at epoch 60 — consistent with the plateau already visible in
the epoch 20-38 trend in `results.csv`, not a discontinuity. Training
converged; the missing log rows reflect a sync gap, not a missing result.
