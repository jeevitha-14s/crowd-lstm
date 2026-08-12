# crowd-risk-lstm

Predictive crowd risk forecasting: the core contribution is reframing panic
*detection* as panic *forecasting* -- predicting an onset k seconds before it
happens, not flagging it as it occurs. See `outputs/reports/` for evaluation
results and `models/*.README.md` for checkpoint provenance notes.

## Setup

Raw dataset zips (CrowdHuman, ShanghaiTech) and the two forecast `.npz`
tensors are gitignored -- too large for a normal git repo (ShanghaiTech's
zip alone is 8.7GB, and `shanghaitech_forecast.npz` at 214MB exceeds
GitHub's 100MB per-file limit). What's checked in instead:

- `data/features/` -- already-extracted per-clip features (small, ~11MB),
  enough to train/evaluate without touching raw images or video.
- `data/raw/umn/Crowd-Activity-All.avi` -- the one raw asset several
  scripts and the deployment quick-start below depend on directly.

To regenerate the forecast tensors from `data/features/`:

    python3 -m src.build_forecast_dataset          # -> data/forecast_dataset.npz
    python3 -m src.build_shanghaitech_forecast      # -> data/shanghaitech_forecast.npz

To re-run feature extraction from scratch you'll need the original
datasets: UMN "Unusual Crowd Activity", ShanghaiTech Campus, and
CrowdHuman (fine-tuning data for the YOLOv8s detector) -- not included
here due to size.

## Deployment layer

Real-time pipeline closing the gap between the offline LOCO evaluation and
the paper's live-monitoring framing: YOLOv8s+ByteTrack -> FeatureExtractor
-> LSTM -> alerting -> dashboard. Deployment uses the UMN-trained model
(`models/umn_forecast_lstm.pt`) since it matches the target scenario
(collective crowd panic); see that checkpoint's README before using it for
anything -- it must never be used to report a metric.

- `src/stream_processor.py` -- real-time pipeline with mandatory frame
  dropping: a single-slot buffer always serves the newest frame to the
  consumer, overwriting anything not yet consumed, rather than queueing --
  a queued design falls further behind reality every second on a live
  source and makes "real-time" a false claim.
- `src/alerting.py` -- three-level (GREEN/AMBER/RED) alerting driven by the
  k=60 (2s-ahead) head by default, with asymmetric rise/fall persistence so
  a single noisy frame can't trigger or clear an alert.
- `src/live_dashboard.py` -- interactive OpenCV view (`q` quit, `space`
  pause, `s` screenshot) or, with `--headless`, an annotated output video.
- `src/benchmark_latency.py` -- per-stage latency table for the paper,
  saved to `outputs/reports/latency_benchmark.csv`. See its module
  docstring for the measurement methodology (sequential, unthrottled,
  fixed imgsz=640 across resolutions).

### Quick test on a video file

    python3 -m src.stream_processor data/raw/umn/Crowd-Activity-All.avi --realtime --min-predictions 500
    python3 -m src.live_dashboard data/raw/umn/Crowd-Activity-All.avi --realtime

`--realtime` paces frame reads to the source's native fps. Without it, a
file decodes far faster than inference can consume it: the reader thread
races through the whole file before the consumer processes more than a
handful of frames, which drops ~everything for the wrong reason (racing a
source that, on a real camera, could never be raced) and never lets the
30-frame `SequenceBuffer` fill enough to produce a single prediction.

### Testing over a genuine RTSP stream (mediamtx)

`stream_processor.py` opens its source via `cv2.VideoCapture`, which accepts
an RTSP URL exactly like a file path. But even a `--realtime`-paced file
read is still fundamentally a file handle: to exercise the pipeline against
something that behaves like an actual camera -- frames arriving over a real
network stream, no way to read ahead of the source no matter what the code
does -- serve the UMN file as RTSP with
[mediamtx](https://github.com/bluenviron/mediamtx) (formerly
rtsp-simple-server).

**1. Install mediamtx and ffmpeg:**

    brew install mediamtx ffmpeg

**2. Start mediamtx** (default config is sufficient -- it listens for RTSP
publishers on `rtsp://localhost:8554/<path>` with no path pre-registration
needed):

    mediamtx &

**3. Publish the UMN file as a looping RTSP stream, paced to its native fps:**

    ffmpeg -re -stream_loop -1 -i data/raw/umn/Crowd-Activity-All.avi \
      -c:v libx264 -f rtsp rtsp://localhost:8554/umn

`-re` is the important flag: it makes ffmpeg read the file at its native
frame rate rather than as fast as disk I/O allows, which is what makes this
a genuine paced stream rather than a fast file dump -- the network, not
`stream_processor.py`, is now what enforces real-time arrival.
`-stream_loop -1` loops the video indefinitely so a test run isn't bounded
by the source's length. Re-encoding to libx264 (rather than `-c copy`) is
used because RTSP requires a streamable codec, which the source AVI's codec
may not be.

**4. Point the pipeline at the RTSP URL** (no `--realtime` flag needed here
-- the stream itself already arrives paced, enforced by ffmpeg/mediamtx
rather than by the pipeline's own reader thread):

    python3 -m src.stream_processor rtsp://localhost:8554/umn --min-predictions 500
    python3 -m src.live_dashboard rtsp://localhost:8554/umn

If `cv2.VideoCapture` reports 0 fps for an RTSP source (common -- RTSP
doesn't reliably advertise frame rate), pass `--fps 30` explicitly rather
than silently falling back to the 30fps default.

**5. Stop the stream:** `Ctrl-C` the `ffmpeg` process, then `kill %1` (or
`Ctrl-C` if `mediamtx` was left in the foreground) to stop the server.

### Latency benchmark

    python3 -m src.benchmark_latency data/raw/umn/Crowd-Activity-All.avi

Runs on CPU always, and on CUDA if `torch.cuda.is_available()`. This repo's
local numbers were produced on a GPU-less Mac; a CUDA run needs a machine
with an NVIDIA GPU (e.g. Colab, matching the workflow already used for the
ShanghaiTech training and the fine-tuned YOLO's own T4 benchmark) -- run the
same command there and merge the resulting CSV rows with the CPU ones for
the paper's combined table.
