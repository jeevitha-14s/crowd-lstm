"""Real-time crowd-risk forecasting pipeline over an RTSP stream or video
file: YOLOv8s+ByteTrack -> FeatureExtractor -> SequenceBuffer -> LSTM.

FRAME DROPPING IS THE CORE DESIGN CONSTRAINT, NOT AN EDGE CASE. A reader
thread continuously decodes frames into a single-slot buffer that always
overwrites with the newest frame (FrameSlot below). If the inference loop
falls behind, older undelivered frames are simply gone -- there is no queue
to fall further behind on. This is what makes "real-time" a true claim
rather than a growing backlog; the alternative (buffering/queueing) looks
fine on a file source and silently lies on a live camera.

CAVEAT worth flagging before this is trusted for anything beyond a demo:
SequenceBuffer accumulates one entry per PROCESSED frame, not per elapsed
video frame. The offline model (train_forecast_model.py) was trained on
STRIDE=5 windows sampled from a continuous 30fps decode, i.e. a fixed 1/6s
gap between consecutive samples. Under sustained frame dropping here, the
wall-clock gap between two consecutive *processed* frames can exceed that --
the buffer has no notion of elapsed time, only of how many frames it has
successfully processed. This is a real train/deploy distribution shift that
grows with drop rate; it is not fixed in this file, only documented, since
fixing it (e.g. timestamp-based resampling) is a modeling decision beyond
the scope of this task.
"""

import argparse
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.feature_extractor import FeatureExtractor
from src.train_forecast_model import HORIZONS, CrowdRiskLSTM

DEFAULT_YOLO_PATH = Path("models/finetuned_yolov8s_cctv.pt")
DEFAULT_LSTM_PATH = Path("models/umn_forecast_lstm.pt")
STAGES = ["decode", "detect", "track", "features", "lstm", "render"]
STAGE_WINDOW = 100

SEQ_LEN = 30
STRIDE = 5  # matches UMN training (build_forecast_dataset.py): 29*5=145 frames history

# On-frame callback signature: (frame_id, frame, detections, probs_or_None, timings_ms) -> None
FrameCallback = Callable[
    [int, np.ndarray, List[Dict[str, Any]], Optional[Dict[int, float]], Dict[str, float]], None
]


class FrameSlot:
    """Single-slot overwrite buffer handing frames from the reader thread to
    the inference loop. put() never blocks; a frame that arrives before the
    previous one was consumed overwrites it and counts as dropped -- this is
    the entire frame-dropping mechanism, everything else just uses it."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = -1
        self._decode_ms = 0.0
        self._has_data = False
        self.produced_count = 0
        self.dropped_count = 0

    def put(self, frame: np.ndarray, frame_id: int, decode_ms: float) -> None:
        with self._cv:
            self.produced_count += 1
            if self._has_data:
                self.dropped_count += 1
            self._frame = frame
            self._frame_id = frame_id
            self._decode_ms = decode_ms
            self._has_data = True
            self._cv.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[Tuple[np.ndarray, int, float]]:
        with self._cv:
            if not self._has_data:
                got = self._cv.wait_for(lambda: self._has_data, timeout=timeout)
                if not got:
                    return None
            frame, frame_id, decode_ms = self._frame, self._frame_id, self._decode_ms
            self._has_data = False
            return frame, frame_id, decode_ms


class SequenceBuffer:
    """Rolling buffer of per-processed-frame feature vectors; produces a
    (seq_len, feature_dim) model input by sampling every `stride`-th entry,
    the same windowing build_forecast_dataset.py applies offline to a
    pre-extracted array, just applied online to a live stream of processed
    frames. See the module docstring for what this means under frame drops.
    """

    def __init__(self, seq_len: int, stride: int, feature_dim: int) -> None:
        self.seq_len = seq_len
        self.stride = stride
        self.feature_dim = feature_dim
        self._maxlen = (seq_len - 1) * stride + 1
        self._buffer: Deque[np.ndarray] = deque(maxlen=self._maxlen)

    def push(self, features: np.ndarray) -> None:
        self._buffer.append(features)

    def is_ready(self) -> bool:
        return len(self._buffer) == self._maxlen

    def get_sequence(self) -> np.ndarray:
        """(seq_len, feature_dim), oldest-to-newest, ending at the most
        recently pushed frame. Caller must check is_ready() first."""
        arr = np.stack(self._buffer, axis=0)
        return arr[:: self.stride]

    def reset(self) -> None:
        self._buffer.clear()


class RollingStageTimer:
    """Keeps the last STAGE_WINDOW per-stage timings (ms) for live reporting."""

    def __init__(self, stages: List[str], window: int = STAGE_WINDOW) -> None:
        self._history: Dict[str, Deque[float]] = {s: deque(maxlen=window) for s in stages}

    def record(self, timings_ms: Dict[str, float]) -> None:
        for stage, ms in timings_ms.items():
            self._history[stage].append(ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for stage, hist in self._history.items():
            if not hist:
                out[stage] = {"mean": float("nan"), "n": 0}
                continue
            arr = np.array(hist)
            out[stage] = {"mean": float(arr.mean()), "n": len(arr)}
        return out


class StreamProcessor:
    def __init__(
        self,
        source: str,
        yolo_path: Path = DEFAULT_YOLO_PATH,
        lstm_path: Path = DEFAULT_LSTM_PATH,
        device: Optional[torch.device] = None,
        grid_rows: int = 4,
        grid_cols: int = 4,
        seq_len: int = SEQ_LEN,
        stride: int = STRIDE,
        imgsz: int = 640,
        conf: float = 0.25,
        realtime: bool = False,
        fps_override: Optional[float] = None,
        on_frame: Optional[FrameCallback] = None,
    ) -> None:
        self.source = source
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.conf = conf
        self.realtime = realtime
        self.on_frame = on_frame

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open source: {source}")
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = fps_override or self.cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = native_fps if native_fps and native_fps > 0 else 30.0

        self.yolo = YOLO(str(yolo_path))
        input_size = grid_rows * grid_cols * 4
        self.lstm = CrowdRiskLSTM(input_size=input_size).to(self.device)
        self.lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
        self.lstm.eval()

        self.feature_extractor = FeatureExtractor(frame_width, frame_height, grid_rows, grid_cols)
        self.sequence_buffer = SequenceBuffer(seq_len, stride, input_size)
        self.timer = RollingStageTimer(STAGES)

        self._slot = FrameSlot()
        self._stop_event = threading.Event()
        self._stream_ended = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_frame_counter = 0
        self.frames_processed = 0
        self.lstm_predictions_made = 0
        self._run_start_time: Optional[float] = None
        self._run_end_time: Optional[float] = None
        self._first_prediction_time: Optional[float] = None

    def _reader_loop(self) -> None:
        # Under --realtime, pace reads to the source's native fps so arrival
        # rate matches a live camera. Without pacing, a file decodes far
        # faster than inference can consume it -- the reader would exhaust
        # the whole file before the consumer processes more than a handful
        # of frames, which drops ~everything for the wrong reason (racing
        # ahead of a source that, on a real camera, could never be raced)
        # and starves decode's own timing measurement via GIL contention
        # with the CPU-bound consumer thread.
        frame_interval = 1.0 / self.native_fps if self.realtime else 0.0
        next_due = time.perf_counter()
        while not self._stop_event.is_set():
            if self.realtime:
                sleep_for = next_due - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                next_due += frame_interval

            t0 = time.perf_counter()
            ok, frame = self.cap.read()
            decode_ms = (time.perf_counter() - t0) * 1000
            if not ok:
                self._stream_ended.set()
                return
            self._reader_frame_counter += 1
            self._slot.put(frame, self._reader_frame_counter, decode_ms)

    def _detect_and_track(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float, float]:
        """Returns (detections, detect_ms, track_ms). Detect/track are split
        using ultralytics' own per-call speed breakdown (preprocess+inference
        vs postprocess, the latter including the ByteTrack update) rather than
        decoupling model.predict() from a manually-driven tracker instance --
        the private tracker API ultralytics exposes for that is fragile
        across versions; this uses only the public, documented result."""
        results = self.yolo.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[0],
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            device=str(self.device),
        )
        speed = results[0].speed
        detect_ms = speed.get("preprocess", 0.0) + speed.get("inference", 0.0)
        track_ms = speed.get("postprocess", 0.0)

        t0 = time.perf_counter()
        detections: List[Dict[str, Any]] = []
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), track_id in zip(xyxy, track_ids):
                detections.append(
                    {
                        "track_id": int(track_id),
                        "cx": float((x1 + x2) / 2),
                        "cy": float((y1 + y2) / 2),
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    }
                )
        track_ms += (time.perf_counter() - t0) * 1000
        return detections, detect_ms, track_ms

    def _run_lstm(self, sequence: np.ndarray) -> Dict[int, float]:
        x = torch.tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            preds = self.lstm(x)
            return {k: float(torch.sigmoid(preds[str(k)]).item()) for k in HORIZONS}

    def _process_frame(self, frame: np.ndarray, frame_id: int, decode_ms: float) -> None:
        detections, detect_ms, track_ms = self._detect_and_track(frame)

        t0 = time.perf_counter()
        features = self.feature_extractor.extract(detections)
        features_ms = (time.perf_counter() - t0) * 1000

        self.sequence_buffer.push(features)
        probs: Optional[Dict[int, float]] = None
        t0 = time.perf_counter()
        if self.sequence_buffer.is_ready():
            probs = self._run_lstm(self.sequence_buffer.get_sequence())
            if self.lstm_predictions_made == 0:
                self._first_prediction_time = t0  # marks the end of warmup
            self.lstm_predictions_made += 1
        lstm_ms = (time.perf_counter() - t0) * 1000

        # render is self-referential (its own duration isn't known until after
        # the callback returns), so the callback gets every other stage's
        # timing for this frame and no "render" key.
        partial_timings = {
            "decode": decode_ms,
            "detect": detect_ms,
            "track": track_ms,
            "features": features_ms,
            "lstm": lstm_ms,
        }
        t0 = time.perf_counter()
        if self.on_frame is not None:
            self.on_frame(frame_id, frame, detections, probs, partial_timings)
        render_ms = (time.perf_counter() - t0) * 1000

        timings = {**partial_timings, "render": render_ms}
        self.timer.record(timings)
        self.frames_processed += 1

    def run(self, max_frames: Optional[int] = None, min_predictions: Optional[int] = None) -> None:
        """min_predictions, if set, keeps the loop running until at least
        that many LSTM forecasts have been produced (SequenceBuffer needs
        history_span+1 processed frames before its first prediction), subject
        to max_frames as a safety cap if both are given."""
        self.feature_extractor.reset()
        self.sequence_buffer.reset()
        if hasattr(self.yolo.predictor, "trackers"):
            self.yolo.predictor.trackers[0].reset()

        self._run_start_time = time.perf_counter()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        while not self._stop_event.is_set():
            item = self._slot.get(timeout=1.0)
            if item is None:
                if self._stream_ended.is_set():
                    break
                continue
            frame, frame_id, decode_ms = item
            self._process_frame(frame, frame_id, decode_ms)
            if min_predictions is not None and self.lstm_predictions_made >= min_predictions:
                break
            if max_frames is not None and self.frames_processed >= max_frames:
                break
        self._run_end_time = time.perf_counter()

        self.stop()

    def request_stop(self) -> None:
        """Safe to call from within on_frame (e.g. a 'quit' keypress) --
        the run() loop checks _stop_event at the top of every iteration, so
        this takes effect right after the current frame finishes."""
        self._stop_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        self.cap.release()

    def print_summary(self) -> None:
        summary = self.timer.summary()
        total_produced = self._slot.produced_count
        dropped = self._slot.dropped_count
        drop_rate = dropped / total_produced if total_produced else 0.0
        elapsed = (
            (self._run_end_time - self._run_start_time)
            if self._run_start_time is not None and self._run_end_time is not None
            else float("nan")
        )
        achieved_fps = self.frames_processed / elapsed if elapsed else float("nan")
        prediction_hz = self.lstm_predictions_made / elapsed if elapsed else float("nan")
        # SequenceBuffer needs a one-time warmup ((seq_len-1)*stride+1 processed
        # frames) before its first prediction; averaging predictions over the
        # whole run (including that silent warmup window) understates the
        # steady-state rate a running deployment actually settles into.
        if self._first_prediction_time is not None and self._run_end_time is not None:
            steady_state_elapsed = self._run_end_time - self._first_prediction_time
            steady_state_hz = self.lstm_predictions_made / steady_state_elapsed if steady_state_elapsed else float("nan")
        else:
            steady_state_elapsed = float("nan")
            steady_state_hz = float("nan")

        print("\n=== StreamProcessor summary ===")
        print(
            f"source: {self.source}  device: {self.device}  "
            f"realtime_pacing={self.realtime} (native_fps={self.native_fps:.1f})"
        )
        print(f"run duration: {elapsed:.1f}s")
        print(f"frames produced by source: {total_produced}  dropped: {dropped} ({drop_rate:.1%})")
        print(f"frames processed: {self.frames_processed}  LSTM predictions made: {self.lstm_predictions_made}")
        print(f"achieved processing rate: {achieved_fps:.2f} FPS")
        print(
            f"effective forecast rate: {prediction_hz:.3f} Hz over the full run "
            f"(includes {(self._first_prediction_time - self._run_start_time) if self._first_prediction_time else float('nan'):.1f}s warmup "
            f"before SequenceBuffer first fills); steady-state: {steady_state_hz:.3f} Hz "
            f"(n={self.lstm_predictions_made} predictions over {steady_state_elapsed:.1f}s post-warmup)"
        )

        print(f"\nper-stage mean latency, last {STAGE_WINDOW} frames (ms):")
        print(
            f"  {'decode':>10}: {summary['decode']['mean']:>8.3f} ms  (n={summary['decode']['n']}) "
            f"-- reader thread, paced to native fps; does NOT serialize with the stages below"
        )
        consumer_stages = ["detect", "track", "features", "lstm", "render"]
        for stage in consumer_stages:
            s = summary[stage]
            print(f"  {stage:>10}: {s['mean']:>8.3f} ms  (n={s['n']})")
        consumer_total = sum(summary[s]["mean"] for s in consumer_stages if not np.isnan(summary[s]["mean"]))
        print(
            f"  {'consumer total':>10}: {consumer_total:>8.3f} ms  "
            f"(serial processing ceiling ~{1000.0 / consumer_total:.2f} FPS if never idle)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time crowd-risk forecasting stream processor")
    parser.add_argument("source", help="RTSP URL or video file path")
    parser.add_argument("--yolo-path", type=Path, default=DEFAULT_YOLO_PATH)
    parser.add_argument("--lstm-path", type=Path, default=DEFAULT_LSTM_PATH)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-predictions", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--realtime", action="store_true", help="pace the reader to the source's native fps"
    )
    parser.add_argument("--fps", type=float, default=None, help="override native fps (e.g. for RTSP)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    processor = StreamProcessor(
        args.source,
        yolo_path=args.yolo_path,
        lstm_path=args.lstm_path,
        device=device,
        realtime=args.realtime,
        fps_override=args.fps,
    )
    print(f"Opened {args.source}  device={processor.device}  realtime={args.realtime}")
    processor.run(max_frames=args.max_frames, min_predictions=args.min_predictions)
    processor.print_summary()


if __name__ == "__main__":
    main()
