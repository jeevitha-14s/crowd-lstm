"""OpenCV live dashboard for the crowd-risk forecasting pipeline: wired as
stream_processor.StreamProcessor's on_frame callback, draws detections, the
zone grid, an alert-level banner, per-horizon probability bars, a rolling
sparkline, and perf stats over each frame -- either to an interactive window
or, with --headless, to an output video file.

The dashboard owns the AlertManager: probabilities flow in from the stream
processor, get fed to AlertManager.update() here (so the banner always shows
the SAME persisted level that's being logged/alerted on, not a separately
computed one), and the resulting level drives both the banner colour and the
probability-bar colour thresholds.
"""

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.alerting import AlertLevel, AlertManager
from src.stream_processor import DEFAULT_LSTM_PATH, DEFAULT_YOLO_PATH, StreamProcessor
from src.train_forecast_model import HORIZONS

TRAINING_FPS = 30.0  # UMN training fps, used only to label horizons in seconds
SPARKLINE_SECONDS = 30
SCREENSHOT_DIR = Path("outputs/figures")

LEVEL_COLORS = {
    AlertLevel.GREEN: (0, 180, 0),
    AlertLevel.AMBER: (0, 165, 255),
    AlertLevel.RED: (0, 0, 255),
}


class LiveDashboard:
    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        frame_width: int,
        frame_height: int,
        alert_manager: AlertManager,
        headless: bool = False,
        output_path: Optional[Path] = None,
        output_fps: float = 15.0,
        sparkline_horizon: int = 60,
        on_quit: Optional[callable] = None,
    ) -> None:
        assert sparkline_horizon in HORIZONS
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.alert_manager = alert_manager
        self.headless = headless
        self.sparkline_horizon = sparkline_horizon
        self.on_quit = on_quit

        # Sparkline is keyed on wall-clock time, not frame count: processed
        # frames arrive at a variable, drop-dependent rate, so a frame-count
        # window would stretch or compress in real time depending on load.
        self._sparkline: Deque[Tuple[float, float]] = deque()

        self._video_writer: Optional[cv2.VideoWriter] = None
        if headless:
            assert output_path is not None, "--output is required with --headless"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(output_path), fourcc, output_fps, (frame_width, frame_height)
            )

        self._paused = False
        self._last_rendered: Optional[np.ndarray] = None
        self.quit_requested = False

        self._fps_history: Deque[float] = deque(maxlen=30)
        self._last_frame_wall_time: Optional[float] = None

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> None:
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                frame, str(det["track_id"]), (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )

    def _draw_grid(self, frame: np.ndarray) -> None:
        zone_w = self.frame_width / self.grid_cols
        zone_h = self.frame_height / self.grid_rows
        for c in range(1, self.grid_cols):
            x = int(c * zone_w)
            cv2.line(frame, (x, 0), (x, self.frame_height), (80, 80, 80), 1)
        for r in range(1, self.grid_rows):
            y = int(r * zone_h)
            cv2.line(frame, (0, y), (self.frame_width, y), (80, 80, 80), 1)

    def _draw_banner(self, frame: np.ndarray, level: AlertLevel) -> None:
        color = LEVEL_COLORS[level]
        cv2.rectangle(frame, (0, 0), (self.frame_width, 24), color, -1)
        cv2.putText(frame, level.name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _bar_color(self, p: float) -> Tuple[int, int, int]:
        if p >= self.alert_manager.red_threshold:
            return LEVEL_COLORS[AlertLevel.RED]
        if p >= self.alert_manager.amber_threshold:
            return LEVEL_COLORS[AlertLevel.AMBER]
        return LEVEL_COLORS[AlertLevel.GREEN]

    def _draw_prob_bars(self, frame: np.ndarray, probs: Optional[Dict[int, float]]) -> None:
        base_y, bar_w, bar_h = 34, 120, 14
        for i, k in enumerate(HORIZONS):
            y = base_y + i * (bar_h + 4)
            seconds = k / TRAINING_FPS
            p = probs.get(k, 0.0) if probs else 0.0
            cv2.putText(
                frame, f"{seconds:.0f}s", (4, y + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )
            cv2.rectangle(frame, (30, y), (30 + bar_w, y + bar_h), (60, 60, 60), 1)
            fill_w = int(bar_w * min(max(p, 0.0), 1.0))
            cv2.rectangle(frame, (30, y), (30 + fill_w, y + bar_h), self._bar_color(p), -1)
            cv2.putText(
                frame, f"{p:.2f}", (30 + bar_w + 6, y + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )

    def _update_sparkline(self, probs: Optional[Dict[int, float]]) -> None:
        if probs is None or self.sparkline_horizon not in probs:
            return
        now = time.time()
        self._sparkline.append((now, probs[self.sparkline_horizon]))
        cutoff = now - SPARKLINE_SECONDS
        while self._sparkline and self._sparkline[0][0] < cutoff:
            self._sparkline.popleft()

    def _draw_sparkline(self, frame: np.ndarray) -> None:
        w, h = 150, 40
        x0, y0 = self.frame_width - w - 8, 8
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), -1)
        cv2.putText(
            frame, f"k={self.sparkline_horizon} {SPARKLINE_SECONDS}s",
            (x0 + 2, y0 + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
        )
        if len(self._sparkline) < 2:
            return
        now = time.time()
        pts = []
        for t, p in self._sparkline:
            x = x0 + int(w * (1 - (now - t) / SPARKLINE_SECONDS))
            y = y0 + h - int(h * min(max(p, 0.0), 1.0))
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            cv2.line(frame, a, b, (0, 200, 255), 1)

    def _draw_perf(self, frame: np.ndarray, timings: Dict[str, float]) -> None:
        now = time.time()
        if self._last_frame_wall_time is not None:
            dt = now - self._last_frame_wall_time
            if dt > 0:
                self._fps_history.append(1.0 / dt)
        self._last_frame_wall_time = now
        fps = float(np.mean(self._fps_history)) if self._fps_history else 0.0
        latency_ms = sum(timings.values())  # excludes render (unknown until after this call)
        cv2.putText(
            frame, f"{fps:.1f} FPS  {latency_ms:.1f} ms", (4, self.frame_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
        )

    def render(
        self,
        frame_id: int,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        probs: Optional[Dict[int, float]],
        timings: Dict[str, float],
    ) -> Tuple[np.ndarray, AlertLevel]:
        annotated = frame.copy()
        self._draw_grid(annotated)
        self._draw_detections(annotated, detections)
        level = self.alert_manager.update(frame_id, probs) if probs else self.alert_manager.current_level
        self._draw_banner(annotated, level)
        self._draw_prob_bars(annotated, probs)
        self._update_sparkline(probs)
        self._draw_sparkline(annotated)
        self._draw_perf(annotated, timings)
        return annotated, level

    def handle_frame(
        self,
        frame_id: int,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        probs: Optional[Dict[int, float]],
        timings: Dict[str, float],
    ) -> None:
        """Wired directly as StreamProcessor's on_frame callback."""
        if self.quit_requested:
            return
        annotated, _ = self.render(frame_id, frame, detections, probs, timings)

        if self.headless:
            if self._video_writer is not None:
                self._video_writer.write(annotated)
            return

        if not self._paused:
            self._last_rendered = annotated
        display = self._last_rendered if self._last_rendered is not None else annotated
        cv2.imshow("crowd-risk-lstm", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.quit_requested = True
            if self.on_quit is not None:
                self.on_quit()
        elif key == ord(" "):
            self._paused = not self._paused
        elif key == ord("s"):
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = SCREENSHOT_DIR / f"screenshot_{int(time.time())}_{frame_id}.png"
            cv2.imwrite(str(out_path), display)
            print(f"saved {out_path}")

    def close(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
        if not self.headless:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live crowd-risk dashboard")
    parser.add_argument("source", help="RTSP URL or video file path")
    parser.add_argument("--yolo-path", type=Path, default=DEFAULT_YOLO_PATH)
    parser.add_argument("--lstm-path", type=Path, default=DEFAULT_LSTM_PATH)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="output video path (--headless)")
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument("--alert-horizon", type=int, default=60, choices=HORIZONS)
    parser.add_argument("--amber-threshold", type=float, default=0.4)
    parser.add_argument("--red-threshold", type=float, default=0.6)
    args = parser.parse_args()

    if args.headless and args.output is None:
        parser.error("--headless requires --output")

    device = torch.device(args.device) if args.device else None
    processor = StreamProcessor(
        args.source,
        yolo_path=args.yolo_path,
        lstm_path=args.lstm_path,
        device=device,
        realtime=args.realtime,
        fps_override=args.fps,
    )

    alert_manager = AlertManager(
        horizon=args.alert_horizon,
        amber_threshold=args.amber_threshold,
        red_threshold=args.red_threshold,
    )
    dashboard = LiveDashboard(
        grid_rows=processor.feature_extractor.grid_rows,
        grid_cols=processor.feature_extractor.grid_cols,
        frame_width=processor.feature_extractor.frame_width,
        frame_height=processor.feature_extractor.frame_height,
        alert_manager=alert_manager,
        headless=args.headless,
        output_path=args.output,
        output_fps=args.output_fps,
        sparkline_horizon=args.alert_horizon,
        on_quit=processor.request_stop,
    )
    processor.on_frame = dashboard.handle_frame

    print(f"Opened {args.source}  device={processor.device}  headless={args.headless}")
    try:
        processor.run(max_frames=args.max_frames)
    finally:
        dashboard.close()
        alert_manager.close()
    processor.print_summary()


if __name__ == "__main__":
    main()
