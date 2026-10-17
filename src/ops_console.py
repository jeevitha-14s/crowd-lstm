"""Multi-camera operator console: a browser UI over N concurrent
StreamProcessor pipelines, one per camera.

Each camera gets its own StreamProcessor (own YOLO+ByteTrack instance, since
ByteTrack's persisted track state is per-model) and its own AlertManager, so
the tier shown on a tile is the SAME persisted, debounced level that is being
logged/webhooked for that camera -- never a separately computed one, same rule
as live_dashboard.py.

Everything reaches the browser over ONE Server-Sent Events stream. One
connection rather than an MJPEG stream per tile matters: browsers cap HTTP/1.1
at 6 concurrent connections per host, so per-tile streams would silently stall
the 6th camera. The SSE generator also only ever sends each camera's NEWEST
packet (Hub.wait coalesces), so a slow browser drops frames rather than
building a backlog -- the same frame-dropping rule stream_processor.py applies
to the source.

Frames, boxes and forecasts travel in the same packet, so the focused feed's
overlay is always drawn on the exact frame it was computed from.

    python3 -m src.ops_console                       # default demo cameras, real models
    python3 -m src.ops_console --simulate            # no models: scripted risk, UI dev only
    python3 -m src.ops_console --camera "Gate=rtsp://localhost:8554/umn" --camera clip.mp4

--simulate output is synthetic and labelled SIMULATED in the UI; like the UMN
checkpoint itself, it must never be used to report a metric.
"""

import argparse
import base64
import json
import math
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, send_from_directory

from src.alerting import AlertEvent, AlertManager
from src.feature_extractor import N_FEATURES_PER_ZONE
from src.train_forecast_model import HORIZONS

UI_DIR = Path(__file__).parent / "ops_console_ui"
ALERT_LOG_DIR = Path("outputs/reports/ops_console")
TRAINING_FPS = 30.0  # labels horizons in seconds (k=30 -> +1s), as in live_dashboard.py
# Single camera for now; add more with --camera (e.g. MED panic clips 001-011
# in "(Copy) Motion_Emotion_Dataset/", see process_med.py).
DEFAULT_SOURCES = [
    ("UMN", Path("data/raw/umn/Crowd-Activity-All.avi")),
]
MAX_STREAM_WIDTH = 960
JPEG_QUALITY = 75
MAX_ALERT_HISTORY = 500


class Hub:
    """Latest packet per camera plus the alert history, with a condition
    variable so SSE clients block until something newer exists."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._version = 0
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._alerts: List[Dict[str, Any]] = []

    def publish_frame(self, cam_id: str, packet: Dict[str, Any]) -> None:
        with self._cv:
            self._version += 1
            packet["seq"] = self._version
            self._latest[cam_id] = packet
            self._cv.notify_all()

    def publish_alert(self, record: Dict[str, Any]) -> None:
        with self._cv:
            self._version += 1
            record["seq"] = self._version
            self._alerts.append(record)
            del self._alerts[:-MAX_ALERT_HISTORY]
            self._cv.notify_all()

    def wait(
        self, since: int, timeout: float
    ) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """(version, frames newer than `since`, alerts newer than `since`).
        Only each camera's latest frame is returned, however many were
        published in between."""
        with self._cv:
            self._cv.wait_for(lambda: self._version > since, timeout=timeout)
            frames = [p for p in self._latest.values() if p["seq"] > since]
            alerts = [a for a in self._alerts if a["seq"] > since]
            return self._version, frames, alerts


def encode_jpeg(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    if w > MAX_STREAM_WIDTH:
        scale = MAX_STREAM_WIDTH / w
        frame = cv2.resize(frame, (MAX_STREAM_WIDTH, int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok, "JPEG encode failed"
    return base64.b64encode(buf.tobytes()).decode("ascii")


class Camera:
    """Shared publishing logic; subclasses supply the frames and forecasts."""

    def __init__(self, cam_id: str, name: str, source: str, hub: Hub, alert_manager: AlertManager) -> None:
        self.cam_id = cam_id
        self.name = name
        self.source = source
        self.hub = hub
        self.alert_manager = alert_manager
        self.alert_manager.add_sink(self._on_alert)
        self._fps = 0.0
        self._last_t: Optional[float] = None

    def describe(self) -> Dict[str, str]:
        return {"id": self.cam_id, "name": self.name, "source": self.source}

    def _on_alert(self, event: AlertEvent) -> None:
        self.hub.publish_alert(
            {
                "cam": self.cam_id,
                "t": event.timestamp,
                "frame_id": event.frame_id,
                "old": event.old_level.name,
                "new": event.new_level.name,
                "probs": {str(k): round(v, 3) for k, v in event.probabilities.items()},
            }
        )

    def publish(
        self,
        frame_id: int,
        frame: np.ndarray,
        boxes: List[List[float]],
        zone_count: Optional[np.ndarray],
        zone_flow: Optional[np.ndarray],
        grid: Tuple[int, int],
        probs: Optional[Dict[int, float]],
    ) -> None:
        level = self.alert_manager.update(frame_id, probs) if probs else self.alert_manager.current_level
        now = time.time()
        if self._last_t is not None and now > self._last_t:
            inst = 1.0 / (now - self._last_t)
            self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_t = now
        self.hub.publish_frame(
            self.cam_id,
            {
                "cam": self.cam_id,
                "t": now,
                "frame_id": frame_id,
                "img": encode_jpeg(frame),
                "boxes": boxes,
                "grid": list(grid),
                "zone_count": None if zone_count is None else [round(float(v), 3) for v in zone_count],
                "zone_flow": None if zone_flow is None else [round(float(v), 3) for v in zone_flow],
                "probs": None if probs is None else {str(k): round(v, 4) for k, v in probs.items()},
                "level": level.name,
                "fps": round(self._fps, 1),
            },
        )

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class LiveCamera(Camera):
    def __init__(self, cam_id: str, name: str, source: str, hub: Hub, alert_manager: AlertManager, args: argparse.Namespace) -> None:
        super().__init__(cam_id, name, source, hub, alert_manager)
        import torch  # deferred so --simulate never pays for torch/ultralytics imports

        from src.stream_processor import StreamProcessor, is_live_source

        live = is_live_source(source)
        self.processor = StreamProcessor(
            source,
            yolo_path=args.yolo_path,
            lstm_path=args.lstm_path,
            device=torch.device(args.device) if args.device else None,
            imgsz=args.imgsz,
            # Files must be paced or the reader races the whole file (see
            # README); a live stream is already paced by the network.
            realtime=not live,
            fps_override=args.fps,
            # For a file, "reconnect" reopens it at EOF -- i.e. loops the demo,
            # with the same scene-state reset a real reconnect gets.
            reconnect=True if live else args.loop,
            on_frame=self._on_frame,
        )
        fe = self.processor.feature_extractor
        self._grid = (fe.grid_rows, fe.grid_cols)
        self._w, self._h = fe.frame_width, fe.frame_height
        self._thread: Optional[threading.Thread] = None

    def _on_frame(self, frame_id, frame, detections, probs, timings) -> None:
        boxes = [
            [round(x1 / self._w, 4), round(y1 / self._h, 4), round(x2 / self._w, 4), round(y2 / self._h, 4)]
            for (x1, y1, x2, y2) in (d["bbox"] for d in detections)
        ]
        zone_count = zone_flow = None
        if self.processor.last_features is not None:
            per_zone = self.processor.last_features.reshape(-1, N_FEATURES_PER_ZONE)
            zone_count, zone_flow = per_zone[:, 0], per_zone[:, 1]
        self.publish(frame_id, frame, boxes, zone_count, zone_flow, self._grid, probs)

    def start(self) -> None:
        self._thread = threading.Thread(target=self.processor.run, name=f"cam-{self.cam_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.processor.request_stop()


class SimulatedCamera(Camera):
    """Plays the source video with a scripted risk curve -- no detector, no
    LSTM. Exists so the UI can be developed and demoed on any machine. Each
    camera gets a different peak height and phase so the console sees
    staggered AMBER/RED escalations, including overlapping ones."""

    PERIOD_S = 80.0
    PEAKS = [0.88, 0.52, 0.28, 0.74]
    WARMUP_S = 5.0  # mimics SequenceBuffer's (30-1)*5-frame warmup at 30fps
    PUBLISH_FPS = 12.0

    def __init__(self, cam_id: str, name: str, source: str, hub: Hub, alert_manager: AlertManager, index: int, n_cams: int) -> None:
        super().__init__(cam_id, name, source, hub, alert_manager)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open source: {source}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = fps if fps and fps > 0 else 30.0
        self.peak = self.PEAKS[index % len(self.PEAKS)]
        self.offset = 18.0 + index * self.PERIOD_S / max(n_cams, 1) * 0.6
        self.rng = np.random.default_rng(index)
        self.hotspot = (int(self.rng.integers(1, 3)), int(self.rng.integers(1, 3)))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _risk(self, t: float) -> float:
        u = (t - self.offset) % self.PERIOD_S
        rise, hold, fall = 8.0, 7.0, 12.0
        if u < rise:
            bump = 0.5 - 0.5 * math.cos(math.pi * u / rise)
        elif u < rise + hold:
            bump = 1.0
        elif u < rise + hold + fall:
            bump = 0.5 + 0.5 * math.cos(math.pi * (u - rise - hold) / fall)
        else:
            bump = 0.0
        return 0.08 + (self.peak - 0.08) * bump

    def _run(self) -> None:
        start = time.time()
        frame_id = 0
        publish_every = max(1, int(round(self.native_fps / self.PUBLISH_FPS)))
        next_due = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_id += 1
            next_due += 1.0 / self.native_fps
            sleep_for = next_due - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            if frame_id % publish_every:
                continue

            t = time.time() - start
            probs = None
            if t >= self.WARMUP_S:
                noise = self.rng.normal(0, 0.02, len(HORIZONS))
                probs = {
                    k: float(np.clip(self._risk(t + k / TRAINING_FPS) + n, 0.0, 1.0))
                    for k, n in zip(HORIZONS, noise)
                }
            rows, cols = 4, 4
            rr, cc = np.mgrid[0:rows, 0:cols]
            dist2 = (rr - self.hotspot[0]) ** 2 + (cc - self.hotspot[1]) ** 2
            heat = self._risk(t) * np.exp(-dist2 / 1.5).ravel()
            flow = np.clip(heat + self.rng.uniform(0, 0.08, rows * cols), 0, 1)
            count = np.clip(0.3 + 0.5 * heat + self.rng.uniform(0, 0.1, rows * cols), 0, 1)
            self.publish(frame_id, frame, [], count, flow, (rows, cols), probs)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"sim-{self.cam_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def build_app(hub: Hub, cameras: List[Camera], args: argparse.Namespace) -> Flask:
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index() -> Response:
        return send_from_directory(UI_DIR, "index.html")

    @app.get("/ui/<path:name>")
    def ui(name: str) -> Response:
        return send_from_directory(UI_DIR, name)

    @app.get("/config")
    def config() -> Response:
        return jsonify(
            {
                "mode": "simulated" if args.simulate else "live",
                "cameras": [c.describe() for c in cameras],
                "horizons": HORIZONS,
                "horizon_seconds": [k / TRAINING_FPS for k in HORIZONS],
                "alert_horizon": args.alert_horizon,
                "amber_threshold": args.amber_threshold,
                "red_threshold": args.red_threshold,
            }
        )

    @app.get("/events")
    def events() -> Response:
        def stream():
            version = 0
            yield "retry: 2000\n\n"
            while True:
                version, frames, alerts = hub.wait(version, timeout=15.0)
                if not frames and not alerts:
                    yield ": keepalive\n\n"
                    continue
                for a in alerts:
                    yield f"event: alert\ndata: {json.dumps(a)}\n\n"
                for f in frames:
                    yield f"event: frame\ndata: {json.dumps(f)}\n\n"

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def parse_cameras(specs: Optional[List[str]]) -> List[Tuple[str, str]]:
    """'NAME=SOURCE' or bare 'SOURCE'. The '=' split is on the first '=' only
    and only when the left side has no '/' or ':', so RTSP URLs with query
    strings aren't mangled."""
    if not specs:
        missing = [str(p) for _, p in DEFAULT_SOURCES if not p.exists()]
        if missing:
            print(f"ops_console: skipping missing default sources: {missing}")
        return [(name, str(p)) for name, p in DEFAULT_SOURCES if p.exists()]
    out = []
    for spec in specs:
        name, sep, source = spec.partition("=")
        if sep and "/" not in name and ":" not in name:
            out.append((name, source))
        else:
            out.append((Path(spec).stem or spec, spec))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-camera crowd-risk operator console")
    parser.add_argument("--camera", action="append", help="NAME=SOURCE or SOURCE (repeatable); default: the UMN video")
    parser.add_argument("--simulate", action="store_true", help="scripted risk, no models (UI development only)")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="stop file sources at EOF instead of looping")
    parser.add_argument("--yolo-path", type=Path, default=Path("models/finetuned_yolov8s_cctv.pt"))
    parser.add_argument("--lstm-path", type=Path, default=Path("models/umn_forecast_lstm.pt"))
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fps", type=float, default=None, help="override native fps (e.g. for RTSP)")
    parser.add_argument("--alert-horizon", type=int, default=60, choices=HORIZONS)
    parser.add_argument("--amber-threshold", type=float, default=0.4)
    parser.add_argument("--red-threshold", type=float, default=0.6)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    sources = parse_cameras(args.camera)
    if not sources:
        parser.error("no camera sources available; pass --camera")

    hub = Hub()
    cameras: List[Camera] = []
    alert_managers: List[AlertManager] = []
    # Simulated alerts go to a throwaway dir so they can never be mistaken
    # for a real run's alert log.
    sim_tmp = tempfile.TemporaryDirectory() if args.simulate else None
    log_dir = Path(sim_tmp.name) if sim_tmp else ALERT_LOG_DIR
    for i, (name, source) in enumerate(sources):
        cam_id = f"cam{i + 1:02d}"
        am = AlertManager(
            horizon=args.alert_horizon,
            amber_threshold=args.amber_threshold,
            red_threshold=args.red_threshold,
            csv_path=log_dir / f"alert_log_{cam_id}.csv",
            webhook_url=None if args.simulate else "unset",
        )
        alert_managers.append(am)
        display = f"CAM {i + 1:02d} · {name}"
        print(f"ops_console: opening {display} <- {source}")
        if args.simulate:
            cameras.append(SimulatedCamera(cam_id, display, source, hub, am, i, len(sources)))
        else:
            cameras.append(LiveCamera(cam_id, display, source, hub, am, args))

    for cam in cameras:
        cam.start()

    app = build_app(hub, cameras, args)
    print(f"ops_console: {'SIMULATED' if args.simulate else 'live'} console at http://{args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    finally:
        for cam in cameras:
            cam.stop()
        for am in alert_managers:
            am.close()
        if sim_tmp:
            sim_tmp.cleanup()


if __name__ == "__main__":
    main()
