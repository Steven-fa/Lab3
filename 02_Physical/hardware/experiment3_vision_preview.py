#!/usr/bin/env python3
"""YOLO and six-region preview for experiment 3, with no robot connection."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime
import json
from pathlib import Path
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from experiment3_region_calibrator import locate_grid, validate_regions


CUBE_CLASSES = {"blue_cube", "green_cube", "pink_cube", "yellow_cube"}
OTHER_CLASSES = {"red_cuboid", "orange_bridge"}
STATE_COLORS = {
    "EMPTY": (150, 150, 150),
    "CANDIDATE": (0, 215, 255),
    "READY_CUBE": (70, 220, 70),
    "READY_OTHER": (255, 180, 40),
    "MULTIPLE": (60, 60, 255),
    "UNKNOWN": (60, 60, 255),
}


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>实验三YOLO六区域验证</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101214;color:#eef1f4;font-family:"Microsoft YaHei",system-ui,sans-serif}
main{width:min(1500px,100%);margin:auto;padding:12px}h1{margin:0 0 4px;font-size:22px}.notice{margin:0 0 10px;color:#ffd166}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:12px;align-items:start}img{display:block;width:100%;height:auto;background:#000;border:1px solid #3d4650}
aside{background:#1a1e22;border:1px solid #394049;border-radius:8px;padding:12px}.mode{color:#59e391;font-weight:700;margin-bottom:10px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:7px 5px;border-bottom:1px solid #3b434c;text-align:center}th{color:#cbd3da}.ready{color:#59e391}.error{color:#ff6b6b}.candidate{color:#ffd166}
.summary{margin-top:10px;line-height:1.5;color:#cbd3da}.small{font-size:13px;color:#aeb7c0}.plans{margin-top:12px;border-top:1px solid #3b434c;padding-top:10px}.plans h2{font-size:16px;margin:0 0 8px}.plan{background:#242a30;border-left:4px solid #59e391;border-radius:4px;margin:8px 0;padding:8px}.route{font-size:12px;line-height:1.45;color:#bdc7d0;margin-top:5px;word-break:break-word}.empty-plan{color:#aeb7c0;font-size:13px}@media(max-width:900px){.layout{grid-template-columns:1fr}}
</style></head><body><main>
<h1>实验三 YOLO 六区域验证</h1>
<p class="notice" id="notice">正在读取运行模式……</p>
<div class="layout"><img src="/stream.mjpg" alt="YOLO实时识别画面"><aside>
<div class="mode" id="mode">正在加载模型……</div>
<table><thead><tr><th>格号</th><th>状态</th><th>分类</th><th>置信度</th></tr></thead><tbody id="rows"></tbody></table>
<div class="summary" id="summary"></div>
<section class="plans"><h2 id="planTitle">自动动作计划</h2><div id="plans"><span class="empty-plan">等待稳定识别结果……</span></div></section>
<p class="small">READY 需要最近7帧中至少5帧结果一致。区域外目标只显示，不进入抓取队列。</p>
</aside></div></main>
<script>
const rows=document.getElementById('rows'),summary=document.getElementById('summary'),mode=document.getElementById('mode'),plans=document.getElementById('plans'),notice=document.getElementById('notice'),planTitle=document.getElementById('planTitle');
function esc(v){return String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});const d=await r.json();mode.textContent=d.model_ready?'模型已加载，等待稳定结果':'模型正在加载';notice.textContent=d.robot_connected?'AUTO：视觉状态机已连接机械臂；现场必须保持急停可用。':'VISION ONLY：只识别，不连接机械臂，不会触发抓取。';planTitle.textContent=d.dry_run?'自动动作计划（不会执行）':'自动任务队列';rows.innerHTML='';
for(const s of d.regions||[]){const tr=document.createElement('tr');const cls=s.state.startsWith('READY')?'ready':(s.state==='CANDIDATE'?'candidate':(s.state==='EMPTY'?'':'error'));tr.className=cls;tr.innerHTML=`<td>${s.grid_id}</td><td>${esc(s.state)}</td><td>${esc(s.group)}</td><td>${s.confidence==null?'—':Number(s.confidence).toFixed(2)}</td>`;rows.appendChild(tr);}summary.innerHTML=`FPS：${Number(d.fps||0).toFixed(1)}<br>画面目标：${d.detection_count??0}<br>区域外目标：${d.outside_count??0}<br>计划数：${d.planned_count??0}<br>机械臂：<strong>${d.robot_connected?'已连接':'未连接'}</strong>${d.task_state?`<br>任务状态：<strong>${esc(d.task_state)}</strong>`:''}${d.current_grid?`<br>当前格：GRID ${esc(d.current_grid)}`:''}`;const events=d.planned_requests||[];plans.innerHTML=events.length?'':'<span class="empty-plan">等待稳定识别结果……</span>';for(const p of [...events].reverse()){const el=document.createElement('div');el.className='plan';el.innerHTML=`<strong>#${p.request_id} GRID${p.grid_id} ${esc(p.group)} → ${esc(p.destination)}</strong><br><span class="small">${esc(p.class_name)} · ${Number(p.confidence||0).toFixed(2)} · ${esc(p.status)}</span><div class="route">${(p.route||[]).map(esc).join(' → ')}</div>`;plans.appendChild(el);}}catch(e){mode.textContent='等待视觉服务……';}}
setInterval(refresh,500);refresh();
</script></body></html>"""


def class_group(name: str) -> str | None:
    if name in CUBE_CLASSES:
        return "cube"
    if name in OTHER_CLASSES:
        return "other"
    return None


def stable_value(history: deque[str], votes: int) -> str | None:
    if len(history) < history.maxlen:
        return None
    value, count = Counter(history).most_common(1)[0]
    return value if count >= votes else None


def load_regions(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    camera = raw.get("camera", {}) if isinstance(raw, dict) else {}
    checked = validate_regions(
        {
            "image_width": camera.get("image_width"),
            "image_height": camera.get("image_height"),
            "regions": raw.get("regions") if isinstance(raw, dict) else None,
        }
    )
    return checked["regions"]


class DryRunPlanner:
    """Turn stable visual states into one-shot plans without importing robot code."""

    def __init__(
        self,
        log_path: Path | None = None,
        max_events: int = 50,
        event_status: str = "PLANNED_ONLY",
    ) -> None:
        self.log_path = log_path
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.issued_grids: set[int] = set()
        self.next_request_id = 1
        self.log_error: str | None = None
        self.event_status = event_status

    @staticmethod
    def route(grid_id: int, destination: str) -> list[str]:
        return [
            "GRIPPER_OPEN",
            "pose_01_safe",
            f"grid_{grid_id}_approach",
            f"grid_{grid_id}_grasp",
            "GRIPPER_CLOSE",
            f"grid_{grid_id}_approach",
            "pose_01_safe",
            f"{destination}_approach",
            f"{destination}_release",
            "GRIPPER_OPEN",
            f"{destination}_approach",
            "pose_01_safe",
        ]

    def update(self, states: dict[int, dict[str, Any]]) -> None:
        # An object must leave a region before that region can create another request.
        for grid_id, state in states.items():
            if state.get("state") == "EMPTY":
                self.issued_grids.discard(grid_id)

        for grid_id in sorted(states):
            state = states[grid_id]
            ready_state = state.get("state")
            group = "cube" if ready_state == "READY_CUBE" else "other" if ready_state == "READY_OTHER" else None
            if group is None or state.get("group") != group or grid_id in self.issued_grids:
                continue
            destination = "right" if group == "cube" else "left"
            event = {
                "request_id": self.next_request_id,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "grid_id": grid_id,
                "class_name": state.get("class_name"),
                "group": group,
                "destination": destination,
                "confidence": state.get("confidence"),
                "pose_library": "experiment3_17_pose",
                "route": self.route(grid_id, destination),
                "status": self.event_status,
            }
            self.next_request_id += 1
            self.issued_grids.add(grid_id)
            self.events.append(event)
            self._record(event)

    def _record(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as error:
            self.log_error = str(error)

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self.events)


class Vision:
    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        model_path: Path,
        shape_model_path: Path,
        red_orange_model_path: Path,
        regions_path: Path,
        confidence: float,
        shape_confidence: float,
        red_orange_confidence: float,
        red_orange_margin: float,
        cuda_device: str,
        history_size: int,
        stable_votes: int,
        plan_log_path: Path | None,
        *,
        dry_run: bool = True,
        robot_connected: bool = False,
        plan_status: str = "PLANNED_ONLY",
        video_path: Path | None = None,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.camera_fps = fps
        self.model_path = model_path
        self.shape_model_path = shape_model_path
        self.red_orange_model_path = red_orange_model_path
        self.regions = load_regions(regions_path)
        self.confidence = confidence
        self.shape_confidence = shape_confidence
        self.red_orange_confidence = red_orange_confidence
        self.red_orange_margin = red_orange_margin
        self.cuda_device = cuda_device
        self.history_size = history_size
        self.stable_votes = stable_votes
        self.dry_run = bool(dry_run)
        self.robot_connected = bool(robot_connected)
        self.video_path = video_path
        self.planner = DryRunPlanner(plan_log_path, event_status=plan_status)
        self.histories = {region["id"]: deque(maxlen=history_size) for region in self.regions}
        self.lock = threading.Condition()
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.running = True
        self.error: str | None = None
        self.model_ready = False
        self.runtime_status: dict[str, Any] = {}
        self.status: dict[str, Any] = {
            "model_ready": False,
            "fps": 0.0,
            "detection_count": 0,
            "outside_count": 0,
            "dry_run": self.dry_run,
            "robot_connected": self.robot_connected,
            "planned_count": 0,
            "planned_requests": [],
            "regions": [
                {"grid_id": region["id"], "state": "EMPTY", "group": None, "confidence": None}
                for region in self.regions
            ],
        }
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        with self.lock:
            self.lock.notify_all()
        self.thread.join(timeout=5)

    def wait_for_frame(self, previous: int) -> tuple[int, bytes | None]:
        with self.lock:
            self.lock.wait_for(
                lambda: self.sequence != previous or not self.running or self.error,
                timeout=2,
            )
            return self.sequence, self.jpeg

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.status))

    def set_runtime_status(self, **values: Any) -> None:
        """Expose controller progress in the shared preview without touching vision state."""
        with self.lock:
            self.runtime_status.update(values)
            self.status.update(self.runtime_status)
            self.lock.notify_all()

    def _refine_red_pink(self, frame: np.ndarray, result: Any, shape_model: YOLO) -> None:
        if result.boxes is None or len(result.boxes) == 0:
            return
        candidates: list[int] = []
        crops: list[np.ndarray] = []
        frame_height, frame_width = frame.shape[:2]
        for index, box in enumerate(result.boxes):
            original = result.names[int(box.cls[0])]
            if original not in ("pink_cube", "red_cuboid"):
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            pad_x, pad_y = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
            right, bottom = min(frame_width, int(x2 + pad_x)), min(frame_height, int(y2 + pad_y))
            crop = frame[top:bottom, left:right]
            if crop.size:
                candidates.append(index)
                crops.append(crop)
        if not crops:
            return
        classifications = shape_model.predict(crops, imgsz=224, device=self.cuda_device, verbose=False)
        result.boxes.data = result.boxes.data.clone()
        red_id = next(key for key, value in result.names.items() if value == "red_cuboid")
        for index, classification in zip(candidates, classifications):
            old_name = result.names[int(result.boxes.cls[index])]
            new_name = shape_model.names[int(classification.probs.top1)]
            score = float(classification.probs.top1conf)
            if old_name == "pink_cube" and new_name == "red_cuboid" and score >= self.shape_confidence:
                result.boxes.data[index, 5] = red_id

    def _refine_red_orange(
        self, frame: np.ndarray, result: Any, pair_model: YOLO
    ) -> None:
        """Resolve only the red-cuboid/orange-bridge ambiguity by crop shape."""
        if result.boxes is None or len(result.boxes) == 0:
            return
        pair_names = {"red_cuboid", "orange_bridge"}
        candidates: list[int] = []
        crops: list[np.ndarray] = []
        frame_height, frame_width = frame.shape[:2]
        for index, box in enumerate(result.boxes):
            original = str(result.names[int(box.cls[0])])
            if original not in pair_names:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            pad_x, pad_y = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
            right, bottom = min(frame_width, int(x2 + pad_x)), min(
                frame_height, int(y2 + pad_y)
            )
            crop = frame[top:bottom, left:right]
            if crop.size:
                candidates.append(index)
                crops.append(crop)
        if not crops:
            return
        classifications = pair_model.predict(
            crops, imgsz=224, device=self.cuda_device, verbose=False
        )
        class_ids = {
            str(value): int(key)
            for key, value in result.names.items()
            if str(value) in pair_names
        }
        updated = result.boxes.data.clone()
        for index, classification in zip(candidates, classifications):
            probabilities = classification.probs.data.tolist()
            order = sorted(
                range(len(probabilities)),
                key=lambda item: float(probabilities[item]),
                reverse=True,
            )
            best, runner_up = order[:2]
            name = str(pair_model.names[best])
            confidence = float(probabilities[best])
            margin = confidence - float(probabilities[runner_up])
            if (
                name in class_ids
                and confidence >= self.red_orange_confidence
                and margin >= self.red_orange_margin
            ):
                updated[index, 5] = class_ids[name]
        result.boxes.data = updated

    def _draw_regions(self, frame: np.ndarray, states: dict[int, dict[str, Any]]) -> None:
        for region in self.regions:
            region_id = int(region["id"])
            polygon = np.asarray(region["polygon"], dtype=np.int32)
            state = states[region_id]["state"]
            color = STATE_COLORS.get(state, (255, 255, 255))
            cv2.polylines(frame, [polygon], True, color, 3, cv2.LINE_AA)
            center = polygon.mean(axis=0).astype(int)
            label = f"G{region_id} {state}"
            cv2.putText(frame, label, (center[0] - 55, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (center[0] - 55, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        video_writer: cv2.VideoWriter | None = None
        try:
            detector = YOLO(str(self.model_path))
            shape_model = YOLO(str(self.shape_model_path))
            red_orange_model = YOLO(str(self.red_orange_model_path))
            self.model_ready = True
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {self.device}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            previous = time.perf_counter()
            smoothed_fps = 0.0
            while self.running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("摄像头已打开，但无法读取画面")
                result = detector.predict(
                    frame,
                    imgsz=640,
                    conf=self.confidence,
                    device=self.cuda_device,
                    verbose=False,
                )[0]
                self._refine_red_pink(frame, result, shape_model)
                self._refine_red_orange(frame, result, red_orange_model)
                observations: dict[int, list[dict[str, Any]]] = {region["id"]: [] for region in self.regions}
                detections: list[dict[str, Any]] = []
                outside_count = 0
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        name = result.names[class_id]
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
                        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                        grid_id = locate_grid(center, self.regions)
                        group = class_group(name)
                        detection = {
                            "name": name,
                            "group": group,
                            "confidence": confidence,
                            "box": [x1, y1, x2, y2],
                            "center": center,
                            "grid_id": grid_id,
                        }
                        detections.append(detection)
                        if grid_id is None:
                            outside_count += 1
                        else:
                            observations[grid_id].append(detection)

                states: dict[int, dict[str, Any]] = {}
                for region in self.regions:
                    grid_id = int(region["id"])
                    current = observations[grid_id]
                    if not current:
                        observation = "EMPTY"
                    elif len(current) > 1:
                        observation = "MULTIPLE"
                    elif current[0]["group"] is None:
                        observation = "UNKNOWN"
                    else:
                        observation = str(current[0]["group"])
                    self.histories[grid_id].append(observation)
                    stable = stable_value(self.histories[grid_id], self.stable_votes)
                    if stable is None:
                        state = "CANDIDATE"
                    elif stable == "EMPTY":
                        state = "EMPTY"
                    elif stable == "cube":
                        state = "READY_CUBE"
                    elif stable == "other":
                        state = "READY_OTHER"
                    else:
                        state = stable
                    confidence = current[0]["confidence"] if len(current) == 1 else None
                    group = current[0]["group"] if len(current) == 1 else None
                    class_name = current[0]["name"] if len(current) == 1 else None
                    states[grid_id] = {
                        "grid_id": grid_id,
                        "state": state,
                        "group": group,
                        "class_name": class_name,
                        "confidence": confidence,
                        "observation": observation,
                        "history": list(self.histories[grid_id]),
                    }

                self.planner.update(states)

                display = result.plot()
                self._draw_regions(display, states)
                overlay = (
                    "AUTONOMOUS TASK - ROBOT CONNECTED"
                    if self.robot_connected
                    else "VISION ONLY - ROBOT DISCONNECTED"
                )
                cv2.putText(display, overlay, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
                now = time.perf_counter()
                instant = 1.0 / max(now - previous, 1e-6)
                previous = now
                smoothed_fps = instant if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant
                cv2.putText(display, f"FPS {smoothed_fps:.1f}", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
                if self.video_path is not None:
                    if video_writer is None:
                        self.video_path.parent.mkdir(parents=True, exist_ok=True)
                        video_writer = cv2.VideoWriter(
                            str(self.video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(max(1, self.camera_fps)),
                            (int(display.shape[1]), int(display.shape[0])),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError(f"无法创建实验录像 {self.video_path}")
                    video_writer.write(display)
                encoded, buffer = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 84])
                if not encoded:
                    continue
                with self.lock:
                    self.jpeg = buffer.tobytes()
                    self.sequence += 1
                    self.status = {
                        "model_ready": self.model_ready,
                        "fps": smoothed_fps,
                        "detection_count": len(detections),
                        "outside_count": outside_count,
                        "dry_run": self.dry_run,
                        "robot_connected": self.robot_connected,
                        "planned_count": len(self.planner.events),
                        "planned_requests": self.planner.snapshot(),
                        "plan_log_error": self.planner.log_error,
                        "detections": detections,
                        "regions": [states[index] for index in sorted(states)],
                        **self.runtime_status,
                    }
                    self.lock.notify_all()
        except Exception as error:
            self.error = str(error)
            with self.lock:
                self.status["error"] = self.error
                self.lock.notify_all()
        finally:
            if video_writer is not None:
                video_writer.release()
            if cap is not None:
                cap.release()


class VisionServer(ThreadingHTTPServer):
    vision: Vision


class Handler(BaseHTTPRequestHandler):
    server: VisionServer

    def send_json(self, status: HTTPStatus, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/vision", "/index.html"):
            payload = PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/health":
            status = self.server.vision.get_status()
            code = HTTPStatus.OK if self.server.vision.jpeg else HTTPStatus.SERVICE_UNAVAILABLE
            self.send_json(code, status)
            return
        if self.path == "/api/status":
            self.send_json(HTTPStatus.OK, self.server.vision.get_status())
            return
        if self.path != "/stream.mjpg":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        sequence = -1
        try:
            while self.server.vision.running:
                sequence, frame = self.server.vision.wait_for_frame(sequence)
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        return


def self_test() -> None:
    assert class_group("blue_cube") == "cube"
    assert class_group("yellow_cube") == "cube"
    assert class_group("red_cuboid") == "other"
    assert class_group("orange_bridge") == "other"
    assert class_group("unexpected") is None
    history = deque(["cube"] * 5 + ["EMPTY"] * 2, maxlen=7)
    assert stable_value(history, 5) == "cube"
    assert stable_value(deque(["cube"] * 4 + ["EMPTY"] * 3, maxlen=7), 5) is None
    planner = DryRunPlanner()
    ready = {
        1: {"state": "READY_CUBE", "group": "cube", "class_name": "blue_cube", "confidence": 0.9},
        2: {"state": "READY_OTHER", "group": "other", "class_name": "red_cuboid", "confidence": 0.8},
    }
    planner.update(ready)
    assert len(planner.events) == 2
    assert planner.events[0]["destination"] == "right"
    assert planner.events[1]["destination"] == "left"
    planner.update(ready)
    assert len(planner.events) == 2
    planner.update({1: {"state": "EMPTY"}, 2: ready[2]})
    planner.update({1: ready[1], 2: ready[2]})
    assert len(planner.events) == 3 and planner.events[-1]["request_id"] == 3
    print("[SELF-TEST] class routing, stability, and one-shot dry-run planning passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 YOLO region preview; no robot control")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shape-model", type=Path, required=True)
    parser.add_argument("--red-orange-model", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--shape-confidence", type=float, default=0.55)
    parser.add_argument("--red-orange-confidence", type=float, default=0.80)
    parser.add_argument("--red-orange-margin", type=float, default=0.30)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--history-size", type=int, default=7)
    parser.add_argument("--stable-votes", type=int, default=5)
    parser.add_argument(
        "--plan-log",
        type=Path,
        default=Path("/home/nvidia/mecharm_tasks/experiment3/logs/vision_dry_run_plans.jsonl"),
    )
    parser.add_argument("--record-video", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if (args.width, args.height) != (1280, 720):
        raise SystemExit("视觉区域配置固定使用 1280x720")
    if not 1 <= args.stable_votes <= args.history_size:
        raise SystemExit("稳定票数必须位于 1 和历史窗口大小之间")

    vision = Vision(
        args.device,
        args.width,
        args.height,
        args.fps,
        args.model,
        args.shape_model,
        args.red_orange_model,
        args.regions,
        args.confidence,
        args.shape_confidence,
        args.red_orange_confidence,
        args.red_orange_margin,
        args.cuda_device,
        args.history_size,
        args.stable_votes,
        args.plan_log,
        video_path=args.record_video,
    )
    server = VisionServer(("0.0.0.0", args.port), Handler)
    server.vision = vision

    def stop_server(_signum: int, _frame: object) -> None:
        vision.running = False
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    vision.start()
    print("[MODE] VISION ONLY - robot is not imported or connected", flush=True)
    print(f"[MODEL] {args.model}", flush=True)
    print(f"[REGIONS] {args.regions}", flush=True)
    print(f"[OPEN] http://0.0.0.0:{args.port}/vision", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        vision.stop()
        server.server_close()


if __name__ == "__main__":
    main()
