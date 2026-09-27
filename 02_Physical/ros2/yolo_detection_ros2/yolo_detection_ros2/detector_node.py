"""Publish camera images and generic YOLO detections as ROS 2 messages."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
from pathlib import Path
import threading

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import yaml

from .red_orange_pair import choose_pair_class


PREVIEW_PAGE = """<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>Experiment 3 Live Detection</title>
<style>body{margin:0;background:#101318;color:#eef;font-family:system-ui}main{max-width:1320px;margin:auto;padding:16px}img{width:100%;border:1px solid #445;border-radius:8px}pre{white-space:pre-wrap;font-size:16px}</style>
</head><body><main><h2>实验3实时识别（ROS 2）</h2><img src=\"/stream.mjpg\"><pre id=\"s\">正在等待检测帧……</pre></main>
<script>setInterval(async()=>{try{let r=await fetch('/status.json',{cache:'no-store'});let x=await r.json();document.getElementById('s').textContent=`FPS: ${x.fps.toFixed(1)}  检测数量: ${x.count}\\n`+x.detections.map(d=>`G${d.grid_id??'-'}  ${d.class_name}  ${(d.confidence*100).toFixed(1)}%`).join('\\n')}catch(e){}},500)</script></body></html>""".encode("utf-8")


class PreviewHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        detector = self.server.detector  # type: ignore[attr-defined]
        if self.path in {"/", "/vision"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PREVIEW_PAGE)))
            self.end_headers()
            self.wfile.write(PREVIEW_PAGE)
            return
        if self.path == "/status.json":
            with detector.preview_condition:
                payload = json.dumps(detector.preview_status, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            try:
                while not detector.finished:
                    with detector.preview_condition:
                        detector.preview_condition.wait_for(
                            lambda: detector.preview_sequence != sequence
                            or detector.finished,
                            timeout=2.0,
                        )
                        frame = detector.preview_jpeg
                        sequence = detector.preview_sequence
                    if not frame:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(404)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def color_for_class(class_id: int) -> tuple[int, int, int]:
    return (
        64 + (37 * class_id) % 192,
        64 + (97 * class_id) % 192,
        64 + (157 * class_id) % 192,
    )


class YoloDetectionPublisher(Node):
    def __init__(self) -> None:
        super().__init__("yolo_detection_publisher")
        self.declare_parameter("model", "/home/nvidia/jetson_yolo/best.pt")
        self.declare_parameter("shape_model", "")
        self.declare_parameter("shape_confidence", 0.55)
        self.declare_parameter("red_orange_model", "")
        self.declare_parameter("red_orange_confidence", 0.80)
        self.declare_parameter("red_orange_margin", 0.30)
        self.declare_parameter("camera", 0)
        self.declare_parameter("source", "")
        self.declare_parameter("camera_width", 1280)
        self.declare_parameter("camera_height", 720)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("image_size", 640)
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("device", "0")
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("topic", "/yolo/detections")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("show", False)
        self.declare_parameter("save_path", "")
        self.declare_parameter("regions_path", "")
        self.declare_parameter("preview_port", 0)

        model_path = Path(str(self.get_parameter("model").value))
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.model = YOLO(str(model_path))
        shape_path_value = str(self.get_parameter("shape_model").value).strip()
        self.shape_model = YOLO(shape_path_value) if shape_path_value else None
        self.shape_confidence = float(self.get_parameter("shape_confidence").value)
        pair_path_value = str(self.get_parameter("red_orange_model").value).strip()
        if pair_path_value and not Path(pair_path_value).is_file():
            raise FileNotFoundError(pair_path_value)
        self.red_orange_model = YOLO(pair_path_value) if pair_path_value else None
        self.red_orange_confidence = float(
            self.get_parameter("red_orange_confidence").value
        )
        self.red_orange_margin = float(self.get_parameter("red_orange_margin").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.confidence = float(self.get_parameter("confidence").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.device = str(self.get_parameter("device").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.show = bool(self.get_parameter("show").value)
        save_path_value = str(self.get_parameter("save_path").value).strip()
        self.save_path = Path(save_path_value).expanduser() if save_path_value else None
        self.regions: list[dict] = []
        regions_path_value = str(self.get_parameter("regions_path").value).strip()
        if regions_path_value:
            regions_path = Path(regions_path_value).expanduser()
            document = yaml.safe_load(regions_path.read_text(encoding="utf-8"))
            self.regions = list(document.get("regions", []))

        detection_topic = str(self.get_parameter("topic").value)
        image_topic = str(self.get_parameter("image_topic").value)
        self.detection_publisher = self.create_publisher(
            Detection2DArray, detection_topic, 10
        )
        self.image_publisher = self.create_publisher(Image, image_topic, 10)
        self.bridge = CvBridge()

        source_value = str(self.get_parameter("source").value).strip()
        camera_index = int(self.get_parameter("camera").value)
        self.is_file_source = bool(source_value)
        if self.is_file_source:
            input_path = Path(source_value).expanduser()
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            capture_source: int | str = str(input_path)
            source_description = str(input_path)
            self.capture = cv2.VideoCapture(capture_source)
        else:
            capture_source = camera_index
            source_description = f"/dev/video{camera_index}"
            self.capture = cv2.VideoCapture(capture_source, cv2.CAP_V4L2)
            if not self.capture.isOpened():
                self.capture.release()
                self.capture = cv2.VideoCapture(capture_source)
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.capture.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                int(self.get_parameter("camera_width").value),
            )
            self.capture.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                int(self.get_parameter("camera_height").value),
            )
            self.capture.set(
                cv2.CAP_PROP_FPS,
                int(self.get_parameter("camera_fps").value),
            )
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open input: {source_description}")

        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not 1.0 <= self.source_fps <= 120.0:
            self.source_fps = 20.0
        self.writer: cv2.VideoWriter | None = None
        self.smoothed_fps = 0.0
        self.previous_frame_time = time.perf_counter()
        self.finished = False
        self.preview_condition = threading.Condition()
        self.preview_jpeg: bytes | None = None
        self.preview_sequence = 0
        self.preview_status: dict = {"fps": 0.0, "count": 0, "detections": []}
        self.preview_server: PreviewHttpServer | None = None
        self.preview_thread: threading.Thread | None = None
        preview_port = int(self.get_parameter("preview_port").value)
        if preview_port > 0:
            self.preview_server = PreviewHttpServer(("127.0.0.1", preview_port), PreviewHandler)
            self.preview_server.detector = self  # type: ignore[attr-defined]
            self.preview_thread = threading.Thread(
                target=self.preview_server.serve_forever,
                name="experiment3-preview",
                daemon=True,
            )
            self.preview_thread.start()
            self.get_logger().info(
                f"Live detection preview: http://127.0.0.1:{preview_port}/vision"
            )
        self.timer = self.create_timer(0.001, self.process_frame)
        self.get_logger().info(
            f"Publishing {image_topic} and {detection_topic} from {source_description}"
        )

    @staticmethod
    def _class_name(result: object, class_id: int) -> str:
        names = result.names  # type: ignore[attr-defined]
        return str(names[class_id])

    def _refine_red_pink(self, frame: object, result: object) -> None:
        if self.shape_model is None or result.boxes is None or len(result.boxes) == 0:  # type: ignore[attr-defined]
            return
        candidates: list[int] = []
        crops: list[object] = []
        frame_height, frame_width = frame.shape[:2]  # type: ignore[attr-defined]
        for index, box in enumerate(result.boxes):  # type: ignore[attr-defined]
            old_name = self._class_name(result, int(box.cls[0]))
            if old_name not in {"pink_cube", "red_cuboid"}:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            pad_x, pad_y = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
            right = min(frame_width, int(x2 + pad_x))
            bottom = min(frame_height, int(y2 + pad_y))
            crop = frame[top:bottom, left:right]  # type: ignore[index]
            if crop.size:  # type: ignore[attr-defined]
                candidates.append(index)
                crops.append(crop)
        if not crops:
            return
        classifications = self.shape_model.predict(
            crops, imgsz=224, device=self.device, verbose=False
        )
        result.boxes.data = result.boxes.data.clone()  # type: ignore[attr-defined]
        red_id = next(
            key for key, value in result.names.items() if value == "red_cuboid"  # type: ignore[attr-defined]
        )
        for index, classification in zip(candidates, classifications):
            old_name = self._class_name(result, int(result.boxes.cls[index]))  # type: ignore[attr-defined]
            new_name = str(self.shape_model.names[int(classification.probs.top1)])
            score = float(classification.probs.top1conf)
            if (
                old_name == "pink_cube"
                and new_name == "red_cuboid"
                and score >= self.shape_confidence
            ):
                result.boxes.data[index, 5] = red_id  # type: ignore[attr-defined]

    def _refine_red_orange(self, frame: object, result: object) -> None:
        """Use a shape-focused second stage only for the red/orange pair."""
        if self.red_orange_model is None or result.boxes is None or len(result.boxes) == 0:  # type: ignore[attr-defined]
            return
        candidates: list[int] = []
        crops: list[object] = []
        frame_height, frame_width = frame.shape[:2]  # type: ignore[attr-defined]
        for index, box in enumerate(result.boxes):  # type: ignore[attr-defined]
            old_name = self._class_name(result, int(box.cls[0]))
            if old_name not in {"red_cuboid", "orange_bridge"}:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            pad_x, pad_y = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
            right = min(frame_width, int(x2 + pad_x))
            bottom = min(frame_height, int(y2 + pad_y))
            crop = frame[top:bottom, left:right]  # type: ignore[index]
            if crop.size:  # type: ignore[attr-defined]
                candidates.append(index)
                crops.append(crop)
        if not crops:
            return
        classifications = self.red_orange_model.predict(
            crops, imgsz=224, device=self.device, verbose=False
        )
        class_ids = {
            str(value): int(key)
            for key, value in result.names.items()  # type: ignore[attr-defined]
            if str(value) in {"red_cuboid", "orange_bridge"}
        }
        updated = result.boxes.data.clone()  # type: ignore[attr-defined]
        for index, classification in zip(candidates, classifications):
            old_name = self._class_name(result, int(result.boxes.cls[index]))  # type: ignore[attr-defined]
            decision = choose_pair_class(
                old_name,
                self.red_orange_model.names,
                classification.probs.data.tolist(),
                min_confidence=self.red_orange_confidence,
                min_margin=self.red_orange_margin,
            )
            if decision.class_name in class_ids:
                updated[index, 5] = class_ids[decision.class_name]
        result.boxes.data = updated  # type: ignore[attr-defined]

    def _grid_for(self, center: tuple[float, float]) -> int | None:
        matches: list[int] = []
        for region in self.regions:
            polygon = np.asarray(region.get("polygon", []), dtype=np.float32)
            if len(polygon) >= 3 and cv2.pointPolygonTest(polygon, center, False) >= 0:
                matches.append(int(region["id"]))
        return matches[0] if len(matches) == 1 else None

    def process_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            if self.is_file_source:
                self.get_logger().info("Input video finished")
                self.finished = True
                self.timer.cancel()
                return
            self.get_logger().warning("Camera did not return a frame")
            return

        stamp = self.get_clock().now().to_msg()
        image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_message.header.stamp = stamp
        image_message.header.frame_id = self.frame_id
        self.image_publisher.publish(image_message)

        result = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]
        self._refine_red_pink(frame, result)
        self._refine_red_orange(frame, result)

        message = Detection2DArray()
        message.header.stamp = stamp
        message.header.frame_id = self.frame_id
        rendered = frame.copy()
        for region in self.regions:
            polygon = np.asarray(region.get("polygon", []), dtype=np.int32)
            if len(polygon) < 3:
                continue
            cv2.polylines(rendered, [polygon], True, (255, 210, 60), 2)
            anchor = tuple(int(value) for value in polygon[0])
            cv2.putText(
                rendered,
                f"G{region['id']}",
                (anchor[0] + 4, anchor[1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 210, 60),
                2,
                cv2.LINE_AA,
            )
        boxes = result.boxes if result.boxes is not None else []
        preview_detections: list[dict] = []
        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = self._class_name(result, class_id)
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detection = Detection2D()
            detection.header = message.header
            detection.bbox.center.position.x = (x1 + x2) / 2.0
            detection.bbox.center.position.y = (y1 + y2) / 2.0
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = x2 - x1
            detection.bbox.size_y = y2 - y1
            detection.id = f"{class_name}_{len(message.detections)}"
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_name
            hypothesis.hypothesis.score = confidence
            detection.results.append(hypothesis)
            message.detections.append(detection)
            grid_id = self._grid_for(
                (detection.bbox.center.position.x, detection.bbox.center.position.y)
            )
            preview_detections.append(
                {
                    "grid_id": grid_id,
                    "class_name": class_name,
                    "confidence": confidence,
                }
            )

            left, top, right, bottom = map(round, (x1, y1, x2, y2))
            color = color_for_class(class_id)
            cv2.rectangle(rendered, (left, top), (right, bottom), color, 2)
            cv2.putText(
                rendered,
                f"G{grid_id or '-'} {class_name} {confidence:.2f}",
                (left, max(25, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )

        try:
            self.detection_publisher.publish(message)
        except Exception:
            if rclpy.ok():
                raise
            return
        now = time.perf_counter()
        instant_fps = 1.0 / max(now - self.previous_frame_time, 1e-6)
        self.previous_frame_time = now
        self.smoothed_fps = (
            0.85 * self.smoothed_fps + 0.15 * instant_fps
            if self.smoothed_fps
            else instant_fps
        )
        cv2.putText(
            rendered,
            f"FPS: {self.smoothed_fps:.1f}  detections: {len(message.detections)}  ROS 2: ON",
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        encoded, jpeg = cv2.imencode(
            ".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
        )
        if encoded:
            with self.preview_condition:
                self.preview_jpeg = jpeg.tobytes()
                self.preview_sequence += 1
                self.preview_status = {
                    "fps": self.smoothed_fps,
                    "count": len(preview_detections),
                    "detections": preview_detections,
                }
                self.preview_condition.notify_all()

        if self.save_path and self.writer is None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = rendered.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.save_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.source_fps,
                (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open output video: {self.save_path}")
        if self.writer is not None:
            self.writer.write(rendered)
        if self.show:
            cv2.imshow("YOLO ROS 2 detections", rendered)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                rclpy.shutdown()

    def destroy_node(self) -> bool:
        self.finished = True
        with self.preview_condition:
            self.preview_condition.notify_all()
        if self.preview_server is not None:
            self.preview_server.shutdown()
            self.preview_server.server_close()
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=2.0)
        self.capture.release()
        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: YoloDetectionPublisher | None = None
    try:
        node = YoloDetectionPublisher()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
