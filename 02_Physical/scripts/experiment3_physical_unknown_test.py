#!/usr/bin/env python3
"""Verify a physical untrained object without enabling robot motion.

The empty grid is captured first.  After the operator places an object, this
script requires both (a) visible foreground occupancy and (b) no v4 YOLO box
inside the selected grid.  It then publishes an explicit unsupported-class
detection to the DISARMED Experiment 3 controller.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
import yaml


def load_polygon(regions_path: Path, grid_id: int) -> np.ndarray:
    document = yaml.safe_load(regions_path.read_text(encoding="utf-8"))
    for region in document.get("regions", []):
        if int(region.get("id", -1)) == grid_id:
            polygon = np.asarray(region.get("polygon", []), dtype=np.int32)
            if len(polygon) < 3:
                raise RuntimeError(f"区域{grid_id}的多边形点不足")
            return polygon
    raise RuntimeError(f"regions.yaml中没有区域{grid_id}")


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开/dev/video{index}")
    return capture


def frame_batch(
    capture: cv2.VideoCapture, count: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    frames: list[np.ndarray] = []
    sharpness: list[float] = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("摄像头读取失败")
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    best_index = int(np.argmax(sharpness))
    metrics = {
        "minimum": round(float(np.min(sharpness)), 3),
        "median": round(float(np.median(sharpness)), 3),
        "maximum": round(float(np.max(sharpness)), 3),
        "sharpest_frame_in_batch": best_index,
    }
    return (
        np.median(np.stack(frames), axis=0).astype(np.uint8),
        frames[best_index],
        metrics,
    )


def occupancy_metrics(
    baseline: np.ndarray, current: np.ndarray, polygon: np.ndarray
) -> tuple[float, float, np.ndarray]:
    base_gray = cv2.GaussianBlur(cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    now_gray = cv2.GaussianBlur(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    roi = np.zeros(base_gray.shape, dtype=np.uint8)
    cv2.fillPoly(roi, [polygon], 255)

    base_level = float(np.median(base_gray[roi > 0]))
    now_level = float(np.median(now_gray[roi > 0]))
    corrected = np.clip(now_gray.astype(np.float32) - (now_level - base_level), 0, 255).astype(np.uint8)
    difference = cv2.absdiff(base_gray, corrected)
    changed = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)[1]
    changed = cv2.bitwise_and(changed, roi)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    roi_area = max(1, cv2.countNonZero(roi))
    changed_ratio = cv2.countNonZero(changed) / roi_area
    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_ratio = max((cv2.contourArea(c) for c in contours), default=0.0) / roi_area
    return changed_ratio, largest_ratio, changed


def known_detections(model: YOLO, frame: np.ndarray, polygon: np.ndarray) -> list[dict]:
    result = model.predict(frame, imgsz=640, conf=0.25, iou=0.45, device="0", verbose=False)[0]
    found: list[dict] = []
    for box in result.boxes or []:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if cv2.pointPolygonTest(polygon.astype(np.float32), center, False) < 0:
            continue
        class_id = int(box.cls[0].item())
        found.append(
            {
                "class_name": str(result.names[class_id]),
                "confidence": round(float(box.conf[0].item()), 4),
                "center": [round(center[0], 1), round(center[1], 1)],
            }
        )
    return found


class UnknownPublisher(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("experiment3_physical_unknown_publisher")
        self.publisher = self.create_publisher(Detection2DArray, topic, 10)

    def send(self, center: tuple[float, float], count: int = 20) -> None:
        deadline = time.monotonic() + 5.0
        while self.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("等待DISARMED任务控制节点订阅超时")
            rclpy.spin_once(self, timeout_sec=0.10)
        for _ in range(count):
            message = Detection2DArray()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "physical_unknown_acceptance"
            detection = Detection2D()
            detection.header = message.header
            detection.id = "physical_unrecognized_object"
            detection.bbox.center.position.x = center[0]
            detection.bbox.center.position.y = center[1]
            detection.bbox.size_x = 30.0
            detection.bbox.size_y = 30.0
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = "unrecognized_object"
            hypothesis.hypothesis.score = 0.99
            detection.results.append(hypothesis)
            message.detections.append(detection)
            self.publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--topic", default="/yolo/detections")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Reuse a previously captured empty-grid image and inspect the current scene immediately.",
    )
    args = parser.parse_args()

    polygon = load_polygon(args.regions, args.grid)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    capture = open_camera(args.camera, 1280, 720)
    try:
        for _ in range(20):
            if not capture.read()[0]:
                raise RuntimeError("摄像头预热失败")
        if args.baseline is not None:
            baseline = cv2.imread(str(args.baseline))
            if baseline is None:
                raise RuntimeError(f"无法读取复用基线：{args.baseline}")
            baseline_clear = baseline.copy()
            baseline_sharpness: dict[str, float] = {}
            print(f"[基线] 复用已保存的空格图像：{args.baseline}", flush=True)
        else:
            print(f"[基线] 正在记录空的{args.grid}号格，请勿放入物体……", flush=True)
            baseline, baseline_clear, baseline_sharpness = frame_batch(capture, 120)
            input(
                f"[现场操作] 现在把一个训练类别之外的物体放入{args.grid}号格中心；"
                "手离开画面后输入 READY 并按Enter："
            )
        cv2.imwrite(str(args.evidence_dir / "empty_baseline.jpg"), baseline)
        cv2.imwrite(str(args.evidence_dir / "empty_baseline_clear.jpg"), baseline_clear)
        for _ in range(10):
            if not capture.read()[0]:
                raise RuntimeError("放置后摄像头读取失败")
        current, current_clear, current_sharpness = frame_batch(capture, 120)
    finally:
        capture.release()

    changed_ratio, largest_ratio, mask = occupancy_metrics(baseline, current, polygon)
    model = YOLO(str(args.model))
    known = known_detections(model, current_clear, polygon)
    cv2.imwrite(str(args.evidence_dir / "unknown_object.jpg"), current_clear)
    cv2.imwrite(str(args.evidence_dir / "unknown_object_median.jpg"), current)
    cv2.imwrite(str(args.evidence_dir / "foreground_mask.png"), mask)
    evidence = {
        "grid_id": args.grid,
        "changed_ratio": round(changed_ratio, 5),
        "largest_changed_component_ratio": round(largest_ratio, 5),
        "known_yolo_detections": known,
        "occupancy_pass": changed_ratio >= 0.035 and largest_ratio >= 0.02,
        "unknown_pass": not known,
        "baseline_sharpness": baseline_sharpness,
        "unknown_object_sharpness": current_sharpness,
    }
    (args.evidence_dir / "physical_unknown_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[实物占用] 变化比例={changed_ratio:.1%}，最大连通区域={largest_ratio:.1%}",
        flush=True,
    )
    if known:
        raise RuntimeError(f"该实物被v4识别为已知类别：{known}")
    if changed_ratio < 0.035 or largest_ratio < 0.02:
        raise RuntimeError("没有可靠检测到区域内的实物变化；请换明显一些的物体重试")

    center = tuple(float(value) for value in polygon.mean(axis=0))
    rclpy.init()
    node = UnknownPublisher(args.topic)
    try:
        node.send(center)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print("[实物判定] 区域有物体且v4无已知分类，发布UNKNOWN", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
