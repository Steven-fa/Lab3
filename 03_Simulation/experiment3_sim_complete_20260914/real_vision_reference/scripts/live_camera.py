from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from red_pink_refiner import RedPinkRefiner


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time six-object detection from a camera.")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--model", type=Path, default=Path("models/cube6_red_focus_v2_best.pt"))
    parser.add_argument("--shape-model", type=Path, default=Path("models/red_pink_shape_classifier_v2.pt"))
    parser.add_argument("--shape-confidence", type=float, default=0.55)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--screenshots", type=Path, default=Path("runs/live_camera"))
    args = parser.parse_args()

    args.screenshots.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    refiner = RedPinkRefiner(args.shape_model, args.device, args.shape_confidence)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头编号 {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    window_name = f"Cube Detection - Camera {args.camera}  [Q/Esc: exit, S: screenshot]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    previous = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头已打开，但无法读取画面")
            result = model.predict(frame, imgsz=640, conf=args.confidence,
                                   device=args.device, verbose=False)[0]
            refiner.refine(frame, result)
            display = result.plot()
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous, 1e-6)
            previous = now
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.90 * smoothed_fps + 0.10 * instant_fps
            cv2.putText(display, f"FPS: {smoothed_fps:.1f}", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output = args.screenshots / f"camera_{args.camera}_{timestamp}.jpg"
                cv2.imwrite(str(output), display)
                print(f"已保存截图：{output.resolve()}", flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
