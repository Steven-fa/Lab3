from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


PINK_CUBE_ID = 2
RED_CUBOID_ID = 4


class RedPinkRefiner:
    """Second-stage classifier for detections confused between pink cube and red cuboid."""

    def __init__(self, model_path: Path | str, device: str = "0", min_confidence: float = 0.55):
        self.model = YOLO(str(model_path))
        self.device = device
        self.min_confidence = min_confidence

    def refine(self, frame: np.ndarray, result) -> list[dict]:
        if result.boxes is None or len(result.boxes) == 0:
            return []

        height, width = frame.shape[:2]
        candidate_indices: list[int] = []
        crops: list[np.ndarray] = []
        for index, box in enumerate(result.boxes):
            detector_class = int(box.cls[0])
            if detector_class not in (PINK_CUBE_ID, RED_CUBOID_ID):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pad_x = 0.08 * (x2 - x1)
            pad_y = 0.08 * (y2 - y1)
            left = max(0, int(x1 - pad_x))
            top = max(0, int(y1 - pad_y))
            right = min(width, int(x2 + pad_x))
            bottom = min(height, int(y2 + pad_y))
            crop = frame[top:bottom, left:right]
            if crop.size:
                candidate_indices.append(index)
                crops.append(crop)

        if not crops:
            return []

        classifications = self.model.predict(crops, imgsz=224, device=self.device, verbose=False)
        # Ultralytics returns an inference tensor that cannot be edited in place.
        result.boxes.data = result.boxes.data.clone()
        changes: list[dict] = []
        for box_index, classification in zip(candidate_indices, classifications):
            top1 = int(classification.probs.top1)
            confidence = float(classification.probs.top1conf)
            class_name = self.model.names[top1]
            refined_class = RED_CUBOID_ID if class_name == "red_cuboid" else PINK_CUBE_ID
            old_class = int(result.boxes.cls[box_index])
            # The observed failure is one-way (red cuboid -> pink cube). Only promote
            # pink detections to red; never downgrade an already-correct red detection.
            should_apply = (
                old_class == PINK_CUBE_ID
                and refined_class == RED_CUBOID_ID
                and confidence >= self.min_confidence
            )
            if should_apply:
                result.boxes.data[box_index, 5] = refined_class
            changes.append({
                "box_index": box_index,
                "old_class": old_class,
                "suggested_class": refined_class,
                "applied": should_apply,
                "classifier_confidence": confidence,
            })
        return changes
