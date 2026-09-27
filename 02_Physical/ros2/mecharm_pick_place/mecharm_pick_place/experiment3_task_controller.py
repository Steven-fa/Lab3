"""ROS 2 state machine for six-region autonomous classification sorting."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
import yaml


CUBE_CLASSES = {"blue_cube", "green_cube", "pink_cube", "yellow_cube"}
OTHER_CLASSES = {"red_cuboid", "orange_bridge"}
GROUP_DESTINATION = {"cube": "right", "other": "left"}
READY_GROUP = {"READY_CUBE": "cube", "READY_OTHER": "other"}


def class_group(name: str) -> str | None:
    if name in CUBE_CLASSES:
        return "cube"
    if name in OTHER_CLASSES:
        return "other"
    return None


def point_in_polygon(point: tuple[float, float], polygon: list[list[int]]) -> bool:
    x, y = point
    inside = False
    previous = len(polygon) - 1
    for current, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[previous]
        if (yi > y) != (yj > y):
            crossing_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def locate_grid(
    point: tuple[float, float], regions: list[dict[str, Any]]
) -> int | None:
    matches = [
        int(region["id"])
        for region in regions
        if point_in_polygon(point, region["polygon"])
    ]
    return matches[0] if len(matches) == 1 else None


def stable_value(history: deque[str], votes: int) -> str | None:
    if len(history) < history.maxlen:
        return None
    value, count = Counter(history).most_common(1)[0]
    return value if count >= votes else None


def collapse_region_observations(
    current: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Resolve detector output for one fixed cell into one routing observation.

    YOLO NMS is class-aware, so one physical cube can occasionally produce two
    overlapping colour-class boxes.  Experiment 3 routes every cube class to
    the same destination, therefore same-group duplicates are one actionable
    observation.  Mixed groups stay unresolved and never trigger motion.
    """
    if not current:
        return "EMPTY", {}
    groups = {item.get("group") for item in current}
    if None in groups:
        return "UNKNOWN", {}
    if len(groups) != 1:
        return "MULTIPLE", {}
    representative = max(
        current, key=lambda item: float(item.get("confidence") or 0.0)
    )
    return str(representative["group"]), representative


def should_auto_complete(
    completed_count: int,
    idle_elapsed: float,
    idle_complete_seconds: float,
    *,
    all_remaining_empty: bool,
) -> bool:
    """Finish only after work occurred and every current region is empty."""
    return (
        completed_count > 0
        and all_remaining_empty
        and idle_elapsed >= idle_complete_seconds
    )


def all_regions_are_empty(region_states: dict[int, dict[str, Any]]) -> bool:
    """Finish only when every currently observed region is stably empty."""
    if not region_states:
        return False
    return all(
        str(state.get("state")) == "EMPTY"
        for state in region_states.values()
    )


def post_transfer_source_decision(state: str) -> str:
    """Resolve the source view after one completed robot transfer.

    A stable READY state can be either an immediately replenished object or the
    original object still present.  Both cases must be queued as the next job;
    treating READY as a fatal error prevents continuous sorting and cannot
    distinguish those two physical situations from one camera frame.
    """
    if state == "EMPTY":
        return "ACCEPT_EMPTY"
    if state in READY_GROUP:
        return "ACCEPT_REQUEUE"
    return "WAIT"


def build_ready_target(
    region_states: dict[int, dict[str, Any]], request_id: int
) -> dict[str, Any] | None:
    """Build the next job without excluding a previously processed grid."""
    for grid in sorted(region_states):
        region = region_states[grid]
        group = READY_GROUP.get(str(region.get("state")))
        if group is None or region.get("group") != group:
            continue
        return {
            "request_id": request_id,
            "grid_id": grid,
            "class_name": str(region.get("class_name") or group),
            "group": group,
            "destination": GROUP_DESTINATION[group],
            "confidence": float(region.get("confidence") or 0.0),
        }
    return None


def transfer_has_timed_out(
    started_at: float, now: float, timeout_seconds: float
) -> bool:
    return now - started_at > timeout_seconds


def load_regions(path: Path) -> list[dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    camera = document.get("camera", {}) if isinstance(document, dict) else {}
    if (
        int(camera.get("image_width", 0)) != 1280
        or int(camera.get("image_height", 0)) != 720
    ):
        raise RuntimeError("regions.yaml必须使用1280x720标定")
    regions = document.get("regions") if isinstance(document, dict) else None
    if not isinstance(regions, list) or len(regions) != 6:
        raise RuntimeError("regions.yaml必须包含六个区域")
    if sorted(int(region.get("id", 0)) for region in regions) != list(range(1, 7)):
        raise RuntimeError("regions.yaml区域编号必须为1～6")
    for region in regions:
        polygon = region.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise RuntimeError(f"GRID {region.get('id')}多边形无效")
    return regions


class Experiment3TaskController(Node):
    def __init__(self) -> None:
        super().__init__("experiment3_task_controller")
        self.declare_parameter(
            "regions_path",
            "/home/nvidia/mecharm_tasks/experiment3/config/regions.yaml",
        )
        self.declare_parameter("armed", False)
        self.declare_parameter("history_size", 7)
        self.declare_parameter("stable_votes", 5)
        self.declare_parameter("idle_complete_seconds", 5.0)
        self.declare_parameter("inter_object_delay", 5.0)
        self.declare_parameter("transfer_timeout_seconds", 180.0)
        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("request_topic", "/experiment3/transfer_requests")
        self.declare_parameter("feedback_topic", "/experiment3/transfer_feedback")
        self.declare_parameter("status_topic", "/experiment3/status")
        self.declare_parameter(
            "log_path",
            "/home/nvidia/mecharm_tasks/experiment3/ros2_runs/task_events.jsonl",
        )

        self.armed = bool(self.get_parameter("armed").value)
        self.history_size = int(self.get_parameter("history_size").value)
        self.stable_votes = int(self.get_parameter("stable_votes").value)
        if not 1 <= self.stable_votes <= self.history_size:
            raise RuntimeError("stable_votes必须位于1和history_size之间")
        self.idle_complete_seconds = float(
            self.get_parameter("idle_complete_seconds").value
        )
        self.inter_object_delay = float(
            self.get_parameter("inter_object_delay").value
        )
        self.transfer_timeout_seconds = float(
            self.get_parameter("transfer_timeout_seconds").value
        )
        if self.transfer_timeout_seconds <= 0.0:
            raise RuntimeError("transfer_timeout_seconds必须大于0")
        self.regions = load_regions(
            Path(str(self.get_parameter("regions_path").value))
        )
        self.histories = {
            int(region["id"]): deque(maxlen=self.history_size)
            for region in self.regions
        }
        self.region_states: dict[int, dict[str, Any]] = {}
        self.completed_grids: set[int] = set()
        self.completed_transfer_count = 0
        self.logged_non_targets: set[tuple[int, str]] = set()
        self.in_flight: dict[str, Any] | None = None
        self.in_flight_started_at: float | None = None
        self.pending_completion: dict[str, Any] | None = None
        self.failed = False
        self.task_complete = False
        self.idle_since: float | None = None
        self.next_request_id = 1
        self.next_allowed_at = 0.0
        self.shutdown_at: float | None = None
        self.exit_signal_sent = False
        self.last_region_signature: tuple[tuple[int, str], ...] | None = None
        self.log_path = Path(str(self.get_parameter("log_path").value))

        self.request_publisher = self.create_publisher(
            String, str(self.get_parameter("request_topic").value), 10
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.create_subscription(
            Detection2DArray,
            str(self.get_parameter("detections_topic").value),
            self.on_detections,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("feedback_topic").value),
            self.on_feedback,
            10,
        )
        self.timer = self.create_timer(0.25, self.tick)
        self._record("controller_started", armed=self.armed)
        if self.armed:
            self.get_logger().warning(
                "ARMED: stable visual detections can trigger real robot motion"
            )
        else:
            self.get_logger().info(
                "DISARMED: publishing visual state only; no transfer request will be sent"
            )

    def _record(self, event: str, **values: Any) -> None:
        record = {
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event,
            **values,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _detection_values(detection: Any) -> tuple[str, float] | None:
        if not detection.results:
            return None
        hypothesis = detection.results[0].hypothesis
        return str(hypothesis.class_id), float(hypothesis.score)

    def on_detections(self, message: Detection2DArray) -> None:
        observations: dict[int, list[dict[str, Any]]] = {
            int(region["id"]): [] for region in self.regions
        }
        for detection in message.detections:
            values = self._detection_values(detection)
            if values is None:
                continue
            class_name, confidence = values
            center = (
                float(detection.bbox.center.position.x),
                float(detection.bbox.center.position.y),
            )
            grid = locate_grid(center, self.regions)
            if grid is not None:
                observations[grid].append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "group": class_group(class_name),
                    }
                )

        states: dict[int, dict[str, Any]] = {}
        for grid, current in observations.items():
            observation, representative = collapse_region_observations(current)
            self.histories[grid].append(observation)
            stable = stable_value(self.histories[grid], self.stable_votes)
            if stable is None:
                state = "CANDIDATE"
            elif stable == "cube":
                state = "READY_CUBE"
            elif stable == "other":
                state = "READY_OTHER"
            else:
                state = stable
            states[grid] = {
                "grid_id": grid,
                "state": state,
                "group": representative.get("group"),
                "class_name": representative.get("class_name"),
                "confidence": representative.get("confidence"),
            }
        self.region_states = states
        signature = tuple(
            (grid, str(state["state"])) for grid, state in sorted(states.items())
        )
        if signature != self.last_region_signature:
            self.last_region_signature = signature
            summary = ", ".join(
                f"G{grid}={state}" for grid, state in signature
            )
            self.get_logger().info(f"视觉区域状态：{summary}")
            self._record(
                "vision_state",
                regions=[states[grid] for grid in sorted(states)],
            )

    def on_feedback(self, message: String) -> None:
        try:
            feedback = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().error("Ignoring invalid worker feedback JSON")
            return
        if self.in_flight is None:
            return
        if int(feedback.get("request_id", -1)) != int(self.in_flight["request_id"]):
            return
        status = str(feedback.get("status", ""))
        self._record("worker_feedback", **feedback)
        if status == "COMPLETED":
            completed_request = dict(self.in_flight)
            grid = int(completed_request["grid_id"])
            now = time.monotonic()
            self.pending_completion = {
                **completed_request,
                "motion_log": feedback.get("motion_log"),
            }
            self.next_allowed_at = now + self.inter_object_delay
            self.get_logger().info(
                f"GRID {grid} motion completed; waiting for stable source state"
            )
            self.in_flight = None
            self.in_flight_started_at = None
            self.idle_since = None
        elif status == "FAILED":
            self.failed = True
            self.get_logger().error(
                f"Transfer failed safely: {feedback.get('error', 'unknown error')}"
            )
            self.in_flight = None
            self.in_flight_started_at = None

    def _ready_target(self) -> dict[str, Any] | None:
        return build_ready_target(self.region_states, self.next_request_id)

    def _publish_status(self, state: str) -> None:
        message = String()
        message.data = json.dumps(
            {
                "state": state,
                "armed": self.armed,
                "completed_grids": sorted(self.completed_grids),
                "completed_transfer_count": self.completed_transfer_count,
                "in_flight": self.in_flight,
                "pending_source_grid": (
                    int(self.pending_completion["grid_id"])
                    if self.pending_completion is not None
                    else None
                ),
                "pending_clear_grid": (
                    int(self.pending_completion["grid_id"])
                    if self.pending_completion is not None
                    else None
                ),
                "regions": [self.region_states[key] for key in sorted(self.region_states)],
            },
            ensure_ascii=False,
        )
        self.status_publisher.publish(message)

    def tick(self) -> None:
        if self.failed:
            self._publish_status("FAILED_SAFE_STOP")
            self.shutdown_at = self.shutdown_at or (time.monotonic() + 1.0)
            if time.monotonic() >= self.shutdown_at and not self.exit_signal_sent:
                self.exit_signal_sent = True
                os.kill(os.getpid(), signal.SIGINT)
            return
        if self.task_complete:
            self._publish_status("COMPLETED")
            self.shutdown_at = self.shutdown_at or (time.monotonic() + 1.0)
            if time.monotonic() >= self.shutdown_at and not self.exit_signal_sent:
                self.exit_signal_sent = True
                os.kill(os.getpid(), signal.SIGINT)
            return
        if not self.region_states:
            self._publish_status("WAITING_FOR_DETECTIONS")
            return
        for grid, region in self.region_states.items():
            state = str(region.get("state"))
            key = (grid, state)
            if state in {"EMPTY", "UNKNOWN", "MULTIPLE"} and key not in self.logged_non_targets:
                self.logged_non_targets.add(key)
                self._record(
                    "region_skipped",
                    grid_id=grid,
                    state=state,
                    action="SKIP_NO_MOTION",
                )
        if self.in_flight is not None:
            assert self.in_flight_started_at is not None
            elapsed = time.monotonic() - self.in_flight_started_at
            if transfer_has_timed_out(
                self.in_flight_started_at,
                time.monotonic(),
                self.transfer_timeout_seconds,
            ):
                request = self.in_flight
                self.failed = True
                self.in_flight = None
                self.in_flight_started_at = None
                self._record(
                    "transfer_timeout",
                    request=request,
                    elapsed_seconds=round(elapsed, 3),
                    timeout_seconds=self.transfer_timeout_seconds,
                )
                self.get_logger().error(
                    f"GRID {request['grid_id']} transfer exceeded "
                    f"{self.transfer_timeout_seconds:.1f}s; stopping the launch"
                )
                self._publish_status("FAILED_TRANSFER_TIMEOUT")
                return
            self._publish_status("EXECUTING")
            return
        if time.monotonic() < self.next_allowed_at:
            self._publish_status("CLEAR_RELEASE_AREA")
            return

        if self.pending_completion is not None:
            pending = self.pending_completion
            grid = int(pending["grid_id"])
            state = str(self.region_states.get(grid, {}).get("state", "CANDIDATE"))
            decision = post_transfer_source_decision(state)
            if decision in {"ACCEPT_EMPTY", "ACCEPT_REQUEUE"}:
                self.completed_grids.add(grid)
                self.completed_transfer_count += 1
                self.pending_completion = None
                self._record(
                    "transfer_accepted",
                    request_id=int(pending["request_id"]),
                    grid_id=grid,
                    source_state_after=state,
                    motion_log=pending.get("motion_log"),
                )
                if decision == "ACCEPT_EMPTY":
                    self._record(
                        "source_clear_verified",
                        request_id=int(pending["request_id"]),
                        grid_id=grid,
                    )
                    self.get_logger().info(
                        f"GRID {grid} source is empty; transfer accepted"
                    )
                else:
                    self._record(
                        "source_ready_requeued",
                        request_id=int(pending["request_id"]),
                        grid_id=grid,
                        state=state,
                    )
                    self.get_logger().info(
                        f"GRID {grid} reports {state} after completion; "
                        "previous transfer accepted and current object queued as the next job"
                    )
            else:
                self._publish_status("VERIFYING_SOURCE_STATE")
                return

        target = self._ready_target()
        if target is not None:
            self.idle_since = None
            if not self.armed:
                self._publish_status("READY_DISARMED")
                return
            self.next_request_id += 1
            self.in_flight = target
            self.in_flight_started_at = time.monotonic()
            request = String()
            request.data = json.dumps(target, ensure_ascii=False)
            self.request_publisher.publish(request)
            self._record("transfer_requested", **target)
            self.get_logger().info(
                f"GRID {target['grid_id']} {target['class_name']} -> {target['destination']}"
            )
            self._publish_status("REQUESTED")
            return

        all_empty = all_regions_are_empty(self.region_states)
        if not all_empty:
            self.idle_since = None
            self._publish_status("SCANNING_UNRESOLVED_REGIONS")
            return
        self.idle_since = self.idle_since or time.monotonic()
        if should_auto_complete(
            self.completed_transfer_count,
            time.monotonic() - self.idle_since,
            self.idle_complete_seconds,
            all_remaining_empty=all_empty,
        ):
            self.task_complete = True
            terminal_states = {
                int(grid): str(state.get("state"))
                for grid, state in self.region_states.items()
            }
            self._record(
                "task_completed",
                completed_grids=sorted(self.completed_grids),
                completed_count=self.completed_transfer_count,
                empty_grids=sorted(
                    grid for grid, state in terminal_states.items() if state == "EMPTY"
                ),
                skipped_states=terminal_states,
            )
            self._publish_status("COMPLETED")
            self.get_logger().info(
                f"Experiment 3 completed with {self.completed_transfer_count} transfers"
            )
        else:
            state = (
                "WAITING_FOR_OBJECTS"
                if self.completed_transfer_count == 0
                else "NO_TARGETS_WAITING"
            )
            self._publish_status(state)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Experiment3TaskController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
