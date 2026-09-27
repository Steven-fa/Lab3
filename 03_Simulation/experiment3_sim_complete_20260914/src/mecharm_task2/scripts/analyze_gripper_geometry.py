#!/usr/bin/env python3
"""Measure the official gripper mesh envelope in the live Gazebo pose."""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from gazebo_msgs.msg import LinkStates
from rclpy.node import Node
from scipy.spatial import cKDTree


VISUAL_ORIGINS = {
    "gripper_base": (0.0, 0.0, -0.012),
    "gripper_left1": (0.039, -0.0133, -0.012),
    "gripper_left2": (0.005, -0.0195, -0.012),
    "gripper_left3": (0.012, 0.0025, -0.012),
    "gripper_right1": (-0.039, -0.0133, -0.012),
    "gripper_right2": (-0.005, -0.0195, -0.012),
    "gripper_right3": (-0.012, 0.0025, -0.012),
}


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def mesh_points(path):
    root = ET.parse(path).getroot()
    unit = 1.0
    for item in root.iter():
        if item.tag.endswith("unit"):
            unit = float(item.attrib.get("meter", "1"))
            break
    geometry_points = {}
    for geometry in root.iter():
        if not geometry.tag.endswith("geometry"):
            continue
        arrays = []
        for item in geometry.iter():
            ident = item.attrib.get("id", "").lower()
            if item.tag.endswith("float_array") and "position" in ident:
                values = np.array([float(v) for v in (item.text or "").split()])
                if len(values) % 3 == 0:
                    arrays.append(values.reshape((-1, 3)))
        if arrays:
            geometry_points[geometry.attrib["id"]] = np.vstack(arrays) * unit
    if not geometry_points:
        raise RuntimeError(f"No position array in {path}")

    # Gazebo applies the matrices in COLLADA's visual scene.  The previous
    # calibration read raw geometry arrays only, which omitted these transforms
    # and made the apparent fingertip centre disagree with the rendered CAD.
    library_nodes = {
        item.attrib["id"]: item
        for item in root.iter()
        if item.tag.endswith("node") and item.attrib.get("id", "").startswith("instance-def-")
    }
    scene_points = []
    visual_scene = next(
        (item for item in root.iter() if item.tag.endswith("visual_scene")), None
    )
    if visual_scene is not None:
        for scene_node in list(visual_scene):
            matrix_item = next(
                (child for child in list(scene_node) if child.tag.endswith("matrix")), None
            )
            instance_item = next(
                (child for child in list(scene_node) if child.tag.endswith("instance_node")), None
            )
            if matrix_item is None or instance_item is None:
                continue
            matrix = np.array(
                [float(value) for value in (matrix_item.text or "").split()]
            ).reshape((4, 4))
            matrix[:3, 3] *= unit
            referenced = library_nodes.get(instance_item.attrib.get("url", "").lstrip("#"))
            if referenced is None:
                continue
            for geometry_item in referenced.iter():
                if not geometry_item.tag.endswith("instance_geometry"):
                    continue
                points = geometry_points.get(
                    geometry_item.attrib.get("url", "").lstrip("#")
                )
                if points is None:
                    continue
                homogeneous = np.column_stack((points, np.ones(len(points))))
                scene_points.append((matrix @ homogeneous.T).T[:, :3])
    if scene_points:
        return np.vstack(scene_points)
    return np.vstack(list(geometry_points.values()))


class Analyzer(Node):
    def __init__(self):
        super().__init__("analyze_gripper_geometry")
        self.message = None
        self.create_subscription(LinkStates, "/gazebo/link_states", self.callback, 1)

    def callback(self, message):
        self.message = message


def main():
    rclpy.init()
    node = Analyzer()
    end = node.get_clock().now().nanoseconds + int(5e9)
    while node.message is None and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    if node.message is None:
        sys.exit(1)
    poses = dict(zip(node.message.name, node.message.pose))
    link6_pose = poses["mecharm_270_m5::link6"]
    link6_rotation = rotation(link6_pose.orientation)
    link6_position = np.array([
        link6_pose.position.x, link6_pose.position.y, link6_pose.position.z
    ])
    mesh_root = "/home/robot/task2_ws/src/mycobot_ros2/mycobot_description/urdf/adaptive_gripper"
    all_points = []
    local_groups = {"left": [], "right": []}
    for link, visual_origin in VISUAL_ORIGINS.items():
        if f"mecharm_270_m5::{link}" not in poses:
            # Gazebo lumps fixed gripper_base into link6.  The six moving finger
            # meshes are sufficient to calibrate the jaw centre.
            continue
        pose = poses[f"mecharm_270_m5::{link}"]
        points = mesh_points(os.path.join(mesh_root, f"{link}.dae"))
        points = points + np.array(visual_origin)
        points = (rotation(pose.orientation) @ points.T).T
        points += np.array([pose.position.x, pose.position.y, pose.position.z])
        all_points.append(points)
        in_link6 = (link6_rotation.T @ (points - link6_position).T).T
        local_groups["left" if "left" in link else "right"].append(in_link6)
        low = points[np.argsort(points[:, 2])[: max(5, len(points) // 100)]]
        print(
            f"{link}: min={np.min(points, axis=0).round(5).tolist()} "
            f"max={np.max(points, axis=0).round(5).tolist()} "
            f"lowest_mean={np.mean(low, axis=0).round(5).tolist()} "
            f"link6_min={np.min(in_link6, axis=0).round(5).tolist()} "
            f"link6_max={np.max(in_link6, axis=0).round(5).tolist()} "
            f"link6_centroid={np.mean(in_link6, axis=0).round(5).tolist()}"
        )
    cloud = np.vstack(all_points)
    print(
        f"GRIPPER_ALL min={np.min(cloud, axis=0).round(5).tolist()} "
        f"max={np.max(cloud, axis=0).round(5).tolist()}"
    )
    left_tree = cKDTree(np.vstack(local_groups["left"]))
    right_tree = cKDTree(np.vstack(local_groups["right"]))
    options = []
    for x in np.linspace(-0.04, 0.04, 17):
        for y in np.linspace(-0.02, 0.08, 21):
            for z in np.linspace(0.045, 0.11, 27):
                centre = np.array([x, y, z])
                dl = float(left_tree.query(centre)[0])
                dr = float(right_tree.query(centre)[0])
                # A 40 mm cube needs roughly 20 mm clearance to each jaw.  Rank
                # balanced left/right cavities ahead of empty space far away.
                if 0.015 <= dl <= 0.040 and 0.015 <= dr <= 0.040:
                    score = abs(dl - 0.022) + abs(dr - 0.022) + abs(dl - dr)
                    options.append((score, centre, dl, dr))
    for score, centre, dl, dr in sorted(options, key=lambda item: item[0])[:15]:
        print(
            f"VOID_CANDIDATE centre={centre.round(5).tolist()} "
            f"left_vertex_distance={dl:.5f} right_vertex_distance={dr:.5f} "
            f"score={score:.5f}"
        )
    cube_centre = np.array([-0.0006, 0.010, 0.105])
    cube_half = 0.020
    for side, groups in local_groups.items():
        points = np.vstack(groups)
        inside = np.all(np.abs(points - cube_centre) < cube_half - 0.0002, axis=1)
        outside_delta = np.maximum(np.abs(points - cube_centre) - cube_half, 0.0)
        print(
            f"CAD_CUBE_CHECK side={side} vertices_inside={int(np.count_nonzero(inside))} "
            f"nearest_vertex_distance={np.min(np.linalg.norm(outside_delta, axis=1)):.6f}"
        )
    desired_pads = {
        "gripper_left1": np.array([-0.0226, 0.010, 0.105]),
        "gripper_right1": np.array([0.0214, 0.010, 0.105]),
    }
    for link, in_link6 in desired_pads.items():
        pose = poses[f"mecharm_270_m5::{link}"]
        world = link6_position + link6_rotation @ in_link6
        local = rotation(pose.orientation).T @ (
            world - np.array([pose.position.x, pose.position.y, pose.position.z])
        )
        print(
            f"PAD_LOCAL {link} link6={in_link6.round(5).tolist()} "
            f"link_local={local.round(6).tolist()}"
        )
    calibrated_locals = {
        "gripper_left1": np.array([0.017862, 0.048691, -0.010549]),
        "gripper_right1": np.array([-0.019062, 0.048691, -0.010549]),
    }
    for link, local in calibrated_locals.items():
        pose = poses[f"mecharm_270_m5::{link}"]
        world = (
            np.array([pose.position.x, pose.position.y, pose.position.z])
            + rotation(pose.orientation) @ local
        )
        current_link6 = link6_rotation.T @ (world - link6_position)
        print(f"PAD_CURRENT {link} link6={current_link6.round(6).tolist()}")
    for link in ("gripper_left1", "gripper_right1"):
        pose = poses[f"mecharm_270_m5::{link}"]
        relative = link6_rotation.T @ (
            np.array([pose.position.x, pose.position.y, pose.position.z]) - link6_position
        )
        print(f"GAZEBO_RELATIVE_{link}={relative.round(6).tolist()}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
