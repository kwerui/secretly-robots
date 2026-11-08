"""Tests for the Unitree G1 onboard sensors and hybrid localization stack.

Run from assistive_g1/:
    pytest -q test_localization.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import math
import mujoco
import numpy as np
import pytest

from localization import G1Localizer, wrap_angle
from scene import DEFAULT_PLACEMENT, STATIONS, build_scene
from sensors import IMUSensor, LegOdometry, LidarSensor, quat_from_euler
from skills import AssistiveRobot


@pytest.fixture(scope="module")
def robot():
    return AssistiveRobot()


def test_imu_and_leg_odometry_no_ground_truth(robot: AssistiveRobot):
    """IMUSensor and LegOdometry estimate roll, pitch, z_pelvis, and velocity without data.qpos[0:7]."""
    imu = robot.localizer.imu
    odom = robot.localizer.leg_odom

    reading = imu.read(0.02)
    assert abs(reading.roll) < 0.08
    assert abs(reading.pitch) < 0.08
    assert reading.gravity_body[2] < -0.95

    lo = odom.update(robot.wbc.q, robot.wbc.dq, reading, 0.02)
    true_z = float(robot.wbc.base_pos[2])
    assert abs(lo.z_pelvis - true_z) < 0.04, f"est z {lo.z_pelvis:.3f} vs true z {true_z:.3f}"


def test_lidar_detects_furniture_and_excludes_robot(robot: AssistiveRobot):
    """360-degree multi-layer LiDAR detects furniture around the room and never hits the robot's own body."""
    scan = robot.localizer.lidar.scan(z_pelvis_est=robot.localizer.z)
    assert len(scan.azimuths) == 180
    assert np.all(scan.ranges >= robot.localizer.lidar.min_range)
    # Should see multiple furniture pieces from the home station (table ahead, shelf left, counter right)
    assert len(scan.points_local) >= 15
    assert len(scan.points_3d_local) >= 20


def test_rgbd_depth_and_point_cloud(robot: AssistiveRobot):
    """HeadCamera renders metric depth, fuses depth with ArUco solvePnP, and generates 3D point clouds."""
    robot.go_to("table")
    rgb, depth = robot.cam.render_rgbd()
    assert rgb.shape == (robot.cam.height, robot.cam.width, 3)
    assert depth.shape == (robot.cam.height, robot.cam.width)
    assert float(np.nanmin(depth)) > 0.10

    det = robot.cam.detect(rgb, depth)
    assert 1 in det, "water_bottle marker (id=1) should be detected on the table"
    mp = det[1]
    assert mp.depth_verified is True

    # Compare estimated world marker position against simulator ground truth
    b_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, "water_bottle")
    true_bottle_top = robot.data.xpos[b_id] + np.array([0.0, 0.0, 0.101])
    err = float(np.linalg.norm(mp.position - true_bottle_top))
    assert err < 0.04, f"RGB-D localized marker error {err:.4f} m exceeds 4 cm"

    cloud = robot.cam.point_cloud_world(depth, stride=16, max_dist=2.0)
    assert cloud.shape[1] == 3
    assert len(cloud) > 50


def test_trimmed_icp_recovery_from_pose_perturbation(robot: AssistiveRobot):
    """Perturbing the localizer's pose estimate by (10 cm, -8 cm, 6 deg) is corrected by LiDAR scan matching."""
    true_x, true_y = float(robot.wbc.base_pos[0]), float(robot.wbc.base_pos[1])
    true_yaw = float(robot.wbc.base_yaw)

    # Inject a deliberate pose error into the EKF state
    robot.localizer.state[0] = true_x + 0.10
    robot.localizer.state[1] = true_y - 0.08
    robot.localizer.state[2] = wrap_angle(true_yaw + math.radians(6.0))
    robot.localizer.P = np.diag([0.05 ** 2, 0.05 ** 2, 0.08 ** 2])

    for _ in range(8):
        scan = robot.localizer.lidar.scan(z_pelvis_est=robot.localizer.z)
        robot.localizer.update_lidar(scan)

    pos_err = math.hypot(robot.localizer.x - true_x, robot.localizer.y - true_y)
    yaw_err = abs(wrap_angle(robot.localizer.yaw - true_yaw))
    assert pos_err < 0.03, f"Position error after scan matching: {pos_err:.4f} m"
    assert yaw_err < math.radians(2.0), f"Yaw error after scan matching: {math.degrees(yaw_err):.2f} deg"


def test_hybrid_map_outlier_rejection_and_live_obstacle_planning(robot: AssistiveRobot):
    """Outlier hits from unmapped obstacles do not corrupt localization and are mapped into DynamicWorldMap."""
    true_x, true_y = float(robot.wbc.base_pos[0]), float(robot.wbc.base_pos[1])
    true_yaw = float(robot.wbc.base_yaw)
    robot.localizer.reset(true_x, true_y, true_yaw, z=float(robot.wbc.base_pos[2]))

    # Synthesize a LiDAR scan that includes both real furniture hits and an unmapped obstacle at (0.8, 0.0)
    scan = robot.localizer.lidar.scan(z_pelvis_est=robot.localizer.z)
    # Inject synthetic obstacle rays at 0.85 m directly ahead (where the static room map is open space)
    c, s = math.cos(true_yaw), math.sin(true_yaw)
    R_inv = np.array([[c, s], [-s, c]])
    obs_world = np.array([[0.85, y] for y in np.linspace(-0.20, 0.20, 9)])
    obs_local = (obs_world - np.array([true_x, true_y])) @ R_inv.T
    scan.points_local = np.vstack([scan.points_local, obs_local])
    scan.max_heights = np.concatenate([scan.max_heights, np.full(len(obs_local), 1.35)])
    scan.ranges = np.concatenate([scan.ranges, np.linalg.norm(obs_local, axis=1)])
    scan.azimuths = np.concatenate([scan.azimuths, np.arctan2(obs_local[:, 1], obs_local[:, 0])])

    robot.localizer.update_lidar(scan)

    # 1. Layer 1 localization stays locked onto the static map despite the unmapped obstacle
    pos_err = math.hypot(robot.localizer.x - true_x, robot.localizer.y - true_y)
    assert pos_err < 0.025, f"Outlier corrupted pose by {pos_err:.4f} m"

    # 2. Layer 2 DynamicWorldMap clusters the unmapped obstacle
    assert len(robot.localizer.dynamic_map.tracks) >= 1

    # 3. Path planner routes around the live obstacle
    path = robot.localizer.dynamic_map.plan_path((0.0, 0.0), (1.42, 0.0), robot_radius=0.28)
    assert len(path) >= 1
    assert np.allclose(path[-1], (1.42, 0.0))


def test_dynamic_person_tracking_triggers_person_too_close(robot: AssistiveRobot):
    """Moving obstacle cluster with human height is classified as a person and triggers person_too_close."""
    dmap = robot.localizer.dynamic_map
    dmap.reset()
    rx, ry = float(robot.est_pos[0]), float(robot.est_pos[1])

    # Feed 4 consecutive observations of a human-sized cluster walking toward the robot
    for k, dist in enumerate((1.40, 1.20, 1.00, 0.85)):
        pts = np.array([[rx + dist, ry - 0.04], [rx + dist, ry], [rx + dist, ry + 0.04]])
        heights = np.array([1.65, 1.65, 1.65])
        clusters = dmap._cluster_outliers(pts, heights)
        dmap._update_tracks(clusters, now=10.0 + k * 0.2)

    too_close, min_d, p_pos = dmap.check_person_clearance(np.array([rx, ry]), now=10.6)
    assert too_close is True
    assert min_d < 1.10
    assert p_pos is not None
