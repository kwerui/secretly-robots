"""Tests for the house scene. Run from sim/:  .venv/bin/python -m pytest -q"""

import itertools
from pathlib import Path

import mujoco
import numpy as np
import pytest

import house_env as H

ROBOT_XML = Path(__file__).resolve().parent.parent / "trash-collection-isaaclab/deploy/mujoco/models/aliengo_z1.xml"
COMBOS = list(itertools.product(H.LAYOUTS, H.TARGETS))


@pytest.fixture(scope="module")
def env():
    return H.HouseEnv(seed=0)


def test_model_compiles(env):
    m = env.model
    assert m.nmocap == len(H.DOORWAYS) + sum(H._obstacle_pool().values())
    for name in ["waterbottle", "user"]:
        m.body(name)
    m.site("waterbottle_site")
    for cam in ["overview"] + [f"cam_{r}" for r in H.ROOMS]:
        m.camera(cam)


def test_xml_roundtrip(env):
    mujoco.MjModel.from_xml_string(env.xml)


@pytest.mark.parametrize("layout,target", COMBOS)
def test_reset_valid(env, layout, target):
    for seed in range(3):
        info = env.reset(layout, target, seed=seed)
        assert info["layout"] == layout and info["target"] == target
        assert np.isfinite(info["path_len"])
        assert env.check().solvable
        assert env.penetrations() == []


@pytest.mark.parametrize("layout,target", COMBOS)
def test_nominal_layout_valid(env, layout, target):
    """The un-jittered layout is the fallback, so it must always be valid."""
    env.reset(layout, target, jitter=False)
    assert env.penetrations() == []
    assert env.check().solvable


@pytest.mark.parametrize("target", list(H.TARGETS))
def test_bottle_stays_on_surface(env, target):
    env.reset("moving_house", target, seed=1)
    start = env.bottle_xyz()
    env.step(int(1.0 / env.model.opt.timestep))  # 1 s
    end = env.bottle_xyz()
    surface = H.FURNITURE[H.TARGETS[target][2]]["h"]
    assert abs(end[2] - surface) < 0.01, f"bottle z {end[2]:.3f} vs surface {surface}"
    assert np.linalg.norm(end[:2] - start[:2]) < 0.02, "bottle slid"
    zaxis = env.data.xmat[env.bottle_body].reshape(3, 3)[:, 2]
    assert zaxis[2] > 0.99, "bottle tipped over"


def test_no_static_contacts_with_mocap(env):
    """After stepping, no obstacle/door is in contact with walls or furniture."""
    for layout in H.LAYOUTS:
        env.reset(layout, "tv_unit", seed=2)
        env.step(50)
        static = set(env.static_geoms)
        for c in env.data.contact[: env.data.ncon]:
            pair = {c.geom1, c.geom2}
            assert not (pair & static and pair & set(env.active_mocap_geoms()) and c.dist < -1e-3)


def test_unused_obstacles_parked(env):
    env.reset("tidy", "tv_unit")
    for ids in env.obs_mocap.values():
        for mid in ids:
            assert env.data.mocap_pos[mid][2] == H.PARK_Z


def test_random_reset(env):
    seen = {env.reset(seed=s)["layout"] for s in range(30)}
    assert len(seen) > 1


@pytest.mark.skipif(not ROBOT_XML.exists(), reason="robot repo not checked out next to sim/")
def test_robot_attach():
    env = H.HouseEnv(robot_xml=str(ROBOT_XML))
    env.reset("tidy", "kitchen_counter")
    assert env.model.body(env.robot_root).name == "trunk"
    q = env.data.qpos[env.robot_qadr : env.robot_qadr + 3]
    assert np.allclose(q[:2], H.ROBOT_START[:2])
