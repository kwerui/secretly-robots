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
    assert m.nmocap == sum(H._obstacle_pool().values())
    for dn in H.DOORWAYS:
        m.joint(f"door_{dn}_hinge")
    for u, spec in H.DRAWER_UNITS.items():
        for i in range(spec["n"]):
            m.site(f"drawer_{u}_{i}_handle")
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
        assert env.sweeps_clear()


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


def _push_door(env, name, sign, force=40.0, seconds=3.0):
    m, d = env.model, env.data
    env.set_door(name, 0.0)
    mujoco.mj_forward(m, d)
    b = m.body(f"door_{name}").id
    for _ in range(int(seconds / m.opt.timestep)):
        R = d.xmat[b].reshape(3, 3)
        d.qfrc_applied[:] = 0
        mujoco.mj_applyFT(m, d, sign * force * R[:, 1], np.zeros(3), d.xpos[b] + 0.8 * R[:, 0], b, d.qfrc_applied)
        mujoco.mj_step(m, d)
    d.qfrc_applied[:] = 0
    return np.degrees(env.door_angle(name))


@pytest.mark.parametrize("door", list(H.DOORWAYS))
def test_door_pushes_both_ways(env, door):
    env.reset("tidy", "tv_unit", seed=0)
    assert _push_door(env, door, +1) > 80
    assert _push_door(env, door, -1) < -80


def test_locked_door_holds(env):
    env.reset("door_shut", "tv_unit", seed=0)
    assert "living_kitchen" in env.locked_doors
    assert abs(_push_door(env, "living_kitchen", +1)) < 2
    assert abs(_push_door(env, "living_kitchen", -1)) < 2


def test_door_start_angles_random(env):
    angles = {env.reset("tidy", "tv_unit", seed=s)["doors"]["kitchen_study"] for s in range(10)}
    assert len(angles) > 5


def test_doors_stay_put(env):
    env.reset("tidy", "tv_unit", seed=4)
    before = {dn: env.door_angle(dn) for dn in H.DOORWAYS}
    env.step(int(2.0 / env.model.opt.timestep))
    for dn, a in before.items():
        assert abs(env.door_angle(dn) - a) < 1e-3


@pytest.mark.parametrize("joint", [f"drawer_{u}_{i}_slide" for u, s in H.DRAWER_UNITS.items() for i in range(s["n"])])
def test_drawer_pulls_out_by_handle(env, joint):
    m, d = env.model, env.data
    env.reset("tidy", "tv_unit", seed=0)
    name = joint.removesuffix("_slide")
    b, j = m.body(name).id, m.joint(joint).id
    for _ in range(int(1.0 / m.opt.timestep)):
        d.qfrc_applied[:] = 0
        pull = 30.0 * d.xmat[b].reshape(3, 3)[:, 0]
        mujoco.mj_applyFT(m, d, pull, np.zeros(3), d.site(name + "_handle").xpos, b, d.qfrc_applied)
        mujoco.mj_step(m, d)
    d.qfrc_applied[:] = 0
    assert d.qpos[m.jnt_qposadr[j]] > 0.95 * m.jnt_range[j, 1]


def test_drawer_handles_are_long(env):
    for u, spec in H.DRAWER_UNITS.items():
        g = env.model.geom(f"drawer_{u}_0_handle")
        assert 2 * g.size[1] >= 0.7 * spec["size"][0]


def test_obstacle_in_door_swing_rejected(env):
    env.reset("tidy", "tv_unit", seed=0)
    mid = env.obs_mocap["box_small"][0]
    env.data.mocap_pos[mid] = [0.5, -1.5, 0.0]  # in the living_kitchen door's swing
    mujoco.mj_forward(env.model, env.data)
    assert not env.sweeps_clear()
