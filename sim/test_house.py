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
    # Obstacles pool + dynamic person + robot head camera mount
    assert m.nmocap == sum(H._obstacle_pool().values()) + 2
    for dn in H.DOORWAYS:
        m.joint(f"door_{dn}_hinge")
        m.site(f"door_{dn}_push_a")
        m.site(f"door_{dn}_push_b")
    for u, spec in H.DRAWER_UNITS.items():
        for i in range(spec["n"]):
            m.site(f"drawer_{u}_{i}_handle")
    for name in ["waterbottle", "user", "person_dynamic"]:
        m.body(name)
    m.site("waterbottle_site")
    m.site("user_hand")
    for cam in ["overview", "cam_robot_head"] + [f"cam_{r}" for r in H.ROOMS]:
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


def test_modes_enumeration(env):
    """Test all three challenge modes: static, shuffled, dynamic."""
    info_static = env.reset("tidy", "kitchen_counter", mode=H.Mode.STATIC, seed=0)
    assert info_static["mode"] == "static"
    assert env.mode == H.Mode.STATIC

    info_shuffled = env.reset(mode=H.Mode.SHUFFLED, seed=1)
    assert info_shuffled["mode"] == "shuffled"
    assert env.mode == H.Mode.SHUFFLED

    info_dynamic = env.reset("tidy", "kitchen_counter", mode=H.Mode.DYNAMIC, seed=2)
    assert info_dynamic["mode"] == "dynamic"
    assert env.mode == H.Mode.DYNAMIC


def test_drawer_target_placement(env):
    """Test target items placed inside drawers."""
    for tgt in H.DRAWER_TARGETS:
        info = env.reset("tidy", tgt, mode=H.Mode.SHUFFLED, seed=0)
        assert info["container"] is not None
        assert info["container"].startswith("drawer_")
        # Check bottle position is at drawer elevation
        bpos = env.bottle_xyz()
        assert bpos[2] > 0.20
        assert env.check().solvable


def test_dynamic_person_walks_and_relocates_item(env):
    """Level 3: Dynamic human enters, picks up bottle, and moves it to another surface."""
    info = env.reset("tidy", "kitchen_counter", mode=H.Mode.DYNAMIC, seed=0)
    ctl = env.dynamic_controller
    st0 = env.dynamic_state
    assert st0["active"] is True
    assert not st0["carrying"]
    assert not st0["item_relocated"]
    assert info["person_drop"] != "kitchen_counter"

    start_bottle = env.bottle_xyz()
    dt = env.model.opt.timestep

    # halfway between pick-up and put-down: carrying, bottle has moved
    env.step(int((ctl.pick_time + ctl.place_time) / 2 / dt))
    assert env.dynamic_state["carrying"] is True
    assert np.linalg.norm(env.bottle_xyz()[:2] - start_bottle[:2]) > 0.3

    # after the route ends: bottle rests on the drop surface
    env.step(int((ctl.total_duration + 1.0 - env.sim_time) / dt))
    assert env.dynamic_state["item_relocated"] is True
    assert np.linalg.norm(env.bottle_xyz() - env.person_drop_pos) < 0.05

    # once put down, the bottle is a free body again (not pinned), so the robot can take it
    a = env.bottle_qadr
    env.data.qpos[a] += 0.05
    env.step(int(0.5 / dt))
    assert env.bottle_xyz()[0] - env.person_drop_pos[0] > 0.04


@pytest.mark.parametrize("layout", list(H.LAYOUTS))
@pytest.mark.parametrize("target", ["kitchen_table", "bedside_table", "dresser_drawer"])
def test_dynamic_person_never_clips(env, layout, target):
    """The person's route goes round furniture, walls, clutter, doors and drawers."""
    m, d = env.model, env.data
    env.reset(layout, target, mode=H.Mode.DYNAMIC, seed=1)
    ctl = env.dynamic_controller
    others = env.static_geoms + env.drawer_geoms + env.all_door_geoms + env.active_mocap_geoms()
    for t in np.arange(0.0, ctl.total_duration, 0.2):
        st = ctl.state_at(t)
        d.mocap_pos[env.dynamic_person_id] = st["pos"]
        d.mocap_quat[env.dynamic_person_id] = st["quat"]
        mujoco.mj_kinematics(m, d)
        for g in env.person_geoms:
            for h in others:
                if np.linalg.norm(d.geom_xpos[g, :2] - d.geom_xpos[h, :2]) > m.geom_rbound[g] + m.geom_rbound[h]:
                    continue
                dist = mujoco.mj_geomDistance(m, d, g, h, 0.05, None)
                assert dist > -1e-3, (t, m.geom(g).name, m.geom(h).name, dist)


def test_dynamic_person_too_close_clearance(env):
    """Dynamic person proximity triggers person_too_close safety condition."""
    env.reset("tidy", "kitchen_counter", mode=H.Mode.DYNAMIC, seed=0)
    # Controller clearance check
    too_close, dist = env.dynamic_controller.check_robot_clearance(
        robot_xy=env.dynamic_controller.state_at(0.0)["pos"][:2],
        t=0.0,
        clearance_threshold=1.0,
    )
    assert too_close is True
    assert dist < 1e-4

    far, dist_far = env.dynamic_controller.check_robot_clearance(
        robot_xy=(-3.0, -2.0),
        t=0.0,
        clearance_threshold=1.0,
    )
    assert far is False
    assert dist_far > 2.0


def test_push_door_aruco_perception(env):
    """ArUco marker on push-only door is detected via simulated camera."""
    from sim.aruco import detect_markers
    env.reset("tidy", "tv_unit", mode=H.Mode.STATIC, seed=0)
    env.set_door("living_kitchen", 0.0)
    mujoco.mj_forward(env.model, env.data)

    target = env.data.site("door_living_kitchen_push_a").xpos
    cam = mujoco.MjvCamera()
    cam.lookat[:] = target
    cam.distance = 0.6
    cam.azimuth = 0
    cam.elevation = 0

    with mujoco.Renderer(env.model, 600, 600) as r:
        r.update_scene(env.data, camera=cam)
        img = r.render()
        dets = detect_markers(img)
        assert len(dets) >= 1
        assert dets[0]["id"] == 10
        assert dets[0]["kind"] == "door_push"


def test_drawer_handle_aruco_perception(env):
    """ArUco marker on chest of drawers is detected via simulated camera."""
    from sim.aruco import detect_markers
    env.reset("tidy", "tv_unit", mode=H.Mode.STATIC, seed=0)
    mujoco.mj_forward(env.model, env.data)

    target = env.data.geom("drawer_dresser_3_marker").xpos
    cam = mujoco.MjvCamera()
    cam.lookat[:] = target
    cam.distance = 0.5
    cam.azimuth = 90
    cam.elevation = 0

    with mujoco.Renderer(env.model, 600, 600) as r:
        r.update_scene(env.data, camera=cam)
        img = r.render()
        dets = detect_markers(img)
        assert len(dets) >= 1
        assert dets[0]["id"] == 23
        assert dets[0]["name"] == "drawer_dresser_3"



# ---- Shuffled furniture (Mode 2) --------------------------------------------
@pytest.mark.parametrize("layout", list(H.LAYOUTS))
def test_shuffled_moves_furniture_validly(env, layout):
    """Shuffled resets move the movable furniture, the bottle moves with it, and the
    result passes every validity check."""
    for seed in range(3):
        info = env.reset(layout, "kitchen_table", mode=H.Mode.SHUFFLED, seed=seed)
        assert info["attempts"] < H.MAX_RESET_TRIES
        moved = [n for n, p in info["furniture"].items() if p != (*H.FURNITURE[n]["pos"], 0.0)]
        assert len(moved) >= 3
        assert env.penetrations() == []
        assert env.sweeps_clear()
        assert env.check().solvable
        assert env.doorways_reachable(env.check().grid)
        assert np.linalg.norm(env.bottle_xyz() - env.surface_xyz("kitchen_table")) < 0.01
        top = env.data.geom("kitchen_table").xpos
        assert np.linalg.norm(env.bottle_xyz()[:2] - top[:2]) < 0.5


def test_other_modes_restore_nominal_furniture(env):
    env.reset("tidy", "study_desk", mode=H.Mode.SHUFFLED, seed=0)
    env.reset("tidy", "study_desk", mode=H.Mode.STATIC, seed=0)
    for n, p in env.furn_pose.items():
        assert p == (*H.FURNITURE[n]["pos"], 0.0)
        assert np.allclose(env.data.xpos[env.furn_body[n], :2], H.FURNITURE[n]["pos"])


# ---- Bottle inside drawers ---------------------------------------------------
@pytest.mark.parametrize("target", list(H.DRAWER_TARGETS))
def test_bottle_fits_in_drawer_and_survives_a_pull(env, target):
    """The bottle stands upright in its drawer without touching the chest, and stays
    upright when the drawer is pulled fully open with a steady 10 N."""
    m, d = env.model, env.data
    env.reset("tidy", target, mode=H.Mode.SHUFFLED, seed=0)
    mujoco.mj_forward(m, d)
    bottle = [g for g in range(m.ngeom) if m.geom_bodyid[g] == env.bottle_body]
    for g in bottle:
        for h in env.static_geoms + env.drawer_geoms:
            assert mujoco.mj_geomDistance(m, d, g, h, 0.05, None) > -1e-3, (m.geom(g).name, m.geom(h).name)

    _, _, unit, idx = H.DRAWER_TARGETS[target]
    j = m.joint(f"drawer_{unit}_{idx}_slide").id
    b = m.body(f"drawer_{unit}_{idx}").id
    for _ in range(int(3.0 / m.opt.timestep)):
        d.xfrc_applied[b, :3] = 10.0 * d.xmat[b].reshape(3, 3)[:, 0]
        env.step(1)
    d.xfrc_applied[:] = 0
    env.step(int(0.5 / m.opt.timestep))
    assert d.qpos[m.jnt_qposadr[j]] > 0.95 * m.jnt_range[j][1]
    assert d.xmat[env.bottle_body].reshape(3, 3)[2, 2] > 0.98  # still upright


# ---- Robot head camera -------------------------------------------------------
TOY_ROBOT = """<mujoco><worldbody>
  <body name="toy_base" pos="0 0 0.3"><freejoint name="toy_root"/>
    <geom type="box" size="0.2 0.15 0.1" mass="10"/>
  </body>
</worldbody></mujoco>"""


def test_head_camera_follows_robot(tmp_path):
    xml = tmp_path / "toy.xml"
    xml.write_text(TOY_ROBOT)
    env = H.HouseEnv(robot_xml=str(xml), seed=0)
    env.reset("tidy", "tv_unit", mode=H.Mode.STATIC, seed=0)
    m, d = env.model, env.data
    cam = m.camera("cam_robot_head").id
    assert np.allclose(d.cam_xpos[cam], [*H.ROBOT_START[:2], H.HEAD_CAM_Z], atol=0.02)

    # teleport the robot to the kitchen, facing +y; the camera goes with it
    a = env.robot_qadr
    d.qpos[a : a + 3] = [2.0, -1.5, 0.3]
    d.qpos[a + 3 : a + 7] = H.yaw_quat(90)
    env.step(1)
    mujoco.mj_forward(m, d)
    assert np.allclose(d.cam_xpos[cam], [2.0, -1.5, H.HEAD_CAM_Z], atol=0.02)
    look = -d.cam_xmat[cam].reshape(3, 3)[:, 2]  # cameras look along their -z
    assert look[1] > 0.8 and abs(look[0]) < 0.1 and look[2] < 0
