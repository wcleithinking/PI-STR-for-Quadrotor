"""Gates V1, V2 and V4 of ``docs/3D_ENVIRONMENT.md`` §5.

* **V1** -- ``build_model()`` succeeds and the unactuated vehicle does not
  explode in 1 s.
* **V2** -- model mass and inertia match ``params.airframe``: mass to 1e-9,
  inertia to 1e-9 *after* the documented ``Iyz = 0`` fix, and ``Iyz`` is
  asserted to be zero.  (The published MATLAB tensor is not physically
  realisable, so MuJoCo rejects it outright -- see
  :func:`pistr_sim.env.mjcf.realisable_inertia`.)
* **V4** -- with :class:`RotorPack`, total hover thrust is ``m g`` within 1 %
  and the vehicle holds altitude for 2 s with less than 0.15 m drift.

Extra tests cover the wiring that M3 (render) and M4 (bridge) depend on:
body/rotor names, the scene-fragment insertion, the FLU body-frame wrench
conventions, and the actuator lag.

Run::

    PYTHONPATH=sim/python:build/mjsim python3 -m pytest sim/python/tests/test_env_model.py -v

The file skips cleanly when MuJoCo is absent, so ``make test`` keeps working on
a bare checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pistr_sim.env import actuation, mjcf  # noqa: E402
from pistr_sim.params import default_params  # noqa: E402

PARAMS = default_params("high")
AIRFRAME = PARAMS.airframe
G = float(AIRFRAME.g)
MASS = float(AIRFRAME.mass)
WEIGHT = MASS * G


def _model(n_drones: int = 1, **kwargs):
    return mjcf.build_model(PARAMS, n_drones=n_drones, **kwargs)


def _data(model):
    return mujoco.MjData(model)


# ---------------------------------------------------------------------------
# V1 -- construction and unactuated stability
# ---------------------------------------------------------------------------
def test_v1_build_model_and_unactuated_stability():
    """V1: a valid model, and 1 s of free fall + landing without blowing up."""
    model = _model()
    assert model.nbody == 6, "world + drone0 + 4 rotor bodies"
    assert model.nq == 7 and model.nv == 6, "one free joint, one vehicle"
    assert model.opt.timestep == pytest.approx(float(PARAMS.sim.dt), abs=0.0)
    assert model.opt.gravity[2] == pytest.approx(-G)

    data = _data(model)
    z0 = float(data.qpos[2])
    assert z0 == pytest.approx(mjcf.DEFAULT_ALTITUDE)

    peak_speed = 0.0
    for _ in range(1000):  # 1 s at the model timestep
        mujoco.mj_step(model, data)
        peak_speed = max(peak_speed, float(np.linalg.norm(data.qvel[:3])))

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    # Ballistic speed after 1.4 m of fall is sqrt(2 g h) ~ 5.2 m/s.
    assert peak_speed < 8.0, f"vehicle accelerated to {peak_speed:.2f} m/s"
    # It must land on its legs and stay there, not sink or bounce forever.
    assert 0.05 < data.qpos[2] < 0.30, f"resting height {data.qpos[2]:.3f} m"
    assert float(np.linalg.norm(data.qvel[:3])) < 0.10
    assert float(np.linalg.norm(data.qvel[3:])) < 0.50


# ---------------------------------------------------------------------------
# V2 -- mass and inertia
# ---------------------------------------------------------------------------
def test_v2_mass_and_inertia_match_params():
    """V2: mass 1e-9; inertia === params with the known-bad ``Iyz`` zeroed."""
    model = _model()
    body_id = mjcf.model_body_id(model, "drone0")
    a = AIRFRAME

    # -- mass: exact, and nothing double-counted on the rotor bodies ---------
    assert model.body_mass[body_id] == pytest.approx(a.mass, abs=1e-9)
    assert float(model.body_mass.sum()) == pytest.approx(a.mass, abs=1e-9)
    for i in range(1, model.nbody):
        if i != body_id:
            assert model.body_mass[i] == 0.0, (
                "rotor bodies must be massless: Airframe.mass already counts "
                "4 rotors + 4 motors"
            )

    # -- inertia: reconstruct the 3x3 tensor MuJoCo is actually using -------
    inertia = mjcf.body_inertia_tensor(model, body_id)
    expected = mjcf.realisable_inertia(a.inertia)
    assert np.allclose(inertia, expected, atol=1e-9, rtol=0.0)

    # ... every diagonal term is exactly the published one:
    assert np.allclose(np.diag(inertia), np.diag(a.inertia), atol=1e-9, rtol=0.0)
    assert inertia[0, 2] == pytest.approx(a.inertia[0, 2], abs=1e-9)
    assert inertia[0, 1] == pytest.approx(0.0, abs=1e-12)
    # ... and the single documented deviation is Iyz (the MATLAB bug):
    assert a.inertia[1, 2] != 0.0, "params no longer sets Iyz = Ixz; revisit the fix"
    assert inertia[1, 2] == 0.0, (
        "Iyz must be zero: the published tensor violates A + B >= C and "
        "MuJoCo refuses to compile it"
    )

    # -- the fix leaves only a 2.9e-6 triangle-inequality margin -------------
    principal = np.linalg.eigvalsh(expected)
    margin = float(principal[0] + principal[1] - principal[2])
    assert margin > 0.0, f"tensor is not realisable: A + B - C = {margin:.3e}"
    assert margin < 1e-4, (
        f"unexpectedly large margin {margin:.3e}: the Iyz=0 fix should leave "
        "almost no room to spare"
    )


def test_v2_raw_matlab_tensor_is_rejected_by_mujoco():
    """Why the inertia is adjusted: MuJoCo refuses the published tensor.

    This is the executable form of the V2 rationale.  If a future MuJoCo
    accepts the raw tensor, this test fails and the ``Iyz = 0`` fix in
    :func:`mjcf.realisable_inertia` should be re-evaluated.
    """
    a = AIRFRAME
    j = a.inertia
    raw = (
        f"{j[0,0]:.12g} {j[1,1]:.12g} {j[2,2]:.12g} "
        f"{j[0,1]:.12g} {j[0,2]:.12g} {j[1,2]:.12g}"
    )
    xml = f"""<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="b">
      <freejoint/>
      <inertial pos="0 0 0" mass="{a.mass:.12g}" fullinertia="{raw}"/>
    </body>
  </worldbody>
</mujoco>"""
    with pytest.raises(ValueError, match="inertia"):
        mujoco.MjModel.from_xml_string(xml)


def test_v2_realisable_inertia_rejects_unfixable_tensors():
    with pytest.raises(ValueError, match="triangle"):
        mjcf.realisable_inertia(np.diag([1.0, 1.0, 5.0]))
    with pytest.raises(ValueError, match="positive definite"):
        mjcf.realisable_inertia(np.diag([1.0, -1.0, 1.0]))


# ---------------------------------------------------------------------------
# V4 -- hover thrust and altitude hold
# ---------------------------------------------------------------------------
def test_v4_hover_thrust_and_two_second_altitude_hold():
    """V4: hover thrust within 1 % of m g; 2 s altitude drift under 0.15 m."""
    pack = actuation.RotorPack(PARAMS)
    trim = pack.trim_hover(soc=1.0)

    # -- trim point ---------------------------------------------------------
    assert trim["thrust"] == pytest.approx(WEIGHT, rel=0.01), (
        f"trim thrust {trim['thrust']:.4f} N vs m g = {WEIGHT:.4f} N"
    )
    # This is the same operating point as the high-fidelity plant (battery
    # sag included), not params.pwm_hover_us (which assumes a constant supply).
    from pistr_sim.plant import QuadrotorPlant

    plant_trim = QuadrotorPlant(PARAMS).trim_hover(1.0)
    assert trim["pwm_us"] == pytest.approx(float(plant_trim["pwm_us"]), abs=1.0)

    # -- pack reset at the trim point is a true fixed point ------------------
    pack.reset(pwm_us=trim["pwm_us"])
    assert pack.thrust.sum() == pytest.approx(trim["thrust"], rel=1e-12)
    assert np.allclose(pack.esc_duty, trim["esc_duty"])

    # -- 2 s of flight driven only through xfrc_applied ----------------------
    model = _model()
    data = _data(model)
    body_id = mjcf.model_body_id(model, "drone0")
    dt = float(model.opt.timestep)
    z0 = float(data.qpos[2])

    for _ in range(int(round(2.0 / dt))):
        pack.step(trim["pwm_us"], dt)
        force, moment = pack.wrench()
        data.xfrc_applied[body_id, :3] = force
        data.xfrc_applied[body_id, 3:] = moment
        mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos))

    drift = abs(float(data.qpos[2]) - z0)
    assert drift < 0.15, f"altitude drift {drift:.3f} m over 2 s"
    # The vehicle must stay level and not spin up: only thrust was applied,
    # at the CoM, so there is no torque at all.
    assert float(np.linalg.norm(data.qvel[3:])) < 0.05
    assert float(np.linalg.norm(data.qpos[3:7] - np.array([1.0, 0.0, 0.0, 0.0]))) < 1e-6


def test_v4_cold_start_settles_to_the_trim_point():
    """A stopped pack reaches the trim state with the expected ~40 ms lag."""
    pack = actuation.RotorPack(PARAMS)
    trim = pack.trim_hover(soc=1.0)
    pack.reset()
    assert np.all(pack.omega == 0.0)

    dt = 1e-3
    for _ in range(500):  # 0.5 s, ~12 combined time constants
        pack.step(np.full(4, trim["pwm_us"]), dt)
    assert pack.omega.sum() == pytest.approx(trim["omega"].sum(), rel=0.01)
    assert pack.thrust.sum() == pytest.approx(WEIGHT, rel=0.01)

    # It must be monotonic in the mean: no overshoot of the target speed.
    pack.reset()
    previous = 0.0
    for _ in range(300):
        pack.step(trim["pwm_us"], dt)
        current = float(pack.omega.mean())
        assert current >= previous - 1e-9
        previous = current


# ---------------------------------------------------------------------------
# Interface wiring that M3/M4 depend on
# ---------------------------------------------------------------------------
def test_body_and_rotor_names():
    model = _model()
    assert mjcf.drone_body_names(1) == ["drone0"]
    for i in range(4):
        rotor_id = mjcf.model_body_id(model, mjcf.rotor_body_name(0, i))
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, mjcf.rotor_site_name(0, i)
        )
        assert rotor_id > 0
        assert site_id >= 0
        assert model.body_parentid[rotor_id] == mjcf.model_body_id(model, "drone0")
        # the rotor body must be welded (no joint) and massless
        assert model.body_jntnum[rotor_id] == 0
        assert model.body_mass[rotor_id] == 0.0
        for suffix in ("_disc", "_blade", "_hub"):
            assert (
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    mjcf.rotor_body_name(0, i) + suffix,
                )
                >= 0
            )


def test_rotor_positions_use_the_ned_to_flu_map():
    """Rotor 1 sits at -y and rotor 3 at +y in the FLU model (S @ arms)."""
    model = _model()
    body_id = mjcf.model_body_id(model, "drone0")
    arms_flu = (mjcf.S_NED_TO_FLU @ np.asarray(AIRFRAME.arm_positions).T).T
    for i in range(4):
        rotor_id = mjcf.model_body_id(model, mjcf.rotor_body_name(0, i))
        pos = np.asarray(model.body_pos[rotor_id])
        # body_pos is relative to the parent, which is at the origin.
        assert np.allclose(pos[:2], arms_flu[i, :2], atol=1e-9)
        assert pos[2] == pytest.approx(0.020, abs=1e-9)
        assert model.body_parentid[rotor_id] == body_id
    # S is a proper rotation: FLU is not a mirror image of NED.
    assert np.linalg.det(mjcf.S_NED_TO_FLU) == pytest.approx(1.0)


def test_multi_drone_layout_and_independence():
    model = _model(n_drones=3)
    assert model.nbody == 1 + 3 * (1 + 4)
    assert model.nq == 3 * 7
    layout = [(0.0, 0.0, 1.5), (2.0, 0.0, 2.0), (0.0, 3.0, 2.5)]
    model = _model(n_drones=3, layout=layout)
    data = _data(model)
    positions = data.qpos.reshape(3, 7)[:, :3]
    assert np.allclose(positions, np.asarray(layout))
    # every vehicle carries the full airframe mass, none shares a free joint
    for i in range(3):
        body_id = mjcf.model_body_id(model, mjcf.drone_body_name(i))
        assert model.body_mass[body_id] == pytest.approx(AIRFRAME.mass, abs=1e-9)
        assert model.body_jntnum[body_id] == 1
        assert model.jnt_type[model.body_jntadr[body_id]] == mujoco.mjtJoint.mjJNT_FREE


def test_layout_validation():
    with pytest.raises(ValueError, match="n_drones"):
        mjcf.build_xml(PARAMS, n_drones=0)
    with pytest.raises(ValueError, match="positions"):
        mjcf.build_xml(PARAMS, n_drones=2, layout=[(0.0, 0.0, 1.0)])
    with pytest.raises(ValueError, match="single position"):
        mjcf.build_xml(PARAMS, n_drones=2, layout=(0.0, 0.0, 1.0))


def test_scene_fragment_is_inserted_into_worldbody():
    scene = '<geom name="crate" type="box" pos="5 0 0.5" size="0.3 0.3 0.3"/>'
    model = _model(scene=scene)
    assert (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "crate") >= 0
    ), "scene fragment must land inside <worldbody>"


def test_scene_asset_block_is_hoisted():
    """M2 may return textures; a single <asset> block is hoisted for it."""
    scene = (
        '<asset><texture name="lawn" type="2d" builtin="checker" '
        'rgb1="0.1 0.4 0.1" rgb2="0.05 0.3 0.05" width="64" height="64"/>'
        '<material name="lawn" texture="lawn"/></asset>'
        '<geom name="lawn_floor" type="plane" size="10 10 0.05" material="lawn"/>'
    )
    model = _model(scene=scene)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lawn_floor") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "lawn") >= 0


def test_a_scene_owns_the_ground():
    """With a scene there is no fallback floor (the scene provides it)."""
    model = _model(scene='<geom name="my_floor" type="plane" size="5 5 0.05"/>')
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "my_floor") >= 0
    # ... and without one, the fallback exists so the V1 drop test is posed.
    assert mujoco.mj_name2id(_model(), mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0


def test_wrench_frame_conventions():
    """FLU body frame: thrust up, rotor 0 at +x pitches down, yaw via -spin."""
    pack = actuation.RotorPack(PARAMS)
    pack.reset()
    omega = np.array([500.0, 400.0, 400.0, 400.0])
    pack.omega = omega
    force, moment = pack.wrench()

    thrust = PARAMS.rotor.kt * omega**2
    torque = PARAMS.rotor.kq * omega**2
    assert force[0] == pytest.approx(0.0)
    assert force[1] == pytest.approx(0.0)
    assert force[2] == pytest.approx(thrust.sum())  # +z is up in FLU

    arm_flu = (mjcf.S_NED_TO_FLU @ np.asarray(AIRFRAME.arm_positions).T).T
    expected_moment = np.cross(
        arm_flu, np.column_stack([np.zeros(4), np.zeros(4), thrust])
    ).sum(axis=0)
    spin_flu = actuation.spin_dirs_flu(PARAMS)
    expected_moment[2] += float(np.dot(spin_flu, torque))
    assert np.allclose(moment, expected_moment, atol=1e-12)
    # Rotor 0 (at +x, spinning faster) produces a negative pitch moment.
    assert moment[1] < 0.0
    # spin_dirs are NED-signed, so rotor 0 (+1 = CW seen from above) gives a
    # negative yaw reaction in FLU.  With rotors 1..3 equal, the yaw moment is
    # -(Q0 - Q1) -- the differential profile torque, not the absolute one.
    assert spin_flu[0] == -1.0
    assert moment[2] == pytest.approx(-(torque[0] - torque[1]), rel=1e-12)


def test_step_semantics_and_limits():
    pack = actuation.RotorPack(PARAMS)
    pack.reset()
    with pytest.raises(ValueError, match="finite"):
        pack.step(np.array([np.nan, 1000, 1000, 1000]), 1e-3)
    with pytest.raises(ValueError, match="dt"):
        pack.step(np.full(4, 1500.0), -1e-3)

    # zero PWM cannot produce thrust, ever
    pack.reset()
    for _ in range(100):
        thrust, torque = pack.step(np.full(4, actuation.PWM_MIN), 1e-3)
    assert np.all(thrust == 0.0) and np.all(torque == 0.0)
    assert np.all(pack.omega == 0.0)

    # commands are clipped to [1000, 2000] us
    pack.step(np.full(4, 5000.0), 1e-3)
    assert np.all(pack.pwm_us == actuation.PWM_MAX)
    pack.step(np.full(4, -5.0), 1e-3)
    assert np.all(pack.pwm_us == actuation.PWM_MIN)

    # a scalar PWM is broadcast to the four rotors
    pack.step(1500.0, 1e-3)
    assert pack.pwm_us.shape == (4,)
    assert np.all(pack.pwm_us == 1500.0)

    # dt == 0 only refreshes the outputs
    before = pack.omega.copy()
    pack.step(np.full(4, 1600.0), 0.0)
    assert np.array_equal(pack.omega, before)


def test_soc_decreases_while_flying():
    pack = actuation.RotorPack(PARAMS)
    trim = pack.trim_hover(soc=1.0)
    pack.reset(pwm_us=trim["pwm_us"])
    soc0 = pack.soc
    for _ in range(1000):
        pack.step(trim["pwm_us"], 1e-3)
    assert pack.soc < soc0
    assert pack.soc > 0.9  # a 3.3 Ah pack cannot empty in one second
    assert pack.v_terminal < PARAMS.battery.open_circuit_voltage(soc0)
