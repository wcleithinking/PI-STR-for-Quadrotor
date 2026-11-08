"""V3 (frame mapping) and V5 (drop-in for ``QuadrotorPlant``) for the MuJoCo bridge.

Also covers the multi-vehicle world, the sensor model and the ``on_frame`` hook --
i.e. everything M4 owns except the V8 closed-loop cross-check, which lives in
``test_env_crosscheck.py``.

Run with the 3-D dependencies unpacked::

    PYTHONPATH=sim/python:build/mjsim python3 -m pytest sim/python/tests/test_env_bridge.py -s

The rotor pack is ``env.actuation.RotorPack`` as soon as M1 has landed; until then a
local fixture pack with the same physics as ``plant.py`` (ESC lag + 1 us
quantisation, coupled ``(I, omega)`` motor equations, battery OCV/sag,
``T = kt w^2``, ``Q = kq w^2 (1 + k_mu mu^2)``) stands in, so the bridge's own code
paths are exercised without copying M1's module into this file.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pistr_sim.control import U_MIX, clamp_pwm, recover_u_actual
from pistr_sim.env.bridge import (
    MAG_EARTH_UT_NED,
    MuJoCoPlant,
    mj_to_ned,
    ned_to_mj,
    quat_mj_to_ned,
    quat_ned_to_mj,
    rotmat_mj_to_ned,
    rotmat_ned_to_mj,
)
from pistr_sim.estimator import (
    AltitudeConfig,
    AltitudeEstimator,
    QuaternionAttitudeEstimator,
    _quat_from_euler,
)
from pistr_sim.params import Params, default_params
from pistr_sim.plant import QuadrotorPlant
from pistr_sim.types import (
    PWM_MAX,
    PWM_MIN,
    Plant,
    VehicleState,
    quat_normalise,
    quat_to_euler,
    quat_to_rotmat,
)

PARAMS: Params = default_params("high")
S = np.diag([1.0, -1.0, -1.0])
H = float(PARAMS.control.h)


# ---------------------------------------------------------------------------
# Fixtures: model, rotor pack
# ---------------------------------------------------------------------------
def _build_model(n_drones: int = 1, altitude: float = 5.0, params: Params = PARAMS):
    """Contract model from M1 when it exists, else the test's own dev MJCF."""
    try:
        from pistr_sim.env.mjcf import build_model  # type: ignore
    except Exception:  # ImportError (not landed) or a broken import while M1 works
        build_model = None
    if build_model is not None:
        try:
            return build_model(params, scene=None, n_drones=n_drones)
        except Exception as exc:  # pragma: no cover - M1 interface drift
            print(f"[fixture] env.mjcf.build_model failed ({exc!r}); using dev MJCF")
    if n_drones == 1:
        from pistr_sim.env._devmodel import build_dev_model

        return build_dev_model(altitude, params=params)
    return mujoco.MjModel.from_xml_string(_dev_multibody_xml(n_drones, altitude, params))


def _dev_multibody_xml(n_drones: int, altitude: float, params: Params) -> str:
    """Minimal N-body MJCF fixture (only used before/without ``env.mjcf``)."""
    a = params.airframe
    J = np.diag(np.asarray(a.inertia, dtype=float))
    diag = " ".join(f"{v:.9f}" for v in J)
    bodies = []
    for i in range(n_drones):
        x = (i - 0.5 * (n_drones - 1)) * 2.0
        bodies.append(
            f'    <body name="drone{i}" pos="{x:.4f} 0 {altitude:.3f}">\n'
            f'      <freejoint name="drone{i}_free"/>\n'
            f'      <inertial pos="0 0 0" mass="{a.mass:.9f}" diaginertia="{diag}"/>\n'
            f'      <geom name="drone{i}_body" type="box" size="0.16 0.16 0.04" '
            f'rgba=".18 .20 .24 1"/>\n'
            f"    </body>"
        )
    return (
        '<mujoco model="pistr-bridge-dev">\n'
        '  <compiler angle="radian" inertiafromgeom="false"/>\n'
        '  <option gravity="0 0 -9.81" timestep="0.001"/>\n'
        "  <worldbody>\n"
        '    <light pos="6 -6 12" dir="-0.4 0.4 -1" diffuse=".8 .8 .8"/>\n'
        '    <geom name="floor" type="plane" size="40 40 .05" rgba=".4 .5 .6 1"/>\n'
        + "\n".join(bodies)
        + "\n  </worldbody>\n</mujoco>\n"
    )


class _DevRotorPack:
    """Test fixture with ``plant.py``'s actuator physics (see the module docstring).

    Deliberately does **not** provide ``trim_hover``/``wrench``: that exercises the
    bridge's numerical-trim fallback and its own wrench assembly.  When M1's pack is
    importable the tests switch to it and the forwarding paths are exercised instead.
    """

    def __init__(self, params: Params = PARAMS, *, sub_steps: int = 6):
        self.params = params
        self.motor = params.motor
        self.esc = params.esc
        self.battery = params.battery
        self.kt = float(params.rotor.kt)
        self.kq = float(params.rotor.kq)
        self.k_mu = float(params.rotor.k_mu)
        self.radius = float(params.airframe.r_rotor)
        self.omega_min = float(params.rotor.omega_min)
        self.sub_steps = int(sub_steps)
        self.reset()

    # -- steady state ---------------------------------------------------
    def _voc(self) -> float:
        return float(self.battery.open_circuit_voltage(self.soc))

    def _denominator(self, duty: np.ndarray) -> np.ndarray:
        return self.motor.resistance + 4.0 * duty * self.battery.r_int

    def _omega_ss(self, duty: np.ndarray) -> np.ndarray:
        den = self._denominator(duty)
        b = self.motor.km * self.motor.ke / den
        c = -self.motor.km * duty * self._voc() / den
        return (-b + np.sqrt(b**2 - 4.0 * self.kq * c)) / (2.0 * self.kq)

    def _current_ss(self, duty: np.ndarray, omega: np.ndarray) -> np.ndarray:
        return (duty * self._voc() - self.motor.ke * omega) / self._denominator(duty)

    def reset(self, *, pwm_us=None, soc: float = 1.0) -> None:
        self.soc = float(soc)
        pwm = np.full(4, 1394.0) if pwm_us is None else np.asarray(pwm_us, dtype=float)
        self.pwm_us = np.clip(pwm, PWM_MIN, PWM_MAX)
        self.duty = np.clip((self.pwm_us - PWM_MIN) / PWM_MIN, 0.0, 1.0)
        self.omega = np.maximum(self._omega_ss(self.duty), 0.0)
        self.current = np.maximum(self._current_ss(self.duty, self.omega), 0.0)
        self.v_terminal = self._voc() - self.battery.r_int * float(np.sum(self.current))

    # -- one actuator step ----------------------------------------------
    def step(self, pwm_us, dt: float, *, v_rel_body=None):
        pwm = np.clip(np.asarray(pwm_us, dtype=float).reshape(4), PWM_MIN, PWM_MAX)
        self.pwm_us = pwm
        d_cmd = np.round(np.clip((pwm - PWM_MIN) / PWM_MIN, 0.0, 1.0) / 1e-3) * 1e-3
        m, bat = self.motor, self.battery
        v_rel = np.zeros(3) if v_rel_body is None else np.asarray(v_rel_body, dtype=float)
        v_plane = float(np.hypot(v_rel[0], v_rel[1]))
        h = float(dt) / self.sub_steps
        w = np.maximum(np.asarray(self.omega, dtype=float), 0.0)
        cur = np.maximum(np.asarray(self.current, dtype=float), 0.0)
        duty = np.clip(np.asarray(self.duty, dtype=float), 0.0, 1.0)
        for _ in range(self.sub_steps):
            v_term = self._voc() - bat.r_int * float(np.sum(cur))
            v_motor = duty * v_term
            d_cur = (v_motor - m.ke * w - m.resistance * cur) / m.inductance
            d_cur = np.where((cur <= 0.0) & (d_cur < 0.0), 0.0, d_cur)
            mu = v_plane / (np.maximum(w, self.omega_min) * self.radius)
            q_load = self.kq * w**2 * (1.0 + self.k_mu * mu**2)
            d_w = (m.km * cur - q_load) / m.inertia
            d_w = np.where((w <= 0.0) & (d_w < 0.0), 0.0, d_w)
            d_duty = (d_cmd - duty) / max(float(self.esc.tau), 1e-9)
            w = np.maximum(w + h * d_w, 0.0)
            cur = np.maximum(cur + h * d_cur, 0.0)
            duty = np.clip(duty + h * d_duty, 0.0, 1.0)
            self.soc = float(
                np.clip(
                    self.soc
                    - h * float(np.dot(v_motor, cur)) / (bat.capacity_ah * bat.v_nom * 3600.0),
                    0.0,
                    1.0,
                )
            )
        self.omega, self.current, self.duty = w, cur, duty
        self.v_terminal = self._voc() - bat.r_int * float(np.sum(cur))
        mu = v_plane / (np.maximum(w, self.omega_min) * self.radius)
        thrust = self.kt * w**2
        torque_z = self.kq * w**2 * (1.0 + self.k_mu * mu**2)
        return thrust, torque_z


def _pack_factory(params: Params = PARAMS):
    """``(params) -> pack``: M1's pack when importable, else the local fixture."""
    try:
        from pistr_sim.env.actuation import RotorPack  # type: ignore
    except Exception:
        RotorPack = None
    if RotorPack is None:
        return lambda: _DevRotorPack(params)
    for args, kwargs in (((params,), {}), ((), {"params": params}), ((), {})):
        try:
            RotorPack(*args, **kwargs)
        except TypeError:
            continue
        return lambda: RotorPack(*args, **kwargs)
    return lambda: _DevRotorPack(params)


def _pack_name() -> str:
    try:
        from pistr_sim.env.actuation import RotorPack  # type: ignore  # noqa: F401

        return "env.actuation.RotorPack"
    except Exception:
        return "test fixture _DevRotorPack (M1 not landed)"


@pytest.fixture(scope="module")
def plant() -> MuJoCoPlant:
    factory = _pack_factory()
    p = MuJoCoPlant(params=PARAMS, dt=0.002, rotor_pack_factory=lambda i: factory())
    print(f"\n[fixtures] rotor pack = {_pack_name()}; {p!r}")
    return p


@pytest.fixture(scope="module")
def trim(plant: MuJoCoPlant) -> dict:
    info = plant.trim_hover()
    print(
        f"[fixtures] trim pwm={info['pwm_us']:.2f} us  thrust={info['thrust']:.4f} N  "
        f"m*g={PARAMS.airframe.mass * PARAMS.airframe.g:.4f} N  "
        f"omega={info['rotor_omega']:.2f} rad/s"
    )
    return info


def _hover_state(altitude: float = 5.0, *, xi=None) -> VehicleState:
    st = VehicleState()
    st.xi = np.array([0.0, 0.0, -altitude]) if xi is None else np.asarray(xi, dtype=float)
    st.q = np.array([1.0, 0.0, 0.0, 0.0])
    st.pwm_us = np.full(4, PARAMS.pwm_hover_us)
    return st


# ---------------------------------------------------------------------------
# V3 -- frame mapping
# ---------------------------------------------------------------------------
def test_v3_level_hover_stays_level(plant: MuJoCoPlant, trim: dict) -> None:
    """A level, at-rest vehicle is level in MuJoCo, and its position maps with S."""
    st = trim["state"].copy()
    st.xi = np.array([1.0, 2.0, -5.0])
    plant.reset(state=st)
    pos, rot = plant.mj_pose()
    assert np.allclose(pos, ned_to_mj(st.xi), atol=1e-12), "position must map p_mj = S p_ned"
    assert np.allclose(rot, np.eye(3), atol=1e-12), "level NED attitude must be level in MuJoCo"

    pwm = np.full(4, trim["pwm_us"])
    state = plant.state
    for _ in range(125):  # 0.25 s at the trim throttle
        state = plant.step(pwm)
    _, rot = plant.mj_pose()
    euler = quat_to_euler(plant.state.q)
    print(
        f"[V3] hover 0.25 s with trim pwm: euler={np.round(euler, 9)} rad  "
        f"alt={plant.height():.5f} m  |R-I|max={np.max(np.abs(rot - np.eye(3))):.2e}"
    )
    assert np.max(np.abs(euler)) < 2e-3, "a level hover must stay level (rad)"
    assert abs(plant.height() - 5.0) < 0.02, "trim throttle must hold altitude"


def test_v3_positive_roll_tilts_toward_minus_y(plant: MuJoCoPlant) -> None:
    """NED ``roll > 0`` tilts the thrust axis toward MuJoCo ``-y`` and equals S@thrust_ned."""
    for roll in (0.1, -0.1):
        q = _quat_from_euler(roll, 0.0, 0.0)
        st = _hover_state(5.0)
        st.q = q
        plant.reset(state=st)
        _, rot = plant.mj_pose()

        c_i2b = quat_to_rotmat(q)
        thrust_ned = c_i2b.T @ np.array([0.0, 0.0, -1.0])  # NED body -z == up
        thrust_mj = rot @ np.array([0.0, 0.0, 1.0])  # MuJoCo body +z == up
        print(
            f"[V3] roll={roll:+.2f}  thrust_ned={np.round(thrust_ned, 5)}  "
            f"thrust_mj={np.round(thrust_mj, 5)}  S@thrust_ned={np.round(ned_to_mj(thrust_ned), 5)}"
        )
        assert np.allclose(thrust_mj, ned_to_mj(thrust_ned), atol=1e-12), (
            "the thrust axis in MuJoCo must be the S image of the NED thrust axis"
        )
        assert thrust_mj[2] > 0.9, "thrust must still point mostly up"
        if roll > 0.0:
            assert thrust_mj[1] < -0.05, "NED roll>0 (right wing down) appears as -y in MuJoCo"
        else:
            assert thrust_mj[1] > 0.05


def test_v3_yaw_matches_ned_convention(plant: MuJoCoPlant) -> None:
    """NED ``yaw`` round-trips through MuJoCo and keeps its sign convention."""
    for yaw in (0.3, -0.3, 1.2):
        q = _quat_from_euler(0.0, 0.0, yaw)
        st = _hover_state(5.0)
        st.q = q
        plant.reset(state=st)
        _, rot = plant.mj_pose()

        mj_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        nose_mj = rot[:, 0]
        back = float(quat_to_euler(plant.state.q)[2])
        print(
            f"[V3] yaw={yaw:+.2f}  mj_yaw={mj_yaw:+.6f}  nose_mj={np.round(nose_mj, 5)}  "
            f"quat_to_euler(back)={back:+.6f}"
        )
        assert mj_yaw == pytest.approx(-yaw, abs=1e-12)
        # nose direction is the S image of the NED nose direction
        assert np.allclose(nose_mj, ned_to_mj([np.cos(yaw), np.sin(yaw), 0.0]), atol=1e-12)
        assert back == pytest.approx(yaw, abs=1e-12)


def test_v3_random_attitudes_roundtrip(plant: MuJoCoPlant) -> None:
    """50 random attitudes: state and rotation round-trip, plus an independent check."""
    rng = np.random.default_rng(11)
    worst_state = worst_rot = worst_scipy = 0.0
    try:
        from scipy.spatial.transform import Rotation as SciRotation
    except Exception:  # pragma: no cover - scipy ships with the wind model
        SciRotation = None
    for _ in range(50):
        euler = rng.uniform(-0.7, 0.7, 3)
        q = _quat_from_euler(*euler)
        st = _hover_state(3.0)
        st.q = q
        st.xi = np.array([1.0, 2.0, -3.0])
        plant.reset(state=st)
        pos, rot = plant.mj_pose()
        c_i2b = quat_to_rotmat(q)

        worst_state = max(worst_state, float(np.max(np.abs(quat_to_euler(plant.state.q) - euler))))
        worst_rot = max(worst_rot, float(np.max(np.abs(rot - S @ c_i2b.T @ S))))
        assert np.allclose(pos, np.array([1.0, -2.0, 3.0]), atol=1e-12)
        if SciRotation is not None:
            r_b2i = SciRotation.from_matrix(c_i2b.T).as_matrix()
            worst_scipy = max(worst_scipy, float(np.max(np.abs(rot - S @ r_b2i @ S))))
    print(
        f"[V3] random attitudes: state round-trip {worst_state:.2e}  "
        f"R_mj vs S R_b2i S {worst_rot:.2e}  scipy {worst_scipy:.2e}"
    )
    assert worst_state < 1e-10
    assert worst_rot < 1e-10
    if SciRotation is not None:
        assert worst_scipy < 1e-10


def test_frame_helpers_are_consistent() -> None:
    q = quat_normalise(_quat_from_euler(0.3, -0.2, 0.5))
    c_i2b = quat_to_rotmat(q)
    assert np.allclose(quat_mj_to_ned(quat_ned_to_mj(q)), q, atol=1e-15)
    assert np.allclose(rotmat_mj_to_ned(rotmat_ned_to_mj(c_i2b)), c_i2b, atol=1e-14)
    # quaternion and matrix forms of the same map must agree
    r_mj = rotmat_ned_to_mj(c_i2b)
    q_mj = quat_ned_to_mj(q)
    from pistr_sim.types import quat_to_rotmat as _q2r

    # MuJoCo stores body->world, so its quaternion resolves to R_b2w under the
    # standard convention -- i.e. quat_to_rotmat(q_mj).T in the repository ordering.
    assert np.allclose(_q2r(q_mj).T, r_mj, atol=1e-14)
    assert np.allclose(ned_to_mj(mj_to_ned([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0], atol=1e-15)
    assert np.allclose(S @ S, np.eye(3), atol=0.0)


# ---------------------------------------------------------------------------
# V5 -- drop-in for QuadrotorPlant
# ---------------------------------------------------------------------------
def test_v5_state_contract(plant: MuJoCoPlant, trim: dict) -> None:
    """Fixed PWM sequence: shapes, units, dt, quat norm, no NaN, pwm echo."""
    st = trim["state"].copy()
    st.xi = np.array([0.0, 0.0, -5.0])
    plant.reset(state=st)
    base = float(trim["pwm_us"])
    pattern = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 20.0, 0.0, 0.0], [-10.0, 0.0, 5.0, 5.0],
                        [0.0, -20.0, 0.0, -5.0]])
    for k in range(100):
        pwm = np.clip(base + pattern[k % len(pattern)], PWM_MIN, PWM_MAX)
        state = plant.step(pwm)
        assert isinstance(state, VehicleState)
        assert state.xi.shape == (3,)
        assert state.v.shape == (3,)
        assert state.omega.shape == (3,)
        assert state.q.shape == (4,)
        assert state.rotor_omega.shape == (4,)
        assert state.esc_duty.shape == (4,)
        assert state.pwm_us.shape == (4,)
        assert np.all(np.isfinite(state.xi)) and np.all(np.isfinite(state.v))
        assert np.all(np.isfinite(state.q)) and np.all(np.isfinite(state.omega))
        assert np.all(np.isfinite(state.rotor_omega))
        assert np.linalg.norm(state.q) == pytest.approx(1.0, abs=1e-12)
        assert np.max(np.abs(np.asarray(state.pwm_us) - pwm)) < 1e-9
        assert state.t == pytest.approx((k + 1) * plant.dt, abs=1e-12)
        assert -8.0 < state.xi[2] < 5.0, "the fixed sequence must not explode"
    # out-of-range commands are clipped, like QuadrotorPlant
    state = plant.step(np.array([0.0, 5000.0, -5000.0, 1000.0]))
    assert np.all(np.asarray(state.pwm_us) >= PWM_MIN)
    assert np.all(np.asarray(state.pwm_us) <= PWM_MAX)


def test_v5_same_units_and_signs_as_python_plant() -> None:
    """The same command produces the same physics as ``fidelity='high'``."""
    py = QuadrotorPlant(PARAMS)
    factory = _pack_factory()
    mj = MuJoCoPlant(params=PARAMS, dt=0.002, rotor_pack_factory=lambda i: factory())
    trim_py = py.trim_hover()
    trim_mj = mj.trim_hover()
    print(
        f"[V5] trim pwm: python {trim_py['pwm_us']:.2f} us vs mujoco {trim_mj['pwm_us']:.2f} us"
        f" (delta {trim_mj['pwm_us'] - trim_py['pwm_us']:+.3f} us)"
    )
    assert abs(trim_mj["pwm_us"] - trim_py["pwm_us"]) < 15.0
    assert abs(trim_mj["thrust"] - PARAMS.airframe.mass * PARAMS.airframe.g) < 0.05

    # (a) free fall: both must accelerate along NED +z at ~g
    st_py = trim_py["state"].copy()
    st_py.xi = np.array([0.0, 0.0, -5.0])
    py.reset(state=st_py)
    st_mj = trim_mj["state"].copy()
    st_mj.xi = np.array([0.0, 0.0, -5.0])
    mj.reset(state=st_mj)
    idle = np.full(4, PWM_MIN)
    for _ in range(200):  # 0.2 s of Python plant (dt = 1 ms)
        s_py = py.step(idle)
    for _ in range(100):  # 0.2 s of MuJoCo plant (dt = 2 ms)
        s_mj = mj.step(idle)
    print(
        f"[V5] free fall 0.2 s: python v_z={s_py.v[2]:+.4f} m/s, "
        f"mujoco v_z={s_mj.v[2]:+.4f} m/s, g*t={PARAMS.airframe.g * 0.2:.4f}"
    )
    assert s_py.v[2] > 0.0 and s_mj.v[2] > 0.0
    assert s_mj.v[2] == pytest.approx(s_py.v[2], rel=0.15)

    # (b) trim hover: both hold altitude
    py.reset(state=st_py)
    mj.reset(state=st_mj)
    for _ in range(200):
        s_py = py.step(np.full(4, trim_py["pwm_us"]))
    for _ in range(100):
        s_mj = mj.step(np.full(4, trim_mj["pwm_us"]))
    print(
        f"[V5] trim hover 0.2 s: python alt={-s_py.xi[2]:.5f} m, "
        f"mujoco alt={-s_mj.xi[2]:.5f} m"
    )
    assert abs(-s_py.xi[2] - 5.0) < 0.02
    assert abs(-s_mj.xi[2] - 5.0) < 0.02

    # (c) the same +roll differential rolls both vehicles the same way
    for _ in range(50):  # 0.05 s of Python plant
        py.step(clamp_pwm(np.full(4, trim_py["pwm_us"]) + U_MIX @ np.array([20.0, 0.0, 0.0, 0.0])))
    for _ in range(25):  # 0.05 s of MuJoCo plant
        s_mj = mj.step(clamp_pwm(np.full(4, trim_mj["pwm_us"]) + U_MIX @ np.array([20.0, 0.0, 0.0, 0.0])))
    roll_py = float(py.state.euler[0])
    roll_mj = float(s_mj.euler[0])
    print(f"[V5] +roll differential 0.05 s: python roll={roll_py:+.5f} rad, mujoco roll={roll_mj:+.5f} rad")
    assert roll_py > 0.0 and roll_mj > 0.0
    assert roll_mj == pytest.approx(roll_py, rel=0.35)


def test_v5_is_a_duck_typed_plant(plant: MuJoCoPlant) -> None:
    assert isinstance(plant, Plant), "MuJoCoPlant must satisfy types.Plant structurally"
    for name in ("dt", "state", "reset", "step", "sense", "set_wind", "trim_hover", "evaluate"):
        assert hasattr(plant, name), f"missing QuadrotorPlant attribute {name}"
    assert plant.dt == pytest.approx(0.002)
    sample = plant.sense()
    assert sample.gyro.shape == (3,) and sample.accel.shape == (3,) and sample.mag.shape == (3,)
    assert np.all(np.isfinite(sample.gyro)) and np.isfinite(sample.baro_alt)
    info = plant.evaluate()
    for key in ("thrust", "F_body", "M_body", "F_drag_body", "c_i2b"):
        assert key in info, f"evaluate() must expose the plant.py key {key}"
    plant.set_wind(np.array([2.0, 0.0, 0.0]), turbulence=False)
    assert np.allclose(plant.wind.at(0.0), [2.0, 0.0, 0.0])
    plant.set_wind(np.zeros(3), turbulence=False)


def test_v5_sense_reports_specific_force_not_gravity() -> None:
    """The accelerometer reads the aero wrench, so level hover is (0, 0, -g) like the plant."""
    factory = _pack_factory()
    p = MuJoCoPlant(
        params=PARAMS, dt=0.002, sensor_noise=False, rotor_pack_factory=lambda i: factory()
    )
    trim = p.trim_hover()
    st = trim["state"].copy()
    st.xi = np.array([0.0, 0.0, -5.0])
    p.reset(state=st)
    sample = p.sense()
    g = PARAMS.airframe.g
    print(
        f"[V5] clean sense at hover: accel={np.round(sample.accel, 6)}  "
        f"baro={sample.baro_alt:.6f}  mag={np.round(sample.mag, 4)}  gyro={np.round(sample.gyro, 8)}"
    )
    assert sample.accel[2] == pytest.approx(-g, rel=0.03)
    assert abs(sample.accel[0]) < 0.05 and abs(sample.accel[1]) < 0.05
    assert sample.baro_alt == pytest.approx(5.0, abs=1e-6)
    assert np.allclose(sample.mag, MAG_EARTH_UT_NED, atol=1e-6)
    assert np.max(np.abs(sample.gyro)) < 1e-9
    # after one more controller period the sample still matches the physics
    for _ in range(20):
        p.step(np.full(4, trim["pwm_us"]))
    sample = p.sense()
    assert sample.accel[2] == pytest.approx(-g, rel=0.05)


def test_v5_control_stack_drives_it_unchanged(plant: MuJoCoPlant, trim: dict) -> None:
    """The unmodified estimator + PI-STR cascade closes the loop on the bridge."""
    from pistr_sim.experiments.controllers import build_controller

    st = trim["state"].copy()
    st.xi = np.array([0.0, 0.0, -5.0])
    plant.reset(state=st)
    controller = build_controller("pi_str", PARAMS, "high")
    controller.reset()
    estimator = QuaternionAttitudeEstimator(dt=H, mag_reference=(45.0, 0.0, 20.0))
    altitude_estimator = AltitudeEstimator(AltitudeConfig(dt=H))
    altitude_estimator.reset(5.0, 0.0)
    throttle = float(trim["pwm_us"])
    sign = np.array([1.0, 1.0, 1.0, -1.0])
    for _ in range(10):
        sample = plant.sense()
        estimate = estimator.update(sample, dt=H)
        c_b2i = quat_to_rotmat(_quat_from_euler(*estimate.angle)).T
        a_up = -float((c_b2i @ np.asarray(sample.accel, dtype=float))[2] + PARAMS.airframe.g)
        altitude, climb = altitude_estimator.update(sample.baro_alt, a_up, dt=H)
        y_ref = np.array([0.0, 0.0, 0.0, 5.0])
        u = np.asarray(
            controller.update(
                y_ref,
                np.array([*estimate.angle, altitude]),
                np.zeros(4),
                np.array([*estimate.rate, climb]),
            ),
            dtype=float,
        )
        pwm = clamp_pwm(throttle + U_MIX @ (u * sign))
        controller.ack(recover_u_actual(pwm, throttle) * sign)
        for _ in range(int(round(H / plant.dt))):
            plant.step(pwm)
    assert np.all(np.isfinite(plant.state.xi))
    assert abs(plant.height() - 5.0) < 0.25
    assert PWM_MIN <= float(np.mean(plant.state.pwm_us)) <= PWM_MAX


# ---------------------------------------------------------------------------
# Multi-vehicle world
# ---------------------------------------------------------------------------
def test_multi_vehicle_independent_dynamics() -> None:
    factory = _pack_factory()
    model = _build_model(2)
    p = MuJoCoPlant(
        params=PARAMS, model=model, n_drones=2, dt=0.002,
        rotor_pack_factory=lambda i: factory(),
    )
    assert p.n_drones == 2
    assert [d.name for d in p._drones] == ["drone0", "drone1"]  # noqa: SLF001 - fixture check
    print(f"[multi] {p!r}")

    trim = p.trim_hover()
    states = []
    for i, x in enumerate((-1.5, 1.5)):
        st = trim["state"].copy()
        st.xi = np.array([x, 0.0, -5.0])
        st.v = np.zeros(3)
        states.append(st)
    p.reset(state=states)
    assert p.state_of(0).xi[0] != p.state_of(1).xi[0]
    pos0, _ = p.mj_pose(0)
    pos1, _ = p.mj_pose(1)
    assert pos0[0] < pos1[0]

    # different commands per vehicle -> different physics (independent free joints)
    pwm = np.array([[1400.0] * 4, [PWM_MIN] * 4])
    for _ in range(100):  # 0.2 s
        p.step(pwm)
    h0, h1 = p.height(0), p.height(1)
    print(f"[multi] after 0.2 s: drone0 alt={h0:.4f} m (hovering), drone1 alt={h1:.4f} m (idle)")
    assert h0 > h1 + 0.05, "each drone must be driven by its own PWM row"
    assert np.all(np.isfinite(p.state_of(1).xi))


def test_multi_vehicle_requires_a_matching_model() -> None:
    factory = _pack_factory()
    model = _build_model(1)
    with pytest.raises(ValueError):
        MuJoCoPlant(
            params=PARAMS, model=model, n_drones=2, dt=0.002,
            rotor_pack_factory=lambda i: factory(),
        )


def test_on_frame_hook_and_multi_drone_trace() -> None:
    from pistr_sim.env.loop import run_flight

    factory = _pack_factory()
    model = _build_model(2)
    frames: list[float] = []

    def on_frame(t: float, data) -> None:
        frames.append(float(t))
        assert data is not None, "the MuJoCo backend must hand MjData to on_frame"

    trace = run_flight(
        scenario="hover",
        t_end=0.8,
        controller="pi_str",
        model=model,
        n_drones=2,
        rotor_pack_factory=lambda i: factory(),
        on_frame=on_frame,
        frame_stride=2,
    )
    expected = len(range(0, trace.metadata["n_ticks"], 2))
    print(f"[multi] frames={len(frames)} expected={expected} drones={trace.n_drones}")
    assert len(frames) == expected
    assert trace.n_drones == 2
    assert trace.stacked("xi").shape == (2, trace.t.size, 3)
    assert np.max(np.abs(trace.stacked("euler"))) < 0.2
    for i in (0, 1):
        altitude = trace.drone(i).altitude
        assert abs(float(altitude[-1]) - 5.0) < 0.3, f"drone{i} must hold altitude"
    assert not trace.metadata["diverged"]
    # the trace's frame-0 position is the NED position, not the MuJoCo one
    assert np.allclose(trace.drone(0).xi[0], [-1.0, 0.0, -5.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Wrench cross-check against M1's own assembly, yaw sign, scene pass-through
# ---------------------------------------------------------------------------
def test_bridge_wrench_matches_pack_wrench(plant: MuJoCoPlant, trim: dict) -> None:
    """The bridge's NED assembly must be the ``S`` image of ``RotorPack.wrench()``.

    ``wrench()`` (M1) returns ``(F, M)`` in the FLU body frame with the arm moments
    and the yaw reaction; the bridge assembles the same wrench in NED body axes and
    maps it to the world.  Comparing them catches any sign slip in either assembly
    without duplicating the formula in the assertion.
    """
    pack = plant._drones[0].pack  # noqa: SLF001 - test-only access to the pack
    wrench = getattr(pack, "wrench", None)
    if wrench is None:
        pytest.skip("this rotor pack has no wrench(); nothing to cross-check")
    st = trim["state"].copy()
    st.xi = np.array([0.0, 0.0, -5.0])
    plant.reset(state=st)
    aero = plant.aero_of(0)
    f_flu, m_flu = wrench()
    f_bridge = S @ np.asarray(aero["F_body"], dtype=float)
    m_bridge = S @ np.asarray(aero["M_body"], dtype=float)
    print(
        f"[wrench] F bridge={np.round(f_bridge, 9)} pack={np.round(f_flu, 9)} | "
        f"M bridge={np.round(m_bridge, 9)} pack={np.round(m_flu, 9)}"
    )
    # The bridge records the wrench of the last 1 ms actuator substep while
    # ``wrench()`` is evaluated on the pack *after* that substep, so the tolerance is
    # the substep's own thrust change (~1e-6 relative at the trim point) plus the
    # residual drag of the 1e-6 m/s wind sample.  Any sign error is O(1) and still fails.
    assert np.allclose(f_bridge, f_flu, rtol=1e-5, atol=1e-6)
    assert np.allclose(m_bridge, m_flu, rtol=1e-5, atol=1e-6)


def test_v5_yaw_command_sign_matches_plant() -> None:
    """A positive yaw differential yaws both vehicles the same way (plant.py G3)."""
    factory = _pack_factory()
    py = QuadrotorPlant(PARAMS)
    mj = MuJoCoPlant(params=PARAMS, dt=0.002, rotor_pack_factory=lambda i: factory())
    trim_py, trim_mj = py.trim_hover(), mj.trim_hover()
    pwm_roll_py = clamp_pwm(
        np.full(4, trim_py["pwm_us"]) + U_MIX @ np.array([0.0, 0.0, 20.0, 0.0])
    )
    pwm_roll_mj = clamp_pwm(
        np.full(4, trim_mj["pwm_us"]) + U_MIX @ np.array([0.0, 0.0, 20.0, 0.0])
    )
    st_py = trim_py["state"].copy()
    st_py.xi = np.array([0.0, 0.0, -5.0])
    py.reset(state=st_py)
    st_mj = trim_mj["state"].copy()
    st_mj.xi = np.array([0.0, 0.0, -5.0])
    mj.reset(state=st_mj)
    for _ in range(50):  # 0.05 s
        py.step(pwm_roll_py)
    for _ in range(25):  # 0.05 s
        s_mj = mj.step(pwm_roll_mj)
    yaw_py = float(py.state.euler[2])
    yaw_mj = float(s_mj.euler[2])
    print(f"[V5] +yaw differential 0.05 s: python yaw={yaw_py:+.6f} rad, mujoco yaw={yaw_mj:+.6f} rad")
    assert yaw_py > 0.0 and yaw_mj > 0.0
    assert yaw_mj == pytest.approx(yaw_py, rel=0.35)


def test_run_flight_accepts_a_scene_spec() -> None:
    """``run_flight(scene=SceneSpec(...))`` must convert the dataclass to MJCF text."""
    from pistr_sim.env.loop import run_flight
    from pistr_sim.env.scene import SceneSpec

    spec = SceneSpec(
        buildings=2, trees=2, obstacles=1, sky=False, wind_markers=False, extent=20.0, seed=3
    )
    trace = run_flight(
        scenario="hover", t_end=0.24, controller="pid", scene=spec, altitude0=3.0
    )
    mj_plant = trace.metadata.get("backend")
    print(f"[scene] SceneSpec accepted, backend={mj_plant}, ticks={trace.metadata['n_ticks']}")
    assert mj_plant == "mujoco"
    assert not trace.metadata["diverged"]
    assert abs(float(trace.altitude[-1]) - 3.0) < 0.3


def test_minimal_pack_only_needs_reset_and_step() -> None:
    """A pack with only the contract's ``reset``/``step`` must still fly the bridge.

    This is the bridge's numerical-trim fallback and its own wrench assembly; it keeps
    working in a checkout where ``env/actuation.py`` is not importable.
    """
    p = MuJoCoPlant(
        params=PARAMS, dt=0.002, rotor_pack_factory=lambda i: _DevRotorPack(PARAMS)
    )
    trim = p.trim_hover()
    hover = PARAMS.airframe.mass * PARAMS.airframe.g
    st = trim["state"].copy()
    st.xi = np.array([0.0, 0.0, -5.0])
    p.reset(state=st)
    for _ in range(250):  # 0.5 s
        p.step(np.full(4, trim["pwm_us"]))
    print(
        f"[pack] minimal pack: trim {trim['pwm_us']:.2f} us thrust {trim['thrust']:.4f} N "
        f"(m*g {hover:.4f}), alt after 0.5 s {p.height():.4f} m"
    )
    assert abs(trim["thrust"] - hover) < 0.1
    assert abs(p.height() - 5.0) < 0.05
