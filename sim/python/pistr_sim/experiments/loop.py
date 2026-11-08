"""One experiment run: plant + estimator + controller + reference.

Two scenarios are supported:

``step``
    Hover at 5 m, then a roll step to 0.1 rad at t = 1 s.  Used for the
    overshoot / settling / RMSE columns of the matrix.
``trajectory``
    The paper's 3-D trajectory (``vx = 4 m/s``, sinusoidal ``vy``/``vz``, yaw
    sweep).  The horizontal position loop is the paper's PD planner; it uses the
    plant's *true* position/velocity as an ideal position aid (there is no GPS in
    the sensor model).  This is a modelling assumption, stated in
    ``docs/RESULTS.md``; the inner-loop comparison is unaffected by it.

The controller always uses the nominal parameters; the fault only changes the
plant (``experiments.faults``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..control import U_MIX, clamp_pwm, recover_u_actual
from ..estimator import (
    AltitudeConfig,
    AltitudeEstimator,
    QuaternionAttitudeEstimator,
    _quat_from_euler,
)
from ..params import default_params
from ..reference import ESO, StepReference, TrajectoryReference3D
from ..types import quat_to_rotmat
from .faults import arm_wind, build_plant

MAG_EARTH_UT = (45.0, 0.0, 20.0)
ALTITUDE_SIGN = np.array([1.0, 1.0, 1.0, -1.0])


@dataclass
class RunTrace:
    scenario: str
    t: np.ndarray
    y_ref: np.ndarray
    y: np.ndarray
    rate_ref: np.ndarray
    rate: np.ndarray
    u: np.ndarray
    u_actual: np.ndarray
    est_y: np.ndarray
    xi: np.ndarray
    xi_ref: np.ndarray
    v: np.ndarray
    theta_hat: np.ndarray
    safe_mode: bool = False
    diverged: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return 4


def _new_controller(controller, plant, fidelity: str):
    controller.reset()
    return controller


def _fly(
    scenario: str,
    fidelity: str,
    controller,
    fault: str,
    seed: int,
    t_end: float,
    altitude0: float = 5.0,
    extras: dict[str, bool] | None = None,
) -> RunTrace:
    started = time.perf_counter()
    nominal = default_params(fidelity)
    h = float(nominal.control.h)
    g = float(nominal.airframe.g)
    plant = build_plant(fidelity, fault, seed=seed, extras=extras)
    trim = plant.trim_hover()
    throttle = float(trim["pwm_us"])

    if scenario == "step":
        reference = StepReference(amplitude=0.1, t_step=1.5, altitude=altitude0)
        planner = None
        state = trim["state"].copy()
        state.xi = np.array([0.0, 0.0, -float(altitude0)])
        state.v = np.zeros(3)
        alt0 = float(altitude0)
    elif scenario == "trajectory":
        # Reduced paper trajectory: keep vx = 4 m/s, the sinusoidal lateral
        # motion and the yaw sweep, but hold altitude (the paper's -20 m + 10 m
        # vertical sinusoid needs a dedicated altitude tracker and is exercised by
        # the step scenario instead).  See docs/RESULTS.md §limitations.
        reference = TrajectoryReference3D(a_vz=0.0, b_vz=-20.0)
        planner = "corrected"
        base = reference.sample(0.0)
        state = trim["state"].copy()
        state.xi = base.xi.copy()
        state.v = base.v_xi.copy()
        alt0 = float(-base.xi[2])
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    state.t = 0.0
    plant.reset(state=state)
    arm_wind(plant)

    eso_roll = ESO(pole=-2.0, h=h)
    eso_pitch = ESO(pole=-2.0, h=h)
    k1_pos = 0.2**2
    k2_pos = 0.4
    max_tilt = np.deg2rad(35.0)
    estimator = QuaternionAttitudeEstimator(
        dt=h, initial_euler=quat_from_euler_zero(), mag_reference=MAG_EARTH_UT
    )
    altitude_estimator = AltitudeEstimator(AltitudeConfig(dt=h))
    altitude_estimator.reset(alt0, 0.0)
    controller = _new_controller(controller, plant, fidelity)

    sub = max(1, int(round(h / float(plant.dt))))
    n_ticks = int(np.floor(t_end / h + 1e-9)) + 1
    order = int(getattr(controller, "order", 2))
    n_state = 2 * order
    rec = {
        "t": np.zeros(n_ticks),
        "y_ref": np.zeros((n_ticks, 4)),
        "y": np.zeros((n_ticks, 4)),
        "rate_ref": np.zeros((n_ticks, 4)),
        "rate": np.zeros((n_ticks, 4)),
        "u": np.zeros((n_ticks, 4)),
        "u_actual": np.zeros((n_ticks, 4)),
        "est_y": np.zeros((n_ticks, 4)),
        "xi": np.zeros((n_ticks, 3)),
        "xi_ref": np.zeros((n_ticks, 3)),
        "v": np.zeros((n_ticks, 3)),
        "theta_hat": np.zeros((n_ticks, 4, n_state)),
    }
    diverged = False

    for k in range(n_ticks):
        t = k * h
        sample = plant.sense()
        estimate = estimator.update(sample, dt=h)
        c_b2i = quat_to_rotmat(_quat_from_euler(*estimate.angle)).T
        a_up = -float((c_b2i @ np.asarray(sample.accel, dtype=float))[2] + g)
        altitude, climb_rate = altitude_estimator.update(sample.baro_alt, a_up, dt=h)

        if planner is None:
            ref = reference.sample(t)
            y_ref = ref.y_ref
            rate_ref = ref.rate_ref
            xi_ref = ref.xi
        else:
            ref = reference.sample(t)
            xi_m, v_m = plant.state.xi, plant.state.v
            a_des = ref.a_xi - k1_pos * (xi_m - ref.xi) - k2_pos * (v_m - ref.v_xi)
            # Thrust-direction inversion for the shared rotation convention:
            # row3(C) = [-sin(pitch), -sin(roll)cos(pitch), cos(roll)cos(pitch)].
            # The published planner's asin(u_x/(-m g)) has the opposite sign for
            # this plant (same class of defect as CONTROL_SPEC §9.9); the exact
            # world->body mapping is used here instead.
            n = np.array([-a_des[0], -a_des[1], g - a_des[2]])
            n /= max(float(np.linalg.norm(n)), 1e-9)
            pitch = float(np.arcsin(np.clip(-n[0], -np.sin(max_tilt), np.sin(max_tilt))))
            c = max(float(np.cos(pitch)), 1e-6)
            roll = float(np.arcsin(np.clip(-n[1] / c, -np.sin(max_tilt), np.sin(max_tilt))))
            _, roll_rate = eso_roll.step(roll)
            _, pitch_rate = eso_pitch.step(pitch)
            if t <= 2.0:
                roll_rate = pitch_rate = 0.0
            # ReferenceSample.altitude is the NED z coordinate; channels are up-positive.
            y_ref = np.array([roll, pitch, ref.yaw, -float(ref.altitude)])
            rate_ref = np.array(
                [roll_rate, pitch_rate, ref.yaw_rate, -float(ref.climb_rate)]
            )
            xi_ref = ref.xi

        y_meas = np.array([estimate.angle[0], estimate.angle[1], estimate.angle[2], altitude])
        rate_meas = np.array([estimate.rate[0], estimate.rate[1], estimate.rate[2], climb_rate])
        u = np.asarray(controller.update(y_ref, y_meas, rate_ref, rate_meas), dtype=float).reshape(4)
        pwm = clamp_pwm(throttle + U_MIX @ (u * ALTITUDE_SIGN))
        u_actual = recover_u_actual(pwm, throttle) * ALTITUDE_SIGN
        controller.ack(u_actual)

        truth = plant.state
        euler = truth.euler
        rec["t"][k] = t
        rec["y_ref"][k] = y_ref
        rec["y"][k] = [euler[0], euler[1], euler[2], -truth.xi[2]]
        rec["rate_ref"][k] = rate_ref
        rec["rate"][k] = [truth.omega[0], truth.omega[1], truth.omega[2], -truth.v[2]]
        rec["u"][k] = u
        rec["u_actual"][k] = u_actual
        rec["est_y"][k] = y_meas
        rec["xi"][k] = truth.xi
        rec["xi_ref"][k] = xi_ref
        rec["v"][k] = truth.v
        theta = np.asarray(controller.theta, dtype=float)
        if theta.shape == (4, n_state):
            rec["theta_hat"][k] = theta

        for _ in range(sub):
            plant.step(pwm)

        if not np.all(np.isfinite(rec["y"][k])) or not np.all(np.isfinite(rec["xi"][k])):
            diverged = True
            rec = {key: val[: k + 1] for key, val in rec.items()}
            break
        if abs(float(rec["y"][k, 3])) > 1e3 or np.max(np.abs(euler)) > np.pi:
            diverged = True
            rec = {key: val[: k + 1] for key, val in rec.items()}
            break

    trace = RunTrace(
        scenario=scenario,
        t=rec["t"],
        y_ref=rec["y_ref"],
        y=rec["y"],
        rate_ref=rec["rate_ref"],
        rate=rec["rate"],
        u=rec["u"],
        u_actual=rec["u_actual"],
        est_y=rec["est_y"],
        xi=rec["xi"],
        xi_ref=rec["xi_ref"],
        v=rec["v"],
        theta_hat=rec["theta_hat"],
        safe_mode=bool(getattr(controller, "safe_mode", False)),
        diverged=diverged,
        metadata={
            "fidelity": fidelity,
            "fault": fault,
            "seed": seed,
            "t_end": t_end,
            "h": h,
            "runtime_s": None,
            "throttle_pwm": throttle,
            "trim": {
                key: float(val)
                for key, val in trim.items()
                if key != "state" and np.isscalar(val)
            },
        },
    )
    trace.metadata["runtime_s"] = float(time.perf_counter() - started)
    return trace


def quat_from_euler_zero() -> tuple[float, float, float]:
    return (0.0, 0.0, 0.0)


def run_step(fidelity, controller, fault="none", seed=0, t_end=6.0, altitude0=5.0, extras=None):
    return _fly("step", fidelity, controller, fault, seed, t_end, altitude0, extras)


def run_trajectory(fidelity, controller, fault="none", seed=0, t_end=6.0, extras=None):
    return _fly("trajectory", fidelity, controller, fault, seed, t_end, 0.0, extras)


__all__ = ["RunTrace", "run_step", "run_trajectory"]
