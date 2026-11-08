"""Physics-plant closed loop: estimator + PI-STR + mixer + ``pistr_sim.plant``.

This module is the *glue* that lets the controller stack fly the high-fidelity
plant.  The controllers themselves never import the plant; only this adapter
does.

Conventions fixed here (asserted by ``tests/test_integration_physics.py``)
-----------------------------------------------------------------------
* The plant is **NED** (``x_i[2]`` down, altitude ``h = -x_i[2]``,
  ``docs/CONTROL_SPEC.md`` §9.7).  The controller channels stay SI/up-positive:
  ``y = [roll, pitch, yaw, altitude]``.
* The plant's accelerometer reports ``C_i2b (a - g e_z)`` with ``e_z`` down, so
  at level rest it reads ``(0, 0, -g)``.  The estimator therefore runs with
  ``accel_sign = -1``.
* The magnetometer sees a 3-D field (``_MAG_EARTH = (45, 0, 20) uT``); the
  estimator uses ``mag_model="vector"`` (exact for the shared rotation
  convention, including the dip angle).
* ``U_MIX`` column 4 is ``[-1,-1,-1,-1]`` (positive differential lowers every
  throttle), while the controller's altitude channel is up-positive.  The
  adapter therefore maps ``u_mix = u * [1, 1, 1, -1]`` and inverts the same map
  for the ``u_actual`` acknowledgement.
* Hover throttle comes from ``plant.trim_hover()``; nothing is hard-coded.

Observer choice for the bench
-----------------------------
``use_accel=False`` by default.  Without position/velocity aiding, a
gravity-referenced tilt observer cannot separate gravity from translational
acceleration (a coordinated acceleration reads as "level"), and in closed loop
that ambiguity produces a slow drift/limit-cycle.  The bench therefore observes
attitude with the gyro + 3-D magnetometer using
:class:`pistr_sim.estimator.QuaternionAttitudeEstimator` (body-rate quaternion
propagation), which stays exact for combined roll/pitch/yaw motion where the
firmware's independent Euler-rate integration does not.  The residual gyro
random-walk drift is a sensor limitation, not controller behaviour.  Pass
``use_accel=True`` to run the firmware-faithful Kalman filter instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..estimator import (
    AltitudeConfig,
    AltitudeEstimator,
    AttitudeEstimator,
    KalmanConfig,
    QuaternionAttitudeEstimator,
    _quat_from_euler,
)
from ..params import Params
from ..types import quat_to_rotmat
from .mixer import U_MIX, clamp_pwm, recover_u_actual

#: Plant-inertial magnetic field (must match ``pistr_sim.plant._MAG_EARTH``).
MAG_EARTH_UT: tuple[float, float, float] = (45.0, 0.0, 20.0)
#: Channel sign map between the controller's up-positive channels and the mixer.
ALTITUDE_SIGN = np.array([1.0, 1.0, 1.0, -1.0])


@dataclass
class PhysicsTrace:
    """Time history of a physics closed-loop run (row index = controller tick)."""

    t: np.ndarray
    y_ref: np.ndarray            # reference [roll,pitch,yaw,alt] (SI)
    y: np.ndarray                # *true* plant [roll,pitch,yaw,alt]
    rate_ref: np.ndarray         # reference rates
    rate: np.ndarray             # true body rates / climb rate
    u: np.ndarray                # design differential commands
    u_actual: np.ndarray         # realised differential commands (after PWM clip)
    pwm: np.ndarray              # clipped PWM microseconds
    est_y: np.ndarray            # estimator angle + altitude estimate
    est_rate: np.ndarray         # estimator rate + climb-rate estimate
    theta_hat: np.ndarray        # RLS parameters [tick, channel, 2*order]
    order: int = 2
    safe_mode: bool = False
    diverged: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return int(self.y.shape[1])

    @property
    def estimator_error(self) -> np.ndarray:
        return self.est_y - self.y


def simulate_physics(
    controller,
    reference,
    params: Params | None = None,
    fidelity: str = "high",
    t_end: float = 8.0,
    altitude0: float = 5.0,
    extras: dict[str, bool] | None = None,
    use_accel: bool = False,
    seed: int = 0,
    divergence_limit: float = 1e3,
) -> PhysicsTrace:
    """Run ``controller`` on the physics plant for ``reference``."""
    from ..plant import make_plant

    if params is None:
        from ..params import default_params

        params = default_params(fidelity)
    if fidelity not in ("paper", "high"):
        raise ValueError("fidelity must be 'paper' or 'high'")

    plant = make_plant(fidelity, extras=extras, seed=seed)
    h = float(params.control.h)
    sub = max(1, int(round(h / float(plant.dt))))
    g = float(params.airframe.g)

    trim = plant.trim_hover()
    throttle = float(trim["pwm_us"])
    state = trim["state"].copy()
    state.t = 0.0
    state.xi = np.array([0.0, 0.0, -float(altitude0)])
    state.v = np.zeros(3)
    plant.reset(state=state)

    if use_accel:
        # Firmware-faithful 2-state-per-axis Kalman (accelerometer tilt correction).
        estimator = AttitudeEstimator(
            KalmanConfig(
                dt=h,
                accel_sign=-1.0,
                use_accel=True,
                mag_model="vector",
                mag_reference=MAG_EARTH_UT,
            ),
            initial_angle=[0.0, 0.0, 0.0],
        )
    else:
        # Body-rate quaternion propagation + 3-D magnetometer heading.  Needed for
        # combined roll/pitch/yaw motion (Euler-rate integration is a small-angle
        # approximation); see the module docstring.
        estimator = QuaternionAttitudeEstimator(
            dt=h, initial_euler=(0.0, 0.0, 0.0), mag_reference=MAG_EARTH_UT
        )
    altitude_estimator = AltitudeEstimator(AltitudeConfig(dt=h))
    altitude_estimator.reset(float(altitude0), 0.0)

    controller.reset()
    order = int(getattr(controller, "order", 2))
    n_state = 2 * order
    n_ticks = int(np.floor(t_end / h + 1e-9)) + 1
    rec = {
        "t": np.zeros(n_ticks),
        "y_ref": np.zeros((n_ticks, 4)),
        "y": np.zeros((n_ticks, 4)),
        "rate_ref": np.zeros((n_ticks, 4)),
        "rate": np.zeros((n_ticks, 4)),
        "u": np.zeros((n_ticks, 4)),
        "u_actual": np.zeros((n_ticks, 4)),
        "pwm": np.zeros((n_ticks, 4)),
        "est_y": np.zeros((n_ticks, 4)),
        "est_rate": np.zeros((n_ticks, 4)),
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

        ref = reference.sample(t)
        y_ref = ref.y_ref
        rate_ref = ref.rate_ref
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
        rec["pwm"][k] = pwm
        rec["est_y"][k] = y_meas
        rec["est_rate"][k] = rate_meas
        theta = np.asarray(controller.theta, dtype=float)
        if theta.shape == (4, n_state):
            rec["theta_hat"][k] = theta

        for _ in range(sub):
            plant.step(pwm)

        if not np.all(np.isfinite(rec["y"][k])) or not np.all(np.isfinite(rec["rate"][k])):
            diverged = True
            rec = {key: val[: k + 1] for key, val in rec.items()}
            break
        if abs(float(rec["y"][k, 3])) > divergence_limit or np.max(np.abs(euler)) > np.pi:
            diverged = True
            rec = {key: val[: k + 1] for key, val in rec.items()}
            break

    return PhysicsTrace(
        t=rec["t"],
        y_ref=rec["y_ref"],
        y=rec["y"],
        rate_ref=rec["rate_ref"],
        rate=rec["rate"],
        u=rec["u"],
        u_actual=rec["u_actual"],
        pwm=rec["pwm"],
        est_y=rec["est_y"],
        est_rate=rec["est_rate"],
        theta_hat=rec["theta_hat"],
        order=order,
        safe_mode=bool(getattr(controller, "safe_mode", False)),
        diverged=diverged,
        metadata={
            "engine": "physics",
            "fidelity": fidelity,
            "h": h,
            "dt": float(plant.dt),
            "throttle_pwm": throttle,
            "trim": {key: (float(val) if np.isscalar(val) else val) for key, val in trim.items() if key != "state"},
            "extras": dict(plant.extras),
            "use_accel": bool(use_accel),
            "theta_true": np.asarray(
                [
                    [np.nan] * n_state
                ]
                * 4
            ),
        },
    )


def realised_commands(pwm: np.ndarray, throttle: float) -> np.ndarray:
    """Inverse of the adapter's mixer sign map (for tests/diagnostics)."""
    return recover_u_actual(pwm, throttle) * ALTITUDE_SIGN


__all__ = [
    "MAG_EARTH_UT",
    "ALTITUDE_SIGN",
    "PhysicsTrace",
    "simulate_physics",
    "realised_commands",
]
