"""PID baselines: the paper's robust PID (MATLAB ``c1``) and the firmware forms.

Two deliberately different structures are kept, because the paper compares the
STR against the *robust PID* and the firmware needs the *cascade PID* for its
``PID_v1``/``PID_v2`` builds:

* :class:`RobustPIDBank` — MATLAB ``c1``: one PID per axis acting on the angle
  error and the (reference - measurement) rate error, output directly the
  differential PWM command.  Gains from ``Load_SystemParameters.m``.
* :class:`CascadePIDBank` — firmware ``PID_v2``: angle loop -> rate command ->
  rate loop, with the outer output clamped (the firmware's ``constrain``).

The STR stack must not depend on :mod:`pistr_sim.plant`; this module only reads
the contract layer (``types``/``params``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Single channel PID
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PIDConfig:
    """One PID channel.

    ``derivative`` selects how the D term is formed:

    * ``"error"``       – ``kd * d(error)/dt`` (classical, MATLAB ``c1`` uses the
      rate error ``rate_meas - rate_ref`` directly, see :class:`RobustPIDBank`).
    * ``"measurement"`` – ``-kd * d(measurement)/dt`` (avoids derivative kick on a
      reference step; preferred for the firmware cascade).

    ``integrator_min/max`` implement clamping anti-windup: the integrator state is
    frozen at the limits, matching ``Rate_middlesire``'s ``constrain`` role.
    """

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    dt: float = 0.04
    out_min: float = -1.0
    out_max: float = 1.0
    derivative: str = "measurement"
    integrator_min: float = -np.inf
    integrator_max: float = np.inf
    d_filter_tau: float = 0.0

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.out_min >= self.out_max:
            raise ValueError("out_min must be smaller than out_max")
        if self.derivative not in ("error", "measurement", "none"):
            raise ValueError("derivative must be 'error', 'measurement' or 'none'")
        if self.d_filter_tau < 0.0:
            raise ValueError("d_filter_tau must be non-negative")


class PID:
    """Single-channel positional PID with integrator clamping."""

    def __init__(self, config: PIDConfig):
        self.cfg = config
        self.reset()

    def reset(self, integrator: float = 0.0, derivative: float = 0.0) -> None:
        self._integ = float(integrator)
        self._d_state = float(derivative)
        self._prev_meas: float | None = None
        self._prev_error: float | None = None
        self._last_u = 0.0

    @property
    def integrator(self) -> float:
        return self._integ

    def update(
        self,
        ref: float,
        meas: float,
        rate_ref: float | None = None,
        rate_meas: float | None = None,
    ) -> float:
        cfg = self.cfg
        ref = float(ref)
        meas = float(meas)
        if not (np.isfinite(ref) and np.isfinite(meas)):
            # Hold the previous command rather than injecting NaN into the mixer.
            return float(self._last_u)

        error = ref - meas

        # --- integral ----------------------------------------------------
        if cfg.ki != 0.0:
            trial = self._integ + cfg.ki * error * cfg.dt
            self._integ = float(np.clip(trial, cfg.integrator_min, cfg.integrator_max))

        # --- derivative --------------------------------------------------
        if cfg.derivative == "none" or cfg.kd == 0.0:
            d_term = 0.0
        elif cfg.derivative == "measurement":
            if rate_ref is not None and rate_meas is not None:
                raw = float(rate_ref) - float(rate_meas)
            else:
                raw = -(meas - self._prev_meas) / cfg.dt if self._prev_meas is not None else 0.0
            if cfg.d_filter_tau > 0.0:
                a = cfg.dt / (cfg.d_filter_tau + cfg.dt)
                self._d_state += a * (raw - self._d_state)
                raw = self._d_state
            d_term = cfg.kd * raw
        else:  # derivative on error
            raw = (error - self._prev_error) / cfg.dt if self._prev_error is not None else 0.0
            if cfg.d_filter_tau > 0.0:
                a = cfg.dt / (cfg.d_filter_tau + cfg.dt)
                self._d_state += a * (raw - self._d_state)
                raw = self._d_state
            d_term = cfg.kd * raw

        self._prev_meas = meas
        self._prev_error = error

        u = cfg.kp * error + self._integ + d_term
        u = float(np.clip(u, cfg.out_min, cfg.out_max))
        self._last_u = u
        return u


# ---------------------------------------------------------------------------
# Paper robust PID (MATLAB c1)
# ---------------------------------------------------------------------------
class RobustPIDBank:
    """4-channel robust PID used as the baseline in the paper's trajectory study.

    MATLAB form (``Sim_Trajectory_Tracking_Control.m``, ``state_c1 == 1``)::

        error      = y_meas - y_ref
        error_dot  = rate_meas - rate_ref
        sum_error += error * h
        u          = clamp(-kP*error - kI*sum_error - kD*error_dot, umin, umax)

    The sign convention (measurement - reference, negated) is kept verbatim so the
    numbers are directly comparable with the published baseline.
    """

    def __init__(
        self,
        kp: Sequence[float] = (10.0, 10.0, 20.0, 30.0),
        ki: Sequence[float] = (0.2, 0.2, 0.4, 0.6),
        kd: Sequence[float] = (2.0, 2.0, 4.0, 60.0),
        u_max: Sequence[float] = (80.0, 80.0, 100.0, 500.0),
        h: float = 0.04,
    ):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.u_max = np.asarray(u_max, dtype=float)
        n = self.kp.size
        if not (self.ki.size == self.kd.size == self.u_max.size == n):
            raise ValueError("kp/ki/kd/u_max must have the same length")
        self.h = float(h)
        self.reset()

    def reset(self) -> None:
        self._sum = np.zeros_like(self.kp)
        self._last = np.zeros_like(self.kp)

    @property
    def integrator(self) -> np.ndarray:
        return self._sum.copy()

    def update(
        self,
        y_ref: Sequence[float],
        y_meas: Sequence[float],
        rate_ref: Sequence[float] | None = None,
        rate_meas: Sequence[float] | None = None,
    ) -> np.ndarray:
        y_ref = np.asarray(y_ref, dtype=float).reshape(-1)
        y_meas = np.asarray(y_meas, dtype=float).reshape(-1)
        error = y_meas - y_ref
        if rate_ref is None or rate_meas is None:
            rate_ref = np.zeros_like(error)
            rate_meas = np.zeros_like(error)
        error_dot = np.asarray(rate_meas, dtype=float) - np.asarray(rate_ref, dtype=float)
        if not (np.all(np.isfinite(error)) and np.all(np.isfinite(error_dot))):
            return self._last.copy()
        self._sum = self._sum + error * self.h
        u = -self.kp * error - self.ki * self._sum - self.kd * error_dot
        u = np.clip(u, -self.u_max, self.u_max)
        self._last = u.copy()
        return u


# ---------------------------------------------------------------------------
# Firmware cascade PID (PID_v1 / PID_v2 from Controller_PID.ino)
# ---------------------------------------------------------------------------
#: Firmware constants, quoted from ``firmware/Quadrotor_Mode_II_Arduino_Due/Controller_PID.ino``.
FIRMWARE_PID_KP = 1.8
FIRMWARE_PID_KI = 0.02
FIRMWARE_PID_KD = 4.8


class CascadePIDBank:
    """Firmware ``PID_v2`` cascade: angle PID -> (clamped) rate command -> rate PID.

    ``PID_v1`` (single loop with rate feed-forward) is obtained with
    ``inner=False``; the D term then uses ``rate_des - rate_meas`` exactly as the
    firmware does.
    """

    def __init__(
        self,
        kp_outer: float = FIRMWARE_PID_KP,
        ki_outer: float = FIRMWARE_PID_KI,
        kd_outer: float = FIRMWARE_PID_KD,
        kp_inner: float = FIRMWARE_PID_KP,
        ki_inner: float = FIRMWARE_PID_KI,
        kd_inner: float = FIRMWARE_PID_KD,
        u_max: Sequence[float] = (80.0, 80.0, 100.0, 500.0),
        rate_max: Sequence[float] | float = 6.0,
        h: float = 0.04,
        inner: bool = True,
    ):
        self.u_max = np.asarray(u_max, dtype=float)
        n = self.u_max.size
        rm = np.asarray(rate_max, dtype=float)
        self.rate_max = np.full(n, float(rm)) if rm.size == 1 else rm.reshape(-1)
        if self.rate_max.size != n:
            raise ValueError("rate_max must be scalar or match u_max")
        self.h = float(h)
        self.inner = bool(inner)
        self.outer = [
            PID(
                PIDConfig(
                    kp=kp_outer,
                    ki=ki_outer,
                    kd=kd_outer,
                    dt=h,
                    out_min=-float(self.rate_max[i]),
                    out_max=float(self.rate_max[i]),
                    derivative="measurement",
                    integrator_min=-float(self.rate_max[i]),
                    integrator_max=float(self.rate_max[i]),
                )
            )
            for i in range(n)
        ]
        if inner:
            self.inner_pids = [
                PID(
                    PIDConfig(
                        kp=kp_inner,
                        ki=ki_inner,
                        kd=kd_inner,
                        dt=h,
                        out_min=-float(self.u_max[i]),
                        out_max=float(self.u_max[i]),
                        derivative="measurement",
                    )
                )
                for i in range(n)
            ]
        else:
            self.inner_pids = []

    def reset(self) -> None:
        for pid in self.outer:
            pid.reset()
        for pid in self.inner_pids:
            pid.reset()

    def update(
        self,
        angle_ref: Sequence[float],
        angle_meas: Sequence[float],
        rate_meas: Sequence[float],
        pilot_rate: Sequence[float] | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        angle_ref = np.asarray(angle_ref, dtype=float).reshape(-1)
        angle_meas = np.asarray(angle_meas, dtype=float).reshape(-1)
        rate_meas = np.asarray(rate_meas, dtype=float).reshape(-1)
        n = self.u_max.size
        out = np.zeros(n)
        for i in range(n):
            if self.inner:
                rate_des = self.outer[i].update(
                    angle_ref[i], angle_meas[i], rate_ref=None, rate_meas=rate_meas[i]
                )
                out[i] = self.inner_pids[i].update(
                    rate_des, rate_meas[i], rate_ref=rate_des, rate_meas=rate_meas[i]
                )
            else:
                # PID_v1: P and I on the angle error, D on the pilot rate error.
                rate_des = 0.0 if pilot_rate is None else float(pilot_rate[i])
                out[i] = self.outer[i].update(
                    angle_ref[i], angle_meas[i], rate_ref=rate_des, rate_meas=rate_meas[i]
                )
        return np.clip(out, -self.u_max, self.u_max)


def robust_pid_from_params(params) -> RobustPIDBank:
    """Paper baseline built from :class:`pistr_sim.params.ControlParams`."""
    c = params.control
    return RobustPIDBank(
        kp=c.pid_kp,
        ki=c.pid_ki,
        kd=c.pid_kd,
        u_max=c.u_max,
        h=c.h,
    )


__all__ = [
    "PIDConfig",
    "PID",
    "RobustPIDBank",
    "CascadePIDBank",
    "robust_pid_from_params",
    "FIRMWARE_PID_KP",
    "FIRMWARE_PID_KI",
    "FIRMWARE_PID_KD",
]
