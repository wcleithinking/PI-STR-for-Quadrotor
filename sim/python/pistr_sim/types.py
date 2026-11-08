"""Shared data contracts and numeric constants.

This module is the interface boundary between the plant (physics) and the
controllers (algorithms). Nothing here may import the plant or the controllers.

See ``docs/PLANT_MODEL.md`` §9 and ``docs/CONTROL_SPEC.md`` for the normative text.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Actuator domain. The firmware, the MATLAB code and every plant model in this
# repository speak PWM microseconds, not normalised thrust.
# ---------------------------------------------------------------------------
PWM_MIN: float = 1000.0
PWM_MAX: float = 2000.0
PWM_SPAN: float = PWM_MAX - PWM_MIN

#: Body frame used by the mixer. ``QuadP`` matches the Arduino firmware
#: (``#define QuadP``); ``QuadX`` is provided for completeness.
FRAME_QUADP = "QuadP"
FRAME_QUADX = "QuadX"


def pwm_to_percent(pwm_us: np.ndarray | float) -> np.ndarray | float:
    """Map PWM microseconds to the [0, 1] throttle used by the MATLAB model."""
    return (np.asarray(pwm_us) - PWM_MIN) / PWM_MIN


def percent_to_pwm(percent: np.ndarray | float) -> np.ndarray | float:
    """Inverse of :func:`pwm_to_percent`."""
    return np.asarray(percent) * PWM_MIN + PWM_MIN


def clamp_pwm(pwm_us: np.ndarray | float) -> np.ndarray | float:
    return np.clip(pwm_us, PWM_MIN, PWM_MAX)


# ---------------------------------------------------------------------------
# Vehicle state
# ---------------------------------------------------------------------------
def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Inertial->body rotation matrix, identical element order to the MATLAB code."""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q
    return np.array(
        [
            [q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
            [2 * (q1 * q2 - q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 + q0 * q1)],
            [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0**2 - q1**2 - q2**2 + q3**2],
        ]
    )


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """``[roll, pitch, yaw]`` in rad, using the same extraction as the MATLAB truth model."""
    c = quat_to_rotmat(q)
    roll = np.arctan2(-c[2, 1], c[2, 2])
    pitch = np.arcsin(np.clip(c[2, 0], -1.0, 1.0))
    yaw = np.arctan2(-c[1, 0], c[0, 0])
    return np.array([roll, pitch, yaw])


def quat_normalise(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """``dq/dt = 0.5 * q (x) [0, omega]`` with the MATLAB matrix layout."""
    q0, q1, q2, q3 = np.asarray(q, dtype=float)
    w = np.asarray(omega, dtype=float)
    return 0.5 * np.array(
        [
            [-q1, -q2, -q3],
            [q0, -q3, q2],
            [q3, q0, -q1],
            [-q2, q1, q0],
        ]
    ) @ w


@dataclass
class VehicleState:
    """Full plant state. Controllers must never read this; they only see ``SensorSample``."""

    t: float = 0.0
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    xi: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotor_omega: np.ndarray = field(default_factory=lambda: np.zeros(4))
    rotor_current: np.ndarray = field(default_factory=lambda: np.zeros(4))
    esc_duty: np.ndarray = field(default_factory=lambda: np.zeros(4))
    pwm_us: np.ndarray = field(default_factory=lambda: np.full(4, PWM_MIN))
    soc: float = 1.0
    v_terminal: float = 12.6

    def copy(self) -> "VehicleState":
        return replace(
            self,
            q=self.q.copy(),
            omega=self.omega.copy(),
            v=self.v.copy(),
            xi=self.xi.copy(),
            rotor_omega=self.rotor_omega.copy(),
            rotor_current=self.rotor_current.copy(),
            esc_duty=self.esc_duty.copy(),
            pwm_us=self.pwm_us.copy(),
        )

    @property
    def euler(self) -> np.ndarray:
        return quat_to_euler(self.q)

    @property
    def c_i2b(self) -> np.ndarray:
        return quat_to_rotmat(self.q)

    @property
    def c_b2i(self) -> np.ndarray:
        return self.c_i2b.T


@dataclass
class SensorSample:
    """What a flight controller is allowed to see. Already delayed and corrupted."""

    t: float
    gyro: np.ndarray          # rad/s, body frame, bias + noise included
    accel: np.ndarray         # m/s^2, body specific force (proper acceleration)
    mag: np.ndarray           # arbitrary units/µT, body frame
    baro_alt: float           # m, positive up


@dataclass
class Estimate:
    """Fused attitude estimate handed to the control law."""

    t: float
    angle: np.ndarray         # rad, [roll, pitch, yaw]
    rate: np.ndarray          # rad/s, bias-corrected body rates


@runtime_checkable
class Plant(Protocol):
    """Physics interface. ``paper`` and ``high`` fidelity both implement it."""

    dt: float

    def reset(self, state: VehicleState | None = None) -> VehicleState: ...

    def step(self, pwm_us: np.ndarray) -> VehicleState: ...

    def sense(self) -> SensorSample: ...

    def set_wind(self, steady: np.ndarray, turbulence: bool = True) -> None: ...


__all__ = [
    "PWM_MIN",
    "PWM_MAX",
    "PWM_SPAN",
    "FRAME_QUADP",
    "FRAME_QUADX",
    "VehicleState",
    "SensorSample",
    "Estimate",
    "Plant",
    "pwm_to_percent",
    "percent_to_pwm",
    "clamp_pwm",
    "quat_to_rotmat",
    "quat_to_euler",
    "quat_normalise",
    "quat_derivative",
]
