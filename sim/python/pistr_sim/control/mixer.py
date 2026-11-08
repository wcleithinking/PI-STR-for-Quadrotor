"""Firmware-compatible mixer between differential commands and PWM microseconds.

Ported from ``sim/matlab/reduced_overshoot/Load_SystemParameters.m`` (``u_mix``)
and ``firmware/Quadrotor_Mode_II_Arduino_Due`` (``QuadP`` frame)::

    PWM         = throttle + U_MIX @ u_design
    u_actual    = U_MIX^-1 @ (PWM - throttle)

The *realised* differential command is what the STR feeds back into its regressor
(``CONTROL_SPEC.md`` §5/§6.4): after PWM clipping it generally differs from the
design value, and that difference is exactly the anti-windup signal.

Column order is ``[roll, pitch, yaw, altitude]``; row order is motor
``[0, 1, 2, 3]`` as wired in the firmware.  The matrix is singular-free
(``det = -8``) so the inverse is exact up to round-off.
"""

from __future__ import annotations

import numpy as np

from ..types import PWM_MAX, PWM_MIN

#: ``u_mix`` from the MATLAB source (motor rows x axis columns).
U_MIX: np.ndarray = np.array(
    [
        [0.0, 1.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0, -1.0],
        [0.0, -1.0, 1.0, -1.0],
        [1.0, 0.0, -1.0, -1.0],
    ]
)

U_MIX_INV: np.ndarray = np.linalg.inv(U_MIX)


def throttle_us(params) -> float:
    """Hover throttle in microseconds (``PWM_min + (PWM_max-PWM_min)*PWM_N``)."""
    return float(params.pwm_hover_us)


def mix(u_design, throttle: float) -> np.ndarray:
    """Map ``[roll, pitch, yaw, alt]`` differential commands to PWM microseconds."""
    u = np.asarray(u_design, dtype=float).reshape(4)
    return np.asarray(throttle + U_MIX @ u, dtype=float)


def clamp_pwm(pwm) -> np.ndarray:
    return np.clip(np.asarray(pwm, dtype=float), PWM_MIN, PWM_MAX)


def recover_u_actual(pwm, throttle: float) -> np.ndarray:
    """Invert the mixer after PWM clipping: the physically realised command."""
    pwm = np.asarray(pwm, dtype=float).reshape(4)
    return np.asarray(U_MIX_INV @ (pwm - float(throttle)), dtype=float)


def mix_and_recover(u_design, throttle: float) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: ``(pwm_clipped, u_actual)`` for one control tick."""
    pwm = clamp_pwm(mix(u_design, throttle))
    return pwm, recover_u_actual(pwm, throttle)


__all__ = [
    "U_MIX",
    "U_MIX_INV",
    "throttle_us",
    "mix",
    "clamp_pwm",
    "recover_u_actual",
    "mix_and_recover",
]
