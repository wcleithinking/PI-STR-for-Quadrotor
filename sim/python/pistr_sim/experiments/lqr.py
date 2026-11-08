"""LQR baseline designed from the *measured* plant small-signal channel.

For each axis the plant is probed at its hover trim with a small differential
step.  The rate step response is ``alpha*u*(t - (1-exp(-beta t))/beta)``, i.e. a
double integrator (angle <- rate) behind a first-order motor lag:

    d(angle)/dt = rate
    d(rate)/dt  = acceleration
    d(acceleration)/dt = -beta*acceleration + beta*alpha*u

Using only ``[angle, rate]`` as the LQR state would omit the motor lag (and the
u -> acceleration integrator) and produce a badly damped loop; the 3-state model
above is what is discretised (ZOH) and fed to
``scipy.linalg.solve_discrete_are``.  The angular acceleration is observed with a
one-step model prediction corrected by the measured rate difference.  The
control law is ``u = -K (x - x_ref)``, ``x_ref = [y_ref, rate_ref, 0]``.

The design uses the *nominal* plant for both fidelities; the fault scenarios are
deliberately unmodelled (that is the point of the robustness matrix).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.linalg import solve_discrete_are
from scipy.optimize import curve_fit

from ..control import ControllerBase, U_MIX
from ..params import default_params
from ..plant import make_plant

ALTITUDE_SIGN = np.array([1.0, 1.0, 1.0, -1.0])
U_MAX = np.array([80.0, 80.0, 100.0, 500.0])


def _probe_axis(plant, axis: int, u0: float = 20.0, t_end: float = 0.35):
    """Step-probe one axis; return ``(alpha, beta)`` in SI (per µs)."""
    trim = plant.trim_hover()
    state = trim["state"].copy()
    state.xi = np.array([0.0, 0.0, -5.0])  # away from the ground-effect region
    state.v = np.zeros(3)
    plant.reset(state=state)
    u_ctrl = np.zeros(4)
    u_ctrl[axis] = u0
    pwm = np.clip(trim["pwm_us"] + U_MIX @ (u_ctrl * ALTITUDE_SIGN), 1000.0, 2000.0)
    dt = float(plant.dt)
    n = int(round(t_end / dt))
    y = np.empty(n)
    for k in range(n):
        plant.step(pwm)
        if axis < 3:
            y[k] = plant.state.omega[axis]
        else:
            y[k] = -plant.state.v[2]
    t = np.arange(1, n + 1) * dt
    y = y / u0

    def model(tt, a, beta):
        return a * (tt - (1.0 - np.exp(-beta * tt)) / beta)

    popt, _ = curve_fit(model, t, y, p0=[0.45, 45.0], maxfev=20000)
    return float(popt[0]), float(popt[1])


def _design_channel(
    alpha: float, beta: float, h: float, q_pos: float, q_rate: float, r: float,
    q_acc: float = 2.0,
):
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -beta]])
    B = np.array([[0.0], [0.0], [beta * alpha]])
    C = np.eye(3)
    D = np.zeros((3, 1))
    Ad, Bd, _, _, _ = signal.cont2discrete((A, B, C, D), h, method="zoh")
    Ad = np.asarray(Ad).reshape(3, 3)
    Bd = np.asarray(Bd).reshape(3, 1)
    Q = np.diag([1.0 / q_pos**2, 1.0 / q_rate**2, 1.0 / q_acc**2])
    R = np.array([[1.0 / r**2]])
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad).reshape(3)
    return Ad, Bd, K


@dataclass
class ChannelLQR:
    alpha: float
    beta: float
    K: np.ndarray
    u_max: float
    h: float = 0.04
    accel_gain: float = 0.3

    def __post_init__(self) -> None:
        self.accel = 0.0
        self.rate_prev: float | None = None
        self.u_prev = 0.0

    def reset(self) -> None:
        self.accel = 0.0
        self.rate_prev = None
        self.u_prev = 0.0

    def command(self, y_ref: float, y: float, rate_ref: float, rate: float) -> float:
        # model prediction + measured-rate correction for the angular acceleration
        a_pred = self.accel + self.h * self.beta * (self.alpha * self.u_prev - self.accel)
        if self.rate_prev is None:
            self.accel = a_pred
        else:
            a_meas = (rate - self.rate_prev) / self.h
            self.accel = a_pred + self.accel_gain * (a_meas - a_pred)
        self.rate_prev = rate
        x_err = np.array([y - y_ref, rate - rate_ref, self.accel])
        u = -float(self.K @ x_err)
        u = float(np.clip(u, -self.u_max, self.u_max))
        self.u_prev = u
        return u


class LQRController(ControllerBase):
    """Per-channel discrete LQR on ``[angle, rate]`` / ``[altitude, climb rate]``."""

    name = "lqr"

    def __init__(self, channels: list[ChannelLQR]):
        if len(channels) != 4:
            raise ValueError("LQRController needs four channels")
        self.channels = channels
        self.n_channels = 4

    def reset(self) -> None:
        for channel in self.channels:
            channel.reset()

    def update(self, y_ref, y_meas, rate_ref=None, rate_meas=None) -> np.ndarray:
        y_ref = np.asarray(y_ref, dtype=float).reshape(4)
        y_meas = np.asarray(y_meas, dtype=float).reshape(4)
        rate_ref = np.zeros(4) if rate_ref is None else np.asarray(rate_ref, dtype=float).reshape(4)
        rate_meas = np.zeros(4) if rate_meas is None else np.asarray(rate_meas, dtype=float).reshape(4)
        u = np.empty(4)
        for j, channel in enumerate(self.channels):
            u[j] = channel.command(y_ref[j], y_meas[j], rate_ref[j], rate_meas[j])
        return u

    @property
    def theta(self) -> np.ndarray:
        return np.empty((0, 0))

    @property
    def safe_mode(self) -> bool:
        return False


def design_lqr(fidelity: str = "high", h: float | None = None) -> LQRController:
    """Probe the nominal plant per axis and return a 4-channel LQR controller."""
    params = default_params(fidelity)
    h = float(params.control.h if h is None else h)
    # Clean probe: no sensor noise or turbulence; the actuator/aero model is the
    # nominal high-fi plant, which is what the LQR is designed against.
    plant = make_plant(fidelity, extras={"sensors": False, "turbulence": False})
    trim = plant.trim_hover()
    alpha = np.empty(4)
    beta = np.empty(4)
    for axis in range(4):
        alpha[axis], beta[axis] = _probe_axis(plant, axis)
    channels = []
    for axis in range(4):
        if axis < 3:
            _, _, K = _design_channel(
                alpha[axis], beta[axis], h, q_pos=0.35, q_rate=2.0, r=60.0, q_acc=2.0
            )
        else:
            _, _, K = _design_channel(
                alpha[axis], beta[axis], h, q_pos=2.0, q_rate=1.0, r=300.0, q_acc=1.0
            )
        channels.append(
            ChannelLQR(
                alpha=float(alpha[axis]),
                beta=float(beta[axis]),
                K=K,
                u_max=U_MAX[axis],
                h=h,
            )
        )
    controller = LQRController(channels)
    controller.plant_trim = trim
    controller.alpha = alpha
    controller.beta = beta
    return controller


_LQR_CACHE: dict[str, LQRController] = {}


def cached_lqr(fidelity: str) -> LQRController:
    if fidelity not in _LQR_CACHE:
        _LQR_CACHE[fidelity] = design_lqr(fidelity)
    return _LQR_CACHE[fidelity]


__all__ = ["ChannelLQR", "LQRController", "design_lqr", "cached_lqr", "ALTITUDE_SIGN", "U_MAX"]
