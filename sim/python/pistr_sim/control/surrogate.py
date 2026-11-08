"""Plant-free ARX bench used to verify the controller stack.

``pistr_sim.plant`` (task-1) is developed in parallel and is *not* a dependency of
the control algorithms.  To exercise the controllers today — and to give the
golden-vector work a deterministic reference loop — this module provides the
minimal test plant implied by ``CONTROL_SPEC.md`` §3:

* ``order == 2`` (Mode II): the ARX maps the differential command ``u`` to the
  **rate** ``y``; the outer variable used by the cascade (angle / altitude) is its
  integral.
* ``order == 3`` (Mode I): the ARX maps ``u`` straight to the outer variable
  (angle / altitude), absorbing the motor lag.

The coefficients are taken from :func:`pistr_sim.control.str.analytic_theta0_mode_ii`
with ``percent = 1`` (the "true" identified model).  This is a *verification
bench*, not a flight-dynamics model: it has no aerodynamics, no sensor model and
no actuator lag.  Use :mod:`pistr_sim.plant` for fidelity studies.

The bench deliberately applies the hardware command saturation **after** the
controller returns, then feeds the realised value back through
``controller.ack()``.  Set ``u_hw`` below ``params.control.u_max`` to exercise
that path (it is the anti-windup mechanism of §5/§6.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .str import analytic_theta0_mode_i, analytic_theta0_mode_ii

RAD2DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# True channel model
# ---------------------------------------------------------------------------
def true_theta(
    params,
    order: int = 2,
    y_unit: str = "rad",
    percent: float = 1.0,
) -> np.ndarray:
    """Ground-truth ARX coefficients per channel, shape ``(4, 2*order)``.

    ``y_unit="rad"`` keeps the Python stack in SI (attitude ``tf_alpha`` is divided
    by 180/pi); ``"deg"`` reproduces the legacy firmware/MATLAB numbers.
    """
    control = params.control
    alpha = np.asarray(params.tf_alpha, dtype=float).copy()
    if y_unit == "rad":
        alpha[:3] /= RAD2DEG
    elif y_unit != "deg":
        raise ValueError("y_unit must be 'rad' or 'deg'")
    beta = np.asarray(params.tf_beta, dtype=float)
    if order == 2:
        return np.stack(
            [
                analytic_theta0_mode_ii(alpha[j], beta[j], control.h, percent)
                for j in range(4)
            ]
        )
    if order == 3:
        return np.stack(
            [
                analytic_theta0_mode_i(alpha[j], beta[j], control.h, percent)
                for j in range(4)
            ]
        )
    raise ValueError("order must be 2 or 3")


# ---------------------------------------------------------------------------
# The discrete channel
# ---------------------------------------------------------------------------
class ArxChannel:
    """One SISO ARX channel ``y(k) = -a y(k-1..n) + b u(k-1..n)``."""

    def __init__(self, theta: Sequence[float]):
        self.theta = np.asarray(theta, dtype=float).reshape(-1)
        n = self.theta.size // 2
        if self.theta.size != 2 * n:
            raise ValueError("theta must contain an even number of coefficients")
        self.order = n
        self.reset()

    def reset(self, y0: float = 0.0) -> None:
        self._y = [float(y0)] * self.order
        self._u = [0.0] * self.order

    @property
    def a(self) -> np.ndarray:
        return self.theta[: self.order]

    @property
    def b(self) -> np.ndarray:
        return self.theta[self.order :]

    def step(self, u: float) -> float:
        a, b = self.a, self.b
        y = -float(a @ np.asarray(self._y)) + float(b @ np.asarray(self._u))
        self._y.insert(0, y)
        del self._y[self.order:]
        self._u.insert(0, float(u))
        del self._u[self.order:]
        return y


# ---------------------------------------------------------------------------
# Closed-loop run
# ---------------------------------------------------------------------------
@dataclass
class ChannelTrace:
    """Time history of a bench run (row index = controller tick)."""

    t: np.ndarray
    y_ref: np.ndarray
    y: np.ndarray
    rate_ref: np.ndarray
    rate: np.ndarray
    u: np.ndarray
    u_actual: np.ndarray
    theta_hat: np.ndarray | None = None
    order: int = 2
    safe_mode: bool = False
    diverged: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return int(self.y.shape[1])


def simulate(
    controller,
    reference,
    params,
    theta_true_values: np.ndarray | None = None,
    order: int | None = None,
    t_end: float | None = None,
    h: float | None = None,
    u_hw: Sequence[float] | float | None = None,
    y0: Sequence[float] | float | None = None,
    rate0: Sequence[float] | float | None = None,
    divergence_limit: float = 1e6,
) -> ChannelTrace:
    """Run ``controller`` against the ARX bench for ``reference``.

    ``u_hw`` is the hardware differential-command limit.  It defaults to
    ``params.control.u_max`` (the same limit the controller uses, so no
    saturation difference); pass a smaller vector to exercise the ``u_actual``
    feedback path.
    """
    control = params.control
    h = float(control.h if h is None else h)
    t_end = float(params.sim.t_end if t_end is None else t_end)
    if order is None:
        order = getattr(controller, "order", None)
    order = int(control.order if order is None else order)

    theta = (
        true_theta(params, order=order)
        if theta_true_values is None
        else np.asarray(theta_true_values, dtype=float)
    )
    if theta.shape != (4, 2 * order):
        raise ValueError(f"theta_true_values must have shape {(4, 2 * order)}")

    u_limit = np.asarray(control.u_max, dtype=float) if u_hw is None else np.asarray(u_hw, dtype=float)
    if u_limit.size == 1:
        u_limit = np.full(4, float(u_limit))
    if u_limit.size != 4:
        raise ValueError("u_hw must be scalar or have four entries")

    y0v = np.zeros(4) if y0 is None else np.asarray(y0, dtype=float).reshape(4)
    rate0v = np.zeros(4) if rate0 is None else np.asarray(rate0, dtype=float).reshape(4)

    n_ticks = int(np.floor(t_end / h + 1e-9)) + 1
    channels = [ArxChannel(theta[j]) for j in range(4)]
    for j in range(4):
        channels[j].reset(rate0v[j] if order == 2 else y0v[j])

    controller.reset()
    n_state = 2 * order
    record = {
        "t": np.zeros(n_ticks),
        "y_ref": np.zeros((n_ticks, 4)),
        "y": np.zeros((n_ticks, 4)),
        "rate_ref": np.zeros((n_ticks, 4)),
        "rate": np.zeros((n_ticks, 4)),
        "u": np.zeros((n_ticks, 4)),
        "u_actual": np.zeros((n_ticks, 4)),
        "theta_hat": np.zeros((n_ticks, 4, n_state)),
    }

    y = y0v.copy()          # outer variable (angle / altitude)
    rate = rate0v.copy()    # outer derivative (rate / climb rate)
    diverged = False

    for k in range(n_ticks):
        t = k * h
        ref = reference.sample(t, xi_meas=None, v_meas=None)
        y_ref = ref.y_ref
        rate_ref = ref.rate_ref

        u = np.asarray(
            controller.update(y_ref, y, rate_ref, rate), dtype=float
        ).reshape(4)
        u_actual = np.clip(u, -u_limit, u_limit)
        controller.ack(u_actual)

        record["t"][k] = t
        record["y_ref"][k] = y_ref
        record["y"][k] = y
        record["rate_ref"][k] = rate_ref
        record["rate"][k] = rate
        record["u"][k] = u
        record["u_actual"][k] = u_actual
        theta_hat = np.asarray(controller.theta, dtype=float)
        if theta_hat.shape == (4, n_state):
            record["theta_hat"][k] = theta_hat

        # ---- advance the channel ---------------------------------------
        for j in range(4):
            yj = channels[j].step(float(u_actual[j]))
            if order == 2:
                rate[j] = yj
                y[j] = y[j] + h * yj
            else:
                y[j] = yj
                rate[j] = (yj - record["y"][k, j]) / h if k > 0 else 0.0

        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(rate)):
            diverged = True
            record = {key: val[: k + 1] for key, val in record.items()}
            break
        if np.max(np.abs(y)) > divergence_limit:
            diverged = True
            record = {key: val[: k + 1] for key, val in record.items()}
            break

    trace = ChannelTrace(
        t=record["t"],
        y_ref=record["y_ref"],
        y=record["y"],
        rate_ref=record["rate_ref"],
        rate=record["rate"],
        u=record["u"],
        u_actual=record["u_actual"],
        theta_hat=record["theta_hat"],
        order=order,
        safe_mode=bool(controller.safe_mode),
        diverged=diverged,
        metadata={
            "theta_true": theta,
            "h": h,
            "order": order,
            "u_hw": u_limit,
        },
    )
    return trace


def open_loop_identification(
    theta_true_values: Sequence[float],
    u: Sequence[float],
    y0: float = 0.0,
) -> np.ndarray:
    """Drive one ARX channel open loop with a given input; return ``y``."""
    channel = ArxChannel(theta_true_values)
    channel.reset(y0)
    return np.array([channel.step(float(ui)) for ui in u], dtype=float)


__all__ = [
    "RAD2DEG",
    "true_theta",
    "ArxChannel",
    "ChannelTrace",
    "simulate",
    "open_loop_identification",
]
