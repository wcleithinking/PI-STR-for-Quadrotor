"""Tracking-quality metrics shared by every experiment.

Definitions (all return ``nan`` when the input cannot be evaluated, and never
raise on NaN/Inf data — a divergent run must be *reported*, not crash the matrix):

* **RMSE** - root mean square of ``y - ref`` over an optional time window.
* **Overshoot** - for a step, ``100 * (peak - final) / |step|`` in the direction
  of the step; negative overshoot (undershoot) is clipped to zero.
* **Settling time (2 %)** - time after the step at which the response last leaves
  the ``±tol * |step|`` band around the final value.
* **Control RMS** - root mean square of the commanded signal.
* **Max attitude error** - largest absolute attitude error (deg) over the window.

The step metrics use the final value estimated from the tail of the record, so
they also work when the reference is not exactly reached (steady-state offset).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np


def _arrays(t: Sequence[float], y: Sequence[float], ref: Sequence[float] | float):
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1) if not np.isscalar(ref) else np.full_like(y, float(ref))
    if not (t.size == y.size == ref.size) or t.size == 0:
        raise ValueError("t, y and ref must have the same non-zero length")
    return t, y, ref


def _window(t: np.ndarray, y: np.ndarray, ref: np.ndarray, t_from: float | None):
    if t_from is None:
        return t, y, ref
    mask = t >= float(t_from) - 1e-12
    return t[mask], y[mask], ref[mask]


def rmse(
    t: Sequence[float],
    y: Sequence[float],
    ref: Sequence[float] | float,
    t_from: float | None = None,
) -> float:
    """Root mean square error between ``y`` and ``ref``."""
    t, y, ref = _arrays(t, y, ref)
    t, y, ref = _window(t, y, ref, t_from)
    if t.size == 0:
        return float("nan")
    error = y - ref
    if not np.all(np.isfinite(error)):
        return float("nan")
    return float(np.sqrt(np.mean(error**2)))


def control_rms(u: Sequence[float], t: Sequence[float] | None = None, t_from: float | None = None) -> float:
    """Root mean square of a control signal (optionally windowed)."""
    u = np.asarray(u, dtype=float).reshape(-1)
    if t is not None:
        t = np.asarray(t, dtype=float).reshape(-1)
        if t.size != u.size:
            raise ValueError("t and u must have the same length")
        if t_from is not None:
            u = u[t >= float(t_from) - 1e-12]
    if u.size == 0 or not np.all(np.isfinite(u)):
        return float("nan")
    return float(np.sqrt(np.mean(u**2)))


def steady_state(y: Sequence[float], t: Sequence[float], fraction: float = 0.1) -> float:
    """Mean of the last ``fraction`` of the record (used as the step final value)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    n = max(1, int(round(y.size * float(np.clip(fraction, 0.01, 0.5)))))
    tail = y[-n:]
    if not np.all(np.isfinite(tail)):
        return float("nan")
    return float(np.mean(tail))


def overshoot_percent(
    t: Sequence[float],
    y: Sequence[float],
    ref: Sequence[float] | float,
    t_from: float | None = None,
) -> float:
    """Peak overshoot of a step response, in percent of the commanded step.

    The step size is taken from the *full* reference record (``ref[-1]-ref[0]``);
    ``t_from`` only restricts where the tail average / peak are measured.
    """
    t, y, ref = _arrays(t, y, ref)
    step = float(ref[-1]) - float(ref[0])
    t, y, ref = _window(t, y, ref, t_from)
    if t.size < 2 or not np.all(np.isfinite(y)) or abs(step) < 1e-12:
        return float("nan")
    final = steady_state(y, t)
    if not np.isfinite(final):
        return float("nan")
    peak = float(np.max(y)) if step > 0 else float(np.min(y))
    return float(max(0.0, (peak - final) / abs(step) * 100.0))


def settling_time(
    t: Sequence[float],
    y: Sequence[float],
    ref: Sequence[float] | float,
    tol: float = 0.02,
    t_from: float | None = None,
) -> float:
    """Last time the response leaves the ``±tol*|step|`` band, relative to ``t_from``.

    The step size is taken from the full reference record (see
    :func:`overshoot_percent`), so passing ``t_from=t_step`` is well defined.
    """
    t, y, ref = _arrays(t, y, ref)
    step = float(ref[-1]) - float(ref[0])
    t, y, ref = _window(t, y, ref, t_from)
    if t.size < 2 or not np.all(np.isfinite(y)) or abs(step) < 1e-12:
        return float("nan")
    final = steady_state(y, t)
    if not np.isfinite(final):
        return float("nan")
    band = abs(float(tol)) * abs(step)
    outside = np.abs(y - final) > band
    if not np.any(outside):
        return 0.0
    idx = int(np.where(outside)[0][-1])
    t0 = float(t[0]) if t_from is None else float(t_from)
    return float(max(0.0, t[idx] - t0))


def max_attitude_error(
    angle: Sequence[float],
    ref: Sequence[float] | float,
    t: Sequence[float] | None = None,
    t_from: float | None = None,
    degrees: bool = True,
) -> float:
    """Largest absolute attitude error over the window (degrees by default)."""
    angle = np.asarray(angle, dtype=float).reshape(-1)
    ref = np.asarray(ref, dtype=float).reshape(-1) if not np.isscalar(ref) else np.full_like(angle, float(ref))
    if angle.size != ref.size:
        raise ValueError("angle and ref must have the same length")
    if t is not None:
        t = np.asarray(t, dtype=float).reshape(-1)
        if t.size != angle.size:
            raise ValueError("t and angle must have the same length")
        if t_from is not None:
            mask = t >= float(t_from) - 1e-12
            angle, ref = angle[mask], ref[mask]
    error = angle - ref
    if error.size == 0 or not np.all(np.isfinite(error)):
        return float("nan")
    value = float(np.max(np.abs(error)))
    return float(np.rad2deg(value)) if degrees else value


def diverged(y: Sequence[float], limit: float = 1e6) -> bool:
    """True when the signal became non-finite or exceeded ``limit`` in magnitude."""
    y = np.asarray(y, dtype=float).reshape(-1)
    return bool((not np.all(np.isfinite(y))) or np.any(np.abs(y) > float(limit)))


@dataclass
class StepMetrics:
    """Step-response summary for one channel."""

    ref: float
    final: float
    overshoot: float          # percent
    settling_time: float      # s, 2 % band
    rmse: float
    steady_state_error: float
    peak: float
    t_peak: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def step_metrics(
    t: Sequence[float],
    y: Sequence[float],
    ref: Sequence[float] | float,
    t_step: float,
    tol: float = 0.02,
    settle_after: float | None = None,
) -> StepMetrics:
    """Complete step-response metrics for the window starting at ``t_step``."""
    t, y, ref = _arrays(t, y, ref)
    mask = t >= float(t_step) - 1e-12
    t_w, y_w, ref_w = t[mask], y[mask], ref[mask]
    if t_w.size == 0 or not np.all(np.isfinite(y_w)) or not np.all(np.isfinite(ref)):
        return StepMetrics(*(float("nan"),) * 8)
    final = steady_state(y_w, t_w)
    step = float(ref[-1]) - float(ref[0])
    if step == 0.0:
        step = 1.0
    peak_idx = int(np.argmax(y_w) if step >= 0 else np.argmin(y_w))
    after = t_step if settle_after is None else float(settle_after)
    return StepMetrics(
        ref=float(ref[-1]),
        final=float(final),
        overshoot=overshoot_percent(t, y, ref),
        settling_time=settling_time(t, y, ref, tol=tol, t_from=after),
        rmse=rmse(t, y, ref, t_from=t_step),
        steady_state_error=float(final - ref[-1]),
        peak=float(y_w[peak_idx]),
        t_peak=float(t_w[peak_idx]),
    )


__all__ = [
    "rmse",
    "control_rms",
    "steady_state",
    "overshoot_percent",
    "settling_time",
    "max_attitude_error",
    "diverged",
    "StepMetrics",
    "step_metrics",
]
