"""Tests for pistr_sim.metrics (step-response and tracking-quality metrics)."""

from __future__ import annotations

import numpy as np
import pytest

from pistr_sim.metrics import (
    StepMetrics,
    control_rms,
    diverged,
    max_attitude_error,
    overshoot_percent,
    rmse,
    settling_time,
    steady_state,
    step_metrics,
)


def _step_record(t_step=1.0, t_end=8.0, h=0.005):
    """Piecewise step response with a known 20 % overshoot.

    Rise 0 -> 0.12 rad/s over [1, 2] s, decay 0.12 -> 0.10 over [2, 3] s, then
    hold exactly 0.10.  The 2 % settling band is last left just before t = 3 s.
    """
    t = np.arange(0.0, t_end, h)
    y = np.zeros_like(t)
    rising = (t >= t_step) & (t < 2.0)
    y[rising] = 0.12 * (t[rising] - t_step)
    decaying = (t >= 2.0) & (t < 3.0)
    y[decaying] = 0.12 - 0.02 * (t[decaying] - 2.0)
    y[t >= 3.0] = 0.1
    ref = np.where(t >= t_step, 0.1, 0.0)
    return t, y, ref


def test_rmse_exact_and_scalar_reference():
    y = np.array([0.0, 1.0, 2.0])
    assert rmse([0, 1, 2], y, 0.0) == pytest.approx(np.sqrt(5.0 / 3.0))
    assert rmse([0, 1, 2], y, [0.0, 1.0, 2.0]) == 0.0
    assert rmse([0, 1, 2], y, 0.0, t_from=1.0) == pytest.approx(np.sqrt(5.0 / 2.0))


def test_rmse_reports_nan_instead_of_raising():
    assert np.isnan(rmse([0, 1], [0.0, np.nan], 0.0))
    assert np.isnan(rmse([0, 1], [np.inf, 0.0], 0.0))


def test_overshoot_percent():
    t, y, ref = _step_record()
    assert overshoot_percent(t, y, ref) == pytest.approx(20.0, abs=1e-6)
    # Monotonic rise: no "overshoot" in the direction of the step.
    monotonic = np.where(t >= 1.0, 0.1, 0.0)
    assert overshoot_percent(t, monotonic, ref) == pytest.approx(0.0, abs=1e-9)


def test_settling_time_2_percent_band():
    t, y, ref = _step_record()
    ts = settling_time(t, y, ref, tol=0.02, t_from=1.0)
    assert 0.0 <= ts <= 5.0
    # After the reported settling time the response never leaves the band again.
    band = 0.02 * abs(ref[-1])
    outside = (t >= 1.0) & (np.abs(y - ref[-1]) > band)
    assert np.any(outside)
    assert not np.any(outside & (t > 1.0 + ts + 1e-9))
    assert np.isclose(t[outside][-1], 1.0 + ts, atol=0.01)


def test_settling_time_handles_a_persistent_oscillation():
    t = np.arange(0.0, 10.0, 0.01)
    ref = np.where(t >= 1.0, 0.1, 0.0)
    # +/-20 % oscillation around the reference: the 2 % band is never entered.
    y = np.where(t >= 1.0, 0.1 + 0.02 * np.sin(20.0 * t), 0.0)
    ts = settling_time(t, y, ref, tol=0.02, t_from=1.0)
    assert np.isfinite(ts)
    assert ts == pytest.approx(t[np.abs(y - 0.1) > 0.002][-1] - 1.0, abs=1e-6)


def test_overshoot_and_settling_need_a_step():
    t = np.arange(0.0, 5.0, 0.01)
    y = np.sin(t)
    assert np.isnan(overshoot_percent(t, y, 0.0))
    assert np.isnan(settling_time(t, y, 0.0))


def test_control_rms():
    u = np.array([3.0, 4.0])
    assert control_rms(u) == pytest.approx(np.sqrt(12.5))
    t = np.array([0.0, 1.0, 2.0])
    assert control_rms([3.0, 4.0, 0.0], t, t_from=1.0) == pytest.approx(np.sqrt(8.0))
    assert np.isnan(control_rms([np.nan, 1.0]))


def test_max_attitude_error_units():
    angle = np.array([0.0, 0.1, -0.2])
    ref = np.zeros(3)
    assert max_attitude_error(angle, ref) == pytest.approx(np.rad2deg(0.2))
    assert max_attitude_error(angle, ref, degrees=False) == pytest.approx(0.2)
    t = np.array([0.0, 1.0, 2.0])
    assert max_attitude_error(angle, ref, t, t_from=1.5) == pytest.approx(np.rad2deg(0.2))


def test_diverged_detection():
    assert not diverged([0.0, 1.0, 2.0])
    assert diverged([0.0, 1e7])
    assert diverged([0.0, np.nan])
    assert diverged([0.0, np.inf])


def test_steady_state_uses_the_tail():
    y = np.concatenate([np.zeros(90), np.full(10, 0.1)])
    t = np.arange(y.size, dtype=float)
    assert steady_state(y, t) == pytest.approx(0.1)


def test_step_metrics_dataclass_and_nan_path():
    t, y, ref = _step_record()
    m = step_metrics(t, y, ref, t_step=1.0)
    assert isinstance(m, StepMetrics)
    assert m.ref == pytest.approx(0.1)
    assert m.final == pytest.approx(0.1, abs=1e-9)
    assert m.overshoot == pytest.approx(20.0, abs=1e-6)
    assert m.steady_state_error == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(m.settling_time) and m.rmse > 0.0
    assert set(m.as_dict()) == {
        "ref",
        "final",
        "overshoot",
        "settling_time",
        "rmse",
        "steady_state_error",
        "peak",
        "t_peak",
    }

    bad = step_metrics(t, np.full_like(y, np.nan), ref, t_step=1.0)
    assert np.isnan(bad.overshoot) and np.isnan(bad.rmse)
