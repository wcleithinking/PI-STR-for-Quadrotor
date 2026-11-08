"""Cascade + estimator + reference/mixer tests (CONTROL_SPEC.md §2, §7).

Everything runs on the plant-free ARX bench, so these tests do not depend on
``pistr_sim.plant``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pistr_sim.control import (
    STRConfig,
    U_MIX,
    U_MIX_INV,
    make_controller,
    mix_and_recover,
    simulate,
    true_theta,
)
from pistr_sim.estimator import (
    AttitudeEstimator,
    KalmanConfig,
    synthetic_sensor_sample,
)
from pistr_sim.metrics import overshoot_percent, rmse, settling_time
from pistr_sim.params import default_params
from pistr_sim.reference import (
    ESO,
    AttitudePlanner,
    AttitudePlanner3D,
    HoverReference,
    SinusoidReference,
    StepReference,
    TrajectoryReference3D,
)

PARAMS = default_params("paper")
H = PARAMS.control.h


def _matlab_str_configs(order: int = 2):
    factory = STRConfig.mode_ii if order == 2 else STRConfig.mode_i
    return [factory(PARAMS, ch, percent=0.9, p_max=100.0, forgetting=0.998) for ch in range(4)]


# ---------------------------------------------------------------------------
# Attitude estimator
# ---------------------------------------------------------------------------
def test_estimator_converges_from_offset_with_noise():
    rng = np.random.default_rng(11)
    est = AttitudeEstimator(KalmanConfig(dt=0.01), initial_angle=[0.0, 0.0, 0.0])
    true_angle = np.array([0.15, -0.08, 0.6])
    for k in range(400):
        sample = synthetic_sensor_sample(
            k * 0.01,
            true_angle,
            np.zeros(3),
            gyro_noise=np.deg2rad(0.1),
            accel_noise=0.05,
            mag_noise=0.01,
            rng=rng,
        )
        estimate = est.update(sample)
    assert np.all(np.isfinite(estimate.angle))
    error = estimate.angle - true_angle
    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    assert np.max(np.abs(error)) < 0.01, error


def test_estimator_estimates_gyro_bias():
    rng = np.random.default_rng(12)
    est = AttitudeEstimator(KalmanConfig(dt=0.01))
    bias = np.array([0.02, -0.03, 0.01])
    for k in range(2000):
        sample = synthetic_sensor_sample(
            k * 0.01,
            np.zeros(3),
            np.zeros(3),
            bias=bias,
            gyro_noise=np.deg2rad(0.1),
            accel_noise=0.05,
            mag_noise=0.01,
            rng=rng,
        )
        est.update(sample)
    assert np.max(np.abs(est.gyro_bias - bias)) < 5e-3, est.gyro_bias


def test_estimator_yaw_wraps_across_pi():
    est = AttitudeEstimator(KalmanConfig(dt=0.01))
    rng = np.random.default_rng(13)
    yaw = 3.05
    for k in range(300):
        yaw += 0.001  # cross +pi
        true_angle = np.array([0.0, 0.0, yaw])
        est.update(
            synthetic_sensor_sample(
                k * 0.01, true_angle, np.zeros(3), accel_noise=0.01, mag_noise=0.005, rng=rng
            )
        )
    err = np.arctan2(
        np.sin(est.yaw.angle - yaw), np.cos(est.yaw.angle - yaw)
    )
    assert abs(err) < 0.05, (est.yaw.angle, yaw)


def test_estimator_handles_bad_sample():
    est = AttitudeEstimator(KalmanConfig(dt=0.04))
    sample = synthetic_sensor_sample(0.0, [0.1, 0.0, 0.0], [0.0, 0.0, 0.0])
    est.update(sample)
    bad = synthetic_sensor_sample(0.04, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    bad.gyro = np.array([np.nan, 0.0, 0.0])
    estimate = est.update(bad)
    assert np.all(np.isfinite(estimate.angle))
    assert np.all(np.isfinite(estimate.rate))


# ---------------------------------------------------------------------------
# Closed-loop: PID, PI-STR, Mode I
# ---------------------------------------------------------------------------
def test_pid_step_response_is_bounded():
    controller = make_controller("pid", PARAMS)
    trace = simulate(controller, StepReference(amplitude=0.1, t_step=1.0), PARAMS, t_end=12.0)
    assert not trace.diverged
    assert np.all(np.isfinite(trace.y))
    assert rmse(trace.t, trace.y[:, 0], trace.y_ref[:, 0]) < 0.06
    assert np.max(np.abs(trace.y[:, 0])) < 0.3


def test_pi_str_step_response():
    controller = make_controller(
        "pi_str", PARAMS, outer_form="paper", str_configs=_matlab_str_configs(2)
    )
    trace = simulate(controller, StepReference(amplitude=0.1, t_step=1.0), PARAMS, t_end=12.0)
    assert not trace.diverged
    y = trace.y[:, 0]
    ref = trace.y_ref[:, 0]

    overshoot = overshoot_percent(trace.t, y, ref)
    settling = settling_time(trace.t, y, ref, tol=0.02, t_from=1.0)
    assert overshoot < 5.0, overshoot
    assert settling < 3.0, settling
    assert abs(y[-1] - 0.1) < 0.01, y[-1]
    assert rmse(trace.t, y, ref) < 0.03
    assert trace.theta_hat.shape == (trace.t.size, 4, 4)


def test_pi_str_hover_station_keeping():
    ref = HoverReference(altitude=5.0)
    controller = make_controller(
        "pi_str", PARAMS, outer_form="paper", str_configs=_matlab_str_configs(2)
    )
    trace = simulate(
        controller,
        ref,
        PARAMS,
        t_end=10.0,
        y0=[0.02, -0.02, 0.05, 5.0],
    )
    assert not trace.diverged
    tail = trace.t >= 3.0
    for j in range(3):
        assert rmse(trace.t[tail], trace.y[tail, j], trace.y_ref[tail, j]) < 0.02
    assert rmse(trace.t[tail], trace.y[tail, 3], trace.y_ref[tail, 3]) < 0.1


def test_pi_str_sinusoid_tracking():
    ref = SinusoidReference()
    controller = make_controller(
        "pi_str", PARAMS, outer_form="paper", str_configs=_matlab_str_configs(2)
    )
    trace = simulate(controller, ref, PARAMS, t_end=15.0)
    assert not trace.diverged
    # The reference model (wn = 5 rad/s) has a known phase lag at 2 rad/s, so a
    # few degrees of RMS error are expected and reported, not asserted away.
    assert np.rad2deg(rmse(trace.t, trace.y[:, 0], trace.y_ref[:, 0])) < 5.0
    assert np.rad2deg(rmse(trace.t, trace.y[:, 2], trace.y_ref[:, 2])) < 8.0


def test_mode1_single_loop_design_verification():
    """Mode I with the analytic model (no deliberate mismatch) tracks a step."""
    configs = [
        STRConfig.mode_i(PARAMS, ch, percent=1.0, p_max=100.0, forgetting=0.998)
        for ch in range(4)
    ]
    controller = make_controller("str", PARAMS, str_configs=configs)
    trace = simulate(controller, StepReference(amplitude=0.1, t_step=1.0), PARAMS, t_end=15.0)
    assert not trace.diverged
    y = trace.y[:, 0]
    assert abs(y[-1] - 0.1) < 0.01, y[-1]
    assert y.max() < 0.15, y.max()
    assert trace.order == 3


def test_u_actual_feedback_under_saturation():
    """The hardware limit is below u_max, so ack() must feed the realised value."""
    controller = make_controller(
        "pi_str", PARAMS, outer_form="paper", str_configs=_matlab_str_configs(2)
    )
    trace = simulate(
        controller,
        StepReference(amplitude=0.1, t_step=1.0),
        PARAMS,
        t_end=6.0,
        u_hw=0.2,  # far below u_max = 80
    )
    assert not trace.diverged
    assert np.max(np.abs(trace.u_actual)) <= 0.2 + 1e-12
    saturated = np.abs(trace.u[:, 0]) > 0.2 + 1e-9
    assert np.any(saturated)
    assert np.allclose(trace.u_actual[saturated, 0], np.sign(trace.u[saturated, 0]) * 0.2)
    assert np.all(np.isfinite(controller.theta))


# ---------------------------------------------------------------------------
# Mixer
# ---------------------------------------------------------------------------
def test_mixer_round_trip_and_saturation():
    assert np.isfinite(np.linalg.det(U_MIX))
    u = np.array([12.0, -8.0, 5.0, 30.0])
    pwm, u_actual = mix_and_recover(u, throttle=1183.0)
    assert np.allclose(pwm, 1183.0 + U_MIX @ u)
    assert np.allclose(u_actual, u, atol=1e-10)
    assert np.allclose(U_MIX_INV @ U_MIX, np.eye(4))

    # Altitude command that drives two motors above 2000 us is clipped; the
    # recovered command differs from the design value (anti-windup signal).
    pwm2, u_actual2 = mix_and_recover(np.array([0.0, 0.0, 0.0, 900.0]), throttle=1183.0)
    assert np.all(pwm2 <= 2000.0 + 1e-9) and np.all(pwm2 >= 1000.0 - 1e-9)
    assert not np.allclose(u_actual2, np.array([0.0, 0.0, 0.0, 900.0]))


# ---------------------------------------------------------------------------
# References, planner and ESO
# ---------------------------------------------------------------------------
def test_trajectory_reference_matches_the_paper_formulas():
    ref = TrajectoryReference3D()
    for t in (0.0, 7.5, 22.0):
        s = ref.sample(t)
        assert np.isclose(s.xi[0], 4.0 * t)
        assert np.isclose(s.xi[1], 60.0 * np.sin(2 * np.pi * t / 30.0 + np.pi))
        assert np.isclose(s.xi[2], 10.0 * np.sin(2 * np.pi * t / 30.0 + np.pi / 2) - 20.0)
        assert np.isclose(s.v_xi[0], 4.0)
        assert np.isclose(
            s.a_xi[1], -(2 * np.pi / 30.0) ** 2 * 60.0 * np.sin(2 * np.pi * t / 30.0 + np.pi)
        )
        amplitude = np.deg2rad(60.0) / np.sqrt(2.0)
        assert np.isclose(
            s.yaw, amplitude * np.sin(2 * np.pi * t / 20.0) + amplitude * np.cos(2 * np.pi * t / 20.0)
        )


def test_sinusoid_rates_are_the_analytic_derivatives():
    ref = SinusoidReference()
    t = 1.234
    dt = 1e-6
    lo, hi = ref.sample(t - dt), ref.sample(t + dt)
    for j, field in enumerate(("roll", "pitch", "yaw", "altitude")):
        derivative = (getattr(hi, field) - getattr(lo, field)) / (2 * dt)
        rate = (hi.rate[j]) if j < 3 else hi.climb_rate
        assert np.isclose(derivative, rate, rtol=1e-6, atol=1e-6)


def test_eso_tracks_a_constant_and_a_ramp():
    eso = ESO(pole=-2.0, h=0.02)
    for k in range(500):
        angle, rate = eso.step(0.3)
    assert abs(angle - 0.3) < 1e-3
    assert abs(rate) < 1e-3

    eso = ESO(pole=-2.0, h=0.02)
    for k in range(1000):
        target = 0.01 * k * 0.02
        angle, rate = eso.step(target)
    assert abs(angle - target) < 0.02
    assert abs(rate - 0.01) < 0.01


def test_attitude_planner_is_bounded_and_consistent():
    planner = AttitudePlanner(mass=PARAMS.airframe.mass)
    roll, pitch = planner.update(
        np.array([10.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([10.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
    )
    assert abs(roll) < 1e-9 and abs(pitch) < 1e-9

    # A large position error must not produce a NaN: the arcsin argument is clipped
    # and the tilt limited.
    roll, pitch = planner.update(
        np.zeros(3), np.zeros(3), np.zeros(3), np.array([1e6, -1e6, 0.0]), np.zeros(3)
    )
    assert np.isfinite(roll) and np.isfinite(pitch)
    assert abs(roll) <= planner.max_tilt + 1e-12
    assert abs(pitch) <= planner.max_tilt + 1e-12


def test_attitude_planner3d_produces_a_usable_reference():
    planner = AttitudePlanner3D(mass=PARAMS.airframe.mass, h=0.02)
    ref = planner.reference
    for k in range(500):
        t = 0.02 * k
        base = ref.sample(t)
        s = planner.sample(t, xi_meas=base.xi, v_meas=base.v_xi)
        assert np.all(np.isfinite([s.roll, s.pitch, s.yaw, s.altitude]))
        assert s.t == pytest.approx(t)
    # The ESO is deliberately bypassed for the first two seconds (§9.4).
    planner.reset()
    early = planner.sample(1.0, xi_meas=ref.sample(1.0).xi, v_meas=ref.sample(1.0).v_xi)
    assert early.roll_rate == 0.0 and early.pitch_rate == 0.0
