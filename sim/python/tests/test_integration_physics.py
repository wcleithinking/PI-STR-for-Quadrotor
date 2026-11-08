"""Physics-plant integration and channel-gain reconciliation (task-6).

Part 1 reconciles the paper's control-effectiveness coefficient ``tf_alpha`` with
the plant: a first-principles linearisation, the paper-fidelity plant and the
``§3`` analytical ARX channel are compared at the hover point with a ±1 µs
differential PWM probe (step slope and the ω = 5 rad/s amplitude).

The reference numbers below were measured on this repository's plant and are
saved (with the full per-extra decomposition) in
``results/control_stack/physics_gain.csv``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal
from scipy.optimize import curve_fit

from pistr_sim.control import (
    STRConfig,
    U_MIX,
    U_MIX_INV,
    make_controller,
    recover_u_actual,
    simulate_physics,
)
from pistr_sim.control.str import analytic_theta0_mode_ii
from pistr_sim.estimator import (
    AltitudeConfig,
    AltitudeEstimator,
    AttitudeEstimator,
    KalmanConfig,
    QuaternionAttitudeEstimator,
    _quat_from_euler,
)
from pistr_sim.metrics import rmse, step_metrics
from pistr_sim.params import default_params
from pistr_sim.plant import DEFAULT_EXTRAS, make_plant
from pistr_sim.reference import HoverReference, StepReference
from pistr_sim.types import quat_to_rotmat

PARAMS = default_params("paper")
H = float(PARAMS.control.h)
DT = float(PARAMS.sim.dt)
RAD2DEG = 180.0 / np.pi

# --- reference channel gains (rad/s per µs), measured on this plant ----------
ARX_A = 0.40911          # §3 analytical channel, a_ss = (b0+b1)/(h(1-a2))
ARX_H5 = 0.08125         # §3 analytical channel, |H(e^{j 5 h})|
PHYS_A = 0.47807         # first-principles linearisation at hover, a_ss
PHYS_H5 = 0.09509        # first-principles linearisation, |H(j5)|


# ---------------------------------------------------------------------------
# Small-signal probes
# ---------------------------------------------------------------------------
def _trim(fidelity, extras=None):
    plant = make_plant(fidelity, extras=extras)
    trim = plant.trim_hover()
    return plant, trim


def _step_gain(fidelity, extras=None, u0=20.0, t_end=0.35):
    """Return ``(a_ss, beta, |H(e^{j5h})|)`` from a ±differential step."""
    plant, trim = _trim(fidelity, extras)
    pwm0 = trim["pwm_us"]
    plant.reset(state=trim["state"])
    pwm = pwm0 + U_MIX @ np.array([u0, 0.0, 0.0, 0.0])
    n = int(round(t_end / DT))
    rate = np.empty(n)
    for k in range(n):
        plant.step(pwm)
        rate[k] = plant.state.omega[0]
    t = np.arange(1, n + 1) * DT
    y = rate / u0

    def model(tt, a, beta):
        return a * (tt - (1.0 - np.exp(-beta * tt)) / beta)

    popt, _ = curve_fit(model, t, y, p0=[0.45, 47.0], maxfev=20000)
    a_ss, beta = float(popt[0]), float(popt[1])
    num, den, _ = signal.cont2discrete(([a_ss * beta], [1.0, beta, 0.0]), H, method="zoh")
    z = np.exp(1j * 5.0 * H)
    h5 = abs(np.polyval(np.ravel(num), z) / np.polyval(np.ravel(den), z))
    return a_ss, beta, float(h5)


def _dither_gain(fidelity, extras=None, amplitude=1.5, omega=5.0, t_end=6.0, skip=1.5):
    """|H(ω)| measured with a cosine dither (no angle accumulation)."""
    plant, trim = _trim(fidelity, extras)
    pwm0 = trim["pwm_us"]
    plant.reset(state=trim["state"])
    n = int(round(t_end / H))
    sub = max(1, int(round(H / DT)))
    y = np.empty(n)
    u = np.empty(n)
    for k in range(n):
        t = k * H
        u[k] = amplitude * np.cos(omega * t)
        pwm = pwm0 + U_MIX @ np.array([u[k], 0.0, 0.0, 0.0])
        for _ in range(sub):
            plant.step(pwm)
        y[k] = plant.state.omega[0]
    t = np.arange(n) * H
    mask = t > skip
    x = np.column_stack([np.cos(omega * t[mask]), np.sin(omega * t[mask])])
    cy, *_ = np.linalg.lstsq(x, y[mask], rcond=None)
    cu, *_ = np.linalg.lstsq(x, u[mask], rcond=None)
    return float(np.hypot(*cy) / np.hypot(*cu))


def _analytic():
    theta = analytic_theta0_mode_ii(
        PARAMS.tf_alpha[0] / RAD2DEG, PARAMS.tf_beta[0], H, 1.0
    )
    a1, a2, b0, b1 = theta
    return theta, float((b0 + b1) / (H * (1.0 - a2)))


# ---------------------------------------------------------------------------
# §1 / §3: the voltage-slope linearisation and the net α ratio
# ---------------------------------------------------------------------------
def test_voltage_slope_linearisation_factors():
    motor, rot, air = PARAMS.motor, PARAMS.rotor, PARAMS.airframe
    p0 = PARAMS.pwm_hover_percent
    true_dv = motor.u_max / (2.0 * np.sqrt(p0)) / 1000.0  # V per µs at hover
    literal = motor.u_max / 250.0                        # the formula's literal term
    effective = motor.u_max / 500.0                      # ... but it is /250/2

    assert literal / true_dv == pytest.approx(3.423, abs=0.01)
    assert effective / true_dv == pytest.approx(1.712, abs=0.01)

    # The formula's torque term uses 2*w (one rotor); the roll differential pair
    # contributes 4*w, i.e. a factor 0.5.  Net: alpha_formula / alpha_physical.
    net = 0.5 * (effective / true_dv)
    alpha_formula = PARAMS.tf_alpha[0] / RAD2DEG / PARAMS.tf_beta[0]
    assert alpha_formula == pytest.approx(net * PHYS_A, rel=2e-3)

    # first-principles steady angular acceleration (rad/s^2 per µs)
    w = PARAMS.w_hover
    dw_dv = (motor.km / motor.resistance) / (
        2 * rot.kq * w + motor.km * motor.ke / motor.resistance
    )
    gamma = air.d_arm * rot.kt * 4.0 * w * dw_dv * true_dv
    alpha_phys = gamma / air.inertia[0, 0]
    assert alpha_phys == pytest.approx(PHYS_A, rel=1e-3)
    assert alpha_formula / alpha_phys == pytest.approx(0.856, abs=0.005)


# ---------------------------------------------------------------------------
# 1 / 2: paper plant, high-fidelity all-off and per-extra decomposition
# ---------------------------------------------------------------------------
def test_paper_plant_matches_first_principles():
    a_ss, beta, h5 = _step_gain("paper")
    assert a_ss == pytest.approx(PHYS_A, rel=0.05)
    assert beta == pytest.approx(PARAMS.tf_beta[0], rel=0.10)
    assert h5 == pytest.approx(PHYS_H5, rel=0.05)

    # The paper's control effectiveness is ~15 % *below* the physical channel:
    # the analytic ARX underestimates the plant gain (not the other way round).
    assert ARX_A / PHYS_A == pytest.approx(0.856, abs=0.005)
    assert a_ss / ARX_A == pytest.approx(1.17, abs=0.05)


def test_high_all_off_reproduces_the_analytical_channel():
    """With every extra switched off the high-fidelity plant is the §3 model."""
    extras = {name: False for name in DEFAULT_EXTRAS}
    a_ss, _, h5 = _step_gain("high", extras)
    assert a_ss == pytest.approx(ARX_A, rel=0.03)
    assert h5 == pytest.approx(ARX_H5, rel=0.04)
    assert _dither_gain("high", extras) == pytest.approx(h5, rel=0.05)


def test_gain_decomposition_per_extra():
    base = {name: False for name in DEFAULT_EXTRAS}
    a_off, _, h_off = _step_gain("high", base)

    # At hover these have no first-order effect: in-plane speed, rotor rates and
    # the inflow recalibration are all zero/compensated at the trim point.
    for name in ("inflow", "h_force", "body_drag", "gyro_moment"):
        extras = dict(base, **{name: True})
        a_on, _, _ = _step_gain("high", extras)
        assert a_on == pytest.approx(a_off, rel=0.01), name

    # Battery sag raises the effective voltage (OCV 12.6 V instead of Umax 11.1 V).
    a_bat, _, _ = _step_gain("high", dict(base, battery_sag=True))
    assert a_bat / a_off == pytest.approx(1.08, abs=0.03)

    # ESC lag + 1 µs PWM quantisation: the steady ramp slope is nearly unchanged
    # but the ω = 5 rad/s amplitude is attenuated (quantisation is a nonlinearity
    # at the 1.5 µs dither amplitude).
    a_esc, beta_esc, _ = _step_gain("high", dict(base, esc_lag=True))
    assert 0.95 < a_esc / a_off < 1.06
    assert beta_esc < 40.0  # extra lag slows the transient

    # Everything on: interaction of sag + inflow + lag raises the channel gain.
    a_all, _, _ = _step_gain(
        "high", {**{k: True for k in DEFAULT_EXTRAS}, "sensors": False, "turbulence": False}
    )
    assert 1.20 < a_all / ARX_A < 1.50
    assert a_all > a_off


# ---------------------------------------------------------------------------
# 1 (closed loop): estimator conventions against the plant
# ---------------------------------------------------------------------------
def test_plant_accelerometer_is_the_thrust_axis_in_free_flight():
    """The high-fidelity accelerometer reports the body thrust axis, (0,0,-g).

    In equilibrium the only non-gravitational force is thrust along body -z, so
    the accelerometer cannot resolve the absolute tilt in free flight (unlike a
    vehicle resting on the ground, where the support force is world-vertical).
    This is why the physics bench observes attitude with gyro + magnetometer.
    """
    plant = make_plant("high", extras={k: False for k in DEFAULT_EXTRAS})
    trim = plant.trim_hover()
    for attitude in ((0.0, 0.0, 0.0), (0.2, -0.1, 0.5), (-0.15, 0.25, -1.0)):
        state = trim["state"].copy()
        state.q = _quat_from_euler(*attitude)
        plant.reset(state=state)
        sample = plant.sense()
        assert sample.accel == pytest.approx([0.0, 0.0, -PARAMS.airframe.g], abs=1e-9)


def test_quaternion_observer_tracks_the_plant_attitude():
    """The gyro + 3-D magnetometer observer is consistent with the plant."""
    plant = make_plant("high", extras={k: False for k in DEFAULT_EXTRAS})
    trim = plant.trim_hover()
    truth = np.array([0.2, -0.1, 0.5])
    state = trim["state"].copy()
    state.q = _quat_from_euler(*truth)
    plant.reset(state=state)
    sample = plant.sense()

    # Start with a 0.3 rad heading error: only the magnetometer can fix it, so
    # this exercises the 3-D field model (45, 0, 20) uT end to end.
    est = QuaternionAttitudeEstimator(
        dt=H, initial_euler=(truth[0], truth[1], truth[2] - 0.3), mag_reference=(45.0, 0.0, 20.0)
    )
    for _ in range(400):
        estimate = est.update(sample, dt=H)
    assert np.max(np.abs(estimate.angle - truth)) < 5e-3

    # The firmware Kalman with accel_sign=-1 would interpret the thrust-axis
    # reading as "level": it is deliberately not used in free flight.
    kalman = AttitudeEstimator(
        KalmanConfig(dt=H, accel_sign=-1.0, mag_model="vector", mag_reference=(45.0, 0.0, 20.0)),
        initial_angle=truth,
    )
    for _ in range(400):
        estimate_kf = kalman.update(sample, dt=H)
    assert abs(estimate_kf.angle[0] - truth[0]) > 0.05


def test_altitude_estimator_converges():
    est = AltitudeEstimator(AltitudeConfig(dt=H, q_alt=1e-4, q_rate=1e-2, r_baro=2.5e-3))
    est.reset(0.0, 0.0)
    rng = np.random.default_rng(0)
    for k in range(300):
        alt, vz = est.update(5.0 + rng.normal(0.0, 0.05), a_up=0.0, dt=H)
    assert alt == pytest.approx(5.0, abs=0.1)
    assert vz == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# 2: physics closed loop (hover and roll step)
# ---------------------------------------------------------------------------
def _physics_controller(params, percent=0.95):
    configs = [
        STRConfig.mode_ii(params, ch, percent=percent, p_max=100.0, forgetting=0.998)
        for ch in range(4)
    ]
    return make_controller(
        "pi_str", params, outer_form="paper", str_configs=configs, rate_max=2.0
    )


def test_physics_hover_closed_loop():
    params = default_params("high")
    controller = _physics_controller(params)
    trace = simulate_physics(
        controller, HoverReference(5.0), params, fidelity="high", t_end=6.0, altitude0=5.0
    )
    assert not trace.diverged
    assert not trace.safe_mode
    assert np.all(np.isfinite(trace.y))
    tail = trace.t > 1.0
    for j in range(3):
        assert rmse(trace.t[tail], trace.y[tail, j], 0.0) < np.deg2rad(3.0)
    assert rmse(trace.t[tail], trace.y[tail, 3], 5.0) < 0.3
    assert np.max(np.abs(trace.estimator_error[tail, :3])) < np.deg2rad(3.0)


def test_physics_roll_step_overshoot():
    params = default_params("high")
    controller = _physics_controller(params)
    trace = simulate_physics(
        controller,
        StepReference(amplitude=0.1, t_step=1.5, altitude=5.0),
        params,
        fidelity="high",
        t_end=8.0,
        altitude0=5.0,
    )
    assert not trace.diverged
    assert not trace.safe_mode
    metrics = step_metrics(trace.t, trace.y[:, 0], trace.y_ref[:, 0], t_step=1.5)
    assert metrics.overshoot < 5.0, metrics
    assert abs(metrics.steady_state_error) < 0.04  # gyro random-walk drift (no aiding)
    assert metrics.rmse < 0.04


def test_physics_u_actual_matches_the_realised_pwm():
    params = default_params("high")
    controller = _physics_controller(params)
    trace = simulate_physics(
        controller,
        StepReference(amplitude=0.1, t_step=0.8, altitude=5.0),
        params,
        fidelity="high",
        t_end=3.0,
        altitude0=5.0,
    )
    throttle = trace.metadata["throttle_pwm"]
    recovered = (
        (trace.pwm - throttle) @ U_MIX_INV.T * np.array([1.0, 1.0, 1.0, -1.0])
    )
    assert np.allclose(recovered, trace.u_actual, atol=1e-9)
    assert np.isfinite(trace.pwm).all()
    assert np.all(trace.pwm >= 1000.0) and np.all(trace.pwm <= 2000.0)
