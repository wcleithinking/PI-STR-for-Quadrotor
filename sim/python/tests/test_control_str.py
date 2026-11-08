"""Unit tests for the STR core (CONTROL_SPEC.md §3-§6).

The tests are deliberately plant-free: synthetic ARX channels stand in for the
physics, so they can run before ``pistr_sim.plant`` exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from pistr_sim.control import (
    A1_MAX,
    A1_MIN,
    A2_MAX,
    A2_MIN,
    B_MAX,
    B_MIN,
    BSUM_MIN,
    FAIL_LIMIT,
    GUARD_DET,
    GUARD_U,
    Gains,
    STR,
    STRConfig,
    am_coeffs,
    analytic_theta0_mode_i,
    analytic_theta0_mode_ii,
    ao_coeffs,
    diophantine_system,
    gaussian_solve,
    project_theta,
    rls_update,
)
from pistr_sim.control.surrogate import ArxChannel, true_theta
from pistr_sim.params import default_params

# A well-conditioned second-order channel (the paper's roll channel in SI units).
PARAMS = default_params("paper")
THETA2_TRUE = true_theta(PARAMS, 2)[0].copy()


def _polynomial_identity(theta, order, am, ao):
    """Return ``conv(A,R)+conv(B,S)`` for the solved gains (z^-1 coefficients)."""
    if order == 2:
        a = np.array([1.0, theta[0], theta[1]])
        b = np.array([0.0, theta[2], theta[3]])
        s = solve_gains(theta, order, am, ao)
        r = np.array([1.0, s["r1"]])
        ss = np.array([s["s0"], s["s1"]])
    else:
        a = np.array([1.0, *theta[:3]])
        b = np.array([0.0, *theta[3:]])
        s = solve_gains(theta, order, am, ao)
        r = np.array([1.0, s["r1"], s["r2"]])
        ss = np.array([s["s0"], s["s1"], s["s2"]])
    lhs = np.polyadd(np.convolve(a, r), np.convolve(b, ss))
    rhs = np.convolve(np.r_[1.0, am], np.r_[1.0, ao])
    return lhs, rhs


def solve_gains(theta, order, am, ao):
    A, rhs = diophantine_system(theta, order, am, ao)
    x, _ = gaussian_solve(A, rhs)
    if order == 2:
        return {"r1": x[0], "s0": x[1], "s1": x[2]}
    return {"r1": x[0], "r2": x[1], "s0": x[2], "s1": x[3], "s2": x[4]}


# ---------------------------------------------------------------------------
# Gaussian elimination
# ---------------------------------------------------------------------------
def test_gaussian_solve_matches_numpy_3x3():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(3, 3))
        b = rng.normal(size=3)
        x, det = gaussian_solve(A, b)
        assert np.allclose(x, np.linalg.solve(A, b), atol=1e-12)
        assert np.isclose(det, np.linalg.det(A), rtol=1e-10, atol=1e-12)


def test_gaussian_solve_matches_numpy_5x5():
    rng = np.random.default_rng(1)
    for _ in range(20):
        A = rng.normal(size=(5, 5))
        b = rng.normal(size=5)
        x, det = gaussian_solve(A, b)
        assert np.allclose(x, np.linalg.solve(A, b), atol=1e-11)
        assert np.isclose(det, np.linalg.det(A), rtol=1e-9, atol=1e-10)


def test_gaussian_solve_needs_partial_pivoting():
    # zero pivot on the diagonal: naive elimination divides by zero.
    A = np.array([[0.0, 2.0], [1.0, 3.0]])
    b = np.array([4.0, 5.0])
    x, det = gaussian_solve(A, b)
    assert np.allclose(x, np.linalg.solve(A, b))
    assert np.isclose(det, -2.0)


def test_gaussian_solve_singular_raises():
    from pistr_sim.control.str import NumericalSingularity

    with pytest.raises(NumericalSingularity):
        gaussian_solve(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Diophantine (§5)
# ---------------------------------------------------------------------------
def test_diophantine_order2_satisfies_identity():
    am = am_coeffs(2, 0.7, 5.0, 0.04)
    ao = ao_coeffs(2, 0.0)
    for theta in (THETA2_TRUE, 0.75 * THETA2_TRUE, np.array([-1.2, 0.3, 0.02, 0.01])):
        lhs, rhs = _polynomial_identity(theta, 2, am, ao)
        # A*R + B*S must reproduce Am*Ao exactly (to round-off).
        assert np.allclose(lhs, rhs, atol=1e-10)


def test_diophantine_order2_with_integral_pole():
    am = am_coeffs(2, 0.7, 5.0, 0.04)
    ao = ao_coeffs(2, -0.5)  # Ao = 1 - 0.5 z^-1
    theta = 0.9 * THETA2_TRUE
    lhs, rhs = _polynomial_identity(theta, 2, am, ao)
    assert np.allclose(lhs, rhs, atol=1e-10)


def test_diophantine_order3_satisfies_identity():
    am = am_coeffs(3, 0.707, 7.9, 0.04, cc=50.0)
    ao = ao_coeffs(3, 0.0)
    theta = true_theta(PARAMS, 3)[0]
    for scale in (1.0, 0.98):
        lhs, rhs = _polynomial_identity(scale * theta, 3, am, ao)
        assert np.allclose(lhs, rhs, atol=1e-8)


def test_diophantine_s0_s1_indexing_is_not_the_legacy_bug():
    """The legacy firmware read s0=x[2], s1=x[3] for a 3-vector (§9.2)."""
    am = am_coeffs(2, 0.7, 5.0, 0.04)
    A, rhs = diophantine_system(THETA2_TRUE, 2, am, ao_coeffs(2, 0.0))
    x, _ = gaussian_solve(A, rhs)
    g = STR(STRConfig(order=2, h=0.04, theta0=tuple(THETA2_TRUE))).gains
    assert np.isclose(g.r1, x[0]) and np.isclose(g.s0, x[1]) and np.isclose(g.s1, x[2])


# ---------------------------------------------------------------------------
# Projection / guards (§6)
# ---------------------------------------------------------------------------
def test_projection_order2_is_idempotent_and_stable():
    theta = np.array([10.0, 10.0, 50.0, -50.0])
    p1 = project_theta(theta, 2)
    p2 = project_theta(p1, 2)
    assert np.array_equal(p1, p2)
    a1, a2 = p1[0], p1[1]
    assert A1_MIN <= a1 <= A1_MAX
    assert A2_MIN <= a2 <= A2_MAX
    assert a2 >= -1.0 and a1 + a2 >= -1.0 and a1 - a2 <= 1.0
    assert B_MIN <= p1[3] <= B_MAX
    # The b-sum fix is applied after the clamp (§6.1 order), so b0 may end at
    # B_MAX + BSUM_MIN by at most the guard value.
    assert B_MIN <= p1[2] <= B_MAX + BSUM_MIN + 1e-12
    # ``b0 = BSUM_MIN - b1`` is exact in real arithmetic but can round a few ulp
    # below the threshold in double; the map is still idempotent (asserted above).
    assert abs(p1[2] + p1[3]) >= BSUM_MIN - 1e-12


def test_projection_preserves_truth_on_the_integrator_boundary():
    """The true rate channel sits on a1+a2 = -1; projection must not move it."""
    p = project_theta(THETA2_TRUE, 2)
    assert np.allclose(p, THETA2_TRUE, atol=1e-12)


def test_projection_does_not_move_the_mode1_analytic_truth():
    """The 3rd-order relative b-sum bound must leave the analytic model untouched."""
    from pistr_sim.control.surrogate import true_theta

    theta3 = true_theta(PARAMS, 3)  # percent = 1.0
    for channel in range(4):
        cfg = STRConfig.mode_i(PARAMS, channel, percent=1.0)
        threshold = cfg.bsum_threshold
        assert threshold == pytest.approx(0.01 * abs(theta3[channel, 3:].sum()), rel=1e-12)
        projected = project_theta(theta3[channel], 3, threshold)
        assert np.array_equal(projected, theta3[channel]), (
            "projection moved the analytic Mode-I truth on channel "
            f"{channel}: {projected} vs {theta3[channel]}"
        )
        # and the resulting gains are valid (no guard freeze at t = 0)
        reg = STR(cfg)
        assert reg.gains.det > GUARD_DET
        assert reg.gains.t0 != 0.0


def test_projection_order3_relative_bsum_restores_a_collapsed_estimate():
    from pistr_sim.control.surrogate import true_theta

    theta3 = true_theta(PARAMS, order=3)
    cfg = STRConfig.mode_i(PARAMS, 0, percent=0.9)
    collapsed = theta3[0].copy()
    collapsed[3:] *= 1e-4  # b vector collapses: |sum b| << threshold
    projected = project_theta(collapsed, 3, cfg.bsum_threshold)
    assert abs(projected[3:].sum()) >= cfg.bsum_threshold - 1e-12
    assert abs(projected[3:].sum()) < abs(theta3[0, 3:].sum())


def test_projection_order3_stabilises_and_is_idempotent():
    p1 = project_theta([4.0, 2.0, 1.0, 5.0, 5.0, 5.0], 3)
    p2 = project_theta(p1, 3)
    assert np.array_equal(p1, p2)
    roots = np.roots(np.r_[1.0, p1[:3]])
    assert np.max(np.abs(roots)) <= 1.0 + 1e-9


def test_projection_enforces_bsum():
    th = np.array([-1.0, 0.0, 5e-4, 5e-4])
    p = project_theta(th, 2)
    assert abs(p[2] + p[3]) >= BSUM_MIN


# ---------------------------------------------------------------------------
# RLS (§4)
# ---------------------------------------------------------------------------
def test_rls_matches_reference_formulation():
    rng = np.random.default_rng(2)
    theta = np.array([-1.0, 0.2, 0.01, 0.005])
    P = 100.0 * np.eye(4)
    lam, pmax = 0.998, 100.0
    for _ in range(30):
        phi = rng.normal(size=4)
        y = float(rng.normal())
        theta_ref = theta + (P @ phi) / (lam + phi @ P @ phi) * (y - theta @ phi)
        K = (P @ phi) / (lam + phi @ P @ phi)
        P_ref = (np.eye(4) - np.outer(K, phi)) @ P / lam
        P_ref = 0.5 * (P_ref + P_ref.T)
        result = rls_update(theta, P, phi, y, lam, pmax)
        assert np.allclose(result.theta, theta_ref, atol=1e-12)
        assert np.allclose(result.P, P_ref, atol=1e-12)
        assert result.symmetric
        theta, P = result.theta, result.P


def test_rls_converges_to_true_arx_parameters():
    """Broadband excitation: the identifier must recover the truth within 10 %."""
    rng = np.random.default_rng(3)
    theta_true = THETA2_TRUE
    cfg = STRConfig(
        order=2, h=0.04, theta0=tuple(0.75 * theta_true), forgetting=0.99, p_max=100.0
    )
    reg = STR(cfg)
    channel = ArxChannel(theta_true)
    channel.reset()
    y1 = y2 = u1 = u2 = 0.0
    N = 4000
    u = np.zeros(N)
    value = 0.0
    for k in range(N):
        if k % int(rng.integers(1, 5)) == 0:
            value = float(rng.choice([-1.0, 1.0]) * 30.0)
        u[k] = value
        y = -theta_true[0] * y1 - theta_true[1] * y2 + theta_true[2] * u1 + theta_true[3] * u2
        reg.step(0.0, y)
        reg.ack(u[k])
        y2, y1 = y1, y
        u2, u1 = u1, u[k]
    rel = np.abs(reg.theta - theta_true) / np.abs(theta_true)
    assert np.all(rel < 0.10), f"relative parameter error {rel}"


def test_rls_p_stays_symmetric_and_bounded():
    rng = np.random.default_rng(4)
    cfg = STRConfig(order=2, h=0.04, theta0=tuple(0.75 * THETA2_TRUE), p_max=5.0)
    reg = STR(cfg)
    for _ in range(500):
        y = float(rng.normal())
        reg.step(float(rng.normal()), y)
        P = reg.P
        assert np.allclose(P, P.T, atol=0.0)
        diag = np.diag(P)
        assert np.all(diag >= 0.0) and np.all(diag <= cfg.p_max + 1e-12)


# ---------------------------------------------------------------------------
# Control law (§5) and u_actual feedback (§6.4)
# ---------------------------------------------------------------------------
def test_control_law_first_ticks_match_manual_formula():
    theta0 = THETA2_TRUE
    cfg = STRConfig(
        order=2, h=0.04, theta0=tuple(theta0), mu=-0.4, p_max=1e-12, forgetting=1.0
    )
    reg = STR(cfg)
    g = reg.gains
    assert g.det > 0.0
    ao1 = float(reg.ao[0])

    u1 = reg.step(1.0, 0.5)
    expected1 = -ao1 * 0.0 + g.t0 * 1.0 + g.t1 * 0.0 - g.s0 * 0.5 - g.s1 * 0.0 + (ao1 - g.r1) * 0.0
    assert np.isclose(u1, expected1, atol=1e-12)

    reg.ack(0.3)
    u2 = reg.step(0.0, 0.2)
    expected2 = (
        -ao1 * expected1
        + g.t0 * 0.0
        + g.t1 * 1.0
        - g.s0 * 0.2
        - g.s1 * 0.5
        + (ao1 - g.r1) * 0.3
    )
    assert np.isclose(u2, expected2, atol=1e-12)


def test_ack_supplies_the_realised_command_to_the_regressor():
    theta0 = THETA2_TRUE
    cfg = STRConfig(order=2, h=0.04, theta0=tuple(theta0), p_max=100.0, forgetting=0.998)
    reg = STR(cfg)
    reg.step(0.0, 0.1)
    reg.ack(0.7)  # realised command differs from the design command

    before = reg.theta
    P_before = reg.P
    reg.step(0.0, 0.2)
    phi = np.array([-0.1, 0.0, 0.7, 0.0])
    expected = rls_update(before, P_before, phi, 0.2, cfg.forgetting, cfg.p_max)
    expected_theta = project_theta(expected.theta, 2, cfg.bsum_min)
    assert np.allclose(reg.theta, expected_theta, atol=1e-12)


def test_output_saturates_at_u_max():
    cfg = STRConfig(order=2, h=0.04, theta0=tuple(THETA2_TRUE), u_max=0.25, u_min=-0.25,
                    p_max=1e-12, forgetting=1.0)
    reg = STR(cfg)
    u = reg.step(100.0, -100.0)
    assert abs(u) == pytest.approx(0.25)


def test_mode1_analytic_guess_matches_zoh_model():
    from scipy import signal

    alpha = PARAMS.tf_alpha[0] / (180.0 / np.pi)
    beta = PARAMS.tf_beta[0]
    num, den, _ = signal.cont2discrete(([beta], [1.0, alpha, 0.0, 0.0]), 0.04, method="zoh")
    num, den = np.ravel(num), np.ravel(den)
    expected = np.array([den[1], den[2], den[3], num[1], num[2], num[3]])
    assert np.allclose(analytic_theta0_mode_i(alpha, beta, 0.04), expected)
    # The percent argument scales the guess linearly (§3, `percent = 0.75`).
    assert np.allclose(
        analytic_theta0_mode_ii(alpha, beta, 0.04, 0.75),
        0.75 * analytic_theta0_mode_ii(alpha, beta, 0.04, 1.0),
    )


# ---------------------------------------------------------------------------
# Numerical safety (§6.2/§6.3)
# ---------------------------------------------------------------------------
def test_det_guard_returns_zero_gains():
    # bsum = 1.001e-3 passes the b-sum guard but det ~ b1^2 = 1e-12 < 1e-9.
    cfg = STRConfig(order=2, h=0.04, theta0=(0.0, 0.0, 1e-3, 1e-6), u_max=10.0, u_min=-10.0)
    reg = STR(cfg)
    assert reg.gains.det == 0.0
    u = reg.step(0.1, 0.0)
    assert reg.status.frozen_gains
    assert np.isfinite(u)


def test_freeze_keeps_the_last_valid_gains(monkeypatch):
    cfg = STRConfig(order=2, h=0.04, theta0=tuple(THETA2_TRUE), u_max=10.0, u_min=-10.0)
    reg = STR(cfg)
    reg.step(0.1, 0.0)
    valid = reg.gains
    assert valid.det > GUARD_DET

    # Force the design step to look singular, as a rank-deficient estimate would.
    monkeypatch.setattr(reg, "_design_gains", lambda theta: Gains.zeros())
    u = reg.step(0.1, 0.0)
    assert reg.status.frozen_gains
    assert reg.gains == valid
    assert np.isfinite(u)


def test_nonfinite_input_restores_snapshot_and_enters_safe_mode():
    cfg = STRConfig(order=2, h=0.04, theta0=tuple(THETA2_TRUE), u_max=10.0, u_min=-10.0)
    reg = STR(cfg)
    reg.step(0.1, 0.0)
    theta_ok = reg.theta.copy()
    P_ok = reg.P.copy()

    for _ in range(FAIL_LIMIT - 1):
        out = reg.step(0.1, float("nan"))
        assert np.isfinite(out)
    assert not reg.safe_mode
    assert np.array_equal(reg.theta, theta_ok)
    assert np.array_equal(reg.P, P_ok)

    out = reg.step(0.1, float("nan"))
    assert out == 0.0
    assert reg.safe_mode
    # Latched until resetdata (§6.3).
    assert reg.step(0.1, 0.0) == 0.0

    reg.resetdata()
    assert not reg.safe_mode
    assert np.allclose(reg.theta, np.asarray(cfg.theta0))
    assert np.allclose(reg.P, cfg.p_max * np.eye(4))


def test_huge_command_is_treated_as_singular():
    cfg = STRConfig(order=2, h=0.04, theta0=tuple(THETA2_TRUE), u_max=10.0, u_min=-10.0)
    reg = STR(cfg)
    # A huge but finite measurement drives |v| beyond GUARD_U; the gains must be
    # frozen and the output clamped, never NaN or a singular design.
    u = reg.step(0.0, 1e9)
    assert reg.status.events.get("large_command", 0) >= 1 or reg.status.frozen_gains
    assert np.isfinite(u) and abs(u) <= cfg.u_max
    assert GUARD_U == 1e6


# ---------------------------------------------------------------------------
# Order 3 closed loop on a well-conditioned synthetic plant
# ---------------------------------------------------------------------------
def test_order3_closed_loop_tracks_step():
    theta = np.array([-1.2, 0.45, -0.05, 0.08, 0.05, 0.01])
    cfg = STRConfig(
        order=3,
        h=0.04,
        theta0=tuple(0.98 * theta),
        zeta=0.707,
        wn=7.9,
        cc=50.0,
        forgetting=0.998,
        p_max=100.0,
        u_max=100.0,
        u_min=-100.0,
    )
    reg = STR(cfg)
    channel = ArxChannel(theta)
    y = 0.0
    trace = []
    for k in range(376):
        t = 0.04 * k
        ref = 0.1 if t >= 1.0 - 1e-12 else 0.0
        u = reg.step(ref, y)
        reg.ack(u)
        y = channel.step(u)
        trace.append(y)
    trace = np.array(trace)
    assert np.all(np.isfinite(trace))
    assert abs(trace[-1] - 0.1) < 0.01, trace[-1]
    assert trace.max() < 0.15, trace.max()
    assert reg.theta[-3:].sum() > 0.0  # b-sum stayed finite and positive


def test_resetdata_is_exact():
    cfg = STRConfig(order=2, h=0.04, theta0=tuple(THETA2_TRUE), p_max=5.0)
    reg = STR(cfg)
    for _ in range(10):
        reg.step(0.1, 0.01)
    reg.resetdata()
    assert np.allclose(reg.theta, np.asarray(cfg.theta0))
    assert np.allclose(reg.P, 5.0 * np.eye(4))
    assert reg.ticks == 0
    assert not reg.safe_mode


def test_config_validation():
    with pytest.raises(ValueError):
        STRConfig(order=4)
    with pytest.raises(ValueError):
        STRConfig(forgetting=0.0)
    with pytest.raises(ValueError):
        STRConfig(theta0=(1.0, 2.0))  # wrong length for order 2
