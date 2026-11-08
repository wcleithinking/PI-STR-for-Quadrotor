"""Validation suite for the two-fidelity quadrotor plant.

Implements ``docs/PLANT_MODEL.md`` §8 (P1-P14) item by item.  Every check
asserts; the numbers behind each assertion are printed so a failing run is
self-explanatory (run ``pytest -s`` to see them).

Conventions used throughout
---------------------------
* The inertial frame is NED (``z`` down), so altitude is ``h = -xi_z``.  Hover
  tests start at ``xi_z = -5 m`` to keep ground effect (§7) out of the way —
  at ``h = 0`` the model correctly gives +30 % thrust augmentation, which would
  otherwise contaminate a free-air trim check.
* ``|a| < 1e-3 g`` means ``9.81e-3 m/s²`` in this file.
* Two actuator effects are switched off where a *static* or *motor-only* check
  is being made, and only there:
    - the 1 µs PWM quantisation steps hover thrust by ~0.5 %, i.e. ~5e-3 g, so a
      ``1e-3 g`` residual is only meaningful without it (P2);
    - the LiPo model makes the hover point decay continuously as the pack
      discharges (P2/P4), which is physical but not a trim error (P6 covers it).
  Both facts are reported in the test output.
"""

from __future__ import annotations

import numpy as np
import pytest

from pistr_sim.params import Params, default_params
from pistr_sim.plant import DEFAULT_EXTRAS, EXTRA_NAMES, QuadrotorPlant
from pistr_sim.types import PWM_MIN, VehicleState
from pistr_sim.wind import DrydenTurbulence

G = 9.81
#: Hover altitude in NED coordinates (h = 5 m, clear of ground effect).
XI_HOVER = np.array([0.0, 0.0, -5.0])

#: Every high-fidelity extra switched off (used by P1 and P12).
NO_EXTRAS = {name: False for name in EXTRA_NAMES}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_high(**overrides) -> QuadrotorPlant:
    """High-fidelity plant with ``overrides`` applied to the default extras."""
    extras = dict(DEFAULT_EXTRAS)
    extras.update(overrides)
    return QuadrotorPlant(default_params("high"), extras=extras)


def make_paper() -> QuadrotorPlant:
    return QuadrotorPlant(default_params("paper"), fidelity="paper")


def hover_state(plant: QuadrotorPlant, trim: dict, xi: np.ndarray | None = None) -> VehicleState:
    """Trim state moved to a given position (the trim solve is at the origin)."""
    state = trim["state"].copy()
    state.xi = XI_HOVER.copy() if xi is None else np.asarray(xi, dtype=float).copy()
    return state


def run(plant: QuadrotorPlant, pwm, steps: int) -> None:
    pwm = np.asarray(pwm, dtype=float) * np.ones(4)
    for _ in range(steps):
        plant.step(pwm)


def relative(a, b, scale: float) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / scale


# ---------------------------------------------------------------------------
# P1 - free fall (NED: a_z = +g)
# ---------------------------------------------------------------------------
def test_p01_free_fall():
    """No thrust, no drag: the vehicle accelerates at exactly +g along ẑ."""
    for plant in (make_paper(), QuadrotorPlant(default_params("high"), extras=NO_EXTRAS)):
        plant.reset()
        for _ in range(100):                       # 0.1 s
            plant.step(np.full(4, PWM_MIN))
        acc = plant.evaluate()["acc_inertial"]
        err = abs(acc[2] - G)
        print(f"P1 {plant.fidelity}: a = {acc}, |a_z-g| = {err:.3e}")
        assert err < 1e-6
        assert np.max(np.abs(acc[:2])) < 1e-12
        # v_z(t) = g t after 0.1 s
        assert abs(plant.state.v[2] - G * 0.1) < 1e-9
        assert np.all(np.isfinite(plant.state.v))


# ---------------------------------------------------------------------------
# P2 - hover trim on the calibrated BEMT model
# ---------------------------------------------------------------------------
def test_p02_hover_trim_calibrated():
    """Hover residual, rotor speed and net torque at the calibrated trim point."""
    plant = make_high(esc_lag=False, battery_sag=False, turbulence=False)
    trim = plant.trim_hover()
    plant.reset(hover_state(plant, trim))
    run(plant, trim["pwm_us"], 1000)               # 1 s

    ev = plant.evaluate()
    acc = ev["acc_inertial"]
    w = plant.state.rotor_omega
    w_hover = plant.params.w_hover
    residual_g = float(np.linalg.norm(acc)) / G
    w_err = float(np.max(np.abs(w - w_hover))) / w_hover
    torque = float(np.linalg.norm(ev["M_body"]))

    print(
        f"P2 calibrated trim: pwm={trim['pwm_us']:.2f} us, "
        f"theta_eff={np.rad2deg(plant.rotor_theta_effective):.4f} deg, "
        f"omega={trim['rotor_omega']:.4f} rad/s (w_hover={w_hover:.4f}), "
        f"|a|={residual_g:.3e} g, |M|={torque:.3e} N.m"
    )
    assert residual_g < 1e-3
    assert w_err < 0.05
    assert torque < 1e-4

    # information only: what the *uncalibrated* collective would give
    uncal = QuadrotorPlant(
        Params.from_matlab("high").with_(rotor=_replace_rotor(plant.params.rotor, calibrate=False)),
        extras={"esc_lag": False, "battery_sag": False, "turbulence": False},
    )
    uncal_trim = uncal.trim_hover()
    print(
        f"P2 (info, not asserted) uncalibrated theta_tw: "
        f"omega={uncal_trim['rotor_omega']:.2f} rad/s "
        f"({uncal_trim['rotor_omega'] / w_hover - 1:+.1%} vs w_hover), "
        f"current={uncal_trim['rotor_current']:.2f} A, "
        f"endurance={3.3 / (4 * uncal_trim['rotor_current']) * 60:.2f} min"
    )

    # information only: the price of the 1 µs PWM quantisation, which is why the
    # asserted residual above is measured on the continuous actuator (§8)
    quantised = make_high(turbulence=False)        # esc_lag on -> 1 µs steps
    q_trim = quantised.trim_hover()
    quantised.reset(hover_state(quantised, q_trim))
    a0 = float(np.linalg.norm(quantised.evaluate()["acc_inertial"]))
    run(quantised, q_trim["pwm_us"], 1000)
    a1 = float(np.linalg.norm(quantised.evaluate()["acc_inertial"]))
    print(
        f"P2 (info, not asserted) 1 µs quantisation: duty={q_trim['esc_duty']:.3f} "
        f"(ideal {q_trim['percent']:.5f}), residual {a0 / G * 1e3:.2f} mg at t=0 and "
        f"{a1 / G * 1e3:.2f} mg after 1 s; worst case is a full step = "
        f"{(1 - (q_trim['esc_duty'] - 1e-3) ** 2 / q_trim['esc_duty'] ** 2) * 1e3:.1f} mg"
    )


def _replace_rotor(rotor, **kwargs):
    from dataclasses import replace

    return replace(rotor, **kwargs)


# ---------------------------------------------------------------------------
# P3 - the published kT fit: blade element with lambda -> 0
# ---------------------------------------------------------------------------
def test_p03_published_kT_fit():
    """`0.5 rho A (wR)^2 sigma a theta/3` must reproduce `kT w^2` (0.5 %)."""
    plant = make_paper()                           # theta_tw untouched
    kt = plant.params.rotor.kt
    worst = 0.0
    for omega in (150.0, 300.0, plant.params.w_hover, 700.0):
        t_be = plant.blade_element_thrust(omega, lam=0.0)
        err = abs(t_be - kt * omega**2) / (kt * omega**2)
        worst = max(worst, err)
        print(f"P3 omega={omega:8.3f}: T_be={t_be:.9f} kT w^2={kt * omega**2:.9f} rel={err:.3e}")
    assert worst < 0.005

    # the same expression must reproduce the analytic inflow loss
    # T/T0 = 1 - 3*lambda/(2*theta)
    theta = plant.params.airframe.theta_tw
    lam = 0.1
    ratio = plant.blade_element_thrust(400.2, lam=lam) / plant.blade_element_thrust(400.2, lam=0.0)
    expected = 1.0 - 1.5 * lam / theta
    print(f"P3 inflow loss at lambda={lam}: {ratio:.6f} (analytic {expected:.6f})")
    assert abs(ratio - expected) < 1e-12
    assert ratio < 1.0                              # thrust falls with inflow/climb


# ---------------------------------------------------------------------------
# P4 - small-signal motor step (63 % rise)
# ---------------------------------------------------------------------------
def test_p04_motor_step_response():
    """Hover -> +5 % motor voltage: 63 % rise time == ln2/tf_beta ... see note.

    ``tf_beta`` is the pole *rate* (1/s) of the linearised motor, so the 63.2 %
    rise time of a first-order response is ``1/tf_beta`` and the 50 % rise time
    is ``ln(2)/tf_beta``.  Both are checked; the spec's "63 % rise ≈ ln2/tf_beta"
    is the 50 % crossing (reported to the Lead).
    """
    plant = make_high(esc_lag=False, battery_sag=False, turbulence=False)
    trim = plant.trim_hover()
    plant.reset(hover_state(plant, trim))

    beta = float(plant.params.tf_beta[0])
    v_term = plant._terminal_voltage(plant.state.rotor_current, 1.0)
    duty = trim["esc_duty"]
    pct = duty * 1.05                              # +5 % motor voltage

    w0 = float(plant.state.rotor_omega[0])
    hist = []
    for _ in range(600):                           # 0.6 s
        plant.step(np.full(4, PWM_MIN + 1000.0 * pct))
        hist.append(float(plant.state.rotor_omega[0]))
    hist = np.asarray(hist)
    w_inf = hist[-1]
    delta = w_inf - w0

    def crossing(fraction: float) -> float:
        idx = int(np.argmax(hist >= w0 + fraction * delta))
        return idx * plant.dt

    t50, t63 = crossing(0.5), crossing(0.632)
    err63 = abs(t63 - 1.0 / beta) * beta
    err50 = abs(t50 - np.log(2.0) / beta) * beta / np.log(2.0)
    print(
        f"P4 small signal: w {w0:.3f}->{w_inf:.3f} rad/s, "
        f"t63={t63 * 1e3:.2f} ms (1/beta={1e3 / beta:.2f} ms, err {err63:.1%}), "
        f"t50={t50 * 1e3:.2f} ms (ln2/beta={1e3 * np.log(2) / beta:.2f} ms, err {err50:.1%}), "
        f"Vterm={v_term:.3f} V"
    )
    assert err63 < 0.10
    assert err50 < 0.10

    # information only: the large-signal 0 -> hover response (not first order)
    big = make_high(esc_lag=False, battery_sag=False, turbulence=False)
    big.reset(VehicleState(xi=XI_HOVER.copy()))
    hist = []
    for _ in range(400):
        big.step(np.full(4, trim["pwm_us"]))
        hist.append(float(big.state.rotor_omega[0]))
    hist = np.asarray(hist)
    idx = int(np.argmax(hist >= 0.632 * plant.params.w_hover))
    print(
        f"P4 (info, not asserted) 0->hover 63 % rise = {idx * big.dt * 1e3:.1f} ms "
        f"(linearised ln2/beta = {1e3 * np.log(2) / beta:.1f} ms)"
    )


# ---------------------------------------------------------------------------
# P5 - energy consistency
# ---------------------------------------------------------------------------
def test_p05_energy_consistency():
    """P_elec / (sum ke*omega*I) in [1, 1.6]; P_elec / (sum Q omega) reported."""
    plant = make_high(esc_lag=False, turbulence=False)
    trim = plant.trim_hover()
    plant.reset(hover_state(plant, trim))
    run(plant, trim["pwm_us"], 200)

    ev = plant.evaluate()
    p_elec = ev["P_electrical"]
    p_emf = ev["P_back_emf"]
    p_shaft = ev["P_shaft"]
    ratio_emf = p_elec / p_emf
    ratio_shaft = p_elec / p_shaft
    print(
        f"P5 P_elec={p_elec:.3f} W, sum(ke w I)={p_emf:.3f} W -> ratio {ratio_emf:.4f}; "
        f"sum(Q w)={p_shaft:.3f} W -> P_elec/sum(Qw) = {ratio_shaft:.4f} "
        f"(known km!=ke inconsistency, not asserted)"
    )
    assert 1.0 <= ratio_emf <= 1.6
    # mechanical absorption is positive and equals the aero load torque * omega
    assert p_shaft > 0.0
    assert abs(p_shaft - float(np.dot(ev["Q_rotor"], plant.state.rotor_omega))) < 1e-9


# ---------------------------------------------------------------------------
# P6 - battery: sag, monotone SoC, endurance
# ---------------------------------------------------------------------------
def test_p06_battery():
    """Hover on the pack: SoC falls, terminal voltage sags by R_int*I, endurance."""
    plant = make_high(esc_lag=False, turbulence=False)
    trim = plant.trim_hover()
    plant.reset(hover_state(plant, trim))

    soc0 = plant.state.soc
    pwm = trim["pwm_us"]
    socs = [soc0]
    energy = 0.0
    for _ in range(2000):                          # 2 s
        plant.step(np.full(4, pwm))
        socs.append(plant.state.soc)
        i_pack = float(np.sum(plant.state.rotor_current))
        energy += plant.state.esc_duty[0] * plant.state.v_terminal * i_pack * plant.dt

    socs = np.asarray(socs)
    pack_current = float(np.sum(plant.state.rotor_current))
    endurance_min = 3.3 / pack_current * 60.0

    # V_term must equal OCV(SoC) - R_int * sum(I) to 1 %
    expected = plant.params.battery.open_circuit_voltage(plant.state.soc) - (
        plant.params.battery.r_int * pack_current
    )
    sag_err = abs(plant.state.v_terminal - expected) / expected

    # energy bookkeeping: dSoC = -P dt / (Q V 3600)
    soc_expected = soc0 - energy / (3.3 * 11.1 * 3600.0)

    print(
        f"P6 SoC {soc0:.6f} -> {socs[-1]:.6f} (analytic {soc_expected:.6f}), "
        f"pack current {pack_current:.3f} A, endurance {endurance_min:.2f} min, "
        f"V_term={plant.state.v_terminal:.4f} V (expected {expected:.4f}, err {sag_err:.2e}), "
        f"sag={12.6 - plant.state.v_terminal:.3f} V ~ R_int*I={plant.params.battery.r_int * pack_current:.3f} V"
    )
    assert np.all(np.diff(socs) < 0.0)             # monotone decreasing
    assert sag_err < 0.01
    assert abs(socs[-1] - soc_expected) / (soc0 - soc_expected) < 0.01
    assert 8.0 <= endurance_min <= 30.0            # sanity band on the calibrated model


# ---------------------------------------------------------------------------
# P7 - gyroscopic moment (G1, G2)
# ---------------------------------------------------------------------------
def test_p07_gyroscopic_moment():
    """G1: equal rotor speeds => exactly zero.  G2: M_y = +Omega_x * h_z."""
    plant = make_high(turbulence=False)

    # G1: equal speeds -> sum(dir_i * omega_i) = 0 -> no gyroscopic moment
    equal = VehicleState(rotor_omega=np.full(4, 400.0), omega=np.array([0.3, -0.2, 0.1]))
    ev = plant.evaluate(equal)
    print(f"P7 G1 equal speeds: h_rotor={ev['h_rotor']}, M_gyro={ev['M_gyro']}")
    assert np.max(np.abs(ev["h_rotor"])) == 0.0
    assert np.max(np.abs(ev["M_gyro"])) == 0.0

    # G2: unequal speeds give h_z != 0; with Omega = (Omega_x, 0, 0),
    #     M_gyro = -Omega x h = (0, +Omega_x h_z, 0)
    unequal = VehicleState(
        rotor_omega=np.array([430.0, 380.0, 430.0, 380.0]), omega=np.array([0.3, 0.0, 0.0])
    )
    ev = plant.evaluate(unequal)
    h_z = plant._jr * float(np.dot(plant._spin, unequal.rotor_omega))
    print(
        f"P7 G2 h_z={h_z:.6e}, M_gyro={ev['M_gyro']}, "
        f"expected (0, {0.3 * h_z:.6e}, 0)"
    )
    assert abs(h_z) > 0.0
    assert abs(ev["M_gyro"][1] - 0.3 * h_z) < 1e-15
    assert abs(ev["M_gyro"][0]) < 1e-15 and abs(ev["M_gyro"][2]) < 1e-15

    # and the coupling really shows up in the yaw/roll rate derivative
    assert abs(ev["domega"][1]) > 0.0

    # switching the extra off removes it entirely
    off = make_high(turbulence=False, gyro_moment=False)
    assert np.max(np.abs(off.evaluate(unequal)["M_gyro"])) == 0.0


# ---------------------------------------------------------------------------
# P8 - body drag
# ---------------------------------------------------------------------------
def test_p08_body_drag():
    """4 m/s level flight at zero attitude: F_x = -0.5 rho CdA_x V^2 (2 %)."""
    plant = make_high(turbulence=False)
    assert plant.extras["rotor_projected_area"] is False   # off by default (§8)
    state = VehicleState(v=np.array([4.0, 0.0, 0.0]))
    ev = plant.evaluate(state)
    cd_a = plant.params.aero.drag_area()[0]
    expected = -0.5 * plant.params.airframe.rho * cd_a * 4.0**2
    err = abs(ev["F_drag_body"][0] - expected) / abs(expected)
    print(f"P8 F_x={ev['F_drag_body'][0]:.9f} N, expected {expected:.9f} N, rel err {err:.2e}")
    assert err < 0.02
    assert abs(ev["F_drag_body"][1]) < 1e-12

    # the optional rotor-projected area only touches the vertical axis
    with_area = make_high(turbulence=False, rotor_projected_area=True)
    print(
        f"P8 (info) CdA_z base={plant._cda[2]:.5f} m^2, "
        f"with rotor_projected_area={with_area._cda[2]:.5f} m^2"
    )
    assert with_area._cda[2] > plant._cda[2]
    assert with_area._cda[0] == plant._cda[0]


# ---------------------------------------------------------------------------
# P9 - inertia sanity
# ---------------------------------------------------------------------------
def test_p09_inertia():
    """Eigenvalues > 0, symmetric, and identical to the MATLAB expressions."""
    af = default_params("paper").airframe
    j = np.asarray(af.inertia, dtype=float)

    # recompute from the MATLAB source (Load_SystemParameters.m lines 51-60)
    m, d = af.m_cog, af.d_arm
    r, h = af.r_cog, af.h_cog
    mr, rr, cr, hr = af.m_rotor, af.r_rotor, af.c_rotor, af.h_rotor
    mm, rm, hm = af.m_motor, af.r_motor, af.h_motor
    ixx = (
        m * (h**2 + 3 * r**2) / 12
        + 2 * mm * d**2
        + 2 * (1 / 12) * mm * (3 * rm**2 + 4 * hm**2)
        + 2 * mr * d**2
        + 2 * (1 / 12 * mr * (2 * rr) ** 2)
    )
    iyy = (
        m * (h**2 + 3 * r**2) / 12
        + 2 * mm * d**2
        + 2 * (1 / 12) * mm * (3 * rm**2 + 4 * hm**2)
        + 2 * mr * d**2
        + 2 * (1 / 12 * mr * cr**2)
    )
    izz = (
        m * r**2 / 2
        + 4 * mm * d**2
        + 4 * mm * rm**2 / 2
        + (4 / 12) * mr * ((2 * rr) ** 2 + cr**2)
        + 4 * mr * d**2
    )
    ixz = 2 * mm * hm * d + 2 * mr * hr * d
    matlab = np.array([[ixx, 0.0, ixz], [0.0, iyy, ixz], [ixz, ixz, izz]])

    rel = np.max(np.abs(j - matlab)) / np.max(np.abs(matlab))
    eig = np.linalg.eigvalsh(j)
    print(f"P9 J=\n{j}\n   eigenvalues={eig}, rel err vs MATLAB={rel:.3e}")
    assert rel < 1e-9
    assert np.all(eig > 0.0)
    assert j[0, 2] == j[2, 0] and j[1, 2] == j[2, 1]
    assert j[2, 1] == j[0, 2]                      # the (documented) Iyz = Ixz quirk


# ---------------------------------------------------------------------------
# P10 - quaternion integrity over 120 s
# ---------------------------------------------------------------------------
def test_p10_quaternion_integrity():
    """120 s at hover plus a fast free rotation: |norm(q)-1| < 1e-9, no NaN."""
    # (a) 120 s at the trim point: nothing may drift at all
    plant = make_paper()
    trim = plant.trim_hover()
    plant.reset(hover_state(plant, trim))
    worst = 0.0
    pwm = np.full(4, trim["pwm_us"])
    for _ in range(120_000):                       # 120 s at 1 kHz
        plant.step(pwm)
        worst = max(worst, abs(float(np.linalg.norm(plant.state.q)) - 1.0))
    s = plant.state
    print(
        f"P10a paper 120 s hover: max||q|-1|={worst:.3e}, q={s.q}, "
        f"xi={s.xi}, w={s.rotor_omega[0]:.6f}"
    )
    assert worst < 1e-9
    assert np.array_equal(s.q, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.max(np.abs(s.xi - XI_HOVER)) < 1e-9
    assert np.all(np.isfinite(s.q)) and np.all(np.isfinite(s.rotor_omega))

    # (b) free rotation at 2.7 rad/s for 20 s (≈170 revolutions): the quaternion
    #     update is the only thing keeping the attitude sane here
    spin = make_paper()
    spin.reset(
        VehicleState(
            omega=np.array([2.0, -1.5, 1.0]),
            xi=XI_HOVER.copy(),
            rotor_omega=np.zeros(4),
            rotor_current=np.zeros(4),
            pwm_us=np.full(4, PWM_MIN),
        )
    )
    worst_spin = 0.0
    for _ in range(20_000):
        spin.step(np.full(4, PWM_MIN))
        worst_spin = max(worst_spin, abs(float(np.linalg.norm(spin.state.q)) - 1.0))
    print(
        f"P10b paper 20 s free rotation: max||q|-1|={worst_spin:.3e}, "
        f"|omega|={np.linalg.norm(spin.state.omega):.4f} rad/s"
    )
    assert worst_spin < 1e-9
    assert np.all(np.isfinite(spin.state.q))

    # (c) the full high-fidelity model (shorter horizon: the BEMT solve makes it
    #     ~20x more expensive per step)
    hi = make_high(turbulence=True, sensors=True)
    hi.reset(VehicleState(q=_euler_to_quat(0.02, 0.02, 0.05), xi=XI_HOVER.copy()))
    worst_hi = 0.0
    for _ in range(10_000):                        # 10 s
        hi.step(np.full(4, 1400.0))
        worst_hi = max(worst_hi, abs(float(np.linalg.norm(hi.state.q)) - 1.0))
    print(f"P10c high 10 s: max||q|-1|={worst_hi:.3e}, finite={np.all(np.isfinite(hi.state.q))}")
    assert worst_hi < 1e-9
    assert np.all(np.isfinite(hi.state.q))


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy - sr * sp * sy,
            sr * cp * cy + cr * sp * sy,
            cr * sp * cy - sr * cp * sy,
            cr * cp * sy + sr * sp * cy,
        ]
    )


# ---------------------------------------------------------------------------
# P11 - Dryden spectrum
# ---------------------------------------------------------------------------
def test_p11_dryden_spectrum():
    """Half-power corner at Vbar/L (factor 2) and time-domain std == sigma."""
    params = default_params("high").wind
    dt = 1e-3
    turb = DrydenTurbulence(params, dt)

    # (a) analytic variance of the recursion must equal sigma^2
    var_err = np.max(np.abs(turb.variance / np.asarray(params.sigma) ** 2 - 1.0))
    print(f"P11 analytic variance {turb.variance} vs sigma^2 {np.asarray(params.sigma) ** 2}")

    # (b) half-power corner of the shaping filter, evaluated on a fine grid
    omega = np.logspace(-5.0, 1.0, 4000)           # rad/s
    for i, name in enumerate("uvw"):
        h = turb.b[i] / (1.0 - turb.a[i] * np.exp(-1j * omega * dt))
        psd = np.abs(h) ** 2
        idx = int(np.argmax(psd <= 0.5 * psd[0]))
        corner = omega[idx]
        ratio = corner / turb.omega_c[i]
        print(
            f"P11 {name}: corner {corner:.5f} rad/s, Vbar/L {turb.omega_c[i]:.5f} rad/s, "
            f"ratio {ratio:.3f}"
        )
        assert 0.5 <= ratio <= 2.0

    # (c) time-domain std == sigma, measured with a resolvable length scale
    fast = DrydenTurbulence(
        params.__class__(sigma=params.sigma, length_scale=(20.0, 20.0, 5.0), v_ref=params.v_ref),
        dt,
    )
    series = fast.series(int(200.0 / dt))
    std = series.std(axis=0)
    print(f"P11 empirical std (L=20 m, 200 s) {np.round(std, 4)} vs sigma {params.sigma}")
    assert np.all(np.abs(std / np.asarray(params.sigma) - 1.0) < 0.25)

    # (d) determinism
    same = DrydenTurbulence(params, dt)
    assert np.array_equal(turb.series(50), same.series(50))
    assert var_err < 1e-9


# ---------------------------------------------------------------------------
# P12 - fidelity equivalence
# ---------------------------------------------------------------------------
def test_p12_fidelity_equivalence():
    """Equal motor voltage, all extras off: high must track paper over 2 s."""
    u_hover = 11.1 * np.sqrt(make_paper().trim_hover()["percent"])

    def compare(integrator: str, step: float, horizon: float = 2.0) -> dict:
        paper = make_paper()
        high = QuadrotorPlant(
            default_params("high"), extras=NO_EXTRAS, integrator=integrator
        )
        tp, th = paper.trim_hover(), high.trim_hover()
        paper.reset(hover_state(paper, tp))
        high.reset(hover_state(high, th))
        worst = {"q": 0.0, "omega": 0.0, "v": 0.0, "xi": 0.0, "rotor_omega": 0.0,
                 "rotor_current": 0.0}
        scales = {"q": 1.0, "omega": 1.0, "v": 1.0, "xi": 1.0,
                  "rotor_omega": 400.2, "rotor_current": 3.8}
        n = int(round(horizon / paper.dt))
        for k in range(n):
            t = k * paper.dt
            factor = 1.0 if t < 0.1 else step
            # equal motor voltage: paper p=(U/11.1)^2, high p=U/V_term
            u = u_hover * factor
            a = paper.step(np.full(4, PWM_MIN + 1000.0 * (u / 11.1) ** 2))
            b = high.step(np.full(4, PWM_MIN + 1000.0 * (u / high.state.v_terminal)))
            for key in worst:
                worst[key] = max(
                    worst[key], relative(getattr(a, key), getattr(b, key), scales[key])
                )
        print(
            f"P12 integrator={integrator} step={step}: "
            + ", ".join(f"{k}={v:.2e}" for k, v in worst.items())
            + f" | max={max(worst.values()):.2e} | xi_z paper={paper.state.xi[2]:.4f}"
            f" high={high.state.xi[2]:.4f}"
        )
        return worst

    loose = compare("rk4", 1.10)                    # paper Euler vs high RK4
    strict = compare("euler", 1.05)                 # identical integrator
    assert max(loose.values()) < 0.05
    assert max(strict.values()) < 1e-6


# ---------------------------------------------------------------------------
# P13 - yaw sign (G3)
# ---------------------------------------------------------------------------
def test_p13_yaw_sign():
    """A positive yaw command at hover must produce dOmega_z/dt > 0."""
    plant = make_high(esc_lag=False, turbulence=False)
    trim = plant.trim_hover()
    plant.reset(hover_state(plant, trim))

    # the MATLAB/firmware mixer column for yaw is [1, -1, 1, -1] (u_mix column 3)
    u_yaw = 20.0
    pwm = trim["pwm_us"] + np.array([1.0, -1.0, 1.0, -1.0]) * u_yaw
    before = plant.evaluate()
    plant.step(pwm)
    after = plant.evaluate()
    # the force/moment of a step is evaluated from its start-of-step rotor state
    # (the MATLAB convention), so the new yaw torque acts from the next step on
    for _ in range(4):
        plant.step(pwm)

    print(
        f"P13 dOmega_z before={before['domega'][2]:.6f}, after={after['domega'][2]:.6f} 1/s^2, "
        f"M_z={after['M_body'][2]:.6e} N.m, Omega_z={plant.state.omega[2]:.6e} rad/s"
    )
    assert before["domega"][2] == pytest.approx(0.0, abs=1e-9)
    assert after["domega"][2] > 0.0
    assert plant.state.omega[2] > 0.0


# ---------------------------------------------------------------------------
# P14 - mixer parity (G4)
# ---------------------------------------------------------------------------
def test_p14_mixer_parity():
    """Same rotor speeds => same yaw torque as MATLAB's w_rotor_mix row 3."""
    paper = make_paper()
    high = make_high(turbulence=False)
    w = np.array([380.0, 425.0, 395.0, 410.0])
    # at 5 m so that ground effect (§7) does not scale the arm moments
    state = VehicleState(rotor_omega=w.copy(), xi=XI_HOVER.copy())
    kq = paper.params.rotor.kq
    matlab = kq * (w[0] ** 2 - w[1] ** 2 + w[2] ** 2 - w[3] ** 2)
    m_paper = float(paper.evaluate(state)["M_yaw"][2])
    m_high = float(high.evaluate(state)["M_yaw"][2])
    print(
        f"P14 M_z: paper={m_paper:.12f}, high={m_high:.12f}, MATLAB={matlab:.12f}, "
        f"diff={abs(m_paper - matlab):.2e}/{abs(m_high - matlab):.2e}"
    )
    assert abs(m_paper - matlab) < 1e-15
    assert abs(m_high - matlab) < 1e-15

    # and the whole mixer matrix (roll/pitch/altitude) matches MATLAB
    tf_paper = paper.evaluate(state)["M_rotor_arms"]
    tf_high = high.evaluate(state)["M_rotor_arms"]
    d, kt = paper.params.airframe.d_arm, paper.params.rotor.kt
    matlab_roll = d * kt * (w[3] ** 2 - w[1] ** 2)
    matlab_pitch = d * kt * (w[0] ** 2 - w[2] ** 2)
    print(
        f"P14 roll {tf_paper[0]:.9f}/{tf_high[0]:.9f} vs {matlab_roll:.9f}; "
        f"pitch {tf_paper[1]:.9f}/{tf_high[1]:.9f} vs {matlab_pitch:.9f}"
    )
    assert abs(tf_paper[0] - matlab_roll) < 1e-12 and abs(tf_high[0] - matlab_roll) < 1e-12
    assert abs(tf_paper[1] - matlab_pitch) < 1e-12 and abs(tf_high[1] - matlab_pitch) < 1e-12


# ---------------------------------------------------------------------------
# P15 - the electrical integration artifact (§10.5)
# ---------------------------------------------------------------------------
def _collective_eigenvalues(plant: QuadrotorPlant) -> np.ndarray:
    """Eigenvalues of the ``(ω, I)`` collective mode at the hover trim point.

    The four motors share one pack, so the mode excited by a throttle command is
    the *collective* one: all four currents move together and the effective
    resistance is ``rm + 4·d·R_int``.  Perturbing a single motor instead would
    give the differential mode (``rm``) and understate the stiffness by ~15 %.
    """
    trim = plant.trim_hover()
    state = trim["state"]
    y0 = plant._pack_actuators(
        state.rotor_omega, state.rotor_current, state.esc_duty, state.soc
    )
    d0 = plant._act_deriv(y0, state.esc_duty, np.zeros(4), 0.0)
    eps = 1e-6
    jac = np.zeros((2, 2))
    for col, sl in enumerate((slice(0, 4), slice(4, 8))):
        y = y0.copy()
        y[sl] += eps
        d = plant._act_deriv(y, state.esc_duty, np.zeros(4), 0.0)
        jac[0, col] = (d[0] - d0[0]) / eps
        jac[1, col] = (d[4] - d0[4]) / eps
    return np.linalg.eigvals(jac)


def _current_step_response(substeps: int, steps: int = 400) -> dict:
    """Hover -> +5 % motor voltage, MATLAB-verbatim ``paper`` plant, return stats."""
    plant = QuadrotorPlant(
        default_params("paper"), fidelity="paper", actuator_substeps=substeps
    )
    trim = plant.trim_hover()
    plant.reset(hover_state(plant, trim))
    pct = trim["percent"] * 1.05**2                # +5 % motor voltage
    pwm = np.full(4, PWM_MIN + 1000.0 * pct)
    trace = np.empty(steps)
    for k in range(steps):
        plant.step(pwm)
        trace[k] = float(plant.state.rotor_current[0])
    tail = trace[-100:]                            # last 0.1 s
    return {
        "mean": float(tail.mean()),
        "p2p": float(tail.max() - tail.min()),
        "min": float(trace.min()),
        "max": float(trace.max()),
        "step": float(np.mean(np.abs(np.diff(tail)))),
        "final": float(trace[-1]),
    }


def test_p15_electrical_integration_artifact():
    """§10.5: at the published parameters the raw one-step current update diverges.

    Three executable facts, so the claim does not rest on prose:

    1. the explicit-Euler amplification factor ``1 - dt*rm/L`` is outside the
       unit circle (and negative, hence Nyquist-frequency ringing);
    2. RK4 does not rescue it: the collective ``(ω, I)`` eigenvalue violates
       RK4's real-axis stability limit at this ``dt``;
    3. ``actuator_substeps=1`` (the MATLAB update order) really does oscillate
       and rectifies against the ``I >= 0`` clamp, while the default substep
       count is smooth *and* converged.
    """
    plant = make_paper()
    dt = plant.dt
    inductance = plant.params.motor.inductance
    resistance = plant.params.motor.resistance
    tau_e = inductance / resistance

    # (1) Euler amplification factor
    euler_gain = 1.0 - dt * resistance / inductance
    print(
        f"P15(1) tau_e = L/rm = {tau_e * 1e3:.3f} ms, "
        f"1 - dt*rm/L = {euler_gain:+.4f} (must be < -1)"
    )
    assert euler_gain < -1.0

    # (2) RK4 stability: z = lambda*dt against the real-axis limit -2.785
    high = make_high(esc_lag=False, turbulence=False)   # battery on: shared pack
    eig = _collective_eigenvalues(high)
    lam_fast = float(np.max(np.abs(eig)))
    z = lam_fast * dt
    print(
        f"P15(2) collective (omega, I) eigenvalues {np.round(eig, 3)}, "
        f"|lambda_max|*dt = {z:.4f} vs RK4 limit 2.785"
    )
    assert z > 2.785

    # (3) the artifact itself, and its absence with substepping
    raw = _current_step_response(1)
    fine = _current_step_response(high.actuator_substeps)
    reference = _current_step_response(32)
    print(
        f"P15(3) substeps=1: mean={raw['mean']:.4f} A, peak-to-peak={raw['p2p']:.4f} A, "
        f"min={raw['min']:.4f} A, max={raw['max']:.4f} A, mean|dI|/step={raw['step']:.4f} A"
    )
    print(
        f"P15(3) substeps={high.actuator_substeps}: mean={fine['mean']:.6f} A, "
        f"peak-to-peak={fine['p2p']:.3e} A, mean|dI|/step={fine['step']:.3e} A, "
        f"converged value {reference['final']:.6f} A (substeps=32)"
    )

    # the MATLAB-verbatim run chatters between the clamp and ~10 A
    assert raw["p2p"] > 1.0
    assert raw["min"] == 0.0                       # rectified by the I >= 0 clamp
    assert raw["step"] > 1.0
    # ... and its mean operating point is wrong by several percent
    assert abs(raw["mean"] - reference["final"]) / reference["final"] > 0.05

    # the substep run is smooth and agrees with a 32-substep reference
    assert fine["step"] < 1e-6
    assert fine["p2p"] < 1e-6
    assert abs(fine["final"] - reference["final"]) < 1e-4
