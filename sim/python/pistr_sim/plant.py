"""Two-fidelity quadrotor plant.

Normative text: ``docs/PLANT_MODEL.md`` (plant) and ``docs/CONTROL_SPEC.md`` §2
(the discrete model the controller assumes).  ``types.py`` and ``params.py`` are
the frozen contract layer; this module only *reads* them.

Frame convention (§4, normative)
--------------------------------
The inertial frame is **NED**: ``z`` points *down*, gravity is ``+g ẑ`` with
``g = +9.81``, thrust acts along the body ``-ẑ``, and altitude is ``h = -xi_z``.
Consequences:

* ``xi[2]`` is *negative* altitude; ground effect (§7) and the barometer
  (``baro = -xi_z``, §5) both use ``h = -xi[2]``.
* A climbing vehicle has ``V_c > 0`` (air flows down through the disc).
* Free fall accelerates at ``+g`` along ``ẑ`` — what validation item P1 asserts.

Rotor numbering and spin direction (§4): rotors 0, 2 spin CW and 1, 3 spin CCW
(viewed from above); ``dir_i = +1`` for CW, i.e. ``spin_dirs = (+1,-1,+1,-1)``.
``dir_i`` enters both the yaw reaction torque ``M_yaw = Σ dir_i Q_i`` and the
rotor angular momentum ``h_rotor = J_r (Σ dir_i ω_i) ẑ``.  Sign conventions are
asserted by tests (G1-G4 / P7, P13, P14), never left to prose.

Fidelity levels
---------------
``fidelity="paper"``
    Port of the ``%% True System`` block of
    ``sim/matlab/reduced_overshoot/AttitudeandAltitude_Tracking_Control_Mode_II.m``
    (attitude + altitude) and of ``sim/matlab/robustness/Sim_Trajectory_Tracking_Control.m``
    (3-D position): ``U = Umax*sqrt(PWM_percent)``, ``T = kT Σω²``,
    ``Γ = w_rotor_mix ω²``, explicit Euler, no aero, no sensors, no battery.
    Every update order is copied from MATLAB, including the facts that ``I(k+1)``
    and ``ω(k+1)`` both use ``I(k)``, that the thrust used in a step comes from
    ``ω(k)`` (pre-update), and that ``xi(k+1) = xi(k) + dt v(k)``.

``fidelity="high"`` (default)
    Engineering model of ``PLANT_MODEL.md`` §§2-7: BEMT inflow solve with the
    re-calibrated collective pitch (§2), H-force, body drag, rotor gyroscopic
    moment, ground effect, LiPo sag, ESC lag with PWM quantisation, Dryden
    turbulence, sensor corruption, RK4.

    Each "extra" can be switched off individually through ``extras`` so that
    ablation studies and the P12 equivalence test are meaningful.

Integration scheme (operator split + actuator sub-stepping)
-----------------------------------------------------------
The published motor has ``L = 150 µH`` and ``rm = 0.377 Ω``, i.e. an electrical
time constant ``L/rm = 0.397 ms``.  At the normative ``dt = 1 ms`` that pole sits
**outside** the stability region of both integrators:

* explicit Euler (the MATLAB update): amplification ``1 − dt·rm/L = −1.516``,
  ``|·| > 1`` → the current rings at the Nyquist frequency, grows, hits the
  ``I ≥ 0`` clamp and injects a spurious torque.  This is a property of the
  reference implementation, reproduced by ``actuator_substeps=1`` (P15).
* RK4: the measured fast eigenvalue of the coupled ``(I, ω)`` pair is ≈ −2880 1/s,
  ``z = −2.88`` is beyond the RK4 real-axis limit (−2.785), ``|R(z)| = 1.15``.

The plant therefore splits each ``dt`` into

1. one aerodynamics evaluation from the step-start rotor state (the same
   "force held over the step" convention MATLAB uses with ``ω(k)``);
2. ``actuator_substeps`` sub-steps of the fast actuator sub-system ``(I, ω, d, SoC)``;
3. one rigid-body step (Euler for ``paper``, RK4 for ``high``) with the frozen
   force/moment of (1).

``actuator_substeps`` defaults to ``ceil(dt / (0.5·L/rm)) = 6``, which puts the electrical
pole inside both stability regions with margin and is cheap (13 states).
``actuator_substeps=1`` restores the raw one-shot update.  The fixed point is unchanged, so
hover trim is bit-identical to MATLAB; only transients differ — and they differ
because MATLAB's are numerically corrupted.  Reported to the Lead as a spec
issue (§3 gives no sub-stepping guidance).

Collective-pitch calibration (§2)
---------------------------------
``kT`` was fitted against the *inflow-free* blade-element expression, so running
BEMT at the published ``theta_tw`` would drop hover thrust by ~42 % and move the
trim point by tens of percent.  With ``Rotor.calibrate`` (default ``True``) the
high-fidelity plant solves, once per parameter set,

    theta_eff  such that  BEMT(ω = w_hover, V_c = 0, θ = theta_eff) = kT w_hover²

by bisection (monotone, 1e-12 rad) and uses ``theta_eff`` for every inflow
solve.  The published ``theta_tw`` is untouched; ``rotor_theta_effective``
exposes the calibrated value for the run log.  Useful consequence: at
``V_c = 0`` the calibrated BEMT reduces *exactly* to ``kT ω²`` for every ω (the
fixed point has a constant inflow ratio), so calibration changes climb/descent
behaviour but not the hover anchor.

Other implementation notes (agreed with the Lead, recorded so they are not
"fixed" silently)
-----------------------------------------------------------------------
* ``esc_lag`` off also disables the 1 µs PWM quantisation — both belong to the
  same actuator path, and this is what makes P12 an exact comparison.
* ``rotor_projected_area`` (the §1 "vertical rotor-projected area") is a
  **separate switch, default off**, because adding ``4πR²`` to ``CdA_z`` would
  double-count the axial momentum drag already inside the BEMT solve.
* ``M_drag`` in §4 has no defining formula; drag is applied at the CoG, so the
  aerodynamic moment is zero.
* ``dSoC/dt = −P/(Q_pack·V_nom·3600)``: ``Q_pack`` is in Ah and ``dt`` in s, so
  the 3600 s/h factor is required for a dimensionless SoC.  §3 omits it; without
  it the pack empties 3600× too fast (dimensional slip, reported to the Lead).
* The BEMT fixed point is seeded with ``v_i = sqrt(kT ω²/(2ρA))`` rather than a
  stateful "previous step" value, so that :meth:`QuadrotorPlant.evaluate` is a
  pure function of the state.  Both converge to the same root (the damped
  iteration is a contraction with factor ≈ 0.41; 12 iterations are ample).

Controllers must never read plant internals — only :meth:`QuadrotorPlant.sense`.
:meth:`QuadrotorPlant.evaluate` exists for the validation suite and logging and
is *not* part of the ``Plant`` protocol.
"""

from __future__ import annotations

import numpy as np

from .params import (
    Aero,
    Airframe,
    Battery,
    Esc,
    Motor,
    Params,
    Rotor,
    SensorParams,
    default_params,
)
from .types import (
    PWM_MAX,
    PWM_MIN,
    SensorSample,
    VehicleState,
    quat_derivative,
    quat_normalise,
    quat_to_rotmat,
)
from .wind import WindModel

__all__ = [
    "DEFAULT_EXTRAS",
    "PAPER_EXTRAS",
    "QuadrotorPlant",
    "make_plant",
    "EXTRA_NAMES",
]

#: Every real-world effect the high-fidelity model can switch off (§8).
EXTRA_NAMES = (
    "inflow",
    "h_force",
    "body_drag",
    "gyro_moment",
    "battery_sag",
    "esc_lag",
    "turbulence",
    "sensors",
    "rotor_projected_area",
    "ground_effect",
)

DEFAULT_EXTRAS: dict[str, bool] = {name: True for name in EXTRA_NAMES}
#: Off by default: it would double-count momentum drag inside the BEMT solve (§8).
DEFAULT_EXTRAS["rotor_projected_area"] = False
PAPER_EXTRAS: dict[str, bool] = {name: False for name in EXTRA_NAMES}

#: Earth magnetic field seen by the magnetometer, inertial (NED) frame, µT.
#: Pointing north and down (northern hemisphere), magnitude ≈ 49.2 µT.
_MAG_EARTH = np.array([45.0, 0.0, 20.0])
#: Deterministic per-axis hard-iron pattern; |offset| = mag_hard_iron * |B|.
_HARD_IRON_SHAPE = np.array([1.0, -0.5, 0.8])

_EZ = np.array([0.0, 0.0, 1.0])


class QuadrotorPlant:
    """Quadrotor plant implementing the ``types.Plant`` protocol.

    Parameters
    ----------
    params:
        Frozen parameter bundle.  Defaults to ``default_params(fidelity)``.
    fidelity:
        ``"paper"`` or ``"high"``; overrides ``params.fidelity`` when given.
    extras:
        Mapping of the names in :data:`EXTRA_NAMES` to bool.  Anything not given
        keeps its default (all on except ``rotor_projected_area`` for ``high``,
        all off for ``paper``).  ``paper`` refuses to switch anything on.
    integrator:
        ``"rk4"`` (default for ``high``) or ``"euler"`` (default and only
        option for ``paper``).  ``"euler"`` on a ``high`` plant is what makes
        the P12 equivalence test integrator-free.
    actuator_substeps:
        Sub-steps used to integrate the actuator sub-system ``(I, ω, d, SoC)``
        per ``dt``.  ``0`` (default) derives it from the electrical time
        constant: ``ceil(dt / (0.5·L/rm)) = 6``.  ``1`` restores the raw one-shot
        MATLAB update, which is numerically unstable at ``dt = 1 ms`` — that is
        exactly what P15 demonstrates.
    seed:
        Seed for the sensor noise stream.  Turbulence has its own seed
        (``WindParams.seed``) and can be overridden via ``reset``.
    """

    def __init__(
        self,
        params: Params | None = None,
        *,
        fidelity: str | None = None,
        extras: dict[str, bool] | None = None,
        integrator: str | None = None,
        actuator_substeps: int = 0,
        seed: int = 0,
    ) -> None:
        if fidelity is None:
            fidelity = params.fidelity if params is not None else "high"
        if fidelity not in ("paper", "high"):
            raise ValueError(f"unknown fidelity {fidelity!r}")
        self.params: Params = params if params is not None else default_params(fidelity)
        self.fidelity: str = fidelity
        self.dt: float = float(self.params.sim.dt)

        # -- effect switches ---------------------------------------------
        if fidelity == "paper":
            self.extras: dict[str, bool] = dict(PAPER_EXTRAS)
            enabled = sorted(k for k, v in (extras or {}).items() if v)
            if enabled:
                raise ValueError(
                    f"fidelity='paper' does not support enabling extras: {enabled}"
                )
        else:
            self.extras = dict(DEFAULT_EXTRAS)
            for name, value in (extras or {}).items():
                if name not in DEFAULT_EXTRAS:
                    raise ValueError(f"unknown extra {name!r}; valid: {EXTRA_NAMES}")
                self.extras[name] = bool(value)

        if integrator is None:
            integrator = "euler" if fidelity == "paper" else self.params.sim.integrator
        if integrator not in ("euler", "rk4"):
            raise ValueError(f"unknown integrator {integrator!r}")
        if fidelity == "paper" and integrator != "euler":
            raise ValueError("fidelity='paper' is explicit-Euler by definition")
        self.integrator = integrator

        # -- cached constants --------------------------------------------
        self.airframe: Airframe = self.params.airframe
        self.rotor: Rotor = self.params.rotor
        self.motor: Motor = self.params.motor
        self.esc: Esc = self.params.esc
        self.battery: Battery = self.params.battery
        self.aero: Aero = self.params.aero
        self.sensors: SensorParams = self.params.sensors

        self._g = float(self.airframe.g)
        self._rho = float(self.airframe.rho)
        self._mass = float(self.airframe.mass)
        self._R = float(self.airframe.r_rotor)
        self._A = float(self.airframe.disc_area)
        self._sigma = float(self.airframe.solidity)
        self._cl_alpha = float(self.airframe.cl_alpha)
        self._theta_tw = float(self.airframe.theta_tw)
        self._kt = float(self.rotor.kt)
        self._kq = float(self.rotor.kq)
        self._omega_min = float(self.rotor.omega_min)
        self._J = np.asarray(self.airframe.inertia, dtype=float)
        self._J_inv = np.linalg.inv(self._J)
        self._arm = np.asarray(self.airframe.arm_positions, dtype=float)      # (4,3)
        self._spin = np.asarray(self.airframe.spin_dirs, dtype=float)         # (4,)
        self._jr = float(self.airframe.rotor_inertia)
        self._w2_mixer = np.array(
            [
                [0.0, -self.airframe.d_arm * self._kt, 0.0, self.airframe.d_arm * self._kt],
                [self.airframe.d_arm * self._kt, 0.0, -self.airframe.d_arm * self._kt, 0.0],
                [self._kq, -self._kq, self._kq, -self._kq],
                [self._kt, self._kt, self._kt, self._kt],
            ]
        )
        self._cda_base = np.asarray(self.aero.drag_area(), dtype=float)
        self._cda = self._effective_drag_area()
        self._quantum = float(self.esc.pwm_quantum_us) / 1000.0

        # -- actuator sub-stepping (see module docstring) ----------------
        tau_e = float(self.motor.inductance) / float(self.motor.resistance)
        if int(actuator_substeps) <= 0:
            actuator_substeps = int(np.ceil(self.dt / (0.5 * tau_e)))
        self.actuator_substeps: int = int(min(max(int(actuator_substeps), 1), 64))

        # -- collective-pitch calibration (§2) ---------------------------
        self.rotor_theta_effective: float = self._solve_collective()

        # -- wind / sensors ----------------------------------------------
        self.wind = WindModel(self.params.wind, self.dt)
        self._period_t = max(1, int(round(float(self.params.control.h) / self.dt)))
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._delay = int(self.sensors.delay_samples) if self.extras["sensors"] else 0
        self._sample_buf: list[SensorSample] = []
        self._gyro_bias = np.zeros(3)
        self._accel_bias = (
            np.array([1.0, -1.0, 1.0], dtype=float) * self.sensors.accel_bias_mg * 1e-3 * self._g
        )
        self._baro_bias = 0.0
        self._hard_iron = (
            self.sensors.mag_hard_iron * float(np.linalg.norm(_MAG_EARTH)) * _HARD_IRON_SHAPE
        )

        # -- state ---------------------------------------------------------
        self.state = VehicleState()
        self._acc_inertial = np.array([0.0, 0.0, self._g])
        self._last: dict | None = None
        self.reset()

    # ------------------------------------------------------------------
    # Plant protocol
    # ------------------------------------------------------------------
    def reset(
        self, state: VehicleState | None = None, *, wind_seed: int | None = None
    ) -> VehicleState:
        """Reset to ``state`` (or a level, at-rest vehicle) and clear history."""
        if state is None:
            s = VehicleState()
            s.v_terminal = self._ocv(1.0)
        else:
            s = state.copy()
        s.q = quat_normalise(np.asarray(s.q, dtype=float))
        s.omega = np.asarray(s.omega, dtype=float).reshape(3).copy()
        s.v = np.asarray(s.v, dtype=float).reshape(3).copy()
        s.xi = np.asarray(s.xi, dtype=float).reshape(3).copy()
        s.rotor_omega = np.maximum(np.asarray(s.rotor_omega, dtype=float).reshape(4), 0.0)
        s.rotor_current = np.maximum(np.asarray(s.rotor_current, dtype=float).reshape(4), 0.0)
        s.esc_duty = np.clip(np.asarray(s.esc_duty, dtype=float).reshape(4), 0.0, 1.0)
        s.pwm_us = np.clip(np.asarray(s.pwm_us, dtype=float).reshape(4), PWM_MIN, PWM_MAX)
        s.soc = float(np.clip(s.soc, 0.0, 1.0))
        s.v_terminal = float(s.v_terminal)
        self.state = s

        self.wind.reset(wind_seed)
        self.wind.set_turbulence(self.extras["turbulence"])
        self._rng = np.random.default_rng(self._seed)
        self._gyro_bias = np.zeros(3)
        self._baro_bias = 0.0
        self._sample_buf = []
        self._last = self.evaluate()
        self._acc_inertial = self._last["acc_inertial"]
        self._push_sample()
        return self.state.copy()

    def step(self, pwm_us: np.ndarray) -> VehicleState:
        """Advance one ``dt`` with the four PWM commands (µs)."""
        pwm = np.asarray(pwm_us, dtype=float).reshape(4)
        if not np.all(np.isfinite(pwm)):
            raise ValueError("pwm_us must be finite")
        pwm = np.clip(pwm, PWM_MIN, PWM_MAX)
        if self.fidelity == "paper":
            self._step_paper(pwm)
        else:
            self._step_high(pwm)
        self.state.t += self.dt
        self._maybe_sample()
        return self.state.copy()

    def sense(self) -> SensorSample:
        """Latest sensor sample, already delayed and corrupted (never internals)."""
        if not self._sample_buf:
            self._push_sample()
        idx = max(0, len(self._sample_buf) - 1 - self._delay)
        sample = self._sample_buf[idx]
        # hand out a copy so a controller cannot mutate the history
        return SensorSample(
            t=sample.t,
            gyro=sample.gyro.copy(),
            accel=sample.accel.copy(),
            mag=sample.mag.copy(),
            baro_alt=float(sample.baro_alt),
        )

    def set_wind(self, steady: np.ndarray, turbulence: bool = True) -> None:
        """Set the steady wind (inertial frame, m/s) and the gust switch."""
        self.wind.set_steady(np.asarray(steady, dtype=float).reshape(3))
        self.wind.set_turbulence(bool(turbulence) and self.extras["turbulence"])

    # ------------------------------------------------------------------
    # Diagnostics (validation suite / logging only — not the Plant protocol)
    # ------------------------------------------------------------------
    def evaluate(self, state: VehicleState | None = None) -> dict:
        """Force/moment/power breakdown at ``state`` (default: current state)."""
        s = self.state if state is None else state
        if self.fidelity == "paper":
            return self._evaluate_paper(s)
        aero = self._aero_forces(s.q, s.omega, s.v, s.rotor_omega, s.xi, s.t)
        q_dot, omega_dot, acc, _ = self._body_deriv(
            s.q, s.omega, s.v, s.xi, aero["F_body"], aero["M_body"]
        )
        d_act = self._act_deriv(
            self._pack_actuators(s.rotor_omega, s.rotor_current, s.esc_duty, s.soc),
            np.clip((np.clip(s.pwm_us, PWM_MIN, PWM_MAX) - PWM_MIN) / PWM_MIN, 0.0, 1.0),
            aero["Vc"],
            aero["Vp"],
        )
        d_w, d_i, d_duty, d_soc = d_act[0:4], d_act[4:8], d_act[8:12], d_act[12]
        v_motor = s.esc_duty * self._terminal_voltage(s.rotor_current, s.soc)
        out = dict(aero)
        out.update(
            {
                "acc_inertial": acc,
                "domega": omega_dot,
                "dquat": q_dot,
                "domega_rotor": d_w,
                "dcurrent": d_i,
                "dduty": d_duty,
                "dsoc": d_soc,
                "motor_voltage": v_motor,
                "P_electrical": float(np.dot(v_motor, s.rotor_current)),
                "P_shaft": float(np.dot(aero["Q_rotor"], s.rotor_omega)),
                "P_back_emf": float(np.dot(self.motor.ke * s.rotor_omega, s.rotor_current)),
                "P_climb": float(np.sum(aero["T_rotor"] * aero["Vc"])),
                "soc": float(s.soc),
            }
        )
        return out

    # ------------------------------------------------------------------
    # Rotor blade element (public: P3 checks the published fit through it)
    # ------------------------------------------------------------------
    def blade_element_thrust(
        self, omega: float, theta: float | None = None, lam: float = 0.0
    ) -> float:
        """Blade-element thrust ``0.5 ρ A (ωR)² σ a (θ/3 − λ/2)`` for one rotor.

        With ``lam = 0`` this is the inflow-free expression that ``kT`` was
        fitted to; P3 asserts it reproduces ``kT ω²`` for the published
        ``theta_tw``.
        """
        theta = self._theta_tw if theta is None else float(theta)
        w_r = max(float(omega), 0.0) * self._R
        k_blade = 0.5 * self._rho * self._A * self._sigma * self._cl_alpha
        return float(max(k_blade * w_r**2 * (theta / 3.0 - 0.5 * float(lam)), 0.0))

    # ------------------------------------------------------------------
    # Paper fidelity: MATLAB update order
    # ------------------------------------------------------------------
    def _step_paper(self, pwm: np.ndarray) -> None:
        s = self.state
        pct = (pwm - PWM_MIN) / PWM_MIN
        u_motor = self.motor.u_max * np.sqrt(np.maximum(pct, 0.0))

        # forces use the PRE-update rotor speeds: MATLAB evaluates T(t), Gamma(t)
        # from w_motor(:,t), which is still w(k) at this point.
        tf = self._w2_mixer @ (s.rotor_omega**2)
        gamma = tf[:3]
        thrust = float(tf[3])

        # actuator sub-system: MATLAB's Euler update, sub-stepped for stability
        i0, w0 = s.rotor_current, s.rotor_omega
        h = self.dt / self.actuator_substeps
        for _ in range(self.actuator_substeps):
            i1 = np.maximum(
                i0
                + h * (u_motor - self.motor.ke * w0 - self.motor.resistance * i0)
                / self.motor.inductance,
                0.0,
            )
            w1 = np.maximum(
                w0 + h * (self.motor.km * i0 - self._kq * w0**2) / self.motor.inertia, 0.0
            )
            i0, w0 = i1, w1

        c_i2b = quat_to_rotmat(s.q)                       # built from q(k)
        f = -(c_i2b.T @ (thrust * _EZ))                  # -C_I2B' * T * e3
        q1, omega1, v1, xi1 = self._advance_body(
            s.q, s.omega, s.v, s.xi, f, gamma, "euler"
        )

        s.q = q1
        s.omega = omega1
        s.v = v1
        s.xi = xi1
        s.rotor_current = i0
        s.rotor_omega = w0
        s.pwm_us = pwm
        s.esc_duty = pct
        s.v_terminal = self.motor.u_max
        self._acc_inertial = self._g * _EZ + f / self._mass

    def _evaluate_paper(self, s: VehicleState) -> dict:
        w = np.maximum(s.rotor_omega, 0.0)
        tf = self._w2_mixer @ (w**2)
        thrust = float(tf[3])
        c_i2b = quat_to_rotmat(s.q)
        f = -(c_i2b.T @ (thrust * _EZ))
        acc = self._g * _EZ + f / self._mass
        q_dot, omega_dot, _, _ = self._body_deriv(s.q, s.omega, s.v, s.xi, f, tf[:3])
        pct = (np.clip(s.pwm_us, PWM_MIN, PWM_MAX) - PWM_MIN) / PWM_MIN
        u_motor = self.motor.u_max * np.sqrt(np.maximum(pct, 0.0))
        q_aero = self._kq * w**2
        m_reaction = np.array([0.0, 0.0, float(np.dot(self._spin, q_aero))])
        return {
            "thrust": thrust,
            "T_rotor": self._kt * w**2,
            "T_ground_effect": np.zeros(4),
            "H_force": np.zeros(4),
            "Q_rotor": q_aero,
            "v_induced": np.zeros(4),
            "Vc": np.zeros(4),
            "Vp": 0.0,
            "F_body": f,
            "F_drag_body": np.zeros(3),
            "M_body": tf[:3] + m_reaction,
            "M_rotor_arms": tf[:3],
            "M_yaw": m_reaction,
            "M_gyro": np.zeros(3),
            "M_reaction": m_reaction,
            "h_rotor": np.zeros(3),
            "v_rel_body": np.zeros(3),
            "c_i2b": c_i2b,
            "acc_inertial": acc,
            "domega": omega_dot,
            "dquat": q_dot,
            "domega_rotor": np.zeros(4),
            "dcurrent": np.zeros(4),
            "dduty": np.zeros(4),
            "dsoc": 0.0,
            "motor_voltage": u_motor,
            "v_terminal": np.full(4, self.motor.u_max),
            "P_electrical": float(np.dot(u_motor, s.rotor_current)),
            "P_shaft": float(np.dot(q_aero, w)),
            "P_back_emf": float(np.dot(self.motor.ke * w, s.rotor_current)),
            "P_climb": 0.0,
            "soc": float(s.soc),
        }

    # ------------------------------------------------------------------
    # High fidelity
    # ------------------------------------------------------------------
    def _step_high(self, pwm: np.ndarray) -> None:
        s = self.state
        pct = (pwm - PWM_MIN) / PWM_MIN
        if self.extras["esc_lag"]:
            d_cmd = np.round(pct / self._quantum) * self._quantum
            duty0 = s.esc_duty
        else:
            # lag and µs quantisation belong to the same actuator path
            d_cmd = pct
            duty0 = d_cmd
        d_cmd = np.clip(d_cmd, 0.0, 1.0)

        # 1. aerodynamics from the step-start rotor state (force held over dt,
        #    the same convention MATLAB uses with w(k))
        aero = self._aero_forces(s.q, s.omega, s.v, s.rotor_omega, s.xi, s.t)
        self._acc_inertial = self._g * _EZ + (aero["c_i2b"].T @ aero["F_body"]) / self._mass

        # 2. fast actuator sub-system
        y = self._advance_actuators(
            self._pack_actuators(s.rotor_omega, s.rotor_current, duty0, s.soc),
            d_cmd,
            aero["Vc"],
            aero["Vp"],
            self.integrator,
        )

        # 3. rigid body with the frozen force/moment
        q1, omega1, v1, xi1 = self._advance_body(
            s.q, s.omega, s.v, s.xi, aero["F_body"], aero["M_body"], self.integrator
        )

        s.q = q1
        s.omega = omega1
        s.v = v1
        s.xi = xi1
        s.rotor_omega = y[0:4]
        s.rotor_current = y[4:8]
        s.esc_duty = y[8:12]
        s.soc = float(y[12])
        s.pwm_us = pwm
        s.v_terminal = self._terminal_voltage(s.rotor_current, s.soc)

    # -- aerodynamics ---------------------------------------------------
    def _aero_forces(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
        t: float,
    ) -> dict:
        """Forces and moments at one instant (§2, §4, §7)."""
        c_i2b = quat_to_rotmat(q)                       # inertial -> body
        v_wind = self.wind.at(t)
        v_rel_b = c_i2b @ (v - v_wind)
        Vc = -v_rel_b[2]                                # > 0 when climbing
        Vp = float(np.hypot(v_rel_b[0], v_rel_b[1]))

        thrust, h_force, q_rotor, v_ind = self._rotor_aero(w, Vc, Vp)
        t_ge = self._ground_effect(thrust, -xi[2])
        f_drag = self._drag_force(v_rel_b)

        if Vp > 1e-9:
            u_p = np.array([v_rel_b[0], v_rel_b[1], 0.0]) / Vp
        else:
            u_p = np.zeros(3)
        f_rotor = np.empty((4, 3))
        f_rotor[:, 0] = -h_force * u_p[0]
        f_rotor[:, 1] = -h_force * u_p[1]
        f_rotor[:, 2] = -(thrust + t_ge)
        f_body = f_rotor.sum(axis=0) + f_drag

        m_arms = np.cross(self._arm, f_rotor).sum(axis=0)
        m_reaction = np.array([0.0, 0.0, float(np.dot(self._spin, q_rotor))])
        if self.extras["gyro_moment"]:
            h_rotor = self._jr * float(np.dot(self._spin, w)) * _EZ
            m_gyro = -np.cross(omega, h_rotor)
        else:
            h_rotor = np.zeros(3)
            m_gyro = np.zeros(3)

        return {
            "thrust": float(np.sum(thrust)),
            "T_rotor": thrust,
            "T_ground_effect": t_ge,
            "H_force": h_force,
            "Q_rotor": q_rotor,
            "v_induced": v_ind,
            "Vc": Vc,
            "Vp": Vp,
            "F_body": f_body,
            "F_drag_body": f_drag,
            "M_body": m_arms + m_reaction + m_gyro,
            "M_rotor_arms": m_arms,
            "M_yaw": m_reaction,
            "M_gyro": m_gyro,
            "M_reaction": m_reaction,
            "h_rotor": h_rotor,
            "v_rel_body": v_rel_b,
            "c_i2b": c_i2b,
        }

    def _rotor_aero(
        self, w: np.ndarray, Vc: float | np.ndarray, Vp: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-rotor ``(T, H, Q, v_i)`` for rotor speeds ``w`` and airspeed."""
        w = np.maximum(np.asarray(w, dtype=float), 0.0)
        active = w > self._omega_min
        w_safe = np.maximum(w, self._omega_min)

        if self.extras["inflow"]:
            thrust, v_ind = self._bemt(w, Vc, active)
        else:
            thrust = np.where(active, self._kt * w**2, 0.0)
            v_ind = np.zeros(4)

        mu = Vp / (w_safe * self._R)
        q_rotor = np.where(active, self._kq * w**2 * (1.0 + self.rotor.k_mu * mu**2), 0.0)
        if self.extras["h_force"]:
            h_force = np.where(
                active, self.rotor.c_h * self._rho * self._A * (w_safe * self._R) * Vp, 0.0
            )
        else:
            h_force = np.zeros(4)
        return thrust, h_force, q_rotor, v_ind

    def _bemt(
        self,
        w: np.ndarray,
        Vc: float | np.ndarray,
        active: np.ndarray,
        theta: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blade-element + momentum theory, fixed point on ``(v_i, T)`` (§2)."""
        rho, area = self._rho, self._A
        theta = self.rotor_theta_effective if theta is None else float(theta)
        w_safe = np.maximum(w, self._omega_min)
        w_r = w_safe * self._R
        vc = np.broadcast_to(np.asarray(Vc, dtype=float), w.shape).astype(float)

        # blade element: T_be = K (wR)^2 (theta/3 - lambda/2), lambda = (Vc+vi)/(wR)
        k_blade = 0.5 * rho * area * self._sigma * self._cl_alpha
        theta_3 = theta / 3.0
        two_rho_a = 2.0 * rho * area

        v_ind = np.sqrt(np.maximum(self._kt * w**2, 0.0) / two_rho_a)
        for _ in range(12):
            lam = (vc + v_ind) / w_r
            t_be = np.maximum(k_blade * w_r**2 * (theta_3 - 0.5 * lam), 0.0)
            v_h = np.sqrt(t_be / two_rho_a)
            # normal working state: momentum theory, T = 2 rho A vi (vi + Vc)
            v_mom = 0.5 * (-vc + np.sqrt(np.maximum(vc**2 + 2.0 * t_be / (rho * area), 0.0)))
            # descent: empirical vortex-ring branch, back to momentum once
            # |Vc| > 2 v_h
            v_vr = np.clip(v_h - 1.125 * vc, 0.0, 2.0 * v_h)
            use_vr = (vc < 0.0) & (np.abs(vc) <= 2.0 * v_h)
            v_ind = 0.5 * v_ind + 0.5 * np.where(use_vr, v_vr, v_mom)
        v_ind = np.maximum(v_ind, 0.0)

        lam = (vc + v_ind) / w_r
        thrust = np.maximum(k_blade * w_r**2 * (theta_3 - 0.5 * lam), 0.0)
        thrust = np.where(active, thrust, 0.0)
        v_ind = np.where(active, v_ind, 0.0)
        return thrust, v_ind

    def _ground_effect(self, thrust: np.ndarray, height: float) -> np.ndarray:
        """Thrust augmentation in ground effect (§7), blended out by ``2R``."""
        if not self.extras["ground_effect"]:
            return np.zeros(4)
        r = self._R
        z = max(float(height), 0.5 * r)
        augment = 1.0 / (1.0 - (r / (4.0 * z)) ** 2) - 1.0
        augment = min(augment, float(self.aero.ground_effect_max))
        # linear fade from full augmentation at z = R/2 to zero at z = 2R
        blend = float(np.clip((2.0 * r - float(height)) / (1.5 * r), 0.0, 1.0))
        return thrust * augment * blend

    def _drag_force(self, v_rel_b: np.ndarray) -> np.ndarray:
        """Quadratic body drag in body axes (§4)."""
        if not self.extras["body_drag"]:
            return np.zeros(3)
        return -0.5 * self._rho * self._cda * np.abs(v_rel_b) * v_rel_b

    # -- rigid body ------------------------------------------------------
    def _body_deriv(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        v: np.ndarray,
        xi: np.ndarray,
        f_body: np.ndarray,
        m_body: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(dq, dOmega, acc_inertial, dxi)`` for a frozen body force/moment."""
        c_i2b = quat_to_rotmat(q)
        acc = self._g * _EZ + (c_i2b.T @ f_body) / self._mass
        domega = self._J_inv @ (m_body - np.cross(omega, self._J @ omega))
        return quat_derivative(quat_normalise(q), omega), domega, acc, v

    def _advance_body(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        v: np.ndarray,
        xi: np.ndarray,
        f_body: np.ndarray,
        m_body: np.ndarray,
        method: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One ``dt`` of the rigid-body EOM with a frozen force/moment field."""
        dt = self.dt
        if method == "rk4":
            d1 = self._body_deriv(q, omega, v, xi, f_body, m_body)
            d2 = self._body_deriv(
                q + 0.5 * dt * d1[0],
                omega + 0.5 * dt * d1[1],
                v + 0.5 * dt * d1[2],
                xi + 0.5 * dt * d1[3],
                f_body,
                m_body,
            )
            d3 = self._body_deriv(
                q + 0.5 * dt * d2[0],
                omega + 0.5 * dt * d2[1],
                v + 0.5 * dt * d2[2],
                xi + 0.5 * dt * d2[3],
                f_body,
                m_body,
            )
            d4 = self._body_deriv(
                q + dt * d3[0],
                omega + dt * d3[1],
                v + dt * d3[2],
                xi + dt * d3[3],
                f_body,
                m_body,
            )
            q1 = quat_normalise(q + (dt / 6.0) * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0]))
            omega1 = omega + (dt / 6.0) * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1])
            v1 = v + (dt / 6.0) * (d1[2] + 2 * d2[2] + 2 * d3[2] + d4[2])
            xi1 = xi + (dt / 6.0) * (d1[3] + 2 * d2[3] + 2 * d3[3] + d4[3])
        else:
            dq, domega, acc, dxi = self._body_deriv(q, omega, v, xi, f_body, m_body)
            q1 = quat_normalise(q + dt * dq)
            omega1 = omega + dt * domega
            v1 = v + dt * acc
            xi1 = xi + dt * dxi
        return q1, omega1, v1, xi1

    # -- actuator sub-system (I, omega, duty, SoC) -----------------------
    @staticmethod
    def _pack_actuators(
        w: np.ndarray, current: np.ndarray, duty: np.ndarray, soc: float
    ) -> np.ndarray:
        y = np.empty(13)
        y[0:4] = w
        y[4:8] = current
        y[8:12] = duty
        y[12] = soc
        return y

    def _act_deriv(
        self, y: np.ndarray, d_cmd: np.ndarray, Vc: float | np.ndarray, Vp: float
    ) -> np.ndarray:
        """Derivative of ``(ω, I, duty, SoC)`` with frozen airspeed."""
        w = np.maximum(y[0:4], 0.0)
        current = np.maximum(y[4:8], 0.0)
        duty = np.clip(y[8:12], 0.0, 1.0)
        soc = float(np.clip(y[12], 0.0, 1.0))

        w_safe = np.maximum(w, self._omega_min)
        active = w > self._omega_min
        mu = Vp / (w_safe * self._R)
        q_load = np.where(active, self._kq * w**2 * (1.0 + self.rotor.k_mu * mu**2), 0.0)

        v_term = self._terminal_voltage(current, soc)
        v_motor = duty * v_term

        d_current = (v_motor - self.motor.ke * w - self.motor.resistance * current) / (
            self.motor.inductance
        )
        d_current = np.where((current <= 0.0) & (d_current < 0.0), 0.0, d_current)
        d_w = (self.motor.km * current - q_load) / self.motor.inertia
        d_w = np.where((w <= 0.0) & (d_w < 0.0), 0.0, d_w)
        if self.extras["esc_lag"]:
            d_duty = (d_cmd - duty) / max(float(self.esc.tau), 1e-9)
        else:
            d_duty = np.zeros(4)
        if self.extras["battery_sag"]:
            # Q_pack is in Ah, dt in s -> the 3600 s/h factor is required (§3
            # omits it; see the module docstring).
            d_soc = -float(np.dot(v_motor, current)) / (
                self.battery.capacity_ah * self.battery.v_nom * 3600.0
            )
        else:
            d_soc = 0.0

        out = np.empty(13)
        out[0:4] = d_w
        out[4:8] = d_current
        out[8:12] = d_duty
        out[12] = d_soc
        return out

    def _advance_actuators(
        self,
        y0: np.ndarray,
        d_cmd: np.ndarray,
        Vc: float | np.ndarray,
        Vp: float,
        method: str,
    ) -> np.ndarray:
        """Integrate ``(ω, I, duty, SoC)`` over one ``dt`` in ``actuator_substeps`` steps."""
        h = self.dt / self.actuator_substeps
        y = np.array(y0, dtype=float)
        for _ in range(self.actuator_substeps):
            if method == "rk4":
                k1 = self._act_deriv(y, d_cmd, Vc, Vp)
                k2 = self._act_deriv(y + 0.5 * h * k1, d_cmd, Vc, Vp)
                k3 = self._act_deriv(y + 0.5 * h * k2, d_cmd, Vc, Vp)
                k4 = self._act_deriv(y + h * k3, d_cmd, Vc, Vp)
                y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            else:
                y = y + h * self._act_deriv(y, d_cmd, Vc, Vp)
            y[0:4] = np.maximum(y[0:4], 0.0)
            y[4:8] = np.maximum(y[4:8], 0.0)
            y[8:12] = np.clip(y[8:12], 0.0, 1.0)
            y[12] = np.clip(y[12], 0.0, 1.0)
        return y

    # -- calibration and constants ---------------------------------------
    def _solve_collective(self) -> float:
        """Re-fit ``theta_tw`` so that hover BEMT matches the measured ``kT`` (§2)."""
        if self.fidelity != "high" or not bool(self.rotor.calibrate):
            return self._theta_tw
        w_hover = float(self.params.w_hover)
        target = self._kt * w_hover**2
        w4 = np.full(4, w_hover)
        active = np.full(4, True)

        def residual(theta: float) -> float:
            thrust, _ = self._bemt(w4, np.zeros(4), active, theta=theta)
            return float(thrust[0]) - target

        lo, hi = 1e-6, 0.5 * np.pi
        f_lo, f_hi = residual(lo), residual(hi)
        if not (f_lo < 0.0 < f_hi):
            raise RuntimeError(
                "collective calibration failed to bracket a root: "
                f"f({lo})={f_lo}, f({hi})={f_hi}"
            )
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-12:
                break
        return float(0.5 * (lo + hi))

    def _effective_drag_area(self) -> np.ndarray:
        cda = self._cda_base.copy()
        if self.extras.get("rotor_projected_area", False):
            cda[2] += 4.0 * self._A
        return cda

    def _ocv(self, soc: float) -> float:
        return float(self.battery.open_circuit_voltage(soc))

    def _terminal_voltage(self, current: np.ndarray, soc: float) -> float:
        if not self.extras["battery_sag"]:
            return float(self.motor.u_max)
        return max(self._ocv(soc) - self.battery.r_int * float(np.sum(current)), 0.0)

    # ------------------------------------------------------------------
    # Sensors (§5)
    # ------------------------------------------------------------------
    def _maybe_sample(self) -> None:
        n = int(round(self.state.t / self.dt))
        if n % self._period_t == 0:
            self._push_sample()

    def _push_sample(self) -> None:
        s = self.state
        c_i2b = quat_to_rotmat(s.q)
        specific_force = self._acc_inertial - self._g * _EZ
        accel_clean = c_i2b @ specific_force
        mag_clean = c_i2b @ _MAG_EARTH
        baro_clean = -s.xi[2]

        if not self.extras["sensors"]:
            sample = SensorSample(
                t=s.t,
                gyro=s.omega.copy(),
                accel=accel_clean,
                mag=mag_clean,
                baro_alt=float(baro_clean),
            )
        else:
            h = float(self.params.control.h)
            sp = self.sensors
            sigma_g = np.deg2rad(sp.gyro_noise_dps)
            sigma_bg = np.deg2rad(sp.gyro_bias_rw_dps) * np.sqrt(h)
            sigma_a = sp.accel_noise_mg * 1e-3 * self._g
            sigma_m = sp.mag_noise_ut
            sigma_baro = sp.baro_noise_m
            sigma_baro_rw = sp.baro_bias_drift_m_sqrt_s * np.sqrt(h)

            self._gyro_bias = self._gyro_bias + sigma_bg * self._rng.standard_normal(3)
            self._baro_bias = self._baro_bias + sigma_baro_rw * self._rng.standard_normal()

            sample = SensorSample(
                t=s.t,
                gyro=s.omega + self._gyro_bias + sigma_g * self._rng.standard_normal(3),
                accel=accel_clean + self._accel_bias + sigma_a * self._rng.standard_normal(3),
                mag=mag_clean + self._hard_iron + sigma_m * self._rng.standard_normal(3),
                baro_alt=float(
                    baro_clean + self._baro_bias + sigma_baro * self._rng.standard_normal()
                ),
            )

        self._sample_buf.append(sample)
        keep = self._delay + 1
        if len(self._sample_buf) > keep:
            del self._sample_buf[: len(self._sample_buf) - keep]

    # ------------------------------------------------------------------
    # Hover trim (steady state; used by the validation suite)
    # ------------------------------------------------------------------
    def trim_hover(self, soc: float = 1.0) -> dict:
        """Solve the steady-state hover operating point.

        Returns ``pwm_us``, ``percent``, ``rotor_omega``, ``rotor_current``,
        ``esc_duty``, ``v_terminal``, ``thrust`` and a ready-to-use ``state``.

        The rotor speed solves ``km I = Q(ω)`` with ``I = (d V_term − ke ω)/rm``
        and ``V_term = V_oc − R_int ΣI`` (when the battery model is on); the
        throttle is then bisected so that ``Σ T = m g``.
        """
        target = self._mass * self._g
        paper = self.fidelity == "paper"

        def duty_of(percent: float) -> float:
            """Physical duty actually commanded for a throttle request.

            With the ESC model on the command is quantised to 1 µs, and the
            steady state must be solved for the *quantised* duty — otherwise the
            returned state is not a true fixed point and the vehicle drifts off
            the trim point by the quantisation error.
            """
            d = float(np.clip(percent, 0.0, 1.0))
            if not paper and self.extras["esc_lag"]:
                d = float(np.round(d / self._quantum) * self._quantum)
            return float(np.clip(d, 0.0, 1.0))

        def rotor_state(duty: float) -> tuple[float, float, float]:
            """``(omega, current, v_terminal)`` at a fixed duty."""
            duty = float(duty)
            if paper:
                v_open = self.motor.u_max * np.sqrt(duty)
                v_term = self.motor.u_max

                def current_of(w: float) -> float:
                    return (v_open - self.motor.ke * w) / self.motor.resistance

            else:
                v_open = self._ocv(soc) if self.extras["battery_sag"] else self.motor.u_max
                v_term = v_open
                r_int = self.battery.r_int if self.extras["battery_sag"] else 0.0

                def current_of(w: float) -> float:
                    # I = (d V_term - ke w)/rm with V_term = Voc - Rint * 4 I
                    return (duty * v_open - self.motor.ke * w) / (
                        self.motor.resistance + 4.0 * duty * r_int
                    )

            lo, hi = 0.0, 5000.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if self.motor.km * current_of(mid) - self._kq * mid**2 > 0.0:
                    lo = mid
                else:
                    hi = mid
            w = 0.5 * (lo + hi)
            cur = max(current_of(w), 0.0)
            if not paper and self.extras["battery_sag"]:
                v_term = max(v_open - self.battery.r_int * 4.0 * cur, 0.0)
            return w, cur, v_term

        def thrust_of(duty: float) -> float:
            w, _, _ = rotor_state(duty)
            if not paper and self.extras["inflow"]:
                thrust, _ = self._bemt(np.full(4, w), np.zeros(4), np.full(4, True))
            else:
                thrust = np.full(4, self._kt * w**2)
            return float(np.sum(thrust))

        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if thrust_of(duty_of(mid)) < target:
                lo = mid
            else:
                hi = mid
        percent = 0.5 * (lo + hi)

        # With a quantised throttle no duty reproduces m*g exactly; pick the
        # closest one instead of the bisection boundary (worst case is still the
        # 1 µs step, ~0.5 % in thrust at hover, which is why P2 disables the ESC).
        duty0 = duty_of(percent)
        candidates = {duty0}
        if not paper and self.extras["esc_lag"]:
            candidates |= {duty0 - self._quantum, duty0 + self._quantum}
        duty = min(
            (d for d in candidates if 0.0 <= d <= 1.0),
            key=lambda d: abs(thrust_of(d) - target),
        )
        w, cur, v_term = rotor_state(duty)
        pwm = PWM_MIN + duty * PWM_MIN

        state = VehicleState(
            t=0.0,
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3),
            v=np.zeros(3),
            xi=np.zeros(3),
            rotor_omega=np.full(4, w),
            rotor_current=np.full(4, cur),
            esc_duty=np.full(4, duty),
            pwm_us=np.full(4, pwm),
            soc=float(soc),
            v_terminal=float(v_term),
        )
        return {
            "pwm_us": float(pwm),
            "percent": float(percent),
            "rotor_omega": float(w),
            "rotor_current": float(cur),
            "esc_duty": float(duty),
            "v_terminal": float(v_term),
            "thrust": thrust_of(duty),
            "state": state,
        }


def make_plant(
    fidelity: str = "high", *, extras: dict[str, bool] | None = None, **kwargs
) -> QuadrotorPlant:
    """Convenience constructor mirroring :func:`params.default_params`."""
    return QuadrotorPlant(
        params=default_params(fidelity), fidelity=fidelity, extras=extras, **kwargs
    )
