"""Rotor actuation chain for the MuJoCo world: PWM -> ESC -> motor -> thrust.

This module is the *simplified* version of the actuator subsystem in
:mod:`pistr_sim.plant` (see ``docs/PLANT_MODEL.md`` §3), **not** a bit-for-bit
port.  The differences are deliberate and are listed here so nobody has to
reverse-engineer them:

* The electrical dynamics are eliminated (``L dI/dt`` dropped): the motor
  current is the algebraic back-EMF solution
  ``I = (duty * V_term - ke * omega) / R``.  The resulting mechanical equation
  ``J d(omega)/dt = km * I - Q_load`` is a **first-order** lag in ``omega``,
  which is what the contract asks for.  ``params.motor.inductance`` is
  therefore unused (the plant's ``tau_e = L/R = 0.4 ms`` is much faster than
  the 20 ms ESC lag anyway).
* The battery is a Thevenin source: ``V_term = OCV(SoC) - R_int * sum(I)``.
  The four motor currents are coupled through the pack resistance, and the
  coupled 4x4 linear system is solved in closed form (rank-one update), so the
  hover point matches ``plant.py``'s high-fidelity ``trim_hover()`` exactly
  (1394 us at SoC = 1, not the 1183 us of ``params.pwm_hover_us``, which
  assumes a constant ``motor.u_max`` supply).
* The ESC is a first-order lag on the duty cycle
  (``d(duty)/dt = (duty_cmd - duty) / params.esc.tau``) plus the 1 us PWM
  quantisation.  ``params.esc.enabled`` / ``params.battery.enabled`` switch the
  corresponding effects off.
* Aerodynamics are ``T = kt * omega^2`` and ``Q = kq * omega^2 * (1 + k_mu *
  mu^2)`` -- the same expressions as ``plant.py`` with BEMT inflow **off**.
  There is no induced-inflow thrust loss and no H-force; that is a known
  simplification (see ``docs/3D_ENVIRONMENT.md`` §7), not an oversight.
* The rotor gyroscopic moment ``-omega x h_rotor`` is not included here: the
  pack does not know the body rate.  Add it in the bridge if wanted
  (it is ~1e-4 N m).

Frame conventions (normative, mirrored in ``docs/3D_ENVIRONMENT.md`` §2)
-----------------------------------------------------------------------
The MuJoCo body frame is FLU (x forward, y left, z **up**); the repository's
body frame is NED (x forward, y right, z **down**).  ``S = diag(1, -1, -1)``
maps NED -> FLU and is a *proper* rotation (det = +1, 180 deg about x), so
forces *and* moments map with the same ``S``.

* ``step()`` returns ``thrust[4]`` **along FLU body +z** (positive up) and
  ``torque_z[4] = Q_i = kq * omega_i^2 * (1 + k_mu * mu^2)``, the *unsigned*
  profile-drag torque -- the same quantity as ``plant.py``'s ``Q_rotor``.
* ``wrench()`` returns ``(F_body[3], M_body[3])`` in the **FLU body frame,
  applied at the body-frame origin**, ready to be rotated to the world frame and
  written into ``mjData.xfrc_applied``.  In particular the yaw reaction is
  ``M_z = -sum(spin_dirs * Q)``: ``spin_dirs`` is defined with the NED z axis
  pointing down, hence the sign flip in FLU.  Rotor moments use
  ``r_i = S @ arm_positions[i]``.
* ``tomj = S @ to_ned``; ``F_z_mj = -F_z_ned``, ``M_z_mj = -M_z_ned``.
"""

from __future__ import annotations

import math

import numpy as np

from ..params import Params, default_params
from ..types import PWM_MAX, PWM_MIN, PWM_SPAN

__all__ = [
    "RotorPack",
    "trim_hover",
    "hover_pwm",
    "spin_dirs_flu",
    "S_NED_TO_FLU",
]

#: NED body frame -> MuJoCo FLU body frame (a 180 deg rotation about x).
S_NED_TO_FLU = np.diag([1.0, -1.0, -1.0])

#: Largest number of internal sub-steps per ``step()`` call (numerical guard).
_MAX_SUBSTEPS = 512


def spin_dirs_flu(params: Params) -> np.ndarray:
    """`Airframe.spin_dirs` re-expressed in the FLU (MuJoCo) body frame.

    ``spin_dirs`` is signed for the NED body z axis (which points down), so in
    FLU the sign flips: ``+1`` (CW seen from above) means a *negative* rotation
    about FLU +z and a *negative* yaw reaction.
    """
    return -np.asarray(params.airframe.spin_dirs, dtype=float).reshape(4)


class RotorPack:
    """Rotor states of **one** vehicle: ``omega``, ``esc_duty``, battery SoC.

    Parameters
    ----------
    params:
        The single source of physical truth.  Defaults to
        ``default_params("high")``.

    Attributes (read-only by convention; ``omega`` is writable for tests/trim)
    ------------------------------------------------------------------------
    omega : (4,) ndarray
        Rotor speed, rad/s, in the firmware mixer order.
    esc_duty : (4,) ndarray
        Physical duty cycle after the ESC lag, in [0, 1].
    current : (4,) ndarray
        Motor current, A (clipped at zero -- no regeneration).
    soc : float
        Battery state of charge, [0, 1].
    pwm_us : (4,) ndarray
        Last PWM command, microseconds (clipped to [1000, 2000]).
    v_terminal : float
        Pack terminal voltage, V.
    thrust : (4,) ndarray
        Last per-rotor thrust, N, along FLU body +z.
    torque_z : (4,) ndarray
        Last per-rotor profile torque ``Q_i``, N m, unsigned.

    Usage
    -----
    ``pack.reset(pwm_us=trim["pwm_us"])`` starts at the steady state of that
    command (a trim start); ``pack.reset()`` starts with stopped rotors.
    ``pack.step(pwm_us, dt)`` advances the chain; ``pack.wrench()`` then returns
    the body wrench to apply through ``mjData.xfrc_applied``.
    """

    #: integration sub-step target: at most this fraction of the fastest lag
    _SUBSTEP_FRACTION = 0.25

    def __init__(self, params: Params | None = None) -> None:
        self.params: Params = params if params is not None else default_params("high")
        a, r, m = self.params.airframe, self.params.rotor, self.params.motor
        e, b = self.params.esc, self.params.battery

        self.airframe = a
        self.rotor = r
        self.motor = m
        self.esc = e
        self.battery = b

        self._mass = float(a.mass)
        self._g = float(a.g)
        self._kt = float(r.kt)
        self._kq = float(r.kq)
        self._k_mu = float(r.k_mu)
        self._omega_min = float(r.omega_min)
        self._r_rotor = float(a.r_rotor)
        self._r_int = float(b.r_int)
        self._capacity_ah = float(b.capacity_ah)
        self._v_nom = float(b.v_nom)
        self._ke = float(m.ke)
        self._km = float(m.km)
        self._resistance = float(m.resistance)
        self._inertia = float(m.inertia)
        self._u_max = float(m.u_max)
        self._quantum = float(e.pwm_quantum_us) / PWM_SPAN

        #: Rotor positions in the FLU body frame (S applied to the NED values).
        self.arm_positions_flu = S_NED_TO_FLU @ np.asarray(
            a.arm_positions, dtype=float
        ).T
        self.arm_positions_flu = np.ascontiguousarray(self.arm_positions_flu.T)
        #: Signed yaw reaction per unit ``Q`` in FLU (+1 = positive FLU yaw).
        self.spin_flu = spin_dirs_flu(self.params)

        #: Hover rotor speed from the *paper* coefficients (start of the trim
        #: bisection; the battery model moves it by ~0.1 %).
        self._w_hover_nom = float(self.params.w_hover)

        self._v_rel = np.zeros(3)
        self.reset()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def reset(self, *, pwm_us=None, soc: float = 1.0) -> None:
        """Reset the pack.

        ``pwm_us=None`` starts with stopped rotors and zero duty;
        ``pwm_us=<(4,)>`` starts at the **steady state** of that command
        (duty, current and ``omega``), which is what a trim start means.
        ``soc`` is clipped to [0, 1].
        """
        soc = float(np.clip(soc, 0.0, 1.0))
        self.soc = soc
        self._v_rel = np.zeros(3)
        if pwm_us is None:
            self.pwm_us = np.full(4, float(PWM_MIN))
            self.esc_duty = np.zeros(4)
            self.omega = np.zeros(4)
            self.current = np.zeros(4)
            self.v_terminal = self._ocv(soc)
        else:
            pwm = self._clip_pwm(pwm_us)
            self.pwm_us = pwm
            try:
                duty = self._duty_command(pwm)
            except ValueError:
                raise
            self.esc_duty = duty.copy()
            w, i, v = self._steady_state(duty, soc)
            self.omega = w
            self.current = i
            self.v_terminal = v
        self._refresh_aero()

    #: -- surface for M3/M4 -------------------------------------------------
    @property
    def mass(self) -> float:
        return self._mass

    @property
    def weight(self) -> float:
        """``m * g``, N -- the hover thrust target."""
        return self._mass * self._g

    # ------------------------------------------------------------------
    # command chain
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_pwm(pwm_us) -> np.ndarray:
        pwm = np.asarray(pwm_us, dtype=float)
        if pwm.ndim == 0:
            pwm = np.full(4, float(pwm))
        pwm = pwm.reshape(4)
        if not np.all(np.isfinite(pwm)):
            raise ValueError("pwm_us must be finite")
        return np.clip(pwm, PWM_MIN, PWM_MAX)

    def _duty_command(self, pwm_us: np.ndarray) -> np.ndarray:
        """PWM (us) -> commanded duty, including the 1 us quantisation."""
        percent = (np.asarray(pwm_us, dtype=float) - PWM_MIN) / PWM_SPAN
        percent = np.clip(percent, 0.0, 1.0)
        if self.esc.enabled:
            percent = np.round(percent / self._quantum) * self._quantum
        return np.clip(percent, 0.0, 1.0)

    def _ocv(self, soc: float) -> float:
        if self.battery.enabled:
            return float(self.battery.open_circuit_voltage(soc))
        return self._u_max

    def _solve_currents(
        self, duty: np.ndarray, omega: np.ndarray, soc: float
    ) -> tuple[np.ndarray, float]:
        """Motor currents and pack terminal voltage at frozen ``omega``.

        With the battery on, ``I_i = (d_i * (OCV - R_int * sum(I)) - ke *
        omega_i) / R``.  Writing ``a_i = (d_i * OCV - ke * omega_i) / R`` and
        ``c_i = R_int * d_i / R`` the 4x4 system is a rank-one update of the
        identity: ``I_i = a_i - c_i * (sum(a) / (1 + sum(c)))``.
        """
        duty = np.asarray(duty, dtype=float)
        omega = np.asarray(omega, dtype=float)
        v_open = self._ocv(soc)
        if self.battery.enabled:
            a = (duty * v_open - self._ke * omega) / self._resistance
            c = self._r_int * duty / self._resistance
            total = float(a.sum() / (1.0 + float(c.sum())))
            current = a - c * total
            v_term = max(v_open - self._r_int * total, 0.0)
        else:
            current = (duty * v_open - self._ke * omega) / self._resistance
            v_term = v_open
        current = np.maximum(current, 0.0)
        return current, v_term

    def _steady_state(
        self, duty: np.ndarray, soc: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Rotor speed fixed point ``km * I = kq * omega^2`` for a fixed duty."""
        duty = np.asarray(duty, dtype=float)
        omega = np.zeros(4)
        current = np.zeros(4)
        v_open = self._ocv(soc)
        for k in range(4):
            d = float(duty[k])
            if d <= 0.0:
                continue

            def residual(w: float, d: float = d) -> float:
                if self.battery.enabled:
                    i = (d * v_open - self._ke * w) / (
                        self._resistance + 4.0 * d * self._r_int
                    )
                else:
                    i = (d * v_open - self._ke * w) / self._resistance
                return self._km * i - self._kq * w * w

            lo, hi = 0.0, 5000.0
            if residual(lo) <= 0.0:
                continue
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if residual(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            w = 0.5 * (lo + hi)
            omega[k] = w
            if self.battery.enabled:
                current[k] = max(
                    (d * v_open - self._ke * w)
                    / (self._resistance + 4.0 * d * self._r_int),
                    0.0,
                )
            else:
                current[k] = max(
                    (d * v_open - self._ke * w) / self._resistance, 0.0
                )
        if self.battery.enabled:
            v_term = max(v_open - self._r_int * float(current.sum()), 0.0)
        else:
            v_term = v_open
        return omega, current, v_term

    # ------------------------------------------------------------------
    # aerodynamics
    # ------------------------------------------------------------------
    def _aero(self) -> tuple[np.ndarray, np.ndarray]:
        """``(thrust, Q)`` for the current ``omega`` and relative airspeed."""
        omega = np.asarray(self.omega, dtype=float)
        active = omega > self._omega_min
        omega_safe = np.maximum(omega, self._omega_min)
        v_plane = float(np.linalg.norm(np.asarray(self._v_rel, dtype=float)[:2]))
        mu = v_plane / (omega_safe * self._r_rotor)
        thrust = np.where(active, self._kt * omega**2, 0.0)
        torque = np.where(
            active, self._kq * omega**2 * (1.0 + self._k_mu * mu**2), 0.0
        )
        return thrust, torque

    def _refresh_aero(self) -> None:
        self.thrust, self.torque_z = self._aero()

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------
    def step(self, pwm_us, dt: float, *, v_rel_body=None):
        """Advance the actuator chain by ``dt`` seconds.

        Parameters
        ----------
        pwm_us:
            Four PWM commands in microseconds (clipped to [1000, 2000]).
        dt:
            Time step in seconds.  ``dt <= 0`` only refreshes the outputs.
        v_rel_body:
            Vehicle velocity relative to the air, **in the FLU body frame**
            (m/s), or ``None`` for still air.  Only the in-plane component is
            used: ``mu = V_xy / (omega * r_rotor)``, which modifies the profile
            torque (``k_mu``).  It does **not** reduce the thrust -- there is no
            BEMT inflow in this simplified model.

        Returns
        -------
        (thrust, torque_z):
            ``thrust[4]`` N along FLU body +z; ``torque_z[4]`` unsigned profile
            torque ``Q_i`` in N m (see the module docstring for the sign of the
            yaw reaction).
        """
        pwm = self._clip_pwm(pwm_us)
        dt = float(dt)
        if not math.isfinite(dt):
            raise ValueError("dt must be finite")
        if dt < 0.0:
            raise ValueError("dt must be non-negative")

        self.pwm_us = pwm
        if v_rel_body is None:
            self._v_rel = np.zeros(3)
        else:
            v = np.asarray(v_rel_body, dtype=float).reshape(3)
            if not np.all(np.isfinite(v)):
                raise ValueError("v_rel_body must be finite")
            self._v_rel = v

        if dt == 0.0:
            self._refresh_aero()
            return self.thrust.copy(), self.torque_z.copy()

        duty_cmd = self._duty_command(pwm)
        n_sub = self._n_substeps(dt)
        h = dt / n_sub

        tau_esc = float(self.esc.tau)
        esc_enabled = bool(self.esc.enabled) and tau_esc > 0.0
        for _ in range(n_sub):
            # ESC: exact exponential relaxation of the duty cycle.
            if esc_enabled:
                blend = 1.0 - math.exp(-h / tau_esc)
                self.esc_duty = self.esc_duty + (duty_cmd - self.esc_duty) * blend
            else:
                self.esc_duty = duty_cmd.copy()
            self.esc_duty = np.clip(self.esc_duty, 0.0, 1.0)

            current, v_term = self._solve_currents(
                self.esc_duty, self.omega, self.soc
            )
            self.current = current
            self.v_terminal = v_term

            thrust, torque = self._aero()
            # Semi-implicit Euler on the mechanical degree of freedom.
            dw = (self._km * current - torque) / self._inertia
            self.omega = np.maximum(self.omega + h * dw, 0.0)

            if self.battery.enabled:
                v_motor = self.esc_duty * v_term
                energy = float(np.dot(v_motor, current))  # W
                self.soc = float(
                    np.clip(
                        self.soc
                        - h
                        * energy
                        / (self._capacity_ah * self._v_nom * 3600.0),
                        0.0,
                        1.0,
                    )
                )

        self._refresh_aero()
        return self.thrust.copy(), self.torque_z.copy()

    def _n_substeps(self, dt: float) -> int:
        """Sub-steps so that each is <= 25 % of the fastest actuator lag."""
        damping = self._km * self._ke / self._resistance + 2.0 * self._kq * (
            self._w_hover_nom
        )
        tau_motor = self._inertia / max(damping, 1e-12)
        tau = tau_motor
        if self.esc.enabled and self.esc.tau > 0.0:
            tau = min(tau, float(self.esc.tau))
        n = int(math.ceil(dt / (self._SUBSTEP_FRACTION * max(tau, 1e-9))))
        return int(min(max(n, 1), _MAX_SUBSTEPS))

    # ------------------------------------------------------------------
    # wrench
    # ------------------------------------------------------------------
    def wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """Whole-vehicle wrench in the FLU body frame, at the body origin.

        Returns ``(F_body, M_body)`` with ``F_body = (0, 0, sum(T))`` and
        ``M_body = sum(r_i x (0, 0, T_i)) + (0, 0, -sum(spin_i * Q_i))``,
        where ``r_i = S @ arm_positions[i]``.  Rotate the pair to the world
        frame before writing ``mjData.xfrc_applied`` (which is world-frame and
        applied at the body CoM -- our CoM is the body-frame origin).
        """
        thrust, torque = self._aero()
        r = self.arm_positions_flu
        f_rotor = np.zeros((4, 3))
        f_rotor[:, 2] = thrust
        force = f_rotor.sum(axis=0)
        moment = np.cross(r, f_rotor).sum(axis=0)
        moment[2] += float(np.dot(self.spin_flu, torque))
        return force, moment

    # ------------------------------------------------------------------
    # hover trim
    # ------------------------------------------------------------------
    def trim_hover(self, soc: float = 1.0) -> dict:
        """Steady-state hover operating point (fixed point of this chain).

        Solves ``km * I = Q(omega)`` together with ``sum(T) = m g`` and the
        battery sag, exactly like ``QuadrotorPlant.trim_hover()`` for
        ``fidelity="high"`` -- so at ``soc=1`` it returns the same ~1394 us
        (not ``params.pwm_hover_us`` = 1183 us, which assumes a constant
        11.1 V supply).  The 1 us ESC quantisation is respected: the solution
        is the *achievable* duty whose thrust is closest to ``m g``, so the
        returned point is a true fixed point of :meth:`step` and not merely the
        continuous solution.

        Returns
        -------
        dict with keys ``pwm_us``, ``percent``, ``esc_duty``, ``omega``,
        ``current``, ``thrust``, ``torque_z``, ``v_terminal``, ``soc``.
        """
        soc = float(np.clip(soc, 0.0, 1.0))
        target = self._mass * self._g

        def state_of(duty_value: float):
            duty = np.full(4, float(np.clip(duty_value, 0.0, 1.0)))
            w, i, v = self._steady_state(duty, soc)
            return float(np.sum(self._kt * w**2)), w, i, v

        # 1) continuous solution of sum(T) = m*g
        if state_of(1.0)[0] < target:
            raise ValueError(
                "full throttle cannot produce m*g with these params "
                f"({state_of(1.0)[0]:.3f} N < {target:.3f} N)"
            )
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if state_of(mid)[0] < target:
                lo = mid
            else:
                hi = mid
        duty_star = 0.5 * (lo + hi)

        # 2) nearest *achievable* duty (the ESC quantises the command to 1 us)
        if self.esc.enabled and self._quantum > 0.0:
            base = round(duty_star / self._quantum) * self._quantum
            candidates = {
                float(np.clip(base + k * self._quantum, 0.0, 1.0))
                for k in (-2, -1, 0, 1, 2)
            }
        else:
            candidates = {float(duty_star)}
        best_duty = min(
            sorted(candidates), key=lambda d: abs(state_of(d)[0] - target)
        )
        thrust, w, current, v_term = state_of(best_duty)
        duty = np.full(4, best_duty)
        pwm_us = float(PWM_MIN + best_duty * PWM_SPAN)
        torque = np.where(w > self._omega_min, self._kq * w**2, 0.0)
        return {
            "pwm_us": pwm_us,
            "percent": best_duty,
            "esc_duty": duty,
            "omega": w,
            "current": current,
            "thrust": thrust,
            "torque_z": torque,
            "v_terminal": v_term,
            "soc": soc,
        }


def trim_hover(params: Params | None = None, soc: float = 1.0) -> dict:
    """Convenience wrapper: ``RotorPack(params).trim_hover(soc)``."""
    return RotorPack(params).trim_hover(soc=soc)


def hover_pwm(params: Params | None = None, soc: float = 1.0) -> float:
    """Hover PWM command in microseconds, solved by :func:`trim_hover`."""
    return float(trim_hover(params, soc)["pwm_us"])
