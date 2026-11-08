"""Physical and controller parameters.

``Params.from_matlab()`` is a faithful port of
``sim/matlab/reduced_overshoot/Load_SystemParameters.m`` — every number below that is
marked ``[matlab]`` comes from there and must not be "improved". Parameters marked
``[realistic]`` only exist at ``fidelity="high"`` (see ``docs/PLANT_MODEL.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from .types import FRAME_QUADP, FRAME_QUADX

Fidelity = Literal["paper", "high"]


@dataclass(frozen=True)
class Airframe:
    """Mass, geometry and inertia of the F450-class vehicle. [matlab]"""

    g: float = 9.81
    rho: float = 1.205
    d_arm: float = 0.225          # arm length, offline measurement
    frame: str = FRAME_QUADP      # matches firmware ``#define QuadP``
    m_cog: float = 0.709          # CoG body mass, offline measurement
    r_cog: float = 0.225 * 0.5    # "through intuition" in the original
    h_cog: float = 0.05           # "through intuition"
    m_rotor: float = 0.013
    r_rotor: float = 0.120
    c_rotor: float = 0.025
    h_rotor: float = 0.045
    m_motor: float = 0.056
    r_motor: float = 0.027 / 2
    h_motor: float = 0.03
    n_blades: int = 2
    theta_tw: float = 15 * np.pi / 180   # collective pitch [matlab]
    cl_alpha: float = 3.320              # [matlab] fitted so kT matches measurements
    cd_zero: float = 0.220               # [matlab] fitted so kT matches measurements
    cl_alpha_scale: float = 3.320        # alias used by the BEMT equations

    @property
    def mass(self) -> float:
        return self.m_cog + 4 * self.m_rotor + 4 * self.m_motor

    @property
    def disc_area(self) -> float:
        return np.pi * self.r_rotor**2

    @property
    def solidity(self) -> float:
        return self.n_blades * self.c_rotor / (np.pi * self.r_rotor)

    @property
    def rotor_inertia(self) -> float:
        """Rotor + blade inertia seen by the motor (dominated by the motor)."""
        return 1.25e-5  # [matlab] J_motor, used as J_r for gyroscopic moments

    @property
    def inertia(self) -> np.ndarray:
        """Full inertia tensor, ported verbatim from the MATLAB source.

        Note the original sets ``Iyz = Ixz``; that is not physically symmetric for a
        quadrotor but it is what the published results were produced with, so it is
        preserved here and flagged in ``docs/PLANT_MODEL.md``.
        """
        m, d = self.m_cog, self.d_arm
        r, h = self.r_cog, self.h_cog
        mr, rr, cr, hr = self.m_rotor, self.r_rotor, self.c_rotor, self.h_rotor
        mm, rm, hm = self.m_motor, self.r_motor, self.h_motor

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
        return np.array([[ixx, 0.0, ixz], [0.0, iyy, ixz], [ixz, ixz, izz]])

    @property
    def arm_positions(self) -> np.ndarray:
        """Rotor positions in the body frame, ordered as the firmware mixer."""
        d = self.d_arm
        if self.frame == FRAME_QUADX:
            s = d / np.sqrt(2.0)
            return np.array([[s, s, 0.0], [-s, s, 0.0], [-s, -s, 0.0], [s, -s, 0.0]])
        return np.array([[d, 0.0, 0.0], [0.0, d, 0.0], [-d, 0.0, 0.0], [0.0, -d, 0.0]])

    @property
    def spin_dirs(self) -> np.ndarray:
        """``+1`` for **CW** (viewed from above), ``-1`` for CCW.

        Rotors 0 and 2 spin CW, rotors 1 and 3 spin CCW, matching the firmware
        comment block in ``Quadrotor_Mode_II_Arduino_Due.ino``. This sign enters
        the yaw reaction torque and the gyroscopic momentum, so it is normative:
        see ``docs/PLANT_MODEL.md`` §4.
        """
        return np.array([1.0, -1.0, 1.0, -1.0])


@dataclass(frozen=True)
class Rotor:
    """Rotor aerodynamic coefficients. [matlab] derivations, BEMT form in PLANT_MODEL."""

    kt: float = 0.0    # thrust coefficient, T = kt * omega^2 at hover
    kh: float = 0.0    # in-plane (H-force) coefficient of the original model
    kq: float = 0.0    # profile torque coefficient, Q = kq * omega^2
    c_h: float = 0.03  # [realistic] empirical H-force coefficient
    k_mu: float = 3.0  # [realistic] advance-ratio torque growth
    omega_min: float = 10.0  # rad/s, below this the rotor is considered stopped
    #: [realistic] re-fit ``theta_tw`` so that BEMT *with inflow* reproduces the
    #: measured ``T = kt*omega^2`` at hover. See docs/PLANT_MODEL.md §2; the
    #: published ``theta_tw`` was fitted against the inflow-free expression, so
    #: using it unchanged with inflow enabled shifts trim by tens of percent.
    calibrate: bool = True


@dataclass(frozen=True)
class Motor:
    """DC motor electrical/mechanical parameters. [matlab]"""

    inductance: float = 15e-5
    ke: float = 0.008286
    resistance: float = 0.3774
    km: float = 0.014490
    inertia: float = 1.25e-5
    u_max: float = 11.1  # offline measurement, "normal" pack voltage


@dataclass(frozen=True)
class Esc:
    """[realistic] ESC dynamics between PWM command and motor voltage."""

    tau: float = 0.02            # first-order lag, s
    pwm_quantum_us: float = 1.0  # command quantisation
    enabled: bool = True


@dataclass(frozen=True)
class Battery:
    """[realistic] 3S LiPo pack with sag and SoC integration."""

    cells: int = 3
    capacity_ah: float = 3.3
    r_int: float = 0.035
    v_nom: float = 11.1
    soc_breakpoints: tuple = (1.00, 0.95, 0.80, 0.50, 0.25, 0.10, 0.00)
    cell_ocv: tuple = (4.20, 4.05, 3.90, 3.80, 3.70, 3.45, 3.20)
    enabled: bool = True

    def open_circuit_voltage(self, soc: float) -> float:
        soc = float(np.clip(soc, 0.0, 1.0))
        # breakpoints are descending; np.interp needs ascending x
        return self.cells * float(
            np.interp(soc, self.soc_breakpoints[::-1], self.cell_ocv[::-1])
        )


@dataclass(frozen=True)
class Aero:
    """[realistic] body drag and ground effect."""

    cda: tuple = (0.06, 0.06, 0.10)  # body-frame drag area, m^2
    ground_effect: bool = True
    ground_effect_max: float = 0.30

    def drag_area(self) -> np.ndarray:
        return np.asarray(self.cda, dtype=float)


@dataclass(frozen=True)
class SensorParams:
    """[realistic] discrete-time sensor corruption, sampled at the controller rate."""

    gyro_noise_dps: float = 0.01
    gyro_bias_rw_dps: float = 0.05
    accel_noise_mg: float = 0.5
    accel_bias_mg: float = 1.0
    mag_noise_ut: float = 0.5
    mag_hard_iron: float = 0.03
    baro_noise_m: float = 0.05
    baro_bias_drift_m_sqrt_s: float = 0.02
    delay_samples: int = 1
    enabled: bool = True


@dataclass(frozen=True)
class WindParams:
    """[realistic] Dryden turbulence parameters."""

    sigma: tuple = (1.5, 1.0, 1.0)   # m/s
    length_scale: tuple = (200.0, 200.0, 50.0)  # m
    v_ref: float = 5.0
    seed: int = 0
    enabled: bool = True


@dataclass(frozen=True)
class SimParams:
    dt: float = 1e-3
    integrator: str = "rk4"     # "euler" for strict MATLAB parity
    t_end: float = 30.0
    log_every: int = 1


@dataclass(frozen=True)
class ControlParams:
    """Controller settings. ``[matlab]`` / ``[firmware]`` mark the diverging sets."""

    h: float = 0.04                # STR sampling period [matlab Mode II / firmware]
    order: int = 2                 # 2 -> Mode II inner loop, 3 -> Mode I
    zeta: float = 0.7              # reference model damping [firmware init_controller]
    wn: float = 5.0                # reference model natural frequency [firmware]
    mu: float = 0.0                # extra integrator pole Ao = 1 + mu z^-1
    forgetting: float = 0.99       # RLS lambda [firmware 0.99, matlab 0.998]
    p_max: float = 5.0             # [firmware 5, matlab 100]
    init_percent: float = 0.75     # deliberate initial parameter mismatch
    kp_outer: float = 1.5          # [firmware] outer PI, angle domain
    ki_outer: float = 0.01
    kd_outer: float = 0.0
    u_max: tuple = (80.0, 80.0, 100.0, 500.0)  # differential PWM limits [matlab umax]
    pid_kp: tuple = (10.0, 10.0, 20.0, 30.0)     # robust PID baseline [matlab c1]
    pid_ki: tuple = (0.2, 0.2, 0.4, 0.6)
    pid_kd: tuple = (2.0, 2.0, 4.0, 60.0)
    # numerical-safety policy, docs/CONTROL_SPEC.md §6
    bsum_min: float = 1e-3
    guard_bsum: float = 1e-6
    guard_det: float = 1e-9
    guard_u: float = 1e6
    fail_limit: int = 50


@dataclass(frozen=True)
class Params:
    """Everything the simulator needs. Immutable; use :meth:`replace` to vary."""

    airframe: Airframe = field(default_factory=Airframe)
    rotor: Rotor = field(default_factory=Rotor)
    motor: Motor = field(default_factory=Motor)
    esc: Esc = field(default_factory=Esc)
    battery: Battery = field(default_factory=Battery)
    aero: Aero = field(default_factory=Aero)
    sensors: SensorParams = field(default_factory=SensorParams)
    wind: WindParams = field(default_factory=WindParams)
    sim: SimParams = field(default_factory=SimParams)
    control: ControlParams = field(default_factory=ControlParams)
    fidelity: Fidelity = "high"

    # -- derived quantities -------------------------------------------------
    @property
    def w_hover(self) -> float:
        """[matlab] w_motor_N = sqrt(m g / (4 kT))."""
        return float(np.sqrt(self.airframe.mass * self.airframe.g / (4 * self.rotor.kt)))

    @property
    def u_hover(self) -> float:
        """[matlab] U_N, motor voltage needed at hover."""
        r, m = self.rotor, self.motor
        w = self.w_hover
        return float(
            (r.kq * w**2 + m.km * m.ke / m.resistance * w) * m.resistance / m.km
        )

    @property
    def pwm_hover_percent(self) -> float:
        """[matlab] PWM_N = (U_N / Umax)^2, a *fraction*, as the MATLAB code uses it."""
        return float((self.u_hover / self.motor.u_max) ** 2)

    @property
    def pwm_hover_us(self) -> float:
        """Hover command in microseconds (the firmware's domain)."""
        return float(self.pwm_hover_percent * 1000.0 + 1000.0)

    @property
    def tf_beta(self) -> np.ndarray:
        """[matlab] motor pole, 1/s (per rotor)."""
        r, m = self.rotor, self.motor
        w = self.w_hover
        return np.full(4, (2 * r.kq * w + m.km * m.ke / m.resistance) / m.inertia)

    @property
    def tf_alpha(self) -> np.ndarray:
        """[matlab] control effectiveness per axis [roll, pitch, yaw, altitude], deg/s."""
        a, r, m = self.airframe, self.rotor, self.motor
        j = a.inertia
        w = self.w_hover
        rad2deg = 180.0 / np.pi
        k = 2 * m.km * w * a.d_arm * r.kt / m.resistance / m.inertia * m.u_max / 250 / 2 * rad2deg
        return np.array(
            [
                k / j[0, 0],
                k / j[1, 1],
                2 * m.km * w * r.kq / m.resistance / j[2, 2] / m.inertia * m.u_max / 250 * rad2deg,
                2 * m.km * w * r.kt / m.resistance / a.mass / m.inertia * m.u_max / 250,
            ]
        )

    @classmethod
    def from_matlab(cls, fidelity: Fidelity = "paper") -> "Params":
        """Port of ``Load_SystemParameters.m`` with the rotor coefficients derived."""
        af = Airframe()
        rho, c, rr = af.rho, af.c_rotor, af.r_rotor
        kt = 1 / 3 * rho * c * rr**3 * af.cl_alpha * af.theta_tw
        kh = 1 / 2 * rho * c * rr**2 * af.cd_zero
        kq = 1 / 4 * rho * c * rr**4 * af.cd_zero
        return cls(
            airframe=af,
            rotor=Rotor(kt=kt, kh=kh, kq=kq),
            fidelity=fidelity,
            sim=SimParams(integrator="euler" if fidelity == "paper" else "rk4"),
        )

    def with_(self, **kwargs) -> "Params":
        """Shallow convenience wrapper around :func:`dataclasses.replace`."""
        return replace(self, **kwargs)


def default_params(fidelity: Fidelity = "high") -> Params:
    return Params.from_matlab(fidelity=fidelity)


__all__ = [
    "Airframe",
    "Rotor",
    "Motor",
    "Esc",
    "Battery",
    "Aero",
    "SensorParams",
    "WindParams",
    "SimParams",
    "ControlParams",
    "Params",
    "default_params",
    "Fidelity",
]
