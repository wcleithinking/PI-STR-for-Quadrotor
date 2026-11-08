"""Reference trajectories, the 3-D attitude planner and the ESO.

Three reference families cover every experiment in the repository:

* :class:`HoverReference` — the paper's ``type_reference = 1`` (hold altitude).
* :class:`SinusoidReference` — the paper's ``type_reference = 2`` (sinusoidal
  attitude and altitude).
* :class:`StepReference` — the deterministic step used by the conformance test and
  the step-response metrics (``docs/CONTROL_SPEC.md`` §8).
* :class:`TrajectoryReference3D` — the paper's robustness study
  (``vx = 4 m/s``, sinusoidal ``vy``/``vz``, yaw sweep), plus
  :class:`AttitudePlanner3D` which turns the position loop into roll/pitch
  references and differentiates them through the ESO.

``ReferenceSample`` carries both the four-channel vector used by the controller
stack (``[roll, pitch, yaw, altitude]``) and the full 3-D position reference so the
same object can drive the trajectory study.  Units are SI: rad, rad/s, m, m/s.

The 3-D block is a line-by-line port of the MATLAB trajectory script
(``sim/matlab/robustness/Sim_Trajectory_Tracking_Control.m``)::

    xi_ref   = [vx*t, A_vy sin(w_vy t + pi), A_vz sin(w_vz t + pi/2) + B_vz]
    yaw_ref  = A_yaw sin(w_yaw t) + A_yaw cos(w_yaw t)
    u_x      = ddot_x_ref - k1 (x - x_ref) - k2 (vx - vx_ref)
    pitch    = asin(u_x / (-m g))
    roll     = asin(u_y / (m g cos(pitch)))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------
@dataclass
class ReferenceSample:
    """One reference instant. Angles in rad, rates in rad/s, positions in m."""

    t: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    altitude: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0
    climb_rate: float = 0.0
    xi: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_xi: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a_xi: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def angle(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw])

    @property
    def rate(self) -> np.ndarray:
        return np.array([self.roll_rate, self.pitch_rate, self.yaw_rate])

    @property
    def y_ref(self) -> np.ndarray:
        """Controller-facing reference ``[roll, pitch, yaw, altitude]``."""
        return np.array([self.roll, self.pitch, self.yaw, self.altitude])

    @property
    def rate_ref(self) -> np.ndarray:
        """Controller-facing rate reference ``[p, q, r, climb_rate]``."""
        return np.array([self.roll_rate, self.pitch_rate, self.yaw_rate, self.climb_rate])


# ---------------------------------------------------------------------------
# Reference generators
# ---------------------------------------------------------------------------
class ReferenceBase:
    """Pure function of time; no measurements are needed."""

    name = "base"

    def reset(self) -> None:
        pass

    def sample(self, t: float, xi_meas: Sequence[float] | None = None,
               v_meas: Sequence[float] | None = None) -> ReferenceSample:
        raise NotImplementedError


class HoverReference(ReferenceBase):
    """Hold attitude at zero and altitude at ``altitude`` (``type_reference = 1``)."""

    name = "hover"

    def __init__(self, altitude: float = 5.0, yaw: float = 0.0):
        self.altitude = float(altitude)
        self.yaw = float(yaw)

    def sample(self, t: float, xi_meas=None, v_meas=None) -> ReferenceSample:
        return ReferenceSample(
            t=float(t), yaw=self.yaw, altitude=self.altitude, xi=np.array([0.0, 0.0, self.altitude])
        )


class StepReference(ReferenceBase):
    """Angle step on one axis at ``t_step`` (there is no rate feed-forward)."""

    name = "step"

    def __init__(
        self,
        amplitude: float = 0.1,
        t_step: float = 1.0,
        axis: int = 0,
        altitude: float = 0.0,
        yaw: float = 0.0,
    ):
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0 (roll), 1 (pitch) or 2 (yaw)")
        self.amplitude = float(amplitude)
        self.t_step = float(t_step)
        self.axis = int(axis)
        self.altitude = float(altitude)
        self.yaw = float(yaw)

    @property
    def target(self) -> float:
        return self.amplitude if self.axis != 2 else self.yaw + self.amplitude

    def sample(self, t: float, xi_meas=None, v_meas=None) -> ReferenceSample:
        value = self.amplitude if t >= self.t_step - _EPS else 0.0
        ref = ReferenceSample(t=float(t), yaw=self.yaw, altitude=self.altitude)
        if self.axis == 0:
            ref.roll = value
        elif self.axis == 1:
            ref.pitch = value
        else:
            ref.yaw = self.yaw + value
        return ref


class SinusoidReference(ReferenceBase):
    """Attitude/altitude sinusoids, the paper's ``type_reference = 2``.

    Defaults are the MATLAB Mode-II amplitudes (5 deg at 2 rad/s, 10 deg yaw,
    1 m at 1 rad/s around 5 m).
    """

    name = "sine"

    def __init__(
        self,
        roll_amplitude: float = np.deg2rad(5.0),
        pitch_amplitude: float = np.deg2rad(5.0),
        yaw_amplitude: float = np.deg2rad(10.0),
        roll_omega: float = 2.0,
        pitch_omega: float = 2.0,
        yaw_omega: float = 2.0,
        altitude_amplitude: float = 1.0,
        altitude_omega: float = 1.0,
        altitude_offset: float = 5.0,
    ):
        self.roll_amplitude = float(roll_amplitude)
        self.pitch_amplitude = float(pitch_amplitude)
        self.yaw_amplitude = float(yaw_amplitude)
        self.roll_omega = float(roll_omega)
        self.pitch_omega = float(pitch_omega)
        self.yaw_omega = float(yaw_omega)
        self.altitude_amplitude = float(altitude_amplitude)
        self.altitude_omega = float(altitude_omega)
        self.altitude_offset = float(altitude_offset)

    def sample(self, t: float, xi_meas=None, v_meas=None) -> ReferenceSample:
        t = float(t)
        roll = self.roll_amplitude * np.sin(self.roll_omega * t)
        pitch = self.pitch_amplitude * np.cos(self.pitch_omega * t)
        yaw = self.yaw_amplitude * np.sin(self.yaw_omega * t)
        alt = self.altitude_amplitude * np.sin(self.altitude_omega * t) + self.altitude_offset
        return ReferenceSample(
            t=t,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            altitude=alt,
            roll_rate=self.roll_amplitude * self.roll_omega * np.cos(self.roll_omega * t),
            pitch_rate=-self.pitch_amplitude * self.pitch_omega * np.sin(self.pitch_omega * t),
            yaw_rate=self.yaw_amplitude * self.yaw_omega * np.cos(self.yaw_omega * t),
            climb_rate=self.altitude_amplitude * self.altitude_omega * np.cos(self.altitude_omega * t),
            xi=np.array([0.0, 0.0, alt]),
            v_xi=np.array([0.0, 0.0, self.altitude_amplitude * self.altitude_omega * np.cos(self.altitude_omega * t)]),
        )


class TrajectoryReference3D(ReferenceBase):
    """The paper's 3-D position reference and yaw sweep (MATLAB trajectory study).

    ``vx_d = 4`` m/s, ``vy = 60 sin(2 pi t/30 + pi)`` m/s-like amplitude (the
    published script uses 60 m amplitude), ``vz = 10 sin(2 pi t/30 + pi/2) - 20``.
    Yaw is ``A sin(w t) + A cos(w t)`` with ``A = 60/sqrt(2)`` deg.
    """

    name = "trajectory"

    def __init__(
        self,
        vx_d: float = 4.0,
        a_vy: float = 60.0,
        b_vy: float = 0.0,
        omega_vy: float = 2.0 * np.pi / 30.0,
        phase_vy: float = np.pi,
        a_vz: float = 10.0,
        b_vz: float = -20.0,
        omega_vz: float = 2.0 * np.pi / 30.0,
        phase_vz: float = np.pi / 2.0,
        yaw_amplitude: float = np.deg2rad(60.0) / np.sqrt(2.0),
        yaw_omega: float = 2.0 * np.pi / 20.0,
    ):
        self.vx_d = float(vx_d)
        self.a_vy = float(a_vy)
        self.b_vy = float(b_vy)
        self.omega_vy = float(omega_vy)
        self.phase_vy = float(phase_vy)
        self.a_vz = float(a_vz)
        self.b_vz = float(b_vz)
        self.omega_vz = float(omega_vz)
        self.phase_vz = float(phase_vz)
        self.yaw_amplitude = float(yaw_amplitude)
        self.yaw_omega = float(yaw_omega)

    def sample(self, t: float, xi_meas=None, v_meas=None) -> ReferenceSample:
        t = float(t)
        xi = np.array(
            [
                self.vx_d * t,
                self.a_vy * np.sin(self.omega_vy * t + self.phase_vy) + self.b_vy,
                self.a_vz * np.sin(self.omega_vz * t + self.phase_vz) + self.b_vz,
            ]
        )
        v_xi = np.array(
            [
                self.vx_d,
                self.omega_vy * self.a_vy * np.cos(self.omega_vy * t + self.phase_vy),
                self.omega_vz * self.a_vz * np.cos(self.omega_vz * t + self.phase_vz),
            ]
        )
        a_xi = np.array(
            [
                0.0,
                -self.omega_vy**2 * self.a_vy * np.sin(self.omega_vy * t + self.phase_vy),
                -self.omega_vz**2 * self.a_vz * np.sin(self.omega_vz * t + self.phase_vz),
            ]
        )
        yaw = self.yaw_amplitude * np.sin(self.yaw_omega * t) + self.yaw_amplitude * np.cos(
            self.yaw_omega * t
        )
        yaw_rate = self.yaw_amplitude * self.yaw_omega * np.cos(
            self.yaw_omega * t
        ) - self.yaw_amplitude * self.yaw_omega * np.sin(self.yaw_omega * t)
        return ReferenceSample(
            t=t,
            yaw=yaw,
            altitude=xi[2],
            yaw_rate=yaw_rate,
            climb_rate=v_xi[2],
            xi=xi,
            v_xi=v_xi,
            a_xi=a_xi,
        )


# ---------------------------------------------------------------------------
# ESO
# ---------------------------------------------------------------------------
class ESO:
    """Second-order extended state observer for one channel (MATLAB port).

    ``k1 = pole^2``, ``k2 = -2 pole`` with ``pole = -2`` in the paper.  The
    controller uses the *current* rate estimate, then the ESO advances one tick
    (matching the MATLAB ``hat(t+1)`` bookkeeping).
    """

    def __init__(self, pole: float = -2.0, h: float = 0.02, initial: float = 0.0):
        self.pole = float(pole)
        self.h = float(h)
        self.k1 = self.pole**2
        self.k2 = -2.0 * self.pole
        self.reset(initial)

    def reset(self, angle: float = 0.0, rate: float = 0.0) -> None:
        self.angle = float(angle)
        self.rate = float(rate)

    def step(self, measurement: float) -> tuple[float, float]:
        """Return ``(angle_hat, rate_hat)`` for this tick and propagate."""
        measurement = float(measurement)
        if not np.isfinite(measurement):
            return self.angle, self.rate
        angle_now, rate_now = self.angle, self.rate
        error = angle_now - measurement
        self.angle = angle_now + self.h * (rate_now - self.k1 * error)
        self.rate = rate_now + self.h * (-self.k2 * error)
        return angle_now, rate_now


# ---------------------------------------------------------------------------
# Attitude planner (position loop -> roll/pitch reference)
# ---------------------------------------------------------------------------
class AttitudePlanner:
    """Position PD to attitude reference, ported from the MATLAB trajectory study.

    ``u_x = ddot_x_ref - k1 (x - x_ref) - k2 (vx - vx_ref)``, then
    ``pitch = asin(u_x / (-m g))`` and ``roll = asin(u_y / (m g cos(pitch)))``.
    The ``arcsin`` argument is clamped so a large position error cannot produce a
    NaN; the resulting tilt is additionally limited to ``max_tilt``.
    """

    def __init__(
        self,
        mass: float,
        gravity: float = 9.81,
        pole_position: float = -0.2,
        max_tilt: float = np.deg2rad(60.0),
    ):
        self.mass = float(mass)
        self.g = float(gravity)
        self.k1 = pole_position**2
        self.k2 = -2.0 * pole_position
        self.max_tilt = float(max_tilt)

    def update(
        self,
        xi_ref: Sequence[float],
        v_ref: Sequence[float],
        a_ref: Sequence[float],
        xi: Sequence[float],
        v: Sequence[float],
    ) -> tuple[float, float]:
        xi_ref = np.asarray(xi_ref, dtype=float).reshape(3)
        v_ref = np.asarray(v_ref, dtype=float).reshape(3)
        a_ref = np.asarray(a_ref, dtype=float).reshape(3)
        xi = np.asarray(xi, dtype=float).reshape(3)
        v = np.asarray(v, dtype=float).reshape(3)

        u_x = a_ref[0] - self.k1 * (xi[0] - xi_ref[0]) - self.k2 * (v[0] - v_ref[0])
        u_y = a_ref[1] - self.k1 * (xi[1] - xi_ref[1]) - self.k2 * (v[1] - v_ref[1])

        pitch = float(np.arcsin(np.clip(u_x / (-self.mass * self.g), -1.0, 1.0)))
        c = np.cos(pitch)
        if abs(c) < 1e-6:
            c = np.sign(c) * 1e-6 if c != 0.0 else 1e-6
        roll = float(np.arcsin(np.clip(u_y / (self.mass * self.g * c), -1.0, 1.0)))

        pitch = float(np.clip(pitch, -self.max_tilt, self.max_tilt))
        roll = float(np.clip(roll, -self.max_tilt, self.max_tilt))
        return roll, pitch


class AttitudePlanner3D:
    """Trajectory reference + planner + ESO, i.e. the full outer loop of the study."""

    def __init__(
        self,
        reference: TrajectoryReference3D | None = None,
        mass: float = 0.985,
        gravity: float = 9.81,
        h: float = 0.02,
        pole_position: float = -0.2,
        pole_eso: float = -2.0,
        use_eso_after: float = 2.0,
    ):
        self.reference = reference or TrajectoryReference3D()
        self.planner = AttitudePlanner(mass, gravity, pole_position)
        self.h = float(h)
        self.eso_roll = ESO(pole_eso, h)
        self.eso_pitch = ESO(pole_eso, h)
        self.use_eso_after = float(use_eso_after)
        self._t0: float | None = None

    def reset(self) -> None:
        self.eso_roll.reset()
        self.eso_pitch.reset()
        self._t0 = None

    def sample(
        self,
        t: float,
        xi_meas: Sequence[float] | None = None,
        v_meas: Sequence[float] | None = None,
    ) -> ReferenceSample:
        ref = self.reference.sample(t)
        xi = np.zeros(3) if xi_meas is None else np.asarray(xi_meas, dtype=float).reshape(3)
        v = np.zeros(3) if v_meas is None else np.asarray(v_meas, dtype=float).reshape(3)

        roll_ref, pitch_ref = self.planner.update(ref.xi, ref.v_xi, ref.a_xi, xi, v)
        ref.roll = roll_ref
        ref.pitch = pitch_ref

        # ESO differentiation of the planned attitude (the no-ESO ramp keeps the
        # first two seconds identical to the MATLAB script).
        _, roll_rate = self.eso_roll.step(roll_ref)
        _, pitch_rate = self.eso_pitch.step(pitch_ref)
        if self._t0 is None:
            self._t0 = float(t)
        if float(t) - self._t0 <= self.use_eso_after:
            ref.roll_rate = 0.0
            ref.pitch_rate = 0.0
        else:
            ref.roll_rate = roll_rate
            ref.pitch_rate = pitch_rate
        return ref


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
REFERENCE_NAMES = ("hover", "step", "sine", "trajectory")


def make_reference(name: str, **kwargs) -> ReferenceBase:
    key = name.lower()
    if key == "hover":
        return HoverReference(**kwargs)
    if key == "step":
        return StepReference(**kwargs)
    if key == "sine" or key == "sinusoid":
        return SinusoidReference(**kwargs)
    if key in ("trajectory", "trajectory3d", "3d"):
        return TrajectoryReference3D(**kwargs)
    raise ValueError(f"unknown reference {name!r}; choose from {REFERENCE_NAMES}")


__all__ = [
    "ReferenceSample",
    "ReferenceBase",
    "HoverReference",
    "StepReference",
    "SinusoidReference",
    "TrajectoryReference3D",
    "ESO",
    "AttitudePlanner",
    "AttitudePlanner3D",
    "REFERENCE_NAMES",
    "make_reference",
]
