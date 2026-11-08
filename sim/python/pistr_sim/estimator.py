"""Attitude estimator: the firmware's two-state (angle + gyro-bias) Kalman filter.

Ported from ``firmware/libraries/MyLibraries_Wenchao/IMU.cpp``
(``kalman_filter_roll`` / ``_pitch`` / ``_yaw``).  One independent 2-state filter
per axis; input is a :class:`pistr_sim.types.SensorSample`, output is a
:class:`pistr_sim.types.Estimate`.

Predict (identical algebra to the firmware)::

    angle_pre = angle - dt*bias + dt*gyro
    bias_pre  = bias
    P00_pre   = P00 - 2 dt P01 + dt^2 P11 + Q_angle
    P01_pre   = P01 - dt P11,      P10_pre = P01_pre,   P11_pre = P11 + Q_bias

Update with a direct angle measurement (accelerometer roll/pitch, magnetometer
yaw)::

    e = meas - angle_pre           (wrapped to [-pi, pi] for the heading)
    S = P00_pre + R
    K = [P00_pre / S, P10_pre / S]
    angle = angle_pre + K0 e ; bias = bias_pre + K1 e
    P00 = (1-K0) P00_pre ; P01 = (1-K0) P01_pre ; P10 = P01
    P11 = -K1 P01_pre + P11_pre

The firmware hard-codes ``Q_angle = dt^2`` (i.e. unit angle process noise),
``Q_bias = 0``, ``R = 1`` and ``P0 = 100``; those are the defaults here too, but
every term is configurable.

Documented deviations from ``IMU.cpp`` (both are behaviour improvements outside
the filter itself, no change to the Kalman recursion):

1. ``IMU.cpp`` returns the raw magnetometer heading ``yaw_mag`` instead of the
   filtered yaw state.  This port returns the filtered state (the control law
   uses it, and it is what a "2-state Kalman" implies).
2. Attitude extraction from the accelerometer uses ``atan2`` instead of the
   firmware's guarded ``atan`` ratio.  Both agree for |roll|, |pitch| < 90 deg
   (the vehicle envelope); ``atan2`` simply cannot divide by zero.

``use_accel=False`` runs the filter as a gyro/magnetometer observer.  It exists
for plant-in-the-loop tests that have no position/velocity aiding: translational
acceleration then corrupts the accelerometer's gravity reference, and a
gravity-referenced tilt loop can settle into a slow drift mode (the "tilt follows
acceleration" ambiguity).  It is a *bench* configuration, not a change to the
firmware filter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .types import (
    Estimate,
    SensorSample,
    quat_derivative,
    quat_normalise,
    quat_to_euler,
    quat_to_rotmat,
)


@dataclass(frozen=True)
class KalmanConfig:
    """Filter and sensor-model settings.

    ``q_angle`` / ``q_bias`` are process-noise *spectral densities*: the predictor
    adds ``q_angle*dt^2`` to ``P00`` and ``q_bias*dt^2`` to ``P11`` (the firmware
    value is ``q_angle = 1``, ``q_bias = 0``).
    """

    dt: float = 0.04
    q_angle: float = 1.0
    q_bias: float = 0.0
    r_accel: float = 1.0
    p0: float = 100.0
    use_accel: bool = True         # accelerometer tilt correction (see note below)
    use_mag: bool = True
    accel_sign: float = 1.0        # +1 when the accelerometer reads +g at rest
    tilt_model: str = "matlab"     # "matlab" (types.quat_to_rotmat) | "standard"
    mag_model: str = "matlab"      # "matlab" | "firmware" | "vector"
    #: Inertial magnetic field (any unit; only the direction is used) for
    #: ``mag_model="vector"``.  Default: the high-fidelity plant's
    #: ``_MAG_EARTH = (45, 0, 20) uT`` (northern-hemisphere dip ~24 deg).
    mag_reference: tuple = (45.0, 0.0, 20.0)
    parallel_update: bool = False  # accepted for API symmetry with the firmware

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.p0 <= 0.0:
            raise ValueError("p0 must be positive")
        if self.r_accel <= 0.0:
            raise ValueError("r_accel must be positive")
        if self.accel_sign not in (1.0, -1.0):
            raise ValueError("accel_sign must be +1 or -1")
        if self.tilt_model not in ("matlab", "standard"):
            raise ValueError("tilt_model must be 'matlab' or 'standard'")
        if self.mag_model not in ("matlab", "firmware", "vector"):
            raise ValueError("mag_model must be 'matlab', 'firmware' or 'vector'")
        if self.mag_model == "vector" and np.linalg.norm(self.mag_reference) < 1e-9:
            raise ValueError("mag_reference must be non-zero for mag_model='vector'")


@dataclass
class _AxisFilter:
    angle: float = 0.0
    bias: float = 0.0
    p00: float = 100.0
    p01: float = 0.0
    p11: float = 100.0

    def clone(self) -> "_AxisFilter":
        return replace(self)

    def predict(self, dt: float, gyro: float, q_angle: float, q_bias: float) -> "_AxisFilter":
        a = self.angle - dt * self.bias + dt * gyro
        b = self.bias
        p00 = self.p00 - 2.0 * dt * self.p01 + dt * dt * self.p11 + q_angle * dt * dt
        p01 = self.p01 - dt * self.p11
        p11 = self.p11 + q_bias * dt * dt
        return _AxisFilter(a, b, p00, p01, p11)

    def update(self, meas: float, r: float, wrap: bool = False) -> "_AxisFilter":
        e = float(meas) - self.angle
        if wrap:
            e = float(np.arctan2(np.sin(e), np.cos(e)))
        s = self.p00 + r
        if not np.isfinite(s) or s <= 0.0:
            return self.clone()
        k0 = self.p00 / s
        k1 = self.p01 / s
        angle = self.angle + k0 * e
        bias = self.bias + k1 * e
        p00 = (1.0 - k0) * self.p00
        p01 = (1.0 - k0) * self.p01
        p11 = -k1 * self.p01 + self.p11
        return _AxisFilter(angle, bias, p00, p01, p11)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    v = np.asarray(vector, dtype=float).reshape(3)
    if not np.all(np.isfinite(v)):
        return None
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def _accel_angles(
    accel: np.ndarray, sign: float, yaw: float, model: str
) -> tuple[float, float]:
    """Roll/pitch from a body-frame specific-force measurement.

    ``model="standard"`` assumes the gravity projection is yaw independent
    (``f/g = [-sin(theta), cos(theta) sin(phi), cos(theta) cos(phi)]``), which is
    what an IMU mounted in the usual way reports.

    ``model="matlab"`` inverts the measurement generated by the shared
    ``types.quat_to_rotmat`` contract, ``f/g = C_I2B @ e3`` (third column of the
    I2B matrix).  That column does depend on the heading for this particular
    Euler parametrisation, so the current yaw estimate is used::

        sin(phi)   = f_x sin(psi) + f_y cos(psi)
        sin(theta) = -(f_x cos(psi) - f_y sin(psi)) / cos(phi)
    """
    unit = _unit(float(sign) * np.asarray(accel, dtype=float))
    if unit is None:
        return 0.0, 0.0
    fx, fy, fz = unit
    if model == "standard":
        return float(np.arctan2(fy, fz)), float(np.arctan2(-fx, np.hypot(fy, fz)))
    sphi = float(np.clip(fx * np.sin(yaw) + fy * np.cos(yaw), -1.0, 1.0))
    phi = float(np.arcsin(sphi))
    cphi = float(np.cos(phi))
    if abs(cphi) < 1e-9:
        return phi, 0.0
    stheta = float(np.clip(-(fx * np.cos(yaw) - fy * np.sin(yaw)) / cphi, -1.0, 1.0))
    return phi, float(np.arcsin(stheta))


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Quaternion from the MATLAB/``types`` Euler convention."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array(
        [
            cy * cp * cr - sy * sp * sr,
            cy * cp * sr + sy * sp * cr,
            cy * sp * cr - sy * cp * sr,
            cy * sp * sr + sy * cp * cr,
        ]
    )


def _mag_yaw(
    mag: np.ndarray,
    roll: float,
    pitch: float,
    model: str = "matlab",
    reference: tuple = (45.0, 0.0, 20.0),
) -> float | None:
    """Magnetometer heading.

    ``model="matlab"``: in the shared rotation convention the first column of
    ``C_I2B`` is ``[cos(psi) cos(theta), -sin(psi) cos(theta), sin(theta)]``, so the
    heading is simply ``atan2(-m_y, m_x)`` (a level, hard-iron-free field along
    inertial +x is assumed).

    ``model="firmware"``: the de-rotation formula of ``IMU.cpp::mag_yaw()``,
    kept for IMUs/plants using that convention.

    ``model="vector"``: exact for the shared ``types.quat_to_rotmat`` convention
    with an arbitrary (non-horizontal) reference field.  The key identity is
    ``C(roll, pitch, yaw) = Rz(-yaw) C(roll, pitch, 0)``, so with the measured
    body field ``m_b`` and ``v = C(roll, pitch, 0) m_ref``::

        yaw = atan2(v_y, v_x) - atan2(m_b,y, m_b,x)

    This is what the physics plant needs: its field is 3-D (dip angle) and a
    level-field assumption biases the heading by several degrees.
    """
    unit = _unit(mag)
    if unit is None:
        return None
    mx, my, mz = unit
    if model == "matlab":
        if abs(mx) + abs(my) < 1e-12:
            return None
        return float(np.arctan2(-my, mx))
    if model == "vector":
        ref = _unit(reference)
        if ref is None:
            return None
        c0 = quat_to_rotmat(_quat_from_euler(roll, pitch, 0.0))
        v = c0 @ ref
        if float(np.hypot(v[0], v[1])) < 1e-6:
            return None
        return float(np.arctan2(v[1], v[0]) - np.arctan2(unit[1], unit[0]))
    m_y = -(my * np.cos(roll) - mz * np.sin(roll))
    m_x = mx * np.cos(pitch) + my * np.sin(roll) * np.sin(pitch) + mz * np.cos(roll) * np.sin(pitch)
    if abs(m_x) + abs(m_y) < 1e-12:
        return None
    return float(np.arctan2(m_y, m_x))


class AttitudeEstimator:
    """Two-state-per-axis attitude estimator (input ``SensorSample``, output ``Estimate``)."""

    def __init__(
        self,
        config: KalmanConfig | None = None,
        initial_angle: Sequence[float] | None = None,
    ):
        self.cfg = config or KalmanConfig()
        self._initial = (
            np.zeros(3) if initial_angle is None else np.asarray(initial_angle, dtype=float).reshape(3)
        )
        self.reset()

    def reset(self, initial_angle: Sequence[float] | None = None) -> None:
        if initial_angle is not None:
            self._initial = np.asarray(initial_angle, dtype=float).reshape(3)
        p0 = self.cfg.p0
        self.roll = _AxisFilter(float(self._initial[0]), 0.0, p0, 0.0, p0)
        self.pitch = _AxisFilter(float(self._initial[1]), 0.0, p0, 0.0, p0)
        self.yaw = _AxisFilter(float(self._initial[2]), 0.0, p0, 0.0, p0)
        self._t = 0.0
        self._last_rate = np.zeros(3)

    @property
    def gyro_bias(self) -> np.ndarray:
        return np.array([self.roll.bias, self.pitch.bias, self.yaw.bias])

    @property
    def covariance(self) -> np.ndarray:
        """Diagonal of the three 2x2 covariance blocks (diagnostics only)."""
        return np.array(
            [
                [self.roll.p00, self.roll.p01, self.roll.p11],
                [self.pitch.p00, self.pitch.p01, self.pitch.p11],
                [self.yaw.p00, self.yaw.p01, self.yaw.p11],
            ]
        )

    def update(self, sample: SensorSample, dt: float | None = None) -> Estimate:
        cfg = self.cfg
        dt = cfg.dt if dt is None else float(dt)
        gyro = np.asarray(sample.gyro, dtype=float).reshape(3)
        accel = np.asarray(sample.accel, dtype=float).reshape(3)
        if not (np.all(np.isfinite(gyro)) and np.all(np.isfinite(accel))):
            # Reject the whole sample rather than propagating NaN into the loop.
            return Estimate(
                t=float(sample.t),
                angle=np.array([self.roll.angle, self.pitch.angle, self.yaw.angle]),
                rate=self._last_rate.copy(),
            )

        roll_meas, pitch_meas = _accel_angles(
            accel, cfg.accel_sign, self.yaw.angle, cfg.tilt_model
        )

        # Predict all three axes, then update with the available measurements.
        roll = self.roll.predict(dt, float(gyro[0]), cfg.q_angle, cfg.q_bias)
        pitch = self.pitch.predict(dt, float(gyro[1]), cfg.q_angle, cfg.q_bias)
        yaw = self.yaw.predict(dt, float(gyro[2]), cfg.q_angle, cfg.q_bias)

        if cfg.use_accel:
            roll = roll.update(roll_meas, cfg.r_accel)
            pitch = pitch.update(pitch_meas, cfg.r_accel)

        if cfg.use_mag:
            yaw_meas = _mag_yaw(
                sample.mag, roll.angle, pitch.angle, cfg.mag_model, cfg.mag_reference
            )
            if yaw_meas is not None:
                yaw = yaw.update(yaw_meas, cfg.r_accel, wrap=True)

        self.roll, self.pitch, self.yaw = roll, pitch, yaw
        self._t = float(sample.t)
        bias = np.array([roll.bias, pitch.bias, yaw.bias])
        self._last_rate = gyro - bias
        return Estimate(
            t=self._t,
            angle=np.array([roll.angle, pitch.angle, yaw.angle]),
            rate=self._last_rate.copy(),
        )

    # convenience: the firmware API in Python clothing ---------------------
    def attitude(self) -> Estimate:
        return Estimate(
            t=self._t,
            angle=np.array([self.roll.angle, self.pitch.angle, self.yaw.angle]),
            rate=-self.gyro_bias,
        )


#: Alias kept for readers coming from the firmware vocabulary.
KalmanEstimator = AttitudeEstimator


@dataclass(frozen=True)
class AltitudeConfig:
    """2-state (altitude, climb-rate) Kalman filter fed by the barometer."""

    dt: float = 0.04
    q_alt: float = 1e-4
    q_rate: float = 1e-2
    r_baro: float = 2.5e-3   # (0.05 m)^2, the high-fidelity baro noise
    p0_alt: float = 1.0
    p0_rate: float = 0.25

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.r_baro <= 0.0:
            raise ValueError("dt and r_baro must be positive")


class AltitudeEstimator:
    """Fuse the barometer with the vertical acceleration for ``(h, v_z)``.

    The plant is NED, so the caller supplies the *upward* acceleration
    ``a_up = -((C_b2i @ accel)[2] + g)``.  The filter is a standard 2-state
    Kalman predictor/corrector; the barometer is the only position measurement.
    """

    def __init__(self, config: AltitudeConfig | None = None):
        self.cfg = config or AltitudeConfig()
        self.reset()

    def reset(self, altitude: float = 0.0, climb_rate: float = 0.0) -> None:
        self.altitude = float(altitude)
        self.climb_rate = float(climb_rate)
        self._p00 = float(self.cfg.p0_alt)
        self._p01 = 0.0
        self._p11 = float(self.cfg.p0_rate)
        self._t = 0.0

    def update(self, baro_alt: float, a_up: float = 0.0, dt: float | None = None) -> tuple[float, float]:
        cfg = self.cfg
        dt = cfg.dt if dt is None else float(dt)
        if not np.isfinite(baro_alt):
            return self.altitude, self.climb_rate
        a_up = float(a_up) if np.isfinite(a_up) else 0.0

        # predict
        h = self.altitude + dt * self.climb_rate + 0.5 * dt * dt * a_up
        v = self.climb_rate + dt * a_up
        p00 = self._p00 + 2.0 * dt * self._p01 + dt * dt * self._p11 + cfg.q_alt * dt
        p01 = self._p01 + dt * self._p11
        p11 = self._p11 + cfg.q_rate * dt

        # correct
        s = p00 + cfg.r_baro
        if s > 0.0 and np.isfinite(s):
            k0 = p00 / s
            k1 = p01 / s
            e = float(baro_alt) - h
            h += k0 * e
            v += k1 * e
            self._p00 = (1.0 - k0) * p00
            self._p01 = (1.0 - k0) * p01
            self._p11 = -k1 * p01 + p11
        else:
            self._p00, self._p01, self._p11 = p00, p01, p11
        if not (np.isfinite(h) and np.isfinite(v)):
            h, v = self.altitude, self.climb_rate
        self.altitude, self.climb_rate = float(h), float(v)
        self._t += dt
        return self.altitude, self.climb_rate


class QuaternionAttitudeEstimator:
    """Quaternion attitude observer for the physics bench.

    The firmware port (:class:`AttitudeEstimator`) propagates three independent
    Euler angles from the body rates, which is exact only for small attitudes.
    With combined roll/pitch/yaw motion (e.g. the sinusoidal reference) the
    Euler-rate approximation drifts and a body-rate-only observer must instead
    integrate the attitude quaternion.  This class does that:

    * predict: ``q <- normalize(q + dt/2 Q(q) (gyro - bias))``;
    * the heading is optionally corrected with the 3-D magnetometer model
      (``mag_model="vector"``), applied as a world-yaw rotation rebuilt from the
      current roll/pitch.

    It is used by :mod:`pistr_sim.control.plant_loop`; the firmware KF remains the
    reference implementation for the embedded core.
    """

    def __init__(
        self,
        dt: float = 0.04,
        initial_euler: Sequence[float] = (0.0, 0.0, 0.0),
        gyro_bias: Sequence[float] = (0.0, 0.0, 0.0),
        use_mag: bool = True,
        mag_reference: Sequence[float] = (45.0, 0.0, 20.0),
        yaw_gain: float = 0.2,
    ):
        self.cfg = KalmanConfig(
            dt=dt, use_mag=use_mag, mag_model="vector", mag_reference=tuple(mag_reference)
        )
        self.cfg_yaw_gain = float(yaw_gain)
        self._initial_euler = np.asarray(initial_euler, dtype=float).reshape(3)
        self._bias = np.asarray(gyro_bias, dtype=float).reshape(3)
        self.reset()

    def reset(self, initial_euler: Sequence[float] | None = None) -> None:
        if initial_euler is not None:
            self._initial_euler = np.asarray(initial_euler, dtype=float).reshape(3)
        self.q = _quat_from_euler(*self._initial_euler)
        self._t = 0.0
        self._last_rate = np.zeros(3)

    @property
    def gyro_bias(self) -> np.ndarray:
        return self._bias.copy()

    def update(self, sample: SensorSample, dt: float | None = None) -> Estimate:
        dt = self.cfg.dt if dt is None else float(dt)
        gyro = np.asarray(sample.gyro, dtype=float).reshape(3)
        if not np.all(np.isfinite(gyro)):
            return Estimate(self._t, quat_to_euler(self.q), self._last_rate.copy())
        rate = gyro - self._bias
        self.q = quat_normalise(self.q + dt * quat_derivative(self.q, rate))
        roll, pitch, yaw = quat_to_euler(self.q)

        if self.cfg.use_mag:
            yaw_meas = _mag_yaw(
                sample.mag, roll, pitch, self.cfg.mag_model, self.cfg.mag_reference
            )
            if yaw_meas is not None:
                err = float(np.arctan2(np.sin(yaw_meas - yaw), np.cos(yaw_meas - yaw)))
                yaw = yaw + self.cfg_yaw_gain * err
                self.q = _quat_from_euler(roll, pitch, yaw)

        self._t = float(sample.t)
        self._last_rate = rate
        return Estimate(self._t, np.array([roll, pitch, yaw]), rate.copy())


def synthetic_sensor_sample(
    t: float,
    angle: Sequence[float],
    rate: Sequence[float],
    bias: Sequence[float] | None = None,
    gyro_noise: float = 0.0,
    accel_noise: float = 0.0,
    mag_noise: float = 0.0,
    baro_alt: float = 0.0,
    rng: np.random.Generator | None = None,
    accel_g: float = 9.81,
) -> SensorSample:
    """Build the ``SensorSample`` a perfect IMU would report (test helper).

    Uses the *shared* ``types.quat_to_rotmat`` contract: the accelerometer reports
    ``g * C_I2B[:, 2]`` (the world-up direction in body coordinates) and the
    magnetometer reports ``C_I2B[:, 0]`` (a level, hard-iron-free field along the
    inertial x axis).  This is the convention the ``tilt_model="matlab"`` and
    ``mag_model="matlab"`` inverses assume, so the round trip is exact.
    """
    rng = rng or np.random.default_rng(0)
    angle = np.asarray(angle, dtype=float).reshape(3)
    rate = np.asarray(rate, dtype=float).reshape(3)
    bias = np.zeros(3) if bias is None else np.asarray(bias, dtype=float).reshape(3)
    roll, pitch, yaw = angle

    q = np.array(
        [
            np.cos(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2)
            - np.sin(yaw / 2) * np.sin(pitch / 2) * np.sin(roll / 2),
            np.cos(yaw / 2) * np.cos(pitch / 2) * np.sin(roll / 2)
            + np.sin(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2),
            np.cos(yaw / 2) * np.sin(pitch / 2) * np.cos(roll / 2)
            - np.sin(yaw / 2) * np.cos(pitch / 2) * np.sin(roll / 2),
            np.cos(yaw / 2) * np.sin(pitch / 2) * np.sin(roll / 2)
            + np.sin(yaw / 2) * np.cos(pitch / 2) * np.cos(roll / 2),
        ]
    )
    c = quat_to_rotmat(q)
    accel = accel_g * c[:, 2]
    mag = c[:, 0].copy()
    gyro = rate + bias

    if gyro_noise:
        gyro = gyro + rng.normal(0.0, gyro_noise, 3)
    if accel_noise:
        accel = accel + rng.normal(0.0, accel_noise, 3)
    if mag_noise:
        mag = mag + rng.normal(0.0, mag_noise, 3)

    return SensorSample(t=float(t), gyro=gyro, accel=accel, mag=mag, baro_alt=float(baro_alt))


__all__ = [
    "KalmanConfig",
    "AttitudeEstimator",
    "KalmanEstimator",
    "QuaternionAttitudeEstimator",
    "AltitudeConfig",
    "AltitudeEstimator",
    "synthetic_sensor_sample",
]
