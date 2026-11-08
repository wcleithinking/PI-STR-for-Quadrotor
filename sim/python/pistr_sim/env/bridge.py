"""MuJoCo <-> NED bridge: the adapter that lets the existing control stack fly a
3-D MuJoCo world without knowing it exists.

Normative text: ``docs/3D_ENVIRONMENT.md`` §2 (frames), §4 (API), V3/V5/V8.

Frame mapping (contract §2, corrected during M4 implementation -- see the derivation
in the document)
----------------------------------------------------------------------------------
MuJoCo is z-up/right-handed with ``(w, x, y, z)`` quaternions; the rest of the
repository is NED (``z`` down, thrust along body ``-z``, ``xi[2]`` is negative
altitude).  One constant carries the whole mapping::

    S = diag(1, -1, -1)                 # a rotation by 180 deg about x (det = +1)
    p_mj       = S @ p_ned              # (x, -y, -z)
    R_b2w_mj   = S @ R_i2b_ned.T @ S    # body->world in MuJoCo
    q_mj       = (w, x, -y, -z)         # same rotation, quaternion form
    v_mj       = S @ v_ned,  omega_mj = S @ omega_ned

``S`` must be applied to positions and to body-frame vectors alike: the "obvious"
``(x, +y, -z)`` position rule has ``det = -1`` (a mirror), and mixing it with a proper
rotation for the attitude makes position feedback a y-axis positive-feedback loop --
invisible in hover/pure attitude control, divergent as soon as an outer loop closes.
Consequence: a NED ``roll > 0`` (right wing down, thrust axis toward NED ``+y``) shows
up in MuJoCo as a tilt toward ``-y`` because MuJoCo's ``+y`` is west; the physics is
the same, only the label flips (V3 asserts both the label and the physics).

Physics coupling
----------------
The bridge owns *nothing* that another stream owns.  Rotor thrust/torque/ESC/motor
dynamics come from ``env.actuation.RotorPack`` (M1) and the world comes from
``env.mjcf.build_model`` (M1) once those land; until then an injected model and an
injected pack (``_devmodel`` + a test fixture) are used, and the same code paths run.
Per actuator sub-step the bridge

1. computes the relative airspeed at the vehicle (steady + Dryden wind from
   :class:`pistr_sim.wind.WindModel`, same model as ``plant.py``),
2. advances the pack, obtains ``(thrust[4], torque_z[4])``,
3. assembles the body-frame wrench *in NED* exactly like ``plant.py``
   (``F = sum (T_i+...)(-z) + drag``, ``M = sum r_i x F_i + yaw reaction + gyro``),
4. maps the wrench with ``S`` and applies it through ``d.xfrc_applied``, which
   MuJoCo 4.0 reads as a **world-frame force/torque at the body CoM**.  Contact is
   MuJoCo's.

The moment is assembled about the body frame origin, which *is* the CoM in the M1
model (``<inertial pos="0 0 0">``), so no lever-arm correction is needed.  If a model
with an offset CoM is ever used, the torque has to be shifted by
``(CoM - origin) x F`` (measured empirically by M1: a world-frame force at
``ipos =/= 0`` produces no moment, i.e. the application point is the CoM).

Sensors
-------
No GPS: position is *not* measured.  ``sense()`` returns a
:class:`pistr_sim.types.SensorSample` built from MuJoCo truth with the same
noise/delay model as ``plant.py`` §5 (gyro bias random walk, accelerometer bias,
barometer bias drift, hard iron, one-sample transport delay at the control rate).
The accelerometer reports the *specific force* from the aero wrench we applied, so a
level hover reads ``(0, 0, -g)`` in the NED body frame, exactly like the plant.

Timing
-------
MuJoCo integrates at ``model.opt.timestep``; the control stack steps the plant with
``MuJoCoPlant.step(pwm)`` at ``plant.dt``.  ``dt`` is quantised to
``n_sub * model.opt.timestep`` (2 ms on the dev/contract model: 2 sub-steps of 1 ms,
or 1 sub-step of 2 ms), and the pack is advanced at the MuJoCo rate, i.e. the
actuator sees the sub-step, not the control period.

Known simplifications (also in the M4 report)
---------------------------------------------
* No ground-effect augmentation of the pack thrust near the floor (the plant has it;
  MuJoCo contact plus the pack is what the world has).  Hover at 5 m is unaffected.
* ``rotor_current`` is reported as zeros unless the pack exposes it; ``v_terminal``
  falls back to ``params.motor.u_max``.
* Rendered rotor geoms are M1's business; the wrench uses ``params.airframe.
  arm_positions`` (NED body) so the physics does not depend on the visual layout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..params import Params, default_params
from ..types import (
    PWM_MAX,
    PWM_MIN,
    SensorSample,
    VehicleState,
    quat_normalise,
    quat_to_rotmat,
)
from ..wind import WindModel

__all__ = [
    "MAG_EARTH_UT_NED",
    "MuJoCoPlant",
    "S_NED_TO_FLU",
    "mj_to_ned",
    "ned_to_mj",
    "quat_mj_to_ned",
    "quat_ned_to_mj",
    "rotmat_mj_to_ned",
    "rotmat_ned_to_mj",
    "scene_argument",
]

#: ``NED -> MuJoCo FLU`` for positions **and** body/world vectors: a 180 deg rotation
#: about ``x``.  ``S = S.T = S^-1``.
S_NED_TO_FLU: np.ndarray = np.diag([1.0, -1.0, -1.0])

#: Earth magnetic field in the inertial (NED) frame, uT.  Same constant as
#: ``plant.py`` / ``experiments.loop`` so the estimator sees one field everywhere.
MAG_EARTH_UT_NED: np.ndarray = np.array([45.0, 0.0, 20.0])

_HARD_IRON_SHAPE = np.array([1.0, -0.5, 0.8])
_EZ = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Frame conversions (pure functions, V3 asserts them)
# ---------------------------------------------------------------------------
def ned_to_mj(vec: Sequence[float]) -> np.ndarray:
    """Map an NED position/vector to MuJoCo world/FLU coordinates."""
    return S_NED_TO_FLU @ np.asarray(vec, dtype=float).reshape(3)


def mj_to_ned(vec: Sequence[float]) -> np.ndarray:
    """Map a MuJoCo position/vector to NED.  ``S`` is its own inverse."""
    return S_NED_TO_FLU @ np.asarray(vec, dtype=float).reshape(3)


def quat_ned_to_mj(q: Sequence[float]) -> np.ndarray:
    """Convert a repository quaternion (inertial->body, ``(w, x, y, z)``) to MuJoCo's.

    MuJoCo stores body->world; the conversion is ``(w, x, y, z) -> (w, x, -y, -z)``,
    which is the quaternion form of ``S R S``.  The map is an involution.
    """
    q = np.asarray(q, dtype=float).reshape(4)
    return quat_normalise(np.array([q[0], q[1], -q[2], -q[3]]))


def quat_mj_to_ned(q: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`quat_ned_to_mj` (same component flip)."""
    q = np.asarray(q, dtype=float).reshape(4)
    return quat_normalise(np.array([q[0], q[1], -q[2], -q[3]]))


def rotmat_ned_to_mj(c_i2b_ned: np.ndarray) -> np.ndarray:
    """Body->world rotation in MuJoCo from the repository's inertial->body matrix."""
    c = np.asarray(c_i2b_ned, dtype=float).reshape(3, 3)
    return S_NED_TO_FLU @ c.T @ S_NED_TO_FLU


def rotmat_mj_to_ned(r_b2w_mj: np.ndarray) -> np.ndarray:
    """Inertial->body (NED) rotation from MuJoCo's body->world matrix."""
    r = np.asarray(r_b2w_mj, dtype=float).reshape(3, 3)
    return S_NED_TO_FLU @ r.T @ S_NED_TO_FLU


# ---------------------------------------------------------------------------
# Sensor corruption (plant.py §5, re-used rather than re-invented)
# ---------------------------------------------------------------------------
class _SensorModel:
    """IMU/magnetometer/barometer corruption plus transport delay.

    Numbers and update order are those of ``plant.QuadrotorPlant._push_sample``
    (§5 of ``docs/PLANT_MODEL.md``); the *clean* values come from MuJoCo truth
    instead of from an integrated state.  Sampling happens at the control rate
    (``params.control.h``) and ``delay_samples`` samples are handed back, so the
    controller sees the same latency as with the Python plant.
    """

    def __init__(self, params: Params, dt: float, *, seed: int = 0, enabled: bool = True):
        self.params = params
        self.sp = params.sensors
        self.dt = float(dt)
        self.period = max(1, int(round(float(params.control.h) / self.dt)))
        self.enabled = bool(enabled)
        self.delay = int(self.sp.delay_samples) if self.enabled else 0
        self._seed = int(seed)
        self._g = float(params.airframe.g)
        self._accel_bias = (
            np.array([1.0, -1.0, 1.0]) * float(self.sp.accel_bias_mg) * 1e-3 * self._g
        )
        self._hard_iron = (
            float(self.sp.mag_hard_iron) * float(np.linalg.norm(MAG_EARTH_UT_NED)) * _HARD_IRON_SHAPE
        )
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(self._seed if seed is None else int(seed))
        self._gyro_bias = np.zeros(3)
        self._baro_bias = 0.0
        self._buf: list[SensorSample] = []

    def push(
        self,
        t: float,
        *,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: np.ndarray,
        baro_alt: float,
    ) -> None:
        gyro = np.asarray(gyro, dtype=float).reshape(3)
        accel = np.asarray(accel, dtype=float).reshape(3)
        mag = np.asarray(mag, dtype=float).reshape(3)
        if not self.enabled:
            sample = SensorSample(
                t=float(t),
                gyro=gyro.copy(),
                accel=accel.copy(),
                mag=mag.copy(),
                baro_alt=float(baro_alt),
            )
        else:
            h = float(self.params.control.h)
            sp = self.sp
            sigma_g = np.deg2rad(float(sp.gyro_noise_dps))
            sigma_bg = np.deg2rad(float(sp.gyro_bias_rw_dps)) * np.sqrt(h)
            sigma_a = float(sp.accel_noise_mg) * 1e-3 * self._g
            sigma_m = float(sp.mag_noise_ut)
            sigma_baro = float(sp.baro_noise_m)
            sigma_baro_rw = float(sp.baro_bias_drift_m_sqrt_s) * np.sqrt(h)

            self._gyro_bias = self._gyro_bias + sigma_bg * self._rng.standard_normal(3)
            self._baro_bias = self._baro_bias + sigma_baro_rw * self._rng.standard_normal()

            sample = SensorSample(
                t=float(t),
                gyro=gyro + self._gyro_bias + sigma_g * self._rng.standard_normal(3),
                accel=accel + self._accel_bias + sigma_a * self._rng.standard_normal(3),
                mag=mag + self._hard_iron + sigma_m * self._rng.standard_normal(3),
                baro_alt=float(baro_alt + self._baro_bias + sigma_baro * self._rng.standard_normal()),
            )

        self._buf.append(sample)
        keep = self.delay + 1
        if len(self._buf) > keep:
            del self._buf[: len(self._buf) - keep]

    def sense(self) -> SensorSample:
        if not self._buf:
            zeros = np.zeros(3)
            return SensorSample(t=0.0, gyro=zeros, accel=zeros, mag=zeros, baro_alt=0.0)
        idx = max(0, len(self._buf) - 1 - self.delay)
        s = self._buf[idx]
        return SensorSample(
            t=s.t,
            gyro=s.gyro.copy(),
            accel=s.accel.copy(),
            mag=s.mag.copy(),
            baro_alt=float(s.baro_alt),
        )


# ---------------------------------------------------------------------------
# Per-drone book-keeping
# ---------------------------------------------------------------------------
@dataclass
class _Drone:
    """One vehicle: its MuJoCo addressing, its actuator pack, its sensors, its state."""

    name: str
    body_id: int
    qposadr: int
    dofadr: int
    pack: Any
    sensors: _SensorModel
    state: VehicleState = field(default_factory=VehicleState)
    last_aero: dict = field(default_factory=dict)


def _pack_vector(pack: Any, names: Iterable[str], size: int, default: float = 0.0) -> np.ndarray:
    """Read one array off a pack (contract names first, then common aliases)."""
    for name in names:
        value = getattr(pack, name, None)
        if value is None:
            continue
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if arr.size == size:
            return arr
    return np.full(size, float(default))


def _pack_scalar(pack: Any, names: Iterable[str], default: float = 1.0) -> float:
    for name in names:
        value = getattr(pack, name, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _as_vec4(value: Any, *, default: float = 0.0) -> np.ndarray:
    """Coerce a pack trim field to a ``(4,)`` float array (scalar -> broadcast)."""
    if value is None:
        return np.full(4, float(default))
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(4, float(arr[0]))
    if arr.size != 4:
        return np.full(4, float(default))
    return arr


def _as_scalar_thrust(value: Any) -> float:
    """Total thrust from a pack trim field, which may be a 4-vector or a scalar."""
    if value is None:
        return float("nan")
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return float(arr[0])
    return float(np.sum(np.maximum(arr, 0.0)))


def scene_argument(scene: Any) -> Any:
    """Turn a ``SceneSpec`` into the MJCF fragment ``env.mjcf.build_model`` expects.

    ``build_model(scene=...)`` takes XML text (or ``None``), not the dataclass, so a
    caller who has M2's ``SceneSpec`` must convert it once.  Strings pass through, so
    this is idempotent and safe to apply on every path (``run_flight`` and
    ``MuJoCoPlant`` both do).
    """
    if scene is None or isinstance(scene, str):
        return scene
    try:
        from .scene import scene_xml
    except ImportError:  # pragma: no cover - scene module is M2
        return scene
    return scene_xml(scene)


def _instantiate_default_pack(params: Params) -> Any:
    """Construct the contract's ``env.actuation.RotorPack`` (frozen: ``RotorPack(params)``)."""
    from .actuation import RotorPack  # ImportError here means "no 3-D deps installed"

    return RotorPack(params)


class MuJoCoPlant:
    """QuadrotorPlant-compatible adapter over a MuJoCo world (duck-typed, not subclassed).

    Parameters
    ----------
    params:
        Frozen :class:`pistr_sim.params.Params` bundle (mass/inertia/rotor geometry,
        aerodynamic coefficients, sensor noise, wind).
    model, xml:
        Inject an ``MjModel`` or MJCF string.  When omitted, ``env.mjcf.build_model``
        is used if it exists (M1), otherwise ``env._devmodel.build_dev_model`` for a
        single vehicle.  A model with several free-joint bodies gives several drones.
    n_drones:
        Number of vehicles to look for.  Names come from ``drone{i}`` when present,
        otherwise from the free-joint bodies in the model.
    dt:
        Control-facing plant step.  Quantised to ``n_sub * model.opt.timestep``.
    scene:
        Passed through to ``env.mjcf.build_model`` (M2 ``SceneSpec``); ``None`` lets
        each owner choose its default.
    seed:
        Sensor noise seed.  Turbulence has its own seed (``params.wind.seed``) or an
        explicit ``reset(..., wind_seed=...)``.
    sensor_noise:
        ``False`` returns clean samples (V3 determinism, debugging).
    rotor_pack, rotor_pack_factory:
        Actuator injection points (see :func:`_instantiate_default_pack`).  A single
        pack instance is deep-copied per drone; a callable is called as
        ``factory(index)`` then ``factory()`` then ``factory``.
    pack_frame:
        Frame of the ``v_rel_body`` handed to ``RotorPack.step``: ``"flu"`` (default,
        what ``env.actuation.RotorPack`` documents -- MuJoCo body axes) or ``"ned"``
        (the ``plant.py`` body axes, for packs ported from the Python plant).
    pack_torque:
        Meaning of ``RotorPack.step``'s ``torque_z``: ``"unsigned"`` (default, M1's
        contract: the per-rotor drag torque ``Q_i``, which the bridge dots with
        ``spin_dirs`` to get the yaw reaction) or ``"signed_flu"`` (a moment already
        signed about FLU ``+z``).
    drag, gyroscopic:
        Include the plant.py body drag / rotor-gyroscopic moment in the wrench.
    """

    #: Marker so callers can tell which backend they hold.
    backend = "mujoco"

    def __init__(
        self,
        params: Params | None = None,
        *,
        model: Any = None,
        xml: str | None = None,
        n_drones: int = 1,
        dt: float = 0.002,
        scene: Any = None,
        seed: int = 0,
        sensor_noise: bool = True,
        rotor_pack: Any = None,
        rotor_pack_factory: Callable[..., Any] | None = None,
        drone_names: Sequence[str] | None = None,
        initial_altitude: float = 5.0,
        pack_frame: str = "flu",
        pack_torque: str = "unsigned",
        drag: bool = True,
        gyroscopic: bool = True,
    ) -> None:
        import mujoco  # local import: mujoco stays an optional dependency

        self._mujoco = mujoco
        self.params: Params = params if params is not None else default_params("high")
        self.fidelity = "mujoco"
        self.n_drones = int(n_drones)
        if self.n_drones < 1:
            raise ValueError("n_drones must be >= 1")
        self.initial_altitude = float(initial_altitude)
        self.seed = int(seed)
        self.sensor_noise = bool(sensor_noise)
        self.pack_frame = str(pack_frame).lower()
        if self.pack_frame not in ("flu", "ned"):
            raise ValueError("pack_frame must be 'flu' or 'ned'")
        self.pack_torque = str(pack_torque).lower()
        if self.pack_torque not in ("unsigned", "signed_flu"):
            raise ValueError("pack_torque must be 'unsigned' or 'signed_flu'")
        self.include_drag = bool(drag)
        self.include_gyro = bool(gyroscopic)

        # -- world ---------------------------------------------------------
        self.model = self._resolve_model(model, xml, scene)
        self.data = mujoco.MjData(self.model)
        self._mj_step = float(self.model.opt.timestep)
        if not np.isfinite(self._mj_step) or self._mj_step <= 0.0:
            raise ValueError("model.opt.timestep must be positive and finite")
        n_sub = max(1, int(round(float(dt) / self._mj_step)))
        self._n_sub = n_sub
        #: Control-facing step.  Always an integer number of MuJoCo sub-steps.
        self.dt: float = n_sub * self._mj_step
        self._h_mj = self._mj_step

        # -- cached parameter constants ------------------------------------
        af = self.params.airframe
        self._mass = float(af.mass)
        self._g = float(af.g)
        self._rho = float(af.rho)
        self._arms = np.asarray(af.arm_positions, dtype=float)          # NED body
        self._spin = np.asarray(af.spin_dirs, dtype=float)              # +1 CW
        self._cda = np.asarray(self.params.aero.drag_area(), dtype=float)
        self._jr = float(af.rotor_inertia)

        # -- actuators ------------------------------------------------------
        self._pack_factory = self._make_pack_factory(rotor_pack, rotor_pack_factory)

        # -- vehicles -------------------------------------------------------
        self._drones: list[_Drone] = []
        names = self._resolve_drone_names(drone_names)
        for i, name in enumerate(names):
            body_id = self._body_id(name)
            jid, qposadr, dofadr = self._free_joint(body_id, name)
            pack = self._new_pack(self._pack_factory, i)
            sensors = _SensorModel(
                self.params, self.dt, seed=self.seed + 101 * i, enabled=self.sensor_noise
            )
            self._drones.append(
                _Drone(
                    name=name,
                    body_id=body_id,
                    qposadr=qposadr,
                    dofadr=dofadr,
                    pack=pack,
                    sensors=sensors,
                )
            )

        # -- wind / time ----------------------------------------------------
        self.wind = WindModel(self.params.wind, self.dt)
        self._wind = self.wind
        self._t = 0.0
        self._tick = 0
        self._period = max(1, int(round(float(self.params.control.h) / self.dt)))
        self._trim_cache: dict[float, dict] = {}

        self.reset()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _resolve_model(self, model: Any, xml: str | None, scene: Any) -> Any:
        if model is not None:
            return model
        if xml is not None:
            return self._mujoco.MjModel.from_xml_string(xml)
        scene = scene_argument(scene)
        try:
            from .mjcf import build_model
        except ImportError:
            build_model = None
        if build_model is not None:
            return build_model(self.params, scene=scene, n_drones=self.n_drones)
        if self.n_drones != 1:
            raise ImportError(
                "multi-vehicle worlds need env.mjcf.build_model (M1); until then pass "
                "model=<MjModel> or xml=<MJCF> explicitly"
            )
        from ._devmodel import build_dev_model

        return build_dev_model(self.initial_altitude, params=self.params)

    def _make_pack_factory(
        self, rotor_pack: Any, rotor_pack_factory: Callable[..., Any] | None
    ) -> Callable[[int], Any]:
        if rotor_pack_factory is not None:
            return lambda i: _call_factory(rotor_pack_factory, i)
        if rotor_pack is not None:
            if hasattr(rotor_pack, "step") and hasattr(rotor_pack, "reset"):
                if self.n_drones == 1:
                    return lambda i: rotor_pack
                return lambda i: copy.deepcopy(rotor_pack)
            if callable(rotor_pack):
                return lambda i: _call_factory(rotor_pack, i)
            raise TypeError("rotor_pack must be a pack instance or a factory")
        return lambda i: _instantiate_default_pack(self.params)

    @staticmethod
    def _new_pack(factory: Callable[[int], Any], index: int) -> Any:
        pack = factory(index)
        for attr in ("reset", "step"):
            if not hasattr(pack, attr):
                raise TypeError(f"rotor pack {pack!r} has no {attr}() (contract §4)")
        return pack

    def _body_id(self, name: str) -> int:
        try:
            return int(self.model.body(name).id)
        except Exception as exc:  # KeyError on missing bodies
            raise ValueError(f"model has no body named {name!r}") from exc

    def _free_joint(self, body_id: int, name: str) -> tuple[int, int, int]:
        mj = self._mujoco
        jntadr = int(self.model.body_jntadr[body_id])
        jntnum = int(self.model.body_jntnum[body_id])
        if jntnum != 1 or int(self.model.jnt_type[jntadr]) != int(
            mj.mjtJoint.mjJNT_FREE
        ):
            raise ValueError(
                f"body {name!r} must have exactly one free joint "
                f"(got {jntnum} joint(s)); this is the M1 build_model contract"
            )
        return jntadr, int(self.model.jnt_qposadr[jntadr]), int(self.model.jnt_dofadr[jntadr])

    def _resolve_drone_names(self, drone_names: Sequence[str] | None) -> list[str]:
        if drone_names is not None:
            names = [str(n) for n in drone_names]
            if len(names) != self.n_drones:
                raise ValueError("drone_names length must equal n_drones")
            return names

        found: list[str] = []
        for i in range(self.n_drones):
            name = None
            for candidate in (f"drone{i}", f"drone_{i}", f"quad{i}", f"quadrotor{i}"):
                try:
                    self.model.body(candidate)
                    name = candidate
                    break
                except Exception:
                    continue
            found.append(name)  # type: ignore[arg-type]
        if all(name is not None for name in found):
            return [str(n) for n in found]

        mj = self._mujoco
        free_bodies = [
            self.model.body(int(self.model.jnt_bodyid[j])).name
            for j in range(self.model.njnt)
            if int(self.model.jnt_type[j]) == int(mj.mjtJoint.mjJNT_FREE)
        ]
        if len(free_bodies) != self.n_drones:
            raise ValueError(
                f"model has {len(free_bodies)} free-joint bodies {free_bodies} but "
                f"n_drones={self.n_drones}; name them drone0..drone{self.n_drones - 1}"
            )
        return [str(n) for n in free_bodies]

    # ------------------------------------------------------------------
    # Plant protocol
    # ------------------------------------------------------------------
    @property
    def state(self) -> VehicleState:
        """Primary (drone 0) state, converted to NED at every step."""
        return self._drones[0].state

    @property
    def states(self) -> list[VehicleState]:
        return [d.state for d in self._drones]

    def state_of(self, index: int) -> VehicleState:
        return self._drones[int(index)].state

    def reset(
        self, state: VehicleState | Sequence[VehicleState] | None = None, *, wind_seed: int | None = None
    ) -> VehicleState:
        """Place the vehicle(s) in ``state`` (NED) and clear history.

        ``state`` may be one :class:`VehicleState` (broadcast to every drone) or one
        per drone.  As with :class:`QuadrotorPlant`, resetting clears the wind; use
        :meth:`set_wind` afterwards.
        """
        states = self._coerce_states(state)
        self._t = 0.0
        self._tick = 0
        for drone, st in zip(self._drones, states):
            q = quat_normalise(np.asarray(st.q, dtype=float).reshape(4))
            if q[0] < 0.0:
                q = -q
            pos_mj = ned_to_mj(np.asarray(st.xi, dtype=float).reshape(3))
            quat_mj = quat_ned_to_mj(q)
            self.data.qpos[drone.qposadr : drone.qposadr + 3] = pos_mj
            self.data.qpos[drone.qposadr + 3 : drone.qposadr + 7] = quat_mj
            self.data.qvel[drone.dofadr : drone.dofadr + 3] = ned_to_mj(st.v)
            self.data.qvel[drone.dofadr + 3 : drone.dofadr + 6] = ned_to_mj(st.omega)

            pwm = np.clip(np.asarray(st.pwm_us, dtype=float).reshape(4), PWM_MIN, PWM_MAX)
            _call_reset(drone.pack, pwm_us=pwm, soc=float(np.clip(st.soc, 0.0, 1.0)))
            drone.sensors.reset(seed=self.seed + 101 * self._drones.index(drone))
            drone.state = st.copy()
            drone.state.t = 0.0
            drone.last_aero = self._zero_aero(q)
            self._update_pack_fields(drone)

        self.data.xfrc_applied[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self.wind.reset(wind_seed)
        self.wind.set_turbulence(bool(self.params.wind.enabled))

        # Prime the aero wrench at t=0 so the first SensorSample carries the real
        # specific force (a zero wrench would tell the accelerometer it is in free
        # fall and the altitude estimator would see -g for one tick).  The pack is
        # reset again afterwards, so priming does not consume actuator state.
        wind0 = self.wind.at(0.0)
        for drone, st in zip(self._drones, states):
            pwm = np.clip(np.asarray(st.pwm_us, dtype=float).reshape(4), PWM_MIN, PWM_MAX)
            self._apply_wrench(drone, pwm, wind0)
            _call_reset(drone.pack, pwm_us=pwm, soc=float(np.clip(st.soc, 0.0, 1.0)))

        self._read_states()
        for drone in self._drones:
            self._push_sample(drone)
        return self.state.copy()

    def step(self, pwm_us: np.ndarray) -> VehicleState:
        """Advance one ``dt`` with four PWM commands in microseconds."""
        pwm = np.asarray(pwm_us, dtype=float)
        if not np.all(np.isfinite(pwm)):
            raise ValueError("pwm_us must be finite")
        if pwm.ndim == 1:
            pwm = np.tile(pwm.reshape(4), (self.n_drones, 1))
        pwm = np.clip(pwm.reshape(self.n_drones, 4), PWM_MIN, PWM_MAX)
        for row in pwm:
            if row.size != 4:  # pragma: no cover - shape guarded above
                raise ValueError("pwm_us must have 4 columns")

        for _ in range(self._n_sub):
            self._t += self._h_mj
            wind = self.wind.at(self._t)
            for i, drone in enumerate(self._drones):
                self._apply_wrench(drone, pwm[i], wind)
            self._mujoco.mj_step(self.model, self.data)
            self._read_states()

        self._tick += 1
        self._maybe_sample()
        return self.state.copy()

    def sense(self) -> SensorSample:
        """Delayed, corrupted sample for the primary vehicle (never internals)."""
        return self._drones[0].sensors.sense()

    def sense_of(self, index: int) -> SensorSample:
        return self._drones[int(index)].sensors.sense()

    def set_wind(self, steady: np.ndarray, turbulence: bool = True) -> None:
        """Set the steady wind (inertial/NED frame, m/s) and the gust switch."""
        self.wind.set_steady(np.asarray(steady, dtype=float).reshape(3))
        self.wind.set_turbulence(bool(turbulence))

    # ------------------------------------------------------------------
    # Diagnostics (validation suite / logging only -- not the Plant protocol)
    # ------------------------------------------------------------------
    def evaluate(self, state: VehicleState | None = None) -> dict:
        """Force/moment breakdown, radial to ``plant.QuadrotorPlant.evaluate``.

        ``state=None`` reports the wrench actually applied on the last sub-step.  A
        given ``state`` re-derives the geometry (drag, moments) for that attitude and
        velocity but re-uses the last rotor thrust, because advancing the pack would
        perturb the closed loop.
        """
        drone = self._drones[0]
        st = drone.state if state is None else state
        aero = dict(drone.last_aero)
        if state is not None:
            c_i2b = quat_to_rotmat(quat_normalise(np.asarray(state.q, dtype=float)))
            v_rel_body = aero.get("v_rel_body", np.zeros(3))
            f_drag = self._drag_force(v_rel_body)
            thrust = np.asarray(aero.get("T_rotor", np.zeros(4)), dtype=float)
            f_rotor = np.zeros((4, 3))
            f_rotor[:, 2] = -thrust
            aero.update(
                {
                    "c_i2b": c_i2b,
                    "F_drag_body": f_drag,
                    "F_body": f_rotor.sum(axis=0) + f_drag,
                    "M_rotor_arms": np.cross(self._arms, f_rotor).sum(axis=0),
                }
            )
        aero["ncon"] = int(self.data.ncon)
        aero["t"] = float(self._t)
        aero["height"] = float(-np.asarray(st.xi, dtype=float)[2])
        aero["soc"] = float(st.soc)
        aero["pwm_us"] = np.asarray(st.pwm_us, dtype=float).copy()
        return aero

    def trim_hover(self, soc: float = 1.0) -> dict:
        """Hover trim from the actuator pack (its steady-state fixed point).

        ``env.actuation.RotorPack.trim_hover`` owns the solve (ESC quantisation,
        battery OCV+sag, four-rotor coupled current); the bridge forwards to it and
        only wraps the result in a ``VehicleState``.  Packs that do not expose a trim
        (the contract only requires ``reset``/``step``) get a numerical bisection
        driven through ``pack.step``, so the number still comes from the pack and
        never from plant internals.

        At ``soc=1`` this reproduces ``QuadrotorPlant.trim_hover`` (pwm ~= 1394 us),
        which is what makes the V8 cross-check apples-to-apples.  Returned keys match
        :meth:`pistr_sim.plant.QuadrotorPlant.trim_hover`.
        """
        soc = float(np.clip(soc, 0.0, 1.0))
        key = round(soc, 6)
        cached = self._trim_cache.get(key)
        if cached is not None:
            out = dict(cached)
            out["state"] = cached["state"].copy()
            return out

        target = self._mass * self._g
        scratch = self._new_pack(self._pack_factory, 0)
        pack_trim = getattr(scratch, "trim_hover", None)
        if callable(pack_trim):
            info = dict(pack_trim(soc=soc))
            pwm_us = float(np.clip(float(info["pwm_us"]), PWM_MIN, PWM_MAX))
            omega = _as_vec4(info.get("omega"), default=0.0)
            duty = _as_vec4(
                info.get("esc_duty"),
                default=float(np.clip((pwm_us - PWM_MIN) / PWM_MIN, 0.0, 1.0)),
            )
            current = _as_vec4(info.get("current"), default=0.0)
            v_terminal = float(info.get("v_terminal", self.params.motor.u_max))
            thrust = _as_scalar_thrust(info.get("thrust"))
            if not np.isfinite(thrust):
                thrust = target
        else:
            pwm_us, thrust, omega, duty, current, v_terminal = self._trim_numerically(
                scratch, target, soc
            )

        state = VehicleState(
            t=0.0,
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=np.zeros(3),
            v=np.zeros(3),
            xi=np.zeros(3),
            rotor_omega=omega.copy(),
            rotor_current=current.copy(),
            esc_duty=duty.copy(),
            pwm_us=np.full(4, pwm_us),
            soc=soc,
            v_terminal=v_terminal,
        )
        out = {
            "pwm_us": float(pwm_us),
            "percent": float((pwm_us - PWM_MIN) / PWM_MIN),
            "rotor_omega": float(np.mean(omega)),
            "rotor_current": float(np.mean(current)),
            "esc_duty": float(np.mean(duty)),
            "v_terminal": float(v_terminal),
            "thrust": float(thrust),
            "hover_error": float(thrust - target),
            "state": state,
        }
        self._trim_cache[key] = dict(out)
        return out

    def _trim_numerically(
        self, scratch: Any, target: float, soc: float
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Fallback trim for a pack without ``trim_hover``: bisect through ``step``."""
        n_settle = max(16, int(round(0.4 / self._h_mj)))
        pwm_vec = np.full(4, self.params.pwm_hover_us)

        def total_thrust(pwm: float) -> float:
            pwm_vec[:] = pwm
            _call_reset(scratch, pwm_us=pwm_vec, soc=soc)
            thrust = np.zeros(4)
            for _ in range(n_settle):
                thrust, _ = scratch.step(pwm_vec, self._h_mj, v_rel_body=np.zeros(3))
            return float(np.sum(np.maximum(np.asarray(thrust, dtype=float), 0.0)))

        lo, hi = float(PWM_MIN), float(PWM_MAX)
        f_lo, f_hi = total_thrust(lo), total_thrust(hi)
        if not (f_lo < target < f_hi):
            pwm_us = float(np.clip(self.params.pwm_hover_us, PWM_MIN, PWM_MAX))
            thrust = total_thrust(pwm_us)
        else:
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if total_thrust(mid) < target:
                    lo = mid
                else:
                    hi = mid
            pwm_us = 0.5 * (lo + hi)
            thrust = total_thrust(pwm_us)

        pwm_vec[:] = pwm_us
        ctrl_h = max(1, int(round(float(self.params.control.h) / self._h_mj)))
        for _ in range(ctrl_h):  # settle the reported rotor state at the trim point
            scratch.step(pwm_vec, self._h_mj, v_rel_body=np.zeros(3))
        omega = _pack_vector(scratch, ("omega", "rotor_omega", "w"), 4)
        duty = _pack_vector(
            scratch, ("esc_duty", "duty"), 4, default=(pwm_us - PWM_MIN) / PWM_MIN
        )
        current = _pack_vector(scratch, ("rotor_current", "current"), 4)
        v_terminal = _pack_scalar(scratch, ("v_terminal",), default=float(self.params.motor.u_max))
        return pwm_us, thrust, omega, duty, current, v_terminal

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _coerce_states(
        self, state: VehicleState | Sequence[VehicleState] | None
    ) -> list[VehicleState]:
        if state is None:
            base = VehicleState(
                t=0.0,
                q=np.array([1.0, 0.0, 0.0, 0.0]),
                xi=np.array([0.0, 0.0, -self.initial_altitude]),
            )
            # Start at the real hover trim, not params.pwm_hover_us (which assumes
            # u_max and no pack sag, i.e. ~1183 us): resetting the pack to a command
            # far below hover leaves a ~0.3 m/s sink transient for the first 0.2 s.
            try:
                trim_pwm = float(self.trim_hover()["pwm_us"])
            except Exception:  # pragma: no cover - trim must not break reset()
                trim_pwm = float(self.params.pwm_hover_us)
            base.pwm_us = np.full(4, min(max(trim_pwm, PWM_MIN), PWM_MAX))
            base.soc = 1.0
            return [base.copy() for _ in self._drones]
        if isinstance(state, VehicleState):
            return [state.copy() for _ in self._drones]
        states = [s.copy() for s in state]  # type: ignore[union-attr]
        if len(states) != self.n_drones:
            raise ValueError(f"expected {self.n_drones} states, got {len(states)}")
        return states

    def _drag_force(self, v_rel_body: np.ndarray) -> np.ndarray:
        if not self.include_drag:
            return np.zeros(3)
        v = np.asarray(v_rel_body, dtype=float).reshape(3)
        return -0.5 * self._rho * self._cda * np.abs(v) * v

    def _apply_wrench(self, drone: _Drone, pwm: np.ndarray, wind_ned: np.ndarray) -> None:
        """One actuator sub-step: advance the pack, apply the world wrench."""
        st = drone.state
        st.pwm_us = np.asarray(pwm, dtype=float).reshape(4).copy()
        c_i2b = quat_to_rotmat(quat_normalise(np.asarray(st.q, dtype=float)))
        c_b2i = c_i2b.T
        v_rel_ned = np.asarray(st.v, dtype=float) - np.asarray(wind_ned, dtype=float)
        v_rel_body = c_i2b @ v_rel_ned

        if self.pack_frame == "flu":
            v_rel_pack = S_NED_TO_FLU @ v_rel_body
        else:
            v_rel_pack = v_rel_body

        thrust_raw, torque_raw = drone.pack.step(pwm, self._h_mj, v_rel_body=v_rel_pack)
        thrust = np.maximum(np.asarray(thrust_raw, dtype=float).reshape(4), 0.0)
        q_rot = np.asarray(torque_raw, dtype=float).reshape(4)

        # -- body wrench in NED, assembled exactly like plant.py §4 ---------
        f_rotor = np.zeros((4, 3))
        f_rotor[:, 2] = -thrust                       # thrust along body -z (NED)
        f_drag = self._drag_force(v_rel_body)
        f_body = f_rotor.sum(axis=0) + f_drag
        m_arms = np.cross(self._arms, f_rotor).sum(axis=0)

        if self.pack_torque == "unsigned":
            # M1's contract: Q_i is the per-rotor drag torque; dir_i Q_i is the yaw
            # reaction about NED +z (which is FLU -z).
            m_yaw_z = float(np.dot(self._spin, q_rot))
        else:  # "signed_flu": torque_z is already a moment about FLU +z
            m_yaw_z = -float(np.sum(q_rot))
        m_yaw = np.array([0.0, 0.0, m_yaw_z])

        if self.include_gyro:
            omega_pack = _pack_vector(drone.pack, ("omega", "rotor_omega", "w"), 4)
            if np.any(omega_pack):
                h_rotor = self._jr * float(np.dot(self._spin, omega_pack)) * _EZ
                m_gyro = -np.cross(np.asarray(st.omega, dtype=float), h_rotor)
            else:
                h_rotor = np.zeros(3)
                m_gyro = np.zeros(3)
        else:
            omega_pack = np.zeros(4)
            h_rotor = np.zeros(3)
            m_gyro = np.zeros(3)

        m_body = m_arms + m_yaw + m_gyro

        f_world = ned_to_mj(c_b2i @ f_body)
        m_world = ned_to_mj(c_b2i @ m_body)
        self.data.xfrc_applied[drone.body_id, :3] = f_world
        self.data.xfrc_applied[drone.body_id, 3:] = m_world

        drone.last_aero = {
            "thrust": float(np.sum(thrust)),
            "T_rotor": thrust,
            "Q_rotor": q_rot,
            "M_yaw": m_yaw,
            "M_rotor_arms": m_arms,
            "M_gyro": m_gyro,
            "h_rotor": h_rotor,
            "F_drag_body": f_drag,
            "F_body": f_body,
            "M_body": m_body,
            "v_rel_body": v_rel_body,
            "v_rel_pack": v_rel_pack,
            "c_i2b": c_i2b,
            "F_world_mj": f_world,
            "M_world_mj": m_world,
            "wind_ned": np.asarray(wind_ned, dtype=float).copy(),
            "rotor_omega": omega_pack.copy(),
            "dt": self._h_mj,
        }

    def _zero_aero(self, q: np.ndarray) -> dict:
        zeros4 = np.zeros(4)
        c_i2b = quat_to_rotmat(quat_normalise(np.asarray(q, dtype=float)))
        return {
            "thrust": 0.0,
            "T_rotor": zeros4.copy(),
            "Q_rotor": zeros4.copy(),
            "M_yaw": np.zeros(3),
            "M_rotor_arms": np.zeros(3),
            "M_gyro": np.zeros(3),
            "h_rotor": np.zeros(3),
            "F_drag_body": np.zeros(3),
            "F_body": np.zeros(3),
            "M_body": np.zeros(3),
            "v_rel_body": np.zeros(3),
            "v_rel_pack": np.zeros(3),
            "c_i2b": c_i2b,
            "F_world_mj": np.zeros(3),
            "M_world_mj": np.zeros(3),
            "wind_ned": np.zeros(3),
            "rotor_omega": np.zeros(4),
            "dt": self._h_mj,
        }

    def _read_states(self) -> None:
        d = self.data
        for drone in self._drones:
            st = drone.state
            st.t = float(self._t)
            st.xi = mj_to_ned(d.qpos[drone.qposadr : drone.qposadr + 3])
            st.q = quat_mj_to_ned(d.qpos[drone.qposadr + 3 : drone.qposadr + 7])
            st.v = mj_to_ned(d.qvel[drone.dofadr : drone.dofadr + 3])
            st.omega = mj_to_ned(d.qvel[drone.dofadr + 3 : drone.dofadr + 6])
            self._update_pack_fields(drone)

    def _update_pack_fields(self, drone: _Drone) -> None:
        st = drone.state
        st.rotor_omega = _pack_vector(drone.pack, ("omega", "rotor_omega", "w"), 4)
        st.rotor_current = _pack_vector(drone.pack, ("rotor_current", "current"), 4)
        st.esc_duty = _pack_vector(
            drone.pack,
            ("esc_duty", "duty"),
            4,
            default=float(np.clip((np.mean(st.pwm_us) - PWM_MIN) / PWM_MIN, 0.0, 1.0)),
        )
        st.soc = float(np.clip(_pack_scalar(drone.pack, ("soc",), default=st.soc), 0.0, 1.0))
        st.v_terminal = _pack_scalar(
            drone.pack, ("v_terminal",), default=float(self.params.motor.u_max)
        )

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    def _maybe_sample(self) -> None:
        if self._tick % self._period == 0:
            for drone in self._drones:
                self._push_sample(drone)

    def _push_sample(self, drone: _Drone) -> None:
        st = drone.state
        aero = drone.last_aero
        c_i2b = np.asarray(aero.get("c_i2b", quat_to_rotmat(st.q)), dtype=float)
        # Specific force in the body frame: the accelerometer reads the aero wrench,
        # never gravity, so a level hover reads (0, 0, -g) like the Python plant.
        accel_clean = np.asarray(aero.get("F_body", np.zeros(3)), dtype=float) / self._mass
        mag_clean = c_i2b @ MAG_EARTH_UT_NED
        drone.sensors.push(
            float(self._t),
            gyro=np.asarray(st.omega, dtype=float),
            accel=accel_clean,
            mag=mag_clean,
            baro_alt=float(-st.xi[2]),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def pwm(self) -> np.ndarray:
        """Current PWM command per drone, ``(n_drones, 4)``."""
        return np.stack([np.asarray(d.state.pwm_us, dtype=float) for d in self._drones])

    @property
    def contact_count(self) -> int:
        return int(self.data.ncon)

    @property
    def time(self) -> float:
        return float(self._t)

    def height(self, index: int = 0) -> float:
        """Altitude of drone ``index`` in metres (positive up)."""
        return float(-self._drones[int(index)].state.xi[2])

    def thrust(self, index: int = 0) -> float:
        """Total rotor thrust (N) applied to drone ``index`` on the last sub-step."""
        return float(self._drones[int(index)].last_aero.get("thrust", 0.0))

    def aero_of(self, index: int = 0) -> dict:
        """Last computed aero wrench of drone ``index`` (diagnostics)."""
        return dict(self._drones[int(index)].last_aero)

    def mj_pose(self, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """``(position, body->world rotation)`` of drone ``index`` in MuJoCo coordinates.

        Kept separate from :attr:`state` on purpose: ``state`` is NED, this is the
        raw MuJoCo truth that V3 checks the mapping against and that a camera rig
        needs.
        """
        bid = self._drones[int(index)].body_id
        return (
            np.asarray(self.data.xpos[bid], dtype=float).copy(),
            np.asarray(self.data.xmat[bid], dtype=float).reshape(3, 3).copy(),
        )

    def mj_velocity(self, index: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """``(world linear velocity, body angular velocity)`` in MuJoCo coordinates."""
        drone = self._drones[int(index)]
        return (
            np.asarray(self.data.qvel[drone.dofadr : drone.dofadr + 3], dtype=float).copy(),
            np.asarray(self.data.qvel[drone.dofadr + 3 : drone.dofadr + 6], dtype=float).copy(),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        names = [d.name for d in self._drones]
        return (
            f"MuJoCoPlant(n_drones={self.n_drones}, bodies={names}, "
            f"dt={self.dt:g}, mj_timestep={self._h_mj:g}, n_sub={self._n_sub})"
        )


# ---------------------------------------------------------------------------
# Small duck-typing helpers.  ``RotorPack.reset``/``step`` are keyword-only by
# contract (§4) and the constructor is ``RotorPack(params)`` (frozen with M1), so
# only the *factory* shape needs adapting: callers may pass ``factory()`` or
# ``factory(index)``.
# ---------------------------------------------------------------------------
def _call_factory(factory: Callable[..., Any], index: int) -> Any:
    try:
        return factory(index)
    except TypeError:
        return factory()


def _call_reset(pack: Any, *, pwm_us: np.ndarray | None = None, soc: float = 1.0) -> None:
    """``RotorPack.reset(*, pwm_us=None, soc=1.0)`` -- steady state at that command."""
    pack.reset(pwm_us=pwm_us, soc=soc)
