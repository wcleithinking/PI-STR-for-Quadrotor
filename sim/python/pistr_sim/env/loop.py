"""Closed-loop flight in the 3-D world: ``run_flight`` + ``FlightTrace`` (M4).

The point of this module is that **no control code is special-cased for MuJoCo**.
The control stack is exactly the one the rest of the repository uses
(``estimator.QuaternionAttitudeEstimator`` + ``estimator.AltitudeEstimator`` +
``experiments.controllers.build_controller`` + ``control.U_MIX``/``ALTITUDE_SIGN``),
driven at the normative STR sampling period ``params.control.h``; the only thing
that changes between backends is the ``plant`` object handed to it.  That is what
makes the V8 cross-check meaningful: the same loop flies
``MuJoCoPlant`` and ``plant.QuadrotorPlant("high")`` and the numbers are compared.

Scenarios
---------
``hover``       hold attitude zero at ``altitude0`` (the V8 gate).
``step``        hover, then a roll step to 0.1 rad at ``t_step`` (CONTROL_SPEC §8).
``sine``        the paper's sinusoidal attitude/altitude reference.
``trajectory``  the paper's 3-D position reference.  There is **no GPS**: the
                horizontal loop needs position/velocity, so this scenario uses the
                plant's *true* position as an ideal aid and says so in
                ``metadata["position_aid"]`` (same modelling assumption as
                ``experiments.loop.run_trajectory``).  Hover/step/sine need no
                position measurement at all.
``gust``        hover plus a wind gust window (``gust_ned``/``gust_t``).

Multi-vehicle
-------------
``n_drones`` vehicles live in one world (independent free joints, one controller
and one rotor pack each, same stack), offset along the NED x axis by
``drone_spacing`` so they are distinguishable in a frame.  ``FlightTrace`` carries
one :class:`DroneTrace` timeline per vehicle.

Frames
------
Everything recorded in the trace is **NED** (``xi[2]`` is negative altitude, quats
are the repository's inertial->body ordering).  MuJoCo coordinates never leak out of
``env/bridge.py``; rendering gets the ``MjData`` through ``on_frame`` and reads
MuJoCo's own frames.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from ..control import U_MIX, clamp_pwm, recover_u_actual
from ..estimator import (
    AltitudeConfig,
    AltitudeEstimator,
    QuaternionAttitudeEstimator,
    _quat_from_euler,
)
from ..params import Params, default_params
from ..reference import ReferenceSample, TrajectoryReference3D, make_reference
from ..types import quat_to_rotmat
from .bridge import MuJoCoPlant, scene_argument

__all__ = [
    "ALTITUDE_SIGN",
    "DroneTrace",
    "FlightTrace",
    "SCENARIOS",
    "run_flight",
]

#: Channel sign map used by every controller in this repository.
ALTITUDE_SIGN = np.array([1.0, 1.0, 1.0, -1.0])

#: Earth magnetic field for the estimator (same constant as ``plant.py``), uT.
MAG_EARTH_UT = (45.0, 0.0, 20.0)

SCENARIOS = ("hover", "step", "sine", "trajectory", "gust")

#: Controllers whose discrete model is tied to ``params.control.h``.
_STR_FAMILY = ("str", "pi_str", "pi_str_1axis", "pi_str_3axis")


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _window_mask(t: np.ndarray, window: tuple[float, float] | None) -> np.ndarray:
    if window is None:
        return np.ones(t.shape, dtype=bool)
    lo, hi = float(window[0]), float(window[1])
    return (t >= lo - 1e-12) & (t <= hi + 1e-12)


# ---------------------------------------------------------------------------
# Trace containers
# ---------------------------------------------------------------------------
@dataclass
class DroneTrace:
    """One vehicle's timeline, all vectors in NED."""

    name: str
    t: np.ndarray                 # (T,)
    xi: np.ndarray                # (T,3) NED position, xi[2] = -altitude
    v: np.ndarray                 # (T,3) NED velocity
    quat: np.ndarray              # (T,4) inertial->body (MATLAB ordering)
    euler: np.ndarray             # (T,3) roll/pitch/yaw, rad
    omega: np.ndarray             # (T,3) body rates, rad/s
    pwm: np.ndarray               # (T,4) commanded PWM, us
    rotor_omega: np.ndarray       # (T,4) rad/s (from the pack)
    thrust: np.ndarray            # (T,) total rotor thrust, N
    y_ref: np.ndarray             # (T,4) [roll, pitch, yaw, altitude]
    rate_ref: np.ndarray          # (T,4)
    y: np.ndarray                 # (T,4) truth [roll, pitch, yaw, altitude]
    est_y: np.ndarray             # (T,4) what the controller saw
    u: np.ndarray                 # (T,4) design command
    u_actual: np.ndarray          # (T,4) realised command after PWM clipping
    pos_ref: np.ndarray           # (T,3) NED position reference
    wind: np.ndarray              # (T,3) NED wind at each tick

    # -- derived ---------------------------------------------------------
    @property
    def altitude(self) -> np.ndarray:
        """Altitude in metres, positive up."""
        return -self.xi[:, 2]

    def attitude_error(self, window: tuple[float, float] | None = None) -> np.ndarray:
        mask = _window_mask(self.t, window)
        return _wrap(self.euler[mask] - self.y_ref[mask, :3])

    def attitude_rmse(self, window: tuple[float, float] | None = None) -> np.ndarray:
        """Per-axis RMS attitude error against the reference, rad."""
        err = self.attitude_error(window)
        return np.sqrt(np.mean(err**2, axis=0))

    def altitude_error(self, window: tuple[float, float] | None = None) -> np.ndarray:
        mask = _window_mask(self.t, window)
        return self.altitude[mask] - self.y_ref[mask, 3]

    def altitude_stats(self, window: tuple[float, float] | None = None) -> dict:
        mask = _window_mask(self.t, window)
        err = self.altitude_error(window)
        alt = self.altitude[mask]
        return {
            "drift_max": float(np.max(np.abs(err))) if err.size else float("nan"),
            "mean_abs": float(np.mean(np.abs(err))) if err.size else float("nan"),
            "final_altitude": float(alt[-1]) if alt.size else float("nan"),
            "min_altitude": float(alt.min()) if alt.size else float("nan"),
            "max_altitude": float(alt.max()) if alt.size else float("nan"),
        }

    def summary(self, window: tuple[float, float] | None = None) -> dict:
        rmse = self.attitude_rmse(window)
        alt = self.altitude_stats(window)
        return {
            "name": self.name,
            "attitude_rmse_deg": {
                "roll": float(np.rad2deg(rmse[0])),
                "pitch": float(np.rad2deg(rmse[1])),
                "yaw": float(np.rad2deg(rmse[2])),
            },
            "altitude": alt,
            "pwm_range": [float(self.pwm.min()), float(self.pwm.max())],
            "rotor_omega_mean": float(np.mean(self.rotor_omega)),
            "thrust_mean": float(np.mean(self.thrust)),
        }


@dataclass
class FlightTrace:
    """``run_flight`` result: one :class:`DroneTrace` per vehicle plus run metadata."""

    t: np.ndarray
    drones: list[DroneTrace]
    wind: np.ndarray
    scenario: str
    controller: str
    metadata: dict = field(default_factory=dict)

    # -- access ----------------------------------------------------------
    @property
    def n_drones(self) -> int:
        return len(self.drones)

    def drone(self, index: int = 0) -> DroneTrace:
        return self.drones[int(index)]

    @property
    def xi(self) -> np.ndarray:
        return self.drones[0].xi

    @property
    def v(self) -> np.ndarray:
        return self.drones[0].v

    @property
    def quat(self) -> np.ndarray:
        return self.drones[0].quat

    @property
    def euler(self) -> np.ndarray:
        return self.drones[0].euler

    @property
    def omega(self) -> np.ndarray:
        return self.drones[0].omega

    @property
    def pwm(self) -> np.ndarray:
        return self.drones[0].pwm

    @property
    def rotor_omega(self) -> np.ndarray:
        return self.drones[0].rotor_omega

    @property
    def altitude(self) -> np.ndarray:
        return self.drones[0].altitude

    def stacked(self, name: str) -> np.ndarray:
        """Stack one field over all drones, ``(n_drones, T, ...)``."""
        return np.stack([np.asarray(getattr(d, name)) for d in self.drones])

    # -- metrics ---------------------------------------------------------
    def attitude_rmse(self, window: tuple[float, float] | None = None) -> np.ndarray:
        """Per-axis RMS attitude error of drone 0, rad (V8 uses this)."""
        return self.drones[0].attitude_rmse(window)

    def altitude_stats(self, window: tuple[float, float] | None = None) -> dict:
        return self.drones[0].altitude_stats(window)

    def summary(self, window: tuple[float, float] | None = None) -> dict:
        return {
            "scenario": self.scenario,
            "controller": self.controller,
            "n_drones": self.n_drones,
            "t_end": float(self.t[-1]) if self.t.size else 0.0,
            "drones": [d.summary(window) for d in self.drones],
            "metadata": self.metadata,
        }

    def write_json(self, path: str, window: tuple[float, float] | None = None) -> str:
        """Write :meth:`summary` (plus the trace metadata) as JSON."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = self.summary(window)
        payload["metadata"] = {
            key: value
            for key, value in self.metadata.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Position loop (trajectory scenario only -- needs position aiding)
# ---------------------------------------------------------------------------
class _PositionLoop:
    """PD position -> roll/pitch planner with the corrected signs.

    The published MATLAB planner's ``asin(u_x / (-m g))`` has the opposite sign for
    this plant (``docs/CONTROL_SPEC.md`` §9.9); this is the exact world->body mapping
    used by ``experiments.loop`` for the same reason.
    """

    def __init__(self, mass: float, gravity: float, *, k1: float = 0.2**2, k2: float = 0.4,
                 max_tilt: float = np.deg2rad(35.0), eso_pole: float = -2.0, h: float = 0.04,
                 use_eso_after: float = 2.0):
        from ..reference import ESO

        self.mass = float(mass)
        self.g = float(gravity)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.max_tilt = float(max_tilt)
        self._eso_roll = ESO(pole=eso_pole, h=h)
        self._eso_pitch = ESO(pole=eso_pole, h=h)
        self._use_eso_after = float(use_eso_after)
        self._t0: float | None = None

    def reset(self) -> None:
        self._eso_roll.reset()
        self._eso_pitch.reset()
        self._t0 = None

    def sample(
        self,
        ref: ReferenceSample,
        xi: Sequence[float],
        v: Sequence[float],
        t: float,
    ) -> ReferenceSample:
        xi = np.asarray(xi, dtype=float).reshape(3)
        v = np.asarray(v, dtype=float).reshape(3)
        a_des = ref.a_xi - self.k1 * (xi - ref.xi) - self.k2 * (v - ref.v_xi)
        n = np.array([-a_des[0], -a_des[1], self.g - a_des[2]])
        n /= max(float(np.linalg.norm(n)), 1e-9)
        pitch = float(np.arcsin(np.clip(-n[0], -np.sin(self.max_tilt), np.sin(self.max_tilt))))
        c = max(float(np.cos(pitch)), 1e-6)
        roll = float(np.arcsin(np.clip(-n[1] / c, -np.sin(self.max_tilt), np.sin(self.max_tilt))))
        _, roll_rate = self._eso_roll.step(roll)
        _, pitch_rate = self._eso_pitch.step(pitch)
        if self._t0 is None:
            self._t0 = float(t)
        if float(t) - self._t0 <= self._use_eso_after:
            roll_rate = pitch_rate = 0.0
        ref.roll, ref.pitch = roll, pitch
        ref.roll_rate, ref.pitch_rate = roll_rate, pitch_rate
        return ref


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_flight(
    *,
    controller: str = "pi_str",
    scenario: str = "hover",
    scene: Any = None,
    t_end: float = 8.0,
    dt: float = 0.002,
    control_h: float | None = None,
    seed: int = 0,
    n_drones: int = 1,
    camera: Any = None,
    out: str | None = None,
    video: str | None = None,
    plant: Any = None,
    model: Any = None,
    params: Params | None = None,
    altitude0: float = 5.0,
    drone_spacing: float = 2.0,
    on_frame: Callable[[float, Any], None] | None = None,
    frame_stride: int = 1,
    rotor_pack: Any = None,
    rotor_pack_factory: Callable[..., Any] | None = None,
    step_amplitude: float = 0.1,
    step_time: float = 1.5,
    gust_ned: Sequence[float] = (0.0, 0.0, 0.0),
    gust_t: tuple[float, float] = (1.0, 2.0),
    position_aid: bool | None = None,
    verbose: bool = False,
) -> FlightTrace:
    """Fly ``controller`` in the 3-D world and return the recorded :class:`FlightTrace`.

    Parameters mirror ``docs/3D_ENVIRONMENT.md`` §4.  ``control_h`` defaults to
    ``params.control.h`` (40 ms): the STR channels identify a discrete model *at that
    sampling period*, so feeding them at a different rate silently invalidates the
    controller (a warning is raised if an explicit different value is used with an
    STR-family controller).  ``dt`` is the physics/plant step handed to
    :class:`MuJoCoPlant` (quantised to whole MuJoCo sub-steps).

    ``plant`` selects the backend: ``None`` builds a :class:`MuJoCoPlant`
    (``model``/``rotor_pack*``/``scene`` are forwarded), ``"high"``/``"paper"`` build
    the corresponding Python plant, or any QuadrotorPlant-compatible object is used
    as-is (this is how V8 compares the two worlds).

    ``on_frame(t, data)`` is called every ``frame_stride`` control ticks with the live
    ``mujoco.MjData`` (``data`` is ``None`` for the pure-Python backend) -- this is
    the M5 recording hook.

    ``camera``: if it is an M3 rig (it has ``set_target``), run_flight drives it
    itself once per frame -- ``tick(frame_stride * control_h)`` then
    ``set_target(drone0_position, yaw=...)`` in MuJoCo coordinates.  ``camera=None``
    means "not my business": drive your own rig from ``on_frame``.

    ``scene`` accepts M2's ``SceneSpec`` (converted to MJCF text with
    ``env.scene.scene_xml``) as well as an XML string; ``out`` (a directory) gets a
    ``flight_summary.json`` and ``video`` (a path) is rendered through ``env.render``
    when that module is available.
    """
    started = time.perf_counter()
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    params = params if params is not None else default_params("high")
    h = float(params.control.h if control_h is None else control_h)
    if h <= 0.0:
        raise ValueError("control_h must be positive")
    if controller.lower() in _STR_FAMILY and abs(h - float(params.control.h)) > 1e-12:
        warnings.warn(
            f"controller {controller!r} identifies its discrete model at "
            f"params.control.h={params.control.h:g} s, but run_flight got "
            f"control_h={h:g} s; the STR loop is not tuned for that rate",
            RuntimeWarning,
            stacklevel=2,
        )
    g = float(params.airframe.g)
    if n_drones < 1:
        raise ValueError("n_drones must be >= 1")

    # -- backend ---------------------------------------------------------
    # ``build_model(scene=...)`` wants MJCF text, not M2's dataclass; convert here so
    # callers (CLI, M5 demos) can pass ``SceneSpec(...)`` straight through.
    scene = scene_argument(scene)
    if plant is None:
        plant = MuJoCoPlant(
            params=params,
            model=model,
            dt=dt,
            seed=seed,
            n_drones=n_drones,
            scene=scene,
            rotor_pack=rotor_pack,
            rotor_pack_factory=rotor_pack_factory,
            initial_altitude=altitude0,
        )
        backend = "mujoco"
    elif isinstance(plant, str):
        from ..plant import make_plant

        plant = make_plant(plant, seed=seed)
        backend = str(plant.fidelity)
    else:
        backend = str(getattr(plant, "backend", getattr(plant, "fidelity", "external")))
    multi = int(getattr(plant, "n_drones", 1))
    if n_drones > 1 and multi < n_drones:
        raise ValueError(
            f"plant only has {multi} vehicle(s); pass n_drones={multi} or a multi-vehicle model"
        )
    if multi > 1 and not hasattr(plant, "sense_of"):
        raise ValueError("multi-vehicle control needs a plant with sense_of(index)")

    plant_dt = float(plant.dt)
    sub = max(1, int(round(h / plant_dt)))
    n_ticks = int(np.floor(t_end / h + 1e-9)) + 1
    if n_ticks < 1:
        raise ValueError("t_end must be >= control_h")

    # -- vehicles: trim, state, controller, estimators -------------------
    trim = plant.trim_hover()
    throttle = float(trim["pwm_us"])
    if not np.isfinite(throttle):
        raise ValueError("trim_hover returned a non-finite throttle")
    if scenario == "trajectory" and n_drones > 1:
        raise ValueError("scenario='trajectory' is single-vehicle in this runner")

    # The trajectory reference defines its own altitude envelope (the paper's
    # -20 m + 10 m vertical sweep), so start *on* it instead of at altitude0.
    ref0 = None
    xi0 = np.zeros(3)
    if scenario == "trajectory":
        ref0 = _make_reference(scenario, altitude0, step_amplitude, step_time).sample(0.0)
        xi0 = np.asarray(ref0.xi, dtype=float).reshape(3)
        altitude0 = float(-xi0[2])

    offsets = (np.arange(n_drones) - 0.5 * (n_drones - 1)) * float(drone_spacing)
    states = []
    for i in range(n_drones):
        st = trim["state"].copy()
        if ref0 is None:
            st.xi = np.array([float(offsets[i]), 0.0, -float(altitude0)])
            st.v = np.zeros(3)
        else:
            st.xi = xi0 + np.array([float(offsets[i]), 0.0, 0.0])
            st.v = np.asarray(ref0.v_xi, dtype=float).reshape(3).copy()
        st.omega = np.zeros(3)
        st.t = 0.0
        states.append(st)
    plant.reset(state=states[0] if n_drones == 1 else states)

    references = [_make_reference(scenario, altitude0, step_amplitude, step_time) for _ in range(n_drones)]
    planners: list[_PositionLoop | None] = [None] * n_drones
    if scenario == "trajectory":
        if position_aid is False:
            raise ValueError("scenario='trajectory' needs position aiding (position_aid=True)")
        position_aid = True
        for i in range(n_drones):
            planners[i] = _PositionLoop(
                params.airframe.mass, g, h=h, use_eso_after=2.0
            )
    elif position_aid is None:
        position_aid = False

    from ..experiments.controllers import build_controller

    controllers = []
    attitude_estimators = []
    altitude_estimators = []
    for i in range(n_drones):
        ctrl = build_controller(controller, params, "high")
        ctrl.reset()
        controllers.append(ctrl)
        attitude_estimators.append(
            QuaternionAttitudeEstimator(dt=h, mag_reference=MAG_EARTH_UT)
        )
        altitude_estimators.append(AltitudeEstimator(AltitudeConfig(dt=h)))
        altitude_estimators[i].reset(float(altitude0), 0.0)

    gust_vector = np.asarray(gust_ned, dtype=float).reshape(3)
    gust_active = scenario == "gust" and bool(np.any(gust_vector != 0.0))
    if gust_active:
        # The window is applied inside the tick loop: a steady wind from
        # ``gust_t[0]`` to ``gust_t[1]``, with the Dryden turbulence left on
        # throughout. Applying it here for the whole run would contradict
        # ``docs/3D_ENVIRONMENT.md`` §4.
        plant.set_wind(np.zeros(3), turbulence=True)

    # -- storage ---------------------------------------------------------
    def _zeros(n: int) -> np.ndarray:
        return np.zeros(n)

    rec = {
        "t": _zeros(n_ticks),
        "xi": np.zeros((n_ticks, 3)),
        "v": np.zeros((n_ticks, 3)),
        "quat": np.zeros((n_ticks, 4)),
        "euler": np.zeros((n_ticks, 3)),
        "omega": np.zeros((n_ticks, 3)),
        "pwm": np.zeros((n_ticks, 4)),
        "rotor_omega": np.zeros((n_ticks, 4)),
        "thrust": _zeros(n_ticks),
        "y_ref": np.zeros((n_ticks, 4)),
        "rate_ref": np.zeros((n_ticks, 4)),
        "y": np.zeros((n_ticks, 4)),
        "est_y": np.zeros((n_ticks, 4)),
        "u": np.zeros((n_ticks, 4)),
        "u_actual": np.zeros((n_ticks, 4)),
        "pos_ref": np.zeros((n_ticks, 3)),
        "wind": np.zeros((n_ticks, 3)),
    }
    records = [{key: value.copy() for key, value in rec.items()} for _ in range(n_drones)]

    # -- optional video (M3's renderer, frames buffered during the run) --
    video_frames: list[np.ndarray] = []
    renderer = None
    if video is not None:
        try:
            from .camera import make_camera
            from .render import HeadlessRenderer
        except ImportError as exc:  # pragma: no cover - rendering is optional
            raise RuntimeError("video capture needs pistr_sim.env.render (M3)") from exc
        if model is None:
            model = getattr(plant, "model", None)
        if model is None:
            raise ValueError("video capture needs a MuJoCo model")
        if camera is None:
            camera = make_camera(model)
        renderer = HeadlessRenderer(model, camera=camera)

        def _record(t: float, data: Any) -> None:
            if data is not None:
                video_frames.append(renderer.render(data, camera=camera))

        if on_frame is not None:
            user_on_frame = on_frame

            def on_frame(t: float, data: Any) -> None:  # type: ignore[misc]
                user_on_frame(t, data)
                _record(t, data)

        else:
            on_frame = _record

    # -- camera rig ------------------------------------------------------
    # If ``camera`` is an M3 rig (has ``set_target``), run_flight drives it itself:
    # ``tick(frame_dt)`` advances the turntable/smoothing clock, then
    # ``set_target(pos, yaw=...)`` aims it at drone 0 in MuJoCo coordinates.  Pass
    # ``camera=None`` and drive it from ``on_frame`` if you need a different policy.
    frame_dt = h * max(1, int(frame_stride))

    def _camera_update(dt: float) -> None:
        if camera is None or not hasattr(camera, "set_target"):
            return
        tick = getattr(camera, "tick", None)
        if tick is not None:
            tick(dt)
        pose = getattr(plant, "mj_pose", None)
        if pose is None:
            return
        position, rotation = pose(0)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        camera.set_target(position, yaw=yaw, dt=dt)


    n_recorded = n_ticks
    diverged = False
    try:
        for k in range(n_ticks):
            t = k * h
            if gust_active:
                if t + 1e-12 >= float(gust_t[0]) and t < float(gust_t[1]):
                    plant.set_wind(gust_vector, turbulence=True)
                else:
                    plant.set_wind(np.zeros(3), turbulence=True)
            pwm_rows = np.zeros((n_drones, 4))
            for i in range(n_drones):
                sample = plant.sense() if n_drones == 1 else plant.sense_of(i)
                att = attitude_estimators[i].update(sample, dt=h)
                c_b2i = quat_to_rotmat(_quat_from_euler(*att.angle)).T
                a_up = -float((c_b2i @ np.asarray(sample.accel, dtype=float))[2] + g)
                altitude, climb = altitude_estimators[i].update(sample.baro_alt, a_up, dt=h)

                truth = plant.state if n_drones == 1 else plant.state_of(i)
                ref = references[i].sample(t)
                if planners[i] is not None:
                    xi_meas = truth.xi if position_aid else np.zeros(3)
                    v_meas = truth.v if position_aid else np.zeros(3)
                    ref = planners[i].sample(ref, xi_meas, v_meas, t)
                y_ref = np.array([ref.roll, ref.pitch, ref.yaw, float(ref.altitude)])
                rate_ref = np.array(
                    [ref.roll_rate, ref.pitch_rate, ref.yaw_rate, float(ref.climb_rate)]
                )

                y_meas = np.array([att.angle[0], att.angle[1], att.angle[2], altitude])
                rate_meas = np.array([att.rate[0], att.rate[1], att.rate[2], climb])
                u = np.asarray(
                    controllers[i].update(y_ref, y_meas, rate_ref, rate_meas), dtype=float
                ).reshape(4)
                pwm = clamp_pwm(throttle + U_MIX @ (u * ALTITUDE_SIGN))
                u_actual = recover_u_actual(pwm, throttle) * ALTITUDE_SIGN
                controllers[i].ack(u_actual)

                euler = truth.euler
                rr = records[i]
                rr["t"][k] = t
                rr["xi"][k] = truth.xi
                rr["v"][k] = truth.v
                rr["quat"][k] = truth.q
                rr["euler"][k] = euler
                rr["omega"][k] = truth.omega
                rr["pwm"][k] = pwm
                rr["rotor_omega"][k] = np.asarray(truth.rotor_omega, dtype=float).reshape(4)
                rr["thrust"][k] = _thrust_of(plant, i)
                rr["y_ref"][k] = y_ref
                rr["rate_ref"][k] = rate_ref
                rr["y"][k] = np.array([euler[0], euler[1], euler[2], -truth.xi[2]])
                rr["est_y"][k] = y_meas
                rr["u"][k] = u
                rr["u_actual"][k] = u_actual
                rr["pos_ref"][k] = ref.xi
                wind = getattr(plant, "wind", None)
                if wind is not None and hasattr(wind, "at"):
                    rr["wind"][k] = np.asarray(wind.at(t), dtype=float).reshape(3)
                pwm_rows[i] = pwm

            # Hold the PWM over the whole control period: `sub` plant steps of
            # `plant.dt` (the plant itself sub-steps MuJoCo's timestep).
            step_pwm = pwm_rows[0] if n_drones == 1 else pwm_rows
            for _ in range(sub):
                plant.step(step_pwm)

            if (k % max(1, int(frame_stride)) == 0) and (on_frame is not None or camera is not None):
                _camera_update(frame_dt)
                if on_frame is not None:
                    on_frame(float(t), getattr(plant, "data", None))

            if _diverged(plant, n_drones):
                diverged = True
                n_recorded = k + 1
                break

            if verbose and (k % 25 == 0 or k == n_ticks - 1):
                alt = -float(plant.state.xi[2])
                print(f"  t={t:6.2f}s  alt={alt:7.3f} m  pwm={np.mean(pwm_rows[0]):7.1f} us")
    finally:
        if renderer is not None:
            renderer.close()

    # -- assemble trace --------------------------------------------------
    traces = []
    for i in range(n_drones):
        r = records[i]
        traces.append(
            DroneTrace(
                name=f"drone{i}",
                t=r["t"][:n_recorded].copy(),
                xi=r["xi"][:n_recorded].copy(),
                v=r["v"][:n_recorded].copy(),
                quat=r["quat"][:n_recorded].copy(),
                euler=r["euler"][:n_recorded].copy(),
                omega=r["omega"][:n_recorded].copy(),
                pwm=r["pwm"][:n_recorded].copy(),
                rotor_omega=r["rotor_omega"][:n_recorded].copy(),
                thrust=r["thrust"][:n_recorded].copy(),
                y_ref=r["y_ref"][:n_recorded].copy(),
                rate_ref=r["rate_ref"][:n_recorded].copy(),
                y=r["y"][:n_recorded].copy(),
                est_y=r["est_y"][:n_recorded].copy(),
                u=r["u"][:n_recorded].copy(),
                u_actual=r["u_actual"][:n_recorded].copy(),
                pos_ref=r["pos_ref"][:n_recorded].copy(),
                wind=r["wind"][:n_recorded].copy(),
            )
        )

    trace = FlightTrace(
        t=records[0]["t"][:n_recorded].copy(),
        drones=traces,
        wind=records[0]["wind"][:n_recorded].copy(),
        scenario=scenario,
        controller=controller,
        metadata={
            "backend": backend,
            "params_fidelity": params.fidelity,
            "dt": plant_dt,
            "control_h": h,
            "sub_steps": sub,
            "seed": seed,
            "n_drones": n_drones,
            "t_end": float(t_end),
            "n_ticks": int(n_recorded),
            "diverged": diverged,
            "throttle_pwm": throttle,
            "trim": {
                key: float(value)
                for key, value in trim.items()
                if key != "state" and np.isscalar(value)
            },
            "mujoco_timestep": getattr(getattr(plant, "model", None), "opt", None)
            and float(plant.model.opt.timestep),
            "position_aid": bool(position_aid),
            "scenario": scenario,
            "gust_ned": np.asarray(gust_ned, dtype=float).tolist()
            if scenario == "gust"
            else None,
            "runtime_s": None,
        },
    )

    # -- optional artefacts ---------------------------------------------
    if video is not None and video_frames:
        from .render import write_mp4

        trace.metadata["video"] = write_mp4(
            video_frames, video, fps=1.0 / h
        )
    if out is not None:
        os.makedirs(out, exist_ok=True)
        trace.metadata["summary_json"] = trace.write_json(
            os.path.join(out, "flight_summary.json")
        )
    trace.metadata["runtime_s"] = float(time.perf_counter() - started)
    return trace


def _make_reference(
    scenario: str, altitude0: float, step_amplitude: float, step_time: float
) -> Any:
    if scenario == "hover":
        return make_reference("hover", altitude=float(altitude0))
    if scenario == "step":
        return make_reference(
            "step", amplitude=float(step_amplitude), t_step=float(step_time),
            axis=0, altitude=float(altitude0),
        )
    if scenario == "sine":
        return make_reference("sine", altitude_offset=float(altitude0))
    if scenario == "gust":
        return make_reference("hover", altitude=float(altitude0))
    if scenario == "trajectory":
        return TrajectoryReference3D()
    raise ValueError(f"unknown scenario {scenario!r}")


def _thrust_of(plant: Any, index: int) -> float:
    """Total rotor thrust currently applied to vehicle ``index``, N."""
    thrust = getattr(plant, "thrust", None)
    if callable(thrust):
        return float(thrust(index))
    evaluate = getattr(plant, "evaluate", None)
    if callable(evaluate) and index == 0:
        try:
            return float(evaluate()["thrust"])
        except (KeyError, TypeError):  # pragma: no cover - defensive
            return float("nan")
    return float("nan")


def _diverged(plant: Any, n_drones: int) -> bool:
    try:
        states = plant.states if n_drones > 1 else [plant.state]
    except AttributeError:  # pragma: no cover - defensive
        states = [plant.state]
    for st in states:
        if not np.all(np.isfinite(st.xi)) or not np.all(np.isfinite(st.v)):
            return True
        if not np.all(np.isfinite(st.q)) or not np.all(np.isfinite(st.omega)):
            return True
        if np.max(np.abs(st.euler)) > np.pi or abs(float(st.xi[2])) > 1e3:
            return True
    return False
