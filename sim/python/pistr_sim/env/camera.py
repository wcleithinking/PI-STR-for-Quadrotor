"""Camera rigs for the MuJoCo 3-D world (M3).

``make_camera(model, name, **kw)`` returns a :class:`mujoco.MjvCamera` subclass, so
it can be handed straight to ``mujoco.Renderer.update_scene(data, camera)`` or to
:class:`pistr_sim.env.render.HeadlessRenderer`.  Three rig families are provided:

``chase``
    Follows the vehicle.  Only its **heading** is followed (azimuth); roll and pitch
    are deliberately *not* copied, so the picture never rolls over when the vehicle
    banks.  Position and heading are low-pass filtered (``tau`` seconds), and the
    vehicle can be tracked frame by frame with ``cam.set_target(pos, quat=...)``.
``orbit``
    Circles the look-at point at a fixed radius/height.  ``tick(dt)`` turns the
    azimuth by ``azimuth_rate`` degrees per second, which is all a turntable shot
    needs.
``fixed``
    A static viewpoint (optionally copied from a model camera).  ``top`` is the
    same rig with a near-vertical elevation, i.e. a map view.

Geometry convention (identical to MuJoCo's free camera, verified on 4.0.0)::

    pos = lookat - distance * (cos(elev)*cos(azim), cos(elev)*sin(azim), sin(elev))

so ``elevation`` is measured from the horizontal plane and is **negative when the
camera is above the look-at point**.  The convenience parameters ``distance`` and
``height`` are the *horizontal* radius and the *vertical* offset above ``lookat``;
``height`` is folded into ``elevation``::

    elevation = -degrees(atan2(height, distance))
    cam.distance = hypot(distance, height)      # MuJoCo's slant distance

Passing ``elevation`` explicitly bypasses ``height`` and treats ``distance`` as the
slant distance, exactly like MuJoCo.

Field of view: ``mujoco.MjvCamera`` has no ``fovy`` field in MuJoCo 4.0 — the free
camera always uses ``model.vis.global_.fovy``.  ``fovy=`` therefore sets that model
field (a documented, model-wide side effect) and is also cached on the rig, so
:class:`HeadlessRenderer` can re-apply it when a rig is shared between renderers.

Nothing here touches a GL context: cameras are pure data and are constructible
headless, as required by ``docs/3D_ENVIRONMENT.md`` §4.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

import mujoco

__all__ = ["Camera", "make_camera"]

_MODES = ("chase", "orbit", "fixed")

#: name -> (mode, top-down?) aliases; ``make_camera`` accepts all of them.
_ALIASES = {
    "chase": ("chase", False),
    "follow": ("chase", False),
    "track": ("chase", False),
    "orbit": ("orbit", False),
    "turntable": ("orbit", False),
    "rotate": ("orbit", False),
    "fixed": ("fixed", False),
    "static": ("fixed", False),
    "lookat": ("fixed", False),
    "top": ("fixed", True),
    "topdown": ("fixed", True),
    "top-down": ("fixed", True),
    "map": ("fixed", True),
}

#: Per-mode defaults: horizontal radius, height above look-at, azimuth (deg), rate.
_DEFAULTS = {
    # A 4.5 m / 1.8 m chase shot at fovy=45 keeps the vehicle readable (~40 px tall
    # at 640x480), the ground texture, the horizon and the whole ground shadow in
    # frame; see results/env/render_check.png.
    "chase": dict(distance=4.5, height=1.8, azimuth=None, azimuth_rate=0.0),
    "orbit": dict(distance=6.0, height=2.5, azimuth=90.0, azimuth_rate=0.0),
    "fixed": dict(distance=8.0, height=5.0, azimuth=90.0, azimuth_rate=0.0),
    # Top-down: `height` is the camera altitude above the look-at point (10 m gives
    # a ~8 m wide patch of ground at fovy=45); `distance` is ignored by this rig.
    "top": dict(distance=8.0, height=10.0, azimuth=90.0, azimuth_rate=0.0),
}


def _wrap_deg(angle: float) -> float:
    """Wrap an angle in degrees to ``[0, 360)``."""
    return float(angle) % 360.0


def _smooth_deg(previous: float, target: float, alpha: float) -> float:
    """Exponentially approach ``target`` from ``previous`` along the short arc."""
    delta = (float(target) - float(previous) + 180.0) % 360.0 - 180.0
    return _wrap_deg(float(previous) + alpha * delta)


def _yaw_from_quat(quat: Iterable[float]) -> float:
    """Heading (rad, math convention, about +z) of a MuJoCo ``(w, x, y, z)`` quat."""
    q = np.asarray(quat, dtype=float).reshape(4)
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _body_index(model, body) -> int:
    """Resolve ``body`` (name, id, None) to a body id, or -1 for "use lookat only".

    ``None`` means "the obvious vehicle": a body literally called ``drone`` /
    ``quadrotor`` / ``base``, else the first child of the world body that owns a
    free joint, else nothing.
    """
    if body is None:
        for candidate in ("drone", "quadrotor", "quad", "base", "body"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if bid >= 0:
                return int(bid)
        for bid in range(1, model.nbody):
            if model.body_jntnum[bid] and model.jnt_type[model.body_jntadr[bid]] == int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                return int(bid)
        return -1
    if isinstance(body, str):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            raise ValueError(f"no body named {body!r} in model")
        return int(bid)
    bid = int(body)
    if not 0 <= bid < model.nbody:
        raise ValueError(f"body id {bid} out of range [0, {model.nbody})")
    return bid


class Camera(mujoco.MjvCamera):
    """A ``MjvCamera`` with a chase/orbit/fixed personality.

    Attributes
    ----------
    mode : {"chase", "orbit", "fixed"}
    name : str
        The name it was created with (``"top"`` is a fixed rig, remembered as such).
    tau : float
        Low-pass time constant in seconds for look-at and heading.  ``0`` snaps.
    azimuth_rate : float
        Orbit turntable speed, degrees per second, applied by :meth:`tick`.
    azimuth_offset : float
        Chase only: azimuth = vehicle heading + this offset.  MuJoCo places the
        camera at ``lookat - radius*(cos az, sin az)``, so offset ``0`` puts it
        behind the vehicle looking the way the vehicle flies.
    fovy : float | None
        Vertical field of view in degrees, or ``None`` to inherit the model value.
    """

    def __init__(
        self,
        mode: str = "chase",
        *,
        name: str = "chase",
        distance: float = 4.5,
        height: float = 1.8,
        lookat=(0.0, 0.0, 1.0),
        azimuth: float = 90.0,
        elevation: float | None = None,
        fovy: float | None = None,
        tau: float = 0.15,
        azimuth_rate: float = 0.0,
        azimuth_offset: float = 0.0,
        lookat_offset=(0.0, 0.0, 0.0),
        top_down: bool = False,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        super().__init__()
        self.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
        self.name = str(name)
        self.mode = mode
        self.tau = max(0.0, float(tau))
        self.azimuth_rate = float(azimuth_rate)
        self.azimuth_offset = float(azimuth_offset)
        self._fovy_deg = None if fovy is None else float(fovy)
        self._horizontal = abs(float(distance))
        self._height = float(height)
        self._explicit_elevation = None if elevation is None else float(elevation)
        self._lookat_offset = np.asarray(lookat_offset, dtype=float).reshape(3)
        self._lookat_smooth = np.asarray(lookat, dtype=float).reshape(3).copy()
        self._has_target = False
        self._dt = 1.0 / 30.0
        self.lookat[:] = self._lookat_smooth
        self._top_down = bool(top_down)
        self._apply_geometry()
        self.azimuth = _wrap_deg(float(azimuth))
        self.elevation = self._elevation_deg()

    # -- fovy (not an MjvCamera field in MuJoCo 4.0) -------------------------
    @property
    def fovy(self) -> float | None:
        return self._fovy_deg

    @fovy.setter
    def fovy(self, value) -> None:
        self._fovy_deg = None if value is None else float(value)

    # -- geometry ------------------------------------------------------------
    def _apply_geometry(self) -> None:
        """Fold (horizontal radius, height) into MuJoCo's (distance, elevation)."""
        if self._explicit_elevation is not None:
            self.distance = self._horizontal
            self._elevation = self._explicit_elevation
        elif self._top_down:
            # A map view: `height` is the altitude above the look-at point and the
            # elevation stops at -89 deg so the camera's up vector stays defined.
            self.distance = max(abs(self._height), 0.1)
            self._elevation = -89.0
        else:
            self.distance = math.hypot(self._horizontal, self._height)
            self._elevation = -math.degrees(
                math.atan2(self._height, max(self._horizontal, 1e-9))
            )

    def _elevation_deg(self) -> float:
        if self._top_down and self._explicit_elevation is None:
            return -89.0
        return float(self._elevation)

    # -- motion --------------------------------------------------------------
    def tick(self, dt: float) -> "Camera":
        """Advance the rig by ``dt`` seconds (orbit turntable, smoothing clock)."""
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be a positive finite number, got {dt!r}")
        self._dt = dt
        if self.azimuth_rate:
            self.azimuth = _wrap_deg(self.azimuth + self.azimuth_rate * dt)
        return self

    def set_target(
        self,
        pos,
        *,
        yaw: float | None = None,
        quat=None,
        dt: float | None = None,
    ) -> "Camera":
        """Aim the rig at ``pos`` (world metres), e.g. ``data.qpos[:3]`` each frame.

        Parameters
        ----------
        pos :
            Look-at target, ``(3,)`` or anything convertible to it.
        yaw :
            Vehicle heading in **radians** (about +z).  Chase rigs put the camera at
            ``yaw + azimuth_offset``.  Only heading is used: roll and pitch are
            ignored on purpose.
        quat :
            MuJoCo ``(w, x, y, z)`` attitude; convenience alternative to ``yaw``
            (``data.qpos[3:7]``).  ``yaw`` wins if both are given.
        dt :
            Time since the previous call, seconds.  Defaults to the last
            :meth:`tick` value, else ``1/30`` s.  Ignored when ``tau == 0``.
        """
        p = np.asarray(pos, dtype=float).reshape(3)
        if not np.all(np.isfinite(p)):
            raise ValueError(f"pos must be finite, got {pos!r}")
        if yaw is None and quat is not None:
            yaw = _yaw_from_quat(quat)
        goal = p + self._lookat_offset

        if self.tau <= 0.0 or not self._has_target:
            alpha = 1.0
        else:
            step = self._dt if dt is None else float(dt)
            if not math.isfinite(step) or step < 0.0:
                raise ValueError(f"dt must be a non-negative finite number, got {dt!r}")
            alpha = 1.0 - math.exp(-step / self.tau)

        self._lookat_smooth = (1.0 - alpha) * self._lookat_smooth + alpha * goal
        self.lookat[:] = self._lookat_smooth
        self._has_target = True

        if yaw is not None and self.mode == "chase":
            # Only the chase rig follows heading.  An orbit/fixed rig must not swing
            # around when the vehicle yaws -- that is the whole point of fixing it.
            desired = math.degrees(float(yaw)) + self.azimuth_offset
            self.azimuth = _smooth_deg(self.azimuth, desired, alpha)
        return self

    #: Alias kept because the task brief called the per-frame hook ``set_target``;
    #: ``target`` reads better in some call sites.
    target = set_target

    def reset(self) -> "Camera":
        """Forget the smoothing state; the next :meth:`set_target` snaps."""
        self._has_target = False
        self._lookat_smooth = np.asarray(self.lookat, dtype=float).copy()
        return self

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Camera({self.name!r}, mode={self.mode!r}, lookat={np.round(self.lookat, 3)}, "
            f"distance={self.distance:.3f}, azimuth={self.azimuth:.1f}, "
            f"elevation={self.elevation:.1f}, fovy={self.fovy})"
        )


def make_camera(model, name: str = "chase", **kw) -> Camera:
    """Build a camera rig for ``model``.

    Parameters
    ----------
    model : mujoco.MjModel
        Only used to resolve a default look-at body (``drone``/first free body) and
        to apply ``fovy`` to ``model.vis.global_.fovy``.
    name : str
        One of ``chase`` / ``orbit`` / ``fixed`` / ``top`` (plus aliases such as
        ``follow``, ``turntable``, ``topdown``); case-insensitive.
    distance, height :
        Horizontal radius (m) and vertical offset above the look-at point (m).
        The ``top`` rig is the exception: its ``height`` is the camera altitude and
        ``distance`` is ignored (the camera looks straight down).
    lookat :
        Look-at point, ``(3,)`` array-like; defaults to the resolved body position
        for chase, else ``(0, 0, 1)``.
    fovy :
        Vertical field of view in degrees; applied to the model.
    azimuth, elevation :
        Degrees.  ``azimuth`` defaults per mode; ``elevation`` overrides ``height``.
    tau :
        Smoothing time constant in seconds for the look-at pan (and, on chase, the
        heading); ``0`` snaps.
    azimuth_rate :
        Turntable speed, degrees per second, applied by ``Camera.tick``.
    azimuth_offset :
        Chase only: camera azimuth = vehicle heading + this offset.
    lookat_offset :
        Added to every ``set_target`` position (e.g. aim slightly above the vehicle).
    smooth :
        Convenience: ``smooth=False`` is equivalent to ``tau=0``.
    body :
        Body name or id used as the default look-at point; ``None`` auto-detects.
    top_down :
        Force the near-vertical elevation of the ``top`` rig even for ``fixed``.
    """
    key = str(name).strip().lower()
    try:
        mode, top_down = _ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unknown camera rig {name!r}; choose from {sorted(_ALIASES)}"
        ) from None

    defaults = _DEFAULTS["top" if top_down else mode]
    smooth = kw.pop("smooth", None)
    if smooth is not None:
        if "tau" in kw:
            raise TypeError("pass either smooth= or tau=, not both")
        kw["tau"] = 0.0 if not smooth else defaults.get("tau", 0.15)

    allowed = {
        "distance",
        "height",
        "lookat",
        "fovy",
        "azimuth",
        "elevation",
        "tau",
        "azimuth_rate",
        "azimuth_offset",
        "lookat_offset",
        "body",
        "top_down",
    }
    unknown = set(kw) - allowed
    if unknown:
        raise TypeError(
            f"make_camera() got unexpected keyword argument(s): {sorted(unknown)}"
        )

    body_hint = kw.pop("body", None)
    body_id = _body_index(model, body_hint)
    if "lookat" in kw:
        lookat = np.asarray(kw.pop("lookat"), dtype=float).reshape(3)
    elif body_id >= 0:
        lookat = np.asarray(model.body_pos[body_id], dtype=float).copy()
    else:
        lookat = np.array([0.0, 0.0, 1.0])

    distance = float(kw.pop("distance", defaults["distance"]))
    height = float(kw.pop("height", defaults["height"]))
    azimuth = kw.pop("azimuth", defaults["azimuth"])
    if azimuth is None:
        # Chase with no heading yet: behave as if the vehicle headed along +x, so
        # the camera starts on -x looking down the flight direction.  The first
        # set_target() (with yaw/quat) takes over from there.
        azimuth = 0.0
    elevation = kw.pop("elevation", None)
    fovy = kw.pop("fovy", None)
    tau = float(kw.pop("tau", 0.15))
    azimuth_rate = float(kw.pop("azimuth_rate", defaults["azimuth_rate"]))
    azimuth_offset = float(kw.pop("azimuth_offset", 0.0))
    lookat_offset = kw.pop("lookat_offset", (0.0, 0.0, 0.0))

    cam = Camera(
        mode,
        name=key,
        distance=distance,
        height=height,
        lookat=lookat,
        azimuth=float(azimuth),
        elevation=elevation,
        fovy=fovy,
        tau=tau,
        azimuth_rate=azimuth_rate,
        azimuth_offset=azimuth_offset,
        lookat_offset=lookat_offset,
        top_down=bool(kw.pop("top_down", top_down)),
    )
    if fovy is not None:
        model.vis.global_.fovy = float(fovy)
    return cam
