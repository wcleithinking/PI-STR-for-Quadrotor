"""MJCF construction: ``pistr_sim.params.Params`` -> MuJoCo XML / model.

Contract: ``docs/3D_ENVIRONMENT.md`` §4.  Everything physical is read from
``params`` -- mass, inertia, arm geometry, rotor radius, spin directions and the
gravity constant are never retyped here.

Body layout (all names stable; M3 renders them, M4 looks them up)
-----------------------------------------------------------------
``world``
  ``floor``                 ground plane (only when ``scene is None``; see below)
  ``drone{i}``              free-jointed vehicle root, body frame origin = **CoM**
    ``drone{i}_body``         fuselage box (collision on)
    ``drone{i}_nose``         +x marker, visual only (helps check yaw in renders)
    ``drone{i}_arm{j}``       arm capsules, visual only
    ``drone{i}_leg{k}``       landing legs, collision on (the vehicle rests on them)
    ``drone{i}_rotor{j}``     welded, **massless** rotor body (so it can be spun
                              visually by rotating ``model.body_quat[rotor]``)
      ``drone{i}_rotor{j}_disc``    swept-area disc, visual only
      ``drone{i}_rotor{j}_blade``   two-blade bar, visual only (makes spin visible)
      ``drone{i}_rotor{j}_hub``     hub, visual only
      ``site drone{i}_rotor{j}_site``  rotor centre, for cameras / metrics

``i`` is 0-based drone index, ``j`` the firmware mixer rotor index (0..3).
The rotor bodies carry **zero mass**: ``Airframe.mass`` and
``Airframe.inertia`` already include the rotors and motors (see ``params.py``),
so adding them again would double-count.  Their geoms are collisionless
(``contype=contype=0``) so welded-body contacts cannot perturb the dynamics.

Frames: MuJoCo is FLU (x forward, y **left**, z up).  The arm positions are
therefore ``S @ Airframe.arm_positions`` with ``S = diag(1, -1, -1)``, i.e.
rotor 1 sits at -y and rotor 3 at +y.  The same ``S`` maps the NED body wrench
to FLU (``S`` is a proper rotation, det = +1), so geometry and physics agree.

Inertia caveat (see ``docs/3D_ENVIRONMENT.md`` §7 and §5/V2)
------------------------------------------------------------
``Airframe.inertia`` is the published MATLAB tensor, whose off-diagonal terms
are set to ``Iyz = Ixz`` -- which ``params.py`` itself flags as *not physically
symmetric*.  This tensor violates the triangle inequality
(``Ixx + Iyy < Izz`` by 2.1e-4) and MuJoCo refuses it at compile time.  The
documented fix is to zero the known-bad coupling term ``Iyz`` only; every
diagonal entry then matches ``params`` exactly (1e-12) and the result is valid
with a 2.9e-6 margin in ``A + B - C``.  :func:`realisable_inertia` does that and
:func:`build_xml` asserts the result really is acceptable before writing it.

Scene integration
-----------------
``scene`` is an optional MJCF fragment produced by ``env/scene.py`` (M2),
inserted inside ``<worldbody>``.  For convenience a single top-level
``<asset>...</asset>`` block in the fragment is hoisted into the model's asset
section, so scene textures/materials work; everything else is inserted verbatim.
When ``scene is None`` a bare checkerboard ground plane and a sun light are
provided instead (enough for the model/actuation gates and for a first render).
When a scene *is* given it owns the ground, so no floor is added here.

``build_model`` never creates a GL context: only ``env/render.py`` may.
"""

from __future__ import annotations

import re

import numpy as np

from ..params import Params

__all__ = [
    "DEFAULT_ALTITUDE",
    "DEFAULT_SPACING",
    "build_xml",
    "build_model",
    "realisable_inertia",
    "drone_body_name",
    "drone_body_names",
    "rotor_body_name",
    "rotor_site_name",
    "model_body_id",
    "body_inertia_tensor",
    "S_NED_TO_FLU",
]

#: NED body frame -> MuJoCo FLU body frame (a 180 deg rotation about x).
S_NED_TO_FLU = np.diag([1.0, -1.0, -1.0])

#: Default initial altitude of each vehicle, m (above the ground plane).
DEFAULT_ALTITUDE = 1.5
#: Default spacing of the ``layout=None`` line arrangement, m.
DEFAULT_SPACING = 3.0

_ROTOR_COLOURS = (
    "0.85 0.30 0.20 1",
    "0.25 0.60 0.90 1",
    "0.90 0.75 0.20 1",
    "0.30 0.80 0.45 1",
)

_LEG_OFFSET = 0.09
_LEG_TOP = -0.040
_LEG_LENGTH = 0.100
_ARM_Z = 0.012
_ROTOR_Z = 0.020

_ASSET_RE = re.compile(r"<asset\b[^>]*>(?P<body>.*?)</asset>", re.DOTALL)


# ----------------------------------------------------------------------------
# naming (single place; M3/M4 import these instead of hard-coding strings)
# ----------------------------------------------------------------------------
def drone_body_name(index: int) -> str:
    """Root (free-jointed) body of drone ``index``."""
    return f"drone{int(index)}"


def drone_body_names(n_drones: int) -> list[str]:
    """Root body names, in drone order: ``['drone0', ...]``."""
    return [drone_body_name(i) for i in range(int(n_drones))]


def rotor_body_name(drone: int, rotor: int) -> str:
    """Welded, massless body of rotor ``rotor`` on drone ``drone``."""
    return f"drone{int(drone)}_rotor{int(rotor)}"


def rotor_site_name(drone: int, rotor: int) -> str:
    """Site at the centre of rotor ``rotor`` on drone ``drone``."""
    return f"drone{int(drone)}_rotor{int(rotor)}_site"


# ----------------------------------------------------------------------------
# inertia
# ----------------------------------------------------------------------------
def realisable_inertia(inertia) -> np.ndarray:
    """Return the MuJoCo-legal tensor derived from ``Airframe.inertia``.

    The published MATLAB tensor sets ``Iyz = Ixz`` (flagged in ``params.py`` as
    a bug).  That makes the principal moments violate ``A + B >= C``, which
    MuJoCo rejects at compile time.  Zeroing the single bad coupling ``Iyz``
    fixes it while keeping **every diagonal term exactly equal to params**:

    * diagonal ``Ixx, Iyy, Izz``: identical to ``params`` (1e-12),
    * ``Ixy = 0`` (already zero in ``params``),
    * ``Ixz``: identical to ``params``,
    * ``Iyz = 0`` (the fix).

    The resulting principal moments are ``(0.00941734, 0.00942911,
    0.01884353)``, i.e. ``A + B - C = +2.9e-6 > 0``: legal, but only just, which
    is why ``test_env_model.py`` asserts ``Iyz == 0`` explicitly.
    """
    K = np.array(inertia, dtype=float)
    if K.shape != (3, 3):
        raise ValueError(f"inertia must be 3x3, got shape {K.shape}")
    K = 0.5 * (K + K.T)
    K[1, 2] = K[2, 1] = 0.0
    principal = np.linalg.eigvalsh(K)
    if principal[0] <= 0.0:
        raise ValueError(f"inertia is not positive definite: {principal}")
    if principal[0] + principal[1] < principal[2]:
        raise ValueError(
            "inertia still violates the MuJoCo triangle inequality after "
            f"zeroing Iyz: principal moments {principal}, "
            f"A + B - C = {principal[0] + principal[1] - principal[2]:.3e}"
        )
    return K


def _fullinertia(K: np.ndarray) -> str:
    """MJCF ``fullinertia`` string, order ``ixx iyy izz ixy ixz iyz``."""
    return " ".join(
        f"{value:.12g}"
        for value in (K[0, 0], K[1, 1], K[2, 2], K[0, 1], K[0, 2], K[1, 2])
    )


# ----------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------
def _layout_positions(n_drones: int, layout) -> list[np.ndarray]:
    """Initial body positions, one per drone.

    ``layout=None`` places the drones on a line along +y spaced by
    :data:`DEFAULT_SPACING`, all at :data:`DEFAULT_ALTITUDE`.  Otherwise
    ``layout`` must be a sequence of ``(x, y, z)`` triples (one per drone), or a
    single triple when ``n_drones == 1``.
    """
    if layout is None:
        return [
            np.array([0.0, DEFAULT_SPACING * i, DEFAULT_ALTITUDE], dtype=float)
            for i in range(n_drones)
        ]
    arr = np.asarray(layout, dtype=float)
    if arr.ndim == 1 and arr.size == 3:
        if n_drones != 1:
            raise ValueError("a single position needs n_drones=1")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"layout must be a sequence of (x, y, z) triples; got shape {arr.shape}"
        )
    if arr.shape[0] < n_drones:
        raise ValueError(
            f"layout has {arr.shape[0]} positions but n_drones={n_drones}"
        )
    if not np.all(np.isfinite(arr[:n_drones])):
        raise ValueError("layout positions must be finite")
    return [arr[i].copy() for i in range(n_drones)]


# ----------------------------------------------------------------------------
# XML
# ----------------------------------------------------------------------------
def _drone_block(params: Params, index: int, position: np.ndarray, K: np.ndarray) -> str:
    a = params.airframe
    mass = float(a.mass)
    arms_flu = (S_NED_TO_FLU @ np.asarray(a.arm_positions, dtype=float).T).T
    r = float(a.r_rotor)
    chord = float(a.c_rotor)
    name = drone_body_name(index)
    x, y, z = (float(v) for v in position)

    lines = [
        f'    <body name="{name}" pos="{x:.6g} {y:.6g} {z:.6g}">',
        f'      <freejoint name="{name}_free"/>',
        f'      <inertial pos="0 0 0" mass="{mass:.12g}" '
        f'fullinertia="{_fullinertia(K)}"/>',
        f'      <geom name="{name}_body" type="box" size="0.120 0.120 0.035" '
        f'rgba="0.16 0.18 0.22 1"/>',
        f'      <geom name="{name}_nose" type="box" pos="0.130 0 0" '
        f'size="0.020 0.045 0.020" contype="0" conaffinity="0" '
        f'rgba="0.90 0.20 0.20 1"/>',
    ]
    for j, (ax, ay, _az) in enumerate(arms_flu):
        lines.append(
            f'      <geom name="{name}_arm{j}" type="capsule" '
            f'fromto="0 0 {_ARM_Z:.4f} {ax:.6f} {ay:.6f} {_ARM_Z:.4f}" '
            f'size="0.010" contype="0" conaffinity="0" rgba="0.10 0.10 0.12 1"/>'
        )
    for k, (sx, sy) in enumerate(
        [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    ):
        lines.append(
            f'      <geom name="{name}_leg{k}" type="cylinder" '
            f'pos="{sx * _LEG_OFFSET:.4f} {sy * _LEG_OFFSET:.4f} '
            f'{_LEG_TOP - 0.5 * _LEG_LENGTH:.4f}" '
            f'size="0.008 {0.5 * _LEG_LENGTH:.4f}" rgba="0.10 0.10 0.12 1"/>'
        )
    for j, (ax, ay, _az) in enumerate(arms_flu):
        colour = _ROTOR_COLOURS[j % len(_ROTOR_COLOURS)]
        rgb = " ".join(colour.split()[:3])
        rotor = rotor_body_name(index, j)
        lines += [
            f'      <body name="{rotor}" pos="{ax:.6f} {ay:.6f} {_ROTOR_Z:.4f}">',
            '        <inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/>',
            f'        <geom name="{rotor}_disc" type="cylinder" '
            f'size="{r:.6f} 0.0030" contype="0" conaffinity="0" '
            f'rgba="{rgb} 0.28"/>',
            f'        <geom name="{rotor}_blade" type="box" '
            f'size="{r:.6f} {0.5 * chord:.6f} 0.0015" contype="0" '
            f'conaffinity="0" rgba="{rgb} 1"/>',
            f'        <geom name="{rotor}_hub" type="cylinder" '
            f'size="0.016 0.0060" contype="0" conaffinity="0" '
            f'rgba="0.08 0.08 0.08 1"/>',
            f'        <site name="{rotor_site_name(index, j)}" pos="0 0 0" '
            f'size="0.010" rgba="{rgb} 0.6"/>',
            "      </body>",
        ]
    lines.append("    </body>")
    return "\n".join(lines)


def build_xml(
    params: Params,
    *,
    scene: str | None = None,
    n_drones: int = 1,
    layout=None,
) -> str:
    """Build the MJCF string for ``params`` (see the module docstring).

    Parameters
    ----------
    params:
        ``pistr_sim.params.Params``; the only source of physical constants.
    scene:
        Optional MJCF fragment inserted inside ``<worldbody>`` (owned by M2's
        ``env/scene.py``).  ``None`` adds a bare checker floor + sun light.
    n_drones:
        Number of independent vehicles (each its own free joint and body).
    layout:
        Initial body positions; see :func:`_layout_positions`.

    Returns
    -------
    str
        A complete ``<mujoco>`` document.  Parsing it needs no GL context.
    """
    if int(n_drones) < 1:
        raise ValueError(f"n_drones must be >= 1, got {n_drones!r}")
    n_drones = int(n_drones)
    dt = float(params.sim.dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"params.sim.dt must be positive, got {dt!r}")

    a = params.airframe
    K = realisable_inertia(a.inertia)
    positions = _layout_positions(n_drones, layout)

    world: list[str] = []
    if scene is None:
        floor = (
            '    <geom name="floor" type="plane" size="60 60 0.05" '
            'material="grid"/>'
        )
    else:
        floor = None
    assets = (
        '    <texture name="grid" type="2d" builtin="checker" '
        'rgb1="0.22 0.30 0.38" rgb2="0.45 0.55 0.65" width="512" height="512"/>\n'
        '    <material name="grid" texture="grid" texrepeat="20 20" '
        'reflectance="0.05"/>'
    )

    scene_world = scene or ""
    scene_assets = ""
    if scene:
        # Be forgiving about M2 returning textures: hoist a single <asset> block.
        found = _ASSET_RE.findall(scene)
        if len(found) > 1:
            raise ValueError("scene fragment may contain at most one <asset> block")
        if found:
            scene_assets = found[0]
            scene_world = _ASSET_RE.sub("", scene)
    if scene_assets:
        assets = assets + "\n" + scene_assets

    lines = [
        '<mujoco model="pistr-3d">',
        '  <compiler angle="radian" inertiafromgeom="false"/>',
        f'  <option timestep="{dt:.12g}" gravity="0 0 {-float(a.g):.12g}" '
        'integrator="implicitfast" cone="elliptic" impratio="10"/>',
        '  <visual>',
        '    <global offwidth="1280" offheight="720"/>',
        '    <quality shadowsize="2048"/>',
        '    <map znear="0.01" zfar="200"/>',
        '  </visual>',
        "  <asset>",
        assets,
        "  </asset>",
        "  <worldbody>",
        '    <light name="sun" pos="6 -6 12" dir="-0.4 0.4 -1" '
        'directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>',
    ]
    if floor is not None:
        lines.append(floor)
    for i, pos in enumerate(positions):
        lines.append(_drone_block(params, i, pos, K))
    if scene_world.strip():
        lines.append(scene_world.strip("\n"))
    lines += ["  </worldbody>", "</mujoco>", ""]
    return "\n".join(lines)


def build_model(
    params: Params,
    *,
    scene: str | None = None,
    n_drones: int = 1,
    layout=None,
):
    """Compile :func:`build_xml` into a ``mujoco.MjModel`` (headless).

    ``mujoco`` is imported here, not at module import time, so
    ``pistr_sim.env.mjcf`` stays importable without the optional dependency.
    """
    import mujoco

    xml = build_xml(params, scene=scene, n_drones=n_drones, layout=layout)
    return mujoco.MjModel.from_xml_string(xml)


# ----------------------------------------------------------------------------
# model introspection helpers (used by the gates and by the bridge/renderers)
# ----------------------------------------------------------------------------
def model_body_id(model, name: str) -> int:
    """Body id for ``name``; raises ``KeyError`` (with the known names) if absent."""
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        known = [model.body(i).name for i in range(model.nbody) if model.body(i).name]
        raise KeyError(f"no body named {name!r} in model; known bodies: {known}")
    return int(body_id)


def body_inertia_tensor(model, body_id: int) -> np.ndarray:
    """Inertia of ``body_id`` **about its body-frame origin**, in the body frame.

    MuJoCo stores ``body_inertia`` as the three principal moments and
    ``body_iquat`` as the principal axes; this reconstructs the full 3x3 tensor
    and adds the parallel-axis term for ``body_ipos`` (zero for our vehicles,
    whose CoM is the body-frame origin).
    """
    import mujoco

    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(model.body_iquat[body_id], dtype=float))
    rot = rot.reshape(3, 3)
    inertia = rot @ np.diag(np.asarray(model.body_inertia[body_id], float)) @ rot.T
    mass = float(model.body_mass[body_id])
    ipos = np.asarray(model.body_ipos[body_id], dtype=float)
    inertia += mass * (float(ipos @ ipos) * np.eye(3) - np.outer(ipos, ipos))
    return 0.5 * (inertia + inertia.T)
