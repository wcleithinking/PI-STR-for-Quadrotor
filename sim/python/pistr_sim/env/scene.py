"""Static scene geometry for the MuJoCo 3-D environment (work stream M2).

``scene_xml(spec)`` turns a :class:`SceneSpec` into an XML **fragment for
``<worldbody>``** — ground, buildings, trees, crates/poles and lights — plus (at
most) one ``<asset>`` block holding two built-in textures, for
``pistr_sim.env.mjcf.build_xml`` to splice into the vehicle model.
The contract lives in ``docs/3D_ENVIRONMENT.md`` §4; this module is the M2 half of
it and is self-contained: it never reaches outside itself for assets, so a scene
can be test-validated without a renderer and without a GL context.

Design rules that are deliberately enforced by code (and by ``test_env_scene.py``)

* **Everything solid is collidable.**  Buildings, trees, crates and poles are
  ``worldbody`` geoms (hence static, infinite mass) with explicit
  ``contype="1" conaffinity="1"``.  No object is a decorative-only mesh.
* **Only decoration is transparent to physics.**  The window bands, the spawn
  pad and the wind markers all carry ``contype="0" conaffinity="0"`` and
  ``group="2"`` so they cannot push, occlude a contact or become part of the
  collision map.  ``wind_markers`` is therefore purely visual — a bit-for-bit
  identical simulation to the same scene without it.  The ground texture is a
  material lookup, so it cannot change contact either.
* **The spawn volume is empty.**  Nothing collidable is placed inside the
  central ``2 * clearance`` box (default ``SPAWN_CLEAR_HALF`` = 5 m, i.e.
  ``10 m x 10 m``) plus a 1.5 m margin, so a vehicle spawned at the origin with
  a 1.5 m drop lands on bare ground and a 3.4 m chase camera does not sit
  against a wall.  ``SceneSpec(clearance=10.0)`` opens it to 20 m x 20 m.
* **Determinism.**  Layout comes from one ``random.Random(spec.seed)`` — never
  the global ``random`` module — and every float is printed with four decimals,
  so ``scene_xml`` is byte-identical for a given ``seed``/spec on any run.
* **Units-safe.**  Orientation is expressed only as ``quat`` (unit
  quaternions), never ``euler``/``axisangle``, and the sky needs no rotation at
  all: the fragment inherits the *caller's* ``<compiler angle=...>``, which M1
  may set to degrees or radians.

Sky: **no geometry, ever.**  MuJoCo has no worldbody-level background colour,
so the sky is a built-in gradient ``<texture type="skybox">`` emitted as the one
``<asset>`` block a fragment may carry (``env/mjcf.build_xml`` hoists it) and
drawn by the renderer behind everything.  Building the sky out of geometry is
tempting and wrong: a shell large enough to enclose the arena inflates
``model.stat.extent``, and MuJoCo sets the camera near plane to
``map.znear * stat.extent`` — a 189 m sky box took ``stat.extent`` from 13 m to
535 m, i.e. ``frustum_near`` from 0.13 m to 5.35 m, which clipped a drone 2 m
away clean out of frame.  Keep the scene compact; see ``_emit_assets``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

__all__ = ["SceneSpec", "scene_xml", "scene_layout", "SPAWN_CLEAR_HALF"]

#: Half-extent of the reserved, empty spawn box at the world origin (metres).
SPAWN_CLEAR_HALF = 5.0
#: Extra ring kept free around the spawn box before the first collidable object.
CLEAR_MARGIN = 1.5
#: Storey height of a building, metres.
_STOREY = 3.0
#: Upper bound of collision-relevant random draws, keeps planning bounded.
_PLACE_TRIES = 256
#: Side of one ground checker **square**, in metres.  Measured mapping for a
#: plane of half-size ``extent``: one texture repeat covers 2 x 2 squares, so
#: ``square = extent / texrepeat``.  ~1.75 m reads as mown grass rather than as
#: a chessboard even when the plane fills 60 % of a 1280x720 frame, and it keeps
#: the distant ground from aliasing.
_GROUND_SQUARE_M = 1.75
#: Radius of the decorative landing pad, metres.
_PAD_RADIUS = 1.3
#: Radius of the object ring as a fraction of ``extent``.  Deliberately well
#: inside ``extent``: MuJoCo derives ``model.stat.extent`` (and with it the
#: camera near plane, see ``_emit_assets``) from the farthest object, and a
#: ring at 0.9 * extent tripled it for no visible gain.
_RING_FLAT = 0.50
_RING_HILLS = 0.40

_COLLIDE = 'contype="1" conaffinity="1"'
_VISUAL = 'contype="0" conaffinity="0" group="2"'

_GROUND_RGBA = (".33 .40 .30 1", ".36 .42 .33 1", ".30 .36 .28 1")
_HILL_RGBA = ".28 .36 .26 1"
_BUILDING_RGBA = (
    ".80 .76 .70 1",
    ".72 .68 .62 1",
    ".82 .72 .60 1",
    ".64 .68 .72 1",
    ".78 .70 .66 1",
)
_ROOF_RGBA = ".30 .32 .36 1"
_WINDOW_LIT = ".98 .86 .52 1"
_WINDOW_DARK = ".14 .20 .28 1"
_TREE_CROWN = (
    ".16 .42 .20 1",
    ".20 .50 .24 1",
    ".26 .46 .18 1",
    ".14 .35 .22 1",
    ".30 .50 .22 1",
)
_TREE_TRUNK = ".36 .24 .14 1"
_CRATE_RGBA = ".60 .40 .20 1"
_POLE_RGBA = ".62 .64 .68 1"
_BARRIER_RGBA = ".85 .48 .12 1"
_SKY_RGBA = ".60 .76 .95 1"
_MARKER_RGBA = ".78 .88 1 1"


@dataclass(frozen=True)
class SceneSpec:
    """What to put in the world.  Frozen: a scene is a value, not a mutable rig.

    Defaults reproduce the contract's example: a 120 m x 120 m flat arena with a
    handful of buildings, trees and crates, lit and sky-backed, from seed 0.

    ``clearance`` is the half-size of the empty box around the origin (the
    contract fixes it at ``SPAWN_CLEAR_HALF``; widening it is a demo knob, e.g.
    ``clearance=10.0`` for a 3.4 m chase camera).  ``ground_texture`` and
    ``pad`` are cosmetic and never affect collision or contact.
    """

    ground: bool = True
    pad: bool = True             # small landing marker at the origin, visual only
    terrain: str = "flat"        # "flat" | "hills"
    buildings: int = 6
    trees: int = 10
    obstacles: int = 8           # crates/poles inside the flight volume
    sky: bool = True
    ground_texture: bool = True  # low-contrast checker, visual only
    wind_markers: bool = False   # visual-only arrows, no collision, no force
    seed: int = 0
    extent: float = 60.0         # world half-size, m
    clearance: float = SPAWN_CLEAR_HALF  # empty half-box around the origin, m

    def __post_init__(self) -> None:
        if self.terrain not in ("flat", "hills"):
            raise ValueError(f"terrain must be 'flat' or 'hills', got {self.terrain!r}")
        for name in ("buildings", "trees", "obstacles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if not math.isfinite(float(self.clearance)) or self.clearance <= 0.0:
            raise ValueError(
                f"clearance must be a positive float, got {self.clearance!r}"
            )
        smallest = 2.0 * float(self.clearance) + 8.0
        if not math.isfinite(float(self.extent)) or self.extent < smallest:
            raise ValueError(
                f"extent must be a finite float >= {smallest:.0f} m "
                "(clearance box plus a ring of objects), "
                f"got {self.extent!r}"
            )


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _f(value: float) -> str:
    """Fixed four decimals; ``-0.0000`` is normalised to ``0.0000``."""
    value = float(value)
    if abs(value) < 5e-5:
        value = 0.0
    return f"{value:.4f}"


def _pos(*xyz: float) -> str:
    return " ".join(_f(v) for v in xyz)


def _quat_z(yaw: float) -> str:
    """Yaw-only unit quaternion, ``w x y z`` order."""
    return _pos(math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


_QUAT_X180 = "0 1 0 0"                       # normal +z -> -z (sky zenith)
_QUAT_Y_M90 = f"{math.sqrt(0.5):.7f} 0 {-math.sqrt(0.5):.7f} 0"   # +z -> -x
_QUAT_Y_P90 = f"{math.sqrt(0.5):.7f} 0 {math.sqrt(0.5):.7f} 0"    # +z -> +x
_QUAT_X_P90 = f"{math.sqrt(0.5):.7f} {math.sqrt(0.5):.7f} 0 0"    # +z -> -y
_QUAT_X_M90 = f"{math.sqrt(0.5):.7f} {-math.sqrt(0.5):.7f} 0 0"   # +z -> +y


def _surface(assets: dict[str, Any], role: str, rgba: str) -> str:
    """``material="..."`` when the caller supplies one, else an inline ``rgba``.

    ``assets`` is the only hook into M1's ``<asset>`` section: map a role name
    (``"ground"``, ``"sky"``, ``"building"``, ``"tree_trunk"``, ...) to an MJCF
    material name and the fragment references it instead of hard-coding colour.
    Unknown keys are ignored, so callers can pass a shared material table.
    """
    name = assets.get(role)
    if name:
        return f'material="{name}"'
    return f'rgba="{rgba}"'


# --------------------------------------------------------------------------- #
# layout planning (pure python, no mujoco)
# --------------------------------------------------------------------------- #
def _clearance_ok(x: float, y: float, radius: float, clear_half: float) -> bool:
    """True when a disc of ``radius`` at (x, y) misses the spawn box + margin."""
    dx = max(0.0, abs(x) - clear_half)
    dy = max(0.0, abs(y) - clear_half)
    return math.hypot(dx, dy) >= radius + CLEAR_MARGIN


def _fit(
    x: float, y: float, radius: float, limit: float | None, clear: float
) -> tuple[float, float]:
    """Pull a point inwards until ``|p| + radius <= limit`` (keeps the world bound)."""
    if limit is None:
        return x, y
    reach = math.hypot(x, y)
    room = max(limit - radius, clear * math.sqrt(2.0) + CLEAR_MARGIN)
    if reach > room and reach > 1e-12:
        scale = room / reach
        return x * scale, y * scale
    return x, y


def _scatter(
    rng: random.Random,
    radius: float,
    rmax: float,
    occupied: list[tuple[float, float, float]],
    *,
    gap: float = 1.5,
    limit: float | None = None,
    clear: float = SPAWN_CLEAR_HALF,
    tries: int = _PLACE_TRIES,
) -> tuple[float, float]:
    """Area-uniform point within ``rmax`` of the origin, outside the spawn box.

    Rejection-samples with a *fixed* number of draws so the result depends only
    on ``rng``'s state; the best candidate seen is kept if the gap constraint
    cannot be met, and a deterministic ring fallback guarantees that a position
    is always returned (never a silent shortage of objects).  ``limit`` is the
    hard maximum distance of the object's *outer edge* from the origin, so the
    scene never spills past ``spec.extent``.
    """
    rmax = max(rmax, clear + radius + CLEAR_MARGIN + 0.5)
    if limit is not None:
        rmax = min(rmax, max(limit - radius, 0.5 * clear))
    best: tuple[float, float] | None = None
    best_score = -math.inf
    for _ in range(tries):
        # sqrt() of a uniform draw keeps the density uniform per unit area.
        r = rmax * math.sqrt(rng.random())
        theta = rng.uniform(0.0, 2.0 * math.pi)
        x, y = r * math.cos(theta), r * math.sin(theta)
        x, y = _fit(x, y, radius, limit, clear)
        if not _clearance_ok(x, y, radius, clear):
            continue
        slack = min(
            (
                math.hypot(x - ox, y - oy) - (radius + orad)
                for ox, oy, orad in occupied
            ),
            default=math.inf,
        )
        if slack >= gap:
            return x, y
        if slack > best_score:
            best_score, best = slack, (x, y)
    if best is not None:
        return best
    # Ring fallback: far enough out that the spawn box cannot be touched even in
    # the corner direction (clear_half * sqrt(2) is the corner distance).
    ring = clear * math.sqrt(2.0) + radius + CLEAR_MARGIN
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return _fit(ring * math.cos(theta), ring * math.sin(theta), radius, limit, clear)


def _plan_buildings(
    rng: random.Random, count: int, band: float, limit: float, clear: float
) -> tuple[list[dict[str, Any]], list[tuple[float, float, float]]]:
    """Footprints scale with the world, so a small ``extent`` still fits."""
    buildings: list[dict[str, Any]] = []
    occupied: list[tuple[float, float, float]] = []
    max_side = max(min(9.0, 0.30 * limit), 2.5)  # footprint scales with the world
    for index in range(count):
        width = rng.uniform(min(4.0, max_side), max_side)
        depth = rng.uniform(min(4.0, max_side), max_side)
        floors = rng.randint(1, 4)
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        radius = 0.5 * math.hypot(width, depth)
        x, y = _scatter(
            rng, radius, band - radius, occupied, gap=3.0, limit=limit, clear=clear
        )
        occupied.append((x, y, radius))
        buildings.append(
            {
                "name": f"bldg_{index:02d}",
                "x": x,
                "y": y,
                "yaw": yaw,
                "w": width,
                "d": depth,
                "floors": floors,
                "rgba": _BUILDING_RGBA[rng.randrange(len(_BUILDING_RGBA))],
                "lit": rng.random() < 0.6,
                "radius": radius,
            }
        )
    return buildings, occupied


def _plan_trees(
    rng: random.Random,
    count: int,
    band: float,
    limit: float,
    clear: float,
    occupied: list[tuple[float, float, float]],
) -> tuple[list[dict[str, Any]], list[tuple[float, float, float]]]:
    trees: list[dict[str, Any]] = []
    for index in range(count):
        trunk_h = rng.uniform(2.2, 3.4)
        trunk_r = rng.uniform(0.14, 0.26)
        crown_r = rng.uniform(0.6, max(min(1.8, 0.05 * limit), 0.9))
        blobs = rng.randint(0, 2)
        crown_rgba = _TREE_CROWN[rng.randrange(len(_TREE_CROWN))]
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        crown_z = trunk_h + 0.55 * crown_r
        offsets = []
        for _ in range(blobs):
            angle = rng.uniform(0.0, 2.0 * math.pi)
            distance = crown_r * rng.uniform(0.45, 0.75)
            offsets.append(
                {
                    "dx": distance * math.cos(angle),
                    "dy": distance * math.sin(angle),
                    "dz": crown_r * rng.uniform(0.15, 0.55),
                    "r": crown_r * rng.uniform(0.55, 0.80),
                }
            )
        radius = crown_r + max((o["r"] * 0.8 for o in offsets), default=0.0) + 0.4
        x, y = _scatter(
            rng, radius, band - radius, occupied, gap=1.2, limit=limit, clear=clear
        )
        occupied.append((x, y, radius))
        trees.append(
            {
                "name": f"tree_{index:02d}",
                "x": x,
                "y": y,
                "yaw": yaw,
                "trunk_h": trunk_h,
                "trunk_r": trunk_r,
                "crown_r": crown_r,
                "crown_z": crown_z,
                "crown_rgba": crown_rgba,
                "offsets": offsets,
                "radius": radius,
            }
        )
    return trees, occupied


def _plan_obstacles(
    rng: random.Random,
    count: int,
    band: float,
    limit: float,
    clear: float,
    occupied: list[tuple[float, float, float]],
) -> tuple[list[dict[str, Any]], list[tuple[float, float, float]]]:
    obstacles: list[dict[str, Any]] = []
    kinds = ("crate", "crate", "pole", "barrier")
    for index in range(count):
        kind = kinds[rng.randrange(len(kinds))]
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        if kind == "crate":
            sx = rng.uniform(0.6, 1.5)
            sy = rng.uniform(0.6, 1.5)
            sz = rng.uniform(0.6, 1.5)
            rgba = _CRATE_RGBA
        elif kind == "barrier":
            sx = rng.uniform(2.0, 3.6)
            sy = rng.uniform(0.22, 0.40)
            sz = rng.uniform(1.0, 1.25)
            rgba = _BARRIER_RGBA
        else:  # pole
            sx = sy = rng.uniform(0.07, 0.14)
            sz = rng.uniform(1.8, 3.6)
            rgba = _POLE_RGBA
        radius = 0.5 * math.hypot(sx, sy) + 0.2
        x, y = _scatter(
            rng, radius, band - radius, occupied, gap=1.0, limit=limit, clear=clear
        )
        occupied.append((x, y, radius))
        obstacles.append(
            {
                "name": f"obs_{index:02d}",
                "kind": kind,
                "x": x,
                "y": y,
                "yaw": yaw,
                "sx": sx,
                "sy": sy,
                "sz": sz,
                "rgba": rgba,
                "radius": radius,
            }
        )
    return obstacles, occupied


def _plan_hills(rng: random.Random, extent: float, rmax_objects: float) -> list[dict[str, Any]]:
    """Domes in the outer ring, outside both the spawn box and the object band."""
    count = max(6, min(24, int(round(extent / 6.0))))
    hills = []
    for index in range(count):
        rx = rng.uniform(0.08, 0.14) * extent
        ry = rng.uniform(0.08, 0.12) * extent
        rz = rng.uniform(0.05, 0.12) * extent
        # Keep the whole hill outside the band where buildings/trees live.
        r_min = max(rmax_objects + rx + CLEAR_MARGIN, 0.70 * extent)
        r_max = max(r_min, 0.82 * extent - max(rx, ry))
        radius = r_min + (r_max - r_min) * rng.random()
        theta = 2.0 * math.pi * (index + rng.uniform(0.1, 0.9)) / count
        hills.append(
            {
                "name": f"hill_{index:02d}",
                "x": radius * math.cos(theta),
                "y": radius * math.sin(theta),
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "rgba": _HILL_RGBA,
            }
        )
    return hills


def _plan_markers(
    rng: random.Random, count: int, extent: float, clear: float
) -> list[dict[str, Any]]:
    """Visual-only arrows: one wind direction per scene, spread over a ring."""
    wind_yaw = rng.uniform(0.0, 2.0 * math.pi)
    r_in = clear + 3.0
    r_out = max(r_in + 4.0, 0.30 * extent)
    markers = []
    for index in range(count):
        theta = 2.0 * math.pi * (index + rng.uniform(0.15, 0.85)) / count
        r = r_in + (r_out - r_in) * rng.random()
        markers.append(
            {
                "name": f"wind_{index:02d}",
                "x": r * math.cos(theta),
                "y": r * math.sin(theta),
                "z": rng.uniform(1.5, 4.5),
                "yaw": wind_yaw + rng.uniform(-0.25, 0.25),
                "length": rng.uniform(1.8, 2.6),
            }
        )
    return markers


def _plan(spec: SceneSpec) -> dict[str, Any]:
    """One deterministic pass over the spec: everything the emitters need."""
    rng = random.Random(spec.seed)
    extent = float(spec.extent)
    hills = spec.terrain == "hills" and spec.ground
    # Without hills objects may use the whole disc; with hills the outer ring
    # belongs to the terrain, so objects stay in a smaller inner band.
    rmax_objects = extent * (_RING_HILLS if hills else _RING_FLAT)
    limit = 0.98 * extent  # outer edge of any object stays inside the world

    clear = float(spec.clearance)
    buildings, occupied = _plan_buildings(
        rng, spec.buildings, rmax_objects, limit, clear
    )
    trees, occupied = _plan_trees(
        rng, spec.trees, rmax_objects, limit, clear, occupied
    )
    obstacles, occupied = _plan_obstacles(
        rng,
        spec.obstacles,
        min(rmax_objects, max(12.0, 0.35 * extent)),
        limit,
        clear,
        occupied,
    )
    return {
        "buildings": buildings,
        "trees": trees,
        "obstacles": obstacles,
        "hills": _plan_hills(rng, extent, rmax_objects) if hills else [],
        "markers": (
            _plan_markers(rng, 8, extent, clear) if spec.wind_markers else []
        ),
        "rmax_objects": rmax_objects,
    }


def scene_layout(spec: SceneSpec) -> dict[str, Any]:
    """The planned placements, for tests and for picking vehicle spawn points.

    Convenience beyond the §4 contract: ``scene_xml`` is a pure function of the
    spec, so this returns exactly what the emitted XML contains (same x/y/radii)
    without anyone having to re-parse XML.  Every entry carries its ``radius``,
    the bounding radius used for the spawn-box clearance guarantee.
    """
    return _plan(spec)


# --------------------------------------------------------------------------- #
# emitters
# --------------------------------------------------------------------------- #
def _ground_texture_name(spec: SceneSpec, assets: dict[str, Any]) -> str | None:
    """Material name the ground should reference, or ``None`` for flat colour."""
    explicit = assets.get("ground")
    if explicit:
        return str(explicit)
    if spec.ground and spec.ground_texture:
        return str(assets.get("ground_texture", "scene_grass"))
    return None


def _emit_assets(spec: SceneSpec, assets: dict[str, Any]) -> list[str]:
    """The single ``<asset>`` block: gradient skybox and (optionally) turf.

    **The sky must not be geometry.**  Any shell big enough to enclose the arena
    inflates ``model.stat.extent``, and MuJoCo sets a free camera's near plane
    to ``map.znear * stat.extent``: an earlier version built the sky from 189 m
    boxes, which took the extent from 13 m to 535 m and the near plane to
    5.35 m -- farther than a 3.4 m chase rig, so the fully simulated drone was
    clipped out of every frame while the physics ran on.  A ``type="skybox"``
    texture is drawn behind all geometry and costs nothing in the scene bounds.

    Both textures are ``builtin``, so no file on disk is ever referenced.

    The embedder must hoist this block into ``<asset>``; ``env.mjcf.build_xml``
    does exactly that (at most one ``<asset>`` block per fragment).  Everything
    after the block is plain ``<worldbody>`` content, so a caller without an
    asset section can cut the string at ``</asset>`` and still splice valid
    worldbody content (with a black sky and a flat ground).

    Returns ``[]`` when nothing is needed, so ``sky=False, ground_texture=False``
    yields a pure worldbody fragment.
    """
    entries: list[str] = []
    if spec.sky:
        sky_name = str(assets.get("skybox", "scene_sky"))
        entries.append(
            f'<texture name="{sky_name}" type="skybox" builtin="gradient" '
            'rgb1=".20 .40 .66" rgb2=".86 .91 .97" width="256" height="1024"/>'
        )
    grass = _ground_texture_name(spec, assets)
    if grass is not None and "ground" not in assets:
        # Low-contrast checker: it must break up the flat plastic look of a
        # 120 m plane without turning the ground into a chessboard.
        entries.append(
            f'<texture name="{grass}" type="2d" builtin="checker" '
            'rgb1=".25 .36 .21" rgb2=".28 .39 .24" width="512" height="512"/>'
        )
        # Square size is extent-independent: for a plane MuJoCo maps texrepeat
        # repeats over the whole 2 * extent span, and one repeat holds 2 x 2
        # squares, i.e. square = extent / n.
        repeat = max(4, int(round(float(spec.extent) / _GROUND_SQUARE_M)))
        entries.append(
            f'<material name="{grass}" texture="{grass}" '
            f'texrepeat="{repeat} {repeat}" '
            'specular="0" shininess="0" reflectance="0"/>'
        )
    if not entries:
        return []
    return ["  <asset>", *[f"    {entry}" for entry in entries], "  </asset>"]


def _emit_lights(spec: SceneSpec) -> list[str]:
    """A shadow-casting sun plus a soft, near-horizontal fill.

    Deliberately *no* light inside the flight volume: a point light there reads
    as a spotlight on the ground and steals all the contrast.  The fill carries
    a little ``ambient`` so a facade facing away from the sun is never pure
    black.
    """
    extent = float(spec.extent)
    sun = (0.95 * extent, -0.85 * extent, 1.05 * extent)
    norm = math.sqrt(sum(c * c for c in sun)) or 1.0
    direction = tuple(-c / norm for c in sun)
    fill = (0.75 * extent, 0.85 * extent, 0.45 * extent)
    fill_norm = math.sqrt(sum(c * c for c in fill)) or 1.0
    fill_dir = tuple(-c / fill_norm for c in fill)
    return [
        "  <!-- lighting: one shadow-casting sun plus a soft sky fill -->",
        '  <light name="scene_sun" directional="true" castshadow="true" '
        f'pos="{_pos(*sun)}" dir="{_pos(*direction)}" '
        'diffuse=".58 .56 .51" specular=".08 .08 .07"/>',
        '  <light name="scene_fill" pos="{}" dir="{}" '
        'ambient=".12 .13 .15" diffuse=".22 .24 .29" '
        'specular=".02 .02 .02" castshadow="false"/>'.format(
            _pos(*fill), _pos(*fill_dir)
        ),
    ]


def _emit_ground(spec: SceneSpec, assets: dict[str, Any]) -> list[str]:
    # Exactly ``extent``: a bigger plane grows ``stat.extent`` (planes count as
    # 0.2 * size) and with it the camera near plane, for no visual gain.
    ground_half = max(float(spec.extent), 20.0)
    rgba = _GROUND_RGBA[spec.seed % len(_GROUND_RGBA)]
    material = _ground_texture_name(spec, assets)
    surf = f'material="{material}"' if material else f'rgba="{rgba}"'
    lines = [
        "  <!-- ground -->",
        # ``castshadow="false"``: the floor has nothing below it to shadow, and
        # a shadow-casting floor darkens the sky shell wherever a camera looks
        # past the ground edge.
        f'  <geom name="scene_ground" type="plane" pos="0 0 0" '
        f'size="{_f(ground_half)} {_f(ground_half)} 0.05" {surf} '
        f'castshadow="false" friction="0.9 0.005 0.0001" {_COLLIDE}/>',
    ]
    if spec.pad:
        # Decoration only: a collidable 1 cm lip would perturb the landing, and
        # a bright/large disc reads as an over-exposed spotlight in chase shots.
        lines += [
            "  <!-- landing pad: 1.3 m grey marker, no collision -->",
            f'  <geom name="scene_pad" type="cylinder" pos="0 0 0.01" '
            f'size="{_f(_PAD_RADIUS)} 0.01" rgba=".44 .45 .42 1" {_VISUAL}/>',
        ]
    return lines


def _emit_hills(plan: dict[str, Any], assets: dict[str, Any]) -> list[str]:
    surf = _surface(assets, "hill", _HILL_RGBA)
    lines = ["  <!-- terrain: ellipsoid domes, half-buried, far ring only -->"]
    for hill in plan["hills"]:
        lines.append(
            f'    <geom name="{hill["name"]}" type="ellipsoid" '
            f'pos="{_pos(hill["x"], hill["y"], 0.0)}" '
            f'size="{_pos(hill["rx"], hill["ry"], hill["rz"])}" {surf} '
            f'friction="0.9 0.005 0.0001" {_COLLIDE}/>'
        )
    return lines


def _emit_buildings(plan: dict[str, Any], assets: dict[str, Any]) -> list[str]:
    """A stack of shrinking boxes plus roof slab and lit/dark window bands."""
    lines = ["  <!-- buildings: stacked collision boxes, window bands are visual -->"]
    for building in plan["buildings"]:
        name = building["name"]
        lines.append(
            f'  <body name="{name}" pos="{_pos(building["x"], building["y"], 0.0)}" '
            f'quat="{_quat_z(building["yaw"])}">'
        )
        half_w = 0.5 * building["w"]
        half_d = 0.5 * building["d"]
        z = 0.0
        top_half_w, top_half_d = half_w, half_d
        for floor in range(building["floors"]):
            half_h = 0.5 * _STOREY
            top_half_w, top_half_d = half_w, half_d
            lines.append(
                f'    <geom name="{name}_block{floor}" type="box" '
                f'pos="0 0 {_f(z + half_h)}" '
                f'size="{_pos(half_w, half_d, half_h)}" '
                f'{_surface(assets, "building", building["rgba"])} '
                f'friction="0.9 0.005 0.0001" {_COLLIDE}/>'
            )
            window = _WINDOW_LIT if building["lit"] and floor % 2 == 0 else _WINDOW_DARK
            window_surf = _surface(assets, "window", window)
            band_z = z + 0.55 * _STOREY
            band_h = 0.34 * _STOREY
            # Four facade bands, 3 cm proud of the wall so they are never buried.
            lines.append(
                f'    <geom name="{name}_win{floor}a" type="box" '
                f'pos="{_f(half_w + 0.03)} 0 {_f(band_z)}" '
                f'size="{_pos(0.03, 0.82 * half_d, 0.5 * band_h)}" '
                f"{window_surf} {_VISUAL}/>"
            )
            lines.append(
                f'    <geom name="{name}_win{floor}b" type="box" '
                f'pos="{_f(-(half_w + 0.03))} 0 {_f(band_z)}" '
                f'size="{_pos(0.03, 0.82 * half_d, 0.5 * band_h)}" '
                f"{window_surf} {_VISUAL}/>"
            )
            lines.append(
                f'    <geom name="{name}_win{floor}c" type="box" '
                f'pos="0 {_f(half_d + 0.03)} {_f(band_z)}" '
                f'size="{_pos(0.82 * half_w, 0.03, 0.5 * band_h)}" '
                f"{window_surf} {_VISUAL}/>"
            )
            lines.append(
                f'    <geom name="{name}_win{floor}d" type="box" '
                f'pos="0 {_f(-(half_d + 0.03))} {_f(band_z)}" '
                f'size="{_pos(0.82 * half_w, 0.03, 0.5 * band_h)}" '
                f"{window_surf} {_VISUAL}/>"
            )
            z += _STOREY
            half_w *= 0.92
            half_d *= 0.92
        # Roof slab, slightly wider than the top storey: reads as a real roofline
        # and is a real collider, so a vehicle can land on the building.
        lines.append(
            f'    <geom name="{name}_roof" type="box" pos="0 0 {_f(z + 0.16)}" '
            f'size="{_pos(top_half_w + 0.30, top_half_d + 0.30, 0.16)}" '
            f'{_surface(assets, "building_roof", _ROOF_RGBA)} '
            f'friction="0.9 0.005 0.0001" {_COLLIDE}/>'
        )
        # Rooftop plant box: breaks the silhouette so towers do not read as slabs.
        lines.append(
            f'    <geom name="{name}_hatch" type="box" '
            f'pos="{_pos(0.30 * top_half_w, -0.32 * top_half_d, z + 0.55)}" '
            f'size="{_pos(0.9, 0.7, 0.45)}" rgba=".26 .28 .30 1" '
            f'{_COLLIDE}/>'
        )
        lines.append("  </body>")
    return lines


def _emit_trees(plan: dict[str, Any], assets: dict[str, Any]) -> list[str]:
    """Cylinder trunk + sphere crown (plus up to two offset blobs)."""
    lines = ["  <!-- trees: cylinder trunk + sphere crown, all collidable -->"]
    trunk_surf = _surface(assets, "tree_trunk", _TREE_TRUNK)
    for tree in plan["trees"]:
        name = tree["name"]
        lines.append(
            f'  <body name="{name}" pos="{_pos(tree["x"], tree["y"], 0.0)}" '
            f'quat="{_quat_z(tree["yaw"])}">'
        )
        lines.append(
            f'    <geom name="{name}_trunk" type="cylinder" '
            f'pos="0 0 {_f(0.5 * tree["trunk_h"])}" '
            f'size="{_pos(tree["trunk_r"], 0.5 * tree["trunk_h"], 0.0)}" '
            f'{trunk_surf} friction="0.9 0.005 0.0001" {_COLLIDE}/>'
        )
        crown_surf = _surface(assets, "tree_crown", tree["crown_rgba"])
        lines.append(
            f'    <geom name="{name}_crown" type="sphere" '
            f'pos="0 0 {_f(tree["crown_z"])}" size="{_f(tree["crown_r"])}" '
            f"{crown_surf} friction=\"0.9 0.005 0.0001\" {_COLLIDE}/>"
        )
        for index, blob in enumerate(tree["offsets"]):
            lines.append(
                f'    <geom name="{name}_crown{index + 1}" type="sphere" '
                f'pos="{_pos(blob["dx"], blob["dy"], tree["crown_z"] + blob["dz"])}" '
                f'size="{_f(blob["r"])}" {crown_surf} '
                f'friction="0.9 0.005 0.0001" {_COLLIDE}/>'
            )
        lines.append("  </body>")
    return lines


def _emit_obstacles(plan: dict[str, Any], assets: dict[str, Any]) -> list[str]:
    lines = ["  <!-- obstacles: wooden crates, poles, safety barriers -->"]
    for obstacle in plan["obstacles"]:
        name = obstacle["name"]
        lines.append(
            f'  <body name="{name}" pos="{_pos(obstacle["x"], obstacle["y"], 0.0)}" '
            f'quat="{_quat_z(obstacle["yaw"])}">'
        )
        surf = _surface(assets, "obstacle", obstacle["rgba"])
        if obstacle["kind"] == "pole":
            lines.append(
                f'    <geom name="{name}_post" type="cylinder" '
                f'pos="0 0 {_f(0.5 * obstacle["sz"])}" '
                f'size="{_pos(0.5 * obstacle["sx"], 0.5 * obstacle["sz"], 0.0)}" '
                f'{surf} friction="0.9 0.005 0.0001" {_COLLIDE}/>'
            )
        else:
            lines.append(
                f'    <geom name="{name}_box" type="box" '
                f'pos="0 0 {_f(0.5 * obstacle["sz"])}" '
                f'size="{_pos(0.5 * obstacle["sx"], 0.5 * obstacle["sy"], 0.5 * obstacle["sz"])}" '
                f'{surf} friction="0.9 0.005 0.0001" {_COLLIDE}/>'
            )
        lines.append("  </body>")
    return lines


def _emit_markers(plan: dict[str, Any], assets: dict[str, Any]) -> list[str]:
    """Wind arrows: shaft capsule + cone-ish ellipsoid head, visual only.

    Deliberately ``contype="0" conaffinity="0"``: a marker must never enter the
    contact computation, so enabling it cannot change a single trajectory bit.
    """
    surf = _surface(assets, "marker", _MARKER_RGBA)
    lines = [
        "  <!-- wind markers: visual-only arrows (no collision, no force) -->",
        "  <!-- arrows point along the scene wind direction; see scene_layout() -->",
    ]
    for marker in plan["markers"]:
        name = marker["name"]
        length = marker["length"]
        head = 0.26 * length
        lines.append(
            f'  <body name="{name}" pos="{_pos(marker["x"], marker["y"], marker["z"])}" '
            f'quat="{_quat_z(marker["yaw"])}">'
        )
        lines.append(
            f'    <geom name="{name}_shaft" type="capsule" '
            f'fromto="0 0 0 {_f(length - head)} 0 0" size="0.085" '
            f"{surf} {_VISUAL}/>"
        )
        lines.append(
            f'    <body name="{name}_tip" pos="{_f(length - head)} 0 0" '
            f'quat="{_QUAT_Y_P90}">'
        )
        lines.append(
            f'      <geom name="{name}_head" type="ellipsoid" '
            f'size="{_pos(0.18, 0.18, head)}" '
            f"{surf} {_VISUAL}/>"
        )
        lines.append("    </body>")
        lines.append("  </body>")
    return lines


def scene_xml(spec: SceneSpec, assets: dict[str, Any] | None = None) -> str:
    """Return a ``<worldbody>``-insertable XML fragment for ``spec``.

    Parameters
    ----------
    spec :
        The scene description.  Purely a value: same spec -> same bytes.
    assets :
        Optional ``{role: name}`` map.  Surface roles are ``ground``,
        ``building``, ``building_roof``, ``window``, ``tree_trunk``,
        ``tree_crown``, ``hill``, ``obstacle`` and ``marker``; when a role is
        present the geom references ``material="<name>"`` (M1's ``<asset>``
        section), otherwise an inline ``rgba`` is used so the fragment is
        self-contained.  The special key ``skybox`` renames the gradient skybox
        texture.  Extra keys are ignored.

    Structure: when ``sky`` is true the string starts with exactly one
    ``<asset>`` block (a built-in gradient skybox texture, no external file);
    everything after it is plain ``<worldbody>`` content -- lighting, ground,
    buildings, trees, crates/poles, optional wind markers.  There is no
    ``<visual>``, ``<option>`` or ``<compiler>`` element and no external file
    reference, so the fragment can never disturb the caller's camera, timestep
    or index conventions.

    The single ``<asset>`` block is the *only* way to paint a sky without
    worldbody geometry, and worldbody geometry must not be used: anything big
    enough to enclose the arena inflates ``model.stat.extent``, which MuJoCo
    multiplies into the camera near plane and which clipped the drone out of
    every chase shot.  ``env/mjcf.build_xml`` hoists this block into
    ``<asset>``; a caller without an asset section can cut the string at
    ``</asset>`` and still splice valid worldbody content.
    """
    if not isinstance(spec, SceneSpec):
        raise TypeError(f"scene_xml() needs a SceneSpec, got {type(spec).__name__}")
    assets = dict(assets or {})
    plan = _plan(spec)

    lines: list[str] = [
        f"  <!-- 3-D scene: seed={spec.seed} terrain={spec.terrain} "
        f"extent={_f(spec.extent)} -->"
    ]
    lines.extend(_emit_assets(spec, assets))
    lines.extend(_emit_lights(spec))
    if spec.ground:
        lines.extend(_emit_ground(spec, assets))
        if plan["hills"]:
            lines.extend(_emit_hills(plan, assets))
    if spec.buildings:
        lines.extend(_emit_buildings(plan, assets))
    if spec.trees:
        lines.extend(_emit_trees(plan, assets))
    if spec.obstacles:
        lines.extend(_emit_obstacles(plan, assets))
    if spec.wind_markers:
        lines.extend(_emit_markers(plan, assets))
    return "\n".join(lines) + "\n"
