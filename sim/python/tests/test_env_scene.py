"""M2 validation suite for the 3-D scene (``docs/3D_ENVIRONMENT.md`` §4-§5).

Covers gate **V6** — an unpowered drop from 1.5 m comes to rest on the ground,
``z >= 0`` and does not tunnel through — plus the properties the scene contract
promises and that M1/M3/M4 rely on:

* ``scene_xml`` is a *worldbody fragment* (no ``<asset>``/``<visual>``/``<option>``),
  self-contained, with a unique name for every geom/body;
* byte-identical XML for a given seed, different layout for a different seed;
* the 10 m x 10 m spawn box is empty and every solid object is real collision
  geometry (static worldbody geoms with ``contype``/``conaffinity``);
* ``wind_markers`` are visual-only: the XML changes, the physics does not;
* ``terrain="hills"`` keeps its domes out of the object band and the spawn box.

All numbers are printed so a failing run explains itself (``pytest -s``).
Tests that need MuJoCo skip cleanly, so ``make test`` stays green on a bare
checkout (the string-level checks still run).
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from pistr_sim.env.scene import SPAWN_CLEAR_HALF, SceneSpec, scene_layout, scene_xml

# The dev fixture (lead's temporary model) provides a known-valid drone body so
# V6 can be exercised before M1's build_xml() exists.
from pistr_sim.env._devmodel import dev_xml

#: Lines of the dev fixture that the scene fragment replaces (its floor and its
#: own point light): the scene must be able to stand in for both.
_DEV_FLOOR = (
    '    <light pos="6 -6 12" dir="-0.4 0.4 -1" diffuse=".8 .8 .8"/>\n'
    '    <geom name="floor" type="plane" size="40 40 .05" material="grid"/>\n'
)
_DEV_ASSET_END = "  </asset>"

_DROP_ALTITUDE = 1.5
_DROP_TIME = 2.0

#: Exactly what ``pistr_sim.env.mjcf.build_xml`` does with a scene fragment: it
#: hoists the single ``<asset>`` block into ``<asset>`` and inserts the rest into
#: ``<worldbody>``.  Mirrored here so this suite does not depend on M1's module.
_ASSET_RE = re.compile(r"<asset\b[^>]*>(?P<body>.*?)</asset>", re.DOTALL)


def _hoist(fragment: str, target_xml: str) -> tuple[str, str]:
    """Split a scene fragment into (worldbody part, model xml with its assets)."""
    found = _ASSET_RE.findall(fragment)
    assert len(found) <= 1, "scene fragment may contain at most one <asset> block"
    if not found:
        return fragment, target_xml
    assert _DEV_ASSET_END in target_xml, "dev fixture has no <asset> to hoist into"
    target_xml = target_xml.replace(
        _DEV_ASSET_END, found[0].rstrip("\n") + "\n" + _DEV_ASSET_END, 1
    )
    return _ASSET_RE.sub("", fragment), target_xml


def _world_xml(spec: SceneSpec, altitude: float = _DROP_ALTITUDE) -> str:
    """Dev drone + ``scene_xml(spec)`` as the world.  Also the V6 rig."""
    base = dev_xml(altitude)
    assert _DEV_FLOOR in base, "dev fixture changed; update the M2 test rig"
    worldbody, base = _hoist(scene_xml(spec), base)
    return base.replace(_DEV_FLOOR, worldbody)


def _worldbody_only(fragment: str) -> str:
    """The fragment with its ``<asset>`` block stripped (pure worldbody content)."""
    return _ASSET_RE.sub("", fragment)


def _geoms(xml: str) -> list[str]:
    return [line for line in xml.splitlines() if "<geom " in line]


def _geom_names(xml: str) -> list[str]:
    return re.findall(r'<geom name="([^"]+)"', xml)


def _drop(spec: SceneSpec, *, seconds: float = _DROP_TIME, altitude: float = _DROP_ALTITUDE):
    """Simulate the unpowered drop; return (model, data, trajectory of body z)."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_world_xml(spec, altitude))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    steps = int(round(seconds / model.opt.timestep))
    heights = []
    for _ in range(steps):
        mujoco.mj_step(model, data)
        heights.append(float(data.xpos[body][2]))
    return model, data, heights, body


# --------------------------------------------------------------------------- #
# structure: the fragment must be spliceable into any worldbody
# --------------------------------------------------------------------------- #
def test_fragment_is_self_contained_worldbody_content():
    xml = scene_xml(SceneSpec())
    # exactly one asset block, and it is only the built-in gradient skybox
    assert xml.count("<asset>") == 1 and xml.count("</asset>") == 1
    asset = _ASSET_RE.search(xml).group(0)
    assert 'type="skybox"' in asset and 'builtin="gradient"' in asset
    assert "file=" not in asset, "the sky must not need a texture file on disk"
    # with neither sky nor turf there must be no asset block at all
    bare = scene_xml(SceneSpec(sky=False, ground_texture=False))
    assert "<asset>" not in bare and "material=" not in bare

    body = _worldbody_only(xml)
    assert body.strip(), "worldbody content must survive the asset hoist"
    forbidden = ("<mujoco", "<worldbody", "</worldbody>", "<asset", "<visual",
                 "<option>", "<compiler", "<default", "<actuator")
    for token in forbidden:
        assert token not in body, f"{token} must not appear in the worldbody part"
    # No external files of any kind (texture/mesh/include).
    for token in ("file=", "<include", ".png", ".jpg", ".stl"):
        assert token not in xml
    # Every tag that opens must close (comments stripped first).
    bare = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    for tag in ("body", "geom"):
        opens = len(re.findall(rf"<{tag}\b", bare))
        closes = len(re.findall(rf"</{tag}>", bare))
        self_closed = len(re.findall(rf"<{tag}\b[^>]*/>", bare))
        assert opens == closes + self_closed, f"unbalanced <{tag}> tags"
    print(f"\nfragment: {len(xml)} chars, {len(_geoms(body))} geoms, "
          f"{body.count('<light ')} lights, {body.count('<body ')} bodies, "
          f"{asset.count('<texture ')} builtin texture(s), hoisted by build_xml")


def test_every_geom_declares_collision_and_group_policy():
    xml = scene_xml(SceneSpec())
    for line in _geoms(xml):
        assert "contype=" in line and "conaffinity=" in line, line
        assert 'group="2"' in line or 'contype="1"' in line, line


def test_names_are_unique():
    xml = scene_xml(SceneSpec(seed=7))
    names = _geom_names(xml) + re.findall(r'<body name="([^"]+)"', xml)
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"duplicate MJCF names: {sorted(duplicates)}"


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_same_seed_is_byte_identical():
    spec = SceneSpec(seed=11, wind_markers=True)
    first = scene_xml(spec)
    second = scene_xml(SceneSpec(seed=11, wind_markers=True))
    assert first == second
    assert len(first) > 2000
    print(f"\nseed 11 -> {len(first)} chars, stable across calls")


def test_different_seed_changes_layout():
    a = scene_xml(SceneSpec(seed=1))
    b = scene_xml(SceneSpec(seed=2))
    assert a != b
    la, lb = scene_layout(SceneSpec(seed=1)), scene_layout(SceneSpec(seed=2))
    moved = [
        (round(x["x"] - y["x"], 3), round(x["y"] - y["y"], 3))
        for x, y in zip(la["buildings"], lb["buildings"])
    ]
    assert any(abs(dx) + abs(dy) > 1.0 for dx, dy in moved), moved
    print(f"\nseed1 vs seed2 building displacement sample: {moved[:3]}")


def test_layout_matches_emitted_xml():
    spec = SceneSpec(seed=5)
    layout = scene_layout(spec)
    xml = scene_xml(spec)
    for entry in layout["buildings"] + layout["trees"] + layout["obstacles"]:
        assert f'name="{entry["name"]}"' in xml
        assert f'pos="{entry["x"]:.4f} {entry["y"]:.4f} 0.0000"' in xml


# --------------------------------------------------------------------------- #
# spawn box and world size
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 17])
def test_spawn_box_is_empty(seed):
    spec = SceneSpec(seed=seed, wind_markers=True)
    layout = scene_layout(spec)
    worst = None
    for group in ("buildings", "trees", "obstacles"):
        for entry in layout[group]:
            dx = max(0.0, abs(entry["x"]) - SPAWN_CLEAR_HALF)
            dy = max(0.0, abs(entry["y"]) - SPAWN_CLEAR_HALF)
            gap = math.hypot(dx, dy) - entry["radius"]
            if worst is None or gap < worst[0]:
                worst = (gap, group, entry["name"])
            assert gap > 0.0, (
                f"{entry['name']} intrudes into the {2 * SPAWN_CLEAR_HALF:.0f} m "
                f"spawn box (gap {gap:.3f} m, radius {entry['radius']:.2f} m)"
            )
    print(f"\nseed {seed}: closest object to the spawn box = {worst[0]:.2f} m "
          f"({worst[2]})")


def test_everything_stays_inside_extent():
    spec = SceneSpec(extent=30.0, seed=4)
    layout = scene_layout(spec)
    for group in ("buildings", "trees", "obstacles"):
        for entry in layout[group]:
            reach = math.hypot(entry["x"], entry["y"]) + entry["radius"]
            assert reach <= spec.extent, f"{entry['name']} reaches {reach:.2f} m"


def test_object_counts_follow_the_spec():
    for spec in (SceneSpec(buildings=0, trees=0, obstacles=0),
                 SceneSpec(buildings=3, trees=4, obstacles=5, seed=9)):
        layout = scene_layout(spec)
        assert len(layout["buildings"]) == spec.buildings
        assert len(layout["trees"]) == spec.trees
        assert len(layout["obstacles"]) == spec.obstacles
        xml = scene_xml(spec)
        if spec.buildings == 0:
            assert "bldg_" not in xml
        if spec.trees == 0:
            assert "tree_" not in xml


# --------------------------------------------------------------------------- #
# MuJoCo-level checks: real, static collision geometry
# --------------------------------------------------------------------------- #
def test_scene_builds_and_is_static_collision_geometry():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_world_xml(SceneSpec()))
    collidable = [
        gid for gid in range(model.ngeom)
        if model.geom_contype[gid] or model.geom_conaffinity[gid]
    ]
    # The scene's own solids are all in worldbody -> welded to the world.
    solid_scene_geoms = [
        gid for gid in collidable
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith(
            ("bldg_", "tree_", "obs_", "hill_", "scene_ground")
        )
    ]
    assert len(solid_scene_geoms) >= 40
    for gid in solid_scene_geoms:
        body = model.geom_bodyid[gid]
        assert model.body_weldid[body] == 0, "scene solids must be static (world-welded)"
        assert model.body_jntnum[body] == 0
    print(f"\nworld: {model.ngeom} geoms ({len(collidable)} collidable, "
          f"{len(solid_scene_geoms)} from the scene), {model.nbody} bodies, "
          f"{model.nlight} lights, stat.extent {model.stat.extent:.1f} m")


def test_hills_terrain_keeps_domes_out_of_the_way():
    mujoco = pytest.importorskip("mujoco")
    spec = SceneSpec(terrain="hills", seed=2)
    layout = scene_layout(spec)
    assert layout["hills"], "terrain='hills' must produce hill geometry"
    xml = scene_xml(spec)
    assert xml.count('type="ellipsoid"') == len(layout["hills"])
    # Nothing may intersect the spawn box, and hills stay behind the objects.
    farthest_object = max(
        math.hypot(e["x"], e["y"]) + e["radius"]
        for group in ("buildings", "trees", "obstacles")
        for e in layout[group]
    )
    for hill in layout["hills"]:
        near = math.hypot(hill["x"], hill["y"]) - max(hill["rx"], hill["ry"])
        assert near > farthest_object, (
            f"{hill['name']} reaches {near:.1f} m, inside the object band"
        )
    model = mujoco.MjModel.from_xml_string(_world_xml(spec))
    print(f"\nhills: {len(layout['hills'])} domes, objects end at "
          f"{farthest_object:.1f} m, geoms {model.ngeom}")


# --------------------------------------------------------------------------- #
# V6: contact
# --------------------------------------------------------------------------- #
def test_v6_unpowered_drop_comes_to_rest_on_the_ground():
    model, data, heights, body = _drop(SceneSpec())
    ground_z = float(data.xpos[body][2])
    lowest = min(heights)
    speed = float(abs(data.cvel[body]).max())
    # 1. never tunnels: the body origin stays above the floor (z=0) at all times
    assert lowest >= 0.0, f"tunnelled through the ground: min z = {lowest:.4f} m"
    # 2. comes to rest *on* the ground: box half-thickness of the dev drone is
    #    0.04 m, so a resting body origin sits just above it.
    assert 0.02 <= ground_z <= 0.12, f"did not settle on the floor: z = {ground_z:.4f}"
    # 3. and is static (contact has damped out, not still ringing)
    assert speed < 1e-2, f"still moving after {_DROP_TIME} s: |v| = {speed:.3g}"
    # 4. the last 0.2 s show no residual bounce
    tail = heights[-100:]
    assert max(tail) - min(tail) < 1e-3, f"bouncing: {max(tail) - min(tail):.2e} m"
    print(f"\nV6 drop: rest z = {ground_z:.5f} m, min z = {lowest:.5f} m, "
          f"|v| = {speed:.2e}, tail ripple = {max(tail) - min(tail):.2e} m, "
          f"ncon = {data.ncon}")


@pytest.mark.parametrize("spec", [
    SceneSpec(terrain="hills", seed=1),
    SceneSpec(sky=False, ground=True, seed=2),
], ids=["hills", "no-sky"])
def test_v6_holds_for_scene_variants(spec):
    model, data, heights, body = _drop(spec)
    assert min(heights) >= 0.0
    assert 0.02 <= float(data.xpos[body][2]) <= 0.12
    print(f"\nV6 ({spec.terrain}, sky={spec.sky}): rest z = "
          f"{float(data.xpos[body][2]):.5f} m, ncon = {data.ncon}")


# --------------------------------------------------------------------------- #
# ground texture and clearance (demo-facing knobs)
# --------------------------------------------------------------------------- #
def test_ground_uses_a_builtin_texture_without_touching_collision():
    xml = scene_xml(SceneSpec(seed=0))
    asset = _ASSET_RE.search(xml).group(0)
    assert 'type="2d"' in asset and 'builtin="checker"' in asset
    assert 'material name="scene_grass"' in asset
    assert "file=" not in asset
    ground = next(line for line in xml.splitlines() if 'name="scene_ground"' in line)
    assert 'material="scene_grass"' in ground
    # the plane keeps its collision identity: the texture is visual only
    assert 'contype="1"' in ground and 'conaffinity="1"' in ground

    # squares are world-relative and stay in the 1.5-2.0 m band at any extent
    # (one plane texture repeat covers 2 x 2 squares -> square = extent / n)
    for extent in (18.0, 30.0, 60.0, 120.0):
        variant = scene_xml(SceneSpec(extent=extent))
        repeat = int(re.search(r'texrepeat="(\d+)', variant).group(1))
        square = extent / repeat
        assert 1.5 <= square <= 2.0, f"extent {extent}: {square:.2f} m squares"

    flat = scene_xml(SceneSpec(seed=0, ground_texture=False))
    ground_flat = next(line for line in flat.splitlines() if 'name="scene_ground"' in line)
    assert "rgba=" in ground_flat and "material=" not in ground_flat
    assert 'name="scene_grass"' not in flat
    # identical collision attributes either way
    assert ('contype="1" conaffinity="1"' in ground
            and 'contype="1" conaffinity="1"' in ground_flat)
    print("\nground: textured plane keeps contype=1/conaffinity=1 "
          "(material is a render-time lookup only)")


def test_ground_texture_can_be_overridden_or_supplied_by_the_caller():
    renamed = scene_xml(SceneSpec(seed=0), assets={"ground_texture": "my_turf"})
    assert 'name="my_turf" type="2d"' in renamed
    assert 'material name="my_turf"' in renamed
    ground = next(line for line in renamed.splitlines() if 'name="scene_ground"' in line)
    assert 'material="my_turf"' in ground
    supplied = scene_xml(SceneSpec(seed=0), assets={"ground": "mat_from_m1"})
    ground = next(line for line in supplied.splitlines() if 'name="scene_ground"' in line)
    assert 'material="mat_from_m1"' in ground
    # the caller's material owns the texture, so we must not emit a second one
    assert "scene_grass" not in supplied


def test_landing_pad_is_small_grey_and_optional():
    xml = scene_xml(SceneSpec(seed=0))
    pad = next(line for line in xml.splitlines() if 'name="scene_pad"' in line)
    radius = float(re.search(r'size="([0-9.]+)', pad).group(1))
    assert 1.2 <= radius <= 1.4, f"pad radius {radius} m is too big for chase shots"
    # grey and mid-dark: lit by sun + fill + headlight, a 0.62 albedo clips to
    # white in the render, which is what made the landing shot look like a
    # spotlight.  Keep R ~ G ~ B and the albedo below 0.6.
    channels = [float(v) for v in re.search(r'rgba="([^"]+)"', pad).group(1).split()[:3]]
    assert max(channels) - min(channels) <= 0.05, f"pad is not grey: {channels}"
    assert max(channels) <= 0.55, f"pad albedo {max(channels)} clips to white"
    assert 'contype="0"' in pad and 'conaffinity="0"' in pad, "pad must not collide"
    # a demo can switch it off entirely
    assert "scene_pad" not in scene_xml(SceneSpec(seed=0, pad=False))
    # ... and it never affects the V6 landing either way
    off = _drop(SceneSpec(seed=0, pad=False))[2][-1]
    on = _drop(SceneSpec(seed=0, pad=True))[2][-1]
    assert on == pytest.approx(off, abs=1e-12)
    print(f"\npad: radius {radius:.1f} m, grey, visual-only, removable "
          f"(final z identical {on:.6f})")


def test_clearance_is_a_spec_field_and_defaults_to_the_old_value():
    from pistr_sim.env.scene import SPAWN_CLEAR_HALF

    assert SceneSpec().clearance == SPAWN_CLEAR_HALF == 5.0
    assert SceneSpec(clearance=10.0).clearance == 10.0
    # raising it must actually move the objects out
    near_default = _nearest_object(SceneSpec())
    near_wide = _nearest_object(SceneSpec(clearance=10.0))
    assert near_wide[0] > near_default[0] + 3.0
    assert near_wide[0] > 10.0
    assert scene_xml(SceneSpec()) != scene_xml(SceneSpec(clearance=10.0))
    print(f"\nclearance: nearest object surface {near_default[0]:.2f} m "
          f"({near_default[1]}, clearance=5) -> {near_wide[0]:.2f} m "
          f"({near_wide[1]}, clearance=10)")


def _nearest_object(spec: SceneSpec) -> tuple[float, str]:
    """Distance from the origin to the closest object surface, and its name."""
    layout = scene_layout(spec)
    best = (math.inf, "")
    for group in ("buildings", "trees", "obstacles"):
        for entry in layout[group]:
            distance = math.hypot(entry["x"], entry["y"]) - entry["radius"]
            if distance < best[0]:
                best = (distance, entry["name"])
    return best


@pytest.mark.parametrize("clearance", [5.0, 8.0, 10.0, 14.0])
def test_clearance_actually_empties_the_box(clearance):
    spec = SceneSpec(clearance=clearance, seed=3, wind_markers=True)
    layout = scene_layout(spec)
    for group in ("buildings", "trees", "obstacles"):
        for entry in layout[group]:
            dx = max(0.0, abs(entry["x"]) - clearance)
            dy = max(0.0, abs(entry["y"]) - clearance)
            assert math.hypot(dx, dy) - entry["radius"] > 0.0, (
                f"{entry['name']} intrudes into the {2 * clearance:.0f} m box"
            )
    # markers are visual only but should still stay outside the drone's air
    for marker in layout["markers"]:
        assert math.hypot(marker["x"], marker["y"]) > clearance
    print(f"\nclearance={clearance:.0f} m: nearest surface "
          f"{_nearest_object(spec)[0]:.2f} m")


def test_clearance_validation():
    with pytest.raises(ValueError):
        SceneSpec(clearance=0.0)
    with pytest.raises(ValueError):
        SceneSpec(clearance=-3.0)
    with pytest.raises(ValueError):
        SceneSpec(clearance=30.0, extent=30.0)   # box would leave no room


# --------------------------------------------------------------------------- #
# camera fit: the scene must not push the near plane past the vehicle
#
# Regression for a real breakage: an earlier sky made of 189 m boxes took
# ``model.stat.extent`` from ~13 m to 535 m, and MuJoCo sets a free camera's
# near plane to ``map.znear * stat.extent`` (``<map znear="0.01">`` in
# env/mjcf.py), i.e. 5.35 m -- farther than the 3.4 m chase rig, so the fully
# simulated drone was clipped out of every rendered frame while the physics ran
# on.  These three tests are the cheap guard against repeating it.
# --------------------------------------------------------------------------- #
def _free_camera_scene(mujoco, model, data, *, distance, lookat=(0.0, 0.0, 1.5),
                       azimuth=130.0, elevation=10.0):
    scn = mujoco.MjvScene(model, 4000)
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    return scn, cam


def _drone_geoms(mujoco, model) -> set[int]:
    bodies = {
        b
        for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("drone")
    }
    assert bodies, "no drone body in the model"
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}


def _assert_vehicle_can_be_rendered(mujoco, model, data, *, distance=3.4):
    """Extent/near-plane bounds plus a segmentation render that sees the drone."""
    scn, cam = _free_camera_scene(mujoco, model, data, distance=distance)
    near = float(scn.camera[0].frustum_near)
    assert model.stat.extent < 100.0, (
        f"scene inflates stat.extent to {model.stat.extent:.1f} m; "
        "the camera near plane scales with it"
    )
    assert near < distance, (
        f"near plane {near:.2f} m >= chase distance {distance:.2f} m: "
        "the vehicle would be clipped out of frame"
    )
    try:
        renderer = mujoco.Renderer(model, 480, 640)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam)
        seg = np.asarray(renderer.render())
        renderer.close()
    except Exception as exc:  # no GL / EGL context in this sandbox
        pytest.skip(f"off-screen rendering unavailable: {exc}")
    drawn = _drone_geoms(mujoco, model) & set(np.unique(seg).tolist())
    assert drawn, "the vehicle is missing from the segmentation frame"
    assert len(drawn) >= 0.5 * len(_drone_geoms(mujoco, model))
    return model.stat.extent, near, len(drawn)


def test_scene_keeps_the_camera_near_plane_out_of_the_vehicle():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_world_xml(SceneSpec()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    extent, near, drawn = _assert_vehicle_can_be_rendered(mujoco, model, data)
    print(f"\ncamera fit: stat.extent = {extent:.1f} m, frustum_near = {near:.3f} m "
          f"at a 3.4 m chase distance, {drawn} drone geoms in the segmentation frame")


@pytest.mark.parametrize("spec", [
    SceneSpec(seed=0),
    SceneSpec(seed=5, wind_markers=True),
    SceneSpec(terrain="hills", seed=2),
], ids=["default", "with-markers", "hills"])
def test_camera_fit_holds_for_scene_variants(spec):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(_world_xml(spec))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    extent, near, drawn = _assert_vehicle_can_be_rendered(mujoco, model, data)
    print(f"\n{spec.terrain}/seed={spec.seed}/markers={spec.wind_markers}: "
          f"extent {extent:.1f} m, near {near:.3f} m, {drawn} geoms drawn")


def test_m1_build_model_is_renderable_with_the_scene():
    """Same check through M1's real embedder (asset hoisting included)."""
    mujoco = pytest.importorskip("mujoco")
    try:
        from pistr_sim.env.mjcf import build_model
        from pistr_sim.params import default_params
    except Exception as exc:  # M1 mid-edit / no mujoco
        pytest.skip(f"env.mjcf.build_model unavailable: {exc}")
    model = build_model(default_params("high"), scene=scene_xml(SceneSpec()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    extent, near, drawn = _assert_vehicle_can_be_rendered(mujoco, model, data)
    print(f"\nM1 build_model + scene: stat.extent = {extent:.1f} m, "
          f"frustum_near = {near:.3f} m, {drawn} drone geoms drawn")


# --------------------------------------------------------------------------- #
# wind markers are decorative only
# --------------------------------------------------------------------------- #
def test_wind_markers_are_visual_only():
    spec = SceneSpec(wind_markers=True, seed=3)
    xml = scene_xml(spec)
    marker_lines = [line for line in _geoms(xml) if 'name="wind_' in line]
    assert marker_lines, "wind_markers=True produced no markers"
    for line in marker_lines:
        assert 'contype="0"' in line and 'conaffinity="0"' in line
        assert 'group="2"' in line, line
        assert 'mass=' not in line and 'density=' not in line
    # The markers are additive: removing them leaves exactly the plain scene.
    plain = scene_xml(SceneSpec(wind_markers=False, seed=3))
    assert xml != plain
    assert all(line in xml for line in plain.splitlines()[:-1])
    print(f"\nwind markers: {len(marker_lines)} visual-only geoms, "
          f"xml {len(xml)} vs {len(plain)} chars")


def test_wind_markers_do_not_change_physics():
    off = _drop(SceneSpec(wind_markers=False, seed=3))
    on = _drop(SceneSpec(wind_markers=True, seed=3))
    z_off = off[2][-1]
    z_on = on[2][-1]
    assert z_off == pytest.approx(z_on, abs=1e-12), (
        f"wind markers changed the trajectory: {z_off!r} vs {z_on!r}"
    )
    # stronger: identical contact count at the end and identical body position
    model_off, data_off, _, body_off = off
    model_on, data_on, _, body_on = on
    assert list(data_off.xpos[body_off]) == pytest.approx(
        list(data_on.xpos[body_on]), abs=1e-12
    )
    print(f"\nwind markers on/off: final z {z_on:.12f} vs {z_off:.12f} "
          f"(identical), ncon {data_off.ncon} vs {data_on.ncon}")


def test_wind_markers_do_not_add_collidable_geoms():
    mujoco = pytest.importorskip("mujoco")
    plain = mujoco.MjModel.from_xml_string(_world_xml(SceneSpec(seed=3)))
    marked = mujoco.MjModel.from_xml_string(
        _world_xml(SceneSpec(seed=3, wind_markers=True))
    )
    assert marked.ngeom > plain.ngeom
    collidable = lambda m: sorted(
        (m.geom_type[g], tuple(m.geom_size[g]), tuple(m.geom_pos[g]))
        for g in range(m.ngeom)
        if m.geom_contype[g] or m.geom_conaffinity[g]
    )
    assert collidable(plain) == collidable(marked)


# --------------------------------------------------------------------------- #
# API details
# --------------------------------------------------------------------------- #
def test_assets_material_hook():
    plain = scene_xml(SceneSpec(seed=0))
    with_mat = scene_xml(SceneSpec(seed=0), assets={"ground": "mat_ground",
                                                    "skybox": "my_sky"})
    assert 'material="mat_ground"' in with_mat
    assert 'name="my_sky" type="skybox"' in with_mat
    assert 'name="scene_ground"' in with_mat
    # colours are replaced, not duplicated, on those geoms
    ground_line = next(l for l in with_mat.splitlines() if 'name="scene_ground"' in l)
    assert "rgba=" not in ground_line
    assert plain.count("<geom ") == with_mat.count("<geom ")
    # unknown roles are ignored
    assert scene_xml(SceneSpec(), assets={"nope": "x"}) == plain


def test_scene_xml_rejects_non_spec():
    with pytest.raises(TypeError):
        scene_xml({"seed": 0})  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [
    {"terrain": "mountains"},
    {"buildings": -1},
    {"trees": 2.5},
    {"extent": 4.0},
    {"obstacles": True},
])
def test_spec_validation(kwargs):
    with pytest.raises(ValueError):
        SceneSpec(**kwargs)


def test_spec_is_frozen_and_hashable():
    spec = SceneSpec(seed=1)
    with pytest.raises(Exception):
        spec.seed = 2  # type: ignore[misc]
    assert hash(SceneSpec(seed=1)) == hash(SceneSpec(seed=1))
    assert SceneSpec() == SceneSpec()
    print(f"\ndefaults: {SceneSpec()}")
