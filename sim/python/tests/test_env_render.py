"""Validation suite for the M3 render stack: cameras, frames, MP4.

Implements gate **V7** of ``docs/3D_ENVIRONMENT.md`` §5::

    V7  Render one 320x240 frame headless to a non-uniform image, and write_mp4()
        produces a playable file with the expected frame count.

The fixture is ``pistr_sim.env._devmodel.build_dev_model``: a deliberately minimal
scene (checker ground, one coloured rotor per arm, a point light) whose geometry is
known to be valid.  It is *unactuated*, so the tests never integrate it for long —
``mj_forward`` after construction keeps the vehicle at its 1.5 m hover pose.

MuJoCo is an optional dependency, so the whole module skips cleanly when it is not
importable (``make test-py`` on a bare checkout).  The MP4 checks additionally skip
when ``ffmpeg``/``ffprobe`` are missing, with the reason printed.

Run with ``pytest -s`` to see the measured numbers behind each assertion.
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="MuJoCo is an optional dependency")

from pistr_sim.env._devmodel import build_dev_model  # noqa: E402
from pistr_sim.env.camera import Camera, make_camera  # noqa: E402
from pistr_sim.env.render import (  # noqa: E402
    HeadlessRenderer,
    ffmpeg_path,
    ffprobe_path,
    write_mp4,
    write_png,
)

V7_SIZE = (320, 240)  # (width, height) required by the gate


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def dev():
    """A fresh dev model plus its MjData, at the 1.5 m hover pose (not integrated)."""
    model = build_dev_model(1.5)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.qpos[2] == pytest.approx(1.5)
    return model, data


@pytest.fixture()
def renderer(dev):
    """A 320x240 renderer with a chase camera aimed at the vehicle."""
    model, data = dev
    cam = make_camera(model, "chase", fovy=45)
    cam.set_target(data.qpos[:3], yaw=0.0)
    with HeadlessRenderer(model, *V7_SIZE, camera=cam) as r:
        yield r, data, cam


def _quat(yaw=0.0, pitch=0.0, roll=0.0) -> np.ndarray:
    """Quaternion ``(w, x, y, z)`` for a ZYX Euler triple, all angles in radians."""
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    qz = np.array([cy, 0.0, 0.0, sy])
    qy = np.array([cp, 0.0, sp, 0.0])
    qx = np.array([cr, sr, 0.0, 0.0])

    def mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    q = mul(mul(qz, qy), qx)
    return q / np.linalg.norm(q)


def _probe(path) -> dict:
    """ffprobe one file as JSON (stream-level frame count, size, duration)."""
    exe = ffprobe_path()
    cmd = [
        exe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,duration,width,height,codec_name,pix_fmt,r_frame_rate",
        "-show_entries",
        "format=duration,format_name",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def _frame_count(probe: dict) -> int:
    stream = probe["streams"][0]
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value not in (None, "", "N/A"):
            return int(value)
    raise AssertionError(f"ffprobe reported no frame count: {stream}")


def _duration(probe: dict) -> float:
    stream = probe["streams"][0]
    for value in (stream.get("duration"), probe.get("format", {}).get("duration")):
        if value not in (None, "", "N/A"):
            return float(value)
    raise AssertionError(f"ffprobe reported no duration: {probe}")


def _require_ffmpeg():
    if not ffmpeg_path(required=False) or not ffprobe_path(required=False):
        pytest.skip("ffmpeg/ffprobe not installed: MP4 gate cannot run here")


# ---------------------------------------------------------------------------
# camera rigs
# ---------------------------------------------------------------------------
def test_make_camera_returns_mjv_camera(dev):
    model, _ = dev
    for name in ("chase", "orbit", "fixed", "top"):
        cam = make_camera(model, name)
        assert isinstance(cam, mujoco.MjvCamera), name
        assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE, name
        assert cam.distance > 0.0
        print(f"camera {name:5s}: {cam!r}")


def test_geometry_parameters_are_honoured(dev):
    model, _ = dev
    # Horizontal radius + height become MuJoCo's slant distance and elevation.
    cam = make_camera(model, "orbit", distance=6.0, height=3.0, azimuth=45.0)
    assert cam.elevation == pytest.approx(-np.degrees(np.arctan2(3.0, 6.0)))
    assert cam.distance == pytest.approx(np.hypot(6.0, 3.0))
    assert cam.azimuth == pytest.approx(45.0)

    # An explicit elevation wins and keeps `distance` as the slant distance.
    cam = make_camera(model, "fixed", distance=9.0, elevation=-30.0)
    assert cam.elevation == pytest.approx(-30.0)
    assert cam.distance == pytest.approx(9.0)

    # Top-down rig: `height` is the altitude above the look-at point.
    top = make_camera(model, "top", height=10.0)
    assert top.elevation <= -85.0
    assert top.distance == pytest.approx(10.0, abs=1e-9)
    target = np.array([0.0, 0.0, 1.5])
    top.set_target(target)
    # The camera ends up ~10 m above the target (MuJoCo's slant distance is 10 and
    # the elevation is -89 deg), not 10 m to the side.
    assert top.distance * np.sin(np.radians(89.0)) == pytest.approx(10.0, abs=0.02)
    print(
        f"orbit elevation={cam.elevation:.2f} deg, slant distance={cam.distance:.3f} m; "
        f"top elevation={top.elevation:.1f} deg"
    )


def test_lookat_accepts_numpy_and_sets_fovy(dev):
    model, _ = dev
    target = np.array([1.0, -2.0, 3.0])
    cam = make_camera(model, "chase", lookat=target, fovy=60.0)
    np.testing.assert_allclose(cam.lookat, target)
    assert cam.fovy == pytest.approx(60.0)
    # MjvCamera has no fovy field in MuJoCo 4.0: the free camera reads this global.
    assert model.vis.global_.fovy == pytest.approx(60.0)


def test_unknown_rig_and_bad_kwargs_raise(dev):
    model, _ = dev
    with pytest.raises(ValueError):
        make_camera(model, "helicopter")
    with pytest.raises(TypeError):
        make_camera(model, "chase", distanse=4.0)  # noqa: E501 - typo on purpose


def test_chase_follows_position(dev):
    model, _ = dev
    cam = make_camera(model, "chase", smooth=False)
    target = np.array([2.0, -1.0, 4.0])
    cam.set_target(target, yaw=0.0)
    np.testing.assert_allclose(cam.lookat, target, atol=1e-12)
    print(f"chase lookat -> {np.round(cam.lookat, 3)}")


def test_chase_follows_yaw_only_not_roll_pitch(dev):
    model, _ = dev
    cam = make_camera(model, "chase", smooth=False)
    cam.set_target([0.0, 0.0, 1.5], yaw=0.0)
    assert cam.azimuth == pytest.approx(0.0, abs=1e-9)

    # A 60 deg roll and 10 deg pitch at zero heading must NOT roll the rig.
    cam.set_target([0.0, 0.0, 1.5], quat=_quat(roll=np.radians(60.0), pitch=np.radians(10.0)))
    assert cam.azimuth == pytest.approx(0.0, abs=1e-9)

    # Heading is followed: yaw=90 deg -> azimuth=90 deg (degrees vs radians).
    cam.set_target([0.0, 0.0, 1.5], quat=_quat(yaw=np.radians(90.0), roll=np.radians(60.0)))
    assert cam.azimuth == pytest.approx(90.0, abs=1e-6)
    print(f"chase azimuth after yaw=90deg + roll=60deg: {cam.azimuth:.3f} deg")


def test_chase_smoothing_lags_then_converges(dev):
    model, _ = dev
    cam = make_camera(model, "chase", tau=0.2)
    cam.set_target([0.0, 0.0, 1.5])  # first call always snaps
    np.testing.assert_allclose(cam.lookat, [0.0, 0.0, 1.5])

    cam.set_target([10.0, 0.0, 1.5], dt=0.1)
    lag_x = float(cam.lookat[0])
    expected = 10.0 * (1.0 - np.exp(-0.1 / 0.2))
    assert lag_x == pytest.approx(expected, rel=1e-9)
    assert 0.0 < lag_x < 10.0

    for _ in range(200):
        cam.set_target([10.0, 0.0, 1.5], dt=0.1)
    assert cam.lookat[0] == pytest.approx(10.0, abs=1e-6)
    print(f"chase smoothing: first step {lag_x:.3f} m of 10 m, converged to {cam.lookat[0]:.6f} m")


def test_orbit_tick_rotates_azimuth(dev):
    model, _ = dev
    cam = make_camera(model, "orbit", azimuth=10.0, azimuth_rate=90.0)
    cam.tick(1.0)
    assert cam.azimuth == pytest.approx(100.0)
    cam.tick(3.0)
    # Azimuth is wrapped into [0, 360): 100 + 270 deg is reported as 10 deg.
    assert cam.azimuth == pytest.approx((100.0 + 270.0) % 360.0)
    cam.tick(1.0)  # wraps through 360
    assert 0.0 <= cam.azimuth < 360.0
    with pytest.raises(ValueError):
        cam.tick(0.0)


def test_fixed_and_orbit_rigs_do_not_follow_heading(dev):
    model, _ = dev
    fixed = make_camera(model, "fixed", lookat=[0.0, 0.0, 1.5], azimuth=37.0)
    orbit = make_camera(model, "orbit", lookat=[0.0, 0.0, 1.5], azimuth=37.0)
    for rig in (fixed, orbit):
        rig.set_target([5.0, 5.0, 5.0], yaw=0.0)  # a 90 deg turn away from 37 deg
        assert rig.azimuth == pytest.approx(37.0), rig.mode
        np.testing.assert_allclose(rig.lookat, [5.0, 5.0, 5.0])  # but it may pan
    assert make_camera(model, "follow").mode == "chase"
    assert make_camera(model, "turntable").mode == "orbit"


# ---------------------------------------------------------------------------
# V7 gate: a non-uniform headless frame
# ---------------------------------------------------------------------------
def test_v7_render_320x240_is_non_uniform(renderer):
    r, data, cam = renderer
    frame = r.render(data)

    assert frame.shape == (V7_SIZE[1], V7_SIZE[0], 3), frame.shape
    assert frame.dtype == np.uint8
    assert frame.flags["C_CONTIGUOUS"]

    std = float(frame.std())
    colors = len(np.unique(frame.reshape(-1, 3), axis=0))
    mean = float(frame.mean())
    print(
        f"V7 frame {V7_SIZE[0]}x{V7_SIZE[1]}: std={std:.2f}, distinct colours={colors}, "
        f"mean luminance={mean:.1f}"
    )

    assert std > 5.0, f"frame is nearly flat (std={std:.3f} <= 5)"
    assert colors > 100, f"only {colors} distinct colours (<= 100)"
    assert frame.max() - frame.min() > 40  # real contrast, not a dim wash

    # The visual contract: the shadow pass is on, and the checker ground texture
    # plus a coloured body actually reach the pixels.
    assert r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] == 1
    assert r.shadows is True
    # Ground texture: several distinct large-area luminances (checker squares).
    quantised = (frame.astype(np.int16) // 16)
    quadrants = [
        quantised[: V7_SIZE[1] // 2, : V7_SIZE[0] // 2],
        quantised[: V7_SIZE[1] // 2, V7_SIZE[0] // 2 :],
        quantised[V7_SIZE[1] // 2 :, : V7_SIZE[0] // 2],
        quantised[V7_SIZE[1] // 2 :, V7_SIZE[0] // 2 :],
    ]
    assert len({c.mean(axis=(0, 1)).astype(int).tobytes() for c in quadrants}) > 1, (
        "all four quadrants have the same colour: the ground texture is missing"
    )

    # `update_scene()` + `render()` (the two-call form) returns the same picture.
    r.update_scene(data)
    again = r.render()
    np.testing.assert_array_equal(frame, again)


def test_render_returns_an_independent_copy(renderer):
    r, data, _ = renderer
    first = r.render(data)
    second = r.render(data)
    assert first is not second
    np.testing.assert_array_equal(first, second)


def test_renderer_camera_and_fovy_plumbing(dev):
    model, data = dev
    with HeadlessRenderer(model, *V7_SIZE) as r:
        assert r.camera is None
        # A per-render camera overrides the default and its fovy is applied.
        cam = make_camera(model, "orbit", fovy=30.0, lookat=[0.0, 0.0, 1.0])
        model.vis.global_.fovy = 45.0
        frame = r.render(data, cam)
        assert frame.shape == (V7_SIZE[1], V7_SIZE[0], 3)
        assert model.vis.global_.fovy == pytest.approx(30.0)
        # A renderer built without a camera must still produce a real picture
        # (MuJoCo's default free camera), because loop.py may not pass one.
        assert r.render(data).std() > 5.0
        r.camera = cam
        assert r.camera is cam
        # A rig name is resolved lazily: run_flight(camera="chase") is a thing.
        r.camera = "orbit"
        assert isinstance(r.camera, Camera) and r.camera.mode == "orbit"


def test_renderer_errors_are_readable(dev):
    model, data = dev
    r = HeadlessRenderer(model, 64, 48)
    with pytest.raises(ValueError):
        r.render()  # no scene yet
    r.close()
    r.close()  # idempotent
    with pytest.raises(RuntimeError):
        r.render(data)  # closed
    with pytest.raises(ValueError):
        HeadlessRenderer(model, 0, 48)


# ---------------------------------------------------------------------------
# V7 gate: a playable MP4 with the expected frame count
# ---------------------------------------------------------------------------
def test_v7_write_mp4_has_expected_frames_and_duration(tmp_path):
    _require_ffmpeg()
    model = build_dev_model(1.5)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = make_camera(model, "chase", fovy=45)
    cam.set_target(data.qpos[:3], yaw=0.0)

    fps = 30.0
    n_frames = 30
    out = tmp_path / "v7.mp4"
    with HeadlessRenderer(model, *V7_SIZE, camera=cam) as r:

        def frames(index: int):
            """Callback form: render one frame per call, streamed straight out."""
            data.qpos[0] = 0.02 * index  # visible motion, no physics needed
            mujoco.mj_forward(model, data)
            return r.render(data)

        path = write_mp4(frames, out, fps=fps, max_frames=n_frames)

    assert path == str(out.resolve())
    assert Path(path).is_file() and Path(path).stat().st_size > 0

    probe = _probe(path)
    stream = probe["streams"][0]
    nb_frames = _frame_count(probe)
    duration = _duration(probe)
    print(
        f"V7 mp4: {nb_frames} frames, {duration:.3f} s, {stream['width']}x{stream['height']}, "
        f"codec={stream.get('codec_name')}, pix_fmt={stream.get('pix_fmt')}, "
        f"rate={stream.get('r_frame_rate')}, {Path(path).stat().st_size} bytes"
    )

    assert nb_frames == n_frames, f"expected {n_frames} frames, ffprobe read {nb_frames}"
    # 30 frames at 30 fps -> 1 s.  Container rounding is tolerated, not ignored.
    assert duration == pytest.approx(n_frames / fps, abs=0.1)
    assert (stream["width"], stream["height"]) == V7_SIZE
    if stream.get("codec_name"):
        assert stream["codec_name"] == "h264"
    if stream.get("pix_fmt"):
        assert stream["pix_fmt"] == "yuv420p"


def test_write_mp4_accepts_iterable_and_pads_odd_sizes(tmp_path):
    _require_ffmpeg()
    # 161x121 is odd on both axes: yuv420p needs even dimensions.
    frames = [
        np.full((121, 161, 3), fill, dtype=np.uint8)
        for fill in (0, 60, 120, 180, 240)
    ]
    out = tmp_path / "odd.mp4"
    path = write_mp4(frames, out, fps=10, preset="ultrafast")
    probe = _probe(path)
    stream = probe["streams"][0]
    assert (stream["width"], stream["height"]) == (162, 122)
    assert _frame_count(probe) == 5
    print(f"odd-size mp4 padded to {stream['width']}x{stream['height']}, 5 frames")


def test_write_mp4_rejects_bad_input(tmp_path):
    _require_ffmpeg()
    with pytest.raises(ValueError):
        write_mp4([], tmp_path / "empty.mp4")
    with pytest.raises(ValueError):
        write_mp4([np.zeros((8, 8, 3), np.uint8)], tmp_path / "badfps.mp4", fps=0)
    with pytest.raises(ValueError):
        write_mp4([np.zeros((8, 8, 2), np.uint8)], tmp_path / "badshape.mp4")
    # A callable that stops immediately also counts as "no frames".
    with pytest.raises(ValueError):
        write_mp4(lambda _i: None, tmp_path / "empty2.mp4")


def test_write_png_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    path = write_png(frame, tmp_path / "frame.png")
    raw = Path(path).read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", raw[16:24])
    assert (width, height) == (160, 120)
    assert Path(path).stat().st_size > 0
    print(f"png {width}x{height}, {Path(path).stat().st_size} bytes")
