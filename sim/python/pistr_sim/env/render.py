"""Off-screen rendering and video encoding for the MuJoCo 3-D world (M3).

Everything here is headless by construction: :class:`HeadlessRenderer` wraps
``mujoco.Renderer`` (CGL on macOS, EGL/OSMesa on Linux) and :func:`write_mp4` pipes
raw frames into the system ``ffmpeg`` through stdin.  No new Python dependencies,
no ``DISPLAY``, no GUI.

Typical use from ``env/loop.py``::

    cam = make_camera(model, "chase", fovy=45)
    renderer = HeadlessRenderer(model, 640, 480, camera=cam)

    def frames(i):                       # one rendered frame per call
        cam.set_target(data.qpos[:3], quat=data.qpos[3:7], dt=dt)
        return renderer.render(data) if i < n_frames else None

    write_mp4(frames, "results/env/hover.mp4", fps=30)   # streamed, O(1) memory
    frame = renderer.render(data)                        # uint8 (H, W, 3)
    write_png(frame, "results/env/frame.png")
    renderer.close()

Both :meth:`HeadlessRenderer.render` and :func:`write_mp4` are streaming: memory
use is O(one frame), never O(number of frames).
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

import mujoco

__all__ = [
    "HeadlessRenderer",
    "write_mp4",
    "write_png",
    "ffmpeg_path",
    "ffprobe_path",
    "DEFAULT_CODEC",
]

#: H.264 in an MP4 container with a web-friendly pixel format.  Playable everywhere.
DEFAULT_CODEC = "libx264"

_FFMPEG_CANDIDATES = (
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
)
_FFPROBE_CANDIDATES = (
    "/opt/homebrew/bin/ffprobe",
    "/usr/local/bin/ffprobe",
    "/usr/bin/ffprobe",
)


def _find_tool(name: str, candidates: tuple[str, ...], *, required: bool) -> str | None:
    env = os.environ.get(name.upper())
    found: list[str] = []
    if env:
        found.append(env)
    which = shutil.which(name)
    if which:
        found.append(which)
    found.extend(candidates)
    for path in found:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    if required:
        raise RuntimeError(
            f"{name!r} was not found. Install it (macOS: `brew install ffmpeg`, "
            f"Debian: `apt install ffmpeg`) or set the {name.upper()} environment "
            f"variable to its absolute path."
        )
    return None


def ffmpeg_path(*, required: bool = True) -> str | None:
    """Absolute path of the ``ffmpeg`` binary (``FFMPEG`` overrides detection)."""
    return _find_tool("ffmpeg", _FFMPEG_CANDIDATES, required=required)


def ffprobe_path(*, required: bool = True) -> str | None:
    """Absolute path of the ``ffprobe`` binary (``FFPROBE`` overrides detection)."""
    return _find_tool("ffprobe", _FFPROBE_CANDIDATES, required=required)


# ---------------------------------------------------------------------------
# frame helpers
# ---------------------------------------------------------------------------
def _as_frame(frame, *, what: str = "frame") -> np.ndarray:
    """Normalise an image-like array to contiguous ``uint8`` ``(H, W, 3)``."""
    a = np.asarray(frame)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"{what} must be (H, W, 3) or (H, W, 4), got {a.shape}")
    if a.shape[2] == 4:
        a = a[:, :, :3]
    if a.dtype != np.uint8:
        a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=255.0, neginf=0.0)
        peak = float(a.max()) if a.size else 0.0
        if peak <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0.0, 255.0).astype(np.uint8)
    if a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"{what} must be at least 1x1, got {a.shape}")
    return np.ascontiguousarray(a)


def _pad_even(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int], bool]:
    """Pad width/height to even numbers, which 4:2:0 chroma requires.

    Returns ``(frame, (width, height), padded)``; padding replicates the last row
    and column, so it is invisible in practice.
    """
    h, w, _ = frame.shape
    dh, dw = h % 2, w % 2
    if not dh and not dw:
        return frame, (w, h), False
    padded = np.pad(frame, ((0, dh), (0, dw), (0, 0)), mode="edge")
    return np.ascontiguousarray(padded), (padded.shape[1], padded.shape[0]), True


def write_png(frame, path) -> str:
    """Write one ``uint8`` frame to ``path`` as a PNG, stdlib only.

    Uses ``zlib`` directly instead of PIL/OpenCV because this repository installs
    neither; the output is a plain 8-bit RGB PNG that any viewer/ffmpeg reads.
    """
    a = _as_frame(frame)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    return os.path.abspath(out)


def _iter_frames(source, max_frames: int | None):
    """Yield frames from an iterable or a ``callable(i) -> frame | None`` source."""
    if callable(source):
        index = 0
        while max_frames is None or index < max_frames:
            frame = source(index)
            if frame is None:
                return
            yield frame
            index += 1
        return
    for index, frame in enumerate(source):
        if max_frames is not None and index >= max_frames:
            return
        yield frame


# ---------------------------------------------------------------------------
# MP4 writer
# ---------------------------------------------------------------------------
def write_mp4(
    source,
    path,
    fps: float = 30.0,
    *,
    codec: str = DEFAULT_CODEC,
    crf: int = 23,
    preset: str = "medium",
    pix_fmt: str = "yuv420p",
    max_frames: int | None = None,
    extra_args: tuple[str, ...] | list[str] = (),
    ffmpeg: str | None = None,
    loglevel: str = "error",
) -> str:
    """Encode frames to an H.264 MP4 by streaming them into ``ffmpeg``.

    Parameters
    ----------
    source :
        Either an iterable of frames, or a callable ``source(i) -> frame | None``
        that is called once per frame (returning ``None`` stops early).  The
        callable form is how ``env/loop.py`` renders while it simulates, so the
        whole video never has to exist in memory.
    path :
        Output file.  Missing parent directories are created; an existing file is
        overwritten.  ``.mp4`` is appended when the path has no suffix.
    fps :
        Frame rate, frames per second (float allowed; written as ``-framerate``).
    codec :
        Video codec name, default ``libx264``.  When the codec is x264, ``crf`` and
        ``preset`` are passed as well; other encoders (e.g. ``h264_videotoolbox``)
        get only the generic flags.
    crf, preset :
        x264 quality/speed knobs (``crf=23`` is visually lossless enough for
        evidence video and small).
    pix_fmt :
        Output pixel format, default ``yuv420p`` for player compatibility.
    max_frames :
        Stop after this many frames; ``None`` means "until the source ends".  With a
        callable source, set this (or return ``None`` from the callback) or the
        encoder will keep asking for frames forever.
    extra_args :
        Extra ffmpeg output arguments, appended before the file name.
    ffmpeg :
        Override the ffmpeg binary (default: detected / ``$FFMPEG``).

    Returns
    -------
    str
        Absolute path of the written file.

    Notes
    -----
    Frames are written to ``ffmpeg``'s stdin with ``-movflags +faststart`` so the
    MP4 is streamable; odd frame sizes are padded by one row/column to satisfy
    4:2:0 chroma subsampling.  A non-zero ffmpeg exit raises ``RuntimeError`` with
    the captured stderr tail, and a zero-byte result is treated as an error.
    """
    if not callable(source) and isinstance(source, np.ndarray):
        # A single 3-D array is one frame, not an iterable of rows.  More useful
        # than silently encoding each row as a 1-pixel-tall image.
        source = [source]
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames!r}")

    out = Path(path)
    if not out.suffix:
        out = out.with_suffix(".mp4")
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    exe = ffmpeg or ffmpeg_path()

    frames = _iter_frames(source, None if max_frames is None else int(max_frames))
    try:
        first = next(frames)
    except StopIteration:
        raise ValueError("write_mp4() got no frames to encode") from None
    first, (width, height), _ = _pad_even(_as_frame(first, what="first frame"))

    cmd = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        loglevel,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        codec,
    ]
    if codec.startswith("libx264"):
        cmd += ["-preset", str(preset), "-crf", str(int(crf))]
    cmd += ["-pix_fmt", pix_fmt, "-movflags", "+faststart"]
    cmd += [str(a) for a in extra_args]
    cmd += [str(out)]

    stderr = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
    written = 0
    try:
        assert proc.stdin is not None
        for frame in _chain_first(first, frames):
            if frame.shape[0] != height or frame.shape[1] != width:
                raise ValueError(
                    f"frame {written} has (even-padded) size {frame.shape[1]}x"
                    f"{frame.shape[0]}, expected {width}x{height}: every frame must "
                    "have the first frame's size"
                )
            proc.stdin.write(frame.tobytes())
            written += 1
        proc.stdin.close()
        returncode = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        stderr.seek(0)
        detail = stderr.read().decode("utf-8", "replace").strip()
        stderr.close()
        # Preserve KeyboardInterrupt/SystemExit semantics and caller errors
        # (ValueError: bad frame shape/size); only wrap real encoder failures.
        exc = sys.exc_info()[1]
        if not isinstance(exc, Exception) or isinstance(exc, ValueError):
            raise
        raise RuntimeError(
            f"ffmpeg failed while encoding {out} after {written} frames"
            + (f":\n{detail}" if detail else "")
        ) from None

    stderr.seek(0)
    detail = stderr.read().decode("utf-8", "replace").strip()
    stderr.close()
    if returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited with status {returncode} while writing {out} "
            f"({written} frames)" + (f":\n{detail}" if detail else "")
        )
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg reported success but {out} is missing or empty")
    return os.path.abspath(out)


def _chain_first(first: np.ndarray, rest):
    yield first
    for frame in rest:
        yield _pad_even(_as_frame(frame))[0]


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------
class HeadlessRenderer:
    """Render MuJoCo scenes to ``uint8 (H, W, 3)`` frames without a display.

    Parameters
    ----------
    model : mujoco.MjModel
    width, height : int
        Framebuffer size in pixels (default 640x480; the contract's size).
    camera : mujoco.MjvCamera | str | None
        Default camera, typically from :func:`pistr_sim.env.camera.make_camera`.
        A rig name (``"chase"``, ``"orbit"``, ``"fixed"``, ``"top"``) is resolved with
        ``make_camera`` on the spot, which keeps ``run_flight(camera="chase")``
        ergonomic.  ``None`` uses MuJoCo's built-in free camera.
    max_geom : int
        Upper bound on geoms per scene buffer (MuJoCo default 20000).
    shadows : bool
        Enable the shadow-map pass (default on; shadows are what makes the
        vehicle legible against the ground).
    shadow_size : int | None
        Raise ``model.vis.quality.shadowsize`` to at least this many texels
        (default 2048).  This is a model-wide setting; pass ``None`` to leave the
        model untouched.
    reflections : bool
        Floor reflection pass (default **off**).  MuJoCo enables it by default, but
        with a reflective ground material it paints a bright, hard-edged patch under
        the vehicle that reads as a rendering bug on video; a matte floor keeps the
        shadow as the only ground cue.
    headlight : tuple[float, float] | None
        MuJoCo's always-on fill light as ``(ambient, diffuse)``, default
        ``(0.35, 0.55)``.  A rasterised scene with only a few point lights is very
        dark (the dev fixture renders at mean luminance 80/255 untouched), so the
        fill is raised from the stock ``(0.1, 0.4)`` to make the body and its rotor
        colours legible.  Pass ``None`` to keep the model's own values.

    Notes
    -----
    Construction touches the GL context, so it is the only part that can fail on a
    machine without one; failures are re-raised as ``RuntimeError`` with a hint
    about ``MUJOCO_GL=egl``/``osmesa`` on Linux.  The class is a context manager and
    :meth:`close` is idempotent, so ``loop.py`` can use ``with`` even when an
    exception aborts a run.
    """

    def __init__(
        self,
        model,
        width: int = 640,
        height: int = 480,
        camera=None,
        *,
        max_geom: int = 20000,
        shadows: bool = True,
        shadow_size: int | None = 2048,
        reflections: bool = False,
        headlight: tuple[float, float] | None = (0.35, 0.55),
    ) -> None:
        width, height = int(width), int(height)
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive, got {width}x{height}")
        self.model = model
        self.width = width
        self.height = height
        self.max_geom = int(max_geom)
        self.shadows = bool(shadows)
        self.reflections = bool(reflections)
        self.headlight = headlight
        self._camera = None
        self.camera = camera
        self._data = None
        self._closed = False

        if shadow_size:
            self.model.vis.quality.shadowsize = max(
                int(self.model.vis.quality.shadowsize), int(shadow_size)
            )
        if headlight is not None:
            ambient, diffuse = (float(v) for v in headlight)
            self.model.vis.headlight.ambient[:] = ambient
            self.model.vis.headlight.diffuse[:] = diffuse

        try:
            self._renderer = mujoco.Renderer(
                model, height=height, width=width, max_geom=self.max_geom
            )
        except Exception as exc:  # GL/EGL/OSMesa/driver failures land here
            raise RuntimeError(
                "MuJoCo off-screen rendering could not start a GL context "
                f"({type(exc).__name__}: {exc}). This needs no DISPLAY: on macOS it "
                "uses CGL, on Linux set MUJOCO_GL=egl (or MUJOCO_GL=osmesa for a "
                "software fallback) and make sure the matching GL libraries exist. "
                "Model construction and physics do not need this."
            ) from exc
        self._apply_scene_flags()

    # -- camera plumbing -----------------------------------------------------
    @property
    def scene(self):
        """The ``MjvScene`` being drawn (flags, lights, background).

        Exposed so callers can toggle render passes (wireframe, segmentation,
        reflections) without reaching for the private ``_renderer``.
        """
        if self._closed or self._renderer is None:
            raise RuntimeError("HeadlessRenderer is closed")
        return self._renderer.scene

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, cam) -> None:
        if isinstance(cam, str):
            from .camera import make_camera

            cam = make_camera(self.model, cam)
        self._camera = cam

    def _camera_for(self, camera):
        cam = self._camera if camera is None else camera
        if cam is not None and self._closed:
            raise RuntimeError("HeadlessRenderer is closed")
        return cam

    def _apply_fovy(self, cam) -> None:
        """Honour a rig's ``fovy`` (MjvCamera has no fovy field in MuJoCo 4.0)."""
        fovy = getattr(cam, "fovy", None) if cam is not None else None
        if fovy:
            self.model.vis.global_.fovy = float(fovy)

    def _apply_scene_flags(self) -> None:
        scene = self._renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if self.shadows else 0
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1 if self.reflections else 0
        # Skybox geometry/material stays on and haze stays off: these are the
        # MuJoCo defaults, set explicitly so a future default change cannot make
        # the video flat.
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0

    # -- rendering -----------------------------------------------------------
    def update_scene(self, data, camera=None) -> None:
        """Point the renderer at ``data``; pair with :meth:`render`."""
        if self._closed:
            raise RuntimeError("HeadlessRenderer is closed")
        cam = self._camera_for(camera)
        self._apply_fovy(cam)
        if cam is None:
            self._renderer.update_scene(data)
        else:
            self._renderer.update_scene(data, camera=cam)
        self._data = data

    def render(self, data=None, camera=None) -> np.ndarray:
        """Return one frame as ``uint8`` ``(height, width, 3)``.

        ``render(data)`` is the one-argument form used by ``loop.py``: it updates
        the scene from ``data`` and immediately renders it.  ``render()`` re-renders
        the state handed to the last :meth:`update_scene`.
        """
        if self._closed:
            raise RuntimeError("HeadlessRenderer is closed")
        if data is not None:
            self.update_scene(data, camera)
        elif camera is not None:
            if self._data is None:
                raise ValueError(
                    "render(camera=...) needs a scene; pass a mujoco.MjData "
                    "(render(data)) or call update_scene(data) first"
                )
            self.update_scene(self._data, camera)
        elif self._data is None:
            raise ValueError(
                "render() has no scene yet; pass a mujoco.MjData (render(data)) or "
                "call update_scene(data) first"
            )
        frame = self._renderer.render()
        # Copy: MuJoCo owns the buffer and reuses it for the next frame, so handing
        # it out by reference would corrupt every previously captured frame.
        return np.array(frame, dtype=np.uint8, copy=True)

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        """Release the GL context.  Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._data = None
        renderer = getattr(self, "_renderer", None)
        self._renderer = None
        if renderer is not None:
            try:
                renderer.close()
            except Exception:  # pragma: no cover - teardown must not mask errors
                pass

    def __enter__(self) -> "HeadlessRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort teardown
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"HeadlessRenderer({self.width}x{self.height}, shadows={self.shadows}, "
            f"camera={self._camera!r}, closed={self._closed})"
        )
