"""Command line front end for the 3-D environment (``docs/3D_ENVIRONMENT.md``).

    python3 -m pistr_sim.env.cli world --out results/env     # scene poster, no flight
    python3 -m pistr_sim.env.cli demo  --out results/env     # five flight segments
    python3 -m pistr_sim.env.cli demo  --segments hover --no-video

MuJoCo is an optional dependency (``make env-setup``). Nothing here is imported by
``pistr_sim`` itself, so the numerical simulator keeps working without MuJoCo.

Rendering owns a CGL/EGL context bound to the thread that created it, so each segment
runs its simulation *and* its renderer in one worker thread and pushes finished frames
into a bounded queue; this process drains the queue into ffmpeg. Memory stays O(queue).
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..params import default_params
from ..types import PWM_MIN

#: Segments of the default demo, in presentation order.
SEGMENTS = ("hover", "step", "gust", "drop", "multi")

#: The STR family is designed at h = 40 ms, i.e. 25 frames/s at one frame per tick.
DEFAULT_FPS = 25
CONTROL_H = 0.04


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _spec_from_args(args) -> Any:
    from .scene import SceneSpec

    return SceneSpec(
        buildings=args.buildings,
        trees=args.trees,
        obstacles=args.obstacles,
        terrain=args.terrain,
        sky=args.sky,
        wind_markers=args.wind_markers,
        seed=args.seed,
        extent=args.extent,
        clearance=args.clearance,
        ground_texture=args.ground_texture,
    )


def _scene_xml(args) -> str | None:
    """``SceneSpec`` -> worldbody fragment (M2's contract). ``--no-scene`` skips it."""
    if getattr(args, "no_scene", False):
        return None
    from .scene import scene_xml

    return scene_xml(_spec_from_args(args))


def _pose(data, model, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    """World position and MuJoCo ``(w, x, y, z)`` attitude of one free body."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    jid = int(model.body_jntadr[bid])
    adr = int(model.jnt_qposadr[jid])
    return (
        np.array(data.qpos[adr : adr + 3], dtype=float),
        np.array(data.qpos[adr + 3 : adr + 7], dtype=float),
    )



def _drone_contact_count(model, data, drone: str = "drone0") -> int:
    """Contacts that involve the vehicle itself (not buildings resting on the ground)."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, drone)
    if bid < 0:
        return int(data.ncon)
    root = int(model.body_rootid[bid])
    count = 0
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        b1 = int(model.geom_bodyid[int(contact.geom1)])
        b2 = int(model.geom_bodyid[int(contact.geom2)])
        if int(model.body_rootid[b1]) == root or int(model.body_rootid[b2]) == root:
            count += 1
    return count

def _camera(args, model, kind: str = "chase", **kw):
    from .camera import make_camera

    if kind == "orbit":
        defaults = dict(distance=8.0, height=3.0, fovy=55.0, azimuth_rate=12.0)
    else:
        defaults = dict(
            distance=getattr(args, "camera_distance", 2.6),
            height=getattr(args, "camera_height", 0.9),
            fovy=55.0,
            tau=0.10,
        )
    defaults.update(kw)
    return make_camera(model, kind, **defaults)


def _stream(callable_body: Callable[[Callable[[np.ndarray], None]], None]):
    """Run ``callable_body(put_frame)`` in a thread and yield its frames."""
    q: queue.Queue = queue.Queue(maxsize=8)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            callable_body(q.put)
        except BaseException as exc:  # re-raised in the consumer
            errors.append(exc)
        finally:
            q.put(None)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def frames() -> Iterable[np.ndarray]:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        thread.join()
        if errors:
            raise errors[0]

    return frames()


# ---------------------------------------------------------------------------
# one rendered segment
# ---------------------------------------------------------------------------
def run_segment(args, name: str, out: Path) -> dict:
    """Simulate, render and encode one segment; return its metrics."""
    from .render import HeadlessRenderer, write_mp4, write_png

    video_path = out / f"{name}.mp4" if args.video else None
    poster_path = out / f"{name}.png" if args.video else None
    metrics: dict[str, Any] = {}

    scene = _scene_xml(args)

    def body(put) -> None:
        from .mjcf import build_model

        n_drones = 3 if name == "multi" else 1
        model = build_model(default_params("high"), scene=scene, n_drones=n_drones)
        if name == "drop":
            # the landing shot needs to look down from a little further: the spawn
            # pad fills the frame at the normal chase distance
            camera = _camera(args, model, distance=4.6, height=2.6)
        else:
            camera = _camera(args, model)
        renderer = HeadlessRenderer(
            model, args.width, args.height, camera=camera,
            shadow_size=getattr(args, "shadow_size", 2048),
        )
        saved = [False]

        def emit(data) -> None:
            frame = renderer.render(data)
            if poster_path is not None and not saved[0]:
                write_png(frame, poster_path)
                saved[0] = True
            put(frame)

        try:
            if name == "drop":
                metrics.update(_drive_drop(args, model, camera, renderer, emit))
            else:
                metrics.update(_drive_flight(args, name, model, camera, renderer, emit, scene))
        finally:
            renderer.close()

    def consume(frames: Iterable[np.ndarray]) -> int:
        if video_path is None:
            return sum(1 for _ in frames)
        write_mp4(frames, video_path, fps=args.fps)
        return 0

    started = time.time()
    consume(_stream(body))
    metrics.setdefault("segment", name)
    metrics["wall_s"] = round(time.time() - started, 2)
    if video_path is not None and video_path.exists():
        metrics["video"] = str(video_path)
    return metrics


def _drive_flight(args, name, model, camera, renderer, emit, scene) -> dict:
    """Closed loop through ``run_flight`` with the chase camera following drone0."""
    from .loop import run_flight

    stride = max(1, int(round(1.0 / (args.fps * CONTROL_H))))
    dt_frame = 1.0 / args.fps

    def frame(t, data) -> None:
        if data is None:
            return
        if name == "multi":
            center = np.mean([_pose(data, model, f"drone{i}")[0] for i in range(3)], axis=0)
            camera.set_target(center, dt=dt_frame)
            camera.tick(dt_frame)
        else:
            pos, quat = _pose(data, model, "drone0")
            camera.set_target(pos, quat=quat, dt=dt_frame)
            camera.tick(dt_frame)
        emit(data)

    kwargs: dict[str, Any] = dict(
        controller=args.controller,
        scenario="hover",
        scene=scene,
        model=model,
        t_end=args.t_end,
        # camera=None: this CLI is the only camera driver (see ``frame`` below), so
        # run_flight must not also tick/target the rig -- otherwise orbit rates and
        # multi-vehicle framing would be applied twice.
        camera=None,
        on_frame=frame,
        frame_stride=stride,
    )
    if name == "step":
        kwargs.update(scenario="step", step_amplitude=0.15, step_time=1.5)
    if name == "gust":
        # the world applies the gust only in the dedicated scenario (loop.py); passing
        # gust_ned with scenario="hover" is silently a no-op, which made the segment
        # bit-identical to hover.
        kwargs.update(scenario="gust", gust_ned=(3.0, 0.0, 0.0),
                      gust_t=(1.0, max(1.2, args.t_end - 0.5)))
    if name == "multi":
        kwargs.update(n_drones=3, drone_spacing=2.2)

    trace = run_flight(**kwargs)
    summary = trace.summary()
    if name == "step":
        summary["step_rad"] = 0.15
    if name == "gust":
        summary["gust_m_s"] = 3.0
    summary["frames"] = len(trace.t)
    return summary


def _drive_drop(args, model, camera, renderer, emit) -> dict:
    """Cut the motors at 1.5 s: free fall, ground contact, rest on the ground."""
    from .bridge import MuJoCoPlant

    params = default_params("high")
    plant = MuJoCoPlant(params, model=model, dt=0.002)
    plant.reset()
    pwm_hover = np.full(4, float(plant.trim_hover()["pwm_us"]))
    physics_dt = 0.002
    every = max(1, int(round(1.0 / (args.fps * physics_dt))))
    n_frames = max(1, int(round(args.t_end * args.fps)))
    t_cut = 1.5

    max_contacts = 0
    lowest = float("inf")
    impact_speed = 0.0
    peak_speed = 0.0
    prev_speed = 0.0
    for k in range(n_frames):
        t = k / args.fps
        pwm = pwm_hover if t < t_cut else np.full(4, PWM_MIN)
        for _ in range(every):
            plant.step(pwm)
        state = plant.state
        pos, quat = _pose(plant.data, model, "drone0")
        camera.set_target(pos, quat=quat, dt=1.0 / args.fps)
        camera.tick(1.0 / args.fps)
        emit(plant.data)
        contacts = _drone_contact_count(model, plant.data)
        if contacts > 0 and max_contacts == 0:
            # speed at the last sample before the ground was touched (the frame that
            # first sees contact is already post-impact: the sample period is 40 ms)
            impact_speed = prev_speed
        max_contacts = max(max_contacts, contacts)
        speed = float(np.linalg.norm(state.v))
        peak_speed = max(peak_speed, speed)
        prev_speed = speed
        altitude = -float(state.xi[2])
        lowest = min(lowest, altitude)
    return dict(
        cut_motors_at_s=t_cut,
        rest_altitude_m=round(float(-plant.state.xi[2]), 3),
        impact_speed_m_s=round(impact_speed, 2),
        peak_fall_speed_m_s=round(peak_speed, 2),
        max_contacts=max_contacts,
        frames=n_frames,
    )


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_world(args) -> int:
    """Build the world for real and write a single poster frame (no flight)."""
    import mujoco

    from .mjcf import build_model
    from .render import HeadlessRenderer, write_png

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(default_params("high"), scene=_scene_xml(args))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pos, quat = _pose(data, model, "drone0")
    camera = _camera(args, model, distance=args.camera_distance, height=args.camera_height, tau=0.0)
    camera.set_target(pos, quat=quat)
    renderer = HeadlessRenderer(model, args.width, args.height, camera=camera)
    try:
        frame = renderer.render(data)
    finally:
        renderer.close()
    path = write_png(frame, out / "world.png")
    print(f"world: {model.nbody} bodies / {model.ngeom} geoms -> {path}")
    return 0


def cmd_demo(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = list(args.segments) if args.segments else list(SEGMENTS)

    summary: dict[str, Any] = {"fps": args.fps, "segments": {}}
    clips: list[Path] = []
    for name in wanted:
        print(f"[demo] {name} ...", flush=True)
        metrics = run_segment(args, name, out)
        summary["segments"][name] = metrics
        if metrics.get("video"):
            clips.append(Path(metrics["video"]))
        print(f"[demo]   {name}: {metrics}", flush=True)

    if args.video and len(clips) > 1:
        final = out / args.video_name
        _concat(clips, final)
        summary["video"] = str(final)
        print(f"[demo] combined -> {final}")
    elif len(clips) == 1:
        summary["video"] = str(clips[0])

    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    text = _summary_text(summary)
    (out / "demo_summary.txt").write_text(text)
    print(text)
    return 0


def _summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "3-D environment demo (MuJoCo world; the controller is the same code as sim/python)",
        f"  fps = {summary['fps']}",
    ]
    for name, m in summary.get("segments", {}).items():
        bits = []
        drones = m.get("drones")
        if isinstance(drones, list) and drones:
            if len(drones) == 1:
                d = drones[0]
                rmse = d.get("attitude_rmse_deg", {})
                alt = d.get("altitude", {})
                bits.append(
                    "attitude RMSE r/p/y = %.2f/%.2f/%.2f deg"
                    % (rmse.get("roll", 0.0), rmse.get("pitch", 0.0), rmse.get("yaw", 0.0))
                )
                if alt:
                    bits.append("alt drift %.2f m" % alt.get("drift_max", 0.0))
            else:
                bits.append("%d drones, alt drift %s m" % (
                    len(drones),
                    "/".join("%.2f" % d.get("altitude", {}).get("drift_max", 0.0) for d in drones),
                ))
        for key, label in (
            ("step_rad", "step %.2f rad"),
            ("gust_m_s", "gust %.1f m/s"),
            ("impact_speed_m_s", "impact %.1f m/s"),
            ("peak_fall_speed_m_s", "peak fall %.1f m/s"),
            ("rest_altitude_m", "rest %.2f m"),
        ):
            if key in m:
                bits.append(label % m[key])
        bits.append("%.1f s" % m.get("wall_s", 0.0) + " render")
        lines.append(f"  {name:6s} " + "  ".join(bits))
    if "video" in summary:
        lines.append(f"  video  {summary['video']}")
    lines.append(
        "  note: the world integrates the rigid body and resolves contact itself;"
        " numbers are a second opinion, not a replay of plant.py"
    )
    return "\n".join(lines)


def _concat(clips: list[Path], out: Path) -> None:
    from .render import ffmpeg_path

    ffmpeg = ffmpeg_path(required=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for clip in clips:
            fh.write(f"file '{clip.resolve()}'\n")
        list_path = fh.name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", "-movflags", "+faststart", str(out)],
            check=True,
        )
    finally:
        Path(list_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m pistr_sim.env.cli",
        description="MuJoCo 3-D environment for the PI-STR quadrotor (optional extra).",
    )
    sub = parser.add_subparsers(dest="command")

    def common(sp) -> None:
        sp.add_argument("--out", default="results/env", help="output directory")
        sp.add_argument("--width", type=int, default=1280)
        sp.add_argument("--height", type=int, default=720)
        sp.add_argument("--fps", type=int, default=DEFAULT_FPS)
        sp.add_argument("--seed", type=int, default=3)
        sp.add_argument("--terrain", default="flat", choices=("flat", "hills"))
        sp.add_argument("--buildings", type=int, default=6)
        sp.add_argument("--trees", type=int, default=10)
        sp.add_argument("--obstacles", type=int, default=6)
        sp.add_argument("--extent", type=float, default=60.0)
        sp.add_argument("--no-sky", dest="sky", action="store_false", default=True)
        sp.add_argument("--no-scene", action="store_true", help="bare floor only")
        sp.add_argument("--wind-markers", action="store_true")
        sp.add_argument("--clearance", type=float, default=10.0,
                        help="empty radius around the spawn point, m")
        sp.add_argument("--no-ground-texture", dest="ground_texture",
                        action="store_false", default=True)

    world = sub.add_parser("world", help="render one poster frame of the scene")
    common(world)
    world.add_argument("--camera-distance", type=float, default=16.0)
    world.add_argument("--camera-height", type=float, default=2.2)
    world.set_defaults(func=cmd_world)

    demo = sub.add_parser("demo", help="fly and render the demo segments")
    common(demo)
    demo.add_argument("--segments", nargs="*", choices=SEGMENTS, default=None)
    demo.add_argument("--t-end", type=float, default=5.0)
    demo.add_argument("--controller", default="pi_str")
    demo.add_argument("--no-video", dest="video", action="store_false", default=True)
    demo.add_argument("--video-name", default="flight_demo.mp4")
    demo.add_argument("--camera-distance", type=float, default=2.6, help="chase radial distance")
    demo.add_argument("--camera-height", type=float, default=0.9, help="chase height above the vehicle")
    demo.add_argument("--shadow-size", type=int, default=4096, help="shadow map resolution")
    demo.set_defaults(func=cmd_demo)

    listing = sub.add_parser("list", help="list demo segments")
    listing.set_defaults(func=lambda a: (print("\n".join(SEGMENTS)), 0)[1])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
