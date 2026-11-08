"""One-command simulation entry point.

Examples
--------
::

    python3 -m pistr_sim.cli --help
    python3 -m pistr_sim.cli --list
    python3 -m pistr_sim.cli run --scenario hover   --controller pi_str --preset matlab --out results/hover
    python3 -m pistr_sim.cli run --scenario step    --controller str    --order 3   --out results/step_mode1
    python3 -m pistr_sim.cli run --scenario sine    --controller pid                --out results/sine_pid
    python3 -m pistr_sim.cli run --scenario trajectory --out results/trajectory

The default engine is the plant-free ARX bench
(:mod:`pistr_sim.control.surrogate`), which verifies the algorithms of
``CONTROL_SPEC.md`` §3-§7 deterministically.  ``--fidelity`` is recorded in the run
metadata and selects the parameter set; the high-fidelity physics plant is
``pistr_sim.plant`` and is exercised by the experiment matrix, not here.

Outputs, in ``--out``:

* ``trace.csv``     - every controller tick (references, outputs, commands, theta)
* ``metrics.csv``   - per-channel RMSE / overshoot / settling time / control RMS
* ``summary.json``  - run metadata and the metrics in machine-readable form
* ``tracking.png``, ``rates_control.png``, ``parameters.png`` and, for the step
  scenario, ``step_response.png``; ``trajectory3d.png`` for ``trajectory``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import __version__
from .control import STRConfig, make_controller, simulate, simulate_physics, true_theta
from .metrics import (
    control_rms,
    max_attitude_error,
    rmse,
    step_metrics,
)
from .params import default_params
from .reference import (
    REFERENCE_NAMES,
    AttitudePlanner3D,
    HoverReference,
    SinusoidReference,
    StepReference,
    make_reference,
)

AXIS_NAMES = ("roll", "pitch", "yaw", "altitude")
AXIS_UNITS = ("deg", "deg", "deg", "m")
SCENARIO_NAMES = ("hover", "step", "sine", "trajectory")
CONTROLLER_CHOICES = ("pid", "str", "pi_str", "cascade_pid")

#: Deliberate presets.  ``firmware`` reproduces the golden-vector settings of
#: ``CONTROL_SPEC.md`` §8 (Pmax=5, lambda=0.99, percent=0.75); ``matlab`` is the
#: trajectory-study configuration (Pmax=100, lambda=0.998); ``exact`` starts from
#: the true model (sanity oracle, not an adaptive run).
PRESETS: dict[str, dict] = {
    "firmware": {"percent": None, "p_max": None, "forgetting": None, "outer_form": "firmware"},
    "matlab": {"percent": 0.9, "p_max": 100.0, "forgetting": 0.998, "outer_form": "paper"},
    "exact": {"percent": 1.0, "p_max": 100.0, "forgetting": 0.998, "outer_form": "paper"},
    # Tuned for the physics plant: realistic sensors (1-sample delay, noise, gyro
    # random walk) plus the measured plant/analytic channel gain ratio of 1.17-1.37
    # make the 10 % deliberate mismatch of "matlab" marginal for the 5 % overshoot
    # target.  5 % mismatch keeps adaptation active and meets the target.
    "physics": {"percent": 0.95, "p_max": 100.0, "forgetting": 0.998, "outer_form": "paper",
                "rate_max": 2.0},
}

SCENARIO_DEFAULTS = {
    "hover": {"t_end": 10.0, "altitude": 5.0},
    "step": {"t_end": 8.0, "amplitude": 0.1, "t_step": 1.5, "altitude": 5.0},
    "sine": {"t_end": 12.0, "altitude": 5.0},
    "trajectory": {"t_end": 30.0},
}


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m pistr_sim.cli",
        description="PI-STR quadrotor simulation: controllers, metrics and figures.",
        epilog="Run '--list' to see scenarios, controllers and presets.",
    )
    parser.add_argument("--version", action="version", version=f"pistr-sim {__version__}")
    parser.add_argument("--list", action="store_true", help="list scenarios/controllers/presets and exit")

    sub = parser.add_subparsers(dest="command", metavar="run")
    run = sub.add_parser("run", help="run one scenario and write CSV + PNG output")
    run.add_argument("--scenario", choices=SCENARIO_NAMES, default="hover")
    run.add_argument("--controller", choices=CONTROLLER_CHOICES, default="pi_str")
    run.add_argument("--preset", choices=tuple(PRESETS), default=None,
                     help="controller tuning preset (default: 'physics' with --plant physics, 'matlab' with surrogate)")
    run.add_argument("--plant", choices=("physics", "surrogate"), default="physics",
                     help="closed-loop engine: 'physics' = pistr_sim.plant (default), "
                          "'surrogate' = plant-free ARX bench")
    run.add_argument("--fidelity", choices=("paper", "high"), default="high",
                     help="plant fidelity for --plant physics; recorded only for the surrogate")
    run.add_argument("--disable-extras", default=None,
                     help="comma-separated plant extras to switch off "
                          "(inflow,h_force,body_drag,gyro_moment,battery_sag,esc_lag,turbulence,sensors,ground_effect)")
    run.add_argument("--use-accel", action="store_true",
                     help="physics only: enable the accelerometer tilt correction (off by default, see plant_loop)")
    run.add_argument("--order", type=int, choices=(2, 3), default=2,
                     help="inner ARX order; 'str' (Mode I) always uses 3")
    run.add_argument("--t-end", type=float, default=None, help="simulation length [s]")
    run.add_argument("--h", type=float, default=None, help="controller period [s] (default: params)")
    run.add_argument("--amplitude", type=float, default=None, help="step amplitude [rad]")
    run.add_argument("--t-step", type=float, default=None, help="step time [s]")
    run.add_argument("--altitude", type=float, default=None, help="hover altitude [m]")
    run.add_argument("--percent", type=float, default=None,
                     help="override the initial-model mismatch scale (preset percent)")
    run.add_argument("--u-hw", type=float, default=None,
                     help="hardware differential limit; below u_max exercises u_actual feedback")
    run.add_argument("--out", type=Path, default=Path("results/run"))
    run.add_argument("--no-plots", action="store_true", help="skip PNG generation")
    run.add_argument("--quiet", action="store_true")

    ident = sub.add_parser("ident", help="open-loop RLS identification on a PRBS-excited ARX channel")
    ident.add_argument("--channel", type=int, choices=(0, 1, 2, 3), default=0)
    ident.add_argument("--order", type=int, choices=(2, 3), default=2)
    ident.add_argument("--preset", choices=tuple(PRESETS), default="firmware")
    ident.add_argument("--percent", type=float, default=None)
    ident.add_argument("--samples", type=int, default=4000)
    ident.add_argument("--amplitude", type=float, default=30.0, help="PRBS differential amplitude")
    ident.add_argument("--out", type=Path, default=Path("results/identification"))
    ident.add_argument("--quiet", action="store_true")
    return parser


def print_catalogue() -> None:
    print("engines   : physics (default, pistr_sim.plant), surrogate (ARX bench)")
    print("scenarios : " + ", ".join(SCENARIO_NAMES))
    print("controllers: " + ", ".join(CONTROLLER_CHOICES))
    print("presets   : " + ", ".join(PRESETS))
    print()
    print("scenario details")
    print("  hover       hold zero attitude and a constant altitude (paper type_reference=1)")
    print("  step        roll step at t_step, used for overshoot/settling metrics (spec §8 shape)")
    print("  sine        sinusoidal roll/pitch/yaw and altitude (paper type_reference=2)")
    print("  trajectory  paper 3-D trajectory + attitude planner + ESO (reference only)")
    print()
    print("note: controller 'str' is Mode I (single loop, 3rd order). The paper's Mode-I")
    print("      channel is a marginal double integrator whose b coefficients are ~1e-4 in")
    print("      SI units, so it is very sensitive to the deliberate initial-model mismatch;")
    print("      use '--preset exact --percent 1.0' for a stable design-verification run.")
    print()
    print("preset details")
    print("  firmware    Pmax=5, lambda=0.99, percent=0.75, outer Kp=1.5/Ki=0.01 (golden §8)")
    print("  matlab      Pmax=100, lambda=0.998, percent=0.9, outer Kp=1.0/Ki=0.02 (paper study)")
    print("  exact       percent=1.0 with the matlab settings (oracle, not adaptive)")
    print("  physics     percent=0.95, rate limit 2 rad/s (tuned for the noisy physics plant)")
    print()
    print("subcommands: run, ident")
    print("  ident       open-loop PRBS identification (RLS convergence evidence, spec §4)")


# ---------------------------------------------------------------------------
# Controller construction
# ---------------------------------------------------------------------------
def build_controller(args, params):
    preset = PRESETS[args.preset]
    order = 3 if args.controller == "str" else int(args.order)
    if args.controller not in ("str", "pi_str"):
        return make_controller(args.controller, params)

    percent = preset["percent"] if args.percent is None else float(args.percent)
    overrides = {}
    if preset["p_max"] is not None:
        overrides["p_max"] = preset["p_max"]
    if preset["forgetting"] is not None:
        overrides["forgetting"] = preset["forgetting"]
    factory = STRConfig.mode_ii if order == 2 else STRConfig.mode_i
    configs = [
        factory(params, channel, percent=percent, **overrides)
        for channel in range(4)
    ]
    return make_controller(
        args.controller,
        params,
        order=order,
        outer_form=preset["outer_form"],
        str_configs=configs,
        rate_max=preset.get("rate_max"),
    )


# ---------------------------------------------------------------------------
# Reference selection
# ---------------------------------------------------------------------------
def build_reference(args):
    defaults = SCENARIO_DEFAULTS[args.scenario]
    altitude = defaults.get("altitude", 0.0) if args.altitude is None else args.altitude
    if args.scenario == "hover":
        return HoverReference(altitude=altitude)
    if args.scenario == "step":
        amplitude = defaults["amplitude"] if args.amplitude is None else args.amplitude
        t_step = defaults["t_step"] if args.t_step is None else args.t_step
        return StepReference(amplitude=amplitude, t_step=t_step, altitude=altitude)
    if args.scenario == "sine":
        ref = SinusoidReference()
        if args.altitude is not None:
            ref.altitude_offset = float(args.altitude)
        return ref
    if args.scenario == "trajectory":
        return make_reference("trajectory")
    raise ValueError(f"unknown scenario {args.scenario!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(trace, scenario: str, t_step: float | None) -> list[dict]:
    rows = []
    t = trace.t
    for j in range(trace.n_channels):
        y_ref = trace.y_ref[:, j]
        y = trace.y[:, j]
        angle_scale = 1.0 if j == 3 else 180.0 / np.pi
        row = {
            "channel": AXIS_NAMES[j],
            "unit": AXIS_UNITS[j],
            "rmse": rmse(t, y, y_ref),
            "control_rms": control_rms(trace.u_actual[:, j], t),
            "max_abs_error": max_attitude_error(
                y, y_ref, t, degrees=(j != 3)
            ),
            "overshoot_pct": float("nan"),
            "settling_time_s": float("nan"),
            "steady_state_error": float("nan"),
        }
        if scenario == "step" and j == 0 and t_step is not None:
            m = step_metrics(t, y, y_ref, t_step=t_step)
            row.update(
                {
                    "overshoot_pct": m.overshoot,
                    "settling_time_s": m.settling_time,
                    "steady_state_error": m.steady_state_error,
                }
            )
        row["diverged"] = bool(trace.diverged) or bool(np.any(~np.isfinite(y)))
        row["angle_scale"] = angle_scale
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_trace_csv(path: Path, trace) -> None:
    n = trace.t.size
    header = ["t"]
    for prefix in ("y_ref", "y", "rate_ref", "rate", "u", "u_actual"):
        header += [f"{prefix}_{name}" for name in AXIS_NAMES]
    theta = trace.theta_hat
    if theta is not None and theta.shape[0] == n and theta.shape[2] > 0:
        header += [f"theta_hat_{AXIS_NAMES[j]}_{i}" for j in range(4) for i in range(theta.shape[2])]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for k in range(n):
            row = [f"{trace.t[k]:.9g}"]
            for block in (trace.y_ref, trace.y, trace.rate_ref, trace.rate, trace.u, trace.u_actual):
                row += [f"{block[k, j]:.9g}" for j in range(4)]
            if theta is not None and theta.shape[0] == n and theta.shape[2] > 0:
                row += [f"{theta[k, j, i]:.9g}" for j in range(4) for i in range(theta.shape[2])]
            writer.writerow(row)


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "channel",
        "unit",
        "rmse",
        "overshoot_pct",
        "settling_time_s",
        "steady_state_error",
        "control_rms",
        "max_abs_error",
        "diverged",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="pistr-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_tracking(trace, out_dir: Path, scenario: str) -> Path:
    plt = _pyplot()
    t = trace.t
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        scale = 1.0 if j == 3 else 180.0 / np.pi
        ax.plot(t, trace.y_ref[:, j] * scale, "k--", lw=1.2, label="reference")
        ax.plot(t, trace.y[:, j] * scale, "C0", lw=1.4, label="output")
        ax.set_title(AXIS_NAMES[j])
        ax.set_ylabel(f"{AXIS_NAMES[j]} [{AXIS_UNITS[j]}]")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.suptitle(f"tracking - {scenario} ({trace.metadata.get('order', '?')}th-order ARX bench)")
    fig.tight_layout()
    path = out_dir / "tracking.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_rates_control(trace, out_dir: Path) -> Path:
    plt = _pyplot()
    t = trace.t
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j in range(3):
        ax = axes.ravel()[j]
        ax.plot(t, np.rad2deg(trace.rate_ref[:, j]), "k--", lw=1.0, label="rate reference")
        ax.plot(t, np.rad2deg(trace.rate[:, j]), "C1", lw=1.2, label="rate")
        ax.set_title(f"{AXIS_NAMES[j]} rate")
        ax.set_ylabel("deg/s")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=8)
    ax = axes.ravel()[3]
    for j in range(4):
        ax.plot(t, trace.u[:, j], lw=0.9, alpha=0.8, label=f"u {AXIS_NAMES[j]}")
    ax.plot(t, trace.u_actual[:, 0], "k.", ms=1.0, alpha=0.4, label="u_actual roll")
    ax.set_title("differential commands")
    ax.set_ylabel("PWM differential")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.suptitle("inner-loop rates and control effort")
    fig.tight_layout()
    path = out_dir / "rates_control.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_parameters(trace, out_dir: Path) -> Path | None:
    theta = trace.theta_hat
    if theta is None or theta.shape[2] == 0:
        return None
    plt = _pyplot()
    t = trace.t
    order = trace.order
    names = ["a1", "a2", "b0", "b1"] if order == 2 else ["a1", "a2", "a3", "b0", "b1", "b2"]
    true = np.asarray(trace.metadata["theta_true"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        for i, name in enumerate(names):
            ax.plot(t, theta[:, j, i], lw=1.1, label=f"hat {name}")
            if np.all(np.isfinite(true[j, i])):
                ax.axhline(true[j, i], color=f"C{i}", ls="--", lw=1.0, alpha=0.8)
        ax.set_title(f"{AXIS_NAMES[j]} parameters" + (" (dashed = true)" if np.all(np.isfinite(true)) else " (no ARX truth for the physics plant)"))
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=7, ncol=3)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.suptitle("RLS parameter convergence")
    fig.tight_layout()
    path = out_dir / "parameters.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_estimator(trace, out_dir: Path) -> Path:
    """Physics runs: estimated vs true attitude/altitude (observer health)."""
    plt = _pyplot()
    t = trace.t
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        scale = 1.0 if j == 3 else 180.0 / np.pi
        ax.plot(t, trace.y[:, j] * scale, "C0", lw=1.3, label="true")
        ax.plot(t, trace.est_y[:, j] * scale, "C3--", lw=1.0, label="estimate")
        ax.set_title(AXIS_NAMES[j])
        ax.set_ylabel(f"{AXIS_NAMES[j]} [{AXIS_UNITS[j]}]")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    err = trace.estimator_error
    fig.suptitle(
        "attitude/altitude estimate vs truth (RMS error: "
        + ", ".join(
            f"{AXIS_NAMES[j]} {np.rad2deg(np.sqrt(np.nanmean(err[:, j] ** 2))) if j < 3 else np.sqrt(np.nanmean(err[:, j] ** 2)):.2f}"
            + ("deg" if j < 3 else "m")
            for j in range(4)
        )
        + ")"
    )
    fig.tight_layout()
    path = out_dir / "estimator.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_step(trace, out_dir: Path, t_step: float, tol: float = 0.02) -> Path:
    plt = _pyplot()
    t = trace.t
    y = trace.y[:, 0]
    ref = trace.y_ref[:, 0]
    m = step_metrics(t, y, ref, t_step=t_step, tol=tol)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, np.rad2deg(y), "C0", lw=1.6, label="roll")
    ax.plot(t, np.rad2deg(ref), "k--", lw=1.2, label="reference")
    if np.isfinite(m.final):
        band = tol * abs(np.rad2deg(m.ref) - 0.0)
        ax.axhspan(np.rad2deg(m.final) - band, np.rad2deg(m.final) + band, color="C2", alpha=0.12)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("roll [deg]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(
        "step response: overshoot {:+.1f} %, settling (2 %) {:s} s, RMSE {:.3g} deg, "
        "ss error {:+.2f} deg".format(
            m.overshoot,
            f"{m.settling_time:.2f}" if np.isfinite(m.settling_time) else "n/a",
            np.rad2deg(m.rmse) if np.isfinite(m.rmse) else float("nan"),
            np.rad2deg(m.steady_state_error) if np.isfinite(m.steady_state_error) else float("nan"),
        )
    )
    fig.tight_layout()
    path = out_dir / "step_response.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_trajectory(out_dir: Path, t_end: float, h: float, params) -> tuple[Path, "object"]:
    """Reference-planner-only run: 3-D path, planned attitude and ESO rates."""
    plt = _pyplot()
    planner = AttitudePlanner3D(mass=params.airframe.mass, h=h)
    n = int(np.floor(t_end / h)) + 1
    t = np.arange(n) * h
    xi = np.zeros((n, 3))
    v = np.zeros((n, 3))
    roll = np.zeros(n)
    pitch = np.zeros(n)
    roll_rate = np.zeros(n)
    pitch_rate = np.zeros(n)
    yaw = np.zeros(n)
    for k, tk in enumerate(t):
        # Reference-module-only demo: assume perfect position tracking, i.e. the
        # planner sees the nominal state, so its output is the feed-forward tilt
        # (a non-zero PD correction would only appear with a plant in the loop).
        base = planner.reference.sample(tk)
        ref = planner.sample(tk, xi_meas=base.xi, v_meas=base.v_xi)
        xi[k] = ref.xi
        v[k] = ref.v_xi
        roll[k], pitch[k] = ref.roll, ref.pitch
        roll_rate[k], pitch_rate[k] = ref.roll_rate, ref.pitch_rate
        yaw[k] = ref.yaw

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(xi[:, 0], xi[:, 1], xi[:, 2], "C0", lw=1.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("paper 3-D position reference")

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, np.rad2deg(roll), label="roll_ref")
    ax.plot(t, np.rad2deg(pitch), label="pitch_ref")
    ax.plot(t, np.rad2deg(yaw), label="yaw_ref")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("attitude planner (position PD)")

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t, np.rad2deg(roll_rate), label="roll rate (ESO)")
    ax.plot(t, np.rad2deg(pitch_rate), label="pitch rate (ESO)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("deg/s")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("ESO-differentiated attitude reference (zero for t <= 2 s)")

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, xi[:, 2], label="z_ref")
    ax.plot(t, v[:, 2], label="vz_ref")
    ax.set_xlabel("time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("altitude reference and climb rate")

    fig.suptitle("3-D trajectory reference and attitude plan (no plant; reference module only)")
    fig.tight_layout()
    path = out_dir / "trajectory3d.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, t


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run_ident(args) -> int:
    """Open-loop identification with broadband (PRBS) excitation.

    The controlled channel is fed a random switching input; there is no feedback,
    so the regressor is persistently exciting and the identified parameters can be
    compared directly with the ground truth (§4 acceptance evidence).
    """
    from .control import STR
    from .control.surrogate import true_theta

    params = default_params(fidelity="paper")
    order = int(args.order)
    preset = PRESETS[args.preset]
    percent = preset["percent"] if args.percent is None else float(args.percent)
    overrides = {}
    if preset["p_max"] is not None:
        overrides["p_max"] = preset["p_max"]
    if preset["forgetting"] is not None:
        overrides["forgetting"] = preset["forgetting"]
    factory = STRConfig.mode_ii if order == 2 else STRConfig.mode_i
    config = factory(params, args.channel, percent=percent, **overrides)

    theta_true = true_theta(params, order=order)[args.channel]
    reg = STR(config)
    rng = np.random.default_rng(20240101 + args.channel)
    n = int(args.samples)
    h = float(config.h)
    a = np.concatenate(([1.0], theta_true[:order]))
    b = np.concatenate(([0.0], theta_true[order:]))
    y_hist = [0.0] * order
    u_hist = [0.0] * order
    records = np.zeros((n, 3 + 2 * order))
    u = 0.0
    for k in range(n):
        if k % int(rng.integers(1, 5)) == 0:
            u = float(rng.choice([-1.0, 1.0]) * args.amplitude)
        y = -float(np.asarray(a[1:]) @ np.asarray(y_hist)) + float(np.asarray(b[1:]) @ np.asarray(u_hist))
        reg.step(0.0, y)
        reg.ack(u)
        y_hist.insert(0, y); del y_hist[order:]
        u_hist.insert(0, u); del u_hist[order:]
        records[k, :] = np.r_[k * h, y, u, reg.theta]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["t", "y", "u"] + [f"theta_{i}" for i in range(2 * order)]
    np.savetxt(out_dir / "ident_trace.csv", records, delimiter=",", header=",".join(header), comments="")

    theta_hat = reg.theta
    rel = (theta_hat - theta_true) / np.where(np.abs(theta_true) > 0, np.abs(theta_true), 1.0)
    summary = {
        "channel": AXIS_NAMES[args.channel],
        "order": order,
        "preset": args.preset,
        "n_samples": n,
        "h": h,
        "theta_true": theta_true.tolist(),
        "theta_hat": theta_hat.tolist(),
        "rel_error_pct": (rel * 100.0).tolist(),
        "max_abs_rel_error_pct": float(np.max(np.abs(rel)) * 100.0),
        "figure": None,
    }

    if not args.quiet:
        plt = _pyplot()
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
        names = ["a1", "a2", "b0", "b1"] if order == 2 else ["a1", "a2", "a3", "b0", "b1", "b2"]
        for i, name in enumerate(names):
            ax = axes.ravel()[i % 4]
            ax.plot(records[:, 0], records[:, 3 + i], lw=1.1)
            ax.axhline(theta_true[i], color="k", ls="--", lw=1.0)
            ax.set_title(f"{name} (dashed = true)")
            ax.grid(True, alpha=0.3)
        for ax in axes[-1, :]:
            ax.set_xlabel("time [s]")
        fig.suptitle(f"RLS identification, {AXIS_NAMES[args.channel]} channel ({order}nd-order ARX)")
        fig.tight_layout()
        path = out_dir / "convergence.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        summary["figure"] = str(path)

    (out_dir / "ident_summary.json").write_text(json.dumps(summary, indent=2))
    if not args.quiet:
        print(f"[pistr] identification channel={AXIS_NAMES[args.channel]} order={order} "
              f"samples={n} preset={args.preset}")
        print(f"{'param':>6s} {'true':>12s} {'estimate':>12s} {'rel err':>9s}")
        for i, name in enumerate(names):
            print(f"{name:>6s} {theta_true[i]:12.6g} {theta_hat[i]:12.6g} {rel[i]*100:8.2f}%")
        print(f"[pistr] max |relative error| = {summary['max_abs_rel_error_pct']:.3f}%")
        if summary["figure"]:
            print(f"[pistr] figure: {summary['figure']}")
        print(f"[pistr] output: {out_dir}")
    return 0


def run(args) -> int:
    started = time.time()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    params = default_params(fidelity=args.fidelity)
    if args.h is not None:
        from dataclasses import replace as _replace

        params = params.with_(control=_replace(params.control, h=float(args.h)))

    if args.preset is None:
        args.preset = "physics" if args.plant == "physics" else "matlab"

    defaults = SCENARIO_DEFAULTS[args.scenario]
    t_end = float(defaults["t_end"] if args.t_end is None else args.t_end)
    if args.scenario == "trajectory":
        path, _ = plot_trajectory(
            out_dir, t_end, float(args.h if args.h is not None else params.control.h), params
        ) if not args.no_plots else (None, None)
        summary = {
            "scenario": args.scenario,
            "controller": "reference_only",
            "preset": args.preset,
            "fidelity": args.fidelity,
            "t_end": t_end,
            "figure": None if path is None else str(path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if not args.quiet:
            print(f"[pistr] trajectory reference written to {out_dir}")
            if path is not None:
                print(f"[pistr] figure: {path}")
        return 0

    reference = build_reference(args)
    controller = build_controller(args, params)
    u_hw = None if args.u_hw is None else float(args.u_hw)

    if args.plant == "physics":
        extras = None
        if args.disable_extras:
            from .plant import DEFAULT_EXTRAS

            extras = {name: False for name in DEFAULT_EXTRAS}
            for name in args.disable_extras.split(","):
                name = name.strip()
                if not name:
                    continue
                if name not in extras:
                    raise ValueError(f"unknown plant extra {name!r}")
                extras[name] = False
        trace = simulate_physics(
            controller,
            reference,
            params,
            fidelity=args.fidelity,
            t_end=t_end,
            altitude0=float(reference.sample(0.0).altitude),
            extras=extras,
            use_accel=bool(args.use_accel),
        )
    else:
        # Hover/step start at the reference altitude; the surrogate has no plant
        # trim, so the initial state is given explicitly (MATLAB initial attitude
        # for hover, level for the others).
        y0 = rate0 = None
        if args.scenario == "hover":
            y0 = [0.02, -0.02, 0.05, float(reference.altitude)]
        elif args.scenario in ("step", "sine"):
            y0 = [0.0, 0.0, 0.0, float(reference.altitude)]
        trace = simulate(
            controller,
            reference,
            params,
            t_end=t_end,
            order=3 if args.controller == "str" else int(args.order),
            u_hw=u_hw,
            y0=y0,
            rate0=rate0,
        )
    rows = compute_metrics(trace, args.scenario, getattr(reference, "t_step", None))

    write_trace_csv(out_dir / "trace.csv", trace)
    write_metrics_csv(out_dir / "metrics.csv", rows)

    figures = []
    if not args.no_plots:
        figures.append(plot_tracking(trace, out_dir, args.scenario))
        figures.append(plot_rates_control(trace, out_dir))
        p = plot_parameters(trace, out_dir)
        if p is not None:
            figures.append(p)
        if getattr(trace, "est_y", None) is not None:
            figures.append(plot_estimator(trace, out_dir))
        if args.scenario == "step":
            figures.append(plot_step(trace, out_dir, float(reference.t_step)))

    theta_final = np.asarray(controller.theta, dtype=float)
    summary = {
        "scenario": args.scenario,
        "engine": "physics" if args.plant == "physics" else "surrogate",
        "controller": controller.name,
        "preset": args.preset,
        "fidelity": args.fidelity,
        "order": trace.order,
        "h": float(trace.metadata["h"]),
        "t_end": float(trace.t[-1]),
        "diverged": bool(trace.diverged),
        "safe_mode": bool(trace.safe_mode),
        "elapsed_s": round(time.time() - started, 3),
        "metrics": rows,
        "theta_true": np.asarray(trace.metadata["theta_true"]).tolist(),
        "theta_final": theta_final.tolist() if theta_final.size else [],
        "figures": [str(p) for p in figures],
        "metadata": {
            key: (val.tolist() if isinstance(val, np.ndarray) else val)
            for key, val in trace.metadata.items()
            if key != "theta_true"
        },
    }
    if getattr(trace, "est_y", None) is not None:
        err = trace.estimator_error
        summary["estimator_rmse"] = {
            AXIS_NAMES[j]: float(
                np.rad2deg(np.sqrt(np.nanmean(err[:, j] ** 2)))
                if j < 3
                else np.sqrt(np.nanmean(err[:, j] ** 2))
            )
            for j in range(4)
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if not args.quiet:
        print(f"[pistr] scenario={args.scenario} controller={controller.name} "
              f"preset={args.preset} order={trace.order} h={trace.metadata['h']:.3f}s")
        print(f"[pistr] diverged={trace.diverged} safe_mode={trace.safe_mode} "
              f"elapsed={summary['elapsed_s']:.2f}s")
        header = f"{'channel':9s} {'rmse':>10s} {'overshoot%':>11s} {'settle[s]':>10s} {'ctrl RMS':>10s}"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['channel']:9s} {row['rmse']:10.4g} {row['overshoot_pct']:11.4g} "
                f"{row['settling_time_s']:10.4g} {row['control_rms']:10.4g}"
            )
        if theta_final.size:
            true = np.asarray(trace.metadata["theta_true"], dtype=float)
            if np.all(np.isfinite(true)):
                print("[pistr] final theta_hat vs true (roll): "
                      + ", ".join(f"{v:+.4f}/{t:+.4f}" for v, t in zip(theta_final[0], true[0])))
            else:
                print("[pistr] final theta_hat (roll): "
                      + ", ".join(f"{v:+.4f}" for v in theta_final[0]))
        for p in figures:
            print(f"[pistr] figure: {p}")
        print(f"[pistr] output: {out_dir}")
    return 0 if not trace.diverged else 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        print_catalogue()
        return 0
    if args.command == "ident":
        return run_ident(args)
    if args.command != "run":
        parser.print_help()
        return 1
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        print(f"[pistr] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
