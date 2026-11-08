"""Experiment matrix (D phase): controllers x fidelities x faults x seeds.

Usage (from the repository root)::

    PYTHONPATH=sim/python python3 -m pistr_sim.experiments.matrix --quick
    PYTHONPATH=sim/python python3 -m pistr_sim.experiments.matrix --full

``--quick`` is the default and runs::

    fidelity {paper, high} x controllers {6} x faults {none, wind_gust} x seed {0}
    + a high-fidelity 3-D trajectory suite for all six controllers

``--full`` expands faults to all five and seeds to three, and also runs the
paper-fidelity trajectory suite.  The measured wall-clock time of each mode is
written to ``results/matrix_timing.json`` and printed at the end.

Outputs
-------
``results/metrics.csv``        one row per run (configuration + metrics)
``results/matrix_summary.json`` medians per (fidelity, controller, fault)
``results/matrix_timing.json`` measured runtimes for the mode that was run
``results/cpu_budget.csv``     per-controller FLOP/cycle budget
``docs/figures/*.png``         trajectory, altitude, RMSE bars, parameter convergence
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..metrics import rmse, step_metrics
from .controllers import CONTROLLERS, build_controller
from .cpu import table as cpu_table
from .faults import FAULT_NAMES
from .loop import RunTrace, run_step, run_trajectory

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = REPO_ROOT / "results"
DEFAULT_FIGURES = REPO_ROOT / "docs" / "figures"


# ---------------------------------------------------------------------------
# Matrix definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatrixConfig:
    fidelities: tuple[str, ...]
    controllers: tuple[str, ...]
    faults: tuple[str, ...]
    seeds: tuple[int, ...]
    step_t_end: float
    traj_fidelities: tuple[str, ...]
    traj_t_end: float


def quick_config() -> MatrixConfig:
    return MatrixConfig(
        fidelities=("paper", "high"),
        controllers=CONTROLLERS,
        faults=("none", "wind_gust"),
        seeds=(0,),
        step_t_end=7.0,
        traj_fidelities=("paper", "high"),
        traj_t_end=6.0,
    )


def full_config() -> MatrixConfig:
    return MatrixConfig(
        fidelities=("paper", "high"),
        controllers=CONTROLLERS,
        faults=("none", "aero_density", "wind_gust", "mass_change", "voltage_decay"),
        seeds=(0, 1, 2),
        step_t_end=8.0,
        traj_fidelities=("paper", "high"),
        traj_t_end=12.0,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _theta_summary(trace: RunTrace, row: dict) -> None:
    """Roll-channel b coefficients (order-aware: order-3 has b0+b1+b2)."""
    if not (trace.theta_hat.size and np.any(np.isfinite(trace.theta_hat))):
        return
    theta = trace.theta_hat[-1, 0]
    order = trace.theta_hat.shape[2] // 2
    if theta.size != 2 * order:
        return
    b = theta[order:]
    row["theta_b0_roll"] = float(b[0])
    row["theta_b1_roll"] = float(b[1]) if b.size > 1 else float("nan")
    row["theta_bsum_roll"] = float(np.sum(b))

def _step_row(trace: RunTrace, fidelity: str, controller: str, fault: str, seed: int) -> dict:
    t = trace.t
    y = trace.y
    y_ref = trace.y_ref
    m = step_metrics(t, y[:, 0], y_ref[:, 0], t_step=1.5) if not trace.diverged else None
    row = {
        "scenario": "step",
        "fidelity": fidelity,
        "controller": controller,
        "fault": fault,
        "seed": seed,
        "t_end": float(t[-1]),
        "diverged": trace.diverged,
        "safe_mode": trace.safe_mode,
        "rmse_roll_deg": np.rad2deg(rmse(t, y[:, 0], y_ref[:, 0])),
        "rmse_pitch_deg": np.rad2deg(rmse(t, y[:, 1], y_ref[:, 1])),
        "rmse_yaw_deg": np.rad2deg(rmse(t, y[:, 2], y_ref[:, 2])),
        "rmse_alt_m": rmse(t, y[:, 3], y_ref[:, 3]),
        "att_rmse_deg": np.rad2deg(
            np.sqrt(
                np.mean(
                    np.mean((y[:, :3] - y_ref[:, :3]) ** 2, axis=1)
                )
            )
        ),
        "max_att_deg": float(np.rad2deg(np.max(np.abs(y[:, :3] - y_ref[:, :3])))),
        "pos_rmse_m": float("nan"),
        "overshoot_pct": float("nan"),
        "settling_s": float("nan"),
        "ss_error_rad": float("nan"),
        "control_rms": float(
            np.sqrt(np.mean(np.mean(trace.u_actual**2, axis=1)))
        ),
        "theta_b0_roll": float("nan"),
        "theta_b1_roll": float("nan"),
        "theta_bsum_roll": float("nan"),
        "runtime_s": trace.metadata.get("runtime_s", float("nan")),
    }
    if m is not None:
        row["overshoot_pct"] = m.overshoot
        row["settling_s"] = m.settling_time
        row["ss_error_rad"] = m.steady_state_error
    _theta_summary(trace, row)
    return row


def _trajectory_row(trace: RunTrace, fidelity: str, controller: str, fault: str, seed: int) -> dict:
    t = trace.t
    y = trace.y
    y_ref = trace.y_ref
    pos_err = np.linalg.norm(trace.xi - trace.xi_ref, axis=1)
    row = {
        "scenario": "trajectory",
        "fidelity": fidelity,
        "controller": controller,
        "fault": fault,
        "seed": seed,
        "t_end": float(t[-1]),
        "diverged": trace.diverged,
        "safe_mode": trace.safe_mode,
        "rmse_roll_deg": np.rad2deg(rmse(t, y[:, 0], y_ref[:, 0])),
        "rmse_pitch_deg": np.rad2deg(rmse(t, y[:, 1], y_ref[:, 1])),
        "rmse_yaw_deg": np.rad2deg(rmse(t, y[:, 2], y_ref[:, 2])),
        "rmse_alt_m": rmse(t, y[:, 3], y_ref[:, 3]),
        "att_rmse_deg": np.rad2deg(
            np.sqrt(np.mean(np.mean((y[:, :3] - y_ref[:, :3]) ** 2, axis=1)))
        ),
        "max_att_deg": float(np.rad2deg(np.max(np.abs(y[:, :3] - y_ref[:, :3])))),
        "pos_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "overshoot_pct": float("nan"),
        "settling_s": float("nan"),
        "ss_error_rad": float("nan"),
        "control_rms": float(np.sqrt(np.mean(np.mean(trace.u_actual**2, axis=1)))),
        "theta_b0_roll": float("nan"),
        "theta_b1_roll": float("nan"),
        "theta_bsum_roll": float("nan"),
        "runtime_s": trace.metadata.get("runtime_s", float("nan")),
    }
    _theta_summary(trace, row)
    return row


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def run_matrix(mode: str, out_dir: Path, figures_dir: Path, quiet: bool = False) -> dict:
    cfg = quick_config() if mode == "quick" else full_config()
    started = time.perf_counter()
    rows: list[dict] = []
    per_run: list[float] = []

    total = (
        len(cfg.fidelities) * len(cfg.controllers) * len(cfg.faults) * len(cfg.seeds)
        + len(cfg.traj_fidelities) * len(cfg.controllers)
    )
    index = 0
    for fidelity in cfg.fidelities:
        for controller in cfg.controllers:
            for fault in cfg.faults:
                for seed in cfg.seeds:
                    index += 1
                    ctrl = build_controller(controller, _nominal_params(fidelity), fidelity)
                    trace = run_step(
                        fidelity, ctrl, fault=fault, seed=seed, t_end=cfg.step_t_end
                    )
                    rows.append(_step_row(trace, fidelity, controller, fault, seed))
                    per_run.append(trace.metadata["runtime_s"])
                    if not quiet:
                        print(
                            f"[{index:3d}/{total:3d}] step {fidelity:5s} {controller:12s} "
                            f"{fault:13s} seed={seed} div={trace.diverged} "
                            f"{trace.metadata['runtime_s']:.1f}s",
                            flush=True,
                        )
    for fidelity in cfg.traj_fidelities:
        for controller in cfg.controllers:
            index += 1
            ctrl = build_controller(controller, _nominal_params(fidelity), fidelity)
            trace = run_trajectory(
                fidelity, ctrl, fault="none", seed=0, t_end=cfg.traj_t_end
            )
            rows.append(_trajectory_row(trace, fidelity, controller, "none", 0))
            per_run.append(trace.metadata["runtime_s"])
            if not quiet:
                print(
                    f"[{index:3d}/{total:3d}] 3d   {fidelity:5s} {controller:12s} "
                    f"none          seed=0 div={trace.diverged} "
                    f"{trace.metadata['runtime_s']:.1f}s",
                    flush=True,
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario", "fidelity", "controller", "fault", "seed", "t_end",
        "diverged", "safe_mode",
        "rmse_roll_deg", "rmse_pitch_deg", "rmse_yaw_deg", "rmse_alt_m",
        "att_rmse_deg", "max_att_deg", "pos_rmse_m",
        "overshoot_pct", "settling_s", "ss_error_rad", "control_rms",
        "theta_b0_roll", "theta_b1_roll", "theta_bsum_roll", "runtime_s",
    ]
    with (out_dir / "metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    # medians per (fidelity, controller, fault) for the report tables
    summary: dict[str, dict] = {}
    for row in rows:
        key = f"{row['scenario']}|{row['fidelity']}|{row['controller']}|{row['fault']}"
        summary.setdefault(key, []).append(row)
    summary_out = {}
    for key, group in summary.items():
        entry = {"n": len(group), "diverged": sum(1 for r in group if r["diverged"])}
        for field in ("att_rmse_deg", "rmse_alt_m", "overshoot_pct", "settling_s",
                      "pos_rmse_m", "control_rms", "max_att_deg"):
            values = np.array([r[field] for r in group], dtype=float)
            values = values[np.isfinite(values)]
            entry[field] = float(np.median(values)) if values.size else float("nan")
        summary_out[key] = entry
    (out_dir / "matrix_summary.json").write_text(json.dumps(summary_out, indent=2))

    cpu_rows = cpu_table()
    with (out_dir / "cpu_budget.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cpu_rows[0]))
        writer.writeheader()
        writer.writerows(cpu_rows)

    # Tables and all four figures are generated by ``experiments.report`` from the
    # metrics.csv just written, so a run can never leave stale figures behind.
    from . import report as _report

    figures = _report.generate(
        out_dir, figures_dir, out=figures_dir.parent / "RESULTS.md"
    )

    wall = time.perf_counter() - started
    timing = {
        "mode": mode,
        "runs": len(rows),
        "step_runs": sum(1 for r in rows if r["scenario"] == "step"),
        "trajectory_runs": sum(1 for r in rows if r["scenario"] == "trajectory"),
        "wall_seconds": round(wall, 1),
        "mean_run_seconds": round(float(np.mean(per_run)), 2),
        "median_run_seconds": round(float(np.median(per_run)), 2),
        "config": {
            "fidelities": list(cfg.fidelities),
            "controllers": list(cfg.controllers),
            "faults": list(cfg.faults),
            "seeds": list(cfg.seeds),
            "step_t_end": cfg.step_t_end,
            "traj_fidelities": list(cfg.traj_fidelities),
            "traj_t_end": cfg.traj_t_end,
        },
        "figures": figures,
    }
    (out_dir / "matrix_timing.json").write_text(json.dumps(timing, indent=2))

    if not quiet:
        print()
        print(f"mode={mode} runs={len(rows)} wall={wall/60.0:.1f} min "
              f"(mean {np.mean(per_run):.2f} s/run)")
        print(f"metrics : {out_dir / 'metrics.csv'}")
        print(f"summary : {out_dir / 'matrix_summary.json'}")
        print(f"timing  : {out_dir / 'matrix_timing.json'}")
        for path in figures:
            print(f"figure  : {path}")
        print()
        print("CPU budget (analytical, 4 cycles/FLOP, 84 MHz, 40 ms period):")
        for row in cpu_rows:
            print(
                f"  {row['controller']:12s} STR axes={row['str_channels']} "
                f"{row['total_flops']:7.0f} FLOPs  {row['cycles_per_tick']:8d} cycles "
                f"{row['budget_pct']:5.2f} % of budget"
            )
    return timing


def _nominal_params(fidelity: str):
    from ..params import default_params

    return default_params(fidelity)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pistr_sim.experiments.matrix",
        description="D-phase controller x fidelity x fault x seed matrix",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="quick subset (default)")
    group.add_argument("--full", action="store_true", help="full matrix (all faults, 3 seeds)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    mode = "full" if args.full else "quick"
    run_matrix(mode, args.out, args.figures, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
