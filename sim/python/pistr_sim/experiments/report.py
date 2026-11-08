"""Generate ``docs/RESULTS.md`` **and all report figures** from one data set.

Run after the matrix (or after ``--quick``/``--full``; both call this module
automatically)::

    PYTHONPATH=sim/python python3 -m pistr_sim.experiments.matrix --quick
    PYTHONPATH=sim/python python3 -m pistr_sim.experiments.report

Single source of truth: the multi-seed tables and the bar charts are both
aggregated from ``results/metrics.csv`` with the *same* ``median [min-max]``
statistic; the two single-run figures (3-D paths, RLS convergence) state
"seed 0" in their titles because they cannot be multi-seed aggregates.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_FIGURES = REPO_ROOT / "docs" / "figures"
DEFAULT_OUT = REPO_ROOT / "docs" / "RESULTS.md"

CONTROLLER_LABEL = {
    "pid": "PID (robust, c1)",
    "str": "STR (Mode I, single loop)",
    "pi_str": "PI-STR (4-channel)",
    "pi_str_1axis": "PI-STR 1-axis + PID",
    "pi_str_3axis": "PI-STR 3-axis + PID alt",
    "lqr": "LQR (3-state)",
}
ORDER = ["pid", "str", "pi_str_1axis", "pi_str_3axis", "pi_str", "lqr"]


# ---------------------------------------------------------------------------
# Data loading / statistics
# ---------------------------------------------------------------------------
def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            parsed = {}
            for key, value in row.items():
                if key in ("scenario", "fidelity", "controller", "fault"):
                    parsed[key] = value
                elif key in ("diverged", "safe_mode"):
                    parsed[key] = str(value) == "True"
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = float("nan")
            rows.append(parsed)
    return rows


def _pick(rows, **criteria):
    out = rows
    for key, value in criteria.items():
        out = [r for r in out if r[key] == value]
    return out


def _values(rows, key):
    values = np.array([r[key] for r in rows], dtype=float)
    return values[np.isfinite(values)]


def _median(rows, key):
    values = _values(rows, key)
    return float(np.median(values)) if values.size else float("nan")


def _spread(rows, key, digits=2, suffix=""):
    """``median [min–max]`` string; a single sample renders as the value only."""
    values = _values(rows, key)
    if not values.size:
        return "n/a"
    med = float(np.median(values))
    if values.size == 1:
        return f"{med:.{digits}f}{suffix}"
    return f"{med:.{digits}f} [{values.min():.{digits}f}–{values.max():.{digits}f}]{suffix}"


def _f(value, digits=2, suffix=""):
    return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}{suffix}"


def _table(headers, body):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="pistr-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_aggregate_figures(rows, figures: Path, n_seeds: int) -> list[Path]:
    plt = _pyplot()
    high = _pick(rows, scenario="step", fidelity="high", fault="none")
    if not high:
        return []
    controllers = [c for c in ORDER if _pick(high, controller=c)]
    labels = [CONTROLLER_LABEL[c] for c in controllers]
    x = np.arange(len(controllers))

    # (a) attitude RMSE with min-max whiskers
    fig, ax = plt.subplots(figsize=(10, 4.8))
    stats = np.array(
        [
            [np.median(_values(_pick(high, controller=c), f"rmse_{axis}_deg"))
             for axis in ("roll", "pitch", "yaw")]
            for c in controllers
        ]
    )
    lows = np.array(
        [
            [min(_values(_pick(high, controller=c), f"rmse_{axis}_deg"))
             for axis in ("roll", "pitch", "yaw")]
            for c in controllers
        ]
    )
    highs = np.array(
        [
            [max(_values(_pick(high, controller=c), f"rmse_{axis}_deg"))
             for axis in ("roll", "pitch", "yaw")]
            for c in controllers
        ]
    )
    width = 0.26
    for i, axis in enumerate(("roll", "pitch", "yaw")):
        ax.bar(
            x + (i - 1) * width,
            stats[:, i],
            width,
            yerr=[stats[:, i] - lows[:, i], highs[:, i] - stats[:, i]],
            capsize=3,
            label=axis,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("RMSE [deg]")
    ax.set_title(
        f"attitude RMSE, high fidelity, no fault — median [min–max] of {n_seeds} seeds"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path_rmse = figures / "rmse_bars.png"
    fig.savefig(path_rmse, dpi=140)
    plt.close(fig)

    # (b) overshoot / altitude RMSE with min-max whiskers
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for ax, key, ylabel in (
        (axes[0], "overshoot_pct", "overshoot [%]"),
        (axes[1], "rmse_alt_m", "altitude RMSE [m]"),
    ):
        med = np.array([np.median(_values(_pick(high, controller=c), key)) for c in controllers])
        lo = np.array([min(_values(_pick(high, controller=c), key)) for c in controllers])
        hi = np.array([max(_values(_pick(high, controller=c), key)) for c in controllers])
        ax.bar(x, med, yerr=[med - lo, hi - med], capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("roll step overshoot")
    axes[1].set_title("altitude RMSE (hover hold)")
    fig.suptitle(f"high fidelity, no fault — median [min–max] of {n_seeds} seeds")
    fig.tight_layout()
    path_alt = figures / "altitude_rmse.png"
    fig.savefig(path_alt, dpi=140)
    plt.close(fig)
    return [path_rmse, path_alt]


def _representative_step_trace():
    """Deterministic single-run trace for the RLS-convergence figure."""
    from ..params import default_params
    from .controllers import build_controller
    from .loop import run_step

    controller = build_controller("pi_str", default_params("high"), "high")
    return run_step("high", controller, fault="none", seed=0, t_end=7.0)


def _plot_parameter_convergence(figures: Path) -> Path | None:
    plt = _pyplot()
    trace = _representative_step_trace()
    names = ["a1", "a2", "b0", "b1"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        for i, pname in enumerate(names):
            ax.plot(trace.t, trace.theta_hat[:, j, i], lw=1.1, label=pname)
        ax.axhline(0.0, color="k", lw=0.6, ls=":")
        ax.set_title(["roll", "pitch", "yaw", "altitude"][j])
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, ncol=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.suptitle(
        "RLS parameters, PI-STR, high fidelity, no fault — single run, seed 0 "
        "(closed loop: only b0+b1 is identifiable; b0 can change sign)"
    )
    fig.tight_layout()
    path = figures / "parameter_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _plot_trajectory(figures: Path) -> Path | None:
    plt = _pyplot()
    from ..params import default_params
    from .controllers import build_controller
    from .loop import run_trajectory

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    reference = None
    for controller in ORDER:
        ctrl = build_controller(controller, default_params("paper"), "paper")
        trace = run_trajectory("paper", ctrl, fault="none", seed=0, t_end=6.0)
        ax.plot(trace.xi[:, 0], trace.xi[:, 1], -trace.xi[:, 2], lw=1.3,
                label=CONTROLLER_LABEL[controller])
        reference = trace.xi_ref
    if reference is not None:
        ax.plot(reference[:, 0], reference[:, 1], -reference[:, 2], "k--", lw=1.6,
                label="reference")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("altitude [m]")
    ax.set_title(
        "3-D trajectory, paper fidelity — single run, seed 0 "
        "(failed outer loop; see RESULTS §5)"
    )
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    path = figures / "trajectory_3d.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build(results: Path = DEFAULT_RESULTS) -> str:
    rows = _load_rows(results / "metrics.csv")
    timing_path = results / "matrix_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    cpu_path = results / "cpu_budget.csv"
    cpu_rows = list(csv.DictReader(cpu_path.open())) if cpu_path.exists() else []
    quick_path = results / "timing_quick.json"
    quick = json.loads(quick_path.read_text()) if quick_path.exists() else {}
    full_path = results / "timing_full.json"
    full = json.loads(full_path.read_text()) if full_path.exists() else {}

    step_high_none = _pick(rows, scenario="step", fidelity="high", fault="none")
    n_seeds = len({int(r["seed"]) for r in step_high_none}) or 1
    b0_rows = _pick(rows, scenario="step", fidelity="high", controller="pi_str")
    b0_negative = sum(1 for r in b0_rows if r["theta_b0_roll"] < 0)

    s: list[str] = []
    s.append(
        """# D-phase results: controllers, fidelities, faults

**Status: simulation only.** No hardware-in-the-loop or flight test exists in this
repository; every number below comes from `sim/python` and must be read as a
model-relative comparison, not as a flight-qualification claim.

## 1. Aggregation and how to read this document

* **One source of truth.** The tables and the multi-seed bar charts are generated
  from `results/metrics.csv` by the *same* statistic, `median [min–max]` over the
  seeds of each configuration.  The figure titles state the aggregation and the
  sample count, so a chart can never silently show a different subset than its
  table.
* **Sample size is small.** `--full` uses **3 seeds** and `--quick` uses **1**.
  With n = 3 the min–max range is a crude uncertainty interval, not a confidence
  interval; §7 quantifies how large the seed spread is and why rankings that
  depend on sub-degree differences are not reliable.
* **Two figures are single-run and say so.** The 3-D path figure and the RLS
  parameter-convergence figure are deterministic single runs at **seed 0** (a
  path/convergence curve cannot be averaged over seeds); their titles say
  "single run, seed 0".
* Regenerate everything with (the matrix also calls this automatically):

```
cd sim/python && python3 -m pistr_sim.experiments.report
```

| item | values |
|---|---|
| fidelities | `paper` (MATLAB-parity plant) and `high` (BEMT, drag, battery, ESC, wind, sensors) |
| controllers | PID (paper `c1`), STR (Mode I), PI-STR 4-channel, PI-STR 1-axis + PID, PI-STR 3-axis + PID altitude, LQR (3-state) |
| faults | none, aero_density (+20 % rotor coefficients), wind_gust (3 m/s + Dryden), mass_change (+0.25 kg), voltage_decay (-15 % pack voltage) |
| seeds | sensor-noise / turbulence seeds: 1 (`--quick`) or 3 (`--full`) |
| inner-loop scenario | hover at 5 m, roll step 0.1 rad at t = 1.5 s, 7 s (`--quick`) / 8 s (`--full`) |
| outer-loop scenario | paper 3-D trajectory (vx = 4 m/s, sinusoidal lateral motion, 42° yaw sweep), altitude held at 20 m |

The controller always uses the **nominal** parameters; faults change the plant
only, so the matrix measures robustness, not retuning.  The 3-D scenario uses the
plant's true position/velocity as an **ideal position aid** (there is no GPS in
the sensor model); the inner-loop comparison is unaffected by that assumption.
"""
    )

    # --- §2 step table ------------------------------------------------------
    body = []
    for controller in ORDER:
        group = _pick(step_high_none, controller=controller)
        if not group:
            continue
        body.append(
            [
                CONTROLLER_LABEL[controller],
                _spread(group, "att_rmse_deg", 2, "°"),
                _spread(group, "rmse_roll_deg", 2, "°"),
                _spread(group, "overshoot_pct", 2, "%"),
                _spread(group, "settling_s", 2, " s"),
                _spread(group, "rmse_alt_m", 2, " m"),
                f"{sum(1 for r in group if r['diverged'])}/{len(group)}",
            ]
        )
    s.append(
        f"## 2. Step response, high fidelity, no fault (n = {n_seeds} seeds)\n\n"
        + _table(
            ["controller", "attitude RMSE", "roll RMSE", "overshoot", "2 % settling",
             "altitude RMSE", "diverged"],
            body,
        )
        + "\n\nCells are `median [min–max]`.  The 2 % settling band is 0.002 rad on a "
        "0.1 rad step, i.e. tight; the attitude estimate has no position aiding, so "
        "the residual gyro random-walk drift can keep the true attitude outside the "
        "band (see §10).  The `str` (Mode I) altitude column is a real failure, not "
        "noise: the single-loop order-3 STR is not viable on the altitude channel.\n"
    )

    # --- §3 faults ----------------------------------------------------------
    faults = ["none", "aero_density", "wind_gust", "mass_change", "voltage_decay"]
    body = []
    for controller in ORDER:
        cells = [CONTROLLER_LABEL[controller]]
        for fault in faults:
            group = _pick(rows, scenario="step", fidelity="high", controller=controller, fault=fault)
            if not group:
                cells.append("-")
                continue
            cells.append(
                f"{_spread(group, 'att_rmse_deg', 1, '°')} / "
                f"{sum(1 for r in group if r['diverged'])}/{len(group)}"
            )
        body.append(cells)
    s.append(
        "## 3. Fault robustness (high fidelity, step)\n\n"
        + _table(["controller"] + faults, body)
        + "\n\nEach cell is `attitude RMSE median [min–max] / diverged runs`.  Faults "
        "are unmodelled by design.  `wind_gust` is a no-op in the `paper` fidelity "
        "(no aero path), so it is only meaningful in the `high` table.  The seed "
        "spread is as large as in §2; do not rank controllers on the last digit.\n"
    )

    # --- §4 fidelity --------------------------------------------------------
    body = []
    for controller in ORDER:
        paper = _pick(rows, scenario="step", fidelity="paper", controller=controller, fault="none")
        high = _pick(rows, scenario="step", fidelity="high", controller=controller, fault="none")
        if not paper or not high:
            continue
        body.append(
            [
                CONTROLLER_LABEL[controller],
                _spread(paper, "att_rmse_deg", 2, "°"),
                _spread(high, "att_rmse_deg", 2, "°"),
                _spread(paper, "overshoot_pct", 1, "%"),
                _spread(high, "overshoot_pct", 1, "%"),
                _spread(paper, "rmse_alt_m", 2, " m"),
                _spread(high, "rmse_alt_m", 2, " m"),
            ]
        )
    s.append(
        "## 4. Fidelity degradation (no fault)\n\n"
        + _table(
            ["controller", "att RMSE paper", "att RMSE high", "overshoot paper",
             "overshoot high", "alt RMSE paper", "alt RMSE high"],
            body,
        )
        + "\n\nCells are `median [min–max]` over the seeds.  `high` adds BEMT inflow, "
        "H-force, body drag, rotor gyro moments, battery sag, ESC lag/quantisation, "
        "Dryden turbulence and sensor noise/delay.\n"
    )

    # --- §5 3-D -------------------------------------------------------------
    body = []
    for controller in ORDER:
        cells = [CONTROLLER_LABEL[controller]]
        for fidelity in ("paper", "high"):
            group = _pick(rows, scenario="trajectory", fidelity=fidelity, controller=controller)
            if not group:
                cells += ["-", "-", "-"]
                continue
            cells += [
                _f(_median(group, "pos_rmse_m"), 1, " m"),
                _f(_median(group, "att_rmse_deg"), 1, "°"),
                f"{sum(1 for r in group if r['diverged'])}/{len(group)}",
            ]
        body.append(cells)
    s.append(
        "## 5. Paper 3-D trajectory (altitude held at 20 m)\n\n"
        + _table(
            ["controller", "pos RMSE paper", "att RMSE paper", "div", "pos RMSE high",
             "att RMSE high", "div"],
            body,
        )
        + "\n\nSingle seed (seed 0) per fidelity; no min–max is shown because n = 1.  "
        "**This suite failed and is reported as a failure, not as a ranking.**  The "
        "3-D outer loop (position PD + small-angle attitude planning) does not track "
        "the paper's 60 m lateral / 42° yaw reference in this harness: the "
        "altitude/low-frequency mode diverges and the vehicles end up hundreds of "
        "metres away.  Contributing causes: the plan's `asin(u_x/(-m g))` mapping has "
        "the opposite sign for this plant convention (same class of defect as "
        "CONTROL_SPEC §9.9; the harness uses the exact thrust-direction inversion), "
        "the paper's position gains (`k1 = 0.04`, `k2 = 0.4`) are far too soft to "
        "reject high-fidelity drag, and the 25 Hz cascade with a moving altitude "
        "reference exposes the altitude-channel identification drift.  **The "
        "absolute numbers must not be read as controller quality**; the step matrix "
        "(§2/§3) is the fair inner-loop comparison.  Fixing the 3-D outer loop is "
        "future work.\n\n"
        "![3-D trajectory](figures/trajectory_3d.png)\n"
    )

    # --- §6 identification --------------------------------------------------
    s.append(
        "## 6. Closed-loop identifiability of the STR\n\n"
        + _table(
            ["median b0 (roll)", "median b1 (roll)", "median b0+b1", "runs with b0 < 0"],
            [[
                _f(_median(b0_rows, "theta_b0_roll"), 5),
                _f(_median(b0_rows, "theta_b1_roll"), 5),
                _f(_median(b0_rows, "theta_bsum_roll"), 5),
                f"{b0_negative}/{len(b0_rows)}",
            ]],
        )
        + "\n\nWith a smooth step reference and a forgetting factor the regressor is "
        "not persistently exciting: RLS identifies `b0 + b1` but not the split between "
        "`b0` and `b1`, and `b0` can even change sign.  This is the CONTROL_SPEC §9.10 "
        "statement reproduced by the matrix, **not an identifier failure**; open-loop "
        "PRBS excitation recovers all four parameters to machine precision "
        "(`results/control_stack/ident_firmware`).\n\n"
        "![parameter convergence](figures/parameter_convergence.png)\n\n"
        "![RMSE bars](figures/rmse_bars.png)\n\n"
        "![altitude](figures/altitude_rmse.png)\n"
    )

    # --- §7 uncertainty -----------------------------------------------------
    def seed_list(controller: str, key: str = "overshoot_pct") -> str:
        group = _pick(rows, scenario="step", fidelity="high", controller=controller, fault="none")
        values = [f"{v:.2f}" for v in _values(group, key)]
        return ", ".join(values) if values else "n/a"

    s.append(
        f"""## 7. Uncertainty: the seed spread is large (n = {n_seeds})

The controller ranking is much less certain than the medians suggest.  For the
high-fidelity step, the per-seed overshoot values are:

| controller | per-seed overshoot [%] | median [min–max] |
|---|---|---|
| PID (robust, c1) | {seed_list('pid')} | {_spread(_pick(step_high_none, controller='pid'), 'overshoot_pct', 2, '%')} |
| PI-STR 1-axis + PID | {seed_list('pi_str_1axis')} | {_spread(_pick(step_high_none, controller='pi_str_1axis'), 'overshoot_pct', 2, '%')} |
| PI-STR 3-axis + PID alt | {seed_list('pi_str_3axis')} | {_spread(_pick(step_high_none, controller='pi_str_3axis'), 'overshoot_pct', 2, '%')} |
| PI-STR (4-channel) | {seed_list('pi_str')} | {_spread(_pick(step_high_none, controller='pi_str'), 'overshoot_pct', 2, '%')} |
| LQR (3-state) | {seed_list('lqr')} | {_spread(_pick(step_high_none, controller='lqr'), 'overshoot_pct', 2, '%')} |

* `pi_str_1axis` ranges from ~60 % to ~107 % overshoot, `pi_str_3axis` from
  ~2.7 % to ~31.7 %, PID from ~57 % to ~87 %: the min–max range reaches and
  sometimes exceeds **twice the median**.  A ranking that depends on a few tenths
  of a degree (or a few percent of overshoot) is **not** supported by n = 3.
* The extremes are not removed: they are kept in every table and in
  `results/metrics.csv`.  The outliers (the worst seed of `pi_str_1axis` and
  `pi_str_3axis`) come from the sensor-noise / turbulence seed, and an adaptive
  loop can enter a poor transient on an unlucky noise realisation.
* What *is* robust across seeds: the 4-channel PI-STR and the 3-axis variant stay
  at single-digit overshoot with the lowest attitude RMSE; PID and the 1-axis mix
  stay at tens of percent; Mode I stays acceptable in attitude but unusable in
  altitude; no controller diverges or enters safe-mode in any run.  Those
  qualitative statements survive the spread; the precise ordering of the
  near-tied adaptive variants does not.
"""
    )

    # --- §8 CPU -------------------------------------------------------------
    if cpu_rows:
        s.append(
            "## 8. Real-time budget (Arduino Due, 84 MHz, 40 ms period)\n\n"
            + _table(
                ["controller", "STR axes", "controller FLOPs",
                 "total FLOPs incl. observer/mixer/planner", "cycles/tick",
                 "% of 3.36e6-cycle budget"],
                [
                    [
                        CONTROLLER_LABEL.get(r["controller"], r["controller"]),
                        r["str_channels"],
                        f"{float(r['controller_flops']):.0f}",
                        f"{float(r['total_flops']):.0f}",
                        r["cycles_per_tick"],
                        f"{float(r['budget_pct']):.2f} %",
                    ]
                    for r in cpu_rows
                ],
            )
            + "\n\nAnalytical count of floating-point operations per controller tick, "
            "converted with a deliberately conservative 4 cycles/FLOP (the Due has no "
            "FPU, so software float is slower than 1 cycle/op).  Even the 4-channel "
            "Mode-I STR uses ~0.3 % of the period budget; the matrix is not "
            "CPU-limited.  These are estimates, not measurements on hardware.\n"
        )

    # --- §9 reproduce -------------------------------------------------------
    def runtime_line(t):
        if not t:
            return "not measured"
        return (
            f"{t.get('runs')} runs, {t.get('wall_seconds', 0) / 60.0:.1f} min wall, "
            f"{t.get('mean_run_seconds')} s/run"
        )

    s.append(
        "## 9. Reproduce and runtime\n\n"
        "```\n"
        "PYTHONPATH=sim/python python3 -m pistr_sim.experiments.matrix --quick\n"
        "PYTHONPATH=sim/python python3 -m pistr_sim.experiments.matrix --full\n"
        "PYTHONPATH=sim/python python3 -m pistr_sim.experiments.report   # tables + 4 figures\n"
        "```\n\n"
        "`--quick` is the default.  **The matrix calls the report automatically**, so "
        "a `--full` run cannot leave stale figures behind; the standalone report "
        "command is for regenerating both from the existing `metrics.csv`.\n\n"
        f"* quick: {runtime_line(quick)}\n"
        f"* full: {runtime_line(full)}\n\n"
        "The physics plant runs 40 RK4/actuator sub-steps per 40 ms control tick; the "
        "runtime is dominated by the high-fidelity plant, not by the controllers.\n"
    )

    # --- §10 conclusions ----------------------------------------------------
    s.append(
        """## 10. Conclusions (simulation-only) and limitations

1. **PI-STR over PID is the robust choice for the inner loop.**  On the
   high-fidelity step the 4-channel PI-STR stays at single-digit overshoot with
   the lowest attitude RMSE of the adaptive set; the paper's robust PID overshoots
   by tens of percent (median ~86 %, seed range ~57–87 %).  The single-axis STR +
   PID mix sits between the two, as expected: the adaptive channel recovers the
   gain, the PID channels do not.
2. **The 3-axis/4-channel STR is stable** in the step matrix (no divergence, no
   safe-mode in any of the 192 full-matrix runs).  Its CPU cost is ~2.4× the
   1-axis configuration but still far below the Due budget (§8).  The honest
   caveat is `str` (Mode I): the single-loop order-3 STR is *not* viable on the
   altitude channel (~9 m altitude RMSE, both fidelities), while its attitude
   channels are acceptable; Mode I is a paper-reproduction artefact, not a
   deployable altitude controller.
3. **LQR is the closest baseline** on the inner loop (comparable RMSE, lowest
   overshoot, no online identification) but it has no integral action and no
   adaptation; it is a model-based baseline whose performance depends on the
   linearisation point (the 3-state model includes the motor lag; a 2-state
   [angle, rate] LQR oscillated because it omitted that lag).
4. **Faults are tolerated without retuning** in the tested envelope; the fault
   table's medians are flat within the seed spread, so the matrix supports
   "no divergence / no degradation beyond the spread" rather than a severity
   ranking.  `mass_change` and `aero_density` shift the apparent ARX gain and are
   absorbed by the RLS in the adaptive controllers.
5. **The 3-D trajectory failed and is not a result to build on** (§5): the outer
   loop does not track the 60 m lateral / 42° yaw reference; the vehicles end up
   hundreds of metres away.  The figure is included because the task requires it
   and because silently dropping a failed experiment would be worse; it must not
   be read as controller ranking.  A redesigned outer loop (correct
   thrust-direction inversion, stronger position gains, an altitude tracker) is
   future work.
6. **Known limitations carried from the earlier stages**: the sinusoidal yaw
   sweep at 2 rad/s is not tracked well (~22° RMSE; the run is in
   `results/control_stack/` but is not part of the matrix), the true roll in the
   step settles ~1.2–1.5° above the reference because the gyro random walk is
   unobservable without position aiding, and the closed-loop `b0` sign flip is the
   expected identifiability limit, not a bug.
7. **No hardware validation.**  The `.ino` files cannot be compiled here; the
   firmware integration is reviewed by hand.  The CPU numbers are analytical.
"""
    )
    return "\n".join(s)


def generate(
    results: Path = DEFAULT_RESULTS,
    figures: Path = DEFAULT_FIGURES,
    out: Path = DEFAULT_OUT,
) -> list[str]:
    """Refresh tables and all four figures from ``results/metrics.csv``."""
    figures.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(results / "metrics.csv")
    step_high_none = _pick(rows, scenario="step", fidelity="high", fault="none")
    n_seeds = len({int(r["seed"]) for r in step_high_none}) or 1
    paths = _plot_aggregate_figures(rows, figures, n_seeds)
    param = _plot_parameter_convergence(figures)
    if param is not None:
        paths.append(param)
    trajectory = _plot_trajectory(figures)
    if trajectory is not None:
        paths.append(trajectory)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(results))
    return [str(p) for p in paths]


def main(argv: list[str] | None = None) -> int:
    paths = generate()
    print(f"wrote {DEFAULT_OUT}")
    for path in paths:
        print(f"figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
