# Quickstart

Three ways to use this repository, ordered by how little you need installed:

| Path | You want to… | Needs |
|---|---|---|
| [1. Run the simulation](#path-1-run-the-simulation) | see the controller fly, change a scenario | Python 3.10+ |
| [2. Change the control law](#path-2-change-the-control-law) | modify / validate an algorithm | Python 3.10+, CMake + C++17 compiler |
| [3. Change the firmware](#path-3-change-the-firmware) | touch the Arduino Due code, fly it | PlatformIO, a Due, **propellers off** |
| [4. MATLAB (legacy)](#path-4-matlab-legacy) | reproduce the published figures | MATLAB + Control System Toolbox |

Everything is run from the repository root unless a `cd` says otherwise. The root
`Makefile` wraps the two most common commands (`make sim`, `make test`), but every
command below is also spelled out so you do not have to use make.

> **Read [CONTROL_SPEC.md](CONTROL_SPEC.md) before changing any
> algorithm.** It is normative: when code and spec disagree, the spec wins.

---

## Prerequisites

```bash
python3 --version        # 3.10 or newer
python3 -c "import numpy, scipy; print(numpy.__version__, scipy.__version__)"
cmake --version          # >= 3.16, only for the C++ tests
c++ --version            # clang++ or g++, C++17, only for the C++ tests
```

The Python package is **not** installed during normal use: `sim/python/pyproject.toml`
sets `pythonpath = ["."]` for pytest and the CLI is run from `sim/python/`, so the
sources are imported in place. Install it only if you want the package importable
from anywhere:

```bash
python3 -m pip install -e "sim/python[dev]"   # optional; adds pytest
```

If your environment has no `pytest`, either install it (`python3 -m pip install pytest`)
or use the interpreter that ships it.

---

## Path 1: run the simulation

From the repository root:

```bash
PYTHONPATH=sim/python python3 -m pistr_sim.cli run \
  --scenario step --controller pi_str --plant physics --fidelity high \
  --out results/control_stack/physics_step
```

Same call from inside the package directory (no `PYTHONPATH` needed):

```bash
cd sim/python && python3 -m pistr_sim.cli run --scenario step --controller pi_str \
  --plant physics --fidelity high --out ../../results/control_stack/physics_step
```

`make sim` wraps the same call: it forwards scenario/controller/fidelity/out and does
not pass `--plant`, which defaults to `physics`. Variations:

```bash
make sim SCENARIO=hover CONTROLLER=pid FIDELITY=paper
PYTHONPATH=sim/python python3 -m pistr_sim.cli --list        # engines/scenarios/controllers/presets
PYTHONPATH=sim/python python3 -m pistr_sim.cli run --help    # every knob
PYTHONPATH=sim/python python3 -m pistr_sim.cli ident --help  # open-loop PRBS identification
```

Two switches that are easy to confuse:

* `--plant {physics,surrogate}` — the **closed-loop engine**. `physics` (default) runs
  `pistr_sim.plant`; `surrogate` runs a plant-free ARX bench with the same controller.
  Use the surrogate for fast unit-level checks and paper-vs-code comparison; it says
  nothing about the aircraft.
* `--fidelity {paper,high}` — the **physics level**, only meaningful with
  `--plant physics` (recorded but unused for the surrogate).

Plant ablation is `--disable-extras inflow,h_force,body_drag,...`; `--use-accel`
enables the accelerometer tilt correction (physics only, off by default).

What to expect: the plant integrates at `dt = 1e-3` s (default `t_end = 30` s, so
30 000 steps); the `step` scenario above takes about 15 s. `--out` is created if
missing and receives `trace.csv` (per-tick state, reference, command, estimates),
`metrics.csv`, `summary.json` and five PNGs (`tracking`, `rates_control`,
`parameters`, `estimator`, `step_response`). Per-run output lives under
`results/<run>/` and is git-ignored; the tracked evidence table is the top-level
`results/metrics.csv` written by the experiment matrix. Curated figures belong in
`docs/figures/` (see [Repository conventions](#repository-conventions)).

Where to look if the numbers look wrong:

| Symptom | First thing to check |
|---|---|
| Roll/pitch diverges immediately | sign of the mixer or of the control gain — compare with `docs/CONTROL_SPEC.md` §5 |
| Attitude tracks but altitude sinks | thrust direction (`f = -C_I2B' * T * e3`) and hover throttle `Params.pwm_hover_us` |
| Estimates of `a1, a2, b0, b1` never converge | excitation: a hover with a constant reference identifies nothing |
| `RuntimeWarning` becomes an error | `filterwarnings = ["error::RuntimeWarning"]` in `pyproject.toml`; a NaN appears before the crash |

With `--plant physics`, two fidelity levels share one interface:

* `--fidelity paper` — line-by-line port of the MATLAB model (Euler, `dt = 1 ms`,
  `U = Umax*sqrt(PWM%)`, no battery/ESC/wind/sensor noise). Use it to reproduce the
  paper, nothing else.
* `--fidelity high` — the default: BEMT rotor inflow, H-force, body drag, rotor
  gyroscopic moments, ESC lag and PWM quantisation, battery sag, Dryden turbulence,
  sensor noise/bias/delay, RK4. Fully specified in
  [`PLANT_MODEL.md`](PLANT_MODEL.md); every extra term can be switched off for
  ablation and for the P12 equivalence test.

The controller still identifies the same 2nd-order ARX channel in both modes. That
is the point: the high-fidelity plant checks whether the identification survives a
plant that does not satisfy the ARX assumption.

---

## Path 2: change the control law

Algorithms live in `sim/python/pistr_sim/` (`control/`, `estimator.py`,
`reference.py`) and, for the flight code, in `firmware/lib/str_core/` (plain C++17,
no Arduino dependency). Both implementations must obey `docs/CONTROL_SPEC.md` §3–§7,
including the numerical-safety guards of §6.

Run the tests:

```bash
python3 -m pytest sim/python/tests        # or: make test-py
```

```bash
cmake -S . -B build && cmake --build build -j

# recommended in CI / sandboxes that refuse to exec a freshly linked binary:
python3 -c "import ctypes; d=ctypes.CDLL('build/libstr_core_verify.dylib'); \
            print(d.str_core_run_tests(b'all'))"     # 0 == PASS, else failure count

# or, on an unrestricted host (this is what `make test-cpp-ctest` runs):
ctest --test-dir build --output-on-failure
```

Both C++ entry points link the same `str_core_testlib`, so they assert the same
things — about 159 000 checks across 8 suites (the count follows the suite:
`make SUITE=guards test-cpp` is 72). `str_core_verify` exposes a C ABI, which is why
`dlopen` works even where `execve` of a freshly linked binary is blocked; if `ctest`
reports `Subprocess killed`, use the ctypes line. The make targets are
`make test-cpp` (build + ctypes, the primary route) and `make test-cpp-ctest`
(build + ctest). The library name is platform-specific:
`libstr_core_verify.dylib` (macOS), `libstr_core_verify.so` (Linux),
`str_core_verify.dll` (Windows).

`pytest sim/python/tests` works from the repository root because pytest picks up
`sim/python/pyproject.toml` as its configuration and adds `sim/python` to
`sys.path`; you do not need `PYTHONPATH`.

Recommended loop when you change an algorithm:

1. Re-read the clause of `CONTROL_SPEC.md` you are implementing.
2. Change the Python implementation first — it is the numerical oracle.
3. Run `python3 -m pytest sim/python/tests` (fast, no hardware).
4. Port the same change to `firmware/lib/str_core/`, rebuild with
   `cmake --build build -j`, and run the ctypes entry (or `ctest`); the golden
   vectors in `tests/golden/` are what keep the two languages honest
   (`CONTROL_SPEC.md` §8: agreement `max|Δ|/max|ref| < 1e-6`).
5. Only then wire it into a `.ino` sketch (path 3).

If a change makes a golden vector fail, the vector wins unless the spec itself was
wrong — in that case update the spec first, in a separate commit.

---

## Path 3: change the firmware

**Safety.** The sketches drive four ESCs on a real airframe. Remove the propellers
and secure the vehicle before the first spin-up.

As shipped, **both** `Config.h` files define `DEBUG`, and `DEBUG` pins every motor
output to `PWM_MIN` at the end of the control function — while the arming state
machine still latches `ARM_flag = 1` and the LED still reports "armed". So a freshly
built vehicle looks armed and never spins: that is the `DEBUG` switch, not a wiring
fault. The refactor in progress replaces this overloaded switch with explicit
`MOTOR_ENABLE` / `MOTOR_TEST` flags — if your `Config.h` already has them, follow its
header comment instead. Read the
[firmware safety section](../README.md#firmware-behaviour-and-safety) before you
comment `DEBUG` out, and keep
[`firmware/SAFETY_NOTES.md`](../firmware/SAFETY_NOTES.md) open next to you: it
separates what stage A fixed, what the audit fixed, and what is identified but
deliberately not fixed — including the missing receiver-loss failsafe (§3.1) and the
`STR_period / 1000` integer division (§3.2).

What is where:

| File | Role |
|---|---|
| `firmware/Quadrotor_Mode_II_Arduino_Due/` | Mode II: PI outer loop + STR inner loop (the paper's structure) |
| `firmware/Quadrotor_Mode_I_Arduino_Due/` | Mode I: single-loop STR on angle |
| `*/Config.h` | compile-time switches: frame (`QuadP`), control algorithm, loop periods, pins, limits |
| `firmware/libraries/MyLibraries_Wenchao/` | IMU/sensor drivers, Kalman attitude filter, legacy `STR` class |
| `firmware/libraries/SdFat/` | vendored third-party SD-card library |
| `firmware/lib/str_core/` | host-testable STR core; the `.ino` files should call into it, not into hand-expanded matrix algebra |

Arduino concatenates every `.ino` in a sketch folder into **one** translation unit,
so a function defined in `Controller.ino` can be called from `Motor.ino`. Adding a
new file means adding a new *section* of one program, not a new module.

Build and flash (PlatformIO):

```bash
python3 -m pip install --upgrade platformio
pio run -e due_mode_ii              # compile
pio run -e due_mode_ii -t upload     # flash to a connected Due
pio device monitor -b 115200         # serial output (Config.h: BaudRate 115200)
pio run -e due_mode_i                # the Mode I sketch
```

> **This path is unverified.** No PlatformIO and no Atmel SAM toolchain were
> available when `platformio.ini` was written, so the configuration has never been
> built. In particular the vendored `SdFat` (2015.4.26) and the library search over
> `firmware/libraries/` are the two things most likely to need adjustment. See the
> comment block at the top of `platformio.ini` for the known risks, and expect to
> fix the environment rather than the flight code if `pio run` fails.

Because the `.ino` files cannot be compiled on a normal host, the rule is:
**algorithmic changes go to `firmware/lib/str_core/` (tested), and the sketch edit
is a mechanical call into it.** Anything else must be reviewed by hand and is not
covered by any test here.

---

## Path 4: MATLAB (legacy)

The MATLAB code is the published study and is kept as-is; it is not required for
anything else in this repository.

```matlab
cd sim/matlab/robustness
Sim_Trajectory_Tracking_Control.m
```

* Needs **Control System Toolbox** (`tf`, `c2d` are used).
* Written for MATLAB R2014a; newer releases should run it, but this is untested.
* Scenario switches are constants at the top of the script: `controller_chosen`
  (1 = robust PID, 2 = PI-STR), `state_add_dist_aero`, `state_add_dist_wind`.
* `sim/matlab/reduced_overshoot/` holds the Mode I / Mode II attitude-and-altitude
  studies; `Load_SystemParameters.m` there is the source of every physical constant
  used by the Python port — if you change a number, change it in both places and
  say so.

---

## Path 5: 3-D environment (MuJoCo, optional)

The rendered world lives in `sim/python/pistr_sim/env/` and is **optional**: nothing else in
the repository imports it. It needs MuJoCo, which is not a core dependency.

```bash
make env-setup        # pip install --target build/mjsim -r requirements-3d.txt
make test-3d          # gates V1-V8; skipped automatically when MuJoCo is absent
make demo-3d          # renders the demo segments into results/env/
```

Manual control over the demo (everything is in `docs/3D_ENVIRONMENT.md`):

```bash
export PYTHONPATH=$PWD/sim/python:$PWD/build/mjsim

python3 -m pistr_sim.env.cli world --out results/env --width 1280 --height 720
python3 -m pistr_sim.env.cli demo  --out results/env \
        --segments hover step gust drop multi --t-end 5 --width 1280 --height 720
python3 -m pistr_sim.env.cli demo  --no-video --segments hover   # numbers, no pixels
python3 -m pistr_sim.env.cli list
```

Useful knobs: `--controller {pi_str,pid,str,cascade_pid}`, `--seed`, `--extent`,
`--buildings/--trees/--obstacles`, `--terrain flat|hills`, `--camera-distance`,
`--camera-height`, `--wind-markers`, `--no-sky`, `--no-scene`, `--shadow-size`.
Outputs: one `*.mp4` per segment plus `flight_demo.mp4` (concatenated), a poster `*.png`
per segment, and `demo_summary.{json,txt}` with the measured numbers.

Scripting against it is the same objects the CLI uses:

```python
from pistr_sim.env import build_model, MuJoCoPlant, SceneSpec, run_flight

model = build_model(params, scene=SceneSpec(seed=4, extent=120, clearance=10))
trace = run_flight(controller="pi_str", scenario="hover", scene=SceneSpec(), t_end=5.0)
print(trace.summary())
```

The controller stack is the *same* code path as Path 1: `MuJoCoPlant` duck-types
`QuadrotorPlant` (`step`/`sense`/`reset`/`trim_hover`), so `experiments.controllers`,
`estimator` and the scenarios work unchanged. What differs is the world: MuJoCo integrates
the rigid body and resolves contact, so the trace is a second opinion, not a bit-exact
replay (see `docs/3D_ENVIRONMENT.md` §7).

---

## Repository conventions

**Line endings.** `.gitattributes` sets `* text=auto`, so new text files are stored
with LF. The legacy trees — `firmware/Quadrotor_Mode_I_Arduino_Due/`,
`firmware/Quadrotor_Mode_II_Arduino_Due/`, `firmware/libraries/`,
`firmware/readme.txt` and `sim/matlab/` — are marked `-text` instead: their bytes are
**locked exactly as the 2018 project left them** (CRLF, a few files mixed), so that
editing one line there cannot turn into a whole-file diff. Do not renormalise them
inside a normal change. If you ever do want to normalise, do it as a dedicated,
review-only commit:

```bash
git add --renormalize .
git commit -m "Normalise line endings"
```

**Generated output vs. tracked figures.** `results/` is scratch space: each run
writes to its own `results/<run>/` directory (ignored — CSVs, `summary.json` and PNGs
are regenerated on demand). The one tracked artefact is the top-level
`results/metrics.csv`, the evidence table produced by the experiment matrix; it is
explicitly un-ignored, while `results/*.png` stays ignored. Curated figures for the
README and the docs go to `docs/figures/*.png`, which **is** tracked — copy the
handful worth keeping rather than pointing documentation at `results/`.
`results/.gitkeep` exists only to keep the directory in the tree.

## Troubleshooting

| Error | Cause / fix |
|---|---|
| `No module named pistr_sim` | you are not in `sim/python` (or `sim/python` is not on `PYTHONPATH`) |
| `No module named pytest` | `python3 -m pip install pytest`, or use a venv that has it |
| `CMakeLists.txt does not exist yet` from `make test-cpp` | the host C++ core has not landed in your checkout — see `docs/architecture.md` §8 |
| `ctest`: "could not find a test file" | configure step did not run; delete `build/` and re-run `make test-cpp-ctest` |
| `ctest`: "Subprocess killed" | the sandbox refuses to exec a freshly linked binary — run the `ctypes` entry instead (Path 2) |
| `ctypes.ArgumentError` / `OSError: dlopen ... not found` | wrong library path or platform suffix; it is `build/libstr_core_verify.dylib`, `.so` or `str_core_verify.dll` |
| CLI: `unrecognized arguments: --plant` | old checkout; the engine switch exists only in the current CLI (`python3 -m pistr_sim.cli run --help`) |
| `pio: command not found` | PlatformIO is not installed (`python3 -m pip install platformio`) |
| `No module named mujoco` | the 3-D world is optional: `make env-setup`, then put `build/mjsim` on `PYTHONPATH` |
| MuJoCo: "off-screen rendering could not start a GL context" | no usable GL backend; on Linux set `MUJOCO_GL=egl` (or `osmesa`), on macOS this needs a CGL-capable session. Physics still works without it |
| `ffmpeg not found` / mp4 test skipped | install ffmpeg; only the video writer needs it, frames/PNG do not |
| 3-D world: vehicle not visible, scene looks like a wall | a giant sky/ground geom can push MuJoCo's auto near plane (`frustum_near` grows with the scene extent) and clip the vehicle — see `docs/3D_ENVIRONMENT.md` §5 V6 and keep `model.stat.extent` small |
| `make: command not found` | use the explicit commands above; Xcode command line tools provide `make` on macOS |
| MATLAB: `Undefined function 'tf'` | Control System Toolbox is missing |

Exit codes are meaningful: `make test-*` exits non-zero on the first failure, and
the guards print why a target cannot run yet instead of failing silently.
