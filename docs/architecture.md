# Architecture

This document explains **how the repository is put together** and why. It is not
normative for the algorithms — [`CONTROL_SPEC.md`](CONTROL_SPEC.md) and
[`PLANT_MODEL.md`](PLANT_MODEL.md) are. When this document disagrees with those,
they win; when it disagrees with the code, one of the two is a bug.

## 1. The problem the layout solves

The paper's contribution is a *discrete-time adaptive inner loop*: recursive least
squares identifies an ARX model of one rate channel, and pole placement retunes the
controller from that model every tick. Three very different consumers of that same
control law exist here:

1. the **2018 flight firmware** (Arduino Due) that actually flew;
2. the **MATLAB study** that produced the published figures;
3. the **Python simulator** added later, which must be usable without MATLAB and
   without hardware.

Keeping one algorithm correct across three implementations is the central
engineering constraint of this repository. Everything below follows from it:

| Rule | Why |
|---|---|
| `docs/CONTROL_SPEC.md` is the single source of truth | three ports, one specification |
| Python is the **numerical oracle** (no MATLAB, no Arduino here) | it is the only side that runs on a plain host |
| the C++ core is **host-compiled, zero Arduino dependency** | so it can be unit-tested on a laptop |
| `.ino` changes are mechanical and hand-reviewed | they cannot be compiled here at all |
| the plant is never importable from a controller | otherwise the "test" is a mirror, not a test |

## 2. Repository map

```
PI-STR-for-Quadrotor/
├── firmware/                       flight code (2018, as published)
│   ├── Quadrotor_Mode_I_Arduino_Due/     single-loop STR, 3rd-order channel
│   ├── Quadrotor_Mode_II_Arduino_Due/    PI outer + STR inner, 2nd-order channel
│   ├── libraries/MyLibraries_Wenchao/    IMU/sensor drivers, attitude Kalman, legacy STR class
│   ├── libraries/SdFat/                  vendored third-party SD library (2015.4.26)
│   └── lib/str_core/                     host-testable STR core, C++17, no Arduino
├── sim/
│   ├── matlab/                     published study (R2014a), kept as-is
│   │   ├── reduced_overshoot/      Mode I / Mode II attitude+altitude tracking
│   │   └── robustness/             PID vs PI-STR over a 3-D trajectory
│   └── python/                     modern simulator, the default entry point
│       ├── pistr_sim/              the package
│       ├── tests/                  pytest suite
│       └── pyproject.toml          deps + pytest configuration
├── tests/
│   ├── cpp/                        C++ unit tests for firmware/lib/str_core
│   └── golden/                     cross-language fixture (CSV, schema in its README)
├── docs/                           specs and guides (this file, quickstart, licensing)
├── CMakeLists.txt                  host build: str_core + tests/cpp
├── Makefile                        test-cpp / test-py / sim / clean
├── platformio.ini                  two envs, one per sketch (unverified, see file header)
├── CITATION.cff                    paper metadata for GitHub's "Cite this repository"
├── .gitignore / .gitattributes
└── docs/CONTROL_SPEC.md, docs/PLANT_MODEL.md
```

## 3. Python simulator: module boundaries

```
             params.py  ─────────── numbers (ported from Load_SystemParameters.m)
                 │
                 │  consumed by everything, mutated by nothing (frozen dataclasses)
                 ▼
  reference.py ──► control/ ──► plant.py ──► sensors ──► estimator.py ──► control/
  (setpoints)      (u, PWM)     (truth)      (noisy)     (Estimate)
       ▲              │            │                        │
       └──────────────┴─── cli.py ─┴──────── metrics.py ◄───┘
                        (wiring, CSV+PNG output)
                                        types.py  ─── the only shared vocabulary
```

Contracts, in dependency order:

| Module | Owns | May import | Must never import |
|---|---|---|---|
| `types.py` | `VehicleState`, `SensorSample`, `Estimate`, `Plant` protocol, quaternion helpers, PWM constants | numpy only | plant, control |
| `params.py` | every physical/controller constant, `fidelity` switch | `types.py` | plant, control |
| `plant.py` | physics: `reset/step/sense/set_wind` at one or both fidelity levels | `types.py`, `params.py` | control, cli |
| `control/` | PID, STR (`str.py`), PI-STR cascade (`pi_str.py`) | `types.py`, `params.py` | plant |
| `estimator.py` | `SensorSample → Estimate` (per-axis 2-state Kalman, from `IMU.cpp`) | `types.py`, `params.py` | plant internals |
| `reference.py` | hover / sinusoid / 3-D trajectory + ESO | `types.py`, `params.py` | plant, control |
| `metrics.py` | RMSE, overshoot, 2 % settling time, control RMS, attitude deviation | numpy only | plant, control |
| `cli.py` | argument parsing, scenario wiring, output | everything | — |

The rule that matters: **a controller sees `Estimate`, never `VehicleState`.** If a
controller reads truth, every robustness claim in the paper becomes untestable. The
`Plant` protocol in `types.py` (`reset`, `step`, `sense`, `set_wind`) is the only
door between physics and algorithms (`PLANT_MODEL.md` §9).

Two fidelity levels live behind that one door and are selected by
`Params.fidelity`:

* `"paper"` — verbatim MATLAB port (Euler, `U = Umax*sqrt(PWM%)`, `T = kT*omega^2`),
  for reproduction and parity;
* `"high"` — BEMT rotor model, H-force, body drag, gyroscopic moments, ESC lag,
  battery sag, Dryden turbulence, sensor corruption, RK4 (`PLANT_MODEL.md` §2–§7).

The controller side is deliberately **not** switched by fidelity: the STR keeps
identifying the same 2nd-order ARX channel in both cases. A plant that violates the
ARX assumption is the whole point of the high-fidelity level.

## 4. Firmware: from one sketch to one translation unit

Each `firmware/Quadrotor_Mode_*/` directory is **one Arduino program**. The Arduino
toolchain concatenates all `.ino` files of a sketch folder into a single `.cpp`
(the `*.ino` named after the folder first, then the rest alphabetically) and
generates prototypes. Consequences:

* a function defined in `Controller_STR.ino` is visible from `Motor.ino`;
* there are no modules, only sections — a "new file" is a new section of one program;
* `Config.h` is the only real header, and it is where the program is configured.

```
Quadrotor_Mode_II_Arduino_Due/
├── Quadrotor_Mode_II_Arduino_Due.ino   setup() + loop() + scheduler, global state
├── Config.h                            compile-time switches: QuadP, STR_v1, PID_v1,
│                                       LOOP_period (10 ms), STR_period (40 ms), pins, limits
├── Driver_*.ino                        motors (Servo), receiver (PWM in), LED, SD card
├── Controller_Main.ino                 arming state, sample/command sequencing
├── Controller_STR.ino                  PI outer loop + call into the STR
├── Controller_PID.ino                  PID baseline
└── Debug_Serial.ino                    serial log

          calls into                        provided by
   ┌───────────────────────┐        ┌───────────────────────────────┐
   │ .ino sections         │───────►│ firmware/lib/str_core/        │  host-testable C++17
   │ (one TU, Arduino API) │        │ RLS · projection · Diophantine│  no Arduino includes,
   └───────────────────────┘        │ · guards · saturation         │  compiled by CMake
             │                     └───────────────────────────────┘
             │ includes                     ▲
             ▼                              │ tests/cpp/
   firmware/libraries/MyLibraries_Wenchao/   └── ctest
   (IMU.cpp: sensor drivers + attitude Kalman; legacy STR.cpp: deprecated)
```

Mode I differs structurally: no PI outer loop, the STR runs directly on the angle
reference with a **3rd-order** model (`STR_period = 100 ms`, λ = 0.98, P0 = 100·I),
and the historical implementation inlined a hand-expanded 5×5 adjugate. Mode II is
the paper's cascade (`LOOP_period = 10 ms`, `STR_period = 40 ms`, λ = 0.99,
P0 = 5·I), 2nd-order model.

The split between `str_core` and the sketch is the key boundary:

* **in `str_core`**: anything with a formula (RLS update, parameter projection,
  Diophantine solve, guards, saturation). Testable, reviewed, both `float` and
  `double` instantiable.
* **in the sketch**: pin configuration, arming logic, PWM mixing, sensor plumbing,
  and the physical-to-differential command conversion (`u_actual`, see
  `CONTROL_SPEC.md` §5). Reviewed by hand, not testable here.

`CONTROL_SPEC.md` §9 lists the known defects of the legacy inline implementations
(including the `s0/s1` index bug) — those are the reason for the boundary, not an
accident of style.

## 5. Cross-language conformance

The Python and C++ implementations are kept aligned by a fixture, not by hope:

* `tests/golden/` holds a plain CSV trace (one row per controller tick) generated
  from the normative scenario of `CONTROL_SPEC.md` §8;
* Python replays it in `sim/python/tests/test_golden_conformance.py`;
* C++ replays it in `tests/cpp/test_golden.cpp` through `str_core`, and again through
  `str_core_check_golden()` from the shared-library entry point;
* acceptance is the fixture's own `# tolerance_rel=` value applied to
  `max|Δ| / max|ref|` per column. The shipped fixture runs at **`tol = 1e-9`** (the
  `RESULT: PASS (tol=1e-09, guard_flags=0x0)` line printed by both C++ runners), well
  inside the `< 1e-6` headline figure of `CONTROL_SPEC.md` §8; the runner uses the
  fixture's tolerance unless one is passed explicitly.

The plant used to generate `y` is the same trivial ARX model on both sides, so the
test isolates the *controller*. Schema and provenance: `tests/golden/README.md`.
A stale fixture is worse than no fixture — do not regenerate it inside a feature
branch.

## 6. What runs where

| Component | Build/run command | Runs on a plain host? | Verifiable on this machine? |
|---|---|---|---|
| Python package + tests | `make test-py` (`python3 -m pytest sim/python/tests`) | yes | yes |
| C++ core + tests | `make test-cpp` (cmake build, then the ctypes entry) / `make test-cpp-ctest` | yes | yes (clang++/g++) |
| Firmware `.ino` | `pio run -e due_mode_ii` | **no** | **no** — no PlatformIO / SAM toolchain |
| MATLAB study | manual, inside MATLAB | no | **no** — no MATLAB, no Control System Toolbox |

Two C++ entry points exist because a restricted environment may refuse to `execve` a
freshly linked binary while `dlopen()` of the same code still works:

| Entry point | Runs | Target |
|---|---|---|
| `str_core_run_tests` (C ABI, called via `ctypes`) | the suites in-process, prints `str_core_run_tests("all") -> PASS` | `make test-cpp` |
| `ctest` | the same suites, one line per suite | `make test-cpp-ctest` |

Both link `str_core_testlib`, so the assertions are identical; only the process model
and the reporting differ. The check count follows the selected suite —
`make SUITE=guards test-cpp` reports 72 checks, `SUITE=all` about 159 000.

Anything in the last two rows of the table is reviewed by reading it, and the
documentation says so rather than implying coverage that does not exist.

No CI workflow is checked in yet. If one is added, the first two rows are the only
jobs that can be gated automatically; the other two need dedicated runners (an
Arduino/SAM toolchain, and a MATLAB licence) that this repository does not have.

## 7. Documentation map

| Document | Authority | Audience |
|---|---|---|
| [`CONTROL_SPEC.md`](CONTROL_SPEC.md) | **normative** for all controller code | anyone touching an algorithm |
| [`PLANT_MODEL.md`](PLANT_MODEL.md) | **normative** for the high-fidelity plant | anyone touching `plant.py` |
| [`quickstart.md`](quickstart.md) | operational | new users |
| this file | descriptive | maintainers |
| [`licensing.md`](licensing.md) | legal status, not legal advice | anyone redistributing |
| `tests/golden/README.md` | fixture schema | anyone regenerating fixtures |
| `../README.md` | entry point | everyone |

Precedence, when two documents disagree: spec > this file > README prose. Code
never overrides a spec without either fixing the code or amending the spec in the
same change, and §9 of `CONTROL_SPEC.md` is where deliberate deviations are
recorded.

## 8. Status

Everything this document describes has landed: the normative specs, the Python
simulator (`plant.py`, `control/`, `estimator.py`, `reference.py`, `metrics.py`,
`cli.py`) with its pytest suite, the host C++ core with both test entry points, the
golden fixture, and the docs/tooling. The README's
[Status, honestly](../README.md#status-honestly) table is the current user-facing
statement of what is and is not validated; this section only maps parts to the
command that exercises them.

| Part | Exercised by |
|---|---|
| Python simulator + control stack | `make test-py`, `make sim` |
| Host C++ core (`firmware/lib/str_core/`) | `make test-cpp` (ctypes) · `make test-cpp-ctest` |
| Cross-language conformance | the golden suite inside both C++ runners (`tests/golden/`) |
| Firmware sketches | **nothing** — `pio run` has never been executed (no SAM toolchain); hand review only |
| MATLAB study | **nothing** — no MATLAB or Control System Toolbox here |

Each `make` target still checks its inputs and names what is missing instead of
failing cryptically, so a partial checkout stays diagnosable.

## 9. Recipes

**Add a controller.** Drop a module in `sim/python/pistr_sim/control/` that maps
`Estimate + parameter dict → PWM differential` per axis, add its name to the CLI's
controller choices, and add a test under `sim/python/tests/`. Do not import the
plant; if you need truth to make it work, the controller is wrong.

**Add a scenario.** Extend `reference.py` (setpoint generation) and the CLI's
scenario table; keep the plant untouched. If a scenario needs new physics, it
belongs in `plant.py` behind a flag listed in `PLANT_MODEL.md` §8 so the P12
equivalence test stays meaningful.

**Change the STR.** Change it in Python, prove it with the Python tests, change
`str_core` identically, prove it with `make test-cpp` (the ctypes entry; run
`make test-cpp-ctest` too if your machine can execute new binaries) and the golden
vector, then (and only then) update the `.ino` call sites. Number formats matter: `str_core` has a `float`
instantiation for the Due and a `double` one for host tests, and the guards in
`CONTROL_SPEC.md` §6 must behave the same in both.

**Add a second airframe.** `params.py` is frozen and flat by design: add a
`Params.from_matlab`-style constructor that returns a different `Airframe` instead of
branching inside the plant.
