# 3-D environment specification (MuJoCo)

Status: **new capability, under construction.** This document is the contract between the
four work streams that build `sim/python/pistr_sim/env/`. It is normative: if an
implementation disagrees with this file, either the file is wrong (fix it, say why) or the
implementation is wrong.

Why this exists: the rest of the simulator is a *headless time-series integrator*. It has a
physically detailed 6-DOF model but **no world** — no ground contact, no scene, no camera,
no pixels. `docs/figures/trajectory_3d.png` is a matplotlib line plot, not a 3-D
environment. This document specifies the missing piece.

---

## 1. Scope

In scope:

* A MuJoCo world containing a ground plane, static obstacles (buildings, trees, crates),
  lighting/sky, and an optional wind field whose effect on the vehicle is physical (the
  visualisation is decoration; the force is real).
* A quadrotor model whose **mass, inertia tensor, arm geometry, rotor radius, spin
  directions and aerodynamic coefficients are imported from `pistr_sim.params`** — not
  retyped, so it cannot drift from the rest of the repository.
* Ground contact and rigid-body contact (the paper plant has neither).
* A camera system (chase / orbit / fixed) and off-screen rendering to PNG frames + MP4.
* A plant adapter that presents the **same control-facing interface as
  `pistr_sim.plant.QuadrotorPlant`**, so the existing controllers, estimators and scenario
  code drive the 3-D world unchanged.
* Multi-vehicle support (N independent quadrotors in one world).
* A recorded demonstration: hover, a roll step, a wind gust, a ground impact / drop.

Out of scope (documented, not silently attempted): photorealistic rendering, GPU
path-tracing, ROS/PX4 SITL, sensor simulation inside MuJoCo (we reuse
`pistr_sim.estimator`), blade flapping / flexible structures, hardware-in-the-loop.

## 2. Frames, units and the one mapping that matters

MuJoCo is **z-up, right-handed, metres, kilograms, seconds** with quaternions ordered
`(w, x, y, z)`. The rest of this repository is **NED-like**: inertial `z` points *down*
(`xi[2]` is negative altitude), the body `z` axis points *down* (thrust is along body `-z`),
and `pistr_sim.types.quat_to_rotmat` returns the inertial→body rotation in the MATLAB
element order.

The bridge converts with a single constant

```
S = diag(1, -1, -1)              # applies to positions AND to body-frame vectors
p_mj   = S @ p_ned = (x_n, -y_e, -z_d)      # position
R_mj   = S @ R_ned.T @ S                    # attitude (R_ned is inertial->body)
q_mj   = (w, x, -y, -z)                     # same mapping in quaternion form
```

Derivation: FLU (x forward, y left, z up) and NED (x forward, y right, z down) share the
forward axis, so the conversion is a **rotation** about x by 180°, and it must be applied
to world positions and body coordinates alike. Both components carry a minus sign on
`y` and `z`; using `(x, +y, -z)` for position while rotating attitude by `S` is a
**mirror** (`det = -1`), and the mismatch makes position feedback a y-axis positive
feedback loop — invisible in hover or pure attitude control, divergent as soon as an
outer/position loop closes. This was caught by M4 during implementation and corrected
here; the mapping above is the only self-consistent choice.

Consequently, a NED `roll > 0` (right wing down, thrust tilts toward NED `+y` = east)
appears in MuJoCo as a tilt toward **`-y`**, because MuJoCo's `+y` is west. The physics is
identical; only the coordinate label flips. **V3 below is a required gate, not a
nicety.**

Units are SI everywhere in the new code. PWM stays in microseconds at the interface
(`PWM_MIN=1000`, `PWM_MAX=2000`) because every controller already speaks that.

## 3. Module map and ownership

| Path | Owner | Contents |
|---|---|---|
| `env/mjcf.py` | M1 model | `build_xml()`, `build_model()`: vehicle `Params` → MJCF |
| `env/actuation.py` | M1 model | rotor thrust/torque + ESC lag + motor first-order lag → body wrench |
| `env/scene.py` | M2 scene | `SceneSpec` → MJCF fragments (ground, buildings, trees, obstacles, sky, wind markers) |
| `env/render.py` | M3 render | off-screen `Renderer`, frame capture, `write_mp4()` |
| `env/camera.py` | M3 render | `make_camera()` chase/orbit/fixed rigs |
| `env/bridge.py` | M4 bridge | `MuJoCoPlant`: QuadrotorPlant-compatible adapter |
| `env/loop.py` | M4 bridge | closed-loop runner + trace, multi-vehicle orchestration |
| `env/__init__.py` | lead | lazy re-exports; `mujoco` must stay an optional dependency |
| `docs/3D_ENVIRONMENT.md` | lead | this file |
| `tests/test_env_*.py` | each owner | see §5 gates |

Rule: **no file is edited by two work streams.** If a contract needs to change, the lead
edits this document and tells everyone.

## 4. API contract

```python
# env/scene.py
@dataclass(frozen=True)
class SceneSpec:
    ground: bool = True
    terrain: str = "flat"        # "flat" | "hills"
    buildings: int = 6
    trees: int = 10
    obstacles: int = 8           # crates/poles inside the flight volume
    sky: bool = True
    wind_markers: bool = False   # visual-only arrows, no collision, no force
    seed: int = 0
    extent: float = 60.0         # world half-size, m

def scene_xml(spec: SceneSpec, assets: dict | None = None) -> str: ...

# env/mjcf.py
def build_xml(params, *, scene=None, n_drones=1, layout=None) -> str: ...
def build_model(params, *, scene=None, n_drones=1) -> "mujoco.MjModel": ...

# env/actuation.py
class RotorPack:
    """Rotor states for one vehicle: omega (4,), esc duty (4,), battery SoC."""
    def reset(self, *, pwm_us=None, soc=1.0) -> None: ...
    def step(self, pwm_us, dt, *, v_rel_body=None) -> tuple[np.ndarray, np.ndarray]:
        """Advance the actuator chain `dt` seconds; return (thrust, torque_z) in N/Nm."""

# env/bridge.py
class MuJoCoPlant:
    """Duck-types pistr_sim.plant.QuadrotorPlant for the control stack."""
    dt: float
    state: VehicleState              # converted from MuJoCo at every step
    def reset(self, state=None, *, wind_seed=None) -> VehicleState: ...
    def step(self, pwm_us) -> VehicleState: ...
    def sense(self) -> SensorSample: ...
    def set_wind(self, steady, turbulence=True) -> None: ...
    def trim_hover(self, soc=1.0) -> dict: ...   # same keys as QuadrotorPlant
    def evaluate(self, state=None) -> dict: ...

# env/loop.py
def run_flight(*, controller="pi_str", scenario="hover", scene=SceneSpec(),
               t_end=8.0, dt=0.002, control_h=None, seed=0,   # None -> params.control.h
               n_drones=1, camera=None, out=None, video=None) -> "FlightTrace"
```

`MjModel` construction must not require a GL context; only rendering does. Scene and model
modules must therefore be importable and testable headless.

## 5. Validation gates (all automated, all in `tests/`)

| # | Gate | Where |
|---|---|---|
| V1 | `build_model()` succeeds and the unactuated vehicle does not explode in 1 s | `test_env_model.py` |
| V2 | Mass equals `params.airframe.mass` to 1e-9; the inertia tensor equals `params.airframe.inertia` with **only** the `Iyz` entry zeroed, to 1e-9, and `Iyz == 0` is asserted explicitly (see §7) | `test_env_model.py` |
| V3 | Frame mapping: level hover stays level; NED `roll>0` tilts the body to MuJoCo **`-y`** and the thrust axis equals `S @ thrust_ned`; round-tripping through `quat_to_euler` restores the NED yaw | `test_env_bridge.py` |
| V4 | With `RotorPack`, total hover thrust ≈ `m g` (1 %) and the vehicle holds altitude for 2 s (< 0.15 m drift) | `test_env_model.py` |
| V5 | `MuJoCoPlant.step()` is a drop-in for `QuadrotorPlant.step()` on a fixed PWM sequence (same shapes, same units, no NaN) | `test_env_bridge.py` |
| V6 | Contact: an unpowered drop from 1.5 m comes to rest on the ground, `z >= 0` and no tunnel-through | `test_env_scene.py` |
| V7 | Render one 320×240 frame headless to a non-uniform image, and `write_mp4()` produces a playable file with the expected frame count | `test_env_render.py` |
| V8 | Cross-check, **multi-seed**: over ≥3 sensor seeds, the mean attitude RMSE of PI-STR in the world is ≤ 2× the same loop on `fidelity="high"`, and the per-seed table is printed. A single 2 s window is not a valid pass/fail unit — with no position aiding, gyro random walk puts one seed on either side of the lightly damped mode (seed 0 favours the plant, seed 4 the world). | `test_env_crosscheck.py` |

Every gate must be able to run in this sandbox: `PYTHONPATH=sim/python:build/mjsim`.
Tests skip cleanly (`pytest.importorskip("mujoco")`) when MuJoCo is absent, so `make test`
keeps passing on a bare checkout.

## 6. Dependencies and how to run

MuJoCo is an **optional** dependency (`requirements-3d.txt`, `make env-setup`). This sandbox
has it unpacked at `build/mjsim`:

```bash
export PYTHONPATH=$PWD/sim/python:$PWD/build/mjsim
python3 -m pistr_sim.env.cli demo --out results/env
```

Rendering uses MuJoCo's off-screen CGL context on macOS (verified on this machine: 640×480
frame, no `DISPLAY`); on Linux the same code path uses EGL. No GUI is ever required.

## 7. Honest limitations (to be repeated in the README, not hidden)

* The 3-D world is a **different rigid-body integrator and contact model** from
  `plant.py`. Cross-checking means "do the two disagree in a way that changes the
  conclusion", not "the traces are identical".
* MuJoCo's contacts are soft by default; ground impact is qualitative, not structural.
* Aerodynamics in the world are the rotor/body model ported from `plant.py`; MuJoCo's own
  fluid model is not used.
* Rendering is rasterised and shadowed, not photorealistic; it is evidence of geometry and
  motion, not of appearance.
* **The actuator chain is a reduced model.** `RotorPack` drops the motor electrical
  inductance pole (`L dI/dt` in `plant.py`, tau_e = L/R = 0.4 ms) and keeps the ESC lag and
  the algebraic back-EMF/battery coupling. Consequence: the open-loop roll response is about
  17 % faster than the high-fidelity plant (0.05 s roll step: 0.00260 rad in the world vs
  0.00222 rad in `plant.py`); hover trim is unaffected (both 1394 µs). This is a difference in
  the plant, not a bug, and it is the reason the world is a second opinion rather than a
  replay.
* **The world does not model ground effect.** The `1/(1-(R/4z)²)` thrust augmentation in
  `plant.py` is absent, which only matters within about a rotor diameter of the ground; the
  5 m hover used by V8 is unaffected.
* **The published inertia tensor is not physically realisable.** `plant.py` preserves the
  MATLAB tensor verbatim, which sets `Iyz = Ixz`; its principal moments are
  `(0.00924387, 0.00949488, 0.01895122)`, so `I1 + I2 − I3 = −2.1e-4 < 0` and MuJoCo
  rejects it (a rigid body must satisfy the triangle inequality). The 3-D model therefore
  zeroes `Iyz` — the documented asymmetry bug, not a tuning parameter. The diagonal entries
  then match the plant **exactly** and the triangle inequality holds with a 2.9e-6 margin
  (verified: the model compiles). Any rotational-inertia difference between the two
  simulators is confined to that one coupling term.

## 8. Implementation status and measured numbers

All gates below were run on this checkout with `PYTHONPATH=sim/python:build/mjsim`
(MuJoCo 4.0.0, macOS arm64, no display). Numbers are printed by the tests themselves.

| Gate | Measured |
|---|---|
| V1 | model builds head-less; an unpowered drop from 1.5 m reaches 5.17 m/s, comes to rest on its legs at `z = 0.13996 m`, `|v| = 2.8e-8` |
| V2 | mass 0.985 kg, deviation **0**; inertia equals `params.airframe.inertia` with `Iyz` zeroed to **3.3e-14**; `Iyz == 0` asserted; triangle margin **+2.9e-6** |
| V3 | 50 random attitudes: NED↔MuJoCo round trip **2.2e-16**; `R_mj` vs `S·R_nedᵀ·S` **5.6e-16** (scipy cross-check 3.3e-16); `roll=+0.1` thrust axis equals `S·thrust_ned` exactly; yaw sign round-trips |
| V4 | `trim_hover` 1394.0 µs; thrust 9.6787 N vs `mg = 9.6629 N` (**+0.16 %**); 2 s hover drift **2.48 cm** |
| V5 | trim PWM identical to `plant.py` (1394.00 vs 1394.00 µs); free-fall, roll and yaw differentials agree in magnitude and sign; clean `sense()` at hover gives `accel ≈ -g` |
| V6 | dropped body rests at `z = 0.03989 m` with 4 contacts; scene XML byte-identical across processes for the same seed; wind markers change physics by **0** |
| V7 | headless frame std **52.9**, 1509 distinct colours; `write_mp4` file probes as 30 frames / 1.000 s / H.264 / yuv420p |
| V8 | 5 seeds, 5 m hover, 2 s, control `h = 40 ms`, physics 2 ms: mean attitude RMSE ratio world/plant **roll 1.00×, pitch 0.36×, yaw 1.06×**; mean altitude drift **0.145 m vs 0.307 m**; no divergence. Clean sensors: world drift **5.7 mm / 2 s** |

Test suite size: **179 passed** with MuJoCo on `PYTHONPATH` (`make test-3d` runs the 99
environment-related ones); without it, the same command gives **110 passed, 17 skipped** —
the env package skips cleanly rather than breaking a checkout.

Two implementation decisions worth knowing:

* **The camera is automatically framed** by `run_flight` when the rig exposes
  `set_target` (it targets drone 0 in MuJoCo coordinates). Pass `camera=None` and drive the
  rig from `on_frame` if you want a different policy — that is what `env.cli` does, so it can
  frame the centroid of a multi-vehicle formation.
* **`control_h` defaults to `params.control.h` (40 ms)**, not to a fast loop rate: the STR
  identifies a discrete model at that sampling period, so feeding it at 4 ms silently
  invalidates it. An explicit conflicting value raises a `RuntimeWarning` for STR-family
  controllers.

## 9. The demo

`make demo-3d` renders five five-second segments at 1280×720 / 25 fps (one frame per control
tick, so playback is real time) and concatenates them:

| Segment | What it shows |
|---|---|
| `hover` | PI-STR holding 5 m in the scene; ground, buildings, trees, shadows |
| `step` | 0.15 rad roll step at t = 1.5 s, chase camera following heading only |
| `gust` | 3 m/s gust from t = 1 s to the end; the loop rejects it without losing altitude |
| `drop` | motors cut at 1.5 s: free fall at **8.3 m/s**, ground contact, rest on the legs at 0.14 m |
| `multi` | three independent vehicles, each with its own rotor pack and PI-STR loop |

The rendered file is 1280×720, 25.160 s, 629 frames, 1.6 MB (`results/env/flight_demo_720p.mp4`);
a poster frame is kept at [`docs/figures/env_flight.png`](figures/env_flight.png). Rendering
costs 1.7–3.0 s of wall time per 5 s segment on this machine, so the whole demo is under a
minute.
