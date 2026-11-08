# Firmware safety notes (stage-A audit)

Scope: the two Arduino Due sketches in this directory (`Quadrotor_Mode_I_Arduino_Due`,
`Quadrotor_Mode_II_Arduino_Due`) and the shared regulator core in `lib/str_core/`.

**Read this before flying anything.** Nothing in this repository has been validated on a
real aircraft, and no `.ino` file can even be compiled in this development environment
(there is no AVR/Arduino toolchain). The firmware changes are mechanical integrations,
reviewed by hand and syntax-checked against stub headers — not executed.

---

## 1. Fixed in stage A

| # | Problem | Fix |
|---|---------|-----|
| 1 | `STR.cpp` read `s0 = xx[2]; s1 = xx[3];` from a 3-element solution vector: out-of-range read, and `s0`/`s1` swapped (`CONTROL_SPEC` §5 requires `x = [r1, s0, s1]`, i.e. `s0 = x[1]`, `s1 = x[2]`). | The Diophantine system is solved by Gaussian elimination with partial pivoting in `str_core`; the mapping is `r = x[0..na-2]`, `s = x[na-1..]`. Old implementation deleted. |
| 2 | `beta = (1 + am1 + am2)/(b0 + b1)` and `detAA_inv = 1/(...)` with no division guard. | §6.2 guards: `\|B(1)\| < 1e-6` or `\|det\| < 1e-9` → gains frozen (last valid solve kept), never a singular solution. |
| 3 | `theta_hat` had no projection or bounds; only `P` was clamped. | §6.1 projection after every RLS update (box + 2nd-order stability triangle + `\|B(1)\| >= BSUM_MIN`). |
| 4 | No NaN/Inf handling: a bad sample went straight to the mixer. | §6.3: non-finite input/state/command → restore the last valid `theta`/`P`, count consecutive failures, and at 50 latch **safe mode** (zero differential command) until `resetdata()`. |
| 5 | `DEBUG` forced `PWM_out[] = PWM_MIN` while the ARM state machine still reported *armed* — the "safe mode" was a side effect of a debug switch. | `DEBUG` now only controls serial/timing output. New explicit interlock `MOTOR_ENABLE` / bench switch `MOTOR_TEST` (see `Config.h`); when interlocked, `ARM_flag` is forced to 0 so the LED and the SD log can never claim "armed". |
| 5b | **Mode II initial parameter guess had a sign error and a millisecond/second slip** (initialisation only, not the control law): `a2 = exp(+atemp*h)` with `atemp = 54` gives `a2 = 8.6711`, and `* STR_period` (40 ms) instead of `h = 0.04 s` gives `b0 = 47.58`, `b1 = -411.24` → `theta0 = [-9.6711, +8.6711, +47.5758, -411.2445]`: an unstable, non-minimum-phase starting model with a negative DC gain, outside every §6.1 limit. | Repaired to the §3 analytic form with the exponent sign and the seconds unit fixed: **`theta0 = [-1.1153, +0.1153, +0.0280, +0.01395]`** (pole radius 0.115, `B(1) = 0.0419`, exactly on the marginal integrator boundary `a1 + a2 = -1`, so it is a fixed point of the projection). Constants `atemp = 54`, `btemp = 64` are the sketch's own and are unchanged. |
| 5d | **Mode II passed `STR_period / 1000` (integer division -> 0) to the reference model.** With `h = 0`, `Am(z) = 1 - 2z^-1 + z^-2` and the reference gain `t0 = Am(1)/B(1)` is exactly 0, i.e. the inner loop had **no reference feedforward** — the outer PI output never reached the plant. | Fixed to `(float)STR_period / 1000.0f` = 0.04 s, the same `h` as the paper's simulation and the Python stack. Behaviour change; real-time behaviour is **not** verified on hardware. |
| 5e | **Mode I: `y_desire[][]` was never written**, so the STR reference channel was dead and the selected axis regulated the measured angle to 0 deg instead of following the stick. | `Angle_desire[Index]` is now shifted into `y_desire[Index][]` and fed to `begin_tick()`; `str_core` owns the history. Behaviour change; needs a flight test. |
| 5c | **Mode I: `PID_out` was used uninitialized.** `Config.h` leaves every `PID_v1/v2/v3` commented out, so the `#if/#elif` chain in `PID_controller()` declared `PID_out` and never assigned it — yet `Controller()` calls it for the two non-STR axes. Undefined behaviour on the roll/pitch feedback path. | Initialised to 0, i.e. deterministic "no feedback on the non-STR axes". **This needs an owner decision:** the ship-ready fix is to enable the intended PID variant in `Config.h`; doing that here would be a control-law change and was not attempted. |

## 2. Fixed after the Lead's audit pass

| # | Problem | Fix |
|---|---------|-----|
| 6 | `Driver_SDcard.ino` did `open → String → println → close` on **every** 100 Hz tick: tens of ms of SD latency per sample (loop overrun), ~10 heap allocations per sample (fragmentation), and an extra CSV header on every power-up (`O_AT_END`). | File opened once and kept open; header written only when `fileSize() == 0`; rows written straight to the file buffer; `sync()` every 50 samples (0.5 s). CSV columns unchanged. |
| 7 | PID integrators (`Controller_PID.ino`, and the same code in Mode I's `Controller.ino`) kept winding up while the output was clamped. | Integrator clamped to `±FeedBackMax[Index]` in all `PID_v1` / `v2` / `v3` branches (including the unused v3, so it cannot bite later). |
| 8 | The ARM state machine was written twice with different conditions (`loop()` vs `Motor_control()`). | One `Arm_update()` per sketch; legacy conditions merged, behaviour unchanged (arm = both sticks top **from disarmed**; kill = roll bottom + pitch top). |

### Initial parameter guess (after the §3 unification)

`theta0 = 0.75 * [a1, a2, b0, b1]` from §3 with `tf_beta = 47.45112887` and the per-channel
`tf_alpha` of `sim/python/pistr_sim/params.py` (deg / PWM-unit domain, `h = 0.04`):

| channel | `tf_alpha` | `theta0` |
|---|---|---|
| roll  | 1112.27298381 | `[-0.862396, +0.112396, +0.388242, +0.209586]` |
| pitch | 1126.85305913 | `[-0.862396, +0.112396, +0.393331, +0.212333]` |
| yaw   |  114.71504431 | `[-0.862396, +0.112396, +0.040042, +0.021616]` (the shipped `STR_Index`) |

All three sit exactly on the marginal integrator boundary `a1 + a2 = -1`, i.e. they are
fixed points of the §6.1 projection, and they match the Python simulator's `mode_ii()`
guess for the same channel. Mode I still uses its hand-written 6-parameter guess: the
§3 3rd-order discretisation (`analytic_theta0_mode_i`) would need the same treatment and
was left out of this pass.

## 3. Identified, **not** fixed (needs the owner's decision — each one changes flight behaviour)

1. **Receiver loss-of-signal is not detected** (`Driver_Receiver.ino`). Pulse widths are
   captured with interrupts and copied into `RC[]`; if the receiver or the wiring dies,
   `pulse1..4` keep their last value and the controller keeps flying the last command.
   This is a genuine safety gap.
   *Suggested fix:* per channel, store the timestamp of the last valid edge and, if no
   edge arrived for ~500 ms, force `ARM_flag = 0` (and `PWM_out[] = PWM_MIN`) plus a
   status flag. Needs a bench test with a receiver that can be unplugged while armed;
   NOT implemented here because it changes in-flight behaviour and cannot be validated
   in this environment.

2. **Mode II passes `STR_period / 1000` (integer division → 0) to the reference model**
   (`Quadrotor_Mode_II_Arduino_Due.ino`, `set_second_order_reference(..., h_str)` with
   `h_str = (float)(STR_period / 1000)`). With `h = 0`, `Am(z) = 1 - 2z^-1 + z^-2` and the
   reference gain is `t0 = Am(1)/B(1) = 0`, i.e. the inner STR loop has **no reference
   feedforward** and behaves as a rate damper. Preserved verbatim (stage A must not
   change the control law). Suspected bug; needs an owner decision and a flight test.

3. **Mode I `/` Mode II motor-unit mismatch of the identified `b` parameters is unverified.**
   The firmware identifies the channel in PWM-microsecond units, the Python side in SI
   units; both are self-consistent, but the `BSUM_MIN`/guard constants only mean the same
   thing in one of them. §6.1's absolute floor is kept for the 2nd order (golden contract)
   and made relative for the 3rd order (1 % of the initial `|B(1)|`), which is the only
   defensible choice for SI numbers; this has not been checked in flight.

4. **Mode I never writes `y_desire[][ ]`.** The reference term of the Mode I STR is
   identically zero, so the selected axis is regulated to 0 deg instead of following the
   stick. Preserved verbatim; almost certainly not what was intended.

5. **Reference gain uses `Am(1)/B(1)` and omits `Ao(1)`.** That matches the legacy code
   and is exactly §5 for Mode II (`mu = 0` ⇒ `Ao(1) = 1`), but in Mode I
   (`ao1 = 0.8`, `ao2 = 0.16` ⇒ `Ao(1) = 1.96`) the closed-loop DC gain is `1/Ao(1)` ≈ 0.51
   instead of 1. Kept for behaviour equivalence; either the spec or the firmware should
   change, not both silently.

6. **Mode I feeds the pre-saturation design command into the `-ao` terms.** §5/§6.4
   require the *realised* command. `str_core` implements the spec, so Mode I behaves
   identically while unsaturated and differs only under mixer saturation (an intentional
   anti-windup improvement, the one deliberate behaviour change of stage A).

7. **The §4.3 covariance clamp is stricter than the legacy code.** `P` diagonals are
   clamped to `[0, Pmax]`; with `lambda < 1` and `P0 = Pmax` the clamp bites from the
   first tick, which the legacy code never did. The clamp is applied **in place**
   (off-diagonals kept), identical to the Python reference, so the golden vectors agree.
   A separate, rarer repair exists: if the clamped `P` is no longer positive semi-definite
   (observed with an almost constant regressor — e.g. hovering with centred sticks), it is
   rebuilt from its clamped diagonal and `kFlagCovarianceRepaired` is set, because
   otherwise `phi' P phi` can go negative and `alpha <= 0` (a real failure reproduced in
   `tests/cpp/suite_rls.cpp`). The first version of this code rebuilt whenever a diagonal
   needed clamping; the golden replay caught it (17 % command error) and it was split into
   "clamp in place" + "rebuild only if not PSD".

## 4. Build / integration status

* `firmware/lib/str_core/` is pure C++17 (it also compiles as C++11, which is what the
  Arduino toolchain uses), has **no** Arduino dependency, and builds warning-free with
  `clang++ -std=c++17 -Wall -Wextra`.
* Host tests: `tests/cpp/` (see `ctest --test-dir build`), covered areas: 3x3/5x5 linear
  solves, Diophantine vs the explicit §5 matrix and vs both legacy adjugates, RLS
  convergence on a known ARX plant, projection idempotence/containment, §6.2/§6.3 guards,
  safe mode, saturation feedback, and behaviour equivalence with the legacy implementations.
* **Not verified:** the `.ino` files are not compiled by any real toolchain here (no AVR
  core installed) and nothing has been run on hardware. They were syntax-checked by
  concatenating them in Arduino order and compiling against stub `Arduino.h`/`Servo.h`/
  `IMU.h`/`SdFat.h` headers, for both `MOTOR_ENABLE` settings.
