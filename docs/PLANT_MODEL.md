# Quadrotor Plant Model Specification

Two fidelity levels, both behind one interface. `fidelity="paper"` exists for
reproducing the paper; `fidelity="high"` is the default for all new evaluation.

Both levels share `sim/python/pistr_sim/types.py` (state/actuator contract) and
`params.py` (parameters). Neither may change the controller-side discrete model
(`docs/CONTROL_SPEC.md` §3) — the STR still identifies the same ARX channel. The
whole point of the high-fidelity plant is to test whether that identification and
the resulting loop survive a plant that does **not** obey the ARX assumption.

---

## 1. What "realistic" means here (and what it does not)

Included (engineering fidelity, all closed-form, no external data files):

| Domain | Model |
|---|---|
| Rotor thrust | Blade-element + momentum theory (BEMT), quasi-steady, solves induced inflow `v_i` together with `T` |
| Rotor in-plane force | Empirical H-force, `H = C_H ρ A (ωR) · V_in_plane`, opposing in-plane relative airspeed |
| Rotor reaction torque | Profile power + induced power, `Q = kQ ω² (1 + k_μ μ²)` |
| Rotor gyroscopic torque | `Σ J_r (Ω × ω_i ẑ)` — this term is what makes yaw/roll coupling appear |
| Body aerodynamics | Quadratic drag in body axes with per-axis `CdA`, plus an optional rotor-projected vertical area (`rotor_projected_area`, default off) |
| Ground effect | Thrust augmentation `1/(1-(R/4z)²)`, blended out above `z = 2R` |
| Wind | Steady offset + Dryden turbulence (u,v,w) as shaped noise; all aero terms use relative airspeed |
| Propulsion | DC motor with inductance + back-EMF + viscous drag, **ESC first-order lag**, PWM µs quantisation |
| Battery | LiPo open-circuit voltage vs SoC (per-cell piecewise), internal resistance, current integration, sag |
| Rigid body | Full inertia tensor including `Ixz`, RK4, quaternion normalisation, no small-angle assumption |
| Sensors | Gyro (white noise + random-walk bias), accel (noise + bias), magnetometer (noise + hard/soft iron), barometer (noise + bias drift), one-sample transport delay |

Explicitly **out** of scope (documented as future work, do not silently add):
blade flapping dynamics and its hub moments, unsteady/attached-flow aerodynamics,
rotor-rotor and rotor-body interference beyond simple ground effect, motor thermal
derating, ESC current limiting and regenerative braking, structural flexibility,
GPS/optical-flow aiding, wind shear profiles.

## 2. Rotor model (BEMT)

Per rotor `i`, with radius `R`, chord `c`, blade count `N_b = 2`, lift slope `a = Cl_alpha`,
pitch `θ_tw`, solidity `σ = N_b c/(πR)`, disc area `A = πR²`.

Relative airspeed at the rotor along the rotor axis (body `-z` direction for thrust-up
configuration): `V_c = (v_rel_body · (-ẑ))` (positive when the vehicle climbs, i.e. air
flows down through the disc). In-plane speed magnitude `V_p = |v_rel_body − (v_rel_body·ẑ)ẑ|`.

Blade element (assume uniform inflow, no twist beyond collective):

```
λ_i   = (V_c + v_i)/(ω R)
T_be  = 0.5 ρ A (ω R)² σ a (θ_tw/3 − λ_i/2)
```

Momentum theory in axial flow (normal working state, `V_c >= 0`):

```
T_mt  = 2 ρ A v_i (v_i + V_c)
```

Solve the two for `(v_i, T)` simultaneously every step by fixed-point iteration
(seed `v_i = sqrt(max(T_prev,0)/(2ρA))`, 12 iterations, damped `0.5`), then apply the
following robust clamps:

* `v_i >= 0` always; `T >= T_min = 0` (a rotor cannot push backwards).
* For descent (`V_c < 0`), blend to the empirical vortex-ring branch:
  `v_i = v_h * (1 − 1.125·V_c/v_h)` clamped to `[0, 2v_h]`, `v_h = sqrt(T/(2ρA))`,
  and switch back to the momentum solution once `|V_c| > 2v_h`.
* If `ω < ω_min = 10 rad/s`, output `T = 0, H = 0, Q = 0` and skip the solve.

**Consistency requirement (two separate identities, do not conflate them):**

1. With `λ_i → 0` and `V_c = 0`, this must reproduce `T = kT ω²` with `kT` and `θ_tw`
   from `Load_SystemParameters.m` to within 0.5 %. That is *why* `kT` has the form it
   has: the published coefficient was fitted against the **inflow-free** blade-element
   expression.
2. Because of (1), enabling inflow at the same `θ_tw` would drop hover thrust by ~42 %
   and move the trim point (ω, current, endurance) by tens of percent — i.e. it would
   contradict the measured thrust curve the whole paper is anchored to. Therefore at
   `fidelity="high"` the collective pitch is **re-calibrated**:

   ```
   solve θ_eff  such that  BEMT(ω = w_hover, V_c = 0, θ = θ_eff) = kT · w_hover²
   ```

   computed once per parameter set (monotone in `θ_eff`, bisection to 1e-10, seed
   `θ_tw`). `Rotor.calibrate = True` enables this and is the default. The published
   `θ_tw` stays untouched in `params.py`; the calibrated value is exposed as
   `Plant.rotor_theta_effective` and logged with every run.

   Rationale: a measured `kT` is a hover-anchored statement. A different functional
   form (with inflow) must be re-fitted to the *same measurement*, not combined with a
   coefficient that was itself the output of that fit. Skipping the calibration is
   physically wrong in a way that is invisible in the attitude loops but obvious in
   trim, current and endurance. Set `calibrate=False` only to reproduce the
   inconsistency deliberately.

Test implication: P2/P4/P6 assert against the **calibrated** model; the uncalibrated
trim numbers are reported alongside as information.

In-plane force (empirical, direction opposite to in-plane relative airflow):

```
H = C_H ρ A (ω R) V_p          # C_H ≈ 0.03, tunable
```

Torque reaction:

```
Q = kQ ω² (1 + k_mu μ²),  μ = V_p/(ωR), k_mu ≈ 3.0
```

## 3. Propulsion: motor, ESC, battery

State per motor: current `I_i`, speed `ω_i`. ESC state: filtered duty `d_i`.

```
d_i(k+1)  = d_i(k) + dt/τ_esc * (d_cmd_i − d_i(k)),     τ_esc = 0.02 s
d_cmd_i   = quantise((PWM_i − 1000)/1000, 1e-3)        # 1 µs of a 1 ms span
V_term    = max(V_oc(SoC) − R_int * Σ I_i, 0)          # pack terminal voltage under load
V_m_i     = d_i * V_term
I_i(k+1)  = max(I_i(k) + dt*(V_m_i − ke·ω_i − rm·I_i(k))/L, 0)
ω_i(k+1)  = max(ω_i(k) + dt*(km·I_i − Q_i(ω_i,V_p))/J_motor, 0)
SoC(k+1)  = SoC(k) − dt * Σ(V_m_i·I_i) / (3600 · Q_pack · V_nom)   # Ah, s -> h
```

Notes:
* `Q_i(ω,V_p)` is the aerodynamic **load** torque from §2, not `kQ ω²` alone — the
  rotors must spin against the full aerodynamic torque.
* The paper's `U = Umax·sqrt(PWM_percent)` is a parity-only artefact; the high-fidelity
  level uses duty × terminal voltage, which makes battery sag physically meaningful.
* **PWM quantisation contaminates small-signal identification.** With a 1 µs
  quantum and a dither amplitude of ~1.5 µs, the measured `|H(omega)|` collapses to
  ~0.81 of the true value while the DC slope stays accurate (`a_ss` ratio 1.03). Any
  identification or gain measurement must use steps of at least ~5 µs, or fit the
  step slope only. This matters because the STR identifies from exactly these small
  differential commands.
* **Electrical substepping is mandatory.** The electrical time constant is
  `τ_e = L/rm = 0.40 ms` (pole at 2516 1/s). At `dt = 1 ms` the explicit Euler update
  has amplification factor `1 − dt·rm/L = −1.52`: the current oscillates at Nyquist,
  diverges, hits the `I >= 0` clamp and injects spurious torque. RK4 does not rescue
  it either (the coupled fast eigenvalue is ≈ −2880 1/s, beyond RK4's real-axis limit
  of −2.785 at this step). Implement the actuator (duty, current, speed, SoC) with
  `n_sub = ceil(dt / (0.5·L/rm))` substeps (≈ 6 at `dt = 1 ms`) and one rigid-body
  step per `dt`, or integrate the full state with a step small enough to
  resolve `τ_e`. See §10.5 for why this matters: the published MATLAB results do
  *not* do this.
* `V_oc(SoC)`: piecewise-linear per cell, `[1.00, 0.95, 0.80, 0.50, 0.25, 0.10, 0.0]`
  ↔ `[4.20, 4.05, 3.90, 3.80, 3.70, 3.45, 3.20]` V; pack = `3 × cell`, `Q_pack = 3300 mAh`,
  `R_int = 0.035 Ω`, `V_nom = 11.1 V`.

## 4. Frames, rigid body, forces, moments

**Frame convention (normative, previously implicit):** the inertial frame is **NED** —
`z` points *down*, gravity is `g ẑ` with `g = +9.81`, thrust acts along the body `−ẑ`,
and **altitude is `h = −xi_z`**. This is the convention of
`sim/matlab/robustness/Sim_Trajectory_Tracking_Control.m` (`f = -C_I2B'*T*e3`,
`v̇ = g*e3 + f/m`) and the only one under which the barometer `−xi_z` and the
free-fall test `a_z = +g` are self-consistent. The single-axis altitude channel in
`AttitudeandAltitude_Tracking_Control_Mode_II.m` is *not* a frame definition — do not
copy its `F_z = m g − T` sign into the 3-D model.

Rotor numbering follows the firmware comment block: **0, 2 spin CW; 1, 3 spin CCW**
(viewed from above), `dir_i = +1` for CW, i.e. `spin_dirs = (+1, −1, +1, −1)`.

```
F_body      = Σ_i [ (T_i + T_ge,i)(−ẑ) + H_i(−v̂_p,i) ] + F_drag_body
M_body      = Σ_i ( r_i × F_rotor,i ) + M_yaw + M_gyro + M_drag
M_yaw       = Σ_i dir_i Q_i ẑ                     # rotor-drag reaction on the body
M_gyro      = −Ω × h_rotor,   h_rotor = J_r Σ_i dir_i ω_i ẑ
F_drag_body = −0.5 ρ (CdA ⊙ |v_rel_body| ⊙ v_rel_body)
```

Corrections against the earlier draft of this document (both were wrong):

* the gyroscopic term is `−Ω × h_rotor`, not `+Σ J_r (Ω × ω_i ẑ)` — the body reacts to
  the rotor momentum, and the per-rotor direction must enter `h_rotor`;
* the yaw reaction is `Σ dir_i Q_i`, not `Q_i(−1)^{dir_i}`, which collapses to a
  constant factor and flips the yaw axis.

`r_i`: arm geometry — `±d` axis-aligned for QuadP (default, matches the firmware
mixer), `±d/√2` for QuadX.

Two classes of sign convention must be **asserted, not argued about** — this is where
the project historically went wrong:

| property | assertion |
|---|---|
| G1 | equal rotor speeds ⇒ `M_gyro = 0` exactly |
| G2 | `Ω = (Ωx, 0, 0)` with `h_rotor = (0, 0, h_z)` ⇒ `M_y = +Ωx·h_z` |
| G3 | a positive yaw command at hover (`u_yaw > 0`) ⇒ `dΩ_z/dt > 0` |
| G4 | the mixer reproduces MATLAB's `Gamma_z = kQ(w0² − w1² + w2² − w3²)` for the same rotor speeds |

G3 is the important one: it forces the plant and the MATLAB reference to agree on the
*closed-loop-observable* sign, so a flipped yaw axis cannot hide behind a flipped
measurement.

> **Known inconsistency, do not silently "fix":** the firmware's QuadP mixer
> (`PWM_ref[0] = Pitch − Yaw + Alt`, `PWM_ref[1] = −Roll + Yaw + Alt`) appears to give a
> *negative* yaw torque for a positive yaw command — the opposite of the MATLAB `u_mix`
> column. It is undetectable without hardware, so record it in
> `firmware/SAFETY_NOTES.md`; do not change it blind.

Equations of motion (NED, RK4):

```
ẋi = v
v̇  = g ẑ + (1/m) R_b2i(q) F_body(v_rel)
q̇  = 0.5 q ⊗ [0, Ω]
Ω̇  = J⁻¹ (M_body − Ω × (J Ω))
```

`v_rel = v − v_wind(xi, t)`, transformed into the body frame before being used in any
aero term. This is the single most important realism fix over the paper model: **all
aerodynamics see airspeed, not ground speed.**

### 4.1 Sensor frame conventions (normative)

Two conventions that silently break the estimator if guessed wrong:

* **Accelerometer sign.** `accel` is the *specific force in the body frame*, so a
  level hover reads `(0, 0, −g)` in NED. Estimators that expect `+g` must flip the
  sign explicitly (`accel_sign=-1`); this is not a free parameter.
* **In free flight the accelerometer cannot see gravity.** It measures the
  thrust-axis specific force, which is `(0, 0, -g)` in the body frame during
  coordinated flight regardless of attitude history. It therefore cannot distinguish
  "tilted with no acceleration" from "level and accelerating", and an
  accelerometer-referenced tilt observer drifts without position/velocity aiding
  (see `docs/CONTROL_SPEC.md` §9.15). Do not "fix" this in the estimator; the real
  vehicle has the same limitation.
* **Magnetometer is a 3-D field, not a heading.** `mag` is the full geomagnetic
  vector (default inertial field `(45, 0, 20) µT` — north component plus a 20 µT
  vertical/down component) rotated into the body frame. A heading-only model that
  assumes a horizontal field will produce a yaw error of `atan(20/45) ≈ 24°` at level
  flight. The estimator **must** tilt-compensate, as the firmware does, and any
  assumption of a horizontal field must be a declared flag with a test.

## 5. Sensors and timing

Controller runs at `h` (default 25 Hz, `h = 0.04` s); simulator integrates at
`dt = 1e-3` s (RK4). Sensor model is sampled at the controller rate with a one-sample
transport delay (the firmware's 100 Hz loop reads the previous sample too).

```
gyro  = Ω + b_g + n_g,        b_g(k+1) = b_g(k) + n_bg·sqrt(h)
accel = R_b2i'(q)(v̇ − g ẑ) + b_a + n_a
mag   = C_hard·ẑ_I_body_normalised + b_m + n_m
baro  = −xi_z + b_baro + n_baro,   b_baro drifts as a slow random walk
```

Defaults: gyro noise σ = 0.01 °/s (σ_density scaled by `1/sqrt(h)`), bias random walk
0.05 °/s/√h; accel σ = 0.5 mg, bias 1 mg; mag σ = 0.5 µT, hard-iron 3 %;
baro σ = 0.05 m, bias drift 0.02 m/√s. All parameters live in `SensorParams`.

## 6. Wind

```
v_wind(xi,t) = v_steady + v_gust(t)
```

`v_steady` from a scenario. `v_gust` is Dryden turbulence: three independent
first-order shaping filters driven by unit-variance white noise,

```
H_u(s) = σ_u sqrt(2 L_u/(π V̄)) / (1 + (L_u/V̄)s)
```

(`V̄` = reference airspeed, default 5 m/s; `σ_u = 1.5, σ_v = σ_w = 1.0` m/s;
`L_u = L_v = 200 m, L_w = 50 m` for light turbulence at low altitude).

Note on normalisation: the transfer function above is written so that the *one-sided
PSD peak* is `K²` with `K = σ√(2L/(πV̄))`. Driving a discrete first-order filter
directly with that gain gives a time-domain standard deviation of `σ/√π`, not `σ`.
Use instead

```
a = exp(−dt·V̄/L),   b = σ·sqrt(2·L/V̄)·(1 − a)      # time-domain std = σ
```

and assert `std(gust) ≈ σ` (§8, P11). Implemented as `scipy.signal.lfilter` one-pole
sections — deterministic given a seed.

## 7. Ground effect

```
T_ge = T · (1/(1 − (R/(4 z))²) − 1)     for z > R/2, blended linearly to 0 at z = 2R
```

Clamped so the augmentation never exceeds 30 % (empirical limit).

## 8. Validation suite (mandatory, automated)

Implement as `sim/python/tests/test_plant_fidelity.py`. Every item must **assert**, not
just print. Tolerances are loose enough to be robust, tight enough to catch sign errors.

| ID | Check | Acceptance |
|---|---|---|
| P1 | Free fall (no thrust, no drag), NED | `a_z = +g`, error `< 1e-6` |
| P2 | Hover trim, **calibrated** BEMT (§2), continuous actuator (`esc_lag` off, no µs quantisation) | residual `|a| < 1e-3 g`; `ω_i` within 5 % of `w_hover`; net torque `< 1e-4 N·m`. The uncalibrated trim is **reported only, never asserted** |
| P3 | Published fit: BEMT with `λ→0`, `θ_tw` unchanged, `V_c=0` | reproduces `kT·ω²` within 0.5 % |
| P4 | Small-signal motor: hover → +5 % motor voltage | 63 % rise within 10 % of `1/tf_beta` **and** 50 % rise within 10 % of `ln 2/tf_beta` (the pole is a *rate*, not a half-time). Large-signal 0→hover is reported only |
| P5 | Energy consistency | `P_elec / (Σ ke·ω·I)` ∈ `[1.0, 1.6]`. `P_elec / (Σ Qω)` is **reported as a known model inconsistency** (`km ≠ ke`, see §10), not asserted |
| P6 | Battery | SoC monotonic ↓; `V_term` sag within 1 % of `R_int·I`; endurance projection printed. Only sanity-banded `[8, 30] min` on the calibrated model |
| P7 | Gyroscopic moment | props G1 and G2 of §4 |
| P8 | Body drag, 4 m/s level flight, zero attitude | `F_x ≈ −0.5ρ CdA_x V²` within 2 % (with the rotor-projected-area term off) |
| P9 | Inertia sanity | eigenvalues > 0, `Ixz = Izx`, matches the ported parameters to 1e-9 |
| P10 | Quaternion integrity over 120 s | `| |q| − 1 | < 1e-9`, no NaN |
| P11 | Dryden spectrum | half-power corner at `V̄/L` within a factor of 2 (the first-order Dryden spectrum peaks at `ω=0`; `V̄/L` is its −3 dB point), and time-domain std ≈ `σ` |
| P12 | Fidelity equivalence | equal **motor voltage** (paper: `p=(U/11.1)²`; high: `p=U/V_term`), `esc_lag` off and no µs quantisation, both Euler: trajectories within 5 % over 2 s. A second, strict comparison with the *same* integrator must agree to 1e-6 |
| P13 | Yaw sign, G3 of §4 | positive yaw command at hover ⇒ `dΩ_z/dt > 0` |
| P14 | Mixer parity, G4 of §4 | same rotor speeds ⇒ same yaw torque as MATLAB's `Gamma_z` |
| P15 | Electrical integration artifact (§10.5) | `1 − dt·rm/L < −1` at the documented parameters; a single-step Euler run exhibits current oscillation while the substep run is smooth (this *proves* the finding instead of asserting it in prose) |

P12 is the guard against "a realistic model that quietly changed the physics".
P13/P14 are the guards against the two historical sign errors (§4).
The `high` plant must accept flags to disable each extra (`inflow`, `h_force`,
`body_drag`, `gyro_moment`, `battery_sag`, `esc_lag`, `turbulence`, `sensors`,
`rotor_projected_area`) so ablation studies are possible and P12 is meaningful.
`rotor_projected_area` defaults to **off** to avoid double-counting momentum drag
inside `body_drag`.

## 9. Interface contract

```python
class Plant(Protocol):
    dt: float
    def reset(self, state: VehicleState | None = None) -> VehicleState: ...
    def step(self, pwm_us: np.ndarray) -> VehicleState: ...   # shape (4,), µs
    def sense(self) -> SensorSample: ...                      # delayed, noisy
    def set_wind(self, steady: np.ndarray, turbulence: bool = True) -> None: ...
```

`VehicleState` and `SensorSample` are defined in `sim/python/pistr_sim/types.py` and are
shared with the controller code. Controllers must never read plant internals — only
`sense()` — otherwise the simulator stops being a fair test.

## 10. Known model inconsistencies carried over from the original parameters

These are properties of the published parameter set, not bugs in the new
implementation. They are recorded so that no test asserts a quantity that the model
cannot satisfy, and so nobody "fixes" them silently in one place only.

1. **`km ≠ ke`.** `km = 0.014490 N·m/A` and `ke = 0.008286 V·s/rad` differ by 1.75×,
   although lossless electromechanical conversion requires `km = ke`. Consequence: at
   hover `P_elec = 72.2 W < Σ Qω = 88.1 W`, i.e. the electrical side manufactures
   energy. Both were fitted independently to different measured curves. **Therefore
   the electrical model may be used for current/sag dynamics but not for energy or
   endurance claims**; P5 and P6 are banded accordingly.
2. **`kT` was fitted with `λ = 0`.** `kT = ρ c R³ a θ_tw / 3` is the inflow-free
   blade-element thrust coefficient. Re-fitting `θ_tw` (the `calibrate` step, §2) is the
   only self-consistent way to keep `kT` and BEMT together.
3. **`Iyz = Ixz`.** The original sets the `yz` product of inertia equal to `xz`, which
   is not physically symmetric for a quadrotor. Preserved for parity; it does not
   affect the diagonal-dominated attitude dynamics materially.
4. **Altitude-thrust sign in the single-axis MATLAB script** (`F_z = m g − T`,
   `altitude += v_z dt`) is not a frame definition and conflicts with the 3-D model.
   The 3-D NED convention of §4 governs.

5. **The published MATLAB study integrates the electrical subsystem unstably.**
   With `L = 150 µH` and `rm = 0.377 Ω` the motor current has a pole at
   `rm/L = 2516 1/s`, i.e. `τ_e = 0.40 ms`, while the study uses `dt = 1 ms` and
   explicit Euler. The amplification factor of the current update is
   `1 − dt·rm/L = −1.52` — outside the unit circle, and negative, so the current
   oscillates at the Nyquist frequency with growing amplitude, is then clipped by
   `I >= 0`, and feeds a rectified, fictitious torque into the rigid body.
   RK4 does not fix it: the coupled fast eigenvalue is ≈ −2880 1/s, beyond RK4's
   real-axis stability limit (−2.785) at this step size.

   Consequence, stated plainly: **the paper's simulated motor currents (and any
   result that depends on them, e.g. the control-effectiveness coefficients derived
   from `tf_alpha`) carry this numerical artifact.** The trim points and the
   low-frequency attitude response are unaffected — they are set by the rotor and
   rigid-body dynamics, whose eigenvalues are three orders of magnitude slower — but
   anything at the electrical time scale is not trustworthy.

   This repository fixes it by substepping the actuator (§3) and keeps
   `fidelity="paper"` otherwise bit-identical to the MATLAB update order, so the
   artifact can be reproduced deliberately (P15) rather than inherited silently.
