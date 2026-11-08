# PI-STR Control Specification (normative)

This document is the single source of truth for **all** re-implementations in this
repository: the Python simulator (`sim/python/`), the host-testable C++ core
(`firmware/lib/str_core/`), and any review of the legacy MATLAB code (`sim/matlab/`).

If an implementation disagrees with this document, the document wins — unless the
disagreement is recorded in §9 (Known deviations).

Reference: W. Lei, C. Li, M. Z. Q. Chen, *"Robust Adaptive Tracking Control for
Quadrotors by Combining PI and Self-Tuning Regulator"*, IEEE TCST 27(6):2663-2671,
2019. DOI [10.1109/TCST.2018.2872462](https://doi.org/10.1109/TCST.2018.2872462).

---

## 1. Scope and verification strategy

No MATLAB/Octave and no Arduino/AVR toolchain are available in the development
environment. Therefore:

* The **Python simulator is the numerical oracle** for everything algorithmic.
* The **C++ core is host-compiled** (clang++/g++) with no Arduino dependency, and is
  checked against Python-generated golden vectors (§8).
* Firmware `.ino` files cannot be compiled here. Changes to them must be
  **behaviour-preserving** integrations, reviewed by hand, and must not alter the
  control law except as mandated by §6.

## 2. Continuous-time plant

Two fidelity levels are required, selected by `Params.fidelity`:

* `"paper"` — ported **verbatim** from MATLAB, used for parity/reproduction only.
* `"high"` — engineering-fidelity model, the default for every new evaluation.
  Fully specified in [`PLANT_MODEL.md`](PLANT_MODEL.md).

The ARX channel model of §3 is *not* changed by the fidelity level: the point of the
high-fidelity plant is to check that the identifier and the resulting loop survive a
plant that does not satisfy the ARX assumption.

Source of truth for parameters: `sim/matlab/reduced_overshoot/Load_SystemParameters.m`.
**Port it line by line. Do not re-derive, do not "improve" the constants.**
Target airframe: DJI F450 frame, APC 1047 props, SunnySky A2212 KV980 motors.

State: quaternion `q=[q0,q1,q2,q3]`, body rate `Omega∈R³`, motor speed `w∈R⁴`,
motor current `I∈R⁴`, position `xi∈R³`, velocity `v∈R³`. Plus per-motor PWM.

Motor electrical/mechanical model (Euler, `dt`):

```
U_motor    = Umax_motor * sqrt(PWM_percent)
I(k+1)     = I(k) + dt*(U_motor - ke*w(k) - rm*I(k))/L_motor ,  then clamp I>=0
w(k+1)     = w(k) + dt*(km*I(k) - kQ*w(k)^2)/J_motor
```

with `PWM_percent = (PWM - PWM_min)/PWM_min`, `PWM_min=1000`, `PWM_max=2000`,
`Umax_motor_normal=11.1`.

Mixing (`w_rotor_mix`, from rotor speeds squared to `[Gamma; T]`):

```
[Gamma; T] = [ 0      -d*kT   0       d*kT ;
               d*kT    0    -d*kT     0    ;
               kQ     -kQ     kQ     -kQ   ;
               kT      kT     kT      kT   ] * w.^2
```

Attitude (quaternion, then Euler extraction):

```
Omega(k+1) = Omega(k) + dt * J^-1 * (-Omega × (J*Omega) + Gamma)
q(k+1)     = normalize( q(k) + dt/2 * [-q1 -q2 -q3; q0 -q3 q2; q3 q0 -q1; -q2 q1 q0] * Omega )
roll = atan2(-c32, c33),  pitch = asin(c31),  yaw = atan2(-c21, c11)
```

where `c_ij` are the elements of `C_I2B` built from `q` exactly as in the MATLAB
source (do not "simplify" into a direct quaternion-to-Euler formula; parity matters).

Position:

```
f      = -C_I2B' * T * e3          # thrust is along -z of body frame
v(k+1) = v(k) + dt*( g*e3 + f/m )
xi(k+1)= xi(k) + dt*v(k)
```

Numerics: `dt = 1e-3` s, explicit Euler, **identical integration order to MATLAB**
so that trajectories are comparable. An RK4 variant may be added as an option but
must not be the default used for parity tests.

## 3. Discrete-time channel model

Each controlled channel is identified as a linear ARX model at the controller
sampling period `h` (`h = 0.04` s for Mode II firmware/MATLAB, `0.02` s for the
trajectory study; keep both as parameters).

**Mode II (inner loop, 2nd order, 4 parameters)** — used by default:

```
y(k) = -a1*y(k-1) - a2*y(k-2) + b0*u(k-1) + b1*u(k-2)
theta = [a1, a2, b0, b1]
phi(k) = [-y(k-1), -y(k-2), u(k-1), u(k-2)]
```

Here `y` is the **measured angular rate** (`dot_y_m`) and `u` the differential
PWM command of that axis (see §5 for how `u` is recovered).

**Mode I (single loop, 3rd order, 6 parameters)**:

```
theta = [a1, a2, a3, b0, b1, b2]
phi(k) = [-y(k-1), -y(k-2), -y(k-3), u(k-1), u(k-2), u(k-3)]
```

Initial parameter guess (analytic, from the motor model; `percent` scales it to
create a deliberate initial mismatch, e.g. `percent = 0.75`):

```
a2 = exp(-tf_beta*h)
a1 = -(1 + a2)
b0 =  (tf_alpha/tf_beta)*h - (tf_alpha/tf_beta^2)*(1 - a2)
b1 =  (1/tf_beta)*(1 - a2) * (tf_alpha/tf_beta)*(1 - a2) - a2*b0
theta0 = percent * [a1, a2, b0, b1]
```

`tf_beta`, `tf_alpha` come from `Load_SystemParameters.m` (§ "Additional
Information"). For Mode I the 3rd-order guess is obtained by discretising
`G(s) = tf_beta/(s^2 + tf_alpha*s)`-style model as the MATLAB file does — port it,
do not invent it.

## 4. RLS with forgetting factor

For each channel, with `lambda ∈ (0,1]` (0.99 firmware, 0.998 simulation):

```
e      = y(k) - theta' * phi(k)
alpha  = lambda + phi' * P * phi
K      = P * phi / alpha
theta  = theta + K * e
P      = (I - K*phi') * P / lambda
```

Implementation requirements:

1. **Never form `I - K*phi'` explicitly** in the C++ core; use the rank-1 update
   `P = (P - K*(phi'*P))/lambda` (identical algebra, half the work, fewer copies).
2. `P` must be symmetrised after each update: `P = 0.5*(P + P')`.
3. `P` diagonal elements clamped to `[0, Pmax]` (`Pmax = 100` in sim, `5` in
   firmware). If clamped, rebuild the matrix from its clamped diagonal (keep
   off-diagonals, they are unaffected by the clamp in practice) — document the choice.
4. Initial `P = Pmax * I`, `theta(0) = theta0` from §3.
5. `resetdata()` restores exactly `theta0` and `P = Pmax*I` — no stale history.

## 5. Pole placement and control law

Reference model (`Am`, second order) and integral extra pole (`Ao`, first order):

```
Am(z) = 1 + am1 z^-1 + am2 z^-2
  am1 = -2*exp(-zeta*wn*h) * cos(wn*sqrt(1-zeta^2)*h)
  am2 = exp(-2*zeta*wn*h)
Ao(z) = 1 + ao1 z^-1          # ao1 = mu  (mu = 0 in the paper's experiments)
```

Reference gain (unity DC gain):

```
t0 = (1 + am1 + am2) / (b0 + b1)      # Mode II
t1 = t0 * ao1
```

Diophantine equation `A*R + B*S = Am*Ao`, with `R = 1 + r1 z^-1` (2nd order case)
and `S = s0 + s1 z^-1`. Unknowns in the order **`x = [r1, s0, s1]`**:

```
[ 1    b0   0  ] [r1]   [ am1 + ao1 - a1      ]
[ a1   b1   b0 ] [s0] = [ am2 + ao1*am1 - a2  ]
[ a2   0    b1 ] [s1]   [ ao1*am2             ]
```

For Mode I (3rd order), `R = 1 + r1 z^-1 + r2 z^-2`, `S = s0 + s1 z^-1 + s2 z^-2`,
unknowns `x = [r1, r2, s0, s1, s2]`, 5x5 system, same construction rule.

**Solve with Gaussian elimination with partial pivoting.** The legacy 3x3 adjugate
and the hand-expanded 5x5 adjugate in the Arduino code are forbidden in new code
(`firmware/lib/str_core/`): they are where the historical indexing bug lived.

Control law (2nd-order case):

```
v(k) = -ao1*u(k-1)
       + t0*y_des(k) + t1*y_des(k-1)
       - s0*y(k)     - s1*y(k-1)
       + (ao1 - r1)*u_actual(k-1)

u(k) = clamp(v(k), -u_max, +u_max)
```

`u_actual` is the **physically realised** differential command recovered from the
saturated motor PWMs (anti-windup for the identifier), never the pre-saturation
design value.

## 6. Numerical-safety policy (new, mandatory)

This section is the core fix of stage A. Every item must be implemented
identically in Python and C++ so the golden-vector test can cover it.

### 6.1 Parameter projection (applied after every RLS update)

```
a1 <- clamp(a1, -4, 4);  a2 <- clamp(a2, -2, 2)
# keep the denominator strictlySchur-stable-ish: project (a1,a2) into the
# triangle  a2 > -1, a1 + a2 > -1, a1 - a2 < 1   (standard 2nd-order stability region)
b0 <- clamp(b0, -10, 10);  b1 <- clamp(b1, -10, 10)
# keep the DC gain finite and the corrector well-posed
if |b0 + b1| < BSUM_MIN:  b0 <- BSUM_MIN - b1
```

`BSUM_MIN = 1e-3` (float firmware) / `1e-3` (double Python). Projection must be
idempotent and must be applied **before** the gains are recomputed.

### 6.2 Singularity guards

```
if |b0 + b1|                  < 1e-6  ->  freeze gains, reuse previous, set flag
if |det(A)|                   < 1e-9  ->  freeze gains, reuse previous, set flag
if |v(k)|                     > 1e6   ->  treat as singular, freeze gains
```

"Freeze gains" means: keep the last valid `(r1, s0, s1, t0, t1)` and keep
controlling; do not emit zero and do not emit the singular solution.

### 6.3 Non-finite guard

If any of `theta`, `P`, `y`, `u`, or the computed command is NaN/Inf:

1. restore `theta` and `P` from the last valid snapshot,
2. recompute nothing from the bad sample,
3. increment a consecutive-failure counter; at `>= 50` consecutive failures enter
   **safe mode**: output `0` differential command, set a status flag, latch until
   `resetdata()`.

### 6.4 Saturation

The design command is clamped to `u_max`; the *actual* value fed back into the
regressor and the control law is the realised one (§5). No integrator windup is
possible in STR (the `-ao1*u(k-1)` term uses the actual value); the outer PI loop
in Mode II still needs its own clamp, which is `Rate_middledesire` limited in
firmware and `u_c = clamp(...)` in simulation.

## 7. PI-STR cascade (Mode II) and single-loop STR (Mode I)

Outer loop (PI, on angle error, runs at the STR rate):

```
e_ang        = angle_des - angle_meas
rate_des     = Kp*e_ang + Ki*∫e_ang dt + Kd*(rate_des_prev - rate_meas)   # firmware form
```

Firmware values: `Kp=1.5, Ki=0.01, Kd=0` (Mode II). Simulation values:
`Kp=1.0, Ki=0.02` on the rate error. **These differ on purpose**; do not unify
without recording it in §9.

The inner STR tracks `rate_des` as `y_des` with `y = rate_meas`.

Mode I has no PI outer loop: the STR is fed the angle reference directly, with the
3rd-order model absorbing the motor/servo lag.

Two outer-loop semantics exist, and both are implemented (`outer_form`):

* `"firmware"` — angle error -> PI -> `rate_des`, `Kp = 1.5`, `Ki = 0.01`, `Kd = 0`.
  This is the deployed law; use it when validating the embedded core.
* `"paper"` — angle error plus ESO rate feedforward, `Kp = 1.0`, `Ki = 0.02`.
  Use it when reproducing the published figures.

Golden vectors (§8) bypass the outer loop entirely: `y_des` is prescribed, so the
fixture tests the STR alone and this choice cannot leak into it.

## 8. Cross-language conformance test (golden vectors)

One scenario, deterministic, no noise, no faults:

* Mode II, `h = 0.04` s, `dt = 1e-3`, `Pmax = 5`, `lambda = 0.99`, `mu = 0`,
  `zeta = 0.7`, `wn = 5`, `percent = 0.75`.
* Initial condition: `roll = 0.02 rad`, everything else zero.
* Reference: roll step to `0.10 rad` at `t = 1.0 s`.
* Run 3 s. Log `y`, `u`, `u_actual`, `theta_hat` at every controller tick.

Acceptance: Python vs C++ agreement `max |Δ| / max |ref| < 1e-6` on all logged
signals, **and** the guard-free path must be bit-comparable in structure (same
update order). Vectors stored as CSV in `tests/golden/` (format and rationale in
`tests/golden/README.md`; CSV is chosen so the C++ test needs no third-party
JSON parser).

All fixture quantities are SI (`rad`, `rad/s`). The fixture prescribes `y_des`
directly — no outer PI, no plant, no sensor model — so it isolates the STR. The
guard constants of §6 are dimension-dependent (the Mode I SI `b0+b1` is ~2.1e-3,
close to `BSUM_MIN`); the §8 scenario is Mode II, where both unit systems stay far
from the guards. Any new fixture must state its unit system in its header.

## 9. Known deviations and honest disclosures

1. **MATLAB is not runnable here.** The Python port is validated by
   line-by-line correspondence with `sim/matlab/`, not by numerical diff against
   MATLAB output. Any place where the MATLAB code is self-contradictory is listed
   below.
2. The legacy `firmware/Quadrotor_Mode_II_Arduino_Due` STR implementation
   (`firmware/libraries/MyLibraries_Wenchao/STR.cpp`) assigned `s0 = xx[2]` and
   `s1 = xx[3]` for a 3-element vector. Per §5 the correct mapping is
   `s0 = x[1]`, `s1 = x[2]`. This is a **bug**, not a design choice.
3. MATLAB Mode II computes `y_des` as the PI output but the ARM firmware feeds
   `output_desired` (angle-domain PI) while the simulator feeds the rate error.
   The behavioural difference is documented here and tracked separately; stage A
   does not change flight behaviour beyond §6's guards.
4. The simulator's outer PI in the trajectory study differentiates the reference
   through an ESO; the firmware has no ESO. This asymmetry is intended (the
   firmware only has a pilot-rate reference).
5. **The plant parameters are not energy-consistent.** `km ≠ ke` by 1.75x in the
   published set, so the electrical model cannot support energy or endurance claims.
   See `docs/PLANT_MODEL.md` §10. Every test that touches power must use the
   electromotive-power definition, and endurance figures are informational only.
6. **`kT` was fitted against the inflow-free blade-element expression**, so enabling
   rotor inflow at the published collective pitch drops hover thrust by ~42%. The
   high-fidelity plant therefore re-calibrates the collective pitch
   (`Rotor.calibrate`, `docs/PLANT_MODEL.md` §2). This is a calibration, not a
   change of the controller's model: the ARX channel identification is untouched.
7. **The inertial frame is NED** (z down, altitude `h = -xi_z`). The single-axis
   MATLAB altitude script uses an inconsistent sign; it is not normative.
8. **The firmware's QuadP yaw mixer appears to have the opposite sign to the MATLAB
   `u_mix` column.** Unverifiable without hardware; recorded in
   `firmware/SAFETY_NOTES.md`, deliberately not changed.
9. **The MATLAB Mode I study indexes the wrong history sample.** For the second-order
   `Ao` term it uses `u_TEMP(t - period)` where the recursion requires
   `t - 2*period`. The Python implementation uses the corrected index, so the Mode I
   MATLAB trajectories are *not* exactly reproducible — this is a defect in the
   published script, not in the port.
10. **Closed-loop identifiability.** With a smooth reference and no persistent
   excitation, RLS with forgetting identifies `b0 + b1` but not `b0` and `b1`
   individually; the estimate can even change sign. Open-loop PRBS recovers all
   parameters to machine precision. Any claim about individual `b` coefficients made
   from closed-loop flight data is unsupported by construction.
11. **Deliberate behaviour change (Mode II firmware).** `init_controller(...)` used
   to receive `STR_period / 1000`, an *integer* division that evaluates to `0` for a
   40 ms period. With `h = 0` the reference model degenerates to
   `Am = 1 - 2z^-1 + z^-2`, so `t0 = Am(1)/B(1) = 0` and the inner loop had **no
   reference feedforward at all**. The firmware now passes `0.04`, which gives
   `t0 = 0.56395` on the yaw channel. This changes flight behaviour; the paper's
   simulation used `h = 0.04` and was never hardware-validated with the `h = 0` path.
12. **Deliberate behaviour change (Mode I firmware).** `y_desire[]` was declared but
   *never assigned*, so the STR axis regulated to zero instead of tracking the
   pilot's stick. It is now fed from `Angle_desire[Index]`. Also a behaviour change.
13. **The IEEE reference's Mode II initial parameters were computed with the wrong
   sign** (`atemp = +54`, giving a pole of radius 8.67 and negative DC gain) and with
   a milliseconds/seconds slip in the `b` terms. Both this repository and the host
   core now derive `theta0` from §3 with the published `tf_alpha/tf_beta` and
   `percent = 0.75`, so firmware and simulator start from the same model.
14. **The published `tf_alpha` is conservative, not aggressive — correction of an
   earlier claim in this project.** The literal `Umax/250` in the formula is 3.42x
   the true hover voltage slope (`Umax / (2*sqrt(p)*1000) = 0.01297 V/us` at
   `p ~ 0.183`), but two other factors cancel most of it: the `/2` written in the
   expression (giving an effective 1.712x) and the use of a *single-rotor* `2*omega`
   thrust derivative where a roll differential pair requires `4*omega` (a factor
   0.5). Net: `1.712 * 0.5 = 0.856` of the first-principles value.

   Measured at the hover point:
   | configuration | `a_ss` vs §3 analytic | `|H(5 rad/s)|` vs §3 analytic |
   |---|---|---|
   | first-principles linearisation | 1.169 | 1.170 |
   | plant, `fidelity="paper"` | 1.177 | 1.179 |
   | plant, `fidelity="high"` all extras off | 1.006 | 1.008 |
   | plant, `fidelity="high"` all extras on | 1.313 | 0.915 |

   So the §3 initial gains are **15-27 % low** (conservative), not high. A previous
   note in this repository asserted they were ~4.8x *high*; that was a
   measurement-window artifact (a 0.2 s rate response compared against the DC
   `alpha/beta` gain) and the claim is **withdrawn**.

   Practical consequence: the STR starts from an under-estimated authority and adapts
   *upwards*, which is the safer direction for a self-tuning loop (an over-estimate
   would begin by over-driving). `fidelity="high"` all-on is 1.31x, i.e. the true
   authority is larger still; `battery_sag` alone contributes +9.4 % because the pack
   sits at 12.6 V while `Umax_motor_normal` is 11.1 V.
15. **Free-flight accelerometer semantics.** In coordinated flight the accelerometer
   measures the **thrust-axis specific force**, which reads `(0, 0, -g)` in the body
   frame; it cannot separate gravity from translational acceleration. An
   accelerometer-referenced tilt observer therefore drifts, or settles into a slow
   limit cycle, without position/velocity aiding. The physics bench consequently uses
   body-rate quaternion propagation with a 3-D (inclined) magnetometer; the firmware's
   2-state-per-axis Kalman path is kept and tested, with its levelling behaviour
   asserted explicitly.
