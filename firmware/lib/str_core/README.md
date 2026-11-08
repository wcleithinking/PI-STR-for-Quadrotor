# str_core — PI-STR regulator core (host-testable)

Pure C++17, **no Arduino dependency**. Normative behaviour: [`docs/CONTROL_SPEC.md`](../../../docs/CONTROL_SPEC.md) §3–§7,
especially §6 (numerical safety).

```
firmware/lib/str_core/str_core.h    the whole implementation (templates)
firmware/lib/str_core/str_core.cpp  explicit instantiations (float/double x order 2/3)
tests/cpp/                          host tests, `ctest`
```

## Why a separate directory

The legacy implementation lived in two places (`MyLibraries_Wenchao/STR.cpp`, 2nd order,
and a hand-expanded 5x5 adjugate inside `Quadrotor_Mode_I_Arduino_Due/Controller.ino`,
3rd order). Both carried the same class of defect: an inverted / out-of-range index in
the Diophantine solution and no numerical guards at all. This directory is the single
implementation; the firmware sketches are thin call sites.

## API (stable — used from the Python golden-vector comparison)

```cpp
#include "str_core.h"

str_core::Str2f str;                        // float, order 2 (Mode II)
// str_core::Str2d / Str3f / Str3d          // double / order 3 / double order 3

str.init_estimator(lambda, Pmax, theta0);   // §4; theta0 = [a1,a2,b0,b1]
str.set_second_order_reference(zeta, wn, mu, h);   // order 2 only  (§5)
str.set_reference(am, ao);                  // generic: am[Order], ao[Order-1]

bool ok = str.begin_tick(y_des, y);         // §3/§6.3 — false = sample rejected
str.update_estimator();                     // §4 RLS + §6.1 projection + §6.3 guard
float u = str.compute_control(u_max);       // §5 Diophantine + control law + §6.2/§6.4
str.commit_actual(u_realised);              // §5/§6.4 realised command feedback

str.resetdata();                            // §4.5 + the only way out of safe mode
```

Diagnostics: `theta_hat()`, `covariance()`, `r()`, `s()`, `t()`, `last_design_command()`,
`last_command()`, `y_history()`, `u_history()`, `flags()`, `frozen()`, `safe_mode()`,
`consecutive_failures()`, `sample_accepted()`.

`str_core::solve_linear_system<Real, N>(A, b, det)` is the shared Gaussian-elimination
routine (partial pivoting) used for both the 3x3 and 5x5 Diophantine systems, and is
exercised directly by the tests.

## Tick ordering (behaviour equivalence matters)

The legacy sketches ran the estimator and the control law in an order that this core
preserves verbatim; the caller decides:

* Mode II: `begin_tick` → `update_estimator` → `compute_control` → (after the mixer) `commit_actual`
* Mode I : `begin_tick` → `compute_control` → (after the mixer) `commit_actual` → `update_estimator`

`compute_control()` must not be called more than once per `begin_tick()`.

## Deliberate differences from the legacy code

1. All §6 guards; **safe mode outputs a zero differential command** (§6.3).
2. The feedback term uses the *realised* command (§5/§6.4). Identical to the legacy code
   whenever the command is not saturated, and identical by construction in Mode II because
   the shipped configuration has `mu = ao1 = 0`.
3. The reference gain is `Am(1)/B(1)`, matching both the legacy Mode II code (where `mu = 0`
   makes `Ao(1) = 1`) and the legacy Mode I code (which never folded in `Ao(1)`). This is a
   known inconsistency with the unity-DC-gain intent of §5 for Mode I; it is preserved on
   purpose so stage A does not change flight behaviour. See `firmware/SAFETY_NOTES.md`.

## Building and testing

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure
```
