// Suite: Diophantine equation A*R + B*S = Am*Ao (§5).
//
// Three independent routes are cross-checked for every test point:
//   1. str_core's general construction + Gaussian elimination (the shipped code),
//   2. the *explicit* 3x3 matrix printed in CONTROL_SPEC §5, solved by the legacy
//      adjugate formula (copied from STR.cpp),
//   3. the legacy hand-expanded 5x5 adjugate (copied from the Mode I Controller.ino).
// Route 2/3 must agree with route 1, which is the formal statement of "the
// refactor did not change the controller".

#include <array>
#include <cmath>

#include "legacy_reference.h"
#include "str_core.h"
#include "str_test_harness.h"
#include "suites.h"

namespace str_test {
namespace {

using str_core::Str2d;
using str_core::Str3d;

// Run one 2nd-order solve through the core and report the gains.
struct Gains2 {
  double r1;
  double s0;
  double s1;
  double t0;
  double t1;
};

Gains2 core_gains_2(double a1, double a2, double b0, double b1, double am1,
                    double am2, double ao1) {
  Str2d reg;
  const Str2d::Vec theta0{{a1, a2, b0, b1}};
  reg.init_estimator(1.0, 100.0, theta0);
  reg.set_reference(std::array<double, 2>{{am1, am2}}, std::array<double, 1>{{ao1}});
  reg.begin_tick(0.0, 0.0);
  (void)reg.compute_control(1e12);
  Gains2 g{reg.r()[0], reg.s()[0], reg.s()[1], reg.t()[0], reg.t()[1]};
  return g;
}

struct Gains3 {
  double r1;
  double r2;
  double s0;
  double s1;
  double s2;
  double t0;
  double t1;
  double t2;
};

Gains3 core_gains_3(const double a[3], const double b[3], const double am[3],
                    const double ao[2]) {
  Str3d reg;
  const Str3d::Vec theta0{{a[0], a[1], a[2], b[0], b[1], b[2]}};
  reg.init_estimator(1.0, 100.0, theta0);
  reg.set_reference(std::array<double, 3>{{am[0], am[1], am[2]}},
                    std::array<double, 2>{{ao[0], ao[1]}});
  reg.begin_tick(0.0, 0.0);
  (void)reg.compute_control(1e12);
  Gains3 g{reg.r()[0], reg.r()[1], reg.s()[0], reg.s()[1], reg.s()[2],
           reg.t()[0], reg.t()[1], reg.t()[2]};
  return g;
}

// --- 2nd order ------------------------------------------------------------
void check_2nd_order_case(double a1, double a2, double b0, double b1, double am1,
                          double am2, double ao1) {
  const Gains2 core = core_gains_2(a1, a2, b0, b1, am1, am2, ao1);

  // Route 2: the explicit 3x3 system of CONTROL_SPEC §5, solved by the legacy
  // adjugate.  Unknowns are x = [r1, s0, s1].
  double xx[3];
  legacy::legacy_adjugate_3(a1, a2, b0, b1, am1, am2, ao1, xx);

  STR_CHECK_NEAR(core.r1, xx[0], 1e-9);
  STR_CHECK_NEAR(core.s0, xx[1], 1e-9);
  STR_CHECK_NEAR(core.s1, xx[2], 1e-9);

  // §5 reference gain.
  STR_CHECK_NEAR(core.t0, (1.0 + am1 + am2) / (b0 + b1), 1e-9);
  STR_CHECK_NEAR(core.t1, core.t0 * ao1, 1e-9);

  // The historical bug would have used x[2] for s0 and read x[3] out of bounds.
  // Show how far apart the two candidate values are: this is why the bug is not
  // a rounding issue but a different controller.
  STR_CHECK_MSG(std::fabs(xx[2] - xx[1]) > 1e-3, "s0/s1 collapse: %.6g vs %.6g",
                xx[1], xx[2]);
}

void check_2nd_order_family() {
  // Shipped Mode II numbers (theta0 at percent = 0.75 and the analytic guess).
  const double zeta = 0.7, wn = 5.0, h = 0.04;
  const double alpha = std::exp(-zeta * wn * h);
  const double omega = wn * std::sqrt(1.0 - zeta * zeta);
  const double am1 = -2.0 * alpha * std::cos(omega * h);
  const double am2 = alpha * alpha;

  check_2nd_order_case(-1.5, 0.6, 0.02, 0.011, am1, am2, 0.0);
  check_2nd_order_case(-1.55, 0.62, 0.03, 0.015, am1, am2, 0.0);
  check_2nd_order_case(-1.2, 0.35, 0.012, 0.008, am1, am2, 0.0);
  // mu != 0 exercises the ao1 terms (the legacy STR.cpp compares v_compute[1]
  // instead of u(k-1); with ao1 = 0 -- the shipped setting -- the two coincide).
  check_2nd_order_case(-1.4, 0.5, 0.02, 0.011, am1, am2, 0.2);
}

// --- 3rd order ------------------------------------------------------------
void check_3rd_order_case(const double a[3], const double b[3], const double am[3],
                          const double ao[2]) {
  const Gains3 core = core_gains_3(a, b, am, ao);

  // Route 3: legacy 5x5 adjugate.
  double bb[5];
  bb[0] = am[0] + ao[0] - a[0];
  bb[1] = am[1] + ao[0] * am[0] + ao[1] - a[1];
  bb[2] = am[2] + ao[0] * am[1] + ao[1] * am[0] - a[2];
  bb[3] = ao[0] * am[2] + ao[1] * am[1];
  bb[4] = ao[1] * am[2];
  double xx[5];
  legacy::legacy_adjugate_5(a[0], a[1], a[2], b[0], b[1], b[2], bb, xx);

  STR_CHECK_NEAR(core.r1, xx[0], 1e-9);
  STR_CHECK_NEAR(core.r2, xx[1], 1e-9);
  STR_CHECK_NEAR(core.s0, xx[2], 1e-9);
  STR_CHECK_NEAR(core.s1, xx[3], 1e-9);
  STR_CHECK_NEAR(core.s2, xx[4], 1e-9);
  STR_CHECK_NEAR(core.t0, (1.0 + am[0] + am[1] + am[2]) / (b[0] + b[1] + b[2]),
                 1e-9);
  STR_CHECK_NEAR(core.t1, core.t0 * ao[0], 1e-9);
  STR_CHECK_NEAR(core.t2, core.t0 * ao[1], 1e-9);
}

void check_3rd_order_family() {
  // Shipped Mode I roll/pitch numbers (Controller.ino am1/am2/am3/ao1/ao2 and the
  // initial a1..b2 guess from Quadrotor_Mode_I_Arduino_Due.ino).
  const double am[3] = {-1.4431986644, 0.7708989340, -0.1380692400};
  const double ao[2] = {0.8, 0.16};
  const double a[3] = {-1.55, 0.55, -0.18};
  const double b[3] = {0.02, 0.02, 0.02};
  check_3rd_order_case(a, b, am, ao);

  const double a2[3] = {-1.9, 1.25, -0.30};
  const double b2[3] = {0.015, 0.014, 0.013};
  check_3rd_order_case(a2, b2, am, ao);

  // Yaw's hard-coded reference polynomial (am2 = 1.2975, not Schur) is kept
  // verbatim from the legacy sketch; the solver must still produce the same
  // numbers as the adjugate.
  const double am_yaw[3] = {-1.92701, 1.29750, -0.30120};
  check_3rd_order_case(a2, b2, am_yaw, ao);
}

// The core must freeze (not emit a singular solution) when the system is
// singular; covered in detail by suite_guards, but assert here that the
// Diophantine path reports the failure rather than returning junk.
void check_singular_freezes() {
  Str2d reg;
  const Str2d::Vec theta0{{2.0, 1.0, 1.0, 1.0}};  // det = b1^2 - a1*b0*b1 + a2*b0^2 = 0
  reg.init_estimator(1.0, 100.0, theta0);
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  reg.begin_tick(0.0, 0.0);
  (void)reg.compute_control(50.0);
  STR_CHECK(reg.frozen());
  STR_CHECK((reg.flags() & str_core::kFlagDetGuard) != 0);
  // No singular solution was emitted: with no previously valid solve the gains
  // stay zero and the command is zero.
  STR_CHECK_NEAR(reg.s()[0], 0.0, 0.0);
  STR_CHECK_NEAR(reg.last_command(), 0.0, 0.0);
}

}  // namespace

void suite_diophantine() {
  check_2nd_order_family();
  check_3rd_order_family();
  check_singular_freezes();
}

}  // namespace str_test
