// Suite: §6.2 singularity guards, §6.3 non-finite guard / safe mode, §6.4
// saturation and realised-command feedback.

#include <array>
#include <cmath>

#include "str_core.h"
#include "str_test_harness.h"
#include "suites.h"

namespace str_test {
namespace {

// ---------------------------------------------------------------------------
// §6.1 vs §6.2: with the shipped limits the projection already keeps |B(1)| at or
// above 1e-3, so the 1e-6 freeze is defence in depth.  Both behaviours are
// asserted: (a) default limits -> no freeze, B(1) == 1e-3; (b) projection's lower
// bound disabled -> the freeze fires instead of dividing by ~0.
// ---------------------------------------------------------------------------
void check_bsum_guard() {
  str_core::Str2d reg;
  const str_core::Str2d::Vec zero_b{{-1.5, 0.6, 0.0, 0.0}};
  reg.init_estimator(1.0, 100.0, zero_b);
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  reg.begin_tick(0.0, 0.0);
  STR_CHECK(reg.update_estimator());
  STR_CHECK_NEAR(reg.theta_hat()[2] + reg.theta_hat()[3], 1e-3, 1e-15);
  (void)reg.compute_control(50.0);
  STR_CHECK(!reg.frozen());

  // (b) disable the projection bound and let |B(1)| = 0 reach the guard.
  str_core::Str2d reg2;
  reg2.init_estimator(1.0, 100.0, zero_b);
  reg2.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  auto limits = str_core::GuardLimits<double>::defaults();
  limits.bsum_min = 0.0;  // projection no longer lifts B(1)
  reg2.set_guard_limits(limits);
  reg2.begin_tick(0.0, 0.0);
  STR_CHECK(reg2.update_estimator());
  STR_CHECK_NEAR(reg2.theta_hat()[2] + reg2.theta_hat()[3], 0.0, 0.0);
  const double u = reg2.compute_control(50.0);
  STR_CHECK((reg2.flags() & str_core::kFlagBsumGuard) != 0);
  STR_CHECK(reg2.frozen());
  STR_CHECK_NEAR(u, 0.0, 0.0);  // no singular solution is emitted
}

// ---------------------------------------------------------------------------
// §6.2 det guard.  theta = [2, 1, 1, 1] has det = b1^2 - a1 b0 b1 + a2 b0^2 = 0
// and lies on the boundary of the stability triangle, so projection leaves it
// alone and the guard has to catch it.
// ---------------------------------------------------------------------------
void check_det_guard() {
  str_core::Str2d reg;
  const str_core::Str2d::Vec singular{{2.0, 1.0, 1.0, 1.0}};
  reg.init_estimator(1.0, 100.0, singular);
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  reg.begin_tick(0.0, 0.0);
  const double u = reg.compute_control(50.0);
  STR_CHECK((reg.flags() & str_core::kFlagDetGuard) != 0);
  STR_CHECK(reg.frozen());
  STR_CHECK_NEAR(u, 0.0, 0.0);
  STR_CHECK(str_test::is_finite(u));

  // "Freeze means keep the last valid gains": after a valid solve, a singular
  // sample must reuse them rather than zeroing the controller.
  str_core::Str2d reg2;
  const str_core::Str2d::Vec good{{-1.5, 0.6, 0.02, 0.011}};
  reg2.init_estimator(1.0, 100.0, good);
  reg2.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  reg2.begin_tick(0.0, 0.0);
  (void)reg2.compute_control(1e6);
  const double s0_before = reg2.s()[0];
  STR_CHECK(std::fabs(s0_before) > 1e-6);

  reg2.set_theta(singular);
  reg2.begin_tick(0.0, 0.0);
  (void)reg2.compute_control(1e6);
  STR_CHECK((reg2.flags() & str_core::kFlagDetGuard) != 0);
  STR_CHECK_NEAR(reg2.s()[0], s0_before, 0.0);  // gains frozen, not zeroed
}

// ---------------------------------------------------------------------------
// §6.2 |v| guard: a wildly out-of-range command must not be emitted.
// ---------------------------------------------------------------------------
void check_v_guard() {
  str_core::Str2d reg;
  const str_core::Str2d::Vec theta0{{-1.5, 0.6, 0.02, 0.011}};
  reg.init_estimator(1.0, 100.0, theta0);
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  reg.begin_tick(1e12, 0.0);  // absurd reference -> |v| >> 1e6
  const double u = reg.compute_control(1e9);
  STR_CHECK((reg.flags() & str_core::kFlagVGuard) != 0);
  STR_CHECK(reg.frozen());
  STR_CHECK(str_test::is_finite(u));
  STR_CHECK(std::fabs(u) <= 1e6);
  // With no earlier valid command the guard holds the previous design (0).
  STR_CHECK_NEAR(u, 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// §6.3 non-finite sample input + latched safe mode + recovery via resetdata().
// ---------------------------------------------------------------------------
void check_non_finite_sample_and_safe_mode() {
  str_core::Str2d reg;
  const str_core::Str2d::Vec theta0{{-1.5, 0.6, 0.02, 0.011}};
  reg.init_estimator(0.99, 5.0, theta0);
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);

  STR_CHECK(!reg.begin_tick(0.0, std::nan("")));
  STR_CHECK(!reg.sample_accepted());
  STR_CHECK((reg.flags() & str_core::kFlagNonFinite) != 0);
  STR_CHECK(reg.consecutive_failures() == 1);

  // An Inf reference is rejected the same way.
  STR_CHECK(!reg.begin_tick(HUGE_VAL, 0.0));
  STR_CHECK(reg.consecutive_failures() == 2);

  // 47 more bad samples -> 49 consecutive failures, still not safe mode.
  for (int k = 0; k < 47; ++k) (void)reg.begin_tick(0.0, std::nan(""));
  STR_CHECK(reg.consecutive_failures() == 49);
  STR_CHECK(!reg.safe_mode());

  // The 50th trips safe mode (§6.3).
  (void)reg.begin_tick(0.0, std::nan(""));
  STR_CHECK(reg.safe_mode());
  STR_CHECK(reg.consecutive_failures() == 50);
  STR_CHECK((reg.flags() & str_core::kFlagSafeMode) != 0);

  // Safe mode outputs a zero differential command and latches: a *good* sample
  // afterwards must not resume control.
  STR_CHECK(reg.begin_tick(0.0, 0.0) == false);
  STR_CHECK_NEAR(reg.compute_control(50.0), 0.0, 0.0);
  STR_CHECK_NEAR(reg.compute_control(50.0), 0.0, 0.0);
  STR_CHECK(reg.safe_mode());

  // Only resetdata() clears it (§6.3).
  reg.resetdata();
  STR_CHECK(!reg.safe_mode());
  STR_CHECK(reg.begin_tick(0.0, 0.0));
  (void)reg.compute_control(50.0);
  STR_CHECK(!reg.safe_mode());
}

// Non-finite state must be rolled back to the last valid snapshot, and the
// estimator must keep working afterwards.
void check_non_finite_state_rollback() {
  str_core::Str2d reg;
  const str_core::Str2d::Vec theta0{{-1.5, 0.6, 0.02, 0.011}};
  reg.init_estimator(1.0, 100.0, theta0);
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);

  // One clean step, then poison theta.
  reg.begin_tick(0.1, 0.0);
  STR_CHECK(reg.update_estimator());
  const str_core::Str2d::Vec clean = reg.theta_hat();
  const auto clean_P = reg.covariance();

  str_core::Str2d::Vec poison = clean;
  poison[0] = std::nan("");
  reg.set_theta(poison);
  reg.begin_tick(0.1, 0.0);
  STR_CHECK(!reg.update_estimator());
  STR_CHECK((reg.flags() & str_core::kFlagNonFinite) != 0);
  for (int i = 0; i < 4; ++i) {
    STR_CHECK_NEAR(reg.theta_hat()[static_cast<std::size_t>(i)],
                   clean[static_cast<std::size_t>(i)], 0.0);
    for (int j = 0; j < 4; ++j) STR_CHECK_NEAR(reg.covariance()[i][j], clean_P[i][j], 0.0);
  }
  STR_CHECK(!reg.safe_mode());

  // ... and the estimator still converges away from the poisoned sample.
  reg.begin_tick(0.1, 0.0);
  STR_CHECK(reg.update_estimator());
}

// ---------------------------------------------------------------------------
// §6.4 saturation + realised-feedback consistency.
// ---------------------------------------------------------------------------
void check_saturation_and_actual_feedback() {
  const str_core::Str2d::Vec theta0{{-1.5, 0.6, 0.02, 0.011}};
  str_core::Str2d reg;
  reg.init_estimator(1.0, 100.0, theta0);
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);

  // First tick: prime the gains, then ask for a command far beyond u_max.
  reg.begin_tick(0.0, 0.0);
  (void)reg.compute_control(1e6);
  reg.commit_actual(0.0);

  reg.begin_tick(1000.0, 0.0);
  const double u = reg.compute_control(25.0);
  STR_CHECK_NEAR(u, 25.0, 0.0);  // clamped
  STR_CHECK((reg.flags() & str_core::kFlagSaturated) != 0);
  STR_CHECK(reg.last_design_command() > 25.0);  // design value was larger

  // The realised command is what goes back into the regressor.
  reg.commit_actual(17.5);
  const auto& uh = reg.u_history();
  STR_CHECK_NEAR(uh[0], 17.5, 0.0);

  // Next tick: the control law must use 17.5 as u(k-1), not the design 1000ish.
  reg.begin_tick(1000.0, 0.0);
  (void)reg.compute_control(1e6);
  const auto& u2 = reg.u_history();
  STR_CHECK_NEAR(u2[1], 17.5, 0.0);
  STR_CHECK_NEAR(u2[2], 0.0, 0.0);

  // y_des history is not exposed (it is an input, not state), so the anti-windup
  // identity of §6.4 (v = -r1 u(k-1) + T y_des - S y) is checked differentially:
  // two regulators whose committed u differs by du must differ by exactly -r1*du.
  str_core::Str2d a;
  str_core::Str2d b;
  a.init_estimator(1.0, 100.0, theta0);
  b.init_estimator(1.0, 100.0, theta0);
  a.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  b.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  a.begin_tick(0.0, 0.0);
  b.begin_tick(0.0, 0.0);
  (void)a.compute_control(1e6);
  (void)b.compute_control(1e6);
  a.commit_actual(0.0);
  b.commit_actual(0.0);
  a.begin_tick(0.0, 1.0);
  b.begin_tick(0.0, 1.0);
  (void)a.compute_control(1e6);
  (void)b.compute_control(1e6);
  const double r1 = a.r()[0];
  a.commit_actual(10.0);
  b.commit_actual(0.0);
  a.begin_tick(0.0, 1.0);
  b.begin_tick(0.0, 1.0);
  const double ua = a.compute_control(1e6);
  const double ub = b.compute_control(1e6);
  STR_CHECK_MSG(std::fabs((ua - ub) + r1 * 10.0) <= 1e-9 * (std::fabs(r1) * 10.0 + 1.0),
                "du = %.12g, expected %.12g", ua - ub, -r1 * 10.0);
}

// A non-finite realised command must not poison the history.
void check_non_finite_actual() {
  str_core::Str2d reg;
  const str_core::Str2d::Vec theta0{{-1.5, 0.6, 0.02, 0.011}};
  reg.init_estimator(1.0, 100.0, theta0);
  reg.begin_tick(0.0, 0.0);
  (void)reg.compute_control(50.0);
  reg.commit_actual(3.0);
  reg.commit_actual(std::nan(""));
  STR_CHECK_NEAR(reg.u_history()[0], 3.0, 0.0);
  STR_CHECK(reg.consecutive_failures() >= 1);
}

}  // namespace

void suite_guards() {
  check_bsum_guard();
  check_det_guard();
  check_v_guard();
  check_non_finite_sample_and_safe_mode();
  check_non_finite_state_rollback();
  check_saturation_and_actual_feedback();
  check_non_finite_actual();
}

}  // namespace str_test
