// Suite: behaviour equivalence with the legacy firmware implementations.
//
// The two regulators are driven with *identical* inputs, tick by tick, and must
// produce identical commands and identical parameter estimates.  That is the
// concrete form of the constraint "integration must be behaviour equivalent; the
// only allowed change is the §6 guards".
//
// The plant runs open loop (exogenous excitation) and both regulators are fed the
// same y(k) and the same realised command, so this is a pure controller/estimator
// equivalence: no feedback-stability effects, no saturation.  Whenever a §6 guard
// fires the two are *expected* to differ, hence the explicit assertion that no
// guard fires on these trajectories -- that is the precondition of the claim.

#include <algorithm>
#include <array>
#include <cmath>

#include "arx_plant.h"
#include "legacy_reference.h"
#include "str_core.h"
#include "str_test_harness.h"
#include "suites.h"

namespace str_test {
namespace {

constexpr unsigned kGuardedFlags =
    str_core::kFlagProjected | str_core::kFlagNonFinite |
    str_core::kFlagBsumGuard | str_core::kFlagDetGuard | str_core::kFlagVGuard |
    str_core::kFlagSafeMode;

// ---------------------------------------------------------------------------
// Mode II (2nd order, PI-STR inner loop).  Legacy order per tick:
//   update_output -> run_estimator -> run_controller -> (later) get_u_actual
// ---------------------------------------------------------------------------
void check_mode2_equivalence() {
  // True plant (§3 structure) and an initial guess that is *inside* the §6.1
  // stability triangle (0.95 x truth).  The shipped analytic guess is outside it --
  // see check_shipped_guess_is_outside_stability_region() -- and then the projection
  // legitimately changes the controller from the first tick.
  const std::array<double, 4> truth{{-1.5, 0.6, 0.02, 0.011}};
  const double zeta = 0.7, wn = 5.0, mu = 0.0, h = 0.04;
  const double kGuessScale = 0.95;
  const std::array<double, 4> guess{{truth[0] * kGuessScale, truth[1] * kGuessScale,
                                     truth[2] * kGuessScale, truth[3] * kGuessScale}};

  // lambda = 1 on purpose: with lambda < 1 the §4.3 P clamp (a spec requirement the
  // legacy code does not have) starts biting from the very first tick, because P0 is
  // initialised at Pmax.  The clamp is a deliberate, documented difference; this test
  // pins the regime where it does not bite so that the algebra can be compared
  // exactly.
  str_core::Str2d core;
  core.init_estimator(1.0, 5.0,
                      str_core::Str2d::Vec{guess[0], guess[1], guess[2], guess[3]});
  core.set_second_order_reference(zeta, wn, mu, h);

  legacy::Legacy2<double> ref;
  ref.init_estimator(1.0, 5.0, guess[0], guess[1], guess[2], guess[3]);
  ref.resetdata();
  ref.init_controller(zeta, wn, mu, h);

  ArxPlant<double, 2> plant(truth);
  Lcg rng(4242u);

  const double u_max = 50.0;
  double max_u_err = 0.0, max_u_ref = 0.0;
  double max_theta_err = 0.0, max_theta_ref = 0.0;
  unsigned guarded_ticks = 0;

  for (int k = 0; k < 1200; ++k) {
    const double y = plant.y();
    const double y_des = (k / 200) % 2 == 0 ? 0.10 : -0.06;

    STR_CHECK(core.begin_tick(y_des, y));
    STR_CHECK(core.update_estimator());
    const double u_core = core.compute_control(u_max);
    core.commit_actual(u_core);

    ref.update_output(y_des, y);
    ref.run_estimator();
    const double u_ref = ref.run_controller(u_max);
    ref.get_u_actual(u_core);  // identical realised value for both histories

    if ((core.flags() & kGuardedFlags) != 0) ++guarded_ticks;

    max_u_err = std::max(max_u_err, std::fabs(u_core - u_ref));
    max_u_ref = std::max(max_u_ref, std::fabs(u_ref));
    for (int i = 0; i < 4; ++i) {
      max_theta_err = std::max(
          max_theta_err,
          std::fabs(
              static_cast<double>(core.theta_hat()[static_cast<std::size_t>(i)]) -
              ref.hattheta[i]));
      max_theta_ref = std::max(max_theta_ref, std::fabs(ref.hattheta[i]));
    }
    // Both regulators see the same realised command, and the plant is driven by it:
    // a genuine closed loop with identical inputs.
    (void)plant.step(u_core);
  }

  STR_CHECK_MSG(guarded_ticks == 0,
                "%u ticks hit a §6 guard; equivalence is only claimed on the "
                "guard-free path",
                guarded_ticks);
  STR_CHECK_MSG(max_u_err / (max_u_ref + 1.0) <= 1e-9,
                "max |du| = %.3g (reference scale %.3g)", max_u_err, max_u_ref);
  STR_CHECK_MSG(max_theta_err / (max_theta_ref + 1.0) <= 1e-9,
                "max |dtheta| = %.3g (reference scale %.3g)", max_theta_err,
                max_theta_ref);
  STR_CHECK(std::fabs(core.s()[0]) > 1e-6);  // sanity: gains really solved
  // The estimator must actually be doing work on this trajectory.
  STR_CHECK(std::fabs(static_cast<double>(core.theta_hat()[0]) - guess[0]) > 1e-3);
}

// The Mode II **initial guess** repair (initialisation only, not a control law).
//
// Before: `a2 = exp(+atemp*h)` with atemp = 54 and `* STR_period` (milliseconds) where
// h = 0.04 s belongs, giving theta0 ~ [-9.67, +8.67, +47.58, -411.24]: an unstable,
// non-minimum-phase starting model with a negative DC gain, outside every §6.1 limit
// (so the projection immediately rewrote it).
// After: the §3 analytic form with the exponent sign and the seconds unit fixed, which
// is what this test pins -- and which lands exactly on the marginal integrator boundary
// a1 + a2 = -1, i.e. a fixed point of the projection.
void check_initial_guess_repair() {
  const double h = 0.04;
  const double atemp = 54.0, btemp = 64.0;

  // (a) the legacy (pre-fix) numbers
  const double a2_old = std::exp(atemp * h);
  const double a1_old = -1.0 - a2_old;
  const double b0_old = btemp / atemp * 40.0 + btemp / (atemp * atemp) * (a2_old - 1.0);
  const double b1_old =
      btemp / atemp * (1.0 - a2_old) * (1.0 - a2_old) / atemp - b0_old * a2_old;
  STR_CHECK_MSG(std::fabs(a2_old - 8.6711) < 1e-3, "legacy a2 = %.6f", a2_old);
  STR_CHECK_MSG(std::fabs(a1_old + 9.6711) < 1e-3, "legacy a1 = %.6f", a1_old);
  STR_CHECK_MSG(b0_old > 40.0 && b1_old < -400.0,
                "legacy b0 = %.4f, b1 = %.4f", b0_old, b1_old);
  STR_CHECK(a1_old * a1_old > 4.0 * 4.0);  // way outside the a1 box

  // ... and the projection has to pull it back inside the region
  std::array<double, 4> legacy_guess{{a1_old, a2_old, b0_old, b1_old}};
  str_core::project_theta<double, 2>(legacy_guess, 1e-3);
  STR_CHECK(std::fabs(legacy_guess[0]) <= 4.0 + 1e-12);
  STR_CHECK(std::fabs(legacy_guess[0]) <= 1.0 + legacy_guess[1] + 1e-12);

  // (b) the fixed guess: §3 analytic form, sign and units corrected
  const double a2_new = std::exp(-atemp * h);
  const double a1_new = -1.0 - a2_new;
  const double b0_new = btemp / atemp * h + btemp / (atemp * atemp) * (a2_new - 1.0);
  const double b1_new =
      btemp / atemp * (1.0 - a2_new) * (1.0 - a2_new) / atemp - b0_new * a2_new;
  STR_CHECK_NEAR(a2_new, 0.11533, 1e-5);
  STR_CHECK_NEAR(a1_new, -1.11533, 1e-5);
  STR_CHECK_NEAR(b0_new, 0.02800, 1e-5);
  STR_CHECK_NEAR(b1_new, 0.013950, 1e-5);
  STR_CHECK_NEAR(a1_new + a2_new, -1.0, 1e-12);  // marginal integrator boundary
  STR_CHECK(std::fabs(b0_new + b1_new) > 1e-3);

  std::array<double, 4> fixed_guess{{a1_new, a2_new, b0_new, b1_new}};
  const std::array<double, 4> before = fixed_guess;
  str_core::project_theta<double, 2>(fixed_guess, 1e-3);
  for (int i = 0; i < 4; ++i) {
    STR_CHECK_MSG(fixed_guess[std::size_t(i)] == before[std::size_t(i)],
                  "the fixed initial guess is modified by the projection at index %d "
                  "(%.9g -> %.9g)", i, before[std::size_t(i)],
                  fixed_guess[std::size_t(i)]);
  }
}

// The historical `s0 = xx[2]; s1 = xx[3];` bug, quantified on the same trajectory.
void check_legacy_s_index_bug_magnitude() {
  const std::array<double, 4> truth{{-1.5, 0.6, 0.02, 0.011}};
  const double zeta = 0.7, wn = 5.0, h = 0.04;
  const double alpha = std::exp(-zeta * wn * h);
  const std::array<double, 4> guess{{-1.0 - alpha, alpha * alpha, 0.018, 0.010}};

  legacy::Legacy2<double> ref;
  ref.init_estimator(1.0, 5.0, guess[0], guess[1], guess[2], guess[3]);
  ref.resetdata();
  ref.init_controller(zeta, wn, 0.0, h);

  ArxPlant<double, 2> plant(truth);
  Lcg rng(2024u);
  double max_diff = 0.0, max_ref = 0.0;
  for (int k = 0; k < 600; ++k) {
    const double u_exc = 20.0 * rng.uniform();
    const double y = plant.y();
    const double y_des = (k / 100) % 2 == 0 ? 0.10 : -0.05;

    ref.update_output(y_des, y);
    ref.run_estimator();
    const double u_ok = ref.run_controller(50.0);

    // Rebuild the same 3x3 solve and apply the buggy index mapping: s0 = x[2]
    // (which is the *correct* s1) and s1 = x[3], an out-of-bounds read whose value
    // is undefined; 0 is the benign assumption for this comparison.
    double xx[3];
    legacy::legacy_adjugate_3(ref.hattheta[0], ref.hattheta[1], ref.hattheta[2],
                              ref.hattheta[3], ref.am1, ref.am2, ref.ao1, xx);
    const double beta =
        (1.0 + ref.am1 + ref.am2) / (ref.hattheta[2] + ref.hattheta[3]);
    const double s0_buggy = xx[2];
    const double s1_buggy = 0.0;
    const double u_buggy = -ref.ao1 * ref.v_compute[1] +
                           beta * ref.y_desired[0] +
                           beta * ref.ao1 * ref.y_desired[1] -
                           s0_buggy * ref.y_measure[0] -
                           s1_buggy * ref.y_measure[1] +
                           (ref.ao1 - xx[0]) * ref.u_compute[1];

    max_diff = std::max(max_diff, std::fabs(u_buggy - u_ok));
    max_ref = std::max(max_ref, std::fabs(u_ok));
    ref.get_u_actual(u_ok);
    (void)plant.step(u_exc);
  }
  STR_CHECK_MSG(max_diff / (max_ref + 1.0) > 0.05,
                "the historical index bug changes the command by only %.3g "
                "(scale %.3g)",
                max_diff, max_ref);
}

// ---------------------------------------------------------------------------
// Mode I (3rd order, single loop).  Legacy order per tick:
//   control law -> estimator -> history shift
// ---------------------------------------------------------------------------
void check_mode3_equivalence() {
  const std::array<double, 6> truth{{-1.9, 1.25, -0.30, 0.015, 0.014, 0.013}};
  const double kGuessScale = 0.95;
  const std::array<double, 6> guess{{truth[0] * kGuessScale, truth[1] * kGuessScale,
                                     truth[2] * kGuessScale, truth[3] * kGuessScale,
                                     truth[4] * kGuessScale, truth[5] * kGuessScale}};
  const std::array<double, 3> am{{-1.4431986644, 0.7708989340, -0.1380692400}};
  const std::array<double, 2> ao{{0.8, 0.16}};

  str_core::Str3d core;
  core.init_estimator(1.0, 100.0,
                      str_core::Str3d::Vec{guess[0], guess[1], guess[2], guess[3],
                                           guess[4], guess[5]});
  core.set_reference(am, ao);

  legacy::Legacy3<double> ref;
  const double theta0[6] = {guess[0], guess[1], guess[2],
                            guess[3], guess[4], guess[5]};
  ref.init_estimator(1.0, 100.0, theta0);
  ref.init_controller(am[0], am[1], am[2], ao[0], ao[1]);

  ArxPlant<double, 3> plant(truth);
  // Kick the plant so that the regulator has something to reject (Mode I regulates
  // the measured angle to the -- unwritten -- zero reference).
  (void)plant.step(0.5);

  // Unsaturated regime on purpose: with ao1 = 0.8 / ao2 = 0.16 (non-zero in Mode I)
  // the legacy law feeds the *design* command into the -ao terms while §5/§6.4
  // requires the realised one, so the two controllers legitimately differ once the
  // command saturates.  Saturation handling is covered by suite_guards.
  const double u_max = 1e6;
  double max_u_err = 0.0, max_u_ref = 0.0;
  double max_theta_err = 0.0, max_theta_ref = 0.0;
  unsigned guarded_ticks = 0;

  for (int k = 0; k < 1200; ++k) {
    const double y = plant.y();
    // Mode I feeds the STR the angle reference directly; the shipped sketch never
    // writes y_desire[], so the reference term is identically zero -- preserved.
    const double y_des = 0.0;

    STR_CHECK(core.begin_tick(y_des, y));
    const double u_core = core.compute_control(u_max);
    core.commit_actual(u_core);
    STR_CHECK(core.update_estimator());

    const double u_ref = ref.control_law(y, u_max);
    ref.estimator_and_shift(u_core);  // identical realised value for both histories

    if ((core.flags() & kGuardedFlags) != 0) ++guarded_ticks;

    max_u_err = std::max(max_u_err, std::fabs(u_core - u_ref));
    max_u_ref = std::max(max_u_ref, std::fabs(u_ref));
    for (int i = 0; i < 6; ++i) {
      max_theta_err = std::max(
          max_theta_err,
          std::fabs(
              static_cast<double>(core.theta_hat()[static_cast<std::size_t>(i)]) -
              ref.hattheta[i]));
      max_theta_ref = std::max(max_theta_ref, std::fabs(ref.hattheta[i]));
    }
    // Both regulators see the same realised command, and the plant is driven by it:
    // a genuine closed loop with identical inputs.
    (void)plant.step(u_core);
  }

  STR_CHECK_MSG(guarded_ticks == 0,
                "%u ticks hit a §6 guard; equivalence is only claimed on the "
                "guard-free path",
                guarded_ticks);
  STR_CHECK_MSG(max_u_err / (max_u_ref + 1.0) <= 1e-9,
                "max |du| = %.3g (reference scale %.3g)", max_u_err, max_u_ref);
  STR_CHECK_MSG(max_theta_err / (max_theta_ref + 1.0) <= 1e-9,
                "max |dtheta| = %.3g (reference scale %.3g)", max_theta_err,
                max_theta_ref);
  STR_CHECK(std::fabs(core.s()[1]) > 1e-9);
}

// Mode I: the ao1/ao2 <-> u(k-1)/u(k-2) pairing.
//
// The MATLAB Mode-I script has a historical slip: the second `Ao` term reads
// `u_TEMP(t - period)` instead of `u_TEMP(t - 2*period)`.  The firmware's C code (and
// therefore str_core) pairs ao1 with u(k-1) and ao2 with u(k-2); this test pins that by
// evaluating the control law against an independently written expression, and shows
// that the historical slip would give a different command.
void check_mode1_ao_index_pairing() {
  const std::array<double, 6> theta0{{-1.9, 1.25, -0.30, 0.015, 0.014, 0.013}};
  const std::array<double, 3> am{{-1.4431986644, 0.7708989340, -0.1380692400}};
  const std::array<double, 2> ao{{0.8, 0.16}};

  str_core::Str3d reg;
  reg.init_estimator(1.0, 100.0,
                     str_core::Str3d::Vec{theta0[0], theta0[1], theta0[2], theta0[3],
                                          theta0[4], theta0[5]});
  reg.set_reference(am, ao);

  // Three ticks with distinct y and u histories so every index is pinned:
  //   y(k) = 0.03, y(k-1) = 0.02, y(k-2) = 0.01
  //   u(k-1) = 3.0, u(k-2) = 2.0
  STR_CHECK(reg.begin_tick(0.0, 0.01));
  reg.commit_actual(2.0);
  STR_CHECK(reg.begin_tick(0.0, 0.02));
  reg.commit_actual(3.0);
  STR_CHECK(reg.begin_tick(0.0, 0.03));
  const double u = reg.compute_control(1e6);
  const double v = reg.last_design_command();

  const double r1 = reg.r()[0], r2 = reg.r()[1];
  const double s0 = reg.s()[0], s1 = reg.s()[1], s2 = reg.s()[2];
  const double y0 = 0.03, y1 = 0.02, y2 = 0.01, u1 = 3.0, u2 = 2.0;

  const double v_correct = -r1 * u1 - r2 * u2 - s0 * y0 - s1 * y1 - s2 * y2;
  STR_CHECK_NEAR(v, v_correct, 1e-9);
  STR_CHECK_NEAR(u, v_correct, 1e-9);

  // The MATLAB-style slip (ao2 paired with u(k-1)) must give something different, so
  // this assertion really distinguishes the two pairings.
  const double v_matlab_slip = -r1 * u1 - r2 * u1 - s0 * y0 - s1 * y1 - s2 * y2;
  STR_CHECK_MSG(std::fabs(v - v_matlab_slip) > 1e-6,
                "the ao2/u(k-1) slip is indistinguishable here (du = %.3g)",
                v - v_matlab_slip);
}

}  // namespace

void suite_equivalence() {
  check_mode2_equivalence();
  check_initial_guess_repair();
  check_legacy_s_index_bug_magnitude();
  check_mode3_equivalence();
  check_mode1_ao_index_pairing();
}

}  // namespace str_test
