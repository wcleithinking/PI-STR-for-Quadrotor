// Suite: §6.1 parameter projection.
//
// The 2nd-order rule is the Python reference (sim/python/pistr_sim/control/str.py,
// project_theta()) and is part of the golden-vector contract; the 3rd-order rule is the
// documented extension of the same function (Schur test + radial contraction).
// Checked here: containment, idempotence, "stable parameters are not modified", "an
// unstable 3rd-order model is pulled back into the unit disc", the B(1) floor (absolute
// for order 2, relative for order 3), and that the regulator really applies it before
// computing gains.

#include <array>
#include <cmath>
#include <cstddef>

#include "str_core.h"
#include "str_test_harness.h"
#include "suites.h"

namespace str_test {
namespace {

constexpr double kBsum2 = 1e-3;

std::size_t idx(int i) { return static_cast<std::size_t>(i); }

void check_order2_containment_and_idempotence() {
  const double tol = 1e-12;
  for (int i = -60; i <= 60; ++i) {
    for (int j = -60; j <= 60; ++j) {
      std::array<double, 4> th{{i * 0.1, j * 0.1, i * 0.02, -j * 0.02}};
      const std::array<double, 4> in = th;
      str_core::project_theta<double, 2>(th, kBsum2);

      STR_CHECK_MSG(std::fabs(th[0]) <= 4.0 + tol, "a1 = %.6g (in %.3g,%.3g)", th[0],
                    in[0], in[1]);
      STR_CHECK_MSG(th[1] >= -1.0 - tol, "a2 = %.6g (in %.3g,%.3g)", th[1], in[0],
                    in[1]);
      STR_CHECK_MSG(th[1] <= 2.0 + tol, "a2 = %.6g", th[1]);
      STR_CHECK_MSG(std::fabs(th[0]) <= 1.0 + th[1] + tol,
                    "a1 = %.6g outside [-1-a2, 1+a2] with a2 = %.6g (in %.3g,%.3g)",
                    th[0], th[1], in[0], in[1]);
      STR_CHECK(std::fabs(th[2]) <= 10.0 + tol);
      STR_CHECK(std::fabs(th[3]) <= 10.0 + tol);
      STR_CHECK_MSG(std::fabs(th[2] + th[3]) >= kBsum2 - tol, "|B(1)| = %.6g",
                    th[2] + th[3]);

      std::array<double, 4> again = th;
      str_core::project_theta<double, 2>(again, kBsum2);
      STR_CHECK_MSG(again == th, "not idempotent at (%.4g,%.4g)", in[0], in[1]);
    }
  }
}

void check_order2_stable_points_are_untouched() {
  // A stable model inside the region, and the marginal integrator boundary
  // a1 + a2 = -1 that the real rate channel sits on: both must be fixed points.
  const std::array<double, 4> stable{{-1.5, 0.6, 0.02, 0.011}};
  std::array<double, 4> th = stable;
  str_core::project_theta<double, 2>(th, kBsum2);
  for (int i = 0; i < 4; ++i) STR_CHECK_NEAR(th[idx(i)], stable[idx(i)], 0.0);

  const std::array<double, 4> integrator{{-1.115, 0.115, 0.028, 0.014}};
  th = integrator;
  str_core::project_theta<double, 2>(th, kBsum2);
  STR_CHECK_NEAR(th[0], integrator[0], 0.0);
  STR_CHECK_NEAR(th[1], integrator[1], 0.0);
  STR_CHECK_NEAR(th[0] + th[1], -1.0, 1e-12);
}

void check_order3_projection() {
  const double tol = 1e-9;

  // (1) idempotence over a grid of denominator coefficients
  for (int i = -20; i <= 20; ++i) {
    for (int j = -20; j <= 20; ++j) {
      for (int k = -12; k <= 12; k += 3) {
        std::array<double, 6> th{{i * 0.3, j * 0.3, k * 0.1, 0.02, 0.02, 0.02}};
        std::array<double, 6> in = th;
        str_core::project_theta<double, 3>(th, 1e-3);
        std::array<double, 6> again = th;
        str_core::project_theta<double, 3>(again, 1e-3);
        STR_CHECK_MSG(again == th,
                      "order-3 projection not idempotent at (%.3g,%.3g,%.3g)", in[0],
                      in[1], in[2]);
      }
    }
  }

  // (2) a stable model is not modified
  const std::array<double, 6> stable{{-1.9, 1.25, -0.30, 0.015, 0.014, 0.013}};
  {
    std::array<double, 3> a{{stable[0], stable[1], stable[2]}};
    STR_CHECK(str_core::is_schur_polynomial(a));
  }
  std::array<double, 6> th = stable;
  str_core::project_theta<double, 3>(th, 1e-3);
  for (int i = 0; i < 6; ++i) {
    STR_CHECK_MSG(th[idx(i)] == stable[idx(i)],
                  "stable order-3 model modified at index %d: %.9g -> %.9g", i,
                  stable[idx(i)], th[idx(i)]);
  }

  // (3) a known unstable model is contracted back into the unit disc
  std::array<double, 6> unstable{{-0.5, 0.3, -0.90, 0.02, 0.02, 0.02}};
  std::array<double, 3> a_before{{unstable[0], unstable[1], unstable[2]}};
  STR_CHECK(!str_core::is_schur_polynomial(a_before));
  str_core::project_theta<double, 3>(unstable, 1e-3);
  std::array<double, 3> a_after{{unstable[0], unstable[1], unstable[2]}};
  STR_CHECK_MSG(str_core::is_schur_polynomial(a_after),
                "order-3 projection left an unstable denominator: a = [%.6g %.6g %.6g]",
                a_after[0], a_after[1], a_after[2]);

  // (4) box: a2 to +-4, a3 to +-1, b to +-10
  std::array<double, 6> box{{7.0, 9.0, 3.0, 100.0, -100.0, 5.0}};
  str_core::project_theta<double, 3>(box, 1e-3);
  STR_CHECK(box[0] <= 4.0 + tol && box[0] >= -4.0 - tol);
  STR_CHECK(box[1] <= 4.0 + tol && box[1] >= -4.0 - tol);
  STR_CHECK(box[2] <= 1.0 + tol && box[2] >= -1.0 - tol);
  STR_CHECK(box[3] <= 10.0 + tol && box[3] >= -10.0 - tol);
}

// The 3rd-order B(1) floor is relative: 1% of |B(1)| at initialisation, fixed after
// that.  The SI Mode I channel has B(1) ~ 2.1e-3, so the absolute 1e-3 would sit within
// a factor of two of the truth and would bias the estimate.
void check_order3_relative_bsum() {
  str_core::Str3d reg;
  const str_core::Str3d::Vec theta0{{-1.9, 1.25, -0.30, 0.0007, 0.0007, 0.0007}};
  reg.init_estimator(1.0, 100.0, theta0);  // B(1) = 2.1e-3 -> floor 2.1e-5
  reg.set_reference(std::array<double, 3>{{-1.44, 0.77, -0.138}},
                    std::array<double, 2>{{0.8, 0.16}});

  // B(1) = 1e-3 is far above the relative floor: the estimate must survive untouched.
  str_core::Str3d::Vec th = theta0;
  th[3] = 0.0004;
  th[4] = 0.0003;
  th[5] = 0.0003;
  reg.set_theta(th);
  reg.begin_tick(0.0, 0.0);
  STR_CHECK(reg.update_estimator());
  STR_CHECK_MSG((reg.flags() & str_core::kFlagProjected) == 0,
                "the absolute 1e-3 floor was applied: B(1) = %.6g",
                reg.theta_hat()[3] + reg.theta_hat()[4] + reg.theta_hat()[5]);
  STR_CHECK_NEAR(reg.theta_hat()[3] + reg.theta_hat()[4] + reg.theta_hat()[5], 1e-3,
                 1e-15);

  // Below the relative floor it is lifted to exactly 1% of the initial B(1).
  str_core::Str3d::Vec th2 = theta0;
  th2[3] = 1e-9;
  th2[4] = 1e-9;
  th2[5] = 1e-9;
  reg.set_theta(th2);
  reg.begin_tick(0.0, 0.0);
  STR_CHECK(reg.update_estimator());
  STR_CHECK((reg.flags() & str_core::kFlagProjected) != 0);
  STR_CHECK_NEAR(reg.theta_hat()[3] + reg.theta_hat()[4] + reg.theta_hat()[5], 2.1e-5,
                 1e-15);
}

void check_regulator_projects_wild_estimates() {
  str_core::Str2d reg;
  const str_core::Str2d::Vec theta0{{-1.5, 0.6, 0.02, 0.011}};
  reg.init_estimator(1.0, 100.0, theta0);
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  const str_core::Str2d::Vec wild{{7.5, 1.9, 250.0, 5.0}};
  reg.set_theta(wild);
  reg.begin_tick(0.0, 0.0);
  STR_CHECK(reg.update_estimator());
  const auto& th = reg.theta_hat();
  STR_CHECK(th[0] >= -4.0 && th[0] <= 4.0);
  STR_CHECK(th[1] >= -1.0 && th[1] <= 2.0);
  STR_CHECK(std::fabs(th[0]) <= 1.0 + th[1] + 1e-12);
  STR_CHECK(th[2] >= -10.0 && th[2] <= 10.0);
  STR_CHECK(th[3] >= -10.0 && th[3] <= 10.0);
  STR_CHECK(std::fabs(th[2] + th[3]) >= 1e-3);
  STR_CHECK((reg.flags() & str_core::kFlagProjected) != 0);
  const double u = reg.compute_control(50.0);
  STR_CHECK(str_test::is_finite(u));
  STR_CHECK(!reg.frozen());
}

// In the pathological corner (b1 pinned at a box limit) the B(1) floor wins over the b0
// box, exactly as in the Python reference: well-posedness of 1/B(1) matters more than
// keeping b0 a hair inside +-10, and the following solve then has a meaningful
// denominator instead of being frozen.
void check_bsum_wins_over_b0_box() {
  // b0 = +10 and b1 = -10 are both inside the box but make B(1) exactly 0.
  std::array<double, 4> th{{-1.5, 0.6, 10.0, -10.0}};
  str_core::project_theta<double, 2>(th, kBsum2);
  STR_CHECK_NEAR(th[2], 1e-3 + 10.0, 1e-12);
  STR_CHECK_MSG(th[2] + th[3] >= kBsum2 * (1.0 - 1e-12), "B(1) = %.17g",
                th[2] + th[3]);
}

}  // namespace

void suite_projection() {
  check_order2_containment_and_idempotence();
  check_order2_stable_points_are_untouched();
  check_order3_projection();
  check_order3_relative_bsum();
  check_regulator_projects_wild_estimates();
  check_bsum_wins_over_b0_box();
}

}  // namespace str_test
