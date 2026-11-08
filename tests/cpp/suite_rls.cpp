// Suite: §4 RLS (forgetting factor, P symmetrisation, P diagonal clamp,
// rank-1 update) and §4.5 resetdata().

#include <array>
#include <cmath>

#include "arx_plant.h"
#include "legacy_reference.h"
#include "str_core.h"
#include "str_test_harness.h"
#include "suites.h"

namespace str_test {
namespace {

// Drive the identifier open-loop with a PRBS and check it recovers the plant.
template <int Order, typename Reg>
void check_convergence(const std::array<double, 2 * Order>& truth,
                       const std::array<double, 2 * Order>& guess,
                       int ticks = 4000, double rel_tol = 0.02) {
  Reg reg;
  typename Reg::Vec theta0{};
  using Real = typename Reg::Vec::value_type;
  for (int i = 0; i < 2 * Order; ++i)
    theta0[static_cast<std::size_t>(i)] =
        static_cast<Real>(guess[static_cast<std::size_t>(i)]);
  reg.init_estimator(Real(1), Real(100), theta0);

  ArxPlant<double, Order> plant(truth);
  Lcg rng(2024u);
  for (int k = 0; k < ticks; ++k) {
    const double u = 50.0 * rng.uniform();
    const double y = plant.y();
    reg.begin_tick(0.0, y);
    // Use the recorded input as the regressor input.
    reg.commit_actual(u);
    reg.update_estimator();
    (void)plant.step(u);
  }

  const auto& th = reg.theta_hat();
  for (int i = 0; i < 2 * Order; ++i) {
    const double want = truth[static_cast<std::size_t>(i)];
    const double got = static_cast<double>(th[static_cast<std::size_t>(i)]);
    const double tol = std::fabs(want) * rel_tol + 1e-6;
    STR_CHECK_MSG(std::fabs(got - want) <= tol,
                  "theta[%d] = %.6f, want %.6f (tol %.2g)", i, got, want, tol);
  }
}

void check_p_invariants() {
  str_core::Str2d reg;
  const str_core::Str2d::Vec theta0{{-1.5, 0.6, 0.02, 0.011}};
  const double Pmax = 5.0;
  reg.init_estimator(0.99, Pmax, theta0);
  ArxPlant<double, 2> plant({{-1.5, 0.6, 0.02, 0.011}});
  Lcg rng(7u);
  for (int k = 0; k < 400; ++k) {
    const double u = 30.0 * rng.uniform();
    reg.begin_tick(0.0, plant.y());
    reg.commit_actual(u);
    reg.update_estimator();
    (void)plant.step(u);
    const auto& P = reg.covariance();
    for (int i = 0; i < 4; ++i) {
      STR_CHECK_MSG(P[i][i] >= 0.0 && P[i][i] <= Pmax + 1e-12,
                    "P[%d][%d] = %.6g out of [0, %.6g]", i, i, P[i][i], Pmax);
      for (int j = 0; j < 4; ++j) {
        // §4.2 symmetrisation, exact to floating point.
        STR_CHECK_MSG(P[i][j] == P[j][i], "P asymmetric at (%d,%d): %.17g vs %.17g",
                      i, j, P[i][j], P[j][i]);
      }
    }
  }
}

void check_resetdata_is_exact() {
  str_core::Str2d reg;
  const str_core::Str2d::Vec theta0{{-1.5, 0.6, 0.02, 0.011}};
  reg.init_estimator(0.99, 5.0, theta0);
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  ArxPlant<double, 2> plant({{-1.5, 0.6, 0.02, 0.011}});
  Lcg rng(9u);
  for (int k = 0; k < 200; ++k) {
    const double u = 20.0 * rng.uniform();
    reg.begin_tick(0.05, plant.y());
    reg.commit_actual(u);
    reg.update_estimator();
    (void)reg.compute_control(50.0);
    (void)plant.step(u);
  }
  // Stir in a couple of guards so that resetdata() has something to clear.
  for (int k = 0; k < 60; ++k) {
    reg.begin_tick(0.0, std::nan(""));
  }
  STR_CHECK(reg.safe_mode());

  reg.resetdata();
  STR_CHECK(!reg.safe_mode());
  STR_CHECK(reg.flags() == str_core::kFlagNone);
  STR_CHECK(reg.consecutive_failures() == 0);
  for (int i = 0; i < 4; ++i) {
    STR_CHECK_NEAR(reg.theta_hat()[static_cast<std::size_t>(i)],
                   static_cast<double>(theta0[static_cast<std::size_t>(i)]), 0.0);
  }
  const auto& P = reg.covariance();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const double want = (i == j) ? 5.0 : 0.0;
      STR_CHECK_NEAR(P[i][j], want, 0.0);
    }
  }
  for (int i = 0; i < str_core::Str2d::kHistory; ++i) {
    STR_CHECK_NEAR(reg.y_history()[static_cast<std::size_t>(i)], 0.0, 0.0);
    STR_CHECK_NEAR(reg.u_history()[static_cast<std::size_t>(i)], 0.0, 0.0);
  }
  STR_CHECK_NEAR(reg.last_command(), 0.0, 0.0);
  // And the regulator works again from the restored state.
  reg.begin_tick(0.1, 0.0);
  STR_CHECK(reg.update_estimator());
  STR_CHECK(str_test::is_finite(reg.compute_control(50.0)));
}

void check_symmetry_of_rank1_update() {
  // The rank-1 rewrite of §4.1 must give exactly the same theta and P as the
  // textbook (I - K phi') P form.  The reference is the legacy copy, whose
  // estimator uses the explicit matrix product.  The data come from a consistent
  // ARX plant so that the projection never bites (it would legitimately make the
  // two differ, and that case is covered by suite_projection).
  const std::array<double, 4> truth{{-1.5, 0.6, 0.02, 0.011}};
  ArxPlant<double, 2> plant(truth);
  Lcg rng(11u);

  str_core::Str2d reg;
  legacy::Legacy2<double> ref;
  const double theta0[4] = {-1.35, 0.54, 0.018, 0.0099};
  reg.init_estimator(1.0, 100.0,
                     str_core::Str2d::Vec{theta0[0], theta0[1], theta0[2], theta0[3]});
  reg.set_second_order_reference(0.7, 5.0, 0.0, 0.04);
  ref.init_estimator(1.0, 100.0, theta0[0], theta0[1], theta0[2], theta0[3]);
  ref.resetdata();
  ref.init_controller(0.7, 5.0, 0.0, 0.04);

  for (int k = 0; k < 600; ++k) {
    const double u_exc = 20.0 * rng.uniform();
    const double y = plant.y();

    ref.update_output(0.0, y);
    ref.run_estimator();
    ref.get_u_actual(u_exc);

    STR_CHECK(reg.begin_tick(0.0, y));
    reg.commit_actual(u_exc);
    STR_CHECK(reg.update_estimator());
    STR_CHECK_MSG((reg.flags() & (str_core::kFlagProjected | str_core::kFlagNonFinite |
                                  str_core::kFlagBsumGuard | str_core::kFlagDetGuard)) == 0,
                  "guard tripped at tick %d: flags = 0x%x", k, reg.flags());

    for (int i = 0; i < 4; ++i) {
      STR_CHECK_NEAR(reg.theta_hat()[static_cast<std::size_t>(i)], ref.hattheta[i],
                     1e-10);
      for (int j = 0; j < 4; ++j) {
        STR_CHECK_NEAR(reg.covariance()[i][j], ref.P[i][j], 1e-9);
      }
    }
    (void)plant.step(u_exc);
  }
}

}  // namespace

void suite_rls() {
  check_convergence<2, str_core::Str2d>({{-1.5, 0.6, 0.02, 0.011}},
                                        {{-1.3, 0.5, 0.015, 0.008}});
  // The 3rd-order ARX is far worse conditioned than the 2nd-order one (its b's are
  // identified from a much thinner information set), so the acceptance band is wider
  // and the run longer; the point of the test is "it recovers the plant", not "it
  // matches to 2% in 4000 samples".
  check_convergence<3, str_core::Str3d>({{-1.9, 1.25, -0.30, 0.015, 0.014, 0.013}},
                                        {{-1.55, 0.55, -0.18, 0.02, 0.02, 0.02}},
                                        8000, 0.05);
  check_p_invariants();
  check_resetdata_is_exact();
  check_symmetry_of_rank1_update();
}

}  // namespace str_test
