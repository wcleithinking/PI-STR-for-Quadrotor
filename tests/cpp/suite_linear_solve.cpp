// Suite: Gaussian elimination with partial pivoting (the shared §5 solver).
//
// Task requirement: "3x3/5x5 线性解正确性（与手算小例对比）".

#include <array>
#include <cmath>

#include "str_core.h"
#include "str_test_harness.h"
#include "suites.h"

namespace str_test {
namespace {

using str_core::solve_linear_system;

// 2x +  y        =  4      (x = 1)
//  x + 3y +  z    = 10      (y = 2)
//      y + 4z    = 18      (z = 4)   -- hand-solved; det = 2*(12-1) - 1*(4-0) = 18
// (b above is for [1,2,3]: 4, 10, 14; use the consistent pair below)
void check_3x3_hand_example() {
  std::array<std::array<double, 3>, 3> a = {{{2, 1, 0}, {1, 3, 1}, {0, 1, 4}}};
  std::array<double, 3> b = {{4, 10, 14}};  // -> x = [1, 2, 3]
  double det = 0;
  const bool ok = solve_linear_system<double, 3>(a, b, det);
  STR_CHECK(ok);
  STR_CHECK_NEAR(det, 18.0, 1e-12);
  STR_CHECK_NEAR(b[0], 1.0, 1e-12);
  STR_CHECK_NEAR(b[1], 2.0, 1e-12);
  STR_CHECK_NEAR(b[2], 3.0, 1e-12);
}

// 5x5 lower-triangular example, hand-solvable:
//   L[i][i] = i+2, L[i][j] = 1 for j < i, x = [1,1,1,1,1]
//   b = row sums = [2, 4, 6, 8, 10],  det = 2*3*4*5*6 = 720
void check_5x5_hand_example() {
  std::array<std::array<double, 5>, 5> a{};
  std::array<double, 5> b{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      a[i][j] = (j < i) ? 1.0 : ((j == i) ? static_cast<double>(i + 2) : 0.0);
    }
    b[i] = static_cast<double>(2 * (i + 1));
  }
  double det = 0;
  const bool ok = solve_linear_system<double, 5>(a, b, det);
  STR_CHECK(ok);
  STR_CHECK_NEAR(det, 720.0, 1e-9);
  for (int i = 0; i < 5; ++i) {
    STR_CHECK_MSG(std::fabs(b[i] - 1.0) < 1e-12, "x[%d] = %.12g", i, b[i]);
  }
}

// A zero leading pivot forces the partial-pivoting branch.
void check_partial_pivoting() {
  std::array<std::array<double, 2>, 2> a = {{{0, 1}, {1, 0}}};
  std::array<double, 2> b = {{2, 3}};  // x = [3, 2]
  double det = 0;
  const bool ok = solve_linear_system<double, 2>(a, b, det);
  STR_CHECK(ok);
  STR_CHECK_NEAR(b[0], 3.0, 1e-15);
  STR_CHECK_NEAR(b[1], 2.0, 1e-15);
  // det of [[0,1],[1,0]] = -1; the row swap flips the sign exactly once.
  STR_CHECK_NEAR(det, -1.0, 1e-15);
}

// A singular matrix must be reported, never silently "solved".
void check_singular_detection() {
  std::array<std::array<double, 3>, 3> a = {{{1, 2, 3}, {2, 4, 6}, {1, 1, 1}}};
  std::array<double, 3> b = {{1, 2, 3}};
  double det = 0;
  const bool solved = solve_linear_system<double, 3>(a, b, det);
  STR_CHECK(!solved);
  STR_CHECK_NEAR(det, 0.0, 0.0);
}

// A NaN in the matrix must be rejected (NaN compares false everywhere, which is
// what makes the pivot test NaN-safe).
void check_nan_rejected() {
  std::array<std::array<double, 2>, 2> a = {{{1, 2}, {NAN, 4}}};
  std::array<double, 2> b = {{1, 2}};
  double det = 0;
  const bool solved = solve_linear_system<double, 2>(a, b, det);
  STR_CHECK(!solved);
}

// Condition number sanity: a badly scaled but non-singular system must still be
// solved to a reasonable accuracy (partial pivoting is what buys this).
void check_badly_scaled() {
  std::array<std::array<double, 3>, 3> a = {{{1e-8, 1, 0}, {1, 1, 1}, {0, 1, 1}}};
  std::array<double, 3> b = {{1.0, 2.0, 2.0}};
  std::array<std::array<double, 3>, 3> a_copy = a;
  std::array<double, 3> x = b;  // the solver takes the RHS and returns the solution
  double det = 0;
  const bool solved = solve_linear_system<double, 3>(a, x, det);
  STR_CHECK(solved);
  // Residual check rather than a reference solution: A * x must reproduce b.
  for (int i = 0; i < 3; ++i) {
    double r = 0;
    for (int j = 0; j < 3; ++j) r += a_copy[i][j] * x[j];
    const double scale = std::fabs(b[i]) + 1.0;
    STR_CHECK_MSG(std::fabs(r - b[i]) <= 1e-9 * scale, "residual[%d] = %.3g", i,
                  r - b[i]);
  }
}

}  // namespace

void suite_linear_solve() {
  check_3x3_hand_example();
  check_5x5_hand_example();
  check_partial_pivoting();
  check_singular_detection();
  check_nan_rejected();
  check_badly_scaled();
}

}  // namespace str_test
