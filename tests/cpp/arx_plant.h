// Deterministic ARX plant used by the RLS-convergence and behaviour-equivalence
// tests.  It is the §3 model exactly:
//
//   y(k) = -a1 y(k-1) - ... - a_na y(k-na) + b0 u(k-1) + ... + b_{nb-1} u(k-nb)

#ifndef TESTS_CPP_ARX_PLANT_H_
#define TESTS_CPP_ARX_PLANT_H_

#include <array>
#include <cstddef>

namespace str_test {

template <typename Real, int Order>
class ArxPlant {
 public:
  explicit ArxPlant(const std::array<Real, 2 * Order>& theta) : theta_(theta) {}

  void set_theta(const std::array<Real, 2 * Order>& theta) { theta_ = theta; }

  void reset() {
    y_.fill(Real(0));
    u_.fill(Real(0));
  }

  // Apply u(k); returns y(k+1).
  //
  //   y(k+1) = -a1 y(k) - ... - a_na y(k-na+1) + b0 u(k) + ... + b_{nb-1} u(k-nb+1)
  //
  // i.e. exactly the §3 model evaluated one step ahead, so that the controller
  // that acts on y(k) and produces u(k) sees a consistent closed loop.  The input
  // is shifted in *before* the update; shifting it afterwards would add a spurious
  // sample of input delay (and make the identifier fit a shifted model).
  Real step(Real u) {
    for (int j = Order - 1; j >= 1; --j) u_[static_cast<std::size_t>(j)] = u_[static_cast<std::size_t>(j - 1)];
    u_[0] = u;
    Real y_next = Real(0);
    for (int i = 0; i < Order; ++i) y_next -= theta_[static_cast<std::size_t>(i)] * y_[i];
    for (int j = 0; j < Order; ++j)
      y_next += theta_[static_cast<std::size_t>(Order + j)] * u_[j];
    for (int i = Order - 1; i >= 1; --i) y_[static_cast<std::size_t>(i)] = y_[static_cast<std::size_t>(i - 1)];
    y_[0] = y_next;
    return y_next;
  }

  Real y() const { return y_[0]; }

 private:
  std::array<Real, 2 * Order> theta_;
  std::array<Real, Order> y_{};
  std::array<Real, Order> u_{};
};

// Deterministic pseudo-random excitation in [-1, 1] (LCG, no <random>: the same
// sequence must be reproducible everywhere).
class Lcg {
 public:
  explicit Lcg(unsigned seed = 12345u) : state_(seed) {}
  double uniform() {
    state_ = (1103515245u * state_ + 12345u) & 0x7fffffffu;
    return 2.0 * (static_cast<double>(state_) / 2147483648.0) - 1.0;
  }

 private:
  unsigned state_;
};

}  // namespace str_test

#endif  // TESTS_CPP_ARX_PLANT_H_
