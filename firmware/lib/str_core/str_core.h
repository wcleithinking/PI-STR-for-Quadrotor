// PI-STR self-tuning regulator core.
//
// Pure C++17, NO Arduino / AVR dependency: this header must compile with
//   clang++ -std=c++17 -Wall -Wextra
// on the host (see tests/cpp/ and the root CMakeLists.txt), which is the only
// place this code is actually executed in CI.  It is *also* the implementation
// used by the two firmware sketches (firmware/Quadrotor_Mode_I_Arduino_Due and
// firmware/Quadrotor_Mode_II_Arduino_Due); the .ino files themselves cannot be
// compiled in this environment (no AVR toolchain), so the firmware integration
// is a mechanical, hand-reviewed replacement.
//
// Normative source of truth: docs/CONTROL_SPEC.md
//   §3 discrete-time ARX model (2nd order = 4 params, 3rd order = 6 params)
//   §4 RLS with forgetting factor, P symmetrisation, P diagonal clamp
//   §5 Diophantine A*R + B*S = Am*Ao solved by Gaussian elimination with
//      partial pivoting (the legacy 3x3 / hand-expanded 5x5 adjugates are
//      forbidden: they are where the historical s0/s1 indexing bug lived)
//   §6 numerical-safety policy (projection, singularity guards, non-finite
//      guard, saturation feedback) -- the whole point of stage A.
//
// Vocabulary used below:
//   theta = [a1 .. a_na, b0 .. b_{nb-1}]      (na = nb = Order)
//   phi(k) = [-y(k-1) .. -y(k-na), u(k-1) .. u(k-nb)]
//   R(z)  = 1 + r1 z^-1 + ... + r_{na-1} z^-(na-1)
//   S(z)  = s0 + s1 z^-1 + ... + s_{nb-1} z^-(nb-1)
//   Ao(z) = 1 + ao1 z^-1 + ... + ao_{Order-1} z^-(Order-1)
//   Am(z) = 1 + am1 z^-1 + ... + am_Order z^-Order
//   T(z)  = t0 + t1 z^-1 + ... ,  t0 = Am(1)/B(1),  t_i = t0 * ao_i
//   u(k)  = -sum_i r_i u(k-i) + T y_des(k) - S y(k)      (see eval_law())
//
// The control law is written in the equivalent legacy form
//   v = -sum_i ao_i u(k-i) + T y_des - S y + sum_i (ao_i - r_i) u(k-i)
// so the arithmetic order matches the code it replaces (behaviour equivalence).
//
// NOTE on u(k-i): per §5/§6.4 the feedback must use the *physically realised*
// command (the value recovered from the saturated motor PWMs), never the
// pre-saturation design value.  The caller therefore has two options per tick:
//   * commit_actual(u_realised) after the mixer ran  (firmware path), or
//   * commit_actual(u_returned_by_compute_control)     (no saturation / tests).

#ifndef STR_CORE_STR_CORE_H_
#define STR_CORE_STR_CORE_H_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace str_core {

// ---------------------------------------------------------------------------
// Guard / status flags.  The low bits are per-tick diagnostics, kFlagSafeMode
// is latched (§6.3) until resetdata().
// ---------------------------------------------------------------------------
enum GuardFlag : std::uint32_t {
  kFlagNone = 0u,
  kFlagProjected = 1u << 0,    // §6.1 projection modified theta this tick
  kFlagBsumGuard = 1u << 1,    // §6.2 |B(1)| < bsum_singular -> gains frozen
  kFlagDetGuard = 1u << 2,     // §6.2 |det(A)| < det_singular -> gains frozen
  kFlagVGuard = 1u << 3,       // §6.2 |v(k)| > v_singular   -> gains frozen
  kFlagNonFinite = 1u << 4,    // §6.3 NaN/Inf in theta, P, y, u or command
  kFlagFrozen = 1u << 5,       // gains were reused from the last valid solve
  kFlagSafeMode = 1u << 6,     // §6.3 latched: output 0 differential command
  kFlagSaturated = 1u << 7,    // design command was clamped to +-u_max
  kFlagCovarianceRepaired = 1u << 8  // §4.3 PSD repair of P (degenerate excitation)
};

// Anything in this mask counts as "this tick hit a guard" for §6.3's
// consecutive-failure counter.  Spelled as an enumerator rather than a C++17
// `inline constexpr` variable: the Arduino toolchain used for the firmware builds with
// gnu++11, and the core has to stay usable there (it is the same header).
enum : std::uint32_t {
  kFlagFailureMask =
      std::uint32_t(kFlagBsumGuard | kFlagDetGuard | kFlagVGuard | kFlagNonFinite)
};

// ---------------------------------------------------------------------------
// §6 guard constants.  Defaults are the values mandated by CONTROL_SPEC §6:
//   BSUM_MIN = 1e-3 (float firmware) / 1e-3 (double simulation)
//   |B(1)| < 1e-6, |det| < 1e-9, |v| > 1e6, 50 consecutive failures
// ---------------------------------------------------------------------------
template <typename Real>
struct GuardLimits {
  Real bsum_min;                   // §6.1 lower bound enforced by projection
  Real bsum_singular;              // §6.2 freeze if |B(1)| below this
  Real det_singular;               // §6.2 freeze if |det(A)| below this
  Real v_singular;                 // §6.2 freeze if |v(k)| above this
  int max_consecutive_failures;    // §6.3 safe mode threshold

  static GuardLimits defaults() {
    return GuardLimits{Real(1e-3), Real(1e-6), Real(1e-9), Real(1e6), 50};
  }
};

// ---------------------------------------------------------------------------
// §6.1 parameter projection.
//
// The 2nd-order rule is the one from the Python reference implementation
// (sim/python/pistr_sim/control/str.py, project_theta()), which is the golden-vector
// contract: sequential clamps, NOT a nearest-point projection onto the triangle.
//
//   a1 <- clip(a1, -4, 4);  a2 <- clip(a2, -2, 2);  a2 <- max(a2, -1)
//   a1 <- clip(a1, -1 - a2, 1 + a2)
//
// It is idempotent (a second application is a no-op) and it keeps the *closed*
// stability triangle, so a model sitting on a1 + a2 = -1 (the pure integrator of the
// motor, which the true channel really has) is a fixed point instead of being pushed
// inside and biasing the estimate.
//
// The 3rd-order rule (the Mode I model) is defined by the same Python function: the
// 2nd-order triangle does not describe a cubic denominator, so a1/a2 are clipped to a
// wider box (a2 to +-4, a3 to +-1) and, if a pole leaves the unit disc by more than
// STABILITY_TOL, all coefficients are scaled by a common factor until the polynomial
// is Schur again (scaling all coefficients scales every root by the same factor, so
// this is a radial contraction).  Stability is decided with the Schur-Cohn /
// reflection-coefficient recursion, and the scale factor is found by bisection on
// [0, 1] -- exactly the algorithm of the Python side.
// ---------------------------------------------------------------------------
template <typename Real>
struct ProjectionConstants {
  static constexpr Real kA1Min = Real(-4), kA1Max = Real(4);
  static constexpr Real kA2Min = Real(-2), kA2Max = Real(2);   // order 2 box
  static constexpr Real kA2MinO3 = Real(-4), kA2MaxO3 = Real(4);  // order 3 box
  static constexpr Real kA3Min = Real(-1), kA3Max = Real(1);
  static constexpr Real kBMin = Real(-10), kBMax = Real(10);
  static constexpr Real kStabilityTol = Real(1e-3);
};

// Schur-Cohn / Jury test through the reflection-coefficient recursion.
// `a` holds the coefficients of 1 + a1 z^-1 + ... + aN z^-N, excluding the leading 1.
// Stable (all poles strictly inside the unit disc) iff every reflection coefficient
// has modulus < 1.  Written generically so the tests can probe 2nd and 3rd order.
template <typename Real, std::size_t N>
inline bool is_schur_polynomial(const std::array<Real, N>& a) {
  const int n = static_cast<int>(N);
  Real cur[static_cast<std::size_t>(n) + 1];
  cur[0] = Real(1);
  for (int i = 0; i < n; ++i) cur[i + 1] = a[static_cast<std::size_t>(i)];
  for (int m = n; m >= 1; --m) {
    const Real k = -cur[m];                       // reflection coefficient
    if (!(std::fabs(k) < Real(1))) return false;  // NaN-safe
    const Real den = Real(1) - k * k;
    if (!(den > Real(0))) return false;
    Real next[static_cast<std::size_t>(n) + 1];
    next[0] = Real(1);
    for (int i = 1; i <= m - 1; ++i) next[i] = (cur[i] + k * cur[m - i]) / den;
    for (int i = 0; i <= m - 1; ++i) cur[i] = next[i];
    cur[m] = Real(0);
  }
  return true;
}

// Radial contraction of the denominator until it is Schur (bisection on the common
// scale factor, 30 iterations -- same as the Python reference).  Idempotent: once
// stable, the caller does not call this at all.
template <typename Real, std::size_t N>
inline void shrink_into_unit_disc(std::array<Real, N>& a) {
  Real lo = Real(0), hi = Real(1);
  for (int it = 0; it < 30; ++it) {
    const Real mid = Real(0.5) * (lo + hi);
    std::array<Real, N> scaled{};
    for (std::size_t i = 0; i < N; ++i) scaled[i] = mid * a[i];
    if (is_schur_polynomial(scaled)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  for (std::size_t i = 0; i < N; ++i) a[i] *= lo;
}

// ---------------------------------------------------------------------------
// §6.1 parameter projection, order 2 and order 3.
//
// `bsum_min` is the B(1) floor.  Order 2 uses the fixed §6.1 value (1e-3); order 3
// uses 1% of the initial |B(1)| (see SelfTuningRegulator::init_estimator), because an
// absolute floor sits within a factor of two of the real SI value (~2.1e-3) and would
// bias the estimate.  Idempotent for both orders; exposed so the tests and the
// golden-vector harness can exercise it directly.
// ---------------------------------------------------------------------------
template <typename Real, int Order>
inline void project_theta(std::array<Real, 2 * Order>& theta, Real bsum_min) {
  using C = ProjectionConstants<Real>;
  const int kB = Order;
  const int kA = Order;

  for (int i = 0; i < kA; ++i) {
    const Real lo = (i == 0) ? C::kA1Min : ((kA == 2) ? C::kA2Min : C::kA2MinO3);
    const Real hi = (i == 0) ? C::kA1Max : ((kA == 2) ? C::kA2Max : C::kA2MaxO3);
    const Real lo2 = (i == 2) ? C::kA3Min : lo;
    const Real hi2 = (i == 2) ? C::kA3Max : hi;
    if (theta[static_cast<std::size_t>(i)] < lo2) theta[static_cast<std::size_t>(i)] = lo2;
    if (theta[static_cast<std::size_t>(i)] > hi2) theta[static_cast<std::size_t>(i)] = hi2;
  }

  if (kA == 2) {
    // Python project_theta(), order 2 -- the golden-vector contract.
    if (theta[1] < Real(-1)) theta[1] = Real(-1);
    const Real a1_lo = Real(-1) - theta[1];
    const Real a1_hi = Real(1) + theta[1];
    if (theta[0] < a1_lo) theta[0] = a1_lo;
    if (theta[0] > a1_hi) theta[0] = a1_hi;
  } else {
    // Order 3: Schur test with a 1e-3 tolerance band, then radial contraction.
    std::array<Real, static_cast<std::size_t>(kA)> a{};
    for (int i = 0; i < kA; ++i) a[static_cast<std::size_t>(i)] = theta[static_cast<std::size_t>(i)];
    const Real probe_scale = Real(1) / (Real(1) + C::kStabilityTol);
    std::array<Real, static_cast<std::size_t>(kA)> probe{};
    for (int i = 0; i < kA; ++i)
      probe[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)] * probe_scale;
    if (!is_schur_polynomial(probe)) {
      shrink_into_unit_disc(a);
      for (int i = 0; i < kA; ++i) theta[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)];
    }
  }

  for (int j = 0; j < kB; ++j) {
    Real& b = theta[static_cast<std::size_t>(kA + j)];
    if (b < C::kBMin) b = C::kBMin;
    if (b > C::kBMax) b = C::kBMax;
  }
  Real bsum = Real(0);
  for (int j = 0; j < kB; ++j) bsum += theta[static_cast<std::size_t>(kA + j)];
  if (!(std::fabs(bsum) >= bsum_min)) {
    Real rest = Real(0);
    for (int j = 1; j < kB; ++j) rest += theta[static_cast<std::size_t>(kA + j)];
    theta[static_cast<std::size_t>(kA)] = bsum_min - rest;
  }
}

// ---------------------------------------------------------------------------
// Dense linear solve, Gaussian elimination with partial pivoting, in place.
// Returns false when a pivot is exactly zero (or NaN), i.e. the system is
// structurally singular.  The *numerical* singularity test mandated by §6.2 is
// |det| < det_singular and is applied by the caller using det_out, so the two
// criteria stay separately visible in the code and in the tests.
// ---------------------------------------------------------------------------
template <typename Real, int N>
inline bool solve_linear_system(std::array<std::array<Real, N>, N>& a,
                                std::array<Real, N>& b, Real& det_out) {
  Real det = Real(1);
  for (int col = 0; col < N; ++col) {
    int pivot = col;
    Real best = std::fabs(a[col][col]);
    for (int row = col + 1; row < N; ++row) {
      const Real mag = std::fabs(a[row][col]);
      if (mag > best) {
        best = mag;
        pivot = row;
      }
    }
    // NaN-safe: any comparison with NaN is false, so a NaN column falls through
    // to the failure branch instead of being used as a pivot.
    if (!(best > Real(0))) {
      det_out = Real(0);
      return false;
    }
    if (pivot != col) {
      std::swap(a[pivot], a[col]);
      std::swap(b[pivot], b[col]);
      det = -det;
    }
    const Real p = a[col][col];
    det *= p;
    for (int row = col + 1; row < N; ++row) {
      const Real f = a[row][col] / p;
      if (f == Real(0)) continue;
      for (int j = col; j < N; ++j) a[row][j] -= f * a[col][j];
      b[row] -= f * b[col];
    }
  }
  for (int row = N - 1; row >= 0; --row) {
    Real acc = b[row];
    for (int j = row + 1; j < N; ++j) acc -= a[row][j] * b[j];
    b[row] = acc / a[row][row];
  }
  det_out = det;
  return true;
}

// ---------------------------------------------------------------------------
// Reference model helper (identical arithmetic to the legacy
// STR::init_controller, which is what §5 specifies).
// ---------------------------------------------------------------------------
template <typename Real>
struct SecondOrderReference {
  Real am1;
  Real am2;
  Real ao1;
};

template <typename Real>
inline SecondOrderReference<Real> make_second_order_reference(Real zeta, Real wn,
                                                              Real mu, Real h) {
  const Real omega = wn * std::sqrt(Real(1) - zeta * zeta);
  const Real alpha = std::exp(-zeta * wn * h);
  const Real beta = std::cos(omega * h);
  SecondOrderReference<Real> ref;
  ref.am1 = Real(-2) * alpha * beta;
  ref.am2 = alpha * alpha;
  ref.ao1 = mu;
  return ref;
}

// ---------------------------------------------------------------------------
// The regulator.
// ---------------------------------------------------------------------------
template <typename Real, int Order>
class SelfTuningRegulator {
 public:
  static_assert(Order == 2 || Order == 3,
                "str_core supports the 2nd-order (Mode II) and 3rd-order (Mode I) ARX models");

  static constexpr int kOrder = Order;
  static constexpr int kNumA = Order;                 // a1 .. a_na
  static constexpr int kNumB = Order;                 // b0 .. b_{nb-1}
  static constexpr int kNumParams = kNumA + kNumB;    // 4 (Mode II) / 6 (Mode I)
  static constexpr int kNumR = kNumA - 1;             // r1 .. r_{na-1}
  static constexpr int kNumS = kNumB;                 // s0 .. s_{nb-1}
  static constexpr int kNumT = Order;                 // t0 .. t_{Order-1}
  static constexpr int kSolveSize = kNumR + kNumS;    // 3 (Mode II) / 5 (Mode I)
  static constexpr int kHistory = Order + 1;          // index 0 = "now"

  using Vec = std::array<Real, kNumParams>;
  using Mat = std::array<std::array<Real, kNumParams>, kNumParams>;

  SelfTuningRegulator() { resetdata(); }

  // -- configuration ------------------------------------------------------
  //
  // lambda  : forgetting factor (§4), e.g. 0.99 firmware / 0.998 simulation
  // Pmax    : initial covariance and diagonal clamp (§4.3)
  // theta0  : initial / restart parameter vector (§3, §4.4)
  void init_estimator(Real lambda, Real Pmax, const Vec& theta0) {
    lambda_ = lambda;
    Pmax_ = Pmax;
    theta0_ = theta0;
    // 3rd-order B(1) floor: 1% of the initial |B(1)|, fixed from now on (see
    // apply_projection() for why an absolute bound is wrong there).  Falls back to the
    // absolute §6.1 constant if the initial guess has B(1) = 0.
    if (kNumA == 3) {
      Real bsum0 = Real(0);
      for (int j = 0; j < kNumB; ++j) bsum0 += theta0[kNumA + j];
      bsum_min_o3_ = Real(0.01) * std::fabs(bsum0);
      if (!(bsum_min_o3_ > Real(0))) bsum_min_o3_ = limits_.bsum_min;
    } else {
      bsum_min_o3_ = limits_.bsum_min;
    }
    resetdata();
  }

  // Am(z), Ao(z) coefficient arrays, excluding their leading 1s:
  //   am[0..Order-1]  = am1 .. am_Order
  //   ao[0..Order-2]  = ao1 .. ao_{Order-1}
  void set_reference(const std::array<Real, kNumA>& am,
                     const std::array<Real, kNumR>& ao) {
    am_ = am;
    ao_ = ao;
  }

  // Mode II convenience wrapper: zeta/wn/mu/h as in the legacy
  // init_controller(0.7, 5, 0, STR_period/1000).
  // Declared as a member template so that explicitly instantiating the class for
  // Order == 3 does not instantiate (and fail) this body; calling it on a 3rd-order
  // regulator is a compile error, which is the intent.
  template <int O = Order>
  void set_second_order_reference(Real zeta, Real wn, Real mu, Real h) {
    static_assert(O == 2,
                  "set_second_order_reference() only describes the 2nd-order model");
    const SecondOrderReference<Real> ref =
        make_second_order_reference(zeta, wn, mu, h);
    am_ = {ref.am1, ref.am2};
    ao_ = {ref.ao1};
  }

  void set_guard_limits(const GuardLimits<Real>& limits) { limits_ = limits; }
  const GuardLimits<Real>& guard_limits() const { return limits_; }

  // Diagnostic override (tests, ground-station retuning).  Does not touch P and
  // does not update the "last valid" snapshot, so a subsequent non-finite check
  // still restores the last clean state.
  void set_theta(const Vec& theta) { theta_ = theta; }

  // §4.5 / §6.3: restore theta0 and P = Pmax*I, clear all history and flags.
  // Also the only way out of latched safe mode.
  void resetdata() {
    theta_ = theta0_;
    for (int i = 0; i < kNumParams; ++i) {
      for (int j = 0; j < kNumParams; ++j) {
        P_[i][j] = (i == j) ? Pmax_ : Real(0);
      }
    }
    last_valid_theta_ = theta_;
    last_valid_P_ = P_;
    for (int i = 0; i < kHistory; ++i) {
      y_meas_[i] = Real(0);
      y_des_[i] = Real(0);
      u_act_[i] = Real(0);
    }
    for (int i = 0; i < kNumR; ++i) gains_r_[i] = Real(0);
    for (int i = 0; i < kNumS; ++i) gains_s_[i] = Real(0);
    for (int i = 0; i < kNumT; ++i) gains_t_[i] = Real(0);
    last_design_ = Real(0);
    last_command_ = Real(0);
    last_good_design_ = Real(0);
    flags_ = kFlagNone;
    failures_ = 0;
    tick_valid_ = false;
    last_tick_failed_ = false;
    safe_mode_ = false;
  }

  // -- per-tick API -------------------------------------------------------
  //
  // Call exactly once per controller tick, then update_estimator() and/or
  // compute_control() in the *same order the legacy code used* (Mode II:
  // estimator first, then control; Mode I: control first, then estimator --
  // see the .ino files), and finally commit_actual() with the realised command.
  //
  // Returns false when the sample was rejected (§6.3): history is then left
  // untouched so the control law keeps running on the last good data.
  bool begin_tick(Real y_des, Real y) {
    flags_ = safe_mode_ ? kFlagSafeMode : kFlagNone;
    if (!last_tick_failed_) failures_ = 0;
    tick_valid_ = false;
    if (safe_mode_) return false;
    if (!is_finite(y) || !is_finite(y_des)) {
      register_failure(kFlagNonFinite);
      note_tick_end();  // keep the §6.3 consecutive-failure counter accumulating
      return false;
    }
    for (int i = kHistory - 1; i >= 1; --i) {
      y_meas_[i] = y_meas_[i - 1];
      y_des_[i] = y_des_[i - 1];
      u_act_[i] = u_act_[i - 1];
    }
    y_meas_[0] = y;
    y_des_[0] = y_des;
    u_act_[0] = Real(0);  // filled by commit_actual() once the mixer ran
    tick_valid_ = true;
    return true;
  }

  // §4 RLS + §6.1 projection + §6.3 non-finite guard.
  bool update_estimator() {
    if (safe_mode_) return false;
    if (!tick_valid_) {
      note_tick_end();
      return false;
    }

    Real phi[kNumParams];
    for (int i = 0; i < kNumA; ++i) phi[i] = -y_meas_[i + 1];
    for (int j = 0; j < kNumB; ++j) phi[kNumA + j] = u_act_[j + 1];

    // e = y(k) - theta' * phi(k)
    Real e = y_meas_[0];
    for (int i = 0; i < kNumParams; ++i) e -= theta_[i] * phi[i];

    // alpha = lambda + phi' P phi
    Real alpha = lambda_;
    for (int i = 0; i < kNumParams; ++i) {
      Real row = Real(0);
      for (int j = 0; j < kNumParams; ++j) row += P_[i][j] * phi[j];
      alpha += phi[i] * row;
    }

    if (!is_finite(alpha) || !(alpha > Real(0))) {
      restore_last_valid();
      register_failure(kFlagNonFinite);
      note_tick_end();
      return false;
    }

    // K = P phi / alpha ; theta += K e
    Real kvec[kNumParams];
    for (int i = 0; i < kNumParams; ++i) {
      Real acc = Real(0);
      for (int j = 0; j < kNumParams; ++j) acc += P_[i][j] * phi[j];
      kvec[i] = acc / alpha;
    }

    Vec theta_new = theta_;
    for (int i = 0; i < kNumParams; ++i) theta_new[i] += kvec[i] * e;

    // P = (P - K (phi' P)) / lambda   -- rank-1 form, never I - K phi' (§4.1)
    Real phip[kNumParams];
    for (int j = 0; j < kNumParams; ++j) {
      Real acc = Real(0);
      for (int i = 0; i < kNumParams; ++i) acc += phi[i] * P_[i][j];
      phip[j] = acc;
    }
    Mat P_new;
    for (int i = 0; i < kNumParams; ++i) {
      for (int j = 0; j < kNumParams; ++j) {
        P_new[i][j] = (P_[i][j] - kvec[i] * phip[j]) / lambda_;
      }
    }
    // §4.2 symmetrise.
    for (int i = 0; i < kNumParams; ++i) {
      for (int j = i + 1; j < kNumParams; ++j) {
        const Real s = Real(0.5) * (P_new[i][j] + P_new[j][i]);
        P_new[i][j] = s;
        P_new[j][i] = s;
      }
    }
    // §4.3 diagonal clamp to [0, Pmax].
    //
    // The clamp itself is in place (off-diagonals kept), which is what the Python
    // reference does and what keeps the golden vectors comparable.  The rebuild from
    // the clamped diagonal is a *separate*, much rarer decision: only when the clamped
    // matrix is no longer positive semi-definite.  With a poorly excited interval (a
    // hovering aircraft with centred sticks gives an almost constant regressor) a
    // diagonal can be driven to zero while the off-diagonals keep growing; phi' P phi
    // then goes negative and alpha <= 0 (reproduced in tests/cpp/suite_rls.cpp).
    // Rebuilding only in that case is what keeps normal operation identical to Python.
    Real diag[kNumParams];
    for (int i = 0; i < kNumParams; ++i) {
      Real d = P_new[i][i];
      if (!(d > Real(0))) d = Real(0);      // also catches a NaN diagonal
      if (d > Pmax_) d = Pmax_;
      diag[i] = d;
      P_new[i][i] = d;  // in place: off-diagonals untouched
    }
    if (!is_positive_semidefinite(P_new)) {
      flags_ |= kFlagCovarianceRepaired;
      for (int i = 0; i < kNumParams; ++i) {
        for (int j = 0; j < kNumParams; ++j) {
          P_new[i][j] = (i == j) ? diag[i] : Real(0);
        }
      }
    }
    // §6.3 non-finite guard on the candidate state before committing it.
    if (!all_finite(theta_new) || !all_finite_mat(P_new)) {
      restore_last_valid();
      register_failure(kFlagNonFinite);
      note_tick_end();
      return false;
    }

    theta_ = theta_new;
    P_ = P_new;
    apply_projection();
    last_valid_theta_ = theta_;
    last_valid_P_ = P_;
    note_tick_end();
    return true;
  }

  // §5 Diophantine + control law, §6.2 singularity guards, §6.4 saturation.
  // Returns the command clamped to +-u_max (0 while in latched safe mode).
  Real compute_control(Real u_max) {
    if (safe_mode_) {
      last_design_ = Real(0);
      last_command_ = Real(0);
      flags_ |= kFlagSafeMode;
      note_tick_end();
      return Real(0);
    }

    const bool solved = recompute_gains();
    if (!solved) flags_ |= kFlagFrozen;

    Real v = evaluate_control_law();
    if (!is_finite(v)) {
      register_failure(kFlagNonFinite);
      flags_ |= kFlagFrozen;
      v = last_good_design_;
    } else if (std::fabs(v) > limits_.v_singular) {
      // §6.2: treat as singular, keep the last valid gains and do not emit it.
      register_failure(kFlagVGuard);
      flags_ |= kFlagFrozen;
      v = last_good_design_;
    }

    last_design_ = v;
    Real u = v;
    if (u > u_max) {
      u = u_max;
      flags_ |= kFlagSaturated;
    } else if (u < -u_max) {
      u = -u_max;
      flags_ |= kFlagSaturated;
    }
    if (!is_finite(u)) {
      register_failure(kFlagNonFinite);
      u = Real(0);
    }
    last_command_ = u;
    if ((flags_ & kFlagFailureMask) == 0) last_good_design_ = v;
    note_tick_end();
    return u;
  }

  // Feed back the physically realised differential command (§5/§6.4).  Call
  // once per tick, after the mixer / PWM saturation produced the real value.
  void commit_actual(Real u_actual) {
    if (!is_finite(u_actual)) {
      register_failure(kFlagNonFinite);
      note_tick_end();
      return;  // keep the previous (finite) history entry
    }
    u_act_[0] = u_actual;
  }

  // -- introspection (stable: Python golden vectors rely on these) --------
  const Vec& theta_hat() const { return theta_; }
  const Mat& covariance() const { return P_; }
  const std::array<Real, kNumR>& r() const { return gains_r_; }
  const std::array<Real, kNumS>& s() const { return gains_s_; }
  const std::array<Real, kNumT>& t() const { return gains_t_; }
  Real last_design_command() const { return last_design_; }
  Real last_command() const { return last_command_; }
  const std::array<Real, kHistory>& y_history() const { return y_meas_; }
  const std::array<Real, kHistory>& u_history() const { return u_act_; }
  std::uint32_t flags() const { return flags_; }
  bool frozen() const { return (flags_ & kFlagFrozen) != 0; }
  bool safe_mode() const { return safe_mode_; }
  int consecutive_failures() const { return failures_; }
  bool sample_accepted() const { return tick_valid_; }

 private:
  Real effective_bsum_min() const {
    return (kNumA == 3) ? bsum_min_o3_ : limits_.bsum_min;
  }

  static bool is_finite(Real v) {
    // <cmath> overloads exist for float and double; the explicit comparison
    // keeps the intent obvious and avoids std::isfinite(int) surprises.
    return v == v && v != std::numeric_limits<Real>::infinity() &&
           v != -std::numeric_limits<Real>::infinity();
  }
  static bool all_finite(const Vec& v) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (!is_finite(v[i])) return false;
    }
    return true;
  }
  static bool all_finite_mat(const Mat& m) {
    for (std::size_t i = 0; i < m.size(); ++i) {
      for (std::size_t j = 0; j < m[i].size(); ++j) {
        if (!is_finite(m[i][j])) return false;
      }
    }
    return true;
  }

  // Cholesky probe (lower triangle only).  Returns false for a non-PSD or singular
  // matrix, and for NaN, which is exactly what the §4.3 repair decision needs.
  static bool is_positive_semidefinite(const Mat& m) {
    Real work[kNumParams][kNumParams];
    for (int i = 0; i < kNumParams; ++i) {
      for (int j = 0; j < kNumParams; ++j) work[i][j] = m[i][j];
    }
    for (int i = 0; i < kNumParams; ++i) {
      Real d = work[i][i];
      for (int k = 0; k < i; ++k) d -= work[i][k] * work[i][k];
      if (!(d > Real(0))) return false;  // not positive definite (incl. NaN)
      const Real sq = std::sqrt(d);
      work[i][i] = sq;
      for (int j = i + 1; j < kNumParams; ++j) {
        Real acc = work[j][i];
        for (int k = 0; k < i; ++k) acc -= work[j][k] * work[i][k];
        work[j][i] = acc / sq;
      }
    }
    return true;
  }

  void register_failure(std::uint32_t flag) {
    flags_ |= flag;
    ++failures_;
    if (failures_ >= limits_.max_consecutive_failures) {
      safe_mode_ = true;
      flags_ |= kFlagSafeMode;
    }
  }

  // A tick counts as failed if any guard tripped during it; flags_ is cleared
  // only by the *next* begin_tick(), so whichever of update_estimator() /
  // compute_control() runs last sees the union of both.
  void note_tick_end() {
    last_tick_failed_ = (flags_ & kFlagFailureMask) != 0;
  }

  void restore_last_valid() {
    theta_ = last_valid_theta_;
    P_ = last_valid_P_;
  }

  // §6.1 projection.  Must run before the gains are recomputed, which is exactly
  // what update_estimator() -> compute_control() ordering guarantees.
  //
  // The B(1) lower bound is `limits_.bsum_min` for the 2nd-order model (fixed 1e-3,
  // part of the golden-vector contract).  For the 3rd-order model an absolute 1e-3
  // would be far too aggressive: in SI units the identified Mode I channel has
  // B(1) ~ 2.1e-3, so a fixed 1e-3 floor sits within a factor of two of the real
  // value and would bias the estimate whenever the forgetting factor dips it a little.
  // The 3rd-order bound is therefore relative -- 1% of |B(1)| at initialisation,
  // computed once in init_estimator() and constant afterwards.
  void apply_projection() {
    const Vec before = theta_;
    project_theta<Real, Order>(theta_, effective_bsum_min());
    for (int i = 0; i < kNumParams; ++i) {
      if (!is_finite(theta_[i])) {  // robust to caller-supplied garbage
        theta_ = before;
        break;
      }
    }
    for (int i = 0; i < kNumParams; ++i) {
      if (theta_[i] != before[i]) {
        flags_ |= kFlagProjected;
        break;
      }
    }
  }

  // Build and solve the (kSolveSize x kSolveSize) Diophantine system
  //   A*R + B*S = Am*Ao ,  unknowns x = [r1..r_{na-1}, s0..s_{nb-1}]
  // which is exactly the general construction of §5 (it reproduces the
  // explicit 3x3 matrix of §5 and the legacy hand-expanded 5x5 system).
  // On success the new gains replace the previous ones; on failure nothing is
  // written, so the caller keeps controlling with the last valid gains.
  bool recompute_gains() {
    const Real* a = &theta_[0];
    const Real* b = &theta_[kNumA];

    Real bsum = Real(0);
    for (int j = 0; j < kNumB; ++j) bsum += b[j];
    if (!is_finite(bsum)) {
      register_failure(kFlagNonFinite);
      return false;
    }
    if (std::fabs(bsum) < limits_.bsum_singular) {
      register_failure(kFlagBsumGuard);  // §6.2
      return false;
    }

    // C = Am * Ao  (degree kSolveSize = kNumA + kNumT - 1).  Plain convolution of
    // the full coefficient vectors: Am = [1, am1..am_na], Ao = [1, ao1..ao_{n-1}].
    // (Multiplying factor by factor as if each am_i were its own (1 + am_i z^-i)
    // introduces spurious cross terms -- that mistake is what the golden vectors
    // would catch, and it is why the tests compare against the explicit §5 matrix.)
    Real c[kSolveSize + 1];
    for (int i = 0; i <= kSolveSize; ++i) c[i] = Real(0);
    for (int i = 0; i <= kNumA; ++i) {
      const Real am_i = (i == 0) ? Real(1) : am_[i - 1];
      for (int j = 0; j < kNumT; ++j) {
        const Real ao_j = (j == 0) ? Real(1) : ao_[j - 1];
        c[i + j] += am_i * ao_j;
      }
    }

    std::array<std::array<Real, kSolveSize>, kSolveSize> m{};
    std::array<Real, kSolveSize> rhs{};
    for (int k = 1; k <= kSolveSize; ++k) {
      for (int i = 1; i <= kNumR; ++i) {
        // coefficient of z^-k in A(z) * z^-i R: a_{k-i}
        const int idx = k - i;
        Real a_coef = Real(0);
        if (idx == 0) {
          a_coef = Real(1);
        } else if (idx >= 1 && idx <= kNumA) {
          a_coef = a[idx - 1];
        }
        m[k - 1][i - 1] = a_coef;
      }
      for (int j = 0; j < kNumS; ++j) {
        // coefficient of z^-k in B(z) * z^-j S: b_{k-1-j}
        const int idx = k - 1 - j;
        Real b_coef = Real(0);
        if (idx >= 0 && idx < kNumB) b_coef = b[idx];
        m[k - 1][kNumR + j] = b_coef;
      }
      Real a_k = Real(0);
      if (k <= kNumA) a_k = a[k - 1];
      rhs[k - 1] = c[k] - a_k;
    }

    Real det = Real(0);
    const bool solved = solve_linear_system<Real, kSolveSize>(m, rhs, det);
    if (!solved || !is_finite(det) || std::fabs(det) < limits_.det_singular) {
      register_failure(kFlagDetGuard);  // §6.2
      return false;
    }
    for (int i = 0; i < kSolveSize; ++i) {
      if (!is_finite(rhs[i])) {
        register_failure(kFlagNonFinite);
        return false;
      }
    }

    // Reference gain: t0 = Am(1)/B(1), t_i = t0 * ao_i  (§5; the legacy
    // Mode I firmware also uses Am(1)/B(1), i.e. it does not fold in Ao(1) --
    // preserved on purpose, see the report / SAFETY_NOTES.md).
    Real am_sum = Real(1);
    for (int i = 0; i < kNumA; ++i) am_sum += am_[i];
    const Real t0 = am_sum / bsum;
    if (!is_finite(t0)) {
      register_failure(kFlagNonFinite);
      return false;
    }
    Real new_t[kNumT];
    new_t[0] = t0;
    for (int i = 1; i < kNumT; ++i) new_t[i] = t0 * ao_[i - 1];

    for (int i = 0; i < kNumR; ++i) gains_r_[i] = rhs[i];
    for (int j = 0; j < kNumS; ++j) gains_s_[j] = rhs[kNumR + j];
    for (int i = 0; i < kNumT; ++i) gains_t_[i] = new_t[i];
    return true;
  }

  Real evaluate_control_law() const {
    Real v = Real(0);
    for (int i = 1; i <= kNumR; ++i) v -= ao_[i - 1] * u_act_[i];
    for (int i = 0; i < kNumT; ++i) v += gains_t_[i] * y_des_[i];
    for (int j = 0; j < kNumS; ++j) v -= gains_s_[j] * y_meas_[j];
    for (int i = 1; i <= kNumR; ++i) v += (ao_[i - 1] - gains_r_[i - 1]) * u_act_[i];
    return v;
  }

  // configuration
  Real lambda_ = Real(1);
  Real Pmax_ = Real(100);
  Vec theta0_{};
  std::array<Real, kNumA> am_{};  // am1 .. am_Order
  std::array<Real, kNumR> ao_{};  // ao1 .. ao_{Order-1}
  GuardLimits<Real> limits_ = GuardLimits<Real>::defaults();
  Real bsum_min_o3_ = Real(1e-3);  // order-3 relative bound, see init_estimator()

  // estimator state
  Vec theta_{};
  Mat P_{};
  Vec last_valid_theta_{};
  Mat last_valid_P_{};

  // signal history, index 0 = "now"
  std::array<Real, kHistory> y_meas_{};
  std::array<Real, kHistory> y_des_{};
  std::array<Real, kHistory> u_act_{};

  // controller state (last valid solve)
  std::array<Real, kNumR> gains_r_{};
  std::array<Real, kNumS> gains_s_{};
  std::array<Real, kNumT> gains_t_{};

  Real last_design_ = Real(0);
  Real last_command_ = Real(0);
  Real last_good_design_ = Real(0);

  std::uint32_t flags_ = kFlagNone;
  int failures_ = 0;
  bool tick_valid_ = false;
  bool last_tick_failed_ = false;
  bool safe_mode_ = false;
};

using Str2f = SelfTuningRegulator<float, 2>;   // Mode II firmware
using Str2d = SelfTuningRegulator<double, 2>;  // Mode II simulation oracle
using Str3f = SelfTuningRegulator<float, 3>;   // Mode I firmware
using Str3d = SelfTuningRegulator<double, 3>;  // Mode I simulation oracle

}  // namespace str_core

#endif  // STR_CORE_STR_CORE_H_
