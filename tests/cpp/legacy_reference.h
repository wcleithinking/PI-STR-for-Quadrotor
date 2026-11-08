// Faithful C++ copies of the *legacy* firmware math, used only by the tests to
// prove that `str_core` is behaviour-preserving.
//
//  * Legacy2 mirrors firmware/libraries/MyLibraries_Wenchao/STR.cpp (Mode II,
//    2nd order, 3x3 adjugate, explicit (I - K phi') P update).
//  * Legacy3 mirrors the 5x5 hand-expanded adjugate + estimator that used to live
//    in firmware/Quadrotor_Mode_I_Arduino_Due/Controller.ino (Mode I, 3rd order).
//  * legacy_adjugate_3 / legacy_adjugate_5 expose the two adjugate solutions so
//    the Diophantine tests can compare them against the Gaussian elimination.
//
// The two "historical bug" switches reproduce the *unfixed* indexing
//   s0 = xx[2]; s1 = xx[3];
// from STR.cpp, so a test can show how large the resulting command error is.
//
// Everything is templated on the scalar type: the equivalence tests run in double
// so that the comparison is limited by algebra, not by float rounding.

#ifndef TESTS_CPP_LEGACY_REFERENCE_H_
#define TESTS_CPP_LEGACY_REFERENCE_H_

#include <array>
#include <cmath>
#include <cstddef>

namespace legacy {

// ---------------------------------------------------------------------------
// 3x3 adjugate solution, copied verbatim from STR.cpp run_controller().
// ---------------------------------------------------------------------------
template <typename Real>
inline void legacy_adjugate_3(Real hata1, Real hata2, Real hatb0, Real hatb1,
                              Real am1, Real am2, Real ao1, Real xx[3]) {
  Real detAA_inv = Real(1) / (hatb1 * hatb1 - hata1 * hatb0 * hatb1 +
                              hata2 * hatb0 * hatb0);
  Real AA_adj[3][3];
  Real bb[3];
  bb[0] = am1 + ao1 - hata1;
  bb[1] = am2 + ao1 * am1 - hata2;
  bb[2] = ao1 * am2;
  AA_adj[0][0] = hatb1 * hatb1;
  AA_adj[0][1] = -hatb0 * hatb1;
  AA_adj[0][2] = hatb0 * hatb0;
  AA_adj[1][0] = hata2 * hatb0 - hata1 * hatb1;
  AA_adj[1][1] = hatb1;
  AA_adj[1][2] = -hatb0;
  AA_adj[2][0] = -hata2 * hatb1;
  AA_adj[2][1] = hata2 * hatb0;
  AA_adj[2][2] = hatb1 - hata1 * hatb0;
  for (int i = 0; i < 3; ++i) {
    xx[i] = 0;
    for (int j = 0; j < 3; ++j) xx[i] += AA_adj[i][j] * bb[j];
    xx[i] *= detAA_inv;
  }
}

// ---------------------------------------------------------------------------
// 5x5 adjugate solution, copied verbatim from the Mode I Controller.ino
// (lines ~252-293 of the pre-refactor file).  `bb` is the right-hand side;
// the solution is returned in `xx` with the legacy ordering
//   x = [r1, r2, s0, s1, s2].
// ---------------------------------------------------------------------------
template <typename Real>
inline void legacy_adjugate_5(Real hata1, Real hata2, Real hata3, Real hatb0,
                              Real hatb1, Real hatb2, const Real bb_in[5],
                              Real xx[5]) {
  const Real hata12 = hata1 * hata1, hata13 = hata12 * hata1;
  const Real hata22 = hata2 * hata2, hata23 = hata22 * hata2;
  const Real hata32 = hata3 * hata3;
  const Real hatb02 = hatb0 * hatb0, hatb03 = hatb02 * hatb0;
  const Real hatb12 = hatb1 * hatb1, hatb13 = hatb12 * hatb1;
  const Real hatb22 = hatb2 * hatb2, hatb23 = hatb22 * hatb2;
  (void)hata13;
  (void)hata23;
  Real detAA_inv = Real(1) /
                   (hata12 * hatb0 * hatb22 - 2 * hata1 * hatb02 * hata3 * hatb2 -
                    hata1 * hatb0 * hata2 * hatb1 * hatb2 +
                    hata1 * hatb0 * hatb12 * hata3 - hata1 * hatb1 * hatb22 +
                    hatb03 * hata32 + hatb02 * hata22 * hatb2 -
                    hatb02 * hata2 * hatb1 * hata3 - 2 * hatb0 * hata2 * hatb22 +
                    3 * hatb0 * hatb1 * hata3 * hatb2 + hata2 * hatb12 * hatb2 -
                    hatb13 * hata3 + hatb23);
  Real aa[5][5];
  aa[0][0] = -hata3 * hatb13 + hata2 * hatb12 * hatb2 - hata1 * hatb1 * hatb22 +
             2 * hatb0 * hata3 * hatb1 * hatb2 + hatb23 - hatb0 * hata2 * hatb22;
  aa[0][1] = -hata3 * hatb02 * hatb2 + hata3 * hatb0 * hatb12 -
             hata2 * hatb0 * hatb1 * hatb2 + hata1 * hatb0 * hatb22;
  aa[0][2] = hata2 * hatb02 * hatb2 - hatb1 * hata3 * hatb02 - hatb0 * hatb22;
  aa[0][3] = hata3 * hatb03 - hata1 * hatb2 * hatb02 + hatb1 * hatb2 * hatb0;
  aa[0][4] = -hata2 * hatb03 + hata1 * hatb02 * hatb1 + hatb2 * hatb02 -
             hatb0 * hatb12;
  aa[1][0] = -hata3 * hatb12 * hatb2 + hata2 * hatb1 * hatb22 - hata1 * hatb23 +
             hatb0 * hata3 * hatb22;
  aa[1][1] = hatb23 - hatb0 * hata2 * hatb22 + hatb0 * hatb1 * hata3 * hatb2;
  aa[1][2] = -hata3 * hatb02 * hatb2 + hata1 * hatb0 * hatb22 - hatb1 * hatb22;
  aa[1][3] = hata2 * hatb02 * hatb2 - hata1 * hatb0 * hatb1 * hatb2 -
             hatb0 * hatb22 + hatb12 * hatb2;
  aa[1][4] = hata3 * hatb03 - hata2 * hatb02 * hatb1 - hata1 * hatb2 * hatb02 +
             hata1 * hatb0 * hatb12 + 2 * hatb2 * hatb0 * hatb1 - hatb13;
  aa[2][0] = hata12 * hatb22 - 2 * hata1 * hatb0 * hata3 * hatb2 -
             hata1 * hata2 * hatb1 * hatb2 + hata1 * hatb12 * hata3 +
             hatb02 * hata32 + hatb0 * hata22 * hatb2 -
             hatb0 * hata2 * hatb1 * hata3 - hata2 * hatb22 + hatb1 * hata3 * hatb2;
  aa[2][1] = -hata3 * hatb12 + hata2 * hatb1 * hatb2 - hata1 * hatb22 +
             hatb0 * hata3 * hatb2;
  aa[2][2] = hatb22 - hatb0 * hata2 * hatb2 + hatb0 * hatb1 * hata3;
  aa[2][3] = -hata3 * hatb02 + hata1 * hatb2 * hatb0 - hatb1 * hatb2;
  aa[2][4] = hata2 * hatb02 - hata1 * hatb0 * hatb1 - hatb2 * hatb0 + hatb12;
  aa[3][0] = -hata22 * hatb1 * hatb2 + hata2 * hatb12 * hata3 +
             hata1 * hata2 * hatb22 - hatb0 * hatb1 * hata32 - hata3 * hatb22;
  aa[3][1] = hatb02 * hata32 + hatb0 * hata22 * hatb2 -
             hatb1 * hatb0 * hata2 * hata3 - hata1 * hatb0 * hata3 * hatb2 -
             hata2 * hatb22 + hatb1 * hata3 * hatb2;
  aa[3][2] = hatb0 * hata3 * hatb2 - hatb12 * hata3 + hata2 * hatb1 * hatb2 -
             hata1 * hatb0 * hata2 * hatb2 + hata1 * hatb0 * hatb1 * hata3;
  aa[3][3] = hata12 * hatb0 * hatb2 - hata3 * hata1 * hatb02 -
             hatb1 * hata1 * hatb2 - hata2 * hatb0 * hatb2 +
             hatb1 * hata3 * hatb0 + hatb22;
  aa[3][4] = -hata12 * hatb0 * hatb1 + hata2 * hata1 * hatb02 + hata1 * hatb12 -
             hata3 * hatb02 - hatb2 * hatb1;
  aa[4][0] = hatb12 * hata32 - hata2 * hatb1 * hata3 * hatb2 -
             hatb0 * hata32 * hatb2 + hata1 * hata3 * hatb22;
  aa[4][1] = -hatb0 * hatb1 * hata32 - hata3 * hatb22 +
             hatb0 * hata2 * hata3 * hatb2;
  aa[4][2] = hatb02 * hata32 - hata1 * hatb2 * hatb0 * hata3 +
             hatb1 * hatb2 * hata3;
  aa[4][3] = -hata2 * hata3 * hatb02 + hata1 * hata3 * hatb0 * hatb1 +
             hata3 * hatb2 * hatb0 - hata3 * hatb12;
  aa[4][4] = hata12 * hatb0 * hatb2 - hata3 * hata1 * hatb02 -
             hata1 * hatb0 * hata2 * hatb1 - hata1 * hatb1 * hatb2 +
             hatb02 * hata22 - 2 * hatb0 * hata2 * hatb2 + hata3 * hatb0 * hatb1 +
             hata2 * hatb12 + hatb22;
  for (int i = 0; i < 5; ++i) {
    xx[i] = 0;
    for (int j = 0; j < 5; ++j) xx[i] += aa[i][j] * bb_in[j];
    xx[i] *= detAA_inv;
  }
}

// ---------------------------------------------------------------------------
// Mode II legacy regulator (STR.cpp), 2nd order.
// ---------------------------------------------------------------------------
template <typename Real>
class Legacy2 {
 public:
  void init_estimator(Real forgetting_factor, Real maxP, Real a1_guess,
                      Real a2_guess, Real b0_guess, Real b1_guess) {
    lambda = forgetting_factor;
    Pmax = maxP;
    for (int i = 0; i < 4; ++i) P[i][i] = Pmax;
    a1_prior = a1_guess;
    a2_prior = a2_guess;
    b0_prior = b0_guess;
    b1_prior = b1_guess;
  }

  void init_controller(Real xi, Real omega_0, Real mu, Real sample_period) {
    Real omega = omega_0 * std::sqrt(Real(1) - xi * xi);
    Real alpha = std::exp(-xi * omega_0 * sample_period);
    Real beta = std::cos(omega * sample_period);
    Real gamma = std::sin(omega * sample_period);
    // The legacy code also computed bm0/bm1 here (the reference numerator); they
    // were never used by the control law, so they are not reproduced.  `gamma` and
    // `omega` stay because they document where the numbers come from.
    (void)gamma;
    am1 = Real(-2) * alpha * beta;
    am2 = alpha * alpha;
    ao1 = mu;
  }

  void update_output(Real output_desired, Real output_measure) {
    y_desired[2] = y_desired[1];
    y_desired[1] = y_desired[0];
    y_desired[0] = output_desired;
    y_measure[2] = y_measure[1];
    y_measure[1] = y_measure[0];
    y_measure[0] = output_measure;
    v_compute[2] = v_compute[1];
    v_compute[1] = v_compute[0];
    u_compute[2] = u_compute[1];
    u_compute[1] = u_compute[0];
  }

  void run_estimator() {
    Real varphi[4] = {-y_measure[1], -y_measure[2], u_compute[1], u_compute[2]};
    Real e = y_measure[0];
    Real alpha = lambda;
    Real K[4];
    Real temp[4][4];
    Real P_o[4][4];
    for (int i = 0; i < 4; ++i) e -= hattheta[i] * varphi[i];
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) alpha += varphi[i] * P[i][j] * varphi[j];
    for (int i = 0; i < 4; ++i) {
      K[i] = 0;
      for (int j = 0; j < 4; ++j) K[i] += P[i][j] * varphi[j];
      K[i] = K[i] / alpha;
      hattheta[i] += K[i] * e;
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        temp[i][j] = (i == j) ? (Real(1) - K[i] * varphi[j]) : (-K[i] * varphi[j]);
        P_o[i][j] = P[i][j];
      }
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        P[i][j] = 0;
        for (int k = 0; k < 4; ++k) P[i][j] += temp[i][k] * P_o[k][j];
        P[i][j] = P[i][j] / lambda;
      }
    }
  }

  Real run_controller(Real input_max) {
    Real r1, s0, s1, t0, t1;
    const Real hata1 = hattheta[0];
    const Real hata2 = hattheta[1];
    const Real hatb0 = hattheta[2];
    const Real hatb1 = hattheta[3];
    Real xx[3];
    legacy_adjugate_3(hata1, hata2, hatb0, hatb1, am1, am2, ao1, xx);
    Real beta = (Real(1) + am1 + am2) / (hatb0 + hatb1);
    // A-stage fix: the legacy STR.cpp read s0 = xx[2]; s1 = xx[3] (out of bounds
    // for a 3-element vector).  Per CONTROL_SPEC §5 the unknown order is
    // x = [r1, s0, s1], so the correct mapping is s0 = xx[1], s1 = xx[2].
    r1 = xx[0];
    s0 = xx[1];
    s1 = xx[2];
    t0 = beta;
    t1 = beta * ao1;
    v_compute[0] = -ao1 * v_compute[1] + t0 * y_desired[0] + t1 * y_desired[1] -
                   s0 * y_measure[0] - s1 * y_measure[1] +
                   (ao1 - r1) * u_compute[1];
    u_compute[0] = v_compute[0];
    if (u_compute[0] > input_max) u_compute[0] = input_max;
    if (u_compute[0] < -input_max) u_compute[0] = -input_max;
    return u_compute[0];
  }

  void get_u_actual(Real u_actual) { u_compute[0] = u_actual; }

  void resetdata() {
    for (int i = 0; i < 4; ++i) {
      if (i < 3) {
        y_desired[i] = 0;
        y_measure[i] = 0;
        v_compute[i] = 0;
        u_compute[i] = 0;
      }
      for (int j = 0; j < 4; ++j) P[i][j] = (i == j) ? Pmax : Real(0);
    }
    hattheta[0] = a1_prior;
    hattheta[1] = a2_prior;
    hattheta[2] = b0_prior;
    hattheta[3] = b1_prior;
  }

  Real am1 = Real(1), am2 = Real(1), ao1 = Real(0);
  Real y_desired[3] = {0, 0, 0};
  Real y_measure[3] = {0, 0, 0};
  Real v_compute[3] = {0, 0, 0};
  Real u_compute[3] = {0, 0, 0};
  Real lambda = Real(1);
  Real Pmax = Real(100);
  Real P[4][4] = {{0}};
  Real hattheta[4] = {0, 0, 0, 0};
  Real a1_prior = 0, a2_prior = 0, b0_prior = 0, b1_prior = 0;
};

// ---------------------------------------------------------------------------
// Mode I legacy regulator (Controller.ino), 3rd order, single axis.
// ---------------------------------------------------------------------------
template <typename Real>
class Legacy3 {
 public:
  void init_estimator(Real forgetting_factor, Real maxP, const Real theta0[6]) {
    lambda = forgetting_factor;
    Pmax = maxP;
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) P[i][j] = (i == j) ? Pmax : Real(0);
    for (int i = 0; i < 6; ++i) hattheta[i] = theta0[i];
    // The sketch keeps the estimate in the globals a1..b2 (STR_estimator writes them
    // back); they must start from theta0 as well, otherwise the control law divides
    // by B(1) = 0 on the first tick.
    a1 = theta0[0];
    a2 = theta0[1];
    a3 = theta0[2];
    b0 = theta0[3];
    b1 = theta0[4];
    b2 = theta0[5];
  }

  void init_controller(Real am1_in, Real am2_in, Real am3_in, Real ao1_in,
                       Real ao2_in) {
    am1 = am1_in;
    am2 = am2_in;
    am3 = am3_in;
    ao1 = ao1_in;
    ao2 = ao2_in;
  }

  // Legacy STR_controller(): computes the command from the *current* measurement.
  Real control_law(Real y_measure_now, Real input_max) {
    const Real hata1 = a1, hata2 = a2, hata3 = a3;
    const Real hatb0 = b0, hatb1 = b1, hatb2 = b2;
    Real xx[5];
    Real bb[5];
    bb[0] = am1 + ao1 - hata1;
    bb[1] = am2 + ao1 * am1 + ao2 - hata2;
    bb[2] = am3 + ao1 * am2 + ao2 * am1 - hata3;
    bb[3] = ao1 * am3 + ao2 * am2;
    bb[4] = ao2 * am3;
    legacy_adjugate_5(hata1, hata2, hata3, hatb0, hatb1, hatb2, bb, xx);
    const Real beta = (Real(1) + am1 + am2 + am3) / (hatb0 + hatb1 + hatb2);
    const Real r1 = xx[0], r2 = xx[1], s0 = xx[2], s1 = xx[3], s2 = xx[4];
    const Real t0 = beta, t1 = beta * ao1, t2 = beta * ao2;
    y_measure[0] = y_measure_now;
    u_compute[0] = -ao1 * u_compute[1] - ao2 * u_compute[2] +
                   t0 * y_desire[0] + t1 * y_desire[1] + t2 * y_desire[2] -
                   s0 * y_measure[0] - s1 * y_measure[1] - s2 * y_measure[2] +
                   (ao1 - r1) * u_actual[1] + (ao2 - r2) * u_actual[2];
    u_actual[0] = u_compute[0];
    if (u_actual[0] > input_max) u_actual[0] = input_max;
    if (u_actual[0] < -input_max) u_actual[0] = -input_max;
    return u_actual[0];
  }

  // Legacy STR_estimator() + Controller_updatedata() history shift.
  void estimator_and_shift(Real u_realised) {
    Real varphi[6] = {-y_measure[1], -y_measure[2], -y_measure[3],
                      u_actual[1],   u_actual[2],   u_actual[3]};
    Real e = y_measure[0];
    Real alpha = lambda;
    Real K[6];
    Real temp[6][6];
    Real P_o[6][6];
    for (int i = 0; i < 6; ++i) e -= hattheta[i] * varphi[i];
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) alpha += varphi[i] * P[i][j] * varphi[j];
    for (int i = 0; i < 6; ++i) {
      K[i] = 0;
      for (int j = 0; j < 6; ++j) K[i] += P[i][j] * varphi[j];
      K[i] = K[i] / alpha;
      hattheta[i] += K[i] * e;
    }
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        temp[i][j] = (i == j) ? (Real(1) - K[i] * varphi[j]) : (-K[i] * varphi[j]);
        P_o[i][j] = P[i][j];
      }
    }
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        P[i][j] = 0;
        for (int k = 0; k < 6; ++k) P[i][j] += temp[i][k] * P_o[k][j];
        P[i][j] = P[i][j] / lambda;
      }
    }
    a1 = hattheta[0];
    a2 = hattheta[1];
    a3 = hattheta[2];
    b0 = hattheta[3];
    b1 = hattheta[4];
    b2 = hattheta[5];
    y_measure[3] = y_measure[2];
    y_measure[2] = y_measure[1];
    y_measure[1] = y_measure[0];
    u_compute[3] = u_compute[2];
    u_compute[2] = u_compute[1];
    u_compute[1] = u_compute[0];
    u_actual[3] = u_actual[2];
    u_actual[2] = u_actual[1];
    u_actual[1] = u_realised;
  }

  Real am1 = 0, am2 = 0, am3 = 0, ao1 = 0, ao2 = 0;
  Real a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0;
  Real lambda = Real(1), Pmax = Real(100);
  Real P[6][6] = {{0}};
  Real hattheta[6] = {0, 0, 0, 0, 0, 0};
  Real y_desire[4] = {0, 0, 0, 0};
  Real y_measure[4] = {0, 0, 0, 0};
  Real u_compute[4] = {0, 0, 0, 0};
  Real u_actual[4] = {0, 0, 0, 0};
};

}  // namespace legacy

#endif  // TESTS_CPP_LEGACY_REFERENCE_H_
