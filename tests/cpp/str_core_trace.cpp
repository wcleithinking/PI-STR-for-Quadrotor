// str_core_trace — deterministic driver for the Python <-> C++ golden-vector
// comparison of CONTROL_SPEC §8.
//
// The C++ core cannot run the python plant and vice versa, so the interface is a
// *replay*: the Python harness records, for every controller tick, the reference
// y_des(k), the measured y(k) and the physically realised command u_actual(k), and
// this tool replays them through the C++ core.  Python and C++ then have to agree
// on the design command, the clamped command and theta_hat.
//
//   stdin  : "<y_des> <y> [u_actual|auto]\n"
//            u_actual omitted or "auto" -> commit the command returned this tick
//   stdout : "tick,y_des,y,u_actual,u_design,u_cmd,theta...,r...,s...,t...,flags"
//
// Usage:
//   str_core_trace [--order 2|3] [--lambda X] [--pmax X] [--zeta X] [--wn X]
//                  [--mu X] [--h X] [--umax X] [--percent X]
//                  [--theta0 a1,a2,b0,b1]        (order 2: 4 values)
//                  [--theta0 a1,a2,a3,b0,b1,b2]  (order 3: 6 values)
//                  [--am1 X --am2 X --am3 X --ao1 X --ao2 X]
//                  [--tf-alpha X --tf-beta X]
//                  [--selftest]
//
// --theta0 is the golden-vector handshake: both sides must start from exactly the same
// parameter vector, otherwise tick 0 already disagrees.  A malformed value is a hard
// error with the exact expected syntax printed, never a silent fallback.
//
// Defaults:
//   order 2: Mode II, h = 0.04, lambda = 0.99, Pmax = 5, zeta = 0.7, wn = 5, mu = 0,
//            theta0 = percent * the §3 analytic guess from tf_alpha/tf_beta
//            (defaults tf_alpha = 64, tf_beta = 47.5, percent = 0.75, §8).
//            NOTE: this is *not* the parameter set hard-coded in the Mode II sketch,
//            which has the exponent sign flipped and a ms/s unit slip (see the report).
//   order 3: Mode I roll axis, lambda = 0.98, Pmax = 100, am/ao from Controller.ino,
//            theta0 = the §3 analytic guess above (uniformly, the 3rd-order sketch still
//            uses its own hand-written guess).
//
// `--selftest` runs an internal closed-loop smoke test (registered with ctest).

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "arx_plant.h"
#include "str_core.h"

namespace {

template <typename Real, int Order>
struct Runner {
  using Reg = str_core::SelfTuningRegulator<Real, Order>;

  Reg reg;
  Real u_max = Real(50);
  bool estimator_first = true;  // Mode II order; Mode I runs the law first
  long tick = 0;
  bool ok = true;

  void print_header() const {
    std::printf("tick,y_des,y,u_actual,u_design,u_cmd");
    for (int i = 0; i < Reg::kNumParams; ++i) std::printf(",theta%d", i);
    for (int i = 0; i < Reg::kNumR; ++i) std::printf(",r%d", i + 1);
    for (int j = 0; j < Reg::kNumS; ++j) std::printf(",s%d", j);
    for (int j = 0; j < Reg::kNumT; ++j) std::printf(",t%d", j);
    std::printf(",flags\n");
  }

  // u_actual_override: value to feed back (the realised command); NaN = use the
  // command this tick produced.
  void step(Real y_des, Real y, Real u_actual_override) {
    (void)reg.begin_tick(y_des, y);
    if (estimator_first) (void)reg.update_estimator();
    const Real u_cmd = reg.compute_control(u_max);
    const Real u_actual =
        std::isfinite(static_cast<double>(u_actual_override)) ? u_actual_override : u_cmd;
    reg.commit_actual(u_actual);
    if (!estimator_first) (void)reg.update_estimator();

    ok = ok && std::isfinite(static_cast<double>(u_cmd)) &&
         std::isfinite(static_cast<double>(u_actual)) &&
         std::isfinite(static_cast<double>(reg.theta_hat()[0]));

    std::printf("%ld,%.17g,%.17g,%.17g,%.17g,%.17g", tick,
                static_cast<double>(y_des), static_cast<double>(y),
                static_cast<double>(u_actual),
                static_cast<double>(reg.last_design_command()),
                static_cast<double>(u_cmd));
    for (int i = 0; i < Reg::kNumParams; ++i)
      std::printf(",%.17g",
                  static_cast<double>(reg.theta_hat()[static_cast<std::size_t>(i)]));
    for (int i = 0; i < Reg::kNumR; ++i)
      std::printf(",%.17g", static_cast<double>(reg.r()[static_cast<std::size_t>(i)]));
    for (int j = 0; j < Reg::kNumS; ++j)
      std::printf(",%.17g", static_cast<double>(reg.s()[static_cast<std::size_t>(j)]));
    for (int j = 0; j < Reg::kNumT; ++j)
      std::printf(",%.17g", static_cast<double>(reg.t()[static_cast<std::size_t>(j)]));
    std::printf(",%u\n", static_cast<unsigned>(reg.flags()));
    ++tick;
  }

  void replay(std::FILE* in) {
    char buf[512];
    while (std::fgets(buf, sizeof(buf), in) != nullptr) {
      double yd = 0, y = 0;
      char actual[64] = "auto";
      const int n = std::sscanf(buf, "%lf %lf %63s", &yd, &y, actual);
      if (n < 2) continue;  // header / blank line
      const bool use_auto = (n < 3) || (std::strcmp(actual, "auto") == 0);
      const double ua = use_auto ? std::nan("") : std::atof(actual);
      step(static_cast<Real>(yd), static_cast<Real>(y), static_cast<Real>(ua));
    }
  }
};

// Parse "--theta0 v1,v2,..." into `out`.  Returns 1 = not present, 0 = parsed,
// -1 = malformed (the caller prints the usage and exits non-zero).
int arg_theta0(int argc, char** argv, double* out, int expected) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--theta0") != 0) continue;
    const char* text = argv[i + 1];
    int count = 0;
    const char* p = text;
    while (*p != '\0') {
      char* end = nullptr;
      const double v = std::strtod(p, &end);
      if (end == p) return -1;  // not a number
      if (count >= expected) return -1;  // too many
      out[count++] = v;
      p = end;
      if (*p == ',') {
        ++p;
      } else if (*p != '\0') {
        return -1;  // garbage between values
      }
    }
    if (count != expected) return -1;
    return 0;
  }
  return 1;
}

double arg_real(int argc, char** argv, const char* name, double fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atof(argv[i + 1]);
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return true;
  }
  return false;
}

// Closed-loop smoke test against an internal ARX plant, so the tool cannot rot.
template <typename Real, int Order>
int self_test(bool estimator_first) {
  Runner<Real, Order> r;
  r.estimator_first = estimator_first;
  std::array<Real, 2 * Order> truth{};
  if constexpr (Order == 2) {
    truth = {Real(-1.5), Real(0.6), Real(0.02), Real(0.011)};
  } else {
    truth = {Real(-1.9), Real(1.25), Real(-0.3), Real(0.015), Real(0.014),
             Real(0.013)};
  }
  str_test::ArxPlant<Real, Order> plant(truth);
  for (int k = 0; k < 800 && r.ok; ++k) {
    const Real y = plant.y();
    const Real y_des = (k / 100) % 2 == 0 ? Real(0.1) : Real(-0.05);
    r.step(y_des, y, std::nan(""));
    (void)plant.step(r.reg.last_command());
  }
  return r.ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const int order = static_cast<int>(arg_real(argc, argv, "--order", 2.0));
  const double lambda = arg_real(argc, argv, "--lambda", order == 2 ? 0.99 : 0.98);
  const double pmax = arg_real(argc, argv, "--pmax", order == 2 ? 5.0 : 100.0);
  const double zeta = arg_real(argc, argv, "--zeta", 0.7);
  const double wn = arg_real(argc, argv, "--wn", 5.0);
  const double mu = arg_real(argc, argv, "--mu", 0.0);
  const double h = arg_real(argc, argv, "--h", order == 2 ? 0.04 : 0.1);
  const double u_max = arg_real(argc, argv, "--umax", order == 2 ? 50.0 : 100.0);
  const double percent = arg_real(argc, argv, "--percent", 0.75);
  const double tf_alpha = arg_real(argc, argv, "--tf-alpha", 64.0);
  const double tf_beta = arg_real(argc, argv, "--tf-beta", 47.5);
  const bool selftest = has_flag(argc, argv, "--selftest");
  const bool use_theta0 = has_flag(argc, argv, "--theta0");
  const char* theta0_text = "";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--theta0") == 0) theta0_text = argv[i + 1];
  }
  if (use_theta0 && order != 2 && order != 3) {
    std::fprintf(stderr, "error: --theta0 needs --order 2 or --order 3\n");
    return 2;
  }

  if (order == 2) {
    using Real = double;
    Runner<Real, 2> r;
    r.u_max = static_cast<Real>(u_max);
    r.estimator_first = true;
    using Vec = typename Runner<Real, 2>::Reg::Vec;
    Vec theta0{};
    int t0_status = -1;
    if (use_theta0) {
      double vals[4];
      t0_status = arg_theta0(argc, argv, vals, 4);
      if (t0_status != 0) {
        std::fprintf(stderr,
                     "error: --theta0 for --order 2 expects 4 comma-separated numbers:\n"
                     "       a1,a2,b0,b1   (got '%s')\n", theta0_text);
        return 2;
      }
      for (int i = 0; i < 4; ++i) theta0[static_cast<std::size_t>(i)] = static_cast<Real>(vals[i]);
    } else {
      // §3 analytic guess, with tf_alpha/tf_beta defaulting to the MATLAB values.
      const Real a2 = static_cast<Real>(std::exp(-tf_beta * h));
      const Real a1 = Real(-1) - a2;
      const Real b0 = static_cast<Real>(tf_alpha / tf_beta * h -
                                        (tf_alpha / (tf_beta * tf_beta)) * (1.0 - static_cast<double>(a2)));
      const Real b1 = static_cast<Real>((1.0 / tf_beta) * (1.0 - static_cast<double>(a2)) *
                                            ((tf_alpha / tf_beta) * (1.0 - static_cast<double>(a2))) -
                                        static_cast<double>(a2) * static_cast<double>(b0));
      const Real p = static_cast<Real>(percent);
      theta0 = Vec{{a1 * p, a2 * p, b0 * p, b1 * p}};
    }
    r.reg.init_estimator(static_cast<Real>(lambda), static_cast<Real>(pmax), theta0);
    r.reg.set_second_order_reference(static_cast<Real>(zeta), static_cast<Real>(wn),
                                     static_cast<Real>(mu), static_cast<Real>(h));
    if (selftest) return self_test<Real, 2>(true);
    r.print_header();
    r.replay(stdin);
    return r.ok ? 0 : 1;
  }

  if (order == 3) {
    using Real = double;
    Runner<Real, 3> r;
    r.u_max = static_cast<Real>(u_max);
    r.estimator_first = false;  // Mode I: control law, then estimator
    const Real am[3] = {
        static_cast<Real>(arg_real(argc, argv, "--am1", -1.4431986644)),
        static_cast<Real>(arg_real(argc, argv, "--am2", 0.7708989340)),
        static_cast<Real>(arg_real(argc, argv, "--am3", -0.1380692400))};
    const Real ao[2] = {static_cast<Real>(arg_real(argc, argv, "--ao1", 0.8)),
                        static_cast<Real>(arg_real(argc, argv, "--ao2", 0.16))};
    using Vec3 = typename Runner<Real, 3>::Reg::Vec;
    Vec3 theta0 = {Real(-1.55), Real(0.55), Real(-0.18),
                   Real(0.02),  Real(0.02), Real(0.02)};
    if (use_theta0) {
      double vals[6];
      if (arg_theta0(argc, argv, vals, 6) != 0) {
        std::fprintf(stderr,
                     "error: --theta0 for --order 3 expects 6 comma-separated numbers:\n"
                     "       a1,a2,a3,b0,b1,b2   (got '%s')\n", theta0_text);
        return 2;
      }
      for (int i = 0; i < 6; ++i) theta0[static_cast<std::size_t>(i)] = static_cast<Real>(vals[i]);
    }
    r.reg.init_estimator(static_cast<Real>(lambda), static_cast<Real>(pmax), theta0);
    r.reg.set_reference(std::array<Real, 3>{{am[0], am[1], am[2]}},
                        std::array<Real, 2>{{ao[0], ao[1]}});
    if (selftest) return self_test<Real, 3>(false);
    r.print_header();
    r.replay(stdin);
    return r.ok ? 0 : 1;
  }

  std::fprintf(stderr, "unsupported --order %d (use 2 or 3)\n", order);
  return 2;
}
