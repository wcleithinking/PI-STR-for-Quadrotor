// Suite: C++ side of the §8 golden-vector comparison.
//
// The Python simulator is the numerical oracle; `tests/golden/step_response_case.csv`
// (schema 2) records one Mode II / order 2 / SI run: for every controller tick it stores
//   # config: h, zeta, wn, mu, lambda, pmax, percent, umax
//   # theta0: the initial parameter vector both sides must start from
//   t, y_des, y, u_actual, u_design, u_cmd, theta0..theta3
// so this suite needs no plant and no sensors: it replays (y_des, y) through
// `str_core::Str2d` and compares the design command, the saturating command and the
// parameter estimate against the oracle.
//
// Entry points:
//   * `str_test::suite_golden()`   -- used by the ctest executable / dispatch table
//   * `extern "C" int str_core_check_golden(const char*, double tol, char*, int)`
//     (`ctypes_entry.cpp`) -- the only route usable where new binaries cannot be
//     executed (this development box), so it is the primary verification path.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "str_core.h"
#include "str_test_harness.h"
#include "suites.h"

// Absolute path to the fixture directory, injected by CMake so the check works from any
// working directory.  Fallback: relative to the repository root.
#ifndef STR_GOLDEN_DIR
#define STR_GOLDEN_DIR "tests/golden"
#endif

namespace str_test {
namespace {

struct GoldenConfig {
  double h = 0.04;
  double zeta = 0.7;
  double wn = 5.0;
  double mu = 0.0;
  double lambda = 0.99;
  double pmax = 5.0;
  double percent = 0.75;
  double umax = 50.0;
  double tolerance = 1e-9;
  bool have_theta0 = false;
  std::array<double, 4> theta0{{0, 0, 0, 0}};
};

char* trim(char* s) {
  while (*s == ' ' || *s == '\t') ++s;
  char* end = s + std::strlen(s);
  while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) {
    --end;
  }
  *end = '\0';
  return s;
}

// Runs one `key=value` pair from the "# config:" line into `cfg`.
bool apply_config_token(const char* token, GoldenConfig* cfg) {
  const char* eq = std::strchr(token, '=');
  if (eq == nullptr) return false;
  const std::string key(token, static_cast<std::size_t>(eq - token));
  const double value = std::atof(eq + 1);
  if (key == "h") cfg->h = value;
  else if (key == "zeta") cfg->zeta = value;
  else if (key == "wn") cfg->wn = value;
  else if (key == "mu") cfg->mu = value;
  else if (key == "lambda") cfg->lambda = value;
  else if (key == "pmax") cfg->pmax = value;
  else if (key == "percent") cfg->percent = value;
  else if (key == "umax") cfg->umax = value;
  else if (key == "tolerance_rel") cfg->tolerance = value;
  else return false;
  return true;
}

// Parses "a,b,c,d" into `out`; returns the number of values found.
int parse_doubles(const char* text, double* out, int max_values) {
  int count = 0;
  const char* p = text;
  while (*p != '\0' && count < max_values) {
    char* end = nullptr;
    const double v = std::strtod(p, &end);
    if (end == p) break;
    out[count++] = v;
    p = end;
    if (*p == ',') ++p;
  }
  return count;
}

struct Stats {
  double max_abs = 0.0;
  double max_ref = 0.0;
  int bad_rows = 0;
};

void accumulate(Stats* s, double got, double want) {
  s->max_abs = std::fmax(s->max_abs, std::fabs(got - want));
  s->max_ref = std::fmax(s->max_ref, std::fabs(want));
}

double ratio(const Stats& s) {
  const double scale = (s.max_ref > 1e-12) ? s.max_ref : 1e-12;
  return s.max_abs / scale;
}

int append(char* report, int cap, int* used, const char* fmt, ...) {
  if (*used >= cap) return *used;
  va_list args;
  va_start(args, fmt);
  const int n = std::vsnprintf(report + *used, static_cast<std::size_t>(cap - *used), fmt, args);
  va_end(args);
  if (n > 0) *used += (n < cap - *used) ? n : (cap - *used - 1);
  return *used;
}

// Replays the fixture.  Returns 0 on PASS, 1 on mismatch, 2 on a structural problem.
int replay_golden(const char* csv_path, double tol_override, char* report, int cap) {
  int used = 0;
  if (report != nullptr && cap > 0) report[0] = '\0';

  std::FILE* f = std::fopen(csv_path, "r");
  if (f == nullptr) {
    append(report, cap, &used, "ERROR: cannot open golden fixture '%s'\n", csv_path);
    return 2;
  }

  GoldenConfig cfg;
  std::string header;
  char line[4096];
  bool header_seen = false;

  // ---- comments + header -------------------------------------------------
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    char* text = trim(line);
    if (text[0] == '\0') continue;
    if (text[0] == '#') {
      if (std::strstr(text, "config:") != nullptr) {
        char* cursor = std::strstr(text, "config:") + 7;
        char* token = std::strtok(cursor, " \t");
        while (token != nullptr) {
          (void)apply_config_token(token, &cfg);
          token = std::strtok(nullptr, " \t");
        }
      } else if (std::strstr(text, "theta0=") != nullptr) {
        char* cursor = std::strstr(text, "theta0=") + 7;
        const int n = parse_doubles(cursor, cfg.theta0.data(), 4);
        cfg.have_theta0 = (n == 4);
      } else if (std::strstr(text, "tolerance_rel=") != nullptr) {
        cfg.tolerance = std::atof(std::strstr(text, "tolerance_rel=") + 14);
      }
      continue;
    }
    header = text;  // first non-comment line is the column header
    header_seen = true;
    break;
  }
  if (!header_seen) {
    std::fclose(f);
    append(report, cap, &used, "ERROR: no column header found in '%s'\n", csv_path);
    return 2;
  }

  // ---- column map --------------------------------------------------------
  int col_t = -1, col_ydes = -1, col_y = -1, col_uact = -1, col_udes = -1, col_ucmd = -1;
  int col_theta[4] = {-1, -1, -1, -1};
  int ncols = 0;
  {
    char buf[4096];
    std::snprintf(buf, sizeof(buf), "%s", header.c_str());
    char* token = std::strtok(buf, ",");
    while (token != nullptr && ncols < 64) {
      const std::string name(trim(token));
      if (name == "t") col_t = ncols;
      else if (name == "y_des") col_ydes = ncols;
      else if (name == "y") col_y = ncols;
      else if (name == "u_actual") col_uact = ncols;
      else if (name == "u_design") col_udes = ncols;
      else if (name == "u_cmd") col_ucmd = ncols;
      else if (name == "theta0") col_theta[0] = ncols;
      else if (name == "theta1") col_theta[1] = ncols;
      else if (name == "theta2") col_theta[2] = ncols;
      else if (name == "theta3") col_theta[3] = ncols;
      ++ncols;
      token = std::strtok(nullptr, ",");
    }
  }
  const bool ok_cols = col_ydes >= 0 && col_y >= 0 && col_udes >= 0 && col_ucmd >= 0 &&
                       col_theta[0] >= 0 && col_theta[1] >= 0 && col_theta[2] >= 0 &&
                       col_theta[3] >= 0;
  if (!ok_cols) {
    std::fclose(f);
    append(report, cap, &used,
           "ERROR: fixture is missing required columns "
           "(y_des, y, u_design, u_cmd, theta0..theta3); header: %s\n",
           header.c_str());
    return 2;
  }

  // ---- replay ------------------------------------------------------------
  str_core::Str2d reg;
  reg.init_estimator(cfg.lambda, cfg.pmax,
                     str_core::Str2d::Vec{cfg.theta0[0], cfg.theta0[1], cfg.theta0[2],
                                          cfg.theta0[3]});
  reg.set_second_order_reference(cfg.zeta, cfg.wn, cfg.mu, cfg.h);

  Stats s_udes, s_ucmd, s_theta[4];
  int rows = 0;
  int rejected_ticks = 0;
  unsigned guard_flags = 0;

  while (std::fgets(line, sizeof(line), f) != nullptr) {
    char* text = trim(line);
    if (text[0] == '\0' || text[0] == '#') continue;

    double value[64];
    int count = 0;
    char* token = std::strtok(text, ",");
    while (token != nullptr && count < 64) {
      value[count++] = std::atof(trim(token));
      token = std::strtok(nullptr, ",");
    }
    if (count < ncols) continue;  // truncated row

    const double y_des = value[col_ydes];
    const double y = value[col_y];
    const double u_actual = (col_uact >= 0) ? value[col_uact] : 0.0;
    (void)col_t;

    if (!reg.begin_tick(y_des, y)) ++rejected_ticks;
    if (!reg.update_estimator()) ++rejected_ticks;
    const double u_cmd = reg.compute_control(cfg.umax);
    guard_flags |= reg.flags();

    accumulate(&s_udes, reg.last_design_command(), value[col_udes]);
    accumulate(&s_ucmd, u_cmd, value[col_ucmd]);
    for (int i = 0; i < 4; ++i) {
      accumulate(&s_theta[i], reg.theta_hat()[static_cast<std::size_t>(i)],
                 value[col_theta[i]]);
    }
    reg.commit_actual(u_actual);
    ++rows;
  }
  std::fclose(f);

  if (rows == 0) {
    append(report, cap, &used, "ERROR: fixture '%s' has no data rows\n", csv_path);
    return 2;
  }

  // ---- report ------------------------------------------------------------
  const double tol = (tol_override > 0.0) ? tol_override : cfg.tolerance;
  bool pass = (rejected_ticks == 0);
  append(report, cap, &used,
         "golden: %s\nrows=%d theta0=[%+.7g, %+.7g, %+.7g, %+.7g] "
         "h=%g zeta=%g wn=%g mu=%g lambda=%g pmax=%g umax=%g\n",
         csv_path, rows, cfg.theta0[0], cfg.theta0[1], cfg.theta0[2], cfg.theta0[3], cfg.h,
         cfg.zeta, cfg.wn, cfg.mu, cfg.lambda, cfg.pmax, cfg.umax);
  if (rejected_ticks != 0) {
    append(report, cap, &used,
           "FAIL rejected_ticks=%d (a §6 guard fired; the guard-free path is required)\n",
           rejected_ticks);
  }

  const struct {
    const char* name;
    const Stats* stats;
  } columns[] = {{"u_design", &s_udes}, {"u_cmd", &s_ucmd},
                 {"theta0", &s_theta[0]}, {"theta1", &s_theta[1]},
                 {"theta2", &s_theta[2]}, {"theta3", &s_theta[3]}};
  for (const auto& c : columns) {
    const double r = ratio(*c.stats);
    const bool ok = (r <= tol);
    pass = pass && ok;
    append(report, cap, &used, "  %-9s max|d|=%.3e max|ref|=%.3e ratio=%.3e %s\n",
           c.name, c.stats->max_abs, c.stats->max_ref, r, ok ? "PASS" : "FAIL");
  }
  append(report, cap, &used, "RESULT: %s (tol=%.3g, guard_flags=0x%x)\n",
         pass ? "PASS" : "FAIL", tol, guard_flags);
  return pass ? 0 : 1;
}

std::string default_fixture_path() {
  std::string path(STR_GOLDEN_DIR);
  if (!path.empty() && path.back() != '/') path.push_back('/');
  path += "step_response_case.csv";
  return path;
}

}  // namespace

int run_golden_check(const char* csv_path, double tol, char* report, int cap) {
  return replay_golden(csv_path, tol, report, cap);
}

void suite_golden() {
  char report[8192];
  const std::string path = default_fixture_path();
  const int rc = replay_golden(path.c_str(), 0.0, report, static_cast<int>(sizeof(report)));
  std::printf("%s", report);
  STR_CHECK_MSG(rc == 0, "golden replay failed (rc=%d) for fixture '%s'", rc, path.c_str());
}

}  // namespace str_test
