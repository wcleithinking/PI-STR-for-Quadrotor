// Minimal, dependency-free test harness: no third-party framework is allowed in
// this repository (see task constraints), and the same suite code has to run both
// through `ctest` (normal machines) and through a shared-library entry point
// (environments that refuse to execute freshly built binaries -- see the report).

#ifndef TESTS_CPP_STR_TEST_HARNESS_H_
#define TESTS_CPP_STR_TEST_HARNESS_H_

#include <cmath>
#include <cstdarg>
#include <cstdio>

namespace str_test {

inline int& check_count() {
  static int n = 0;
  return n;
}
inline int& failure_count() {
  static int n = 0;
  return n;
}

inline bool is_finite(double v) {
  return v == v && v != HUGE_VAL && v != -HUGE_VAL;
}

inline void record(bool ok, const char* what, const char* file, int line) {
  ++check_count();
  if (!ok) {
    ++failure_count();
    std::printf("FAIL %s:%d  %s\n", file, line, what);
  }
}

// Same as record(), but with a printf-formatted message so failures carry the
// offending numbers.
inline void record_msg(bool ok, const char* file, int line, const char* fmt, ...) {
  ++check_count();
  if (ok) return;
  ++failure_count();
  char buf[512];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  std::printf("FAIL %s:%d  %s\n", file, line, buf);
}

inline int summary(const char* suite) {
  std::printf("[%s] %d checks, %d failures\n", suite, check_count(), failure_count());
  return failure_count();
}

}  // namespace str_test

#define STR_CHECK(cond) ::str_test::record((cond), #cond, __FILE__, __LINE__)

#define STR_CHECK_MSG(cond, ...) \
  ::str_test::record_msg((cond), __FILE__, __LINE__, __VA_ARGS__)

// Absolute tolerance comparison that also fails on NaN.
#define STR_CHECK_NEAR(a, b, tol)                                              \
  ::str_test::record_msg(                                                      \
      ::str_test::is_finite((a)) && ::str_test::is_finite((b)) &&              \
          std::fabs((a) - (b)) <= (tol),                                       \
      __FILE__, __LINE__, "|%s - %s| = %.12g > %.3g", #a, #b,                  \
      static_cast<double>((a) - (b)), static_cast<double>(tol))

// Relative (scale-aware) comparison, used for "behaviour equivalent" claims.
#define STR_CHECK_REL(a, b, scale, tol)                                        \
  ::str_test::record_msg(                                                      \
      ::str_test::is_finite((a)) && ::str_test::is_finite((b)) &&              \
          std::fabs((a) - (b)) <= (tol) * ((scale) + 1.0),                     \
      __FILE__, __LINE__, "|%s - %s| = %.12g > %.3g * (1+%.3g)", #a, #b,       \
      static_cast<double>((a) - (b)), static_cast<double>(tol),                \
      static_cast<double>(scale))

#endif  // TESTS_CPP_STR_TEST_HARNESS_H_
