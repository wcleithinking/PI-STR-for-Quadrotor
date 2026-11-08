// Shared-library entry point for environments that refuse to execute freshly
// built binaries (this session: every new Mach-O is killed with SIGKILL, while
// dlopen() of the same code works).  The suite code is identical to the ctest
// executable, so the assertions verified this way are the same ones ctest runs.
//
//   python3 -c "import ctypes; l=ctypes.CDLL('./build/libstr_core_verify.dylib'); \
//               print(l.str_core_run_tests(b'all'))"

#include <cstring>
#include <string>

#include "dispatch.h"

#ifndef STR_GOLDEN_DIR
#define STR_GOLDEN_DIR "tests/golden"
#endif

extern "C" int str_core_run_tests(const char* suite) {
  const int failures = str_test::run_suites(suite);
  std::fflush(stdout);
  return failures;
}

// Replay one golden fixture against the C++ core (CONTROL_SPEC §8).  This is the
// verification route that works where freshly built binaries cannot be executed.
//
//   int str_core_check_golden(const char* csv_path, double tol, char* report, int cap);
//     csv_path : fixture path; "" / nullptr -> tests/golden/step_response_case.csv
//     tol      : relative tolerance; <= 0 -> use the fixture's own "# tolerance_rel="
//     report   : optional caller buffer, NUL-terminated, one line per compared column
//                plus a "RESULT: PASS|FAIL" line
//     return   : 0 = PASS, 1 = mismatch, 2 = structural problem (no file / bad columns)
extern "C" int str_core_check_golden(const char* csv_path, double tol, char* report,
                                     int cap) {
  static char internal[8192];
  char* out = (report != nullptr && cap > 0) ? report : internal;
  const int out_cap = (report != nullptr && cap > 0) ? cap : static_cast<int>(sizeof(internal));
  const std::string path =
      (csv_path != nullptr && csv_path[0] != '\0') ? std::string(csv_path) : std::string();
  int rc = 0;
  if (path.empty()) {
    // No path given: use the compile-time fixture directory.
    std::string def(STR_GOLDEN_DIR);
    if (!def.empty() && def.back() != '/') def.push_back('/');
    def += "step_response_case.csv";
    rc = str_test::run_golden_check(def.c_str(), tol, out, out_cap);
  } else {
    rc = str_test::run_golden_check(path.c_str(), tol, out, out_cap);
  }
  std::fflush(stdout);
  return rc;
}
