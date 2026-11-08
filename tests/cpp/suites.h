// Test suite entry points.  Each suite is implemented in its own .cpp and is run
// either by the `str_core_tests` executable (one ctest entry per suite) or by the
// shared-library entry point used where native execution is unavailable.

#ifndef TESTS_CPP_SUITES_H_
#define TESTS_CPP_SUITES_H_

namespace str_test {

void suite_linear_solve();
void suite_diophantine();
void suite_projection();
void suite_rls();
void suite_guards();
void suite_equivalence();
void suite_golden();

// Replays one golden fixture (see suite_golden.cpp).  Returns 0 = PASS, 1 = mismatch,
// 2 = structural problem (missing file / columns).  `report` receives the per-column
// max |delta| and relative error; `tol` <= 0 means "use the fixture's tolerance_rel".
int run_golden_check(const char* csv_path, double tol, char* report, int cap);

}  // namespace str_test

#endif  // TESTS_CPP_SUITES_H_
