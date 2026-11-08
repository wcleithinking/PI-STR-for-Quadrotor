// Suite registry shared by the ctest executable and the shared-library entry
// point (some environments cannot execute freshly built binaries).

#ifndef TESTS_CPP_DISPATCH_H_
#define TESTS_CPP_DISPATCH_H_

#include <cstdio>
#include <cstring>

#include "str_test_harness.h"
#include "suites.h"

namespace str_test {

struct SuiteEntry {
  const char* name;
  void (*fn)();
};

inline const SuiteEntry* suite_table(int* count) {
  static const SuiteEntry kSuites[] = {
      {"linear_solve", &str_test::suite_linear_solve},
      {"diophantine", &str_test::suite_diophantine},
      {"projection", &str_test::suite_projection},
      {"rls", &str_test::suite_rls},
      {"guards", &str_test::suite_guards},
      {"equivalence", &str_test::suite_equivalence},
      {"golden", &str_test::suite_golden},
  };
  *count = static_cast<int>(sizeof(kSuites) / sizeof(kSuites[0]));
  return kSuites;
}

// Runs every suite, or the single one named `name` ("all" / nullptr = everything).
// Returns the number of failed checks.
inline int run_suites(const char* name) {
  int count = 0;
  const SuiteEntry* table = suite_table(&count);
  const bool all = (name == nullptr) || (std::strcmp(name, "all") == 0);
  int matched = 0;
  for (int i = 0; i < count; ++i) {
    if (!all && std::strcmp(table[i].name, name) != 0) continue;
    ++matched;
    std::printf("=== suite: %s ===\n", table[i].name);
    table[i].fn();
  }
  if (matched == 0) {
    std::printf("unknown suite '%s'; available:", name == nullptr ? "" : name);
    for (int i = 0; i < count; ++i) std::printf(" %s", table[i].name);
    std::printf("\n");
    return 1;
  }
  return str_test::summary(all ? "all suites" : name);
}

}  // namespace str_test

#endif  // TESTS_CPP_DISPATCH_H_
