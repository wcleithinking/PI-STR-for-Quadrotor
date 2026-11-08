// ctest entry point: `str_core_tests [suite]` (no argument or "all" runs everything).

#include <cstring>

#include "dispatch.h"

int main(int argc, char** argv) {
  const char* name = (argc > 1) ? argv[1] : nullptr;
  return str_test::run_suites(name) == 0 ? 0 : 1;
}
