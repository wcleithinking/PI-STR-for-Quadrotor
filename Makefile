# ===========================================================================
# PI-STR-for-Quadrotor -- developer entry points.
#
# Run from anywhere: every path is derived from this Makefile's own location, so
# `make -f /abs/path/Makefile <target>` behaves identically to `make <target>`
# in the repository root.
#
#   make test            pytest + C++ host tests
#   make test-py         pytest on sim/python/tests
#   make test-cpp        cmake build + C++ assertions via the shared-library entry
#                        point (works where exec of a fresh binary is blocked)
#   make test-cpp-ctest  the same suites through ctest (needs to exec new binaries)
#   make build-cpp       configure + build the host core only
#   make sim             one Python simulation run into results/
#   make clean           remove build/, caches, egg-info (results/ is kept)
#
# Nothing here installs packages and nothing here touches the network.
# See README.md and docs/quickstart.md.
# ===========================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Directory of this Makefile, without a trailing slash.
REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

SIM_DIR   := $(REPO_ROOT)/sim/python
BUILD_DIR ?= $(REPO_ROOT)/build
RESULTS   ?= $(REPO_ROOT)/results

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
CMAKE  ?= cmake
CTEST  ?= ctest

# Primary C++ verification route: the shared library built by CMake exports a C
# entry point, so `dlopen` works in sandboxes/CI that SIGKILL freshly linked
# binaries. `make test-cpp-ctest` is the alternative when exec is allowed.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  VERIFY_LIB ?= $(BUILD_DIR)/libstr_core_verify.dylib
else
  VERIFY_LIB ?= $(BUILD_DIR)/libstr_core_verify.so
endif
# Suite selector handed to str_core_run_tests(); 'all' runs every suite. Note the
# reported check count shifts with the selection (golden suite adds a few).
SUITE ?= all

# `make sim` knobs. The flags mirror `python3 -m pistr_sim.cli run --help`.
SCENARIO   ?= hover
CONTROLLER ?= pi_str
FIDELITY   ?= high
ARGS       ?=

.PHONY: help test test-cpp test-cpp-ctest build-cpp test-py sim \
        env-setup test-3d demo-3d clean

# Optional 3-D environment (MuJoCo). Wheels are unpacked into the build tree so
# nothing is installed into site-packages; `make clean` removes them too.
DEPS_3D ?= $(BUILD_DIR)/mjsim
ENV_OUT ?= $(RESULTS)/env

help:
	@printf '%s\n' \
	  'PI-STR-for-Quadrotor' \
	  '' \
	  '  make test-py         pytest on sim/python/tests' \
	  '  make test-cpp        cmake build + C++ assertions via ctypes (0 == PASS)' \
	  '  make test-cpp-ctest  the same suites through ctest' \
	  '  make build-cpp       configure + build the host core only' \
	  '  make test            test-py + test-cpp' \
	  '  make sim             python3 -m pistr_sim.cli run (hover / pi_str / high)' \
	  '  make env-setup       unpack MuJoCo into build/mjsim (needs network)' \
	  '  make test-3d         MuJoCo env gates V1-V8 (skipped if not set up)' \
	  '  make demo-3d         render the 3-D demo flight into results/env' \
	  '  make clean           drop build/, __pycache__, .pytest_cache, egg-info' \
	  '' \
	  'Overrides: PYTHON=, BUILD_DIR=, RESULTS=, SCENARIO=, CONTROLLER=,' \
	  '           FIDELITY=paper|high, SUITE=all|<suite>, VERIFY_LIB=,' \
	  '           ARGS="<extra CLI flags>"' \
	  '' \
	  'Details: docs/quickstart.md'

test: test-py test-cpp

# --- Python simulation (the numerical oracle of this repository) ------------
test-py:
	@test -d "$(SIM_DIR)/tests" || { \
	  echo "error: $(SIM_DIR)/tests does not exist yet."; \
	  echo "       The pytest suite is part of the simulator work in progress;"; \
	  echo "       see docs/architecture.md section 8."; \
	  exit 1; }
	$(PYTEST) "$(SIM_DIR)/tests"

# --- C++ host tests (firmware/lib/str_core, no Arduino dependency) ----------
# Two entry points, one test library (str_core_testlib), so the assertions are
# identical -- see README "Run the host tests".
#
#   test-cpp        builds, then calls the str_core_verify C entry point via
#                   ctypes. This is the route that also works in restricted
#                   environments where a freshly linked binary cannot be exec'd.
#   test-cpp-ctest  builds, then runs the same suites through ctest; keep it for
#                   machines where exec works and for per-suite progress output.
#
# `check` here means: ctypes entry returns 0 (number of failed checks).
build-cpp:
	@test -f "$(REPO_ROOT)/CMakeLists.txt" || { \
	  echo "error: $(REPO_ROOT)/CMakeLists.txt does not exist yet."; \
	  echo "       The host build for firmware/lib/str_core comes with the firmware"; \
	  echo "       refactor; see docs/architecture.md section 8."; \
	  exit 1; }
	$(CMAKE) -S "$(REPO_ROOT)" -B "$(BUILD_DIR)"
	$(CMAKE) --build "$(BUILD_DIR)" --parallel

test-cpp: build-cpp
	@test -f "$(VERIFY_LIB)" || { \
	  echo "error: $(VERIFY_LIB) was not produced by the build."; \
	  echo "       Override the name with VERIFY_LIB=<path> if your platform differs."; \
	  exit 1; }
	@$(PYTHON) -c 'import ctypes,sys; d=ctypes.CDLL("$(VERIFY_LIB)"); rc=d.str_core_run_tests(b"$(SUITE)"); print("str_core_run_tests(\"%s\") -> %s" % ("$(SUITE)", "PASS (0 failures)" if rc==0 else "%d failures" % rc)); sys.exit(0 if rc==0 else 1)'

test-cpp-ctest: build-cpp
	$(CTEST) --test-dir "$(BUILD_DIR)" --output-on-failure

# --- One simulation run ----------------------------------------------------
# `cd` is required: pistr_sim is a plain package under sim/python, not an
# installed distribution, so `python3 -m pistr_sim.cli` must see it on sys.path.
sim:
	@test -f "$(SIM_DIR)/pistr_sim/cli.py" || { \
	  echo "error: $(SIM_DIR)/pistr_sim/cli.py does not exist yet."; \
	  echo "       The CLI is part of the simulator work in progress; see"; \
	  echo "       docs/architecture.md section 8."; \
	  exit 1; }
	@mkdir -p "$(RESULTS)"
	cd "$(SIM_DIR)" && $(PYTHON) -m pistr_sim.cli run \
	  --scenario "$(SCENARIO)" \
	  --controller "$(CONTROLLER)" \
	  --fidelity "$(FIDELITY)" \
	  --out "$(RESULTS)" $(ARGS)

# --- Optional 3-D environment (MuJoCo) -------------------------------------
# This is the only target in this Makefile that touches the network. Everything
# else stays offline; `make test` keeps working on a checkout without MuJoCo
# because the env tests call pytest.importorskip("mujoco") first.
env-setup:
	$(PYTHON) -m pip install --quiet --upgrade --target "$(DEPS_3D)" -r "$(REPO_ROOT)/requirements-3d.txt"
	@echo "3-D deps unpacked into $(DEPS_3D)"
	@echo "run with: PYTHONPATH=$(SIM_DIR):$(DEPS_3D) $(PYTHON) -m pistr_sim.env.cli --help"

test-3d:
	@test -d "$(DEPS_3D)/mujoco" || { \
	  echo "error: MuJoCo is not in $(DEPS_3D). Run 'make env-setup' first."; \
	  exit 1; }
	PYTHONPATH="$(SIM_DIR):$(DEPS_3D)" $(PYTEST) "$(SIM_DIR)/tests" -k env

demo-3d:
	@test -d "$(DEPS_3D)/mujoco" || { \
	  echo "error: MuJoCo is not in $(DEPS_3D). Run 'make env-setup' first."; \
	  exit 1; }
	@mkdir -p "$(ENV_OUT)"
	PYTHONPATH="$(SIM_DIR):$(DEPS_3D)" $(PYTHON) -m pistr_sim.env.cli demo --out "$(ENV_OUT)" $(ARGS)

# --- Housekeeping ----------------------------------------------------------
# results/ is deliberately NOT deleted: it holds the output of long runs and of
# the experiment matrix. Remove it by hand (`rm -rf results`) if you mean it.
clean:
	rm -rf "$(BUILD_DIR)"
	find "$(REPO_ROOT)" -type d -name __pycache__ -prune -exec rm -rf {} +
	find "$(REPO_ROOT)" -type d -name .pytest_cache -prune -exec rm -rf {} +
	find "$(SIM_DIR)" -maxdepth 2 -type d -name '*.egg-info' -prune -exec rm -rf {} +
	@echo "cleaned: $(BUILD_DIR), __pycache__, .pytest_cache, egg-info (results/ kept on purpose)"
