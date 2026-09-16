# F15-L07 build. Nothing here is hard-coded to one machine: the MuJoCo header directory and
# the shared library are discovered by asking the interpreter that is running the lesson where
# its own mujoco package lives. Pass PYTHON=... to use a different interpreter (the lesson's
# Python wrapper always passes its own sys.executable, so the binary is always linked against
# the same libmujoco the notebook imports).
#
#   make                                                    build lesson.cpp -> lesson_bin
#   make test                                               build, then run the self-test
#   make SRC=solutions/lesson.cpp BIN=solutions/lesson_bin test    the reference build
#   make paths                                              show what was discovered
#   make clean
#
# The binary names match the build products already listed in the repository's .gitignore.

PYTHON ?= python3
CXX    ?= clang++
SRC    ?= lesson.cpp
BIN    ?= lesson_bin
MODEL  ?= assets/cartpole.xml

CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter

UNAME_S := $(shell uname -s)

# Where the installed mujoco package unpacked its headers and its shared library.
MJ_DIR := $(shell $(PYTHON) -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")
MJ_LIB := $(shell $(PYTHON) -c "import glob, os, mujoco; d = os.path.dirname(mujoco.__file__); c = sorted(glob.glob(os.path.join(d, 'libmujoco*.dylib')) + glob.glob(os.path.join(d, 'libmujoco.so*'))); print(c[0] if c else '')")

ifeq ($(MJ_LIB),)
$(error could not find libmujoco next to $(MJ_DIR) — is mujoco installed for $(PYTHON)?)
endif

# On macOS the wheel's dylib announces itself as @rpath/mujoco.framework/... but ships no
# framework directory, so the freshly linked binary has to be pointed at the real file.
# -headerpad_max_install_names reserves room in the Mach-O header for that rewrite.
ifeq ($(UNAME_S),Darwin)
LDEXTRA := -Wl,-headerpad_max_install_names -Wl,-rpath,$(MJ_DIR)
else
LDEXTRA := -Wl,-rpath,$(MJ_DIR)
endif

.PHONY: all test clean paths

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(MJ_DIR)/include $(SRC) $(MJ_LIB) $(LDEXTRA) -o $@
ifeq ($(UNAME_S),Darwin)
	@iname=`otool -D $(MJ_LIB) | grep -v ':$$' | head -1`; \
	case "$$iname" in \
	  @*) install_name_tool -change "$$iname" $(MJ_LIB) $@ ;; \
	esac
endif

# The gate the student runs, and the one the Python wrapper shells out to. The binary's own
# self-test reports which of the three TODOs is still unfinished, and exits non-zero if any is.
test: $(BIN)
	./$(BIN) selftest --model $(MODEL)

paths:
	@echo "PYTHON = $(PYTHON)"
	@echo "MJ_DIR = $(MJ_DIR)"
	@echo "MJ_LIB = $(MJ_LIB)"
	@echo "SRC    = $(SRC)"
	@echo "BIN    = $(BIN)"

clean:
	rm -rf lesson_bin solutions/lesson_bin lesson_bin.dSYM solutions/lesson_bin.dSYM build
