# F15-L06 build. Nothing here is hard-coded to one machine: the MuJoCo header directory and
# the shared library are discovered by asking the interpreter that is running the lesson where
# its own mujoco package lives. Pass PYTHON=... to use a different interpreter (the lesson's
# Python wrapper always passes its own sys.executable, so the binary is always linked against
# the same libmujoco the notebook imports).
#
#   make                                                   build lesson.c -> lesson_bin
#   make test                                              build, then run the self-test
#   make SRC=solutions/lesson.c BIN=solutions/lesson_bin test   the reference build
#   make paths                                             show what was discovered
#   make clean
#
# The binary names match the build products already listed in the repository's .gitignore.

PYTHON ?= python3
CC     ?= clang

SRC ?= lesson.c
BIN ?= lesson_bin
MODEL ?= assets/arm2.xml

# -std=c11 because this lesson is C, not C++: the point is the API the engine actually
# exposes. -Wall -Wextra because a control loop that indexes the wrong array is exactly the
# bug a warning catches for free.
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter

UNAME_S := $(shell uname -s)

# Where the installed mujoco package unpacked its headers and its shared library. The wheel
# ships both: include/mujoco/*.h and libmujoco.<version>.dylib (or .so) sit side by side in
# the package directory, so one query finds both.
MJ_DIR := $(shell $(PYTHON) -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")
MJ_LIB := $(shell $(PYTHON) -c "import glob, os, mujoco; d = os.path.dirname(mujoco.__file__); c = sorted(glob.glob(os.path.join(d, 'libmujoco*.dylib')) + glob.glob(os.path.join(d, 'libmujoco.so*'))); print(c[0] if c else '')")

# `clean` and `paths` keep working even when the interpreter on PATH has no mujoco. `make
# clean` is the documented fix for a binary whose baked-in library path has gone stale, and
# refusing to run it because the library cannot be found would be exactly backwards.
ifeq ($(MJ_LIB),)
ifeq ($(filter clean paths,$(MAKECMDGOALS)),)
$(error could not find libmujoco next to $(MJ_DIR) — is mujoco installed for $(PYTHON)?)
endif
endif

# On macOS the wheel's dylib announces its own install name as
# @rpath/mujoco.framework/Versions/A/libmujoco.<version>.dylib, but the wheel ships no
# framework directory — so a freshly linked binary asks the loader for a path that does not
# exist and dies at startup with "Library not loaded". Two things fix it together:
#   -rpath $(MJ_DIR)                    tells the loader where to search, and
#   -headerpad_max_install_names        reserves room in the Mach-O header so the recorded
#                                       path can be rewritten after linking (below).
ifeq ($(UNAME_S),Darwin)
LDEXTRA := -Wl,-headerpad_max_install_names -Wl,-rpath,$(MJ_DIR)
else
LDEXTRA := -Wl,-rpath,$(MJ_DIR)
endif

.PHONY: all test clean paths

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(MJ_DIR)/include $(SRC) $(MJ_LIB) $(LDEXTRA) -o $@
ifeq ($(UNAME_S),Darwin)
	@iname=`otool -D $(MJ_LIB) | grep -v ':$$' | head -1`; \
	case "$$iname" in \
	  @*) install_name_tool -change "$$iname" $(MJ_LIB) $@ ;; \
	esac
endif

# The gate the student runs, and the one the Python wrapper shells out to. The binary's own
# self-test reports which of the three TODOs is still unfinished, and exits 2 if any is.
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
