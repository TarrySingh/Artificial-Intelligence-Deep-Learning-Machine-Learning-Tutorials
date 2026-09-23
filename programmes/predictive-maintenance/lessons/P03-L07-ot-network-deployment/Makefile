# P03-L07 build.
#
#   make                                                        build lesson.c -> lesson_bin
#   make test                                                   build, then run the self-test
#   make SRC=solutions/lesson.c BIN=solutions/lesson_bin test    the reference build
#   make paths                                                  show what was discovered
#   make clean                                                  delete every build product
#
# Nothing here is hard-coded to one machine. The one external thing this lesson needs is the
# C math library, and where it lives differs by platform: on macOS libm is part of libSystem
# and -lm is a no-op, on Linux it is a separate library. The recipe below probes for it by
# compiling a two-line program rather than asserting which platform you are on.
#
# Worth noticing, because it is the difference between this compiled lesson and the ones that
# link against the virtualenv: this binary links NOTHING out of .venv, so no absolute path
# into the virtualenv is baked into it and the dyld half of the trap in QUALITY.md cannot
# bite here. The other half still can — see `make clean` below.

PYTHON ?= python3
CC     ?= clang

SRC ?= lesson.c
BIN ?= lesson_bin

# -std=c11 for a fixed language level. -Wall -Wextra because an off-by-one in a ring buffer
# is exactly what a warning catches for free.
#
# -ffp-contract=off and -fno-fast-math are here for the same reason they are in P04-L06: the
# gateway's arithmetic is integer, but this notebook compares it against a float64 reference,
# and a compiler that is free to fuse a multiply and an add, or to reassociate, would make
# that reference a moving target. The gap this lesson measures has to be a property of the
# fixed-point design, not of the optimiser.
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -ffp-contract=off -fno-fast-math -Wno-unused-parameter

# Probe for libm rather than guessing: link a trivial program with -lm and keep the flag only
# if this toolchain accepts it. On Linux -lm is required; on macOS libm is inside libSystem
# and the flag resolves to a stub; on a toolchain that has neither, the probe fails and the
# flag is dropped rather than breaking the build. No platform name appears anywhere below.
# The probe writes its throwaway binary to a fresh mktemp path and deletes it afterwards: a
# fixed name under /tmp belongs to whoever created it first, so on a shared machine the
# second student's probe cannot write it and the flag is silently dropped.
LIBM := $(shell t=$$(mktemp) && printf 'int main(void){return 0;}' | \
          $(CC) -xc - -lm -o $$t >/dev/null 2>&1 && echo -lm; rm -f $$t)

.PHONY: all test clean paths

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(SRC) $(LIBM) -o $@

# The gate the student runs, and the one the notebook shells out to. The binary's own
# self-test reports each exercise separately and exits 2 while any of them is still a stub.
test: $(BIN)
	./$(BIN) selftest

paths:
	@echo "PYTHON = $(PYTHON)"
	@echo "CC     = $(CC)"
	@echo "CFLAGS = $(CFLAGS)"
	@echo "LIBM   = $(LIBM)   (empty means libm is part of the system C library here)"
	@echo "SRC    = $(SRC)"
	@echo "BIN    = $(BIN)"
	@$(PYTHON) -c "import sys; print('PY VER =', sys.version.split()[0])"

# Run this after moving or copying your checkout. make decides a binary is up to date by
# comparing timestamps against its source, and a binary that arrived with a copied directory
# is newer than the .c file next to it — so make will refuse to rebuild it and you will be
# testing yesterday's answers. Deleting the artefact is the only reliable fix.
clean:
	rm -rf lesson_bin solutions/lesson_bin lesson_bin.dSYM solutions/lesson_bin.dSYM build
