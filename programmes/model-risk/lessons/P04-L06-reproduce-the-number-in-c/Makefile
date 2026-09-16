# P04-L06 build.
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
# into the virtualenv is baked into it and the dyld trap in QUALITY.md cannot bite here. The
# other half of that trap still can — see `make clean` below.

PYTHON ?= python3
CC     ?= clang

SRC ?= lesson.c
BIN ?= lesson_bin

# -std=c11 for a fixed language level. -Wall -Wextra because an off-by-one in a summation
# loop is exactly what a warning catches for free.
#
# -ffp-contract=off is load-bearing, not decoration. Left on, the compiler may fuse a
# multiply and an add into a single FMA instruction that rounds ONCE instead of twice, and
# `a*b + c` then produces a different double on a chip with FMA than on one without. A lesson
# about reproducing a number to the last bit cannot ship with that switch left at the
# compiler's discretion. -fno-fast-math for the same reason: fast-math permits reassociating
# additions, which would let the compiler silently turn your naive_sum into something else.
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -ffp-contract=off -fno-fast-math -Wno-unused-parameter

# Probe for libm rather than guessing: link a trivial program with -lm and keep the flag only
# if this toolchain accepts it. On Linux -lm is required; on macOS libm is inside libSystem
# and the flag resolves to a stub; on a toolchain that has neither, the probe fails and the
# flag is dropped rather than breaking the build. No platform name appears anywhere below.
# The probe writes its throwaway binary to a fresh mktemp path and deletes it afterwards. A
# fixed name under /tmp belongs to whoever created it first, so on a shared machine the second
# student's probe cannot write it, the flag is silently dropped, and on Linux the link then
# fails for a reason that has nothing to do with their code.
LIBM := $(shell t=$$(mktemp) && printf 'int main(void){return 0;}' | \
          $(CC) -xc - -lm -o $$t >/dev/null 2>&1 && echo -lm; rm -f $$t)

.PHONY: all test clean paths

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(SRC) $(LIBM) -o $@

# The gate the student runs, and the one the Python wrapper shells out to. The binary's own
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
