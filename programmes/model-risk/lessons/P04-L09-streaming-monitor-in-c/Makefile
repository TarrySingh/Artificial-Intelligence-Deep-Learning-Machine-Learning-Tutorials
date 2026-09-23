# P04-L09 build.
#
#   make                                                        build lesson.c -> lesson_bin
#   make test                                                   build, then run the self-test
#   make SRC=solutions/lesson.c BIN=solutions/lesson_bin test    the reference build
#   make paths                                                  show what was discovered
#   make clean                                                  delete every build product
#
# Nothing here is tied to one machine. The compiler is discovered on PATH, and the one
# external thing the binary needs — the C maths library, for log() — lives in different places
# on different systems: on macOS it is part of libSystem and -lm is a no-op, on Linux it is a
# separate library. The recipe below probes for it by linking a two-line program rather than
# asserting which platform you are on.
#
# This binary links NOTHING out of the virtualenv, so no absolute path into .venv is baked into
# it and the dyld half of the compiled-lesson trap cannot bite here. The stale-timestamp half
# still can — see `make clean` below.

# `make` predefines CC=cc; only choose one when you have not.
ifeq ($(origin CC), default)
CC := $(shell command -v clang 2>/dev/null || command -v gcc 2>/dev/null || echo cc)
endif
PYTHON ?= python3

SRC ?= lesson.c
BIN ?= lesson_bin

# -std=c11 for a fixed language level; the source switches on the POSIX calls it needs
# (fork, wait4, getrusage) itself. -Wall -Wextra because an off-by-one in a loop over a buffer
# is exactly what a warning catches for free. -ffp-contract=off so the compiler may not fuse
# `psi += (a - e) * log(a / e)` into one fused multiply-add, which rounds once instead of
# twice and would make the running PSI depend on the chip. -fno-fast-math for the same reason.
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off -fno-fast-math

# Probe for libm: link a trivial program with -lm and keep the flag only if this toolchain
# accepts it. The throwaway binary goes to a fresh mktemp path and is deleted, because a fixed
# name under /tmp belongs to whoever created it first.
LIBM := $(shell t=$$(mktemp) && printf 'int main(void){return 0;}' | \
          $(CC) -xc - -lm -o $$t >/dev/null 2>&1 && echo -lm; rm -f $$t)

.PHONY: all test clean paths

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(SRC) $(LIBM) -o $@

# The gate you run, and the one the notebook shells out to. The binary's own self-test
# reports each exercise separately and exits 2 while any of them is still a stub.
test: $(BIN)
	./$(BIN) selftest

paths:
	@echo "CC     = $(CC) ($(shell $(CC) --version 2>/dev/null | head -1))"
	@echo "CFLAGS = $(CFLAGS)"
	@echo "LIBM   = $(LIBM)   (empty means libm is part of the system C library here)"
	@echo "SRC    = $(SRC)"
	@echo "BIN    = $(BIN)"
	@$(PYTHON) -c "import sys; print('PY VER =', sys.version.split()[0])"

# Run this after moving or copying your checkout. make decides a binary is up to date by
# comparing timestamps against its source, and a binary that arrived with a copied directory
# is newer than the .c file beside it — so make will refuse to rebuild it and you will be
# testing yesterday's answers. Deleting the artefact is the only reliable fix.
clean:
	rm -rf lesson_bin solutions/lesson_bin lesson_bin.dSYM solutions/lesson_bin.dSYM build
