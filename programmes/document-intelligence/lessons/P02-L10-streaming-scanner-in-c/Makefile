# P02-L10-streaming-scanner-in-c build. Nothing here is tied to one machine or one checkout: the lesson directory is
# derived from this makefile's own path, and the compiler and interpreter are discovered on
# PATH. There is no third-party library to link — the scanner uses the C standard library and
# POSIX getrusage, both of which the system compiler already knows where to find. `make paths`
# prints everything that was discovered, including the header the RSS counter comes from, so a
# build that goes wrong tells you WHICH tool it went wrong in.
#
#   make                                    build build/lesson from lesson.c
#   make test                               build, then run the binary's selftest
#   make bench BYTES=33554432               scan a generated export, print throughput
#   make paths                              show what was discovered
#   make clean                              delete every compiled artefact
#   make SRC=<file> BIN=build/<name> test   build some other source to some other binary
#
# The lesson's Python wrapper always passes SRC and BIN explicitly, which is what lets the
# course build compile a second source beside yours without either binary overwriting the
# other. You never need the last form by hand.

LESSON_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# `make` predefines CC=cc; only override it when the user has not asked for a compiler.
ifeq ($(origin CC), default)
CC := $(shell command -v clang 2>/dev/null || command -v gcc 2>/dev/null || echo cc)
endif
PYTHON ?= $(shell command -v python3 2>/dev/null || echo python3)

SRC   ?= lesson.c
BIN   ?= build/lesson
BYTES ?= 33554432
CHUNK ?= 65536

# -O2 because the whole lesson is a measurement: an unoptimised build would win its race
# against Python for reasons that have nothing to do with the language.
STD    ?= c11
CFLAGS ?= -O2 -std=$(STD) -Wall -Wextra -Wno-unused-parameter

# Discovered, not assumed: where the header behind the peak-RSS counter actually lives on
# this machine. Empty output here is the first thing to look at if the build cannot find
# getrusage. Nothing is compiled against the result; it is diagnostic only.
INCLUDE_DIRS = $(shell echo | $(CC) -E -Wp,-v -x c - -o /dev/null 2>&1 | grep '^ /' | tr -d ' ')
RUSAGE_HEADER = $(firstword $(wildcard $(addsuffix /sys/resource.h,$(INCLUDE_DIRS))))

.PHONY: all test bench paths clean

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(SRC) -o $@

# The gate the Python wrapper shells out to: the binary checks its own four functions.
test: $(BIN)
	./$(BIN) selftest

bench: $(BIN)
	./$(BIN) bench --bytes $(BYTES) --chunk $(CHUNK)

paths:
	@echo "LESSON_DIR     = $(LESSON_DIR)"
	@echo "CC             = $(CC) ($(shell $(CC) --version 2>/dev/null | head -1))"
	@echo "PYTHON         = $(PYTHON) ($(shell $(PYTHON) -c 'import sys; print(sys.version.split()[0])' 2>/dev/null))"
	@echo "SRC            = $(SRC)"
	@echo "BIN            = $(BIN)"
	@echo "CFLAGS         = $(CFLAGS)"
	@echo "RUSAGE_HEADER  = $(RUSAGE_HEADER)"
	@echo "LIBRARIES      = none (C standard library and POSIX only)"

# Read section 11 of the notebook before you wonder why this target exists.
clean:
	rm -rf $(LESSON_DIR)/build
