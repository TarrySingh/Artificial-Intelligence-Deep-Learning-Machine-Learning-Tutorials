# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Synapsa Commons - Copyright 2026 RealAI - free to learn from and share, not to sell; see NOTICE.
# T05-L04 build. Nothing here is tied to one machine or one checkout: the lesson directory is
# derived from this makefile's own path, and the compiler is discovered on PATH. Only the C++
# standard library is linked, so there is no third-party dependency to find and no absolute path
# into a virtualenv is baked into the binary. The stale-timestamp trap still exists, see `make clean`.
#
#   make                                                build lesson_bin from lesson.cpp
#   make test                                           build, then run the binary's self-test
#   make SRC=other.cpp BIN=other_bin test              build and test another file beside yours
#   make paths                                          show what was discovered
#   make clean
#
# The notebook always passes SRC and BIN explicitly, so two builds can sit side by side without ever
# overwriting each other's binary.

LESSON_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# `make` predefines CXX=g++; only override it when the user has not asked for a compiler.
ifeq ($(origin CXX), default)
CXX := $(shell command -v clang++ 2>/dev/null || command -v g++ 2>/dev/null || echo c++)
endif
PYTHON ?= $(shell command -v python3 2>/dev/null || echo python3)

SRC ?= lesson.cpp
BIN ?= lesson_bin

# -O2 for a realistic build. -ffp-contract=off and -fno-fast-math keep the compiler from fusing a multiply and
# an add or rewriting a float expression, so that every single-precision operation in lesson.cpp is the one
# written, on every chip: the notebook compares your output with a numpy reference bit for bit.
STD      ?= c++17
CXXFLAGS ?= -O2 -std=$(STD) -Wall -Wextra -Wno-unused-parameter -ffp-contract=off -fno-fast-math

.PHONY: all test paths clean

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

# The gate the notebook shells out to: the binary checks its own functions, one line per exercise.
test: $(BIN)
	./$(BIN) selftest

paths:
	@echo "LESSON_DIR = $(LESSON_DIR)"
	@echo "CXX        = $(CXX) ($(shell $(CXX) --version 2>/dev/null | head -1))"
	@echo "CXXFLAGS   = $(CXXFLAGS)"
	@echo "PYTHON     = $(PYTHON) ($(shell $(PYTHON) -c 'import sys; print(sys.version.split()[0])' 2>/dev/null))"
	@echo "SRC        = $(SRC)"
	@echo "BIN        = $(BIN)"

# Run this after moving or copying your checkout. make decides a binary is up to date by comparing
# timestamps against its source, and a binary that arrived with a copied directory is newer than the
# .cpp beside it, so make will refuse to rebuild it and you will be testing yesterday's answers.
# Deleting the artefact is the only reliable fix.
clean:
	rm -rf $(LESSON_DIR)/lesson_bin $(LESSON_DIR)/lesson_bin.dSYM $(LESSON_DIR)/solutions/lesson_bin $(LESSON_DIR)/solutions/lesson_bin.dSYM $(LESSON_DIR)/build
