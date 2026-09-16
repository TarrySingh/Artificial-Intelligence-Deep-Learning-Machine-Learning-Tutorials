# T03-L02 build. Nothing here is tied to one machine or one checkout: the lesson directory is
# derived from this makefile's own path, and the compiler and interpreter are discovered on
# PATH. Only the C++ standard library is linked — there is no third-party dependency to find.
#
#   make                                                build build/lesson from lesson.cpp
#   make test                                           build, then run the binary's selftest
#   make SRC=solutions/lesson.cpp BIN=build/lesson_ref test    build the reference instead
#   make bench MERGES=300                               train on the shipped corpus
#   make paths                                          show what was discovered
#   make clean
#
# The lesson's Python wrapper always passes SRC and BIN explicitly, so the student file and
# the reference build side by side without ever overwriting each other's binary.

LESSON_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# `make` predefines CXX=g++; only override it when the user has not asked for a compiler.
ifeq ($(origin CXX), default)
CXX := $(shell command -v clang++ 2>/dev/null || command -v g++ 2>/dev/null || echo c++)
endif
PYTHON ?= $(shell command -v python3 2>/dev/null || echo python3)

SRC    ?= lesson.cpp
BIN    ?= build/lesson
CORPUS ?= $(LESSON_DIR)/assets/corpus.txt
MERGES ?= 200

# -O2 because the whole lesson is a measurement: an unoptimised build would win the race
# against Python for reasons that have nothing to do with the language.
STD      ?= c++17
CXXFLAGS ?= -O2 -std=$(STD) -Wall -Wextra -Wno-unused-parameter

.PHONY: all test bench paths clean

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

# The gate the Python wrapper shells out to: the binary checks its own four functions.
test: $(BIN)
	./$(BIN) selftest

bench: $(BIN)
	./$(BIN) train --corpus $(CORPUS) --merges $(MERGES)

paths:
	@echo "LESSON_DIR = $(LESSON_DIR)"
	@echo "CXX        = $(CXX) ($(shell $(CXX) --version 2>/dev/null | head -1))"
	@echo "PYTHON     = $(PYTHON) ($(shell $(PYTHON) -c 'import sys; print(sys.version.split()[0])' 2>/dev/null))"
	@echo "SRC        = $(SRC)"
	@echo "BIN        = $(BIN)"
	@echo "CORPUS     = $(CORPUS) ($(shell wc -c < $(CORPUS) 2>/dev/null | tr -d ' ') bytes)"

clean:
	rm -rf $(LESSON_DIR)/build
