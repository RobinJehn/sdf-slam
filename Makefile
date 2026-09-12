BUILD_DIR := build
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy

CXX_SOURCES := $(shell find src apps tests -name '*.cpp' -o -name '*.hpp')

.PHONY: configure build test format format-check tidy bench check check-fast clean

configure:
	cmake -B $(BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release

build: configure
	ninja -C $(BUILD_DIR)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

format:
	$(CLANG_FORMAT) -i $(CXX_SOURCES)

format-check:
	$(CLANG_FORMAT) --dry-run --Werror $(CXX_SOURCES)

# Homebrew clang-tidy on macOS needs the SDK sysroot passed explicitly. Its
# clang is newer than AppleClang and warns about the __COUNTER__ C2y extension
# inside the BENCHMARK macros.
ifeq ($(shell uname),Darwin)
TIDY_EXTRA_ARGS := --extra-arg=-isysroot --extra-arg=$(shell xcrun --show-sdk-path) \
	--extra-arg=-Wno-c2y-extensions
endif

tidy: configure
	$(CLANG_TIDY) -p $(BUILD_DIR) $(TIDY_EXTRA_ARGS) $(shell find src apps -name '*.cpp')

bench: build
	./$(BUILD_DIR)/bench_solvers

# Full local gate: what CI runs.
check: format-check build test tidy

# Pre-commit gate: skips clang-tidy for speed.
check-fast: format-check build test

clean:
	rm -rf $(BUILD_DIR)
