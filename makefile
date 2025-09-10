# Makefile for building ai_runtime and packaging with fpm

NAME := omniengine
VERSION := $(shell cargo metadata --no-deps --format-version 1 | jq -r '.packages[0].version')
ARCH := $(shell uname -m)
BUILD_DIR := target/release
BIN := $(BUILD_DIR)/omniengine

PREFIX := /usr/local
DESTDIR := ./pkg

# ---- Python & Linker ----
PYVER ?= 3.10
PYLIBDIR ?= /usr/lib/x86_64-linux-gnu

# Export as environment so cargo sees them
PYTHON ?= $(shell uv python find --resolve 3.10 2>/dev/null || uv python find 2>/dev/null)
export PYO3_PYTHON := $(PYTHON)
export RUSTFLAGS   := -L $(PYLIBDIR) -l python$(PYVER) -C link-args=-Wl,-rpath,$(PYLIBDIR)

# Optional (hilft z.B. beim lokalen Ausführen ohne Deb):
export LD_LIBRARY_PATH := $(PYLIBDIR):$(LD_LIBRARY_PATH)

all: build

# Build Rust project in release mode
build:
	cargo build --release

build-cli:
	cargo build --release --bin omniengine-cli

python-wheel:
	uv run maturin build --release

deb: build-cli
	fpm -s dir -t deb -n omniengine -v $(VERSION) \
		--prefix /usr/local/bin \
		target/release/omniengine-cli=usr/local/bin/omniengine

rpm: build-cli
	fpm -s dir -t rpm -n omniengine -v $(VERSION) \
		--prefix /usr/local/bin \
		target/release/omniengine-cli=usr/local/bin/omniengine

clean:
	cargo clean
	rm -rf $(DESTDIR) *.deb *.rpm

.PHONY: all build build-cli python-wheel deb rpm clean
