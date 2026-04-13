CC ?= cc
OPENMP_CFLAGS ?=
OPENMP_LDFLAGS ?=
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -pedantic -Iinclude -DOSDZU3_FRAMEWORK_VARIANT=\"OpenMP-Windows\" $(OPENMP_CFLAGS)
LDFLAGS ?= -lm $(OPENMP_LDFLAGS)

SRC := \
	src/main.c \
	src/osdzu3_board.c \
	src/osdzu3_cli.c \
	src/osdzu3_config.c \
	src/osdzu3_dataset.c \
	src/osdzu3_json.c \
	src/osdzu3_metrics.c \
	src/osdzu3_network.c

OBJ := $(SRC:src/%.c=build/%.o)
BIN := build/speculative_framework
DDD_PACK := build/ddd_pack_pgm_dataset
DDD_COMPARE := build/ddd_compare_runs

.PHONY: all clean

all: $(BIN) $(DDD_PACK) $(DDD_COMPARE)

build:
	mkdir -p build

$(BIN): build $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

$(DDD_PACK): build ../projects/driver_drowsiness/tools/pack_pgm_dataset.c
	$(CC) $(CFLAGS) ../projects/driver_drowsiness/tools/pack_pgm_dataset.c -o $@

$(DDD_COMPARE): build ../projects/driver_drowsiness/tools/compare_runs.c
	$(CC) $(CFLAGS) ../projects/driver_drowsiness/tools/compare_runs.c -o $@

build/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf build
