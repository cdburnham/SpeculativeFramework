# OSDZU3 Baseline Framework

Contributors:
- Cameron Burnham, (student of Western New England University)
- Arnab Purkayastha (Assistant Professor of Electrical & Computer Engineering at Western New England University)
- Ray Rimar (Professor in the Practice, Electrical and Computer Engineering at Rice University)

Acknowledgements:
This framework was produced atop of prior research conducted by Sed Centino and Chris Prague, both graduates of Western New England University.

In addition, AI tools (GPT-5.4) sourced from OpenAI were leveraged to accelerate research, development and the documentation phases of this project.

This repository is organized as a baseline directory for speculative backpropagation experiments on the OSDZU3 platform. It provides three framework variants with the same CLI contract and a shared top-level `projects/` directory.

## Layout

- `Standard/`: non-pipelined baseline framework
- `OpenMP-Mac/`: macOS OpenMP-oriented framework copy
- `OpenMP-Windows/`: Windows OpenMP-oriented framework copy
- `projects/`: shared project directory used by every framework variant

Each framework variant builds the same executable name:

```bash
./build/speculative_framework
```

## Shared CLI Contract

Run these commands from inside one of the framework variant directories.

Train a network from a JSON config:

```bash
./build/speculative_framework train ../projects/driver_drowsiness/configs/speculative.json
```

Train with runtime overrides:

```bash
./build/speculative_framework train ../projects/driver_drowsiness/configs/speculative.json \
  --epochs 50 \
  --threshold 0.25 \
  --metrics ../projects/driver_drowsiness/results/speculative_metrics.csv \
  --checkpoint ../projects/driver_drowsiness/results/speculative_checkpoint.bin
```

Validate or inspect a config:

```bash
./build/speculative_framework validate ../projects/driver_drowsiness/configs/speculative.json
./build/speculative_framework describe ../projects/driver_drowsiness/configs/speculative.json
```

Run benchmarking mode with parallel training processes:

```bash
./build/speculative_framework benchmark ../projects/driver_drowsiness/results/benchmark \
  ../projects/driver_drowsiness/configs/speculative.json \
  --thresholds 0.10,0.20,0.25,0.35 \
  --repeat 3 \
  --epochs 15 \
  --max-parallel 3
```

Compile results from one or more CSV files or a benchmark directory:

```bash
./build/speculative_framework compile-results \
  ../projects/driver_drowsiness/results/speculative_metrics.csv \
  ../projects/driver_drowsiness/results/baseline_metrics.csv \
  --output ../projects/driver_drowsiness/results/summary.csv
```

Or compile an entire benchmark directory:

```bash
./build/speculative_framework compile-results \
  ../projects/driver_drowsiness/results/benchmark \
  --output ../projects/driver_drowsiness/results/benchmark_summary.csv
```

## Threshold Control

Speculative mode supports threshold customization in the range:

- `0.10` to `0.35`

Threshold can be set in JSON or overridden at runtime with:

```bash
--threshold <value>
```

The CLI override takes precedence over the config file for that run.

## Per-Epoch Benchmark Data

Every training run exports per-epoch CSV data. The CSV includes:

- `epoch`
- `samples`
- `batches`
- `speculative_updates`
- `fallback_updates`
- `average_loss`
- `train_accuracy`
- `eval_accuracy`
- `epoch_ms`
- `average_sample_us`
- `threshold`
- `framework_variant`
- `config_path`
- `run_id`
- `benchmark_group`

Benchmark mode also writes:

- per-run metadata files
- per-run checkpoint files
- aggregate benchmark summary CSV

## Building Variants

Build the standard variant:

```bash
make standard
```

Build the macOS OpenMP variant:

```bash
make openmp-mac
```

If your macOS toolchain needs explicit OpenMP flags, use:

```bash
make openmp-mac OPENMP_CFLAGS="-Xpreprocessor -fopenmp" OPENMP_LDFLAGS="-lomp"
```

Build the Windows OpenMP variant structure:

```bash
make openmp-windows
```

## Project Example

The driver drowsiness project is shared across all variants at:

- [projects/driver_drowsiness]

Use the Python preprocessor from the repository root:

```bash
python3 projects/driver_drowsiness/scripts/preprocess_ddd.py \
  --input-root projects/driver_drowsiness/data \
  --output-root projects/driver_drowsiness/data \
  --image-size 24 \
  --train-ratio 0.7 \
  --label-map "drowsy=1" "non drowsy=0"
```

Then choose a framework variant and train from within that variant directory.
