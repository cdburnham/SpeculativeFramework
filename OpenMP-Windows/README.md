# OpenMP-Windows Variant

This is the Windows OpenMP-oriented framework copy. It preserves the same CLI contract as `Standard`.

Build:

```bash
make OPENMP_CFLAGS="<windows-openmp-flags>" OPENMP_LDFLAGS="<windows-openmp-linker-flags>"
```

Core CLI:

```bash
./build/speculative_framework train ../projects/driver_drowsiness/configs/speculative.json
./build/speculative_framework benchmark ../projects/driver_drowsiness/results/benchmark ../projects/driver_drowsiness/configs/speculative.json --thresholds 0.10,0.25,0.35 --repeat 3 --epochs 15 --max-parallel 3
./build/speculative_framework compile-results ../projects/driver_drowsiness/results/benchmark --output ../projects/driver_drowsiness/results/benchmark_summary.csv
```

Use [README.md](/Users/cameronburnham/Downloads/SpeculativeFramework/OSDZU3/README.md) at the repository root as the main guide.
