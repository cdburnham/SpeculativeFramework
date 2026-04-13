# Architecture Notes

The framework targets software execution on the Zynq UltraScale+ processing system first, because that is the fastest path to a working OSDZU3 board demo with UART control.

## Core Modules

- `osdzu3_config`: parses JSON network and training descriptions
- `osdzu3_dataset`: abstracts dataset access for host and board runs
- `osdzu3_network`: forward pass, baseline SGD, source-matched speculative scheduling, batching
- `osdzu3_metrics`: CSV epoch logging
- `osdzu3_board`: timer and console abstraction for host or Vitis
- `osdzu3_cli`: command-line and UART-shell command handling

## Speculative Backprop Strategy

The original experiments cached prior activations by label and reused them when the new output distribution stayed close to the cached one. This framework preserves that idea in a more general form:

- each class label owns a cached copy of activations and preactivations
- after each sample, the current forward-pass state is stored for that label
- on the next sample of that label, the mean absolute output delta is measured
- if the delta is below the configured threshold, gradients are computed from cached states
- otherwise the framework falls back to a standard backprop pass

## Board-Friendly Design Choices

- one-time memory allocation during initialization
- single-precision math instead of `double`
- no Windows headers
- no OpenMP dependency
- CLI compatible with stdin/stdout over UART

## Natural Extension Points

- replace dense layer loops with HLS or PL accelerators
- map metrics output to SD card or DDR-backed filesystem
- add DMA-fed dataset paths
- add UART commands for checkpoint save/load
