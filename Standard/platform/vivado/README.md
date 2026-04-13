# Vivado Handoff Notes

The current framework runs on the Zynq UltraScale+ processing system and is intentionally arranged so dense-layer kernels can be accelerated later.

## Minimum Hardware Requirements

- Zynq UltraScale+ PS enabled
- DDR enabled for model weights, gradients, and dataset buffers
- one UART enabled for console control
- optional SD interface for JSON configs, logs, and datasets

## Suggested Next Hardware Steps

1. Keep training control, JSON parsing, and UART command handling on the PS.
2. Identify hot loops in `osdzu3_network.c`:
   - matrix-vector multiply
   - bias add
   - activation pass
   - gradient accumulation
3. Move those loops into HLS kernels or custom PL IP.
4. Replace the corresponding software routines with accelerator calls.

## Why This Split Fits OSDZU3

- command parsing and orchestration stay flexible on the PS
- UART and CLI behavior remain unchanged
- only the numerically intensive parts need hardware migration
- the JSON-driven topology still instantiates networks without rewriting the board interface
