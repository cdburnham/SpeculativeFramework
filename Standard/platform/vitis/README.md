# Vitis Integration

Use the files in `include/` and `src/` directly in a Vitis application project.

## Recommended Flow

1. Create the hardware platform in Vivado for the OSDZU3-REF board.
2. Enable the Zynq UltraScale+ PS UART used by your terminal connection.
3. Export the XSA to Vitis.
4. Create a standalone application project.
5. Import:
   - `src/main.c`
   - `src/osdzu3_board.c`
   - `src/osdzu3_cli.c`
   - `src/osdzu3_config.c`
   - `src/osdzu3_dataset.c`
   - `src/osdzu3_json.c`
   - `src/osdzu3_metrics.c`
   - `src/osdzu3_network.c`
   - all headers in `include/`
6. Add the preprocessor symbol `OSDZU3_PLATFORM_XILINX`.
7. Ensure stdin/stdout are mapped to the chosen PS UART in the BSP settings.
8. Place the JSON config and dataset where the application can access them:
   - SD card
   - RAM filesystem
   - embedded arrays if you later choose to compile-in a config

## UART Usage

With stdin/stdout mapped to UART, these commands work naturally from a serial console:

```text
speculative_framework shell /sd/examples/mnist_osdzu3.json
```

Inside the shell:

```text
describe
train
infer test 0
quit
```

## Filesystem Notes

The framework expects ordinary path strings. In Vitis, those paths should match however your BSP exposes storage:

- `/sd/...` for SD-backed filesystems
- local BSP file paths if you add a filesystem layer
- compile-time embedded datasets if you later replace file I/O
