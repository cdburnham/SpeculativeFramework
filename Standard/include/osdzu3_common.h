#ifndef OSDZU3_COMMON_H
#define OSDZU3_COMMON_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define OSDZU3_MAX_PATH 256
#define OSDZU3_MAX_LAYERS 8
#define OSDZU3_MAX_INPUTS 4096
#define OSDZU3_MAX_NEURONS 512
#define OSDZU3_MAX_CLASSES 256
#define OSDZU3_MAX_JSON_TOKENS 4096
#define OSDZU3_MAX_CMD 256
#define OSDZU3_MAX_RUN_ID 64
#define OSDZU3_MAX_VARIANT 32

#ifndef OSDZU3_FRAMEWORK_VARIANT
#define OSDZU3_FRAMEWORK_VARIANT "Standard"
#endif

#define OSDZU3_UNUSED(x) ((void)(x))

#endif
