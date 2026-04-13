#include "osdzu3_board.h"

#include <sys/time.h>
#include <stdarg.h>
#include <stdio.h>

#ifdef OSDZU3_PLATFORM_XILINX
#include "xtime_l.h"
#include "xil_printf.h"
#endif

void osdzu3_board_printf(const char *fmt, ...) {
    va_list args;
    char buffer[512];

    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

#ifdef OSDZU3_PLATFORM_XILINX
    xil_printf("%s", buffer);
#else
    fputs(buffer, stdout);
#endif
}

void osdzu3_board_flush(void) {
    fflush(stdout);
}

uint64_t osdzu3_board_time_us(void) {
#ifdef OSDZU3_PLATFORM_XILINX
    XTime now;
    XTime_GetTime(&now);
    return (uint64_t) ((now * 1000000ULL) / COUNTS_PER_SECOND);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((uint64_t) tv.tv_sec * 1000000ULL) + (uint64_t) tv.tv_usec;
#endif
}
