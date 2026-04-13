#ifndef OSDZU3_BOARD_H
#define OSDZU3_BOARD_H

#include "osdzu3_common.h"

void osdzu3_board_printf(const char *fmt, ...);
void osdzu3_board_flush(void);
uint64_t osdzu3_board_time_us(void);

#endif
