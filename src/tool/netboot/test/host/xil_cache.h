/* Host stub for the x86 unit test (streamload_x86.sh): the cache calls are counted. */
#ifndef XIL_CACHE_H
#define XIL_CACHE_H
#include <stdint.h>
typedef intptr_t INTPTR;
void Xil_DCacheFlushRange(INTPTR adr, INTPTR len);
void Xil_DCacheFlush(void);
void Xil_ICacheInvalidate(void);
#endif
