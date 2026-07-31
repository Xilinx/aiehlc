/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef AIE_TIMER_H
#define AIE_TIMER_H

#ifdef __AIESIM__
#include <stdint.h>
#include <time.h>
typedef uint64_t XTime;
#define COUNTS_PER_SECOND 1000000000ULL
static inline void XTime_GetTime(XTime *t) {
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    *t = (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;
}
#elif defined(AIE_GEN) && AIE_GEN == 5
#include "xiltimer.h"
#else
#include "xtime_l.h"
#endif

#if defined(__aarch64__) && !defined(__AIESIM__)
static inline unsigned long long __ps_cntvct(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static inline unsigned long long __ps_cntfrq(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}
#else
static inline unsigned long long __ps_cntvct(void) { return 0ULL; }
static inline unsigned long long __ps_cntfrq(void) { return 0ULL; }
#endif

#if defined(__aarch64__) && !defined(__AIESIM__)
static inline void __ps_pmccntr_enable(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, pmcr_el0" : "=r"(v));
    v |= (1ULL << 0) | (1ULL << 2); /* E: enable counters, C: reset cycle counter */
    v &= ~(1ULL << 3);
    __asm__ volatile("msr pmcr_el0, %0" : : "r"(v));
    __asm__ volatile("msr pmcntenset_el0, %0" : : "r"(1ULL << 31));
    __asm__ volatile("isb");
}
static inline unsigned long long __ps_pmccntr(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, pmccntr_el0" : "=r"(v));
    return v;
}
static inline unsigned long long __ps_pmcr(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, pmcr_el0" : "=r"(v));
    return v;
}
#else
static inline void __ps_pmccntr_enable(void) {}
static inline unsigned long long __ps_pmccntr(void) { return 0ULL; }
static inline unsigned long long __ps_pmcr(void) { return 0ULL; }
#endif

#endif /* AIE_TIMER_H */
