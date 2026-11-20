void aiednn_prof_print(double graph_run_ms, double aie_wall_ms); /* graph_driver.c (P1 PROF) */
int aiednn_aie_stats(int *launches, int *run_rc, int *calls, unsigned long long *macs);
int aiednn_aie_launch_stats(double *aie_ms, int *kloads, unsigned *cold_tiles);
/*KT+*/int aiednn_aie_ktimer_stats(unsigned long long *max_sum, unsigned long long *all_sum, int *n, int *bad);
/*KT-*/int aiednn_aie_split_stats(unsigned long long *pre, unsigned long long *call, unsigned long long *glue, unsigned long long *post);
/* the runtime is C++ (aie_runtime.o in the kernel archive): bind its mangled symbol */
void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls) __asm__("_Z22__Runtime_phase_cyclesPyPj");
void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls) __asm__("_Z24__Runtime_wait_io_cyclesPyPj");
void __Runtime_kload_split_cycles(unsigned long long *elf_cyc, unsigned int *elf_n, unsigned long long *rst_cyc, unsigned int *rst_n) __asm__("_Z28__Runtime_kload_split_cyclesPyPjS_S0_");
void __Runtime_setup_split_cycles(unsigned long long *plan_cyc, unsigned int *plan_n, unsigned long long *sync_cyc, unsigned int *sync_n) __asm__("_Z28__Runtime_setup_split_cyclesPyPjS_S0_");
void __Runtime_launch_words(unsigned long long *mmio_w, unsigned long long *cp_w, unsigned long long *ph_w, unsigned int *bd_n, unsigned int *rst_n, unsigned long long *other_w) __asm__("_Z22__Runtime_launch_wordsPyS_S_PjS0_S_");
unsigned long long __Runtime_ctrl_aot_fallbacks(void) __asm__("_Z28__Runtime_ctrl_aot_fallbacksv");
void __Runtime_ctrl_shim_bd_ctrl_stats_print(void) __asm__("_Z39__Runtime_ctrl_shim_bd_ctrl_stats_printv");
/* A78 PMU cycle counter (the runtime's RT_PROF_* clock): enabled here without the C reset, read
 * around graph_run with XTime so the PMU rate is measured, not assumed */
static inline unsigned long long @M@_pmc(void) {
    unsigned long long v;
    __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(v));
    return v;
}
static inline void @M@_pmc_enable(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, pmcr_el0" : "=r"(v));
    v |= 1ULL;
    v &= ~(1ULL << 3);
    __asm__ volatile("msr pmcr_el0, %0" : : "r"(v));
    __asm__ volatile("msr pmcntenset_el0, %0" : : "r"(1ULL << 31));
    __asm__ volatile("isb");
}
