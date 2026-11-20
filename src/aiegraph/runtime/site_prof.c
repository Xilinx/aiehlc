
/* P1 (b), build_unet.py --aie conv: A78 wall time of every call site in graph_run (XTime),
 * always on. AIE sites count only their A78 part (site time - aie_wall of their launches);
 * graph_bf16_expand time goes to the site that follows it. aiednn_prof_print() from main.c. */
#include "aie_timer.h"
int aiednn_aie_launch_stats(double *aie_ms, int *kloads, unsigned *cold_tiles);
void aiednn_aie_host_split_site(int site);                   /* GAP-RT1: kernel archive (aie_host_split.h) */
int aiednn_aie_host_split_line(int site, char *buf, int n);
#if defined(AIEHLC_SITE_PERF) && AIEHLC_SITE_PERF
/* M3.1 site-perf build (diagnostic only, not for L): the runtime arms the core perf counters only under
 * AIEHLC_PROFILING + debug flag 0x100 (AIE_DEBUG_FLAG_CORE_PERF_COUNTER); this strong definition overrides
 * aie_runtime.c's weak 0, so the board run needs no arguments. */
int g_runtime_debug_level = 0x100;
#endif
#define UP_MAX 1024
static unsigned long long up_ticks[UP_MAX], up_exp_ticks[UP_MAX], up_pending_exp;
static double up_aie_ms[UP_MAX];
static const char *up_tag[UP_MAX];
static unsigned up_n, up_exp_calls;
#undef GRAPH_PROF_BEGIN
#undef GRAPH_PROF_END
#define GRAPH_PROF_BEGIN(i) XTime graph_prof_t0_##i; up_exp_ticks[i] += up_pending_exp; up_pending_exp = 0; \
    XTime_GetTime(&graph_prof_t0_##i)
#define GRAPH_PROF_END(i, tag) do { XTime t1_; XTime_GetTime(&t1_);                      \
        up_ticks[i] += t1_ - graph_prof_t0_##i; up_tag[i] = (tag);                        \
        if ((unsigned)(i) + 1U > up_n) up_n = (unsigned)(i) + 1U; } while (0)
#define UP_AIE_BEGIN(i) double up_a0_##i = 0.0; aiednn_aie_host_split_site(i); aiednn_aie_launch_stats(&up_a0_##i, 0, 0)
#define UP_AIE_END(i) do { double a1_ = 0.0; aiednn_aie_launch_stats(&a1_, 0, 0); up_aie_ms[i] += a1_ - up_a0_##i; \
        aiednn_aie_host_split_site(-1); } while (0)
void aiednn_prof_print(double graph_run_ms, double aie_wall_ms) {
    const double tk = 1000.0 / (double)COUNTS_PER_SECOND;
    double tot = 0.0, cpu = 0.0, aieh = 0.0, exp_ms = 0.0, aie_in = 0.0;
    unsigned ns = 0;
    for (unsigned i = 0; i < up_n; i++) {
        if (!up_tag[i])
            continue;
        const int is_aie = strncmp(up_tag[i], "aie ", 4) == 0;
        const double site = (double)up_ticks[i] * tk, e = (double)up_exp_ticks[i] * tk;
        const double a78 = is_aie ? site - up_aie_ms[i] : site;
        const char *op = strncmp(up_tag[i], "cpu ", 4) == 0 ? up_tag[i] + 4 : up_tag[i];
        if (is_aie && strncmp(up_tag[i], "aie aiednn_aie_lin ", 19) == 0) /* PDN linear site: AIE host glue */
            printf("PROF part=%u op=aie_host_linear%s ms=%.3f bf16_expand_ms=%.3f site_ms=%.3f aie_ms=%.3f\n", i,
                   up_tag[i] + 19, a78 + e, e, site, up_aie_ms[i]);
        else if (is_aie)
            printf("PROF part=%u op=aie_host_conv%s ms=%.3f bf16_expand_ms=%.3f site_ms=%.3f aie_ms=%.3f\n", i,
                   up_tag[i] + 20, a78 + e, e, site, up_aie_ms[i]);
        else
            printf("PROF part=%u op=%s ms=%.3f bf16_expand_ms=%.3f\n", i, op, a78 + e, e);
        if (is_aie) { /* GAP-RT1: PROF aie_host_split part=<i> kind=<attn|conv|linear> pack_ms= scatter_ms= ... */
            char hs_[192];
            if (aiednn_aie_host_split_line((int)i, hs_, (int)sizeof hs_) > 0)
                printf("%s\n", hs_);
            /* M3.1: PROF aie_site_perf part=<i> ... (launches > 0, site-perf archives only; 0x10000 = AHS_PERF_LINE) */
            if (aiednn_aie_host_split_line((int)i + 0x10000, hs_, (int)sizeof hs_) > 0)
                printf("%s\n", hs_);
        }
        tot += a78 + e;
        exp_ms += e;
        aie_in += is_aie ? up_aie_ms[i] : 0.0;
        if (is_aie)
            aieh += a78;
        else
            cpu += a78;
        ns++;
    }
    const double tail = (double)up_pending_exp * tk;
    tot += tail;
    exp_ms += tail;
    const double den = graph_run_ms - aie_wall_ms;
    printf("PROF total_nonaie_ms=%.3f vs_graph_minus_aie=%.4f graph_run_ms=%.3f aie_wall_ms=%.3f sites=%u "
           "cpu_ms=%.3f aie_host_ms=%.3f bf16_expand_ms=%.3f (calls=%u tail=%.3f) aie_ms_in_sites=%.3f\n",
           tot, den > 0.0 ? tot / den : 0.0, graph_run_ms, aie_wall_ms, ns, cpu, aieh, exp_ms, up_exp_calls, tail,
           aie_in);
    {
        char hs_[192];
        if (aiednn_aie_host_split_line(-1, hs_, (int)sizeof hs_) > 0)
            printf("%s\n", hs_);
        else
            printf("PROF aie_host_split_total host_split_n=0\n");
    }
}
