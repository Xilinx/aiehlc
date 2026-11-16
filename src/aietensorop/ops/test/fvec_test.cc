// x86 check of aie_modes_fvec.h (the FVEC mode) against double precision, through the AIE-API
// emulation: g++ -std=c++20 -DAIE_FVEC_EMU -I src/mlir/runtime -I src/aietensorop/ops fvec_test.cc
// Prints each program's worst |got - want| / (1e-5 + 1e-5 |want|) (contract: <= 1).
#include "aie_api_x86emu.h"
using int32 = int32_t;
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "aie_modes_fvec.h"

static double sig(double v) { return 1 / (1 + std::exp(-v)); }
static double softplus(double v) { return std::fmax(v, 0) + std::log1p(std::exp(-std::fabs(v))); }

static double ref(int f, double x, double y, double a, double b) {
    switch (f) {
    case AIE_FVF_LINEAR:
        return a * x + b;
    case AIE_FVF_RELU:
        return x > 0 ? x : a * x;
    case AIE_FVF_CLIP:
        return std::fmin(std::fmax(x, a), b);
    case AIE_FVF_ABS:
        return std::fabs(x);
    case AIE_FVF_SQUARE:
        return x * x;
    case AIE_FVF_SQRT:
        return std::sqrt(x);
    case AIE_FVF_EXP:
        return std::exp(x);
    case AIE_FVF_LOG:
        return std::log(x);
    case AIE_FVF_LOGISTIC:
        return sig(x);
    case AIE_FVF_TANH:
        return std::tanh(x);
    case AIE_FVF_GELU_TANH:
        return 0.5 * x * (1 + std::tanh(std::sqrt(2 / M_PI) * (x + 0.044715 * x * x * x)));
    case AIE_FVF_SWISH:
        return x * sig(a * x);
    case AIE_FVF_ELU:
        return x > 0 ? x : a * (std::exp(x) - 1);
    case AIE_FVF_SELU:
        return b * (x > 0 ? x : a * (std::exp(x) - 1));
    case AIE_FVF_SOFTPLUS:
        return softplus(a * x) / a;
    case AIE_FVF_MISH:
        return x * std::tanh(softplus(x));
    case AIE_FVF_HARDSIGMOID:
        return std::fmin(std::fmax(a * x + b, 0), 1);
    case AIE_FVF_HARDSWISH:
        return x * std::fmin(std::fmax(a * x + b, 0), 1);
    case AIE_FVF_GELU_ERF:
        return 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
    case AIE_FVF_ERF:
        return std::erf(x);
    case AIE_FVF_SIGN:
        return (x > 0) - (x < 0);
    case AIE_FVF_SOFTSIGN:
        return x / (1 + std::fabs(x));
    case AIE_FVF_LOG_SIGMOID:
        return -softplus(-x);
    case AIE_FVF_RECIPROCAL:
        return 1 / x;
    case AIE_FVF_RSQRT:
        return 1 / std::sqrt(x);
    case AIE_FVF_ADD:
        return x + y;
    case AIE_FVF_SUB:
        return x - y;
    case AIE_FVF_MUL:
        return x * y;
    case AIE_FVF_DIV:
        return x / y;
    case AIE_FVF_MAX:
        return std::fmax(x, y);
    case AIE_FVF_MIN:
        return std::fmin(x, y);
    case AIE_FVF_ROUND:
        return std::nearbyint(x);
    case AIE_FVF_FLOOR:
        return std::floor(x);
    case AIE_FVF_CEIL:
        return std::ceil(x);
    case AIE_FVF_LOGICAL_NOT:
        return x == 0;
    case AIE_FVF_POW:
        return a * std::pow(x, b);
    case AIE_FVF_EQ:
        return x == y;
    case AIE_FVF_NE:
        return x != y;
    case AIE_FVF_GT:
        return x > y;
    case AIE_FVF_GE:
        return x >= y;
    case AIE_FVF_LT:
        return x < y;
    case AIE_FVF_LE:
        return x <= y;
    case AIE_FVF_LAND:
        return x != 0 && y != 0;
    case AIE_FVF_LOR:
        return x != 0 || y != 0;
    case AIE_FVF_LXOR:
        return (x != 0) != (y != 0);
    case AIE_FVF_ADD_SQUARE:
        return x + y * y;
    }
    return 0;
}

/* f'(x) in double, as ref: (aiednn_rt_bwd.h eltwise_df) */
static double dref(int f, double x, double a, double b) {
    const double s = 1 / (1 + std::exp(-x));
    switch (f) {
    case AIE_FVF_RELU:
        return x > 0 ? 1 : a;
    case AIE_FVF_ELU:
        return x > 0 ? 1 : a * std::exp(x);
    case AIE_FVF_SELU:
        return b * (x > 0 ? 1 : a * std::exp(x));
    case AIE_FVF_LOGISTIC:
        return s * (1 - s);
    case AIE_FVF_TANH:
        return 1 - std::tanh(x) * std::tanh(x);
    case AIE_FVF_SWISH: {
        const double t = 1 / (1 + std::exp(-a * x));
        return t + a * x * t * (1 - t);
    }
    case AIE_FVF_SOFTPLUS:
        return 1 / (1 + std::exp(-a * x));
    case AIE_FVF_MISH: {
        const double t = std::tanh(x > 20 ? x : std::log1p(std::exp(x)));
        return t + x * (1 - t * t) * s;
    }
    case AIE_FVF_HARDSWISH: {
        const double t = a * x + b;
        return t <= 0 ? 0 : t >= 1 ? 1 : t + a * x;
    }
    case AIE_FVF_HARDSIGMOID: {
        const double t = a * x + b;
        return t <= 0 || t >= 1 ? 0 : a;
    }
    case AIE_FVF_CLIP:
        return x >= a && x <= b ? 1 : 0;
    case AIE_FVF_LINEAR:
        return a;
    case AIE_FVF_ABS:
        return x > 0 ? 1 : x < 0 ? -1 : 0;
    case AIE_FVF_EXP:
        return std::exp(x);
    case AIE_FVF_LOG:
        return 1 / x;
    case AIE_FVF_SQRT:
        return 0.5 / std::sqrt(x);
    case AIE_FVF_SQUARE:
        return 2 * x;
    case AIE_FVF_RECIPROCAL:
        return -1 / (x * x);
    case AIE_FVF_RSQRT:
        return -0.5 / (x * std::sqrt(x));
    case AIE_FVF_ERF:
        return 1.1283791670955126 * std::exp(-x * x);
    case AIE_FVF_SOFTSIGN:
        return 1 / ((1 + std::fabs(x)) * (1 + std::fabs(x)));
    case AIE_FVF_LOG_SIGMOID:
        return 1 - s;
    case AIE_FVF_GELU_TANH: {
        const double k = 0.79788456080286535588, t = std::tanh(k * (x + 0.044715 * x * x * x));
        return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * 0.044715 * x * x);
    }
    default:
        return NAN;
    }
}

static void run(int op, int fl, int n, int L, float a, float b, const float *in, const float *aux, float *out) {
    uint8_t D[16] = {};
    aie_fvec_entry(D, n, L);
    D[4] = (uint8_t)op, D[5] = (uint8_t)fl;
    std::memcpy(D + 6, &a, 4), std::memcpy(D + 10, &b, 4);
    aie_fvec_apply(in, out, D, aux);
}

int main() {
    std::mt19937 rng(11);
    alignas(64) static float in[AIE_FVEC_N], out[AIE_FVEC_N], aux_store[AIE_FVEC_AUX + 4];
    float *aux = aux_store + 4; /* 16-byte aligned only, as the gemm kernel's dwb */
    int bad = 0, maxins = 0;
    auto uni = [&](float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); };
    struct U {
        int f;
        float lo, hi, a, b;
    } us[] = {
        {AIE_FVF_LINEAR, -8, 8, 0.75f, -0.5f},
        {AIE_FVF_RELU, -8, 8, 0.1f, 0},
        {AIE_FVF_RELU, -8, 8, 0.f, 0},
        {AIE_FVF_CLIP, -8, 8, -1.5f, 2.f},
        {AIE_FVF_ABS, -8, 8, 0, 0},
        {AIE_FVF_SQUARE, -8, 8, 0, 0},
        {AIE_FVF_SQRT, 0, 100, 0, 0},
        {AIE_FVF_EXP, -80, 80, 0, 0},
        {AIE_FVF_LOG, 1e-6f, 1e6f, 0, 0},
        {AIE_FVF_LOGISTIC, -20, 20, 0, 0},
        {AIE_FVF_TANH, -10, 10, 0, 0},
        {AIE_FVF_GELU_TANH, -8, 8, 0, 0},
        {AIE_FVF_SWISH, -8, 8, 1.f, 0},
        {AIE_FVF_ELU, -8, 8, 1.f, 0},
        {AIE_FVF_SELU, -8, 8, 1.6732632f, 1.0507010f},
        {AIE_FVF_SOFTPLUS, -30, 30, 1.f, 0},
        {AIE_FVF_SOFTPLUS, -8, 8, 2.f, 0},
        {AIE_FVF_MISH, -8, 8, 0, 0},
        {AIE_FVF_HARDSIGMOID, -6, 6, 1.f / 6, 0.5f},
        {AIE_FVF_HARDSWISH, -6, 6, 1.f / 6, 0.5f},
        {AIE_FVF_GELU_ERF, -8, 8, 0, 0},
        {AIE_FVF_ERF, -4, 4, 0, 0},
        {AIE_FVF_SIGN, -4, 4, 0, 0},
        {AIE_FVF_SOFTSIGN, -8, 8, 0, 0},
        {AIE_FVF_LOG_SIGMOID, -30, 30, 0, 0},
        {AIE_FVF_RECIPROCAL, 0.01f, 100, 0, 0},
        {AIE_FVF_RSQRT, 1e-4f, 1e4f, 0, 0},
        {AIE_FVF_ADD, -8, 8, 0, 0},
        {AIE_FVF_SUB, -8, 8, 0, 0},
        {AIE_FVF_MUL, -8, 8, 0, 0},
        {AIE_FVF_DIV, -8, 8, 0, 0},
        {AIE_FVF_MAX, -8, 8, 0, 0},
        {AIE_FVF_MIN, -8, 8, 0, 0},
    };
    for (const U &u : us) {
        aie_fv_prog_t p;
        if (!aie_fvec_prog(u.f, u.a, u.b, &p)) {
            std::printf("no program for %d\n", u.f);
            bad++;
            continue;
        }
        maxins = p.nins > maxins ? p.nins : maxins;
        for (int i = 0; i < AIE_FVEC_N; i++) {
            const bool lg = u.f == AIE_FVF_LOG || u.f == AIE_FVF_RSQRT; /* log-uniform positives */
            in[i] = lg ? std::exp(uni(std::log(u.lo), std::log(u.hi))) : uni(u.lo, u.hi);
            aux[i] = uni(0.25f, 8) * (rng() % 2 ? 1 : -1);
        }
        in[0] = 0.0f;
        if (u.f == AIE_FVF_LOG || u.f == AIE_FVF_RSQRT || u.f == AIE_FVF_RECIPROCAL)
            in[0] = u.lo;
        aie_fvec_prog_store(&p, aux);
        run(AIE_FV_PROG, 0, AIE_FVEC_N, 0, 0, 0, in, aux, out);
        double worst = 0;
        for (int i = 0; i < AIE_FVEC_N; i++) {
            const double w = ref(u.f, in[i], aux[i], u.a, u.b);
            worst = std::fmax(worst, std::fabs(out[i] - w) / (1e-5 + 1e-5 * std::fabs(w)));
        }
        bad += worst > 1.0;
        std::printf("prog f %2d (%2d ins) contract %.3f%s\n", u.f, p.nins, worst, worst <= 1.0 ? "" : "  MISMATCH");
    }
    for (int op = AIE_FV_SOFTMAX; op <= AIE_FV_RMSNORM; op++)
        for (int L : {16, 64, 256, 1024})
            for (int fl : {0, 3}) {
                if ((fl && op < AIE_FV_LNORM) ||
                    (op >= AIE_FV_LNORM && 2 * L > AIE_FVEC_AUX)) /* norms: gamma + beta fit AUX */
                    continue;
                for (int i = 0; i < AIE_FVEC_N; i++)
                    in[i] = uni(-6, 6) + (i % L == 0 ? 3 : 0);
                for (int i = 0; i < 2 * L && i < AIE_FVEC_AUX; i++) /* fl 0: the host sends gamma 1 / beta 0 */
                    aux[i] = fl ? uni(-2, 2) : i < L ? 1.0f : 0.0f;
                const float eps = 1e-5f;
                run(op, op >= AIE_FV_LNORM ? 3 : fl, AIE_FVEC_N, L, eps, 1.0f / L, in, aux, out);
                double worst = 0;
                for (int r = 0; r < AIE_FVEC_N; r += L) {
                    double mx = -1e300, s = 0, mean = 0, q = 0;
                    for (int i = 0; i < L; i++)
                        mx = std::fmax(mx, in[r + i]), mean += in[r + i] / L;
                    for (int i = 0; i < L; i++)
                        s += std::exp(in[r + i] - mx), q += op == AIE_FV_LNORM
                                                                ? (in[r + i] - mean) * (in[r + i] - mean) / L
                                                                : (double)in[r + i] * in[r + i] / L;
                    for (int i = 0; i < L; i++) {
                        const double x = in[r + i];
                        double w = op == AIE_FV_SOFTMAX      ? std::exp(x - mx) / s
                                   : op == AIE_FV_LOGSOFTMAX ? x - mx - std::log(s)
                                   : op == AIE_FV_LNORM      ? (x - mean) / std::sqrt(q + eps)
                                                             : x / std::sqrt(q + eps);
                        if (fl)
                            w = w * aux[i] + aux[L + i];
                        worst = std::fmax(worst, std::fabs(out[r + i] - w) / (1e-5 + 1e-5 * std::fabs(w)));
                    }
                }
                bad += worst > 1.0;
                std::printf("row    op %2d L %4d fl %d contract %.3f%s\n", op, L, fl, worst,
                            worst <= 1.0 ? "" : "  MISMATCH");
            }
    for (int op : {(int)AIE_FV_RMAX, (int)AIE_FV_RSUM}) /* row reductions: max, sum, mean */
        for (int L : {16, 64, 1024})
            for (float bscale : {1.0f, 1.0f / L}) {
                if (op == AIE_FV_RMAX && bscale != 1.0f)
                    continue;
                for (int i = 0; i < AIE_FVEC_N; i++)
                    in[i] = uni(-6, 6);
                run(op, 0, AIE_FVEC_N, L, 0, bscale, in, aux, out);
                double worst = 0;
                for (int r = 0; r < AIE_FVEC_N / L; r++) {
                    double v = op == AIE_FV_RMAX ? -1e300 : 0;
                    for (int i = 0; i < L; i++)
                        v = op == AIE_FV_RMAX ? std::fmax(v, in[r * L + i]) : v + in[r * L + i];
                    if (op == AIE_FV_RSUM)
                        v *= bscale;
                    worst = std::fmax(worst, std::fabs(out[r] - v) / (1e-5 + 1e-5 * std::fabs(v)));
                }
                bad += worst > 1.0;
                std::printf("reduce op %2d L %4d scale %g contract %.3f%s\n", op, L, bscale, worst,
                            worst <= 1.0 ? "" : "  MISMATCH");
            }
    for (int f :
         {(int)AIE_FVF_ROW_AFFINE, (int)AIE_FVF_ROW_AFFINE_RELU, (int)AIE_FVF_ROW_PRELU}) /* per-row constants */
        for (int L : {16, 64, 784 / 49 * 16, 1024}) {
            aie_fv_prog_t p;
            aie_fvec_prog(f, 0, 0, &p);
            const int rows = AIE_FVEC_N / L;
            for (int i = 0; i < AIE_FVEC_N; i++)
                in[i] = uni(-6, 6);
            for (int r = 0; r < rows; r++)
                aux[2 * r] = uni(-2, 2), aux[2 * r + 1] = uni(-1, 1);
            aie_fvec_prog_store(&p, aux);
            run(AIE_FV_ROWPROG, 0, rows * L, L, 0, 0, in, aux, out);
            double worst = 0;
            for (int i = 0; i < rows * L; i++) {
                const double x = in[i], c2 = aux[2 * (i / L)], c3 = aux[2 * (i / L) + 1];
                const double w = f == AIE_FVF_ROW_PRELU    ? (x > 0 ? x : c2 * x)
                                 : f == AIE_FVF_ROW_AFFINE ? x * c2 + c3
                                                           : std::fmax(x * c2 + c3, 0);
                worst = std::fmax(worst, std::fabs(out[i] - w) / (1e-5 + 1e-5 * std::fabs(w)));
            }
            bad += worst > 1.0;
            std::printf("rowprog f %2d L %4d contract %.3f%s\n", f, L, worst, worst <= 1.0 ? "" : "  MISMATCH");
        }
    /* training rows: ROWSUM (one sum per row) and ROWPROG with a per-element operand (AIE_FV_F_ELEM) */
    for (int L : {16, 48, 256, 1024}) {
        const int rows = AIE_FVEC_N / L;
        for (int i = 0; i < AIE_FVEC_N; i++)
            in[i] = uni(0.001f, 1), aux[i] = uni(-2, 2); /* y (a softmax output), dy */
        struct Case {
            int op, f, el;
        } cs[] = {{AIE_FV_ROWSUM, AIE_FVF_MUL, 1},
                  {AIE_FV_ROWSUM, AIE_FVF_PICK1, 1},
                  {AIE_FV_ROWSUM, AIE_FVF_ROW_SQDEV, 0},
                  {AIE_FV_ROWPROG, AIE_FVF_ROW_SMBWD, 1},
                  {AIE_FV_ROWPROG, AIE_FVF_ROW_LSMBWD, 1}};
        for (const Case &c : cs) {
            aie_fv_prog_t p;
            aie_fvec_prog(c.f, 0, 0, &p);
            aie_fvec_prog_store(&p, aux);
            float *cst = c.el ? aux + AIE_FVEC_ROWAT : aux;
            std::vector<float> keep(aux, aux + AIE_FVEC_N);
            if (!c.el) /* row constants at AUX[2r]: the operand area is not read */
                for (int r = 0; r < rows; r++)
                    cst[2 * r] = uni(-1, 1), cst[2 * r + 1] = 0;
            else
                for (int r = 0; r < rows; r++)
                    cst[2 * r] = uni(-1, 1), cst[2 * r + 1] = 0;
            run(c.op, c.el ? AIE_FV_F_ELEM : 0, rows * L, L, 0, 0, in, aux, out);
            double worst = 0;
            for (int r = 0; r < rows; r++) {
                double sum = 0, ssum = 0;
                for (int i = r * L; i < (r + 1) * L; i++) {
                    const double y = in[i], dy = c.el ? aux[i] : 0, c2 = cst[2 * r];
                    const double v = c.f == AIE_FVF_MUL         ? y * dy
                                     : c.f == AIE_FVF_PICK1     ? dy
                                     : c.f == AIE_FVF_ROW_SQDEV ? (y - c2) * (y - c2)
                                     : c.f == AIE_FVF_ROW_SMBWD ? y * (dy - c2)
                                                                : dy - std::exp(y) * c2;
                    sum += v, ssum += std::fabs(v);
                    if (c.op == AIE_FV_ROWPROG)
                        worst = std::fmax(worst, std::fabs(out[i] - v) / (1e-5 + 1e-5 * std::fabs(v)));
                }
                if (c.op == AIE_FV_ROWSUM) /* a sum: the contract on the sum of magnitudes (reassociation) */
                    worst = std::fmax(worst, std::fabs(out[r] - sum) / (1e-5 + 1e-5 * ssum));
            }
            bad += worst > 1.0;
            std::printf("train op %2d f %3d el %d L %4d contract %.3f%s\n", c.op, c.f, c.el, L, worst,
                        worst <= 1.0 ? "" : "  MISMATCH");
            (void)keep;
        }
    }
    { /* PReLU as an elementwise program: w = the AUX element */
        aie_fv_prog_t p;
        aie_fvec_prog(AIE_FVF_ROW_PRELU, 0, 0, &p);
        for (int i = 0; i < AIE_FVEC_N; i++)
            in[i] = uni(-6, 6), aux[i] = uni(-2, 2);
        aie_fvec_prog_store(&p, aux);
        run(AIE_FV_PROG, 0, AIE_FVEC_N, 0, 0, 0, in, aux, out);
        double worst = 0;
        for (int i = 0; i < AIE_FVEC_N; i++) {
            const double w = in[i] > 0 ? in[i] : (double)aux[i] * in[i];
            worst = std::fmax(worst, std::fabs(out[i] - w) / (1e-5 + 1e-5 * std::fabs(w)));
        }
        bad += worst > 1.0;
        std::printf("prelu elementwise contract %.3f%s\n", worst, worst <= 1.0 ? "" : "  MISMATCH");
    }
    { /* exact-valued programs (round / floor / ceil, compares, logical, pow): ties, equal operands, zeros */
        struct E {
            int f;
            float a, b;
        } es[] = {
            {AIE_FVF_ROUND, 0, 0},      {AIE_FVF_FLOOR, 0, 0},  {AIE_FVF_CEIL, 0, 0},   {AIE_FVF_LOGICAL_NOT, 0, 0},
            {AIE_FVF_POW, 1.5f, 0},     {AIE_FVF_POW, 1.5f, 1}, {AIE_FVF_POW, 1.5f, 2}, {AIE_FVF_POW, -0.5f, 3},
            {AIE_FVF_POW, 1.f, 4},      {AIE_FVF_POW, 2.f, -1}, {AIE_FVF_POW, 1.f, -2}, {AIE_FVF_EQ, 0, 0},
            {AIE_FVF_NE, 0, 0},         {AIE_FVF_GT, 0, 0},     {AIE_FVF_GE, 0, 0},     {AIE_FVF_LT, 0, 0},
            {AIE_FVF_LE, 0, 0},         {AIE_FVF_LAND, 0, 0},   {AIE_FVF_LOR, 0, 0},    {AIE_FVF_LXOR, 0, 0},
            {AIE_FVF_ADD_SQUARE, 0, 0},
        };
        for (const E &u : es) {
            aie_fv_prog_t p;
            if (!aie_fvec_prog(u.f, u.a, u.b, &p)) {
                std::printf("no program for %d\n", u.f);
                bad++;
                continue;
            }
            for (int i = 0; i < AIE_FVEC_N; i++) {
                const int k = (int)(rng() % 4);
                in[i] = k == 0   ? (float)((int)(rng() % 41) - 20) + 0.5f
                        : k == 1 ? 0.0f
                                 : uni(-20, 20);                                        /* ties, zeros */
                aux[i] = rng() % 3 == 0 ? in[i] : rng() % 3 == 0 ? 0.0f : uni(-20, 20); /* equal, zeros */
                if (u.f == AIE_FVF_POW && u.b < 0 && in[i] == 0.0f)
                    in[i] = 0.75f;
            }
            if (u.f == AIE_FVF_ROUND || u.f == AIE_FVF_FLOOR ||
                u.f == AIE_FVF_CEIL) /* beyond the round trick's exact range */
                in[1] = 3000000.5f, in[2] = -1e30f;
            aie_fvec_prog_store(&p, aux);
            run(AIE_FV_PROG, 0, AIE_FVEC_N, 0, 0, 0, in, aux, out);
            double worst = 0;
            for (int i = 0; i < AIE_FVEC_N; i++) {
                const double w = ref(u.f, in[i], aux[i], u.a, u.b);
                worst = std::fmax(worst, std::fabs(out[i] - w) / (1e-5 + 1e-5 * std::fabs(w)));
            }
            bad += worst > 1.0;
            std::printf("exact f %2d b %2g (%2d ins) contract %.3f%s\n", u.f, u.b, p.nins, worst,
                        worst <= 1.0 ? "" : "  MISMATCH");
        }
    }
    { /* eltwise backward programs: R0 = x, R1 = dy, out dy f'(x) (aie_fvec_prog_bwd) */
        struct B {
            int f;
            float lo, hi, a, b;
        } bs[] = {
            {AIE_FVF_RELU, -4, 4, 0.f, 0},
            {AIE_FVF_RELU, -4, 4, 0.1f, 0},
            {AIE_FVF_ELU, -4, 4, 1.f, 0},
            {AIE_FVF_SELU, -4, 4, 1.6732632f, 1.0507010f},
            {AIE_FVF_LOGISTIC, -12, 12, 0, 0},
            {AIE_FVF_TANH, -6, 6, 0, 0},
            {AIE_FVF_SWISH, -8, 8, 1.f, 0},
            {AIE_FVF_SOFTPLUS, -8, 8, 2.f, 0},
            {AIE_FVF_MISH, -8, 8, 0, 0},
            {AIE_FVF_HARDSWISH, -6, 6, 1.f / 6, 0.5f},
            {AIE_FVF_HARDSIGMOID, -6, 6, 1.f / 6, 0.5f},
            {AIE_FVF_CLIP, -4, 4, -1.5f, 2.f},
            {AIE_FVF_LINEAR, -4, 4, 0.75f, 0.5f},
            {AIE_FVF_ABS, -4, 4, 0, 0},
            {AIE_FVF_EXP, -20, 20, 0, 0},
            {AIE_FVF_LOG, 0.01f, 100, 0, 0},
            {AIE_FVF_SQRT, 0.01f, 100, 0, 0},
            {AIE_FVF_SQUARE, -4, 4, 0, 0},
            {AIE_FVF_RECIPROCAL, 0.1f, 10, 0, 0},
            {AIE_FVF_RSQRT, 0.01f, 100, 0, 0},
            {AIE_FVF_ERF, -3, 3, 0, 0},
            {AIE_FVF_SOFTSIGN, -8, 8, 0, 0},
            {AIE_FVF_LOG_SIGMOID, -12, 12, 0, 0},
            {AIE_FVF_GELU_TANH, -6, 6, 0, 0},
        };
        for (const B &u : bs) {
            aie_fv_prog_t p;
            if (!aie_fvec_prog_bwd(u.f, u.a, u.b, &p)) {
                std::printf("no backward program for %d\n", u.f);
                bad++;
                continue;
            }
            for (int i = 0; i < AIE_FVEC_N; i++)
                in[i] = uni(u.lo, u.hi), aux[i] = uni(-2, 2);
            in[0] = u.lo > 0 ? u.lo : 0.0f; /* the x = 0 branch */
            aie_fvec_prog_store(&p, aux);
            run(AIE_FV_PROG, 0, AIE_FVEC_N, 0, 0, 0, in, aux, out);
            double worst = 0;
            for (int i = 0; i < AIE_FVEC_N; i++) {
                const double w = aux[i] * dref(u.f, in[i], u.a, u.b);
                worst = std::fmax(worst, std::fabs(out[i] - w) / (1e-5 + 1e-5 * std::fabs(w)));
            }
            bad += worst > 1.0;
            std::printf("bwd f %2d (%2d ins) contract %.3f%s\n", u.f, p.nins, worst, worst <= 1.0 ? "" : "  MISMATCH");
        }
    }
    /* an unused item of a partial launch: every op with n = rowlen = 0 must return */
    for (int op : {(int)AIE_FV_PROG, (int)AIE_FV_SOFTMAX, (int)AIE_FV_LOGSOFTMAX, (int)AIE_FV_LNORM,
                   (int)AIE_FV_RMSNORM, (int)AIE_FV_RMAX, (int)AIE_FV_RSUM, (int)AIE_FV_ROWPROG, (int)AIE_FV_ROWSUM})
        run(op, 3, 0, 0, 1e-5f, 0.f, in, aux, out);
    std::printf("empty items return\n");
    std::printf("longest program %d instructions\n%d mismatching\n", maxins, bad);
    return bad != 0;
}
