/* E1: one generic 3x3 stride-1 conv layer (TVM call 04 of int8 ResNet-18:
 * 64 -> 64 channels, 56x56, pad 1), shared by kernel and host. Plain C.
 *
 * Activation layout L8 (doc/design/resnet_aie_engine.md):
 *   [H + 2][C / 8][WST][8]   WST = roundup8(W + 2) px, zero-point padded border
 * One input row = CIN * WST bytes; each (row, channel block) segment is
 * WST*8 bytes and 64-byte aligned, so A tiles come from aligned loads. */
#ifndef E1_LAYER_H
#define E1_LAYER_H
#define E1_CIN 64
#define E1_COUT 64
#define E1_COUT_CORE 16 /* COUT / mesh cols */
#define E1_H 56         /* output rows == input rows (s1, pad 1) */
#define E1_W 56
#define E1_WST 64       /* input/output L8 row stride in px */
#define E1_ROWS_CORE 14 /* output rows per mesh row */
#define E1_ROWS_IN_CORE (E1_ROWS_CORE + 2)
#define E1_IN_ROW_BYTES (E1_CIN * E1_WST)                        /* 4096 */
#define E1_OUT_ROW_BYTES (E1_COUT_CORE * E1_W)                   /* 896: [2][56][8] per core */
#define E1_PARAM_BYTES 256                                       /* 16 oc x {bz, mult, shift} int32, padded */
#define E1_BP_BYTES (9 * (E1_CIN / 8) * (E1_COUT_CORE / 8) * 64) /* 9216 */
#define E1_W_BYTES (E1_PARAM_BYTES + E1_BP_BYTES)                /* 9472 */
#define E1_RING 4
#endif
