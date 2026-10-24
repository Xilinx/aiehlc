/* eng_desc.h -- the persistent conv engine's per-layer contract, shared by the
 * AIE kernel, the host driver and gen_engine.py (which writes the bytes).
 * Plain C, no includes. doc/design/resnet_aie_engine.md has the full picture.
 *
 * Each mesh column's weight channel delivers, per layer:
 *   header  ENG_HDR_BYTES: int32 words indexed by EH_* below
 *   P*Q chunks, pass p outer, ic chunk q inner, each ENG_PRM_BYTES of params
 *   then B tiles [tap][kc 8][nb 2][k 8][n 8] (int8)
 *     params: main oc i<16 {bz, mult, shift} at i*12, down tap at 256 + i*12
 *     tap 0..8 = (kh, kw) of the 3x3; tap 9 = the fused 1x1 s2 downsample
 * Core (mesh row r, col c), pass p owns output channel blocks (4p + c)*2 + nb,
 * so pass p of all four columns is the contiguous 64-channel group p.
 *
 * Activations in DDR: L8 = [H + 2][C / 8][WST][8] int8 (value - 128),
 * border = -128. WST = roundup8(W + 2) of the tensor's own width.
 * Core-memory buffers (all addresses are core-view, set by gen_engine.py):
 *   RING [slots][8 blk][WST_in][8]           input rows of one ic chunk
 *   ACC  [row][mblk][nb][64] int32           partial sums when Q > 1
 *   ACCD [row][mblk][nb][64] int32           same, downsample tap
 *   OUT  [slot][p][nb][Wr][8]                output rows (packet = one slot)
 *   RES  [row][p][nb][Wr][8]                 residual for the next layer
 */
#ifndef ENG_DESC_H
#define ENG_DESC_H

#define ENG_MAGIC 0x31474E45 /* "ENG1" */
#define ENG_HDR_BYTES 256
#define ENG_PRM_BYTES 512
#define ENG_PRM_DOWN 256
#define ENG_CHUNK_BLK 8
#define ENG_TILE_BYTES 128 /* one (tap, kc): [nb 2][k 8][n 8] */

enum {
    EH_MAGIC = 0,
    EH_LAYER, /* TVM call index, for logs */
    EH_S,     /* stride 1 / 2 (K is always 3) */
    EH_Q,     /* ic chunks (Cin / 64) */
    EH_P,     /* passes (Cout / 64) */
    EH_WST,   /* input L8 row stride, px */
    EH_WO,    /* output width */
    EH_MB,    /* M-blocks of 8 output px */
    EH_FLAGS, /* ENG_F_* */
    EH_C0,    /* added to the requantized value before clamp(0, 255) */
    EH_C0D,   /* same, downsample tap */
    EH_SZP,   /* combine: Q(v - szp, sl, sM, ss) */
    EH_SL,
    EH_SM,
    EH_SS,
    EH_RZR, /* combine: Q(res - rzr, rl, rM, rs) */
    EH_RL,
    EH_RM,
    EH_RS,
    EH_NTAP,  /* 9, or 10 with the downsample tap */
    EH_CHUNK, /* bytes per chunk incl. params */
    EH_RING_SLOTS,
    EH_A_RING,
    EH_A_ACC,
    EH_A_ACCD,
    EH_A_OUT,
    EH_A_RES,
    EH_OUT_SLOTS,
    EH_A_EPD,  /* downsample epilogue constants (EpSet), ENG_F_DOWN layers */
    EH_A_SROW, /* STEM: 3 stem-row buffers [nb 2][112][8] for the fused maxpool */
    EH_ROWS,   /* [4]: output rows of mesh row r */
    EH_NWORDS = EH_ROWS + 4
};

#define ENG_F_DOWN 1   /* fused 1x1 s2 downsample tap -> RES */
#define ENG_F_GRAB 2   /* keep own slice of the input's centre rows -> RES */
#define ENG_F_COMB 4   /* out = clamp(Qself(v) + Qres(res), 0, 255) */
#define ENG_F_RES_IN 8 /* all input rows of all chunks resident: streamed once, not P times */
#define ENG_F_STEM 16  /* the stem: 7x7 s2 over [230][232][4] + fused 3x3 s2 maxpool, 56-px L8 out */
/* Layer-major (batched) schedule: a block's conv1 and conv2 do not run back to
 * back for one image, so the residual cannot stay in core memory. */
#define ENG_F_RESX                                                                                                     \
    32 /* COMB: before the conv, the input channel first delivers the residual                                         \
        * tensor's rows (own output rows, P chunks); the core grabs its slice -> RES */
#define ENG_F_DOWNOUT                                                                                                  \
    64 /* DOWN: v_d leaves as output passes P..2P-1 (2C-channel output tensor),                                        \
        * each OUT row slot = [main P passes | down P passes] */
#define ENG_F_GO                                                                                                       \
    128 /* hold the image's first output row until the host sets lock ENG_LK_GO: its                                   \
         * out-of-order shim BDs are re-armed per image, and an idle OOO channel                                       \
         * would land early packets at the previous image's addresses */
#define ENG_LK_GO 8
#define ENG_F_RI                                                                                                       \
    256 /* row=image: this mesh row's own image, all output rows; q-outer -- input                                     \
         * chunk q's rows stay while all P passes use them, ACC holds P x rows */

/* Stem geometry (ENG_F_STEM). Input rows are 232 px x 4 B = 928 B, landing in
 * 960 B (64-aligned) ring slots. Mesh row r computes stem rows 28r-1 .. 28r+27
 * (one halo row, recomputed) and emits maxpool rows 14r .. 14r+13. */
#define ENG_STEM_IN_ROW 928
#define ENG_STEM_SLOT 960
#define ENG_STEM_W 112
#define ENG_STEM_TAPS 28 /* kh 7 x kw-pair 4; k = (kw & 1) * 4 + channel */

#endif
