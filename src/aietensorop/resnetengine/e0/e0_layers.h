/* E0 bring-up layer table, shared by e0 kernel and host (plain C, no includes).
 * Each "layer": every core receives R input blocks of IN bytes (broadcast per
 * mesh row), one weight block of W bytes (broadcast per mesh col) and returns R
 * blocks of IN bytes: out[j] = in[j] + w[j % W] + layer + mesh_col.  R is even
 * (finite ping-pong task: chain ping->pong, repeat R/2). */
#ifndef E0_LAYERS_H
#define E0_LAYERS_H
#define E0_NUM_LAYERS 3
#define E0_IN_MAX 4096
#define E0_W_MAX 4096
/* {IN bytes, W bytes, R rounds} -- sizes multiple of 64 */
#define E0_LAYER_TABLE                                                                                                 \
    {                                                                                                                  \
        {1024, 256, 4}, {4096, 2048, 2}, { 512, 4096, 6 }                                                              \
    }
#endif
