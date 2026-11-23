#ifndef AIE_CTRLPKT_ENCODE_H
#define AIE_CTRLPKT_ENCODE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RT_CPE_CHUNK_WORDS 4U
#define RT_CPE_SKIP_ADDR 0xFFFFFFFFU

#define RT_CPE_MAGIC 0x43504B31U
#define RT_CPE_VERSION 2U

typedef struct {
    uint32_t prog_mem_size;
    uint32_t prog_mem_host_off;
    uint32_t data_mem_addr;
    uint32_t data_mem_size;
    uint32_t core_ctrl_off;
    uint32_t core_rst_mask;
    uint8_t checkerboard;
    uint32_t dma_ch_ctrl_off;
    uint32_t dma_ch_stride;
    uint32_t dma_nch;
    uint32_t dma_ch_rst_mask;
} rt_cpe_dev;

uint32_t rt_cpe_reset_accesses(const rt_cpe_dev *d);

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t aie_gen;
    uint32_t fingerprint;
    uint32_t stream_id;
    uint32_t nresp;
    uint32_t body_words;
    uint32_t enc_tile;
} rt_cpe_blob_hdr;

uint32_t rt_cpe_pktize(uint32_t *out, uint32_t cap, uint32_t stream_id, uint32_t tile_addr, const uint32_t *data,
                       uint32_t nwords, int lastack, uint32_t ret_sid);

int rt_cpe_target(const rt_cpe_dev *d, int col, int row, uint32_t paddr, uint32_t *tile_addr);

/* .bss handling shared by the AOT (aiehlc_ctrlpkt) and JIT (aie_runtime.c) encoders. */
int rt_cpe_elf_range_is_port_bufs(const unsigned char *elf, uint32_t lo, uint32_t hi);
uint32_t rt_cpe_seg_load_len(const unsigned char *elf, const void *phdr); /* phdr: const Elf32_Phdr* */

uint32_t rt_cpe_count_pkts(const rt_cpe_dev *d, const unsigned char *elf, int col, int row);

int rt_cpe_encode_body(const rt_cpe_dev *d, const unsigned char *elf, int col, int row, uint32_t stream_id,
                       int want_reset, uint32_t *pkt, uint32_t cap, uint32_t *body_words_out, uint32_t *nacc_out);

#define RT_CPE_NONE 0xFFFFFFFFU
typedef struct {
    uint32_t pw, last, last_ctrl;
} rt_cpe_seg;

void rt_cpe_seg_reset(rt_cpe_seg *s);
int rt_cpe_seg_write(rt_cpe_seg *s, uint32_t *p, uint32_t cap, uint32_t stream_id, uint32_t off, const uint32_t *data,
                     uint32_t nwords);
uint32_t rt_cpe_seg_finish(rt_cpe_seg *s, uint32_t *p);

#define RT_CPE_WR_UNDECIDED 0U
#define RT_CPE_WR_BCAST 1U
#define RT_CPE_WR_MMIO 2U
typedef struct {
    uint8_t col, row;
    uint32_t off, val;
    uint8_t state;
} rt_cpe_wr;

/* An offset is broadcast-safe when every write to it carries the same value and
 * each of the @ntiles consumer tiles receives it exactly once. */
uint32_t rt_cpe_ret_split(rt_cpe_wr *wr, uint32_t n, uint32_t ntiles, uint32_t *bc_off, uint32_t *bc_val, uint32_t max);

uint32_t rt_cpe_fingerprint(const rt_cpe_dev *d, uint32_t stream_id);

#ifdef __cplusplus
}
#endif

#endif
