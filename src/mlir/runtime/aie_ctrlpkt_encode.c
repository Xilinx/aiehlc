#include "aie_ctrlpkt_encode.h"

#include <elf.h>
#include <string.h>

static inline uint32_t cpe_parity(uint32_t v) { return (uint32_t)!__builtin_parity(v); }

static uint32_t cpe_stream_hdr(uint32_t stream_id) {
    uint32_t h = stream_id & 0x1FU;
    return h | (cpe_parity(h) << 31U);
}

static uint32_t cpe_pktize(uint32_t *out, uint32_t cap, uint32_t stream_id, uint32_t tile_addr, const uint32_t *data,
                           uint32_t nwords, int lastack, uint32_t ret_sid) {
    uint32_t idx = 0U;
    uint32_t nwrite = (lastack && nwords > 0U) ? (nwords - 1U) : nwords;
    for (uint32_t i = 0U; i < nwrite; i += 4U) {
        uint32_t ps = (nwrite - i) > 4U ? 4U : (nwrite - i);
        if (idx + 2U + ps > cap)
            return 0U;
        uint32_t ch = ((ps - 1U) & 0x3U) << 20U;
        ch |= (tile_addr + i * (uint32_t)sizeof(uint32_t)) & 0xFFFFFU;
        ch |= cpe_parity(ch) << 31U;
        out[idx++] = cpe_stream_hdr(stream_id);
        out[idx++] = ch;
        for (uint32_t j = 0U; j < ps; j++)
            out[idx++] = data ? data[i + j] : 0U;
    }
    if (lastack && nwords > 0U) {
        if (idx + 3U > cap)
            return 0U;
        uint32_t la = tile_addr + (nwords - 1U) * (uint32_t)sizeof(uint32_t);
        uint32_t ch = la & 0xFFFFFU;
        ch |= (0x2U & 0x3U) << 22U;
        ch |= (ret_sid & 0x1FU) << 24U;
        ch |= cpe_parity(ch) << 31U;
        out[idx++] = cpe_stream_hdr(stream_id);
        out[idx++] = ch;
        out[idx++] = data ? data[nwords - 1U] : 0U;
    }
    return idx;
}

uint32_t rt_cpe_pktize(uint32_t *out, uint32_t cap, uint32_t stream_id, uint32_t tile_addr, const uint32_t *data,
                       uint32_t nwords, int lastack, uint32_t ret_sid) {
    return cpe_pktize(out, cap, stream_id, tile_addr, data, nwords, lastack, ret_sid);
}

int rt_cpe_target(const rt_cpe_dev *d, int col, int row, uint32_t paddr, uint32_t *tile_addr) {
    if (paddr < d->prog_mem_size) {
        *tile_addr = d->prog_mem_host_off + paddr;
        return 0;
    }
    if (paddr < d->data_mem_addr || paddr >= d->data_mem_addr + d->data_mem_size * 4U)
        return -1;
    uint8_t dir = (uint8_t)(paddr / d->data_mem_size);
    uint8_t parity = d->checkerboard ? (uint8_t)(row & 1) : 1U;
    int tcol = col, trow = row;
    if (dir == 4U)
        trow--;
    else if (dir == 5U && parity)
        tcol--;
    else if (dir == 6U)
        trow++;
    else if (dir == 7U && !parity)
        tcol++;
    else if (dir < 4U || dir > 7U)
        return -1;
    if (tcol != col || trow != row) {
        *tile_addr = RT_CPE_SKIP_ADDR;
        return 0;
    }
    *tile_addr = paddr & (d->data_mem_size - 1U);
    return 0;
}

/* True if [lo, hi) is fully covered by the kernel's DMA port buffers
 * (buf_in_ping_N / buf_in_pong_N / buf_out_ping_N / buf_out_pong_N symbols in
 * the ELF .symtab). Port buffers are always written before they are read, so
 * they need no zero-fill. No symtab, or any byte that is not a port buffer -> 0. */
int rt_cpe_elf_range_is_port_bufs(const unsigned char *elf, uint32_t lo, uint32_t hi) {
    const Elf32_Ehdr *ehdr = (const Elf32_Ehdr *)elf;
    if (ehdr->e_shoff == 0U || ehdr->e_shnum == 0U || ehdr->e_shentsize != sizeof(Elf32_Shdr))
        return 0;
    const Elf32_Shdr *sh = (const Elf32_Shdr *)(elf + ehdr->e_shoff);
    const Elf32_Shdr *symsh = NULL;
    for (uint32_t i = 0U; i < ehdr->e_shnum; i++) {
        if (sh[i].sh_type == (uint32_t)SHT_SYMTAB) {
            symsh = &sh[i];
            break;
        }
    }
    if (!symsh || symsh->sh_entsize != sizeof(Elf32_Sym) || symsh->sh_link >= ehdr->e_shnum)
        return 0;
    const Elf32_Sym *sym = (const Elf32_Sym *)(elf + symsh->sh_offset);
    const char *strtab = (const char *)(elf + sh[symsh->sh_link].sh_offset);
    const uint32_t nsym = symsh->sh_size / (uint32_t)sizeof(Elf32_Sym);
    uint32_t cur = lo;
    while (cur < hi) {
        uint32_t best_end = cur;
        for (uint32_t k = 0U; k < nsym; k++) {
            const uint32_t a = sym[k].st_value, e = sym[k].st_value + sym[k].st_size;
            if (sym[k].st_size == 0U || a > cur || e <= best_end)
                continue;
            const char *n = strtab + sym[k].st_name;
            if (strncmp(n, "buf_in_ping_", 12) != 0 && strncmp(n, "buf_in_pong_", 12) != 0 &&
                strncmp(n, "buf_out_ping_", 13) != 0 && strncmp(n, "buf_out_pong_", 13) != 0)
                continue;
            best_end = e;
        }
        if (best_end == cur)
            return 0;
        cur = best_end;
    }
    return 1;
}

/* Bytes of a PT_LOAD segment the control-packet loader must write: the file
 * bytes plus the zero-initialised tail [filesz, memsz) (.bss), unless that tail
 * is made only of DMA port buffers. XAie_LoadElfMem zero-fills the tail; the
 * control-packet encoders used to drop it, leaving stale DM in kernel statics. */
uint32_t rt_cpe_seg_load_len(const unsigned char *elf, const void *phdr_v) {
    const Elf32_Phdr *phdr = (const Elf32_Phdr *)phdr_v;
    if (phdr->p_type != (uint32_t)PT_LOAD)
        return 0U;
    uint32_t filesz = phdr->p_filesz, memsz = phdr->p_memsz;
    if (memsz <= filesz)
        return filesz;
    if (rt_cpe_elf_range_is_port_bufs(elf, phdr->p_vaddr + filesz, phdr->p_vaddr + memsz))
        return filesz;
    return memsz;
}

uint32_t rt_cpe_count_pkts(const rt_cpe_dev *d, const unsigned char *elf, int col, int row) {
    const Elf32_Ehdr *ehdr = (const Elf32_Ehdr *)elf;
    uint32_t npkt = 0U;
    for (uint32_t i = 0U; i < ehdr->e_phnum; i++) {
        const Elf32_Phdr *phdr = (const Elf32_Phdr *)(elf + ehdr->e_phoff + (uint64_t)i * ehdr->e_phentsize);
        const uint32_t len = rt_cpe_seg_load_len(elf, phdr);
        if (len == 0U)
            continue;
        uint32_t tile_addr = 0U;
        if (rt_cpe_target(d, col, row, phdr->p_paddr, &tile_addr) != 0 || tile_addr == RT_CPE_SKIP_ADDR)
            continue;
        npkt += (len + (uint32_t)sizeof(uint32_t) * RT_CPE_CHUNK_WORDS - 1U) /
                ((uint32_t)sizeof(uint32_t) * RT_CPE_CHUNK_WORDS);
    }
    return npkt;
}

/* Encodes @len bytes at @tile_addr: the first @filesz come from @src, the rest
 * (the .bss tail) are zero. */
static int cpe_seg(uint32_t stream_id, const unsigned char *src, uint32_t filesz, uint32_t len, uint32_t tile_addr,
                   uint32_t *pkt, uint32_t cap, uint32_t *pw_io, uint32_t *nacc, uint32_t *last_addr,
                   uint32_t *last_val) {
    const uint32_t hdr = cpe_stream_hdr(stream_id);
    uint32_t pw = *pw_io, acc = *nacc, off = 0U, nwords = 0U;
    uint32_t words[RT_CPE_CHUNK_WORDS];
    for (; off < len; off += (uint32_t)sizeof(words)) {
        uint32_t chunk = len - off < (uint32_t)sizeof(words) ? len - off : (uint32_t)sizeof(words);
        nwords = (chunk + 3U) / 4U;
        if (pw + 2U + nwords > cap)
            return -1;
        memset(words, 0, sizeof(words));
        if (off < filesz)
            memcpy(words, src + off, filesz - off < chunk ? filesz - off : chunk);
        uint32_t ctrl = (((nwords - 1U) & 0x3U) << 20U) | ((tile_addr + off) & 0xFFFFFU);
        pkt[pw] = hdr;
        pkt[pw + 1U] = ctrl | (cpe_parity(ctrl) << 31U);
        memcpy(pkt + pw + 2U, words, nwords * sizeof(uint32_t));
        pw += 2U + nwords;
        acc++;
    }
    if (nwords) {
        uint32_t last_off = off - (uint32_t)sizeof(words);
        *last_addr = tile_addr + last_off + (nwords - 1U) * (uint32_t)sizeof(uint32_t);
        *last_val = words[nwords - 1U];
    }
    *pw_io = pw;
    *nacc = acc;
    return 0;
}

uint32_t rt_cpe_reset_accesses(const rt_cpe_dev *d) { return 2U + 4U * d->dma_nch; }

static int cpe_put1(uint32_t *pkt, uint32_t cap, uint32_t *pw, uint32_t *acc, uint32_t stream_id, uint32_t addr,
                    uint32_t val) {
    uint32_t g = cpe_pktize(pkt + *pw, cap - *pw, stream_id, addr, &val, 1U, 0, 0U);
    if (g == 0U)
        return -1;
    *pw += g;
    (*acc)++;
    return 0;
}

static int cpe_reset(const rt_cpe_dev *d, uint32_t stream_id, uint32_t *pkt, uint32_t cap, uint32_t *pw,
                     uint32_t *acc) {
    if (cpe_put1(pkt, cap, pw, acc, stream_id, d->core_ctrl_off, d->core_rst_mask) != 0)
        return -1;
    for (uint32_t v = 0U; v < 2U; v++)
        for (uint32_t c = 0U; c < 2U * d->dma_nch; c++)
            if (cpe_put1(pkt, cap, pw, acc, stream_id, d->dma_ch_ctrl_off + c * d->dma_ch_stride,
                         v ? 0U : d->dma_ch_rst_mask) != 0)
                return -1;
    return cpe_put1(pkt, cap, pw, acc, stream_id, d->core_ctrl_off, 0U);
}

int rt_cpe_encode_body(const rt_cpe_dev *d, const unsigned char *elf, int col, int row, uint32_t stream_id,
                       int want_reset, uint32_t *pkt, uint32_t cap, uint32_t *body_words_out, uint32_t *nacc_out) {
    uint32_t pw = 0U, acc = 0U, last_addr = 0U, last_val = 0U;
    if (want_reset && cpe_reset(d, stream_id, pkt, cap, &pw, &acc) != 0)
        return -1;
    const Elf32_Ehdr *ehdr = (const Elf32_Ehdr *)elf;
    for (uint32_t i = 0U; i < ehdr->e_phnum; i++) {
        const Elf32_Phdr *phdr = (const Elf32_Phdr *)(elf + ehdr->e_phoff + (uint64_t)i * ehdr->e_phentsize);
        const uint32_t len = rt_cpe_seg_load_len(elf, phdr);
        if (len == 0U)
            continue;
        uint32_t tile_addr = 0U;
        if (rt_cpe_target(d, col, row, phdr->p_paddr, &tile_addr) != 0)
            return -1;
        if (tile_addr == RT_CPE_SKIP_ADDR)
            continue;
        if (cpe_seg(stream_id, elf + phdr->p_offset, phdr->p_filesz, len, tile_addr, pkt, cap, &pw, &acc, &last_addr,
                    &last_val) != 0)
            return -1;
    }
    if (acc == 0U || pw == 0U)
        return -1;
    uint32_t g = cpe_pktize(pkt + pw, cap - pw, stream_id, last_addr, &last_val, 1U, 1, 0U);
    if (g == 0U)
        return -1;
    pw += g;
    *body_words_out = pw;
    *nacc_out = acc;
    return 0;
}

static uint32_t cpe_ctrl_word(uint32_t ctrl) { return ctrl | (cpe_parity(ctrl) << 31U); }

static uint32_t cpe_words_to_boundary(uint32_t byte_addr) {
    uint32_t left = (16U - (byte_addr & 0xFU)) / (uint32_t)sizeof(uint32_t);
    return left ? left : 1U;
}

void rt_cpe_seg_reset(rt_cpe_seg *s) {
    s->pw = 0U;
    s->last = RT_CPE_NONE;
}

int rt_cpe_seg_write(rt_cpe_seg *s, uint32_t *p, uint32_t cap, uint32_t stream_id, uint32_t off, const uint32_t *data,
                     uint32_t nwords) {
    uint32_t hdr = cpe_stream_hdr(stream_id);
    for (uint32_t i = 0U; i < nwords; i++, off += 4U) {
        if (s->pw + 5U > cap)
            return -1;
        if (s->last != RT_CPE_NONE) {
            uint32_t aoff = p[s->last] & 0xFFFFFU, nw = ((p[s->last] >> 20U) & 0x3U) + 1U;
            if (off == aoff + 4U * nw && nw < cpe_words_to_boundary(aoff)) {
                p[s->last] = cpe_ctrl_word(aoff | (nw << 20U));
                p[s->pw++] = data[i];
                continue;
            }
        }
        p[s->pw++] = hdr;
        s->last = s->pw;
        p[s->pw++] = cpe_ctrl_word(off & 0xFFFFFU);
        p[s->pw++] = data[i];
    }
    return 0;
}

uint32_t rt_cpe_seg_finish(rt_cpe_seg *s, uint32_t *p) {
    if (s->last == RT_CPE_NONE)
        return 0U;
    uint32_t aoff = p[s->last] & 0xFFFFFU, nw = ((p[s->last] >> 20U) & 0x3U) + 1U;
    if (nw > 1U) {
        uint32_t d = p[s->pw - 1U];
        p[s->last] = cpe_ctrl_word(aoff | ((nw - 2U) << 20U));
        p[s->pw - 1U] = p[s->last - 1U];
        s->last = s->pw;
        aoff = (aoff + 4U * (nw - 1U)) & 0xFFFFFU;
        p[s->pw++] = 0U;
        p[s->pw++] = d;
    }
    p[s->last] = cpe_ctrl_word(aoff | (0x2U << 22U));
    return s->pw;
}

#define CPE_RET_MAX_ROWS 16U
#define CPE_RET_MAX_COLS 64U

static int cpe_ret_uniform(const rt_cpe_wr *wr, uint32_t n, uint32_t i, uint32_t ntiles) {
    uint64_t seen[CPE_RET_MAX_ROWS + 1U] = {0};
    const rt_cpe_wr *a = &wr[i];
    uint32_t hits = 0U;
    for (uint32_t j = i; j < n; j++) {
        const rt_cpe_wr *b = &wr[j];
        if (b->off != a->off)
            continue;
        if (b->val != a->val || b->row > CPE_RET_MAX_ROWS || b->col >= CPE_RET_MAX_COLS ||
            (seen[b->row] >> b->col) & 1U)
            return 0;
        seen[b->row] |= 1ULL << b->col;
        hits++;
    }
    return hits == ntiles;
}

uint32_t rt_cpe_ret_split(rt_cpe_wr *wr, uint32_t n, uint32_t ntiles, uint32_t *bc_off, uint32_t *bc_val, uint32_t max) {
    uint32_t nb = 0U;
    for (uint32_t i = 0U; i < n; i++) {
        if (wr[i].state != RT_CPE_WR_UNDECIDED)
            continue;
        uint8_t kind = RT_CPE_WR_MMIO;
        if (nb < max && cpe_ret_uniform(wr, n, i, ntiles)) {
            bc_off[nb] = wr[i].off;
            bc_val[nb] = wr[i].val;
            nb++;
            kind = RT_CPE_WR_BCAST;
        }
        for (uint32_t j = i; j < n; j++)
            if (wr[j].off == wr[i].off)
                wr[j].state = kind;
    }
    return nb;
}

uint32_t rt_cpe_fingerprint(const rt_cpe_dev *d, uint32_t stream_id) {
    uint32_t h = 2166136261U;
    const uint32_t vals[] = {d->prog_mem_size,   d->prog_mem_host_off, d->data_mem_addr, d->data_mem_size,
                             d->core_ctrl_off,   d->core_rst_mask,     d->checkerboard,  d->dma_ch_ctrl_off,
                             d->dma_ch_stride,   d->dma_nch,           d->dma_ch_rst_mask, stream_id};
    for (uint32_t i = 0U; i < sizeof(vals) / sizeof(vals[0]); i++) {
        h ^= vals[i];
        h *= 16777619U;
    }
    return h;
}
