/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "aie_ctrlpkt_devparams.h"
#include "aie_ctrlpkt_encode.h"
#include "aie_ctrlpkt_sites.h"
#include "ctrlpkt_dbg.h"

#include <elf.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CPE_ENC_COL 0
#define CPE_ENC_ROW 2
#define CPE_MAX_SEGS 64

typedef struct {
    int phdr_idx;
    uint32_t paddr, filesz, tile_addr;
} cpe_seg_info;

static unsigned char *read_file(const char *path, long *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(len > 0 ? (size_t)len : 1);
    if (buf && len > 0 && fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    if (buf)
        *len_out = len;
    return buf;
}

static long manifest_int(const char *json, const char *func, const char *key, long def) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"func\":\"%s\"", func);
    const char *p = strstr(json, needle);
    if (!p)
        return def;
    const char *k = strstr(p, key);
    if (!k)
        return def;
    k += strlen(key);
    while (*k && (*k == ':' || *k == ' ' || *k == '"'))
        k++;
    return strtol(k, NULL, 10);
}

static int collect_segs(const rt_cpe_dev *d, const unsigned char *elf, cpe_seg_info *segs) {
    const Elf32_Ehdr *eh = (const Elf32_Ehdr *)elf;
    int n = 0;
    for (uint32_t i = 0U; i < eh->e_phnum && n < CPE_MAX_SEGS; i++) {
        const Elf32_Phdr *ph = (const Elf32_Phdr *)(elf + eh->e_phoff + (uint64_t)i * eh->e_phentsize);
        uint32_t ta = 0U;
        const uint32_t len = rt_cpe_seg_load_len(elf, ph);
        if (len == 0U ||
            rt_cpe_target(d, CPE_ENC_COL, CPE_ENC_ROW, ph->p_paddr, &ta) != 0 || ta == RT_CPE_SKIP_ADDR)
            continue;
        segs[n].phdr_idx = (int)i;
        segs[n].paddr = ph->p_paddr;
        segs[n].filesz = len; /* file bytes + zero-filled .bss tail */
        segs[n].tile_addr = ta;
        n++;
    }
    return n;
}

static void emit_skipped(FILE *f, const char *pfx, const rt_cpe_dev *d, const unsigned char *elf) {
    const Elf32_Ehdr *eh = (const Elf32_Ehdr *)elf;
    int any = 0;
    for (uint32_t i = 0U; i < eh->e_phnum; i++) {
        const Elf32_Phdr *ph = (const Elf32_Phdr *)(elf + eh->e_phoff + (uint64_t)i * eh->e_phentsize);
        if (ph->p_type != (uint32_t)PT_LOAD)
            continue;
        uint32_t ta = 0U;
        const char *why = NULL;
        const uint32_t len = rt_cpe_seg_load_len(elf, ph);
        if (len == 0U)
            why = "only DMA port buffers (.bss); not zeroed";
        else if (rt_cpe_target(d, CPE_ENC_COL, CPE_ENC_ROW, ph->p_paddr, &ta) != 0)
            why = "address outside program/data memory";
        else if (ta == RT_CPE_SKIP_ADDR)
            why = "neighbour tile's data memory";
        if (!why)
            continue;
        if (!any)
            fprintf(f, "%s\n%s PT_LOAD segments not sent:\n", pfx, pfx);
        any = 1;
        fprintf(f, "%s   PT_LOAD[%u] paddr 0x%x, filesz %u, memsz %u: %s\n", pfx, (unsigned)i, (unsigned)ph->p_paddr,
                (unsigned)ph->p_filesz, (unsigned)ph->p_memsz, why);
    }
}

static void emit_hdr_words(FILE *f, const rt_cpe_blob_hdr *h) {
    fprintf(f, "    /* rt_cpe_blob_hdr (checked by the runtime, not sent) */\n");
    fprintf(f, "    0x%08xu, /* magic \"CPK1\" */\n", (unsigned)h->magic);
    fprintf(f, "    0x%08xu, /* version */\n", (unsigned)h->version);
    fprintf(f, "    0x%08xu, /* aie_gen */\n", (unsigned)h->aie_gen);
    fprintf(f, "    0x%08xu, /* fingerprint of device params + stream id */\n", (unsigned)h->fingerprint);
    fprintf(f, "    0x%08xu, /* stream_id (broadcast) */\n", (unsigned)h->stream_id);
    fprintf(f, "    0x%08xu, /* nresp: tiles that acknowledge */\n", (unsigned)h->nresp);
    fprintf(f, "    0x%08xu, /* body_words */\n", (unsigned)h->body_words);
    fprintf(f, "    0x%08xu, /* enc_tile (col << 16 | row) */\n", (unsigned)h->enc_tile);
}

static int is_dma_ch_ctrl(const rt_cpe_dev *d, uint32_t addr) {
    return addr >= d->dma_ch_ctrl_off && addr < d->dma_ch_ctrl_off + 2U * d->dma_nch * d->dma_ch_stride &&
           (addr - d->dma_ch_ctrl_off) % d->dma_ch_stride == 0U;
}

typedef struct {
    void (*banner)(void *ctx, uint32_t at, const char *text);
    void (*access)(void *ctx, uint32_t at, uint32_t len, const char *note);
    void *ctx;
} body_sink;

static void seg_banner(const body_sink *k, uint32_t at, const rt_cpe_dev *d, const cpe_seg_info *s) {
    int prog = s->tile_addr >= d->prog_mem_host_off && s->tile_addr < d->prog_mem_host_off + d->prog_mem_size;
    char b[320];
    snprintf(b, sizeof(b),
             "ELF PT_LOAD[%d]: paddr 0x%x, %u bytes -> %s memory 0x%05x on every core tile\n"
             "(MMIO path: XAie_LoadElfMem %s-memory write, one XAie_BlockWrite32 per tile)",
             s->phdr_idx, (unsigned)s->paddr, (unsigned)s->filesz, prog ? "program" : "data", (unsigned)s->tile_addr,
             prog ? "program" : "data");
    k->banner(k->ctx, at, b);
}

static void walk_body(const body_sink *k, const rt_cpe_dev *d, const uint32_t *body, uint32_t nwords,
                      const cpe_seg_info *segs, int nseg, uint32_t nresp) {
    int cur = -2, seg = -1;
    uint32_t acc = 0U, seg_left = 0U;
    char b[320], note[256];
    for (uint32_t i = 0U; i + 2U <= nwords; acc++) {
        uint32_t ctrl = body[i + 1U];
        uint32_t addr = ctrl & 0xFFFFFU, beats = ((ctrl >> 20U) & 0x3U) + 1U, op = (ctrl >> 22U) & 0x3U;
        uint32_t len = 2U + beats;
        if (i + len > nwords)
            break;
        if (op == 0x2U) {
            snprintf(b, sizeof(b),
                     "Completion barrier: last ELF word re-written as a write-with-return.\n"
                     "Each of the %u tiles answers one response word, so the runtime's\n"
                     "response BD completes only after every earlier write has landed.",
                     (unsigned)nresp);
            k->banner(k->ctx, i, b);
            snprintf(note, sizeof(note), "acc %u @%u: write-with-return 0x%05x", (unsigned)acc, (unsigned)i,
                     (unsigned)addr);
        } else if (addr == d->core_ctrl_off) {
            if (cur != -1)
                k->banner(k->ctx, i,
                          body[i + 2U] ? "Core reset pulse on every core tile"
                                       : "Release the core reset; the core stays disabled until launch");
            cur = -1;
            snprintf(note, sizeof(note), "acc %u @%u: Core_Control <- 0x%x (MMIO path: %s)", (unsigned)acc,
                     (unsigned)i, (unsigned)body[i + 2U], body[i + 2U] ? "XAie_CoreReset" : "XAie_CoreUnreset");
        } else if (is_dma_ch_ctrl(d, addr)) {
            if (cur != -3)
                k->banner(k->ctx, i,
                          "DMA channel reset pulse on every core tile: a previous launch leaves each\n"
                          "channel running its ping-pong BD chain (MMIO path: XAie_DmaChannelResetAll)");
            cur = -3;
            uint32_t c = (addr - d->dma_ch_ctrl_off) / d->dma_ch_stride;
            snprintf(note, sizeof(note), "acc %u @%u: DMA_%s_%u_Ctrl <- 0x%x (%s)", (unsigned)acc, (unsigned)i,
                     c < d->dma_nch ? "S2MM" : "MM2S", (unsigned)(c % d->dma_nch), (unsigned)body[i + 2U],
                     body[i + 2U] ? "DMA_CHANNEL_RESET" : "DMA_CHANNEL_UNRESET");
        } else {
            if (seg_left == 0U && seg + 1 < nseg) {
                seg++;
                seg_left = (segs[seg].filesz + 4U * RT_CPE_CHUNK_WORDS - 1U) / (4U * RT_CPE_CHUNK_WORDS);
                seg_banner(k, i, d, &segs[seg]);
            }
            if (seg_left)
                seg_left--;
            if (seg < 0 || addr < segs[seg].tile_addr || addr >= segs[seg].tile_addr + segs[seg].filesz)
                fprintf(stderr, "aiehlc_ctrlpkt: WARNING access %u at 0x%05x outside its PT_LOAD annotation\n",
                        (unsigned)acc, (unsigned)addr);
            cur = seg;
            snprintf(note, sizeof(note), "acc %u @%u: write %u word%s at 0x%05x", (unsigned)acc, (unsigned)i,
                     (unsigned)beats, beats > 1U ? "s" : "", (unsigned)addr);
        }
        k->access(k->ctx, i, len, note);
        i += len;
    }
}

typedef struct {
    FILE *f;
    const uint32_t *body;
} hdr_sink;

static void hdr_banner(void *ctx, uint32_t at, const char *text) {
    (void)at;
    FILE *f = ((hdr_sink *)ctx)->f;
    fprintf(f, "\n    /* ");
    for (const char *s = text; *s; s++)
        s[0] == '\n' ? fputs("\n     * ", f) : fputc(*s, f);
    fprintf(f, " */\n");
}

static void hdr_access(void *ctx, uint32_t at, uint32_t len, const char *note) {
    hdr_sink *h = (hdr_sink *)ctx;
    fprintf(h->f, "    ");
    for (uint32_t w = 0U; w < len; w++)
        fprintf(h->f, "0x%08xu, ", (unsigned)h->body[at + w]);
    fprintf(h->f, "/* %s */\n", note);
}

static void emit_body(FILE *f, const rt_cpe_dev *d, const uint32_t *body, uint32_t nwords, const cpe_seg_info *segs,
                      int nseg, uint32_t nresp) {
    hdr_sink h = {f, body};
    body_sink k = {hdr_banner, hdr_access, &h};
    walk_body(&k, d, body, nwords, segs, nseg, nresp);
}

#define DBG_MAX_MARKS 512
#define DBG_MAX_ACC 8192
typedef struct {
    cpd_mark marks[DBG_MAX_MARKS];
    char *mark_text[DBG_MAX_MARKS];
    int nmarks;
    char (*notes)[256];
    uint32_t nacc;
} dbg_sink;

static void dbg_banner(void *ctx, uint32_t at, const char *text) {
    dbg_sink *s = (dbg_sink *)ctx;
    if (s->nmarks >= DBG_MAX_MARKS)
        return;
    s->mark_text[s->nmarks] = strdup(text);
    s->marks[s->nmarks].at = at;
    s->marks[s->nmarks].text = s->mark_text[s->nmarks];
    s->nmarks++;
}

static void dbg_access(void *ctx, uint32_t at, uint32_t len, const char *note) {
    (void)at, (void)len;
    dbg_sink *s = (dbg_sink *)ctx;
    const char *c = strstr(note, ": ");
    c = c && !strncmp(note, "acc ", 4) ? c + 2 : note;
    if (!strncmp(c, "write ", 6))
        c = "";
    if (s->nacc < DBG_MAX_ACC)
        snprintf(s->notes[s->nacc++], 256, "%s", c);
}

static void dbg_note(void *ctx, uint32_t acc, uint32_t at, const uint32_t *words, uint32_t len, char *buf,
                     size_t n) {
    (void)at, (void)words, (void)len;
    const dbg_sink *s = (const dbg_sink *)ctx;
    snprintf(buf, n, "%s", acc < s->nacc ? s->notes[acc] : "");
}

static int write_debug(const char *dir, const char *func, int gen, const rt_cpe_blob_hdr *h, const uint32_t *body,
                       const rt_cpe_dev *d, const unsigned char *elf, const cpe_seg_info *segs, int nseg) {
    static dbg_sink s;
    memset(&s, 0, sizeof(s));
    s.notes = (char (*)[256])calloc(DBG_MAX_ACC, 256);
    if (!s.notes)
        return -1;
    body_sink k = {dbg_banner, dbg_access, &s};
    walk_body(&k, d, body, h->body_words, segs, nseg, h->nresp);
    char *skipped = NULL;
    size_t skipped_len = 0U;
    FILE *sf = open_memstream(&skipped, &skipped_len);
    if (sf) {
        emit_skipped(sf, "", d, elf);
        fclose(sf);
    }
    char name[200], title[4096];
    snprintf(name, sizeof(name), "kernel_%s", func);
    snprintf(title, sizeof(title),
             "%s: kernel-ELF load body of '%s' (__Runtime_load_kernel_group_*_ctrl)\n"
             "Broadcast (stream id 0x%x) to %u core tile(s), AIE gen %d, %u words. Same words as\n"
             "kernel_%s_ctrlpkt[] after its 8-word blob header (magic 0x%08x version %u\n"
             "fingerprint 0x%08x nresp %u enc_tile 0x%x, checked by the runtime, not sent).\n"
             "On the wire the runtime sends plan_ret (return-route prefix) first, then these words, in one\n"
             "shim MM2S BD with one TLAST; see %s_wire.h.%s",
             name, func, (unsigned)h->stream_id, (unsigned)h->nresp, gen, (unsigned)h->body_words, func,
             (unsigned)h->magic, (unsigned)h->version, (unsigned)h->fingerprint, (unsigned)h->nresp,
             (unsigned)h->enc_tile, name, skipped && skipped[0] ? skipped : "");
    size_t tl = strlen(title);
    while (tl && title[tl - 1U] == '\n')
        title[--tl] = '\0';
    int rc = cpd_dump(dir, name, title, body, h->body_words, s.marks, s.nmarks, dbg_note, &s);
    free(skipped);
    for (int i = 0; i < s.nmarks; i++)
        free(s.mark_text[i]);
    free(s.notes);
    return rc;
}

static int write_header(const char *path, const char *func, int gen, const rt_cpe_blob_hdr *h, const uint32_t *body,
                        const rt_cpe_dev *d, const unsigned char *elf, const cpe_seg_info *segs, int nseg) {
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f,
            "/* Auto-generated by aiehlc_ctrlpkt -- do not edit.\n"
            " *\n"
            " * AOT control packets for the kernel-ELF load of '%s' (#pragma control_packet_mode(aot)).\n"
            " * Replaces the runtime JIT encode inside __Runtime_load_kernel_group_*_ctrl(): every\n"
            " * access below is broadcast (stream id 0x%x) to all %u core tiles of the fabric.\n"
            " * At load time the runtime checks the header, prepends the deferred return-route\n"
            " * writes from __Runtime_ctrl_plan_init(), and sends the body in one shim MM2S BD.\n"
            " * AIE gen %d, %u body words. Access format: stream header, control header\n"
            " * ([19:0] tile address, [21:20] words-1, [23:22] op), data words.\n",
            func, (unsigned)h->stream_id, (unsigned)h->nresp, gen, (unsigned)h->body_words);
    emit_skipped(f, " *", d, elf);
    fprintf(f,
            " */\n"
            "#ifndef KERNEL_%s_CTRLPKT_H\n#define KERNEL_%s_CTRLPKT_H\n\n#include <stdint.h>\n\n"
            "static const uint32_t kernel_%s_ctrlpkt[] = {\n",
            func, func, func);
    emit_hdr_words(f, h);
    emit_body(f, d, body, h->body_words, segs, nseg, h->nresp);
    fprintf(f, "};\n\n#endif\n");
    return fclose(f) == 0 ? 0 : -1;
}

static int write_bin(const char *path, const rt_cpe_blob_hdr *h, const uint32_t *body) {
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    fwrite(h, sizeof(*h), 1, f);
    fwrite(body, sizeof(uint32_t), h->body_words, f);
    return fclose(f) == 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *manifest = NULL, *func = NULL, *elf_path = NULL, *out = NULL, *out_bin = NULL;
    const char *sites = NULL, *host_cc = NULL, *out_sites = NULL, *debug_dir = NULL;
    int gen = 5;
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--sites"))
            sites = argv[++i];
        else if (!strcmp(argv[i], "--host-cc"))
            host_cc = argv[++i];
        else if (!strcmp(argv[i], "--out-sites"))
            out_sites = argv[++i];
        else if (!strcmp(argv[i], "--manifest"))
            manifest = argv[++i];
        else if (!strcmp(argv[i], "--func"))
            func = argv[++i];
        else if (!strcmp(argv[i], "--elf"))
            elf_path = argv[++i];
        else if (!strcmp(argv[i], "--gen"))
            gen = (int)strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--out"))
            out = argv[++i];
        else if (!strcmp(argv[i], "--out-bin"))
            out_bin = argv[++i];
        else if (!strcmp(argv[i], "--debug-dir"))
            debug_dir = argv[++i];
    }
    if (sites)
        return out_sites ? ctrlpkt_emit_sites(sites, host_cc, out_sites, debug_dir) : 2;
    if (!manifest || !func || !elf_path || !out) {
        fprintf(stderr, "usage: aiehlc_ctrlpkt --manifest j --func n --elf e --gen g --out hdr.h [--out-bin b] "
                        "[--debug-dir d]\n");
        return 2;
    }
    rt_cpe_dev dev;
    if (rt_cpe_dev_for_gen(gen, &dev) != 0) {
        fprintf(stderr, "aiehlc_ctrlpkt: unsupported --gen %d\n", gen);
        return 2;
    }
    long jlen = 0, elen = 0;
    unsigned char *json = read_file(manifest, &jlen);
    if (!json) {
        fprintf(stderr, "aiehlc_ctrlpkt: cannot read manifest %s\n", manifest);
        return 1;
    }
    long nresp = manifest_int((const char *)json, func, "\"nresp\"", -1);
    long stream_id = manifest_int((const char *)json, func, "\"stream_id\"", -1);
    long reset = manifest_int((const char *)json, func, "\"reset\"", 1);
    free(json);
    if (nresp < 0 || stream_id < 0) {
        fprintf(stderr, "aiehlc_ctrlpkt: no manifest entry for func '%s'\n", func);
        return 1;
    }
    unsigned char *elf = read_file(elf_path, &elen);
    if (!elf) {
        fprintf(stderr, "aiehlc_ctrlpkt: cannot read ELF %s\n", elf_path);
        return 1;
    }
    uint32_t npkt = rt_cpe_count_pkts(&dev, elf, CPE_ENC_COL, CPE_ENC_ROW);
    uint32_t cap = (npkt + rt_cpe_reset_accesses(&dev) + 1U) * (2U + RT_CPE_CHUNK_WORDS) + 16U;
    uint32_t *body = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    uint32_t body_words = 0U, nacc = 0U;
    int rc = body ? rt_cpe_encode_body(&dev, elf, CPE_ENC_COL, CPE_ENC_ROW, (uint32_t)stream_id, reset ? 1 : 0, body,
                                       cap, &body_words, &nacc)
                  : -1;
    if (rc != 0) {
        fprintf(stderr, "aiehlc_ctrlpkt: encode failed rc=%d (npkt=%u cap=%u)\n", rc, npkt, cap);
        free(elf);
        free(body);
        return 1;
    }
    cpe_seg_info segs[CPE_MAX_SEGS];
    int nseg = collect_segs(&dev, elf, segs);
    rt_cpe_blob_hdr hdr;
    hdr.magic = RT_CPE_MAGIC;
    hdr.version = RT_CPE_VERSION;
    hdr.aie_gen = (uint32_t)gen;
    hdr.fingerprint = rt_cpe_fingerprint(&dev, (uint32_t)stream_id);
    hdr.stream_id = (uint32_t)stream_id;
    hdr.nresp = (uint32_t)nresp;
    hdr.body_words = body_words;
    hdr.enc_tile = ((uint32_t)CPE_ENC_COL << 16) | (uint32_t)CPE_ENC_ROW;
    rc = write_header(out, func, gen, &hdr, body, &dev, elf, segs, nseg);
    if (rc == 0 && out_bin)
        rc = write_bin(out_bin, &hdr, body);
    if (rc == 0 && debug_dir && write_debug(debug_dir, func, gen, &hdr, body, &dev, elf, segs, nseg) != 0)
        fprintf(stderr, "aiehlc_ctrlpkt: WARNING cannot write the debug dump to %s\n", debug_dir);
    free(elf);
    free(body);
    if (rc != 0) {
        fprintf(stderr, "aiehlc_ctrlpkt: cannot write %s\n", out);
        return 1;
    }
    fprintf(stderr, "aiehlc_ctrlpkt: %s func=%s gen=%d nresp=%ld body_words=%u accesses=%u fp=%08x\n", out, func, gen,
            nresp, body_words, nacc + 1U, hdr.fingerprint);
    return 0;
}
