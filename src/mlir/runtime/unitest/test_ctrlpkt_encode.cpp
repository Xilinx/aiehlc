/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "aie_ctrlpkt_devparams.h"
#include "aie_ctrlpkt_encode.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <elf.h>
#include <vector>

static uint32_t odd_parity_expect(uint32_t v) {
    for (uint8_t i = 16U; i > 0U; i >>= 1U)
        v ^= (v >> i);
    return (v & 1U) ? 0U : 1U;
}
static void check_parity(uint32_t word) { assert(((word >> 31) & 1U) == odd_parity_expect(word & 0x7FFFFFFFU)); }

static void add_seg(std::vector<uint8_t> &img, std::vector<Elf32_Phdr> &phdrs, uint32_t paddr, uint32_t filesz,
                    uint32_t memsz, uint8_t fill) {
    Elf32_Phdr ph{};
    ph.p_type = PT_LOAD;
    ph.p_paddr = paddr;
    ph.p_vaddr = paddr;
    ph.p_memsz = memsz;
    ph.p_filesz = filesz;
    ph.p_offset = (uint32_t)img.size();
    for (uint32_t i = 0; i < filesz; i++)
        img.push_back((uint8_t)(fill + i));
    phdrs.push_back(ph);
}

int main() {
    rt_cpe_dev d;
    assert(rt_cpe_dev_for_gen(5, &d) == 0);

    std::vector<uint8_t> img(sizeof(Elf32_Ehdr), 0);
    std::vector<Elf32_Phdr> phdrs;

    add_seg(img, phdrs, 0x100, 16, 16, 0x10);
    add_seg(img, phdrs, 0x70000, 8, 8, 0x40);
    add_seg(img, phdrs, 0x40000, 32, 32, 0x80);
    add_seg(img, phdrs, 0x200, 0, 64, 0x00);

    uint32_t phoff = (uint32_t)img.size();
    for (auto &ph : phdrs) {
        const uint8_t *p = (const uint8_t *)&ph;
        img.insert(img.end(), p, p + sizeof(ph));
    }
    Elf32_Ehdr *eh = (Elf32_Ehdr *)img.data();
    eh->e_phoff = phoff;
    eh->e_phnum = (uint16_t)phdrs.size();
    eh->e_phentsize = sizeof(Elf32_Phdr);

    const int col = 0, row = 2;
    uint32_t npkt = rt_cpe_count_pkts(&d, img.data(), col, row);
    assert(npkt == 2);

    std::vector<uint32_t> pkt(256, 0xCCCCCCCCu);
    uint32_t body_words = 0, nacc = 0;
    int rc =
        rt_cpe_encode_body(&d, img.data(), col, row, 0x10, 1, pkt.data(), (uint32_t)pkt.size(), &body_words, &nacc);
    assert(rc == 0);
    assert(rt_cpe_reset_accesses(&d) == 10);
    assert(nacc == 12);
    assert(body_words == 43);

    check_parity(pkt[0]);
    check_parity(pkt[1]);
    assert((pkt[0] & 0x1F) == 0x10);
    assert((pkt[1] & 0xFFFFF) == 0x38000);
    assert(pkt[2] == d.core_rst_mask);
    for (uint32_t v = 0; v < 2; v++)
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t a = 3 + (v * 4 + c) * 3;
            check_parity(pkt[a + 1]);
            assert((pkt[a + 1] & 0xFFFFF) == 0x1DE00u + 8u * c);
            assert(pkt[a + 2] == (v ? 0u : d.dma_ch_rst_mask));
        }
    assert((pkt[28] & 0xFFFFF) == 0x38000);
    assert(pkt[29] == 0u);

    uint32_t ack_ctrl = pkt[body_words - 2];
    check_parity(ack_ctrl);
    assert(((ack_ctrl >> 22) & 0x3u) == 0x2u);
    assert((ack_ctrl & 0xFFFFF) == 4u);

    uint32_t fp = rt_cpe_fingerprint(&d, 0x10);
    assert(fp == rt_cpe_fingerprint(&d, 0x10));
    rt_cpe_dev d2 = d;
    d2.core_rst_mask ^= 0x1u;
    assert(rt_cpe_fingerprint(&d2, 0x10) != fp);
    d2 = d;
    d2.dma_ch_ctrl_off += 0x10u;
    assert(rt_cpe_fingerprint(&d2, 0x10) != fp);

    rt_cpe_blob_hdr hdr{};
    hdr.magic = RT_CPE_MAGIC;
    hdr.version = RT_CPE_VERSION;
    hdr.fingerprint = fp;
    hdr.stream_id = 0x10;
    hdr.nresp = 16;
    hdr.body_words = body_words;
    hdr.enc_tile = ((uint32_t)col << 16) | (uint32_t)row;
    assert(hdr.magic == RT_CPE_MAGIC && hdr.body_words == body_words);
    assert((hdr.enc_tile >> 16) == 0u && (hdr.enc_tile & 0xFFFF) == 2u);

    {
        uint32_t p[64] = {0};
        rt_cpe_seg s;
        rt_cpe_seg_reset(&s);
        const uint32_t d[6] = {0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1};
        assert(rt_cpe_seg_write(&s, p, 64, 0x3, 0x100, d, 4) == 0);
        assert(rt_cpe_seg_write(&s, p, 64, 0x3, 0x110, d + 4, 2) == 0);
        assert(s.pw == 2 + 4 + 2 + 2);
        assert((p[0] & 0x1F) == 0x3);
        check_parity(p[0]);
        check_parity(p[1]);
        assert((p[1] & 0xFFFFF) == 0x100 && ((p[1] >> 20) & 3) == 3);
        assert((p[7] & 0xFFFFF) == 0x110 && ((p[7] >> 20) & 3) == 1);
        uint32_t q[16];
        rt_cpe_seg t;
        rt_cpe_seg_reset(&t);
        assert(rt_cpe_seg_write(&t, q, 16, 0, 0x200, d, 1) == 0);
        assert(rt_cpe_seg_write(&t, q, 16, 0, 0x208, d, 1) == 0);
        assert(t.pw == 6);
        uint32_t len = rt_cpe_seg_finish(&s, p);
        assert(len == s.pw && len == 2 + 4 + 2 + 1 + 2 + 1);
        assert(((p[7] >> 20) & 3) == 0 && ((p[7] >> 22) & 3) == 0);
        uint32_t ack = p[s.last];
        check_parity(ack);
        assert(((ack >> 22) & 3) == 2 && (ack & 0xFFFFF) == 0x114 && p[s.last + 1] == 0xB1);
        assert((p[s.last - 1] & 0x1F) == 0x3);
        uint32_t before[64];
        memcpy(before, p, sizeof(p));
        assert(rt_cpe_seg_finish(&s, p) == len && memcmp(before, p, sizeof(p)) == 0);
        rt_cpe_seg u;
        rt_cpe_seg_reset(&u);
        uint32_t r[6];
        assert(rt_cpe_seg_write(&u, r, 6, 0, 0x300, d, 1) == 0);
        assert(rt_cpe_seg_write(&u, r, 6, 0, 0x400, d, 1) != 0);
    }

    {
        rt_cpe_wr w[5] = {
            {1, 3, 0x3F230, 0x101, RT_CPE_WR_UNDECIDED}, {2, 3, 0x3F230, 0x101, RT_CPE_WR_UNDECIDED},
            {1, 3, 0x3F024, 0x19, RT_CPE_WR_UNDECIDED},  {2, 3, 0x3F024, 0x29, RT_CPE_WR_UNDECIDED},
            {1, 3, 0x3F100, 0x1, RT_CPE_WR_UNDECIDED},
        };
        uint32_t off[4], val[4];
        uint32_t nb = rt_cpe_ret_split(w, 5, 2, off, val, 4);
        assert(nb == 1 && off[0] == 0x3F230 && val[0] == 0x101);
        assert(w[0].state == RT_CPE_WR_BCAST && w[1].state == RT_CPE_WR_BCAST);
        assert(w[2].state == RT_CPE_WR_MMIO && w[3].state == RT_CPE_WR_MMIO);
        assert(w[4].state == RT_CPE_WR_MMIO);
        rt_cpe_wr dup[2] = {{1, 3, 0x10, 7, RT_CPE_WR_UNDECIDED}, {1, 3, 0x10, 7, RT_CPE_WR_UNDECIDED}};
        assert(rt_cpe_ret_split(dup, 2, 2, off, val, 4) == 0);
    }

    {
        uint32_t o[8], one = 5;
        assert(rt_cpe_pktize(o, 8, 0x10, 0x38000, &one, 1, 0, 0) == 3);
        assert(o[0] == pkt[0] && (o[1] & 0xFFFFF) == 0x38000 && o[2] == 5);
        assert(rt_cpe_pktize(o, 2, 0x10, 0x38000, &one, 1, 0, 0) == 0);
    }

    printf("PASS\n");
    return 0;
}
