/* Validate an ELF64/AArch64 executable and place its PT_LOAD segments into the
 * app's DDR_LOW0. Nothing is written until every header has been checked, so a
 * rejected upload leaves the previous image intact.
 *
 * Two ways in: elf_place() from a whole image in staging (RERUN), and the
 * streaming elf_stream_*() that places a LOAD_RUN as its bytes arrive (an ELF
 * with bf16 weights is larger than staging). Both produce the same bytes in
 * DDR_LOW0 (test/streamload_x86.sh compares them). */
#include <elf.h>
#include <stdio.h>
#include <string.h>

#include "netboot.h"
#include "xil_cache.h"

static int fail(char *msg, size_t msglen, const char *why) {
    snprintf(msg, msglen, "%s", why);
    return -1;
}

static int check_headers(const uint8_t *img, uint64_t size, char *msg, size_t msglen) {
    if (size < sizeof(Elf64_Ehdr))
        return fail(msg, msglen, "file shorter than an ELF header");
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)img;
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0)
        return fail(msg, msglen, "not an ELF file");
    if (eh->e_ident[EI_CLASS] != ELFCLASS64 || eh->e_ident[EI_DATA] != ELFDATA2LSB)
        return fail(msg, msglen, "not ELF64 little endian");
    if (eh->e_machine != EM_AARCH64 || eh->e_type != ET_EXEC)
        return fail(msg, msglen, "not an AArch64 ET_EXEC image");
    if (eh->e_phentsize != sizeof(Elf64_Phdr) || eh->e_phnum == 0 ||
        eh->e_phoff + (uint64_t)eh->e_phnum * sizeof(Elf64_Phdr) > size)
        return fail(msg, msglen, "program headers out of range");
    if (eh->e_entry >= NB_APP_LIMIT)
        return fail(msg, msglen, "entry point outside DDR_LOW0");

    const Elf64_Phdr *ph = (const Elf64_Phdr *)(img + eh->e_phoff);
    for (unsigned i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_LOAD || ph[i].p_memsz == 0)
            continue;
        uint64_t lo = ph[i].p_paddr, hi = lo + ph[i].p_memsz;
        if (hi < lo || hi > NB_APP_LIMIT) {
            snprintf(msg, msglen, "PT_LOAD %u [0x%llx,0x%llx) outside DDR_LOW0", i, (unsigned long long)lo,
                     (unsigned long long)hi);
            return -1;
        }
        if (ph[i].p_filesz > ph[i].p_memsz || ph[i].p_offset + ph[i].p_filesz > size ||
            ph[i].p_offset + ph[i].p_filesz < ph[i].p_offset) {
            snprintf(msg, msglen, "PT_LOAD %u file range out of bounds", i);
            return -1;
        }
    }
    return 0;
}

/* Where the section-level passes read the file from: a whole image, or what a
 * stream kept (header prefix, staging copy, or the gap cache). */
struct elf_src {
    const uint8_t *img; /* whole image, or NULL: read through s */
    const struct nb_stream *s;
    uint64_t size;
};

static const uint8_t *src_at(const struct elf_src *src, uint64_t off, uint64_t len) {
    if (off + len < off || off + len > src->size)
        return NULL;
    if (src->img)
        return src->img + off;
    const struct nb_stream *s = src->s;
    if (off + len <= s->hdr_len)
        return s->hdr + off;
    if (s->stage)
        return s->stage + off;
    for (unsigned g = 0; g < s->ngap; g++) {
        const struct nb_gap *gp = &s->gap[g];
        if (off >= gp->coff && off + len <= gp->off + gp->len)
            return s->cache + gp->cpos + (off - gp->coff);
    }
    return NULL;
}

/* Zero the app's .bss-like sections exactly as a cold-boot crt0 would.
 * NOT the whole p_memsz tail: the BSP linker script folds the 1 GB .heap and the
 * .stack into the same PT_LOAD, and zero-filling + cache-cleaning that wedged the
 * supervisor. crt0 cannot be relied on either: it skips BSS clearing when the
 * core's APU_PWRCTL bit says "warm boot", which a PLM wake-up can look like.
 * *missing: sections skipped because their name was not available (stream cache full). */
static uint64_t zero_bss(const struct elf_src *src, unsigned *missing) {
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)src_at(src, 0, sizeof(Elf64_Ehdr));
    *missing = 0;
    if (!eh || eh->e_shoff == 0 || eh->e_shentsize != sizeof(Elf64_Shdr) || eh->e_shstrndx >= eh->e_shnum ||
        eh->e_shoff + (uint64_t)eh->e_shnum * sizeof(Elf64_Shdr) > src->size)
        return 0;
    const Elf64_Shdr *sh = (const Elf64_Shdr *)src_at(src, eh->e_shoff, (uint64_t)eh->e_shnum * sizeof(Elf64_Shdr));
    if (!sh) {
        *missing = eh->e_shnum;
        return 0;
    }
    const Elf64_Shdr *str = &sh[eh->e_shstrndx];
    const char *names = NULL;
    if (str->sh_offset + str->sh_size <= src->size)
        names = (const char *)src_at(src, str->sh_offset, str->sh_size);
    uint64_t zeroed = 0;
    for (unsigned i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_type != SHT_NOBITS || !(sh[i].sh_flags & SHF_ALLOC) || (sh[i].sh_flags & SHF_TLS) ||
            sh[i].sh_size == 0)
            continue;
        if (sh[i].sh_name < str->sh_size && str->sh_offset + str->sh_size <= src->size) {
            if (!names) { /* cannot tell .heap from .bss: leave it alone */
                (*missing)++;
                continue;
            }
            const char *name = names + sh[i].sh_name;
            uint64_t room = str->sh_size - sh[i].sh_name;
            if (strncmp(name, ".heap", room < 6 ? room : 6) == 0 || strncmp(name, ".stack", room < 7 ? room : 7) == 0)
                continue;
        }
        uint64_t lo = sh[i].sh_addr, hi = lo + sh[i].sh_size;
        if (hi < lo || hi > NB_APP_LIMIT)
            continue;
        uint8_t *dst = NB_PHYS(lo);
        memset(dst, 0, sh[i].sh_size);
        Xil_DCacheFlushRange((INTPTR)dst, (INTPTR)sh[i].sh_size);
        zeroed += sh[i].sh_size;
    }
    return zeroed;
}

int elf_place(const uint8_t *img, uint64_t size, uint64_t *entry, char *msg, size_t msglen) {
    if (check_headers(img, size, msg, msglen) != 0)
        return -1;
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)img;
    const Elf64_Phdr *ph = (const Elf64_Phdr *)(img + eh->e_phoff);
    uint64_t placed = 0;
    for (unsigned i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_LOAD || ph[i].p_filesz == 0)
            continue;
        /* Address 0 is a valid destination here: build with
         * -fno-delete-null-pointer-checks so the copy is not optimized away. */
        uint8_t *dst = NB_PHYS(ph[i].p_paddr);
        memcpy(dst, img + ph[i].p_offset, ph[i].p_filesz);
        /* The app core starts MMU/caches off: the bytes must be in DDR. */
        Xil_DCacheFlushRange((INTPTR)dst, (INTPTR)ph[i].p_filesz);
        placed += ph[i].p_filesz;
    }
    struct elf_src src = {img, NULL, size};
    unsigned missing;
    uint64_t zeroed = zero_bss(&src, &missing);
    *entry = eh->e_entry;
    snprintf(msg, msglen, "placed %llu bytes, zeroed %llu bss bytes, entry 0x%llx", (unsigned long long)placed,
             (unsigned long long)zeroed, (unsigned long long)eh->e_entry);
    return 0;
}

/* Redirect the placed app's _exit to the supervisor's park routine so a finished
 * app core can be reused without the PLM (park.S). Patched only when _exit is the
 * BSP's `b .` followed by 12 bytes of padding, room for
 *   ldr x16, #8 ; br x16 ; .quad park_addr
 * Returns 1 if patched, 0 if the image does not match (core then stays busy). */
static int patch_exit(const struct elf_src *src, uint64_t park_addr) {
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)src_at(src, 0, sizeof(Elf64_Ehdr));
    if (!eh || eh->e_shoff == 0 || eh->e_shoff + (uint64_t)eh->e_shnum * sizeof(Elf64_Shdr) > src->size)
        return 0;
    const Elf64_Shdr *sh = (const Elf64_Shdr *)src_at(src, eh->e_shoff, (uint64_t)eh->e_shnum * sizeof(Elf64_Shdr));
    if (!sh)
        return 0;
    for (unsigned i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_type != SHT_SYMTAB || sh[i].sh_link >= eh->e_shnum)
            continue;
        const Elf64_Shdr *st = &sh[sh[i].sh_link];
        if (sh[i].sh_offset + sh[i].sh_size > src->size || st->sh_offset + st->sh_size > src->size)
            return 0;
        const Elf64_Sym *sym = (const Elf64_Sym *)src_at(src, sh[i].sh_offset, sh[i].sh_size);
        const char *strs = (const char *)src_at(src, st->sh_offset, st->sh_size);
        if (!sym || !strs)
            return 0;
        for (uint64_t k = 0; k < sh[i].sh_size / sizeof(Elf64_Sym); k++) {
            if (sym[k].st_name >= st->sh_size)
                continue;
            uint64_t room = st->sh_size - sym[k].st_name;
            if (strncmp(strs + sym[k].st_name, "_exit", room < 6 ? room : 6) != 0)
                continue;
            uint64_t a = sym[k].st_value;
            if (a + 16 > NB_APP_LIMIT || (a & 3))
                return 0;
            volatile uint32_t *w = (volatile uint32_t *)NB_PHYS(a);
            if (w[0] != 0x14000000U || w[1] || w[2] || w[3])
                return 0;       /* not the expected `b .` + padding: leave it alone */
            w[0] = 0x58000050U; /* ldr x16, #8 */
            w[1] = 0xD61F0200U; /* br  x16     */
            w[2] = (uint32_t)park_addr;
            w[3] = (uint32_t)(park_addr >> 32);
            Xil_DCacheFlushRange((INTPTR)NB_PHYS(a), 16);
            return 1;
        }
    }
    return 0;
}

int elf_patch_exit(const uint8_t *img, uint64_t size, uint64_t park_addr) {
    struct elf_src src = {img, NULL, size};
    return patch_exit(&src, park_addr);
}

/* ------------------------------------------------------------ streaming */

void elf_stream_begin(struct nb_stream *s, uint64_t size, uint8_t *stage, uint8_t *cache, uint64_t cache_size) {
    s->size = size;
    s->got = 0;
    s->stage = stage;
    s->cache = cache;
    s->cache_size = cache_size;
    s->phase = 0;
    s->wrote = 0;
    s->hdr_len = 0;
    s->placed = 0;
    s->ngap = 0;
    s->err[0] = 0;
    nb_sha256_init(&s->sha);
}

static int stream_reject(struct nb_stream *s, const char *why) {
    if (why)
        snprintf(s->err, sizeof(s->err), "%s", why);
    s->phase = -1;
    return -1;
}

/* Headers are in and valid: list the file ranges no PT_LOAD covers and give each
 * the cache room it can have, last range first (section headers, .shstrtab,
 * .strtab and .symtab sit at the end of the file, after the debug sections). */
static int stream_plan(struct nb_stream *s) {
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)s->hdr;
    const Elf64_Phdr *ph = (const Elf64_Phdr *)(s->hdr + eh->e_phoff);
    uint64_t pos = 0;
    s->ngap = 0;
    while (pos < s->size) {
        /* the PT_LOAD file range covering pos, or the nearest one after it */
        uint64_t cover_end = 0, next = s->size;
        for (unsigned i = 0; i < eh->e_phnum; i++) {
            if (ph[i].p_type != PT_LOAD || ph[i].p_filesz == 0)
                continue;
            uint64_t a = ph[i].p_offset, b = a + ph[i].p_filesz;
            if (a <= pos && pos < b && b > cover_end)
                cover_end = b;
            else if (a > pos && a < next)
                next = a;
        }
        if (cover_end) {
            pos = cover_end;
            continue;
        }
        if (s->ngap == NB_STREAM_GAP_MAX)
            return stream_reject(s, "too many file ranges outside PT_LOAD segments to stream");
        struct nb_gap *g = &s->gap[s->ngap++];
        g->off = pos;
        g->len = next - pos;
        g->coff = next; /* nothing cached yet */
        g->cpos = 0;
        pos = next;
    }
    if (s->stage || !s->cache)
        return 0;
    uint64_t room = s->cache_size;
    for (unsigned k = s->ngap; k-- > 0 && room > 16;) {
        struct nb_gap *g = &s->gap[k];
        /* keep (cpos - coff) a multiple of 16 so cached tables stay aligned */
        uint64_t n = g->len < room - 16 ? g->len : room - 16;
        g->coff = g->off + g->len - n;
        uint64_t base = s->cache_size - room;
        g->cpos = base + ((g->coff - base) & 15U);
        room -= (g->cpos - base) + n;
    }
    return 0;
}

static void stream_place(struct nb_stream *s, uint64_t off, const uint8_t *data, uint64_t len) {
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)s->hdr;
    const Elf64_Phdr *ph = (const Elf64_Phdr *)(s->hdr + eh->e_phoff);
    uint64_t end = off + len;
    for (unsigned i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_LOAD || ph[i].p_filesz == 0)
            continue;
        uint64_t a = ph[i].p_offset, b = a + ph[i].p_filesz;
        uint64_t lo = off > a ? off : a, hi = end < b ? end : b;
        if (lo >= hi)
            continue;
        /* address 0 is valid: -fno-delete-null-pointer-checks (build.sh) */
        memcpy(NB_PHYS(ph[i].p_paddr + (lo - a)), data + (lo - off), hi - lo);
        s->placed += hi - lo;
        s->wrote = 1;
    }
    if (s->stage || !s->cache)
        return;
    for (unsigned g = 0; g < s->ngap; g++) {
        const struct nb_gap *gp = &s->gap[g];
        uint64_t a = gp->coff, b = gp->off + gp->len;
        uint64_t lo = off > a ? off : a, hi = end < b ? end : b;
        if (lo < hi)
            memcpy(s->cache + gp->cpos + (lo - a), data + (lo - off), hi - lo);
    }
}

int elf_stream_feed(struct nb_stream *s, const uint8_t *data, uint64_t len) {
    if (s->phase < 0)
        return -1;
    if (len > s->size - s->got)
        return stream_reject(s, "more payload than elf_size");
    uint64_t off = s->got;
    s->got += len;
    nb_sha256_update(&s->sha, data, len);
    if (s->stage)
        memcpy(s->stage + off, data, len);
    if (s->phase == 1) {
        stream_place(s, off, data, len);
        return 0;
    }
    /* phase 0: collect the ELF and program headers */
    if (off < NB_STREAM_HDR_MAX) {
        uint64_t take = NB_STREAM_HDR_MAX - off < len ? NB_STREAM_HDR_MAX - off : len;
        memcpy(s->hdr + off, data, take);
        s->hdr_len = (uint32_t)(off + take);
    }
    if (s->hdr_len < sizeof(Elf64_Ehdr))
        return 0;
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)s->hdr;
    uint64_t need = sizeof(Elf64_Ehdr);
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) == 0 && eh->e_phentsize == sizeof(Elf64_Phdr)) {
        if (eh->e_phoff > NB_STREAM_HDR_MAX || eh->e_phoff + (uint64_t)eh->e_phnum * sizeof(Elf64_Phdr) > NB_STREAM_HDR_MAX)
            return stream_reject(s, "program headers not within the first 64 KiB: cannot stream this ELF");
        need = eh->e_phoff + (uint64_t)eh->e_phnum * sizeof(Elf64_Phdr);
        if (need < sizeof(Elf64_Ehdr))
            need = sizeof(Elf64_Ehdr);
    }
    if (s->hdr_len < need && s->got < s->size)
        return 0;
    /* every header checked before the first byte reaches the app's DDR */
    if (check_headers(s->hdr, s->size, s->err, sizeof(s->err)) != 0)
        return stream_reject(s, NULL);
    if (stream_plan(s) != 0)
        return -1;
    s->phase = 1;
    /* what arrived so far: the buffered prefix, then the rest of this chunk */
    stream_place(s, 0, s->hdr, s->hdr_len);
    if (s->got > s->hdr_len)
        stream_place(s, s->hdr_len, data + (s->hdr_len - off), s->got - s->hdr_len);
    return 0;
}

int elf_stream_finish(struct nb_stream *s, const uint8_t want_sha[32], uint64_t *entry, char *msg, size_t msglen) {
    if (s->phase < 0) {
        snprintf(msg, msglen, "%s", s->err);
        return -1;
    }
    if (s->got != s->size) {
        snprintf(msg, msglen, "short upload: %llu of %llu bytes", (unsigned long long)s->got,
                 (unsigned long long)s->size);
        return -1;
    }
    uint8_t digest[32];
    nb_sha256_final(&s->sha, digest);
    if (memcmp(digest, want_sha, 32) != 0) {
        snprintf(msg, msglen,
                 "sha256 mismatch: upload corrupted after %llu bytes were placed%s; NEED_RELOAD (send LOAD_RUN again)",
                 (unsigned long long)s->placed, s->wrote ? ", app area invalid" : "");
        s->phase = -1;
        return -2;
    }
    if (s->phase == 0) { /* the whole file was shorter than its headers claim */
        if (check_headers(s->hdr, s->size, msg, msglen) != 0)
            return -1;
        snprintf(msg, msglen, "ELF headers incomplete");
        return -1;
    }
    struct elf_src src = {NULL, s, s->size};
    unsigned missing;
    uint64_t zeroed = zero_bss(&src, &missing);
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)s->hdr;
    *entry = eh->e_entry;
    snprintf(msg, msglen, "placed %llu bytes, zeroed %llu bss bytes, entry 0x%llx", (unsigned long long)s->placed,
             (unsigned long long)zeroed, (unsigned long long)eh->e_entry);
    if (missing) {
        size_t n = strlen(msg);
        snprintf(msg + n, msglen - n, " (%u NOBITS sections not zeroed: names not cached)", missing);
    }
    return 0;
}

int elf_stream_patch_exit(struct nb_stream *s, uint64_t park_addr) {
    struct elf_src src = {NULL, s, s->size};
    return patch_exit(&src, park_addr);
}

void elf_stream_sync(struct nb_stream *s) {
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)s->hdr;
    const Elf64_Phdr *ph = (const Elf64_Phdr *)(s->hdr + eh->e_phoff);
    for (unsigned i = 0; i < eh->e_phnum; i++)
        if (ph[i].p_type == PT_LOAD && ph[i].p_filesz)
            Xil_DCacheFlushRange((INTPTR)NB_PHYS(ph[i].p_paddr), (INTPTR)ph[i].p_filesz);
    Xil_ICacheInvalidate();
}
