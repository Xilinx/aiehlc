/* Validate an ELF64/AArch64 executable and place its PT_LOAD segments into the
 * app's DDR_LOW0. Nothing is written until every header has been checked, so a
 * rejected upload leaves the previous image intact. */
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

/* Zero the app's .bss-like sections exactly as a cold-boot crt0 would.
 * NOT the whole p_memsz tail: the BSP linker script folds the 1 GB .heap and the
 * .stack into the same PT_LOAD, and zero-filling + cache-cleaning that wedged the
 * supervisor. crt0 cannot be relied on either: it skips BSS clearing when the
 * core's APU_PWRCTL bit says "warm boot", which a PLM wake-up can look like. */
static uint64_t zero_bss(const uint8_t *img, uint64_t size) {
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)img;
    if (eh->e_shoff == 0 || eh->e_shentsize != sizeof(Elf64_Shdr) || eh->e_shstrndx >= eh->e_shnum ||
        eh->e_shoff + (uint64_t)eh->e_shnum * sizeof(Elf64_Shdr) > size)
        return 0;
    const Elf64_Shdr *sh = (const Elf64_Shdr *)(img + eh->e_shoff);
    const Elf64_Shdr *str = &sh[eh->e_shstrndx];
    uint64_t zeroed = 0;
    for (unsigned i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_type != SHT_NOBITS || !(sh[i].sh_flags & SHF_ALLOC) || (sh[i].sh_flags & SHF_TLS) ||
            sh[i].sh_size == 0)
            continue;
        if (sh[i].sh_name < str->sh_size && str->sh_offset + str->sh_size <= size) {
            const char *name = (const char *)img + str->sh_offset + sh[i].sh_name;
            if (strcmp(name, ".heap") == 0 || strcmp(name, ".stack") == 0)
                continue;
        }
        uint64_t lo = sh[i].sh_addr, hi = lo + sh[i].sh_size;
        if (hi < lo || hi > NB_APP_LIMIT)
            continue;
        uint8_t *dst = (uint8_t *)(uintptr_t)lo;
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
        uint8_t *dst = (uint8_t *)(uintptr_t)ph[i].p_paddr;
        memcpy(dst, img + ph[i].p_offset, ph[i].p_filesz);
        /* The app core starts MMU/caches off: the bytes must be in DDR. */
        Xil_DCacheFlushRange((INTPTR)dst, (INTPTR)ph[i].p_filesz);
        placed += ph[i].p_filesz;
    }
    uint64_t zeroed = zero_bss(img, size);
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
int elf_patch_exit(const uint8_t *img, uint64_t size, uint64_t park_addr) {
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)img;
    if (eh->e_shoff == 0 || eh->e_shoff + (uint64_t)eh->e_shnum * sizeof(Elf64_Shdr) > size)
        return 0;
    const Elf64_Shdr *sh = (const Elf64_Shdr *)(img + eh->e_shoff);
    for (unsigned i = 0; i < eh->e_shnum; i++) {
        if (sh[i].sh_type != SHT_SYMTAB || sh[i].sh_link >= eh->e_shnum)
            continue;
        const Elf64_Shdr *st = &sh[sh[i].sh_link];
        if (sh[i].sh_offset + sh[i].sh_size > size || st->sh_offset + st->sh_size > size)
            return 0;
        const Elf64_Sym *sym = (const Elf64_Sym *)(img + sh[i].sh_offset);
        for (uint64_t k = 0; k < sh[i].sh_size / sizeof(Elf64_Sym); k++) {
            if (sym[k].st_name >= st->sh_size ||
                strcmp((const char *)img + st->sh_offset + sym[k].st_name, "_exit") != 0)
                continue;
            uint64_t a = sym[k].st_value;
            if (a + 16 > NB_APP_LIMIT || (a & 3))
                return 0;
            volatile uint32_t *w = (volatile uint32_t *)(uintptr_t)a;
            if (w[0] != 0x14000000U || w[1] || w[2] || w[3])
                return 0;       /* not the expected `b .` + padding: leave it alone */
            w[0] = 0x58000050U; /* ldr x16, #8 */
            w[1] = 0xD61F0200U; /* br  x16     */
            w[2] = (uint32_t)park_addr;
            w[3] = (uint32_t)(park_addr >> 32);
            Xil_DCacheFlushRange((INTPTR)a, 16);
            return 1;
        }
    }
    return 0;
}
