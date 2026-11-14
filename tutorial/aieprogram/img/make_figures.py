#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Generate the SVG figures of tutorial/aieprogram (stdlib only).

    python3 tutorial/aieprogram/img/make_figures.py

Every number drawn here is the one the example uses (and the compiler reports
in its [KernelAnalysis] block), so regenerate after changing an example.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent

A_COL = "#f4a261"   # A / IFM (row broadcast)
B_COL = "#6fa8dc"   # B / WEIGHTS (column broadcast)
C_COL = "#93c47d"   # C / OFM (gather)
HI = "#e63946"      # highlighted core
GRID = "#555555"
FONT = "font-family='DejaVu Sans, Arial, sans-serif'"


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def rect(self, x, y, w, h, fill="none", stroke=GRID, sw=1.0, dash=None, op=1.0, rx=0):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{rx}' "
                          f"fill='{fill}' fill-opacity='{op}' stroke='{stroke}' stroke-width='{sw}'{d}/>")

    def line(self, x1, y1, x2, y2, stroke=GRID, sw=1.0, dash=None, arrow=False):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        m = " marker-end='url(#arr)'" if arrow else ""
        self.parts.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
                          f"stroke='{stroke}' stroke-width='{sw}'{d}{m}/>")

    def text(self, x, y, s, size=12, anchor="middle", color="#222", weight="normal", rot=None):
        r = f" transform='rotate({rot} {x:.1f} {y:.1f})'" if rot is not None else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' text-anchor='{anchor}' "
                          f"fill='{color}' font-weight='{weight}'{r}>{s}</text>")

    def save(self, name):
        head = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.w}' height='{self.h}' "
                f"viewBox='0 0 {self.w} {self.h}'>\n<defs><marker id='arr' markerUnits='userSpaceOnUse' markerWidth='12' markerHeight='10' "
                "refX='11' refY='5' orient='auto'><path d='M0,0 L12,5 L0,10 z' fill='#444'/></marker></defs>\n"
                f"<rect width='100%' height='100%' fill='white'/>\n")
        (OUT / name).write_text(head + "\n".join(self.parts) + "\n</svg>\n")
        print("wrote", OUT / name)


# ---------------------------------------------------------------- GEMM ----
def gemm_tiling():
    """A[64,256], B[64,256], C[64,64]; core (pm, pn) = (1, 2) highlighted."""
    s = Svg(1040, 420)
    s.text(520, 24, "simplegemm_kernel_port.cc:  C[64,64] = A[64,256] x B[64,256]^T  on the 4x4 mesh",
           15, weight="bold")
    s.text(520, 44, "highlighted: the core with program_id(0) = pm = 1, program_id(1) = pn = 2", 12, color=HI)
    pm_hi, pn_hi = 1, 2
    # A: 64 rows x 256 cols -> 4 row bands (pm) x 4 K chunks (j)
    ax, ay, cw, rh = 60, 80, 70, 60
    s.text(ax + 2 * cw, ay - 14, "A  [64 x 256]   (input a, aie::IFM)", 13, weight="bold")
    for r in range(4):
        for j in range(4):
            hi = r == pm_hi
            s.rect(ax + j * cw, ay + r * rh, cw, rh, A_COL if hi else "#fde8d4", GRID)
            s.text(ax + j * cw + cw / 2, ay + r * rh + rh / 2 + 4, f"j={j}", 11, color="#333")
        s.text(ax - 8, ay + r * rh + rh / 2 + 4, f"pm={r}", 11, "end", HI if r == pm_hi else "#333")
    s.text(ax + 2 * cw, ay + 4 * rh + 18, "rows 16*pm .. +16  |  K chunk 64*j .. +64", 11)
    s.text(ax + 2 * cw, ay + 4 * rh + 34, "block {16, 64}: load_buffer(a, {pm*16, j*64}, {16, 64})", 11,
           color="#a0522d")
    # B: 64 rows x 256 cols -> 4 row bands (pn) x 4 K chunks (j)
    bx = 400
    s.text(bx + 2 * cw, ay - 14, "B  [64 x 256]   (input b, aie::WEIGHTS)", 13, weight="bold")
    for r in range(4):
        for j in range(4):
            hi = r == pn_hi
            s.rect(bx + j * cw, ay + r * rh, cw, rh, B_COL if hi else "#dbe8f5", GRID)
            s.text(bx + j * cw + cw / 2, ay + r * rh + rh / 2 + 4, f"j={j}", 11, color="#333")
        s.text(bx - 8, ay + r * rh + rh / 2 + 4, f"pn={r}", 11, "end", HI if r == pn_hi else "#333")
    s.text(bx + 2 * cw, ay + 4 * rh + 18, "rows 16*pn .. +16  |  K chunk 64*j .. +64", 11)
    s.text(bx + 2 * cw, ay + 4 * rh + 34, "block {16, 64}: load_buffer(b, {pn*16, j*64}, {16, 64})", 11,
           color="#1f4e79")
    # C: 64 x 64 -> 4 x 4 blocks (pm, pn)
    cx, cs = 750, 52
    s.text(cx + 2 * cs, ay - 14, "C  [64 x 64]  (output c, aie::OFM)", 13, weight="bold")
    for r in range(4):
        for c in range(4):
            hi = (r, c) == (pm_hi, pn_hi)
            s.rect(cx + c * cs, ay + r * cs, cs, cs, HI if hi else "#e2f0d9", GRID, sw=2 if hi else 1)
            s.text(cx + c * cs + cs / 2, ay + r * cs + cs / 2 + 4, f"({r},{c})", 11,
                   color="white" if hi else "#333")
    s.text(cx + 2 * cs, ay + 4 * cs + 18, "block (pm, pn) = 16 x 16", 11)
    s.text(cx + 2 * cs, ay + 4 * cs + 34, "store_buffer(c, {pm*16, pn*16}, {16, 16})", 11, color="#38761d")
    s.text(cx + 2 * cs, ay + 4 * cs + 50, "C(pm,pn) = sum over j of A(pm,j) x B(pn,j)^T", 11)
    s.save("gemm_tiling.svg")


def gemm_mesh():
    """The 4x4 mesh: which core gets which pid, and the derived DMA."""
    s = Svg(940, 720)
    s.text(470, 24, "The 4x4 mesh: program_id = position in the mesh, the DMA follows the blocks", 15,
           weight="bold")
    x0, y0, cs, gap = 260, 60, 100, 26

    def core_xy(pm, pn):  # mesh row pm sits at array row 3 + pm; row 3 is next to the shim (bottom)
        return x0 + pn * (cs + gap), y0 + (3 - pm) * (cs + gap)

    ys = y0 + 4 * (cs + gap) + 10           # shim row
    # streams first (drawn under the cores, visible in the gaps)
    for pm in range(4):
        y = core_xy(pm, 0)[1] + cs / 2
        s.line(x0 - 60, y, x0 + 4 * (cs + gap) - gap + 12, y, A_COL, 5)
        s.text(x0 - 66, y - 4, f"A rows {16 * pm}..{16 * pm + 15}", 11, "end", "#a0522d", "bold")
        s.text(x0 - 66, y + 11, f"from shim ({pm},0), MM2S ch1", 10, "end", "#a0522d")
    for pn in range(4):
        x = core_xy(0, pn)[0] + cs / 2
        s.line(x, ys, x, y0 - 6, B_COL, 5)
    for pm in range(4):
        for pn in range(4):
            x, y = core_xy(pm, pn)
            s.rect(x, y, cs, cs, "#ffffff", GRID, 1.5, rx=6)
            s.text(x + cs / 2, y + 22, f"pid ({pm},{pn})", 13, weight="bold")
            s.text(x + cs / 2, y + 40, f"tile ({3 + pn},{3 + pm})", 11, color="#666")
            s.text(x + cs / 2, y + 66, f"C rows {16 * pm}..{16 * pm + 15}", 10, color="#38761d")
            s.text(x + cs / 2, y + 82, f"C cols {16 * pn}..{16 * pn + 15}", 10, color="#38761d")
    for c in range(4):
        x = x0 + c * (cs + gap)
        s.rect(x, ys, cs, 46, "#eeeeee", GRID, 1.5, rx=4)
        s.text(x + cs / 2, ys + 18, f"shim ({c},0)", 11, weight="bold")
        s.text(x + cs / 2, ys + 34, f"ch0: B rows {16 * c}..", 10, color="#1f4e79")
    s.text(x0 - 66, ys + 18, "shim row (DDR side)", 11, "end", "#333", "bold")
    s.text(x0 - 66, ys + 34, "tile (col, row): array column", 10, "end", "#666")
    s.text(x0 - 66, ys + 47, "3..6 = partition {3, 6, ...}", 10, "end", "#666")
    ly = ys + 80
    s.text(470, ly, "orange = A: one stream per mesh ROW, broadcast to its 4 cores (the block depends only on pm)",
           12, color="#a0522d")
    s.text(470, ly + 18, "blue = B: one stream per mesh COLUMN, broadcast up it (the block depends only on pn)", 12,
           color="#1f4e79")
    s.text(470, ly + 36, "C: each core stores its own 16x16 block; the 4 cores of a mesh row are gathered back "
           "through one shim S2MM channel", 12, color="#38761d")
    s.save("gemm_mesh.svg")


def gemm_rounds():
    """One core over its 4 rounds: what each load_buffer brings, what the core keeps."""
    s = Svg(940, 312)
    s.text(470, 24, "Core (pm, pn): the K loop -- 4 rounds, each a {16, 64} block of A and of B", 15,
           weight="bold")
    x0, w, gap = 40, 200, 26
    for j in range(4):
        x = x0 + j * (w + gap)
        s.rect(x, 50, w, 170, "#fafafa", GRID, 1, rx=6)
        s.text(x + w / 2, 70, f"round j = {j}", 13, weight="bold")
        s.rect(x + 14, 84, 80, 40, A_COL, GRID)
        s.text(x + 54, 102, "x = A blk", 11)
        s.text(x + 54, 117, f"K {64 * j}..{64 * j + 63}", 10)
        s.rect(x + 106, 84, 80, 40, B_COL, GRID)
        s.text(x + 146, 102, "w = B blk", 11)
        s.text(x + 146, 117, f"K {64 * j}..{64 * j + 63}", 10)
        s.text(x + w / 2, 146, "acc[16x16] += x * w^T", 12, color="#38761d")
        s.text(x + w / 2, 166, "release(a); release(b)", 11, color="#666")
        s.text(x + w / 2, 182, "-> DMA refills that buffer", 10, color="#666")
        s.text(x + w / 2, 202, "with round j+1's block" if j < 3 else "(last round)", 10, color="#666")
        if j < 3:
            s.line(x + w + 2, 135, x + w + gap - 2, 135, GRID, 1.5, arrow=True)
    s.rect(x0, 236, 4 * w + 3 * gap, 62, "#e2f0d9", GRID, 1, rx=6)
    s.text(x0 + (4 * w + 3 * gap) / 2, 254, "before the loop:  y = store_buffer(c, {pm*16, pn*16}, {16, 16})", 11)
    s.text(x0 + (4 * w + 3 * gap) / 2, 270, "after the loop:  y[i] = (int8)acc[i];  commit(c)  -> the DMA sends the "
           "16x16 block home", 11)
    s.text(x0 + (4 * w + 3 * gap) / 2, 288, "acc stays in core data memory for the whole launch; "
           "only the 1 KB A / B windows are streamed (ping-pong)", 11, color="#666")
    s.save("gemm_rounds.svg")


# ---------------------------------------------------------------- CONV ----
def conv_tiling():
    """Output 112x112x64: OH bands per mesh row, OC groups per mesh column, 4x4 blocks per core."""
    s = Svg(1010, 520)
    s.text(505, 24, "simpleconv2dkernel_port_less.cc: output 112 x 112 x 64, how pid and (i, j) pick a block",
           15, weight="bold")
    x0, y0, sc = 100, 70, 3.4   # 112 px * 3.4 = 381
    side = 112 * sc
    s.text(x0 + side / 2, y0 - 12, "output plane  OH x OW = 112 x 112  (one OC group shown)", 12, weight="bold")
    for p in range(4):
        y = y0 + p * 28 * sc
        hi = p == 1
        s.rect(x0, y, side, 28 * sc, "#e2f0d9" if not hi else "#cfe8c2", GRID, 1.5)
        s.text(x0 - 8, y + 14 * sc + 4, f"pid_h={p}", 11, "end", HI if hi else "#333")
        s.text(x0 - 8, y + 14 * sc + 18, f"rows {28 * p}..{28 * p + 27}", 9, "end", "#666")
    # blocks of band 1
    for i in range(4):
        for j in range(4):
            x = x0 + j * 28 * sc
            y = y0 + 28 * sc + i * 7 * sc
            s.rect(x, y, 28 * sc, 7 * sc, HI if (i, j) == (2, 1) else "none", "#2d6a2d", 1, op=0.6)
            s.text(x + 14 * sc, y + 7 * sc / 2 + 4, f"i={i} j={j}", 9, color="white" if (i, j) == (2, 1) else "#2d6a2d")
    s.text(x0 + side / 2, y0 + side + 20, "each band: 4 x 4 blocks of 7 rows x 28 columns (i = row block, "
           "j = column block)", 11)
    s.text(x0 + side / 2, y0 + side + 36, "red block: oh0 = pid_h*28 + i*7 = 42, ow0 = j*28 = 28", 11, color=HI)
    # channel axis
    cx = 600
    s.text(cx + 150, y0 - 12, "output channels OC = 64", 12, weight="bold")
    for c in range(4):
        hi = c == 2
        s.rect(cx + c * 75, y0, 75, 40, B_COL if hi else "#dbe8f5", GRID, 1.5)
        s.text(cx + c * 75 + 37, y0 + 18, f"pid_c={c}", 11, color=HI if hi else "#333")
        s.text(cx + c * 75 + 37, y0 + 33, f"oc {16 * c}..{16 * c + 15}", 9, color="#333")
    s.text(cx + 150, y0 + 60, "mesh column pid_c owns 16 output channels,", 11)
    s.text(cx + 150, y0 + 76, "so it needs only filters 16*pid_c .. +16 (one weight block)", 11)
    # the store_buffer of the red block
    by = y0 + 120
    s.rect(cx - 15, by, 330, 150, "#fafafa", GRID, 1, rx=6)
    s.text(cx + 150, by + 22, "core (pid_h, pid_c) = (1, 2), round (i, j) = (2, 1)", 12, weight="bold")
    s.text(cx + 12, by + 48, "y = store_buffer(win_c,", 11, "start", "#38761d")
    s.text(cx + 24, by + 66, "{oh0, ow0, pid_c*16},  {7, 28, 16})", 11, "start", "#38761d")
    s.text(cx + 24, by + 84, "= {42, 28, 32}, a 7 x 28 x 16 block (3136 B)", 11, "start", "#38761d")
    s.text(cx + 12, by + 108, "x = load_buffer(win_a,", 11, "start", "#a0522d")
    s.text(cx + 24, by + 126, "{oh0*2, ow0*2, 0}, {19, 61, 4})", 11, "start", "#a0522d")
    s.text(cx + 24, by + 144, "= {84, 56, 0}: its input footprint (4636 B)", 11, "start", "#a0522d")
    s.text(cx + 150, by + 180, "16 cores x 16 rounds x 3136 B = all 802,816 outputs", 11, color="#666")
    s.save("conv_tiling.svg")


def conv_halo():
    """Input slabs of consecutive blocks overlap: the halo."""
    s = Svg(960, 360)
    s.text(480, 24, "Input footprint of an output block: 7x7 kernel, stride 2 -> overlapping slabs (halo)",
           15, weight="bold")
    s.text(480, 44, "input 230 x 230 x 4 (zero-padded in DDR), top-left corner -- mesh row pid_h = 0", 12,
           color="#555")
    x0, y0, sc = 110, 80, 4.2           # one input row / column = 4.2 px
    R, Cc = 40, 125                     # rows / cols drawn
    s.rect(x0, y0, Cc * sc, R * sc, "#fffaf4", GRID, 1)
    blocks = [((0, 0), "#f4a261", "round i=0, j=0"), ((14, 0), "#e76f51", "round i=1, j=0"),
              ((0, 56), "#2a9d8f", "round i=0, j=1")]
    for (r, c), col, _ in blocks:
        s.rect(x0 + c * sc, y0 + r * sc, 61 * sc, 19 * sc, col, "#333", 1.5, op=0.30)
    # halo strips, darker
    s.rect(x0, y0 + 14 * sc, 61 * sc, 5 * sc, "#8b0000", "none", 0, op=0.35)
    s.rect(x0 + 56 * sc, y0, 5 * sc, 19 * sc, "#8b0000", "none", 0, op=0.35)
    # labels in the non-overlapping parts
    s.text(x0 + 28 * sc, y0 + 8 * sc, "round i=0, j=0", 12, weight="bold")
    s.text(x0 + 28 * sc, y0 + 27 * sc, "round i=1, j=0", 12, weight="bold")
    s.text(x0 + 90 * sc, y0 + 8 * sc, "round i=0, j=1", 12, weight="bold")
    s.text(x0 + 28 * sc, y0 + 17 * sc + 4, "halo: rows 14..18", 10, color="white", weight="bold")
    s.text(x0 + 62 * sc + 4, y0 + 17 * sc + 4, "<- halo: cols 56..60", 10, "start", "#8b0000", "bold")
    # dimension lines
    s.line(x0 - 12, y0, x0 - 12, y0 + 19 * sc, "#333", 1.2)
    s.text(x0 - 16, y0 + 9.5 * sc - 4, "19 rows", 11, "end", weight="bold")
    s.text(x0 - 16, y0 + 9.5 * sc + 10, "(7-1)*2+7", 10, "end", "#666")
    s.line(x0 - 40, y0 + 14 * sc, x0 - 4, y0 + 14 * sc, "#333", 1, dash="3,2")
    s.text(x0 - 16, y0 + 14 * sc + 24, "next block", 10, "end", "#666")
    s.text(x0 - 16, y0 + 14 * sc + 37, "starts at row 14", 10, "end", "#666")
    s.line(x0, y0 + R * sc + 12, x0 + 61 * sc, y0 + R * sc + 12, "#333", 1.2)
    s.text(x0 + 30 * sc, y0 + R * sc + 28, "61 columns = (28-1)*2+7", 11, weight="bold")
    s.line(x0 + 56 * sc, y0 + R * sc + 4, x0 + 56 * sc, y0 + R * sc + 40, "#333", 1, dash="3,2")
    s.text(x0 + 56 * sc + 4, y0 + R * sc + 48, "next block starts at column 56", 10, "start", "#666")
    # explanation panel
    tx = x0 + Cc * sc + 24
    lines = [("block - step = halo", True, "#222"),
             ("rows:  19 - 14 = 5", False, "#222"),
             ("cols:  61 - 56 = 5", False, "#222"),
             ("", False, "#222"),
             ("step = output block * stride", True, "#222"),
             ("rows:  7 * 2 = 14", False, "#222"),
             ("cols:  28 * 2 = 56", False, "#222"),
             ("", False, "#222"),
             ("halo bytes are sent twice:", True, "#8b0000"),
             ("each round gets its whole slab", False, "#8b0000"),
             ("", False, "#222"),
             ("mesh row pid_h starts at", True, "#222"),
             ("input row 56 * pid_h", False, "#222")]
    for k, (t, bold, col) in enumerate(lines):
        s.text(tx, y0 + 6 + k * 17, t, 11, "start", col, "bold" if bold else "normal")
    s.text(480, 345, "The compiler reads exactly this from the trace: block (19,61,4), core offset (56,0,0)*p0, "
           "rounds 4x4, steps (14,0,0) / (0,56,0)", 11, color="#a0522d")
    s.save("conv_halo.svg")


def conv_mesh():
    s = Svg(940, 720)
    s.text(470, 24, "conv2d on the 4x4 mesh: mesh row = OH band, mesh column = OC group", 15, weight="bold")
    x0, y0, cs, gap = 260, 60, 100, 26

    def core_xy(ph, pc):
        return x0 + pc * (cs + gap), y0 + (3 - ph) * (cs + gap)

    ys = y0 + 4 * (cs + gap) + 10
    for ph in range(4):
        y = core_xy(ph, 0)[1] + cs / 2
        s.line(x0 - 60, y, x0 + 4 * (cs + gap) - gap + 12, y, A_COL, 5)
        s.text(x0 - 66, y - 4, f"input rows {56 * ph}..{56 * ph + 60}", 11, "end", "#a0522d", "bold")
        s.text(x0 - 66, y + 11, "16 slabs 19 x 61 x 4", 10, "end", "#a0522d")
    for pc in range(4):
        x = core_xy(0, pc)[0] + cs / 2
        s.line(x, ys, x, y0 - 6, B_COL, 5)
    for ph in range(4):
        for pc in range(4):
            x, y = core_xy(ph, pc)
            s.rect(x, y, cs, cs, "#ffffff", GRID, 1.5, rx=6)
            s.text(x + cs / 2, y + 22, f"pid ({ph},{pc})", 13, weight="bold")
            s.text(x + cs / 2, y + 44, f"oh {28 * ph}..{28 * ph + 27}", 10, color="#38761d")
            s.text(x + cs / 2, y + 60, f"oc {16 * pc}..{16 * pc + 15}", 10, color="#38761d")
            s.text(x + cs / 2, y + 82, "16 output blocks", 10, color="#666")
    for c in range(4):
        x = x0 + c * (cs + gap)
        s.rect(x, ys, cs, 46, "#eeeeee", GRID, 1.5, rx=4)
        s.text(x + cs / 2, ys + 18, f"shim ({c},0)", 11, weight="bold")
        s.text(x + cs / 2, ys + 34, f"filters {16 * c}..{16 * c + 15}", 10, color="#1f4e79")
    s.text(x0 - 66, ys + 26, "shim row (DDR side)", 11, "end", "#333", "bold")
    ly = ys + 80
    s.text(470, ly, "orange: input slabs, the same for every core of a mesh ROW (they depend only on pid_h)", 12,
           color="#a0522d")
    s.text(470, ly + 18, "blue: one 16x7x7x4 filter block (3136 B), the same for every core of a mesh COLUMN", 12,
           color="#1f4e79")
    s.text(470, ly + 36, "each core sends its own 7x28x16 output blocks to their place in the HWC output", 12,
           color="#38761d")
    s.save("conv_mesh.svg")


# ---------------------------------------------------------- SCHEDULING ----
def sched_timeline():
    s = Svg(1060, 400)
    s.text(530, 24, "Per-port buffering: what {.depth} and {.resident} change (one core, input port + weights)",
           15, weight="bold")
    x0, unit = 150, 58

    def bar(row_y, start, length, color, label):
        s.rect(x0 + start * unit, row_y, length * unit - 3, 24, color, "#333", 1, rx=3)
        s.text(x0 + start * unit + (length * unit - 3) / 2, row_y + 16, label, 10)

    # depth 1
    y = 60
    s.text(20, y + 12, "{.depth = 1}", 12, "start", weight="bold")
    s.text(20, y + 28, "one buffer", 10, "start", color="#666")
    for k in range(4):
        bar(y, k * 3, 1, A_COL, f"DMA x{k}")
        bar(y + 30, k * 3 + 1, 2, C_COL, f"compute {k}")
    s.text(x0 + 12 * unit + 10, y + 30, "DMA waits for release(),", 10, "start", "#666")
    s.text(x0 + 12 * unit + 10, y + 44, "core waits for the DMA", 10, "start", "#666")
    # depth 2
    y = 160
    s.text(20, y + 12, "{.depth = 2}", 12, "start", weight="bold")
    s.text(20, y + 28, "ping-pong", 10, "start", color="#666")
    for k in range(4):
        bar(y, k * 2, 1, A_COL if k % 2 == 0 else "#f9c89b", f"DMA x{k} {'ping' if k % 2 == 0 else 'pong'}")
        bar(y + 30, 1 + k * 2, 2, C_COL, f"compute {k}")
    s.text(x0 + 12 * unit + 10, y + 30, "the next block arrives", 10, "start", "#666")
    s.text(x0 + 12 * unit + 10, y + 44, "while the core computes", 10, "start", "#666")
    # resident
    y = 270
    s.text(20, y + 12, "weights", 12, "start", weight="bold")
    s.text(20, y + 28, "{.resident = true}", 10, "start", color="#666")
    bar(y, 0, 1, B_COL, "DMA w once")
    bar(y + 30, 1, 11, "#dbe8f5", "w held in ONE buffer for the whole launch (16 x 7 x 7 x 4 = 3136 B)")
    s.text(530, 380, "Illustrative timing (not to scale). The kernel code is identical in every case: "
           "options configure the DMA and the buffers, not the compute.", 11, color="#666")
    s.save("sched_timeline.svg")


# --------------------------------------------------------------- VECTOR ----
def vec_mmul():
    s = Svg(1000, 510)
    s.text(500, 24, "simpleconv2dkernel_port.cc: one aie::mmul<8,8,8> step = 8 output columns x 8 tap bytes x 8 "
           "channels", 14, weight="bold")
    # slab row
    x0, y0, bw = 40, 70, 3.5
    s.text(x0, y0 - 10, "one slab row  r = x + (oh*2 + kh) * 244   (61 input columns x 4 channels = 244 bytes)", 12,
           "start", weight="bold")
    s.rect(x0, y0, 244 * bw, 26, "#fff7ef", GRID)
    for col in range(0, 61, 2):
        s.line(x0 + col * 4 * bw, y0, x0 + col * 4 * bw, y0 + 26, "#ddd", 0.6)
    # output column ow reads bytes [8*ow, 8*ow + 28): stacked, they overlap by 20 bytes
    for k, (m, color) in enumerate([(0, "#f4a261"), (1, "#e9c46a"), (7, "#e76f51")]):
        yy = y0 + 34 + k * 16
        s.rect(x0 + 8 * m * bw, yy, 28 * bw, 12, color, "#333", 1, op=0.85)
        s.text(x0 + 8 * m * bw + 28 * bw + 6, yy + 10, f"output column ow={m}: bytes {8 * m}..{8 * m + 27}", 10,
               "start")
    s.text(x0 + 470, y0 + 50, "(7 taps x 4 channels = 28 bytes; the next column starts 2 px x 4 ch = 8 bytes later)",
           10, "start", "#666")
    # the 64-byte unaligned load = 8 rows of 8 bytes
    y1 = y0 + 110
    s.text(x0, y1 - 8, "aie::load_unaligned_v<64>(r + owA*8 + off, 4): 64 consecutive bytes", 12, "start",
           weight="bold")
    s.rect(x0, y1, 64 * bw, 20, A_COL, "#333", 1, op=0.8)
    for m in range(8):
        s.line(x0 + 8 * m * bw, y1, x0 + 8 * m * bw, y1 + 20, "#333", 0.8)
        s.text(x0 + 8 * m * bw + 4 * bw, y1 + 14, f"ow{m}", 9)
    s.text(x0 + 64 * bw + 10, y1 + 14, "= an 8x8 A matrix: row m = output column owA+m, 8 tap bytes of it", 10,
           "start")
    s.text(x0 + 64 * bw + 10, y1 + 30, "(stride 2 x 4 channels = 8 bytes = exactly one A row: im2col for free)",
           10, "start", "#666")
    # K chunks
    y2 = y1 + 74
    s.text(x0, y2 - 8, "K: the 28 bytes of a kernel row in 4 chunks of 8 (off = 0, 8, 16, 20)", 12, "start",
           weight="bold")
    for kb, off in enumerate([0, 8, 16, 20]):
        s.rect(x0 + off * 9, y2 + kb * 18, 8 * 9, 14, "#cfe2f3" if kb < 3 else "#f4cccc", "#333", 1)
        s.text(x0 + off * 9 + 36, y2 + kb * 18 + 11, f"kb={kb}", 9)
    s.rect(x0 + 20 * 9, y2 + 3 * 18, 4 * 9, 14, "#999999", "#333", 1, op=0.8)
    s.text(x0 + 28 * 9 + 14, y2 + 3 * 18 + 11, "bytes 20..23 overlap kb=2: their weights in Bp are 0", 10, "start",
           "#990000")
    for b in range(0, 29, 4):
        s.text(x0 + b * 9, y2 + 4 * 18 + 12, str(b), 8)
    # mmul
    y3 = y2 + 120
    s.text(x0, y3 - 8, "per kernel row kh (7) and chunk kb (4):  4 accumulators", 12, "start", weight="bold")
    s.rect(x0, y3, 90, 90, A_COL, "#333", 1, op=0.8)
    s.text(x0 + 45, y3 + 40, "aA", 12, weight="bold")
    s.text(x0 + 45, y3 + 58, "8 ow x 8 K", 10)
    s.text(x0 + 110, y3 + 50, "x", 16)
    s.rect(x0 + 130, y3, 90, 90, B_COL, "#333", 1, op=0.8)
    s.text(x0 + 175, y3 + 40, "b0 / b1", 12, weight="bold")
    s.text(x0 + 175, y3 + 58, "8 K x 8 oc", 10)
    s.text(x0 + 240, y3 + 50, "->", 16)
    s.rect(x0 + 270, y3, 90, 90, C_COL, "#333", 1, op=0.8)
    s.text(x0 + 315, y3 + 40, "cA0 / cA1", 12, weight="bold")
    s.text(x0 + 315, y3 + 58, "8 ow x 8 oc", 10)
    notes = ["cA0 += aA * b0   (ow owA..+7,  oc 0..7)",
             "cA1 += aA * b1   (ow owA..+7,  oc 8..15)",
             "cB0 += aB * b0   (ow owB..+7,  oc 0..7)",
             "cB1 += aB * b1   (ow owB..+7,  oc 8..15)",
             "pass 0: owA = 0, owB = 8;  pass 1: owA = 16, owB = 20",
             "(28 columns = 0..7, 8..15, 16..23, 20..27; 20..23 written twice, same values)",
             "7 kh x 4 kb = 28 mmul steps per accumulator: K = 224 (196 real taps)"]
    for k, n in enumerate(notes):
        s.text(x0 + 390, y3 + 10 + k * 17, n, 11, "start", "#333")
    s.save("vec_mmul.svg")


# ------------------------------------------------------------- COMPILER ----
def trace_to_dma():
    """Kernel trace -> affine fit -> DMA pattern, for A of the GEMM."""
    s = Svg(1060, 430)
    s.text(530, 24, "Step 1: from load_buffer calls to a DMA pattern (matrix A of the GEMM)", 15, weight="bold")
    cols = [(20, "1. run the kernel on x86 for every core", "(trace)"),
            (370, "2. fit each port's blocks to one formula", "(affine fit)"),
            (720, "3. that formula IS a DMA buffer descriptor", "(DMA pattern)")]
    for x, t1, t2 in cols:
        s.rect(x, 44, 320, 330, "#fafafa", GRID, 1, rx=8)
        s.text(x + 160, 66, t1, 12, weight="bold")
        s.text(x + 160, 82, t2, 11, color="#666")
    for x in (346, 696):
        s.line(x, 210, x + 18, 210, GRID, 2, arrow=True)
    # column 1: recorded calls of two cores
    x = 34
    s.text(x, 108, "core (pm=1, pn=2) records:", 11, "start", weight="bold")
    for j in range(4):
        s.text(x + 8, 128 + j * 17, f"load a  offset ({16},{64 * j:>3})  shape (16,64)", 10, "start", "#a0522d")
    s.text(x, 210, "core (pm=2, pn=0) records:", 11, "start", weight="bold")
    for j in range(4):
        s.text(x + 8, 230 + j * 17, f"load a  offset ({32},{64 * j:>3})  shape (16,64)", 10, "start", "#a0522d")
    s.text(x, 316, "... 16 cores x 4 rounds, plus every", 10, "start", "#666")
    s.text(x, 330, "store, release and commit", 10, "start", "#666")
    s.text(x, 352, "stubs: aie::load_buffer records,", 10, "start", "#666")
    s.text(x, 366, "ASan checks every byte the kernel reads", 10, "start", "#666")
    # column 2: the fit
    x = 384
    s.text(x, 112, "offset = base_p0 * pm + base_p1 * pn", 11, "start", weight="bold")
    s.text(x, 128, "       + step_i * i + step_j * j", 11, "start", weight="bold")
    rows = [("block", "(16, 64)"), ("base_p0", "(16, 0)    moves with pm"), ("base_p1", "(0, 0)     same for all pn"),
            ("rounds", "1 x 4"), ("step_j", "(0, 64)    the K loop")]
    for k, (a, b) in enumerate(rows):
        s.text(x + 6, 156 + k * 18, a, 11, "start", "#a0522d", "bold")
        s.text(x + 76, 156 + k * 18, b, 11, "start")
    s.text(x, 262, "=> A depends on pm only:", 11, "start", weight="bold")
    s.text(x, 278, "   ROW BROADCAST", 11, "start", "#a0522d", "bold")
    s.text(x, 300, "B: base_p1 = (16, 0) -> COLUMN broadcast", 10, "start", "#1f4e79")
    s.text(x, 316, "C: moves with pm and pn -> GATHER", 10, "start", "#38761d")
    s.text(x, 344, "not affine / not one shape -> build error", 10, "start", "#990000")
    s.text(x, 358, "(the DMA could not stream it)", 10, "start", "#990000")
    # column 3: the DMA pattern
    x = 734
    s.text(x, 112, "shim MM2S descriptor, mesh row pm:", 11, "start", weight="bold")
    dims = [("start", "A + 4096 * pm   (16 rows x 256 B)"), ("d0", "16 words  = 64 B of one row"),
            ("d1", "16 rows, pitch 256 B"), ("d2", "4 chunks, step 64 B   (j)"), ("length", "1024 B per round, 4 rounds")]
    for k, (a, b) in enumerate(dims):
        s.text(x + 6, 136 + k * 18, a, 11, "start", "#a0522d", "bold")
        s.text(x + 60, 136 + k * 18, b, 11, "start")
    s.text(x, 240, "routed to all 4 cores of mesh row pm", 11, "start", weight="bold")
    s.text(x, 258, "core side: 1 KB ping + 1 KB pong window,", 10, "start")
    s.text(x, 272, "a lock pair; load_buffer = acquire,", 10, "start")
    s.text(x, 286, "release = give the buffer back", 10, "start")
    s.text(x, 316, "written to aout/newfile.cpp as", 10, "start", "#666")
    s.text(x, 330, "constexpr aie::GemmSpace __aie_gemm_IFM", 10, "start", "#666")
    s.text(x, 344, "  .d1 = {64, tile_round 4, tile 16, stride 16}", 10, "start", "#666")
    s.text(x, 358, "  .d2 = {256, tile_round 4, tile 64, stride 64}", 10, "start", "#666")
    s.text(530, 405, "Nothing here knows it is a GEMM: the same three steps route the conv2d of tutorial 2.", 12,
           color="#333")
    s.save("trace_to_dma.svg")


def compiler_flow():
    """aiehlc frontend + the TilingLinalg MLIR pipeline down to host.cc / kernel.cc / routing.cc."""
    s = Svg(1100, 760)
    s.text(550, 24, "Step 2: the pipeline from the kernel's blocks to host.cc, kernel.cc and routing.cc", 15,
           weight="bold")
    s.text(550, 42, "every stage dumps its IR: ir/dfschedule/N_<Pass>.mlir and ir/simplerouting/N_<Pass>.mlir "
           "(numbers below)", 11, color="#666")

    def box(x, y, w, h, title, lines, fill):
        s.rect(x, y, w, h, fill, "#333", 1.3, rx=8)
        s.text(x + w / 2, y + 18, title, 12, weight="bold")
        for k, l in enumerate(lines):
            s.text(x + w / 2, y + 35 + k * 14, l, 10, color="#333")

    FE, IR, PH, OUT = "#fff2cc", "#dbe8f5", "#e2f0d9", "#f4cccc"
    # frontend lane
    s.text(20, 76, "aiehlc frontend (src/llvm/aiehlc.cc, convautotile/)", 11, "start", "#7f6000", "bold")
    box(20, 86, 230, 74, "your .cc", ["__global__ kernel with ports,", "load_buffer / store_buffer,",
                                      "main() with <<<mesh>>>"], "#ffffff")
    box(290, 86, 250, 74, "kernel analysis (trace + fit)", ["run on x86 for all 16 cores,",
                                                           "fit blocks, check, plan:", "row / col broadcast, gather"], FE)
    box(580, 86, 230, 74, "explicit descriptors", ["aout/newfile.cpp:", "aie::GemmSpace per port",
                                                   "+ #routing.kernel_block"], FE)
    box(850, 86, 230, 74, "Clang AST -> routing IR", ["mesh, tensors, partitions,", "tiling levels,",
                                                      "IO per mesh row / column"], FE)
    for x1, x2 in ((250, 290), (540, 580), (810, 850)):
        s.line(x1 + 2, 123, x2 - 2, 123, "#333", 1.5, arrow=True)
    s.line(965, 160, 965, 196, "#333", 1.5, arrow=True)
    # shared MLIR lane
    s.text(20, 192, "TilingLinalg MLIR pipeline (src/mlir/mlirfront/tilinglinalg/) -- shared stages", 11, "start",
           "#1f4e79", "bold")
    shared = [(850, "routing IR", ["0_initial.mlir", "createhwmesh, partitiontensor,", "movedatabyio (abstract tiles)"]),
              (580, "routing unrolled", ["1_RoutingUnrollingLowerPass", "one op per mesh row / col", "(scf loops unrolled)"]),
              (310, "dmap  (logical dataflow)", ["2_RoutingToDmapPass", "io engine -> core group,", "stream, push / pull"]),
              (40, "dmaphop  (physical hops)", ["3_DmapToDmaphopPass", "tiles, DMA ports + channels,", "hop path shim->core->core"])]
    for x, t, ls in shared:
        box(x, 202, 230, 80, t, ls, IR)
    for x1, x2 in ((850, 810), (580, 540), (310, 270)):
        s.line(x1 - 2, 242, x2 + 2, 242, "#333", 1.5, arrow=True)
    # three paths from dmaphop
    s.text(300, 322, "the module is cloned: three paths read the same dmaphop IR", 11, "start", "#1f4e79", "bold")
    s.line(155, 282, 155, 336, "#333", 1.5, arrow=True)          # -> blueprint
    s.line(155, 300, 875, 300, "#333", 1.5)                       # -> routing path
    s.line(875, 300, 875, 336, "#333", 1.5, arrow=True)
    box(40, 338, 230, 80, "dfscheblueprint", ["5_DmaphopTodfscheblueprintPass", "per flow: data slices,",
                                               "tile groups, flow configs"], IR)
    box(760, 338, 230, 80, "routinghw  (stream switches)", ["simplerouting/1_DmaphopToRoutinghw",
                                                            "switch port connections,", "packet flows, shim mux"], PH)
    # host and kernel from blueprint
    s.line(155, 418, 155, 450, "#333", 1.5)
    s.line(155, 450, 520, 450, "#333", 1.5)
    s.line(155, 450, 155, 470, "#333", 1.5, arrow=True)
    s.line(520, 450, 520, 470, "#333", 1.5, arrow=True)
    box(40, 472, 230, 94, "dfschedule  (host)", ["6_BlueprintToSchedulePass", "DMA BDs, locks, IOs, events",
                                                  "7..11 canonicalize, merge waits", "12_DfscheduleToApiPass"], PH)
    box(405, 472, 230, 94, "dfschedule  (kernel)", ["18_BlueprintToScheduleKernelPass", "core buffers ping / pong,",
                                                     "locks, window_init", "19_DfscheduleToKernelApiPass"], PH)
    s.text(338, 440, "both lower the SAME core tiles: shared helper/", 10, color="#666")
    s.line(875, 418, 875, 472, "#333", 1.5, arrow=True)
    box(760, 472, 230, 94, "routinghw lowered", ["2_RoutingHWVerifyPass", "5_RoutingHWLowerPass",
                                                 "-> XAie_StrmConn*,", "XAie_Enable*ShimPort calls"], PH)
    # outputs
    for x, name, lines in ((40, "host.cc", ["runs on the Arm: program BDs,", "load + start cores, wait"]),
                           (405, "kernel.cc", ["runs on each core: windows,", "then your kernel function"]),
                           (760, "routing.cc", ["stream-switch setup,", "run at partition init"])):
        s.line(x + 115, 566, x + 115, 602, "#333", 1.5, arrow=True)
        box(x, 604, 230, 66, name, lines, OUT)
    s.text(550, 700, "EmitC turns the last IR of each path into C++; aiehlc.sh compiles host.cc + routing.cc for "
           "the Arm (main.elf)", 11, color="#333")
    s.text(550, 718, "and kernel.cc with the AIE compiler (embedded in main.elf). flowcheck re-checks routing.cc + "
           "host.cc against the dmaphop flows.", 11, color="#333")
    s.save("compiler_flow.svg")


if __name__ == "__main__":
    gemm_tiling()
    gemm_mesh()
    gemm_rounds()
    conv_tiling()
    conv_halo()
    conv_mesh()
    sched_timeline()
    vec_mmul()
    trace_to_dma()
    compiler_flow()
