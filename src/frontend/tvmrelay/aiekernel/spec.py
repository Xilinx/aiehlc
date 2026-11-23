###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Read a user kernel source: splice its ``.cc`` includes, parse its ``AIEK_SPEC``.

The source is read the way aiehlc reads it
------------------------------------------
``src/llvm/aiehlc.cc:inlineSourceIncludes`` splices every quoted ``.cc`` /
``.cpp`` / ``.cxx`` include into the entry text *before* any rewrite runs, and
resolves it against the ENTRY file's directory (not the including file's).
:func:`load_source` does the same, so the lint rules see exactly the text the
aiehlc frontend sees -- a user kernel that is a thin wrapper around
``#include "conv2dstem.cc"`` is checked as the whole thing. Every line keeps its
origin (file, line) so a finding points at the file the user has to edit.

The spec grammar
----------------
``AIEK_SPEC(entry, "..." "...")`` holds one directive per line (``#`` starts a
comment)::

    kernel    <display name>              optional, default = entry
    op        <family label>              informational (conv2d, matmul, ...)
    match     <substring>                 repeatable; every one must occur in the
                                          TVM kernel symbol for a layer to match
    reference <file.c>                    x86 reference defining <entry>_ref,
                                          relative to the file holding the spec
    export    <symbol>                    keep this global in the archive (the
                                          rest of the user's symbols are made
                                          local so two kernels can share an ELF)
    target    <board> ...                 boards the AIE body is built for
                                          (vek280, vek385); aiednn only links a
                                          kernel on a board it names
    in|param|out <dtype> [d0,d1,...] [name] [key=value ...]

Argument lines are TVM call order: every input as the graph passes it
(activations and constants interleaved), then the outputs. ``param`` = a graph
constant (weights, bias, quant params), ``in`` = an activation. Dims are an
integer, ``*`` (anything), or a NAME bound on first use and required to agree
everywhere after (``[1,CI,H,W,4]`` ... ``[CO,CI,3,3,4,4]``). Keys:

    require=zero|nonzero|uniform|nonneg|range(lo,hi)   param values (repeatable)
    tol=N                                               max |ref - tvm| on an out

Nothing here imports TVM or numpy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SourceText", "ArgSpec", "KernelSpec", "SpecError", "load_source",
           "BufSpec", "ConstSpec", "StepSpec", "parse_steps", "parse_step_text",
           "parse_specs", "DTYPES", "strip_comments_and_strings"]

#: dtype name -> (DLPack code, bits). Code 0 int, 1 uint, 2 float.
DTYPES = {
    "int8": (0, 8), "int16": (0, 16), "int32": (0, 32), "int64": (0, 64),
    "uint8": (1, 8), "uint16": (1, 16), "uint32": (1, 32), "uint64": (1, 64),
    "float16": (2, 16), "float32": (2, 32), "float64": (2, 64),
}

_KINDS = ("in", "param", "out")
_REQUIRES = ("zero", "nonzero", "uniform", "nonneg", "any")

#: aiehlc's own include regex (aiehlc.cc), so the splice decisions agree.
_INC_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"[ \t]*(//[^\n]*)?$', re.M)
_SOURCE_EXT = (".cc", ".cpp", ".cxx")


class SpecError(ValueError):
    """A malformed ``AIEK_SPEC``. Carries the origin so lint can point at it."""

    def __init__(self, msg: str, where: str = ""):
        super().__init__(f"{where}: {msg}" if where else msg)
        self.where = where


# ═══════════════════════════════════════════════════════════════════════════
#  Source text with origins
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SourceText:
    """The spliced text of one entry file, plus where every line came from."""

    path: Path
    text: str
    origins: list                     # per output line: (Path, 1-based line)
    included: list                    # .cc files spliced in, in order
    missing: list                     # quoted .cc includes that did not resolve
    _starts: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._starts = [0]
        for m in re.finditer("\n", self.text):
            self._starts.append(m.end())

    def line_of(self, offset: int) -> int:
        """0-based line index of a character offset into :attr:`text`."""
        lo, hi = 0, len(self._starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def where(self, offset: int) -> str:
        """``file.cc:123`` for a character offset -- the ORIGINAL file and line."""
        path, line = self.origins[min(self.line_of(offset), len(self.origins) - 1)]
        return f"{path.name}:{line}"


def load_source(path) -> SourceText:
    """Read *path* and splice its quoted ``.cc`` includes exactly as aiehlc does."""
    path = Path(path).resolve()
    base = path.parent
    seen: set = set()
    included, missing = [], []

    def splice(p: Path, depth: int) -> list:
        lines = p.read_text(errors="replace").split("\n")
        out = []
        for no, line in enumerate(lines, 1):
            m = _INC_RE.match(line)
            name = m.group(1) if m else ""
            if not m or not name.endswith(_SOURCE_EXT) or depth > 8:
                out.append((line, p, no))
                continue
            # aiehlc resolves against the ENTRY directory at every depth.
            target = Path(name) if name.startswith("/") else base / name
            target = target.resolve()
            if target in seen:
                continue                      # aiehlc drops repeats
            if not target.is_file():
                missing.append((name, p, no))
                out.append((line, p, no))
                continue
            seen.add(target)
            included.append(target)
            out.extend(splice(target, depth + 1))
        return out

    rows = splice(path, 0)
    return SourceText(path=path, text="\n".join(r[0] for r in rows),
                      origins=[(r[1], r[2]) for r in rows],
                      included=included, missing=missing)


def strip_comments_and_strings(text: str, keep_strings: bool = False) -> str:
    """*text* with comments (and, unless *keep_strings*, string/char literals)
    blanked to spaces. Offsets and newlines are preserved, so a match in the
    result is a match at the same offset in the original."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = [ch if ch == "\n" else " " for ch in text[i:j]]
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            if not keep_strings:
                out[i + 1:j - 1] = " " * max(0, j - 1 - (i + 1))
            i = j
        else:
            i += 1
    return "".join(out)


# ═══════════════════════════════════════════════════════════════════════════
#  Spec model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ArgSpec:
    """One argument line of the spec."""

    index: int
    kind: str                 # in | param | out
    dtype: str
    dims: tuple               # int | "*" | NAME per dim
    name: str = ""
    requires: tuple = ()      # ("zero",), ("range", lo, hi), ...
    tol: int = 0

    @property
    def code_bits(self) -> tuple:
        return DTYPES[self.dtype]

    def shape_text(self) -> str:
        return "[" + ",".join(str(d) for d in self.dims) + "]"

    def __str__(self) -> str:
        return f"{self.kind} {self.dtype}{self.shape_text()} {self.name}".rstrip()


@dataclass
class KernelSpec:
    """Everything one ``AIEK_SPEC`` says, plus where it was found."""

    entry: str
    source: Path              # the entry file the user passed
    spec_file: Path           # the file the AIEK_SPEC text physically sits in
    where: str
    name: str = ""
    op: str = ""
    match: tuple = ()
    reference: Path = None
    exports: tuple = ()
    targets: tuple = ()       # `target` lines: boards the AIE body runs on
    mesh: tuple = ()          # `mesh` line: (arg name, dim, coef, "rows"|"cols"|"cores") scaled per mesh
    modes: tuple = ()         # `modes` line: the kernel source's optional modes this entry needs
                              # (AIEK_MODE_<X> in aie_modes_cfg.h); a per-app build compiles only those
    args: list = field(default_factory=list)
    raw: str = ""

    @property
    def inputs(self) -> list:
        return [a for a in self.args if a.kind != "out"]

    @property
    def outputs(self) -> list:
        return [a for a in self.args if a.kind == "out"]

    @property
    def symbolic(self) -> bool:
        """True when any dim is ``*`` or a name -- one spec, many geometries."""
        return any(not isinstance(d, int) for a in self.args for d in a.dims)

    def describe(self) -> str:
        lines = [f"{self.name} (entry {self.entry}, op {self.op or '?'})"]
        lines += [f"  arg {a.index}: {a}" for a in self.args]
        return "\n".join(lines)


_SPEC_CALL = re.compile(r"\bAIEK_SPEC\s*\(\s*([A-Za-z_]\w*)\s*,")
_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'", "0": "\0", "r": ""}


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(1)), s)


def _spec_body(text: str, start: int) -> tuple:
    """String-literal payload of the macro call starting at *start*.

    Returns ``(payload, end_offset)``. Only string literals and comments may
    sit between the comma and the closing paren; anything else is an error,
    because the Python side would silently ignore it while the compiler did
    not.
    """
    clean = strip_comments_and_strings(text, keep_strings=True)
    i, parts = start, []
    while i < len(text):
        c = clean[i]
        if c.isspace():
            i += 1
        elif c == '"':
            m = _STRING.match(text, i)
            if not m:
                raise SpecError("unterminated string in AIEK_SPEC")
            parts.append(_unescape(m.group(1)))
            i = m.end()
        elif c == ")":
            return "".join(parts), i + 1
        else:
            raise SpecError(f"unexpected {text[i:i + 12]!r} in AIEK_SPEC "
                            "(only adjacent string literals are allowed)")
    raise SpecError("AIEK_SPEC( has no closing parenthesis")


def _parse_dims(token: str, where: str) -> tuple:
    if not (token.startswith("[") and token.endswith("]")):
        raise SpecError(f"shape {token!r} must be written [d0,d1,...]", where)
    out = []
    for d in token[1:-1].split(","):
        d = d.strip()
        if not d:
            raise SpecError(f"empty dim in {token!r}", where)
        if d == "*" or re.fullmatch(r"[A-Za-z_]\w*", d):
            out.append(d)
        elif re.fullmatch(r"\d+", d):
            out.append(int(d))
        else:
            raise SpecError(f"dim {d!r} is not an integer, '*' or a NAME", where)
    return tuple(out)


def _parse_require(val: str, where: str) -> tuple:
    m = re.fullmatch(r"range\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", val)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            raise SpecError(f"require={val}: lo > hi", where)
        return ("range", lo, hi)
    if val not in _REQUIRES:
        raise SpecError(f"require={val!r}; want one of {', '.join(_REQUIRES)}, "
                        "range(lo,hi)", where)
    return (val,)


def _parse_arg(index: int, toks: list, where: str) -> ArgSpec:
    """``in|param|out <dtype> [dims] [name] [k=v ...]`` -- the shape may contain
    spaces, so it is re-joined from the bracket tokens first."""
    kind = toks[0]
    rest = " ".join(toks[1:])
    m = re.match(r"\s*(\w+)\s*(\[[^\]]*\])\s*(.*)$", rest)
    if not m:
        raise SpecError(f"want '{kind} <dtype> [dims] [name] [key=value...]'", where)
    dtype, shape, tail = m.group(1), m.group(2), m.group(3).split()
    if dtype not in DTYPES:
        raise SpecError(f"dtype {dtype!r}; want one of {', '.join(DTYPES)}", where)
    arg = ArgSpec(index=index, kind=kind, dtype=dtype,
                  dims=_parse_dims(shape.replace(" ", ""), where))
    reqs = []
    for tok in tail:
        if "=" not in tok:
            if arg.name:
                raise SpecError(f"two names ({arg.name!r}, {tok!r})", where)
            arg.name = tok
            continue
        key, val = tok.split("=", 1)
        if key == "require":
            if kind != "param":
                raise SpecError("require= only applies to a param (its values "
                                "are known before the run)", where)
            reqs.append(_parse_require(val, where))
        elif key == "tol":
            if kind != "out" or not val.isdigit():
                raise SpecError("tol=N (non-negative integer) only applies to an out", where)
            arg.tol = int(val)
        else:
            raise SpecError(f"unknown key {key!r} (require=, tol=)", where)
    arg.requires = tuple(r for r in reqs if r != ("any",))
    return arg


def _parse_mesh(tok: str, where: str) -> tuple:
    """``name[dim]=[coef*]rows|cols|cores``: that dim is coef x the mesh's rows / cols /
    rows*cols (a kernel whose block grows with the array, gemm_uk.cc); the literal dim in
    the arg line is its value on the default 4x4 mesh"""
    m = re.fullmatch(r"(\w+)\[(\d+)\]=(?:(\d+)\*)?(rows|cols|cores)", tok)
    if not m:
        raise SpecError(f"mesh {tok!r}; want name[dim]=[coef*]rows|cols|cores", where)
    return (m.group(1), int(m.group(2)), int(m.group(3) or 1), m.group(4))


MESH_DEFAULT = (4, 4)


def mesh_dim(coef: int, unit: str, rows: int, cols: int) -> int:
    return coef * {"rows": rows, "cols": cols, "cores": rows * cols}[unit]


def _check_mesh(spec: KernelSpec) -> None:
    named = {a.name: a for a in spec.args if a.name}
    for name, dim, coef, unit in spec.mesh:
        a = named.get(name)
        if a is None or dim >= len(a.dims):
            raise SpecError(f"mesh {name}[{dim}]: no such argument dim", spec.where)
        want = mesh_dim(coef, unit, *MESH_DEFAULT)
        if a.dims[dim] != want:
            raise SpecError(f"mesh {name}[{dim}]={coef}*{unit}: the literal dim is {a.dims[dim]}, "
                            f"but on the default {MESH_DEFAULT[0]}x{MESH_DEFAULT[1]} mesh it is {want}",
                            spec.where)


def _parse_text(spec: KernelSpec, payload: str) -> None:
    """Fill *spec* from the decoded payload. Raises :class:`SpecError`."""
    for lineno, line in enumerate(payload.split("\n"), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{spec.where} (spec line {lineno})"
        toks = line.split()
        key = toks[0]
        if key in _KINDS:
            spec.args.append(_parse_arg(len(spec.args), toks, where))
        elif key in ("kernel", "op") and len(toks) == 2:
            setattr(spec, "name" if key == "kernel" else "op", toks[1])
        elif key == "match" and len(toks) >= 2:
            spec.match += tuple(toks[1:])
        elif key == "export" and len(toks) >= 2:
            spec.exports += tuple(toks[1:])
        elif key == "target" and len(toks) >= 2:
            spec.targets += tuple(toks[1:])
        elif key == "mesh" and len(toks) >= 2:
            spec.mesh += tuple(_parse_mesh(t, where) for t in toks[1:])
        elif key == "modes" and len(toks) >= 2:
            bad = [t for t in toks[1:] if not re.fullmatch(r"[A-Z][A-Z0-9_]*", t)]
            if bad:
                raise SpecError(f"modes {' '.join(bad)}: want upper-case names (AIEK_MODE_<NAME>)", where)
            spec.modes += tuple(toks[1:])
        elif key == "reference" and len(toks) == 2:
            spec.reference = (spec.spec_file.parent / toks[1]).resolve()
        else:
            raise SpecError(f"cannot read {line!r}", where)
    if not spec.args:
        raise SpecError("no in/param/out lines", spec.where)
    if not spec.outputs:
        raise SpecError("no `out` argument -- the kernel must produce something",
                        spec.where)
    _check_mesh(spec)
    if spec.args[-1].kind != "out" or any(
            a.kind == "out" for a in spec.args[:len(spec.inputs)]):
        raise SpecError("outputs must come last (TVM passes inputs, then outputs)",
                        spec.where)


def parse_specs(src: SourceText) -> list:
    """Every ``AIEK_SPEC`` in *src*, parsed. Raises :class:`SpecError` on the
    first malformed one -- a half-read spec would match the wrong layers."""
    clean = strip_comments_and_strings(src.text)
    specs = []
    for m in _SPEC_CALL.finditer(clean):
        if clean[max(0, m.start() - 8):m.start()].rstrip().endswith("define"):
            continue                                   # the macro definition itself
        where = src.where(m.start())
        payload, _ = _spec_body(src.text, m.end())
        spec_file = src.origins[src.line_of(m.start())][0]
        spec = KernelSpec(entry=m.group(1), source=src.path, spec_file=spec_file,
                          where=where, raw=payload)
        try:
            _parse_text(spec, payload)
        except SpecError as exc:
            raise exc if exc.where else SpecError(str(exc), where)
        spec.name = spec.name or spec.entry
        specs.append(spec)
    return specs


# ═══════════════════════════════════════════════════════════════════════════
#  AIEK_STEP: the group-offload step ABI (include/aie_user_kernel.h)
# ═══════════════════════════════════════════════════════════════════════════
#
#   AIEK_STEP(<entry>, reads(<bufs>), writes(<bufs>), consts(<consts>));
#
#   <bufs>   = N                     N unnamed buffers (0 allowed)
#            | item {, item}         item = NAME[:qual ...], qual = dense | <dtype>
#   <consts> = (empty) | NAME[:int|:float] {, NAME[:int|:float]}
#
# Unlike AIEK_SPEC the clauses are C tokens, not string literals (the macro
# stringifies them), so they are read from the comment-stripped text.

_STEP_CALL = re.compile(r"\bAIEK_STEP\s*\(")
_STEP_CLAUSES = ("reads", "writes", "consts")
CONST_TYPES = ("int", "float")


@dataclass
class BufSpec:
    """One aiek_buf of a step: ``in[index]`` (reads) or ``out[index]`` (writes)."""

    index: int
    name: str = ""
    dtype: str = ""            # "" = any
    dense: bool = False        # True: the step ignores ld

    def __str__(self) -> str:
        quals = ([self.dtype] if self.dtype else []) + (["dense"] if self.dense else [])
        return (self.name or f"#{self.index}") + "".join(":" + q for q in quals)


@dataclass
class ConstSpec:
    index: int
    name: str
    type: str = "int"


@dataclass
class StepSpec:
    """Everything one ``AIEK_STEP`` declares."""

    entry: str
    source: Path
    where: str
    reads: list = field(default_factory=list)
    writes: list = field(default_factory=list)
    consts: list = field(default_factory=list)
    raw: str = ""

    @property
    def const_names(self) -> tuple:
        return tuple(c.name for c in self.consts)

    def describe(self) -> str:
        return (f"AIEK_STEP({self.entry}, reads({', '.join(map(str, self.reads)) or 0}), "
                f"writes({', '.join(map(str, self.writes)) or 0}), "
                f"consts({', '.join(c.name + ':' + c.type for c in self.consts)}))")


def _split_top(text: str) -> list:
    """Split on commas at paren depth 0."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [s.strip() for s in out]


def _parse_bufs(body: str, where: str, what: str) -> list:
    body = body.strip()
    if re.fullmatch(r"\d+", body):
        return [BufSpec(index=i) for i in range(int(body))]
    if not body:
        raise SpecError(f"{what}() is empty; write {what}(0) for none", where)
    bufs, names = [], set()
    for i, item in enumerate(_split_top(body)):
        parts = [p.strip() for p in item.split(":")]
        if not re.fullmatch(r"[A-Za-z_]\w*", parts[0]):
            raise SpecError(f"{what}: {item!r} is not NAME[:dense][:dtype] or a count", where)
        if parts[0] in names:
            raise SpecError(f"{what}: buffer {parts[0]!r} declared twice", where)
        names.add(parts[0])
        b = BufSpec(index=i, name=parts[0])
        for q in parts[1:]:
            if q == "dense":
                b.dense = True
            elif q in DTYPES:
                if b.dtype:
                    raise SpecError(f"{what}: {item!r} has two dtypes", where)
                b.dtype = q
            else:
                raise SpecError(f"{what}: qualifier {q!r} in {item!r}; want dense or one of "
                                f"{', '.join(DTYPES)}", where)
        bufs.append(b)
    return bufs


def _parse_consts(body: str, where: str) -> list:
    body = body.strip()
    if not body:
        return []
    out, names = [], set()
    for i, item in enumerate(_split_top(body)):
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*(?::\s*(\w+))?", item)
        if not m:
            raise SpecError(f"consts: {item!r} is not NAME[:int|:float]", where)
        name, typ = m.group(1), m.group(2) or "int"
        if typ not in CONST_TYPES:
            raise SpecError(f"consts: {name}:{typ}; want :int or :float", where)
        if name in names:
            raise SpecError(f"consts: {name!r} declared twice", where)
        names.add(name)
        out.append(ConstSpec(index=i, name=name, type=typ))
    return out


def _paren_end(clean: str, open_idx: int) -> int:
    """Offset just past the ``)`` matching the ``(`` at *open_idx*."""
    depth = 0
    for i in range(open_idx, len(clean)):
        if clean[i] == "(":
            depth += 1
        elif clean[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    raise SpecError("AIEK_STEP( has no closing parenthesis")


def parse_step_text(entry: str, args: str, where: str = "", source=None) -> StepSpec:
    """Parse the clause text of one AIEK_STEP (everything after ``entry,``)."""
    parts = _split_top(args)
    step = StepSpec(entry=entry, source=Path(source) if source else None, where=where, raw=args)
    if len(parts) != 3:
        raise SpecError(f"AIEK_STEP({entry}, ...) needs exactly reads(...), writes(...), "
                        f"consts(...); got {len(parts)} clause(s)", where)
    for want, part in zip(_STEP_CLAUSES, parts):
        m = re.fullmatch(r"(\w+)\s*\((.*)\)", part, re.S)
        if not m or m.group(1) != want:
            raise SpecError(f"AIEK_STEP({entry}): clause {part!r}; want {want}(...) "
                            f"(order: reads, writes, consts)", where)
        body = m.group(2)
        if want == "consts":
            step.consts = _parse_consts(body, where)
        else:
            setattr(step, want, _parse_bufs(body, where, want))
    if not step.reads and not step.writes:
        raise SpecError(f"AIEK_STEP({entry}) reads and writes nothing", where)
    return step


def parse_steps(src: SourceText) -> list:
    """Every ``AIEK_STEP`` in *src*, parsed; each entry must also be defined
    in the text (``AIEK_STEP_ENTRY(entry)`` or a plain ``entry(`` definition).
    Raises :class:`SpecError` on the first malformed one."""
    clean = strip_comments_and_strings(src.text)
    steps, seen = [], {}
    for m in _STEP_CALL.finditer(clean):
        if clean[max(0, m.start() - 8):m.start()].rstrip().endswith("define"):
            continue                                   # the macro definition itself
        where = src.where(m.start())
        end = _paren_end(clean, m.end() - 1)
        inner = clean[m.end():end - 1]
        em = re.match(r"\s*([A-Za-z_]\w*)\s*,", inner)
        if not em:
            raise SpecError("AIEK_STEP(entry, reads(...), writes(...), consts(...))", where)
        entry = em.group(1)
        if entry in seen:
            raise SpecError(f"AIEK_STEP({entry}) declared twice (first at {seen[entry]})", where)
        seen[entry] = where
        step = parse_step_text(entry, inner[em.end():], where, src.path)
        rest = clean[:m.start()] + " " * (end - m.start()) + clean[end:]
        if not re.search(r"\bAIEK_STEP_ENTRY\s*\(\s*%s\s*\)|\b%s\s*\([^;{]*\)\s*\{" % (entry, entry), rest):
            raise SpecError(f"AIEK_STEP({entry}) but no definition of {entry} "
                            f"(write AIEK_STEP_ENTRY({entry}) {{ ... }})", where)
        steps.append(step)
    return steps
