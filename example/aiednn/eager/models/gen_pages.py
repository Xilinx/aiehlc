#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""One Markdown page per benchmark model, showing the code at every layer of aiednn's EAGER mode (PyTorch model ->
generated C app -> eager plan -> generated instance entry -> kernel template -> board results), all read from the
VEK280 board builds under worklocal/aiednn/models/. Writes <model>.md and models.md next to this script.

    python3 example/aiednn/eager/models/gen_pages.py

Read-only on everything else; stdlib only; the output depends only on the files read (no timestamps).
"""
import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BUILDS = REPO / "worklocal" / "aiednn" / "models"
KDIR = REPO / "src" / "aietensorop" / "eager"
ZOO = REPO / "example" / "aiednn" / "models" / "zoo.py"
UP = "../../../../"                                  # this directory -> repo root

MODELS = ["gpt2_small", "tiny_llama", "bert_base", "resnet18", "resnet50", "mobilenet_v2", "yolov8n"]
VARIANTS = [("4x4", "baseline", "{m}_int8_mlir_4x4_eager"), ("4x4", "warm", "{m}_int8_mlir_4x4_warm_eager"),
            ("8x4", "baseline", "{m}_int8_mlir_8x4_eager"), ("8x4", "warm", "{m}_int8_mlir_8x4_warm_eager")]
ZOO_BLOCKS = {"gpt2_small": ["GPT2Block", "GPT2Small"], "tiny_llama": ["RMSNorm", "rope", "LlamaBlock", "TinyLlama"],
              "bert_base": ["BertLayer", "BertBase"], "yolov8n": ["_yolov8n", "_c2f_forward"]}
TORCHVISION_DESC = {
    "resnet18": "torchvision ResNet-18 (ImageNet weights, `weights=\"DEFAULT\"`), an image classifier",
    "resnet50": "torchvision ResNet-50 (ImageNet weights, `weights=\"DEFAULT\"`), an image classifier with "
                "bottleneck blocks",
    "mobilenet_v2": "torchvision MobileNetV2 (ImageNet weights, `weights=\"DEFAULT\"`), an image classifier built "
                    "from inverted residual blocks with depthwise convolutions",
}
WARM_NOTE = ("warm = partition reused across launches and warm ELF loads (commit cca26a2b, branch eager-warm); "
             "baseline = the builds made before it")
NA = "not available"


def rel(p: Path) -> str:
    return UP + str(p.resolve().relative_to(REPO))


def read(p: Path):
    return p.read_text(errors="replace") if p.is_file() else None


# ----------------------------------------------------------------------------------------------------------- builds

def parse_run_log(text):
    r = {"model": None, "cold": None, "verdict": None, "result": None, "prof": [], "stats": [], "gemmchk": []}
    for ln in text.replace("\r", "").splitlines():
        ln = ln.strip()
        if ln.startswith("MODEL "):
            r["model"] = dict(kv.split("=", 1) for kv in ln.split()[2:] if "=" in kv)
        elif ln.startswith("MODEL_COLD "):
            m = re.search(r"time_ms=([\d.]+)", ln)
            r["cold"] = m.group(1) if m else None
        elif ln.startswith(("PASS:", "FAIL:")):
            r["verdict"] = ln
        elif ln.startswith("===== RESULT:"):
            r["result"] = ln.strip("= ").replace("RESULT: ", "")
        elif ln.startswith("GEMMCHK"):
            r["gemmchk"].append(ln)
        elif m := re.match(r"PROF (\S+)\s+(\d+) ops\s+([\d.]+) ms", ln):
            r["prof"].append((m.group(1), int(m.group(2)), float(m.group(3))))
        elif m := re.match(r"aiednn_stats: (\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)$", ln):
            r["stats"].append((m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5))))
    return r


def load_build(model, mesh, mode, pattern):
    d = BUILDS / pattern.format(m=model)
    b = {"dir": d, "name": d.name, "mesh": mesh, "mode": mode, "exists": d.is_dir()}
    if not b["exists"]:
        return b
    b["app"] = d / f"{model}_app.cc"
    b["inst_cc"] = d / "vek280" / "eager_instances.cc"
    b["plan_json"] = d / "vek280" / "eager_instances.json"
    b["run_log"] = d / "vek280" / "run.log"
    b["mlir"] = d / "ir" / f"{model}.opt.mlir"
    log = read(b["run_log"])
    b["log"] = parse_run_log(log) if log is not None else None
    pj = read(b["plan_json"])
    b["plan"] = json.loads(pj) if pj else None
    b["ok"] = bool(b["log"] and b["log"]["model"] and b["log"]["model"].get("gemm_bad") == "0"
                   and (b["log"]["verdict"] or "").startswith("PASS"))
    return b


def verdict(b):
    if not b.get("log"):
        return "no run.log"
    v = b["log"]["verdict"] or b["log"]["result"] or "no verdict"
    return "PASS" if v.startswith("PASS") else ("FAIL" if v.startswith("FAIL") else v)


def pick_primary(builds):
    for mesh in ("8x4", "4x4"):
        for mode in ("warm", "baseline"):
            for b in builds:
                if b["exists"] and b["ok"] and b["mesh"] == mesh and b["mode"] == mode:
                    return b
    return next((b for b in builds if b["exists"]), None)


# -------------------------------------------------------------------------------------------------------- zoo.py

def zoo_blocks(src):
    """top-level `class X` / `def X` blocks of zoo.py by name, text-parsed"""
    lines, out, i = src.splitlines(), {}, 0
    while i < len(lines):
        m = re.match(r"(class|def)\s+(\w+)", lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j][0] in " \t"):
            j += 1
        blk = lines[i:j]
        while blk and not blk[-1].strip():
            blk.pop()
        out[m.group(2)] = blk
        i = j
    return out


def zoo_inputs(src):
    return {m.group(1): m.group(3) for m in re.finditer(r'"(\w+)": \((\w+), \(([\d, ]+)\)\)', src)}


def docstring(block):
    m = re.search(r'"""(.*?)"""', "\n".join(block), re.S)
    return " ".join(m.group(1).split()) if m else None


# ----------------------------------------------------------------------------------------------------------- app

def app_blocks(text):
    """(op index, layer name, create function, lines) per primitive block of the generated app"""
    lines = text.splitlines()
    starts = [(i, int(m.group(1)), m.group(2)) for i, ln in enumerate(lines)
              if (m := re.match(r'\s*g_op_name\[(\d+)\] = "(.*)";', ln))]
    out = []
    for k, (i, idx, name) in enumerate(starts):
        end = starts[k + 1][0] - 1 if k + 1 < len(starts) else len(lines)
        blk = lines[i - 1:end]
        cm = re.search(r"(aiednn_\w*create\w*)\(", "\n".join(blk))
        out.append((idx, name, cm.group(1) if cm else "?", blk))
    return out


def shorten_array(ln, keep=3):
    m = re.match(r"(.*?= \{)(.*)(\};)\s*$", ln)
    if len(ln) <= 160 or not m:
        return ln, False
    vals = [v.strip() for v in m.group(2).split(",")]
    return f"{m.group(1)}{', '.join(vals[:keep])}, /* ... {len(vals)} values */{m.group(3)}", True


def representative(blocks):
    for idx, name, fn, blk in blocks:
        if fn in ("aiednn_matmul_forward_create_attr", "aiednn_convolution_forward_create_attr") \
                and any("aiednn_s8" in ln for ln in blk):
            cut = []
            for ln in blk:
                if ln.strip().startswith("#if VERIFY_GEMM"):
                    break
                cut.append(ln)
                if f'return fail({idx}, "execute", st);' in ln:
                    break
            cut, short = cut[:40], False
            for k, ln in enumerate(cut):
                cut[k], s = shorten_array(ln)
                short |= s
            return idx, name, fn, cut, short
    return None


# ----------------------------------------------------------------------------------------------- instance entry

def parse_instances(text):
    lines = text.splitlines()
    first = next((i for i, ln in enumerate(lines) if ln.startswith("static void")), len(lines))
    header = [ln for ln in lines[:first] if ln.startswith(("#include", "#define", "// Generated", "// model"))]
    inst = {}
    for i, ln in enumerate(lines):
        m = re.match(r"static void (\w+)_launch\(", ln)
        if not m:
            continue
        j = i
        while j < len(lines) and not lines[j].startswith("AIEK_ENTRY("):
            j += 1
        code = lines[i:j + 1]
        call = next((re.search(r"(\w+)<([^<>]*)><<<", c) for c in code if "<<<" in c), None)
        inst[m.group(1)] = {"comment": lines[i - 1] if lines[i - 1].startswith("//") else None, "code": code,
                            "args": call.group(2) if call else None}
    hw_rows = next((re.search(r"#define HW_ROWS (\d+)", ln).group(1) for ln in header
                    if ln.startswith("#define HW_ROWS")), None)
    return header, inst, hw_rows


def short_comment(c, nlayers):
    if c is None or len(c) <= 160:
        return c
    names = c[3:].split(", ")
    return f"// {', '.join(names[:4])}, ... ({nlayers} layers; comment shortened here)"


def template_args(i, inst):
    if inst.get(i["name"], {}).get("args"):
        return inst[i["name"]]["args"]
    vals = i["params"] or [i["M"], i["N"], i["K"], i["TK"], i["TM"]]
    return ", ".join(str(v) for v in vals)


# ------------------------------------------------------------------------------------------------------ templates

def template_signature(path, tmpl):
    text = read(path)
    if text is None:
        return None
    lines = text.splitlines()
    g = next((i for i, ln in enumerate(lines) if re.match(rf"__global__ void {tmpl}\(", ln)), None)
    if g is None:
        return None
    s = g - 1 if g > 0 and lines[g - 1].startswith("template") else g
    while s > 0 and lines[s - 1].startswith("//"):
        s -= 1
    e = g
    while e < len(lines) and not lines[e].rstrip().endswith("{"):
        e += 1
    return lines[s:e + 1]


# ------------------------------------------------------------------------------------------------------- tables

def fmt_ms(v):
    return f"{float(v):.1f}" if v is not None else NA


def results_row(b):
    if not b["exists"]:
        return None
    lg = b.get("log")
    mdl = lg["model"] if lg else None
    if not mdl:
        return (f"| {b['mesh']} | {b['mode']} | {NA} | {NA} | {NA} | {NA} | {NA} | {verdict(b)} | "
                f"`{b['name']}` |")
    chk, bad = int(mdl.get("gemm_checked", 0)), int(mdl.get("gemm_bad", 0))
    t1 = f"{mdl.get('top1')} vs {mdl.get('ref_top1')}" + (" (match)" if mdl.get("top1") == mdl.get("ref_top1")
                                                          else " (miss)")
    return (f"| {b['mesh']} | {b['mode']} | {fmt_ms(mdl.get('time_ms'))} | {fmt_ms(lg['cold'])} | "
            f"{chk - bad}/{chk} | {t1} | {mdl.get('relerr', NA)} | {verdict(b)} | "
            f"[`{b['name']}`]({rel(b['run_log'])}) |")


def page(model, builds, zoo_src):
    P = []
    w = P.append
    prim = pick_primary(builds)
    if prim is None:
        return f"# {model}\n\nNo build directory found under `{BUILDS.relative_to(REPO)}`.\n", ["no build dir"]
    missing = []
    app_text = read(prim["app"])
    blocks = app_blocks(app_text) if app_text else []
    header_comment = app_text.splitlines()[0] if app_text else None
    plan = prim["plan"]
    inst_text = read(prim["inst_cc"])
    inst_header, inst, hw_rows = parse_instances(inst_text) if inst_text else ([], {}, None)
    stats = prim["log"]["stats"] if prim.get("log") else []
    prof = prim["log"]["prof"] if prim.get("log") else []
    zblocks = zoo_blocks(zoo_src) if zoo_src else {}
    inputs = zoo_inputs(zoo_src) if zoo_src else {}

    # ---- title + summary
    w(f"# {model} in aiednn eager mode\n")
    if model in ZOO_BLOCKS:
        if model == "yolov8n":
            what = ("ultralytics YOLOv8n, an object detector, with the released COCO weights (`yolov8n.pt`, the "
                    "head in export mode: one output tensor, no NMS)")
        else:
            what = docstring(zblocks.get(ZOO_BLOCKS[model][-1], [])) or f"`{ZOO_BLOCKS[model][-1]}` from zoo.py"
        shape = inputs.get(model)
        inp = (f"input `[{shape}]`" + (" (ultralytics' bus.jpg resized to 320 x 320)" if model == "yolov8n"
                                        else " (random embeddings; random weights)")) if shape else f"input {NA}"
    else:
        what = TORCHVISION_DESC.get(model, f"torchvision {model}")
        inp = "input `[1, 3, 224, 224]`, a real ImageNet image (`zoo.image_input()`)"
    nprim = re.search(r"(\d+) primitives", header_comment or "")
    nprim = int(nprim.group(1)) if nprim else len(blocks)
    ninst = len(plan["instances"]) if plan else 0
    nlay = sum(len(i["ops"]) for i in plan["instances"]) if plan else 0
    nlaunch = sum(s[2] for s in stats)
    w(f"{what.rstrip('. ')}; {inp}. Quantized to int8 and lowered through the aiednn MLIR pipeline, it becomes "
      f"**{nprim} aiednn primitives** in the generated C app. The eager plan of the primary build "
      f"(`{prim['name']}`, mesh {prim['mesh']}, {prim['mode']}) makes **{ninst} kernel instances** that serve "
      f"**{nlay} of those layers**; on the board `aiednn_stats` counts **{nlaunch} AIE launches** in the timed run. "
      f"Each instance is one shape-specialized template instantiation, loaded right before the layer it serves runs.")
    if plan and plan.get("refused"):
        w("")
        w("Layers the planner refused (from `eager_instances.json`):\n")
        for r in plan["refused"]:
            w(f"- `{r['name']}`: {r['reason']}")
    w("")

    # ---- run it
    w("## Run it\n")
    w("```bash")
    w(f"python example/aiednn/eager/models/run.py --model {model}                # x86 emulation, 4x4")
    w(f"python example/aiednn/eager/models/run.py --model {model} --show         # + plan and instance code")
    w(f"python example/aiednn/eager/models/run.py --model {model} --mesh 8x4 --target vek280 --board galerina13 \\")
    w("    --vitis-settings <.../Vitis/settings64.sh>")
    w("# the same through the model runner (builds worklocal/aiednn/models/"
      f"{model}_int8_mlir_8x4[_warm]_eager):")
    w(f"python example/aiednn/models/run_models.py --models {model} --modes int8 --app vek280 --board galerina13 \\")
    w("    --eager --eager-mesh 8x4 --mlir [--tag warm]")
    w("```\n")
    w("`--tag warm` only names the build directory; the warm launch is in the current code.\n")

    # ---- 1. PyTorch
    w("## 1. The PyTorch model\n")
    if zoo_src is None:
        w(f"`zoo.py` {NA}.\n")
        missing.append("zoo.py")
    elif model in ZOO_BLOCKS:
        names = ZOO_BLOCKS[model]
        w(f"From [`example/aiednn/models/zoo.py`]({rel(ZOO)}) ({', '.join(f'`{n}`' for n in names)}):\n")
        w("```python")
        for k, n in enumerate(names):
            if n not in zblocks:
                w(f"# {n}: {NA}")
                missing.append(f"zoo.py:{n}")
                continue
            if k:
                w("")
                w("")
            P.extend(zblocks[n])
        w("```\n")
        bm = re.search(rf'"{model}": \(\w+, \([\d, ]+\)\)', zoo_src)
        if bm:
            w(f"Model and input shape, from `zoo.BENCH`: `{bm.group(0)}`\n")
    else:
        ln = next((ln.strip() for ln in zoo_src.splitlines() if "getattr(tvm, name)" in ln), None)
        w(f"A torchvision model; [`zoo.py`]({rel(ZOO)}) `build()` constructs it as:\n")
        w("```python")
        w(ln or f"# construction line {NA}")
        w("```\n")
        w(f"i.e. `torchvision.models.{model}(weights=\"DEFAULT\").eval()`.\n")

    # ---- 2. app
    w("## 2. aiednn primitives (generated C app)\n")
    if app_text is None:
        w(f"`{prim['name']}/{model}_app.cc` {NA}.\n")
        missing.append(f"{prim['name']}/{model}_app.cc")
    else:
        w(f"[`{prim['name']}/{model}_app.cc`]({rel(prim['app'])}), first line:\n")
        w("```c")
        w(header_comment)
        w("```\n")
        cnt = collections.Counter(fn for _, _, fn, _ in blocks)
        w("| create function | primitives |")
        w("|---|---:|")
        for fn, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            w(f"| `{fn}` | {n} |")
        w(f"| **total** | **{sum(cnt.values())}** |")
        w("")
        if sum(cnt.values()) != nprim:
            w(f"(The header says {nprim} primitives; {sum(cnt.values())} `g_op_name` blocks were found.)\n")
        rep = representative(blocks)
        if rep:
            idx, name, fn, cut, short = rep
            w(f"Op {idx} (`{name}`), the first int8 `{fn}` block, verbatim up to its execute"
              + (" (long constant arrays shortened to their first values)" if short else "") + ":\n")
            w("```c")
            P.extend(cut)
            w("```\n")
            w("In the app this is followed by an `#if VERIFY_GEMM` block that re-runs the layer with the `ref:` "
              "implementation and `memcmp`s the result (the `gemm_checked` / `gemm_bad` counters).\n")
        else:
            w(f"No int8 matmul / convolution `create_attr` block found: {NA}.\n")
    mlir = read(prim["mlir"]) if prim.get("mlir") else None
    if mlir:
        ops = collections.Counter(re.findall(r"= aiednn\.(\w+)", mlir))
        body = ", ".join(f"{k} {v}" for k, v in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0])) if k != "const")
        w(f"The aiednn MLIR after the pass pipeline, [`ir/{prim['mlir'].name}`]({rel(prim['mlir'])}), "
          f"holds: {body} (+ {ops.get('const', 0)} `aiednn.const`).\n")
    else:
        w(f"aiednn MLIR (`ir/{model}.opt.mlir`): {NA}.\n")

    # ---- 3. plan
    w("## 3. The eager plan\n")
    if not plan:
        w(f"`vek280/eager_instances.json` {NA}.\n")
        missing.append(f"{prim['name']}/vek280/eager_instances.json")
    else:
        mesh = "x".join(str(v) for v in plan.get("mesh", [])) or NA
        w(f"[`{prim['name']}/vek280/eager_instances.json`]({rel(prim['plan_json'])}): mesh {mesh}, "
          f"{plan.get('gemm_layers', NA)} GEMM-family layers, {ninst} instances for {nlay} layers.\n")
        w("| instance | template | template arguments | layers | first layers |")
        w("|---|---|---|---:|---|")
        for i in plan["instances"]:
            names = [o["name"] for o in i["ops"]]
            first = ", ".join(f"`{n}`" for n in names[:3]) + (", ..." if len(names) > 3 else "")
            kfile = KDIR / f"{i['template']}_eager_kernel.cc"
            t = f"[`{i['template']}`]({rel(kfile)})" if kfile.is_file() else f"`{i['template']}`"
            w(f"| `{i['name']}` | {t} | `<{template_args(i, inst)}>` | {len(names)} | {first} |")
        w("")

    # ---- 4. instance code
    w("## 4. Instance code (generated entry)\n")
    if inst_text is None:
        w(f"`vek280/eager_instances.cc` {NA}.\n")
        missing.append(f"{prim['name']}/vek280/eager_instances.cc")
    else:
        w(f"[`{prim['name']}/vek280/eager_instances.cc`]({rel(prim['inst_cc'])}) (mesh `HW_ROWS` = {hw_rows}); "
          "header:\n")
        w("```cpp")
        P.extend(inst_header)
        w("```\n")
        order = sorted(plan["instances"] if plan else [], key=lambda i: -len(i["ops"]))[:3]
        if order:
            w(f"The {len(order)} most-used instances (launch function + `AIEK_ENTRY`; each also has an `AIEK_SPEC` "
              "and an `AIEK_REFERENCE` line, not shown):\n")
        for i in order:
            e = inst.get(i["name"])
            if not e:
                w(f"`{i['name']}`: {NA} in eager_instances.cc.\n")
                continue
            w(f"`{i['name']}`, {len(i['ops'])} layers:\n")
            w("```cpp")
            c = short_comment(e["comment"], len(i["ops"]))
            if c:
                w(c)
            P.extend(e["code"])
            w("```\n")

    # ---- 5. templates
    w("## 5. The template\n")
    tmpls = list(dict.fromkeys(i["template"] for i in plan["instances"])) if plan else []
    if not tmpls:
        w(f"Templates: {NA} (no plan).\n")
    for t in tmpls:
        kfile = KDIR / f"{t}_eager_kernel.cc"
        sig = template_signature(kfile, t)
        if sig is None:
            w(f"### `{t}`\n\n`src/aietensorop/eager/{t}_eager_kernel.cc` signature: {NA}.\n")
            missing.append(f"template {t}")
            continue
        w(f"### [`{t}`]({rel(kfile)})\n")
        w("```cpp")
        P.extend(sig)
        w("```\n")
        built = template_signature(prim["dir"] / "vek280" / f"{t}_eager_kernel.cc", t)
        if built is not None and built != sig:
            w(f"(The copy compiled into `{prim['name']}` has a different signature: the source changed after "
              "that build.)\n")

    # ---- 6. results
    w("## 6. Board results (galerina13, VEK280)\n")
    w(f"From each build's `vek280/run.log` ({WARM_NOTE}). time = the `MODEL` line's `time_ms` (timed run), "
      "cold = `MODEL_COLD` (first run, incl. one-time weight packing and ELF load). GEMM exact = layers bit-exact "
      "vs `ref:` / layers checked.\n")
    w("| mesh | launch | time ms | cold ms | GEMM exact | top-1 vs reference | rel err | result | build |")
    w("|---|---|---:|---:|---:|---|---:|---|---|")
    for b in builds:
        row = results_row(b)
        if row:
            w(row)
    absent = [b["name"] for b in builds if not b["exists"]]
    w("")
    if absent:
        w(f"Not built: {', '.join(f'`{a}`' for a in absent)}.\n")
    for b in builds:
        if b["exists"] and b.get("log") and not b["ok"]:
            m = b["log"]["model"] or {}
            w(f"`{b['name']}` failed (gemm_bad={m.get('gemm_bad', NA)}):\n")
            w("```text")
            P.extend(b["log"]["gemmchk"])
            if b["log"]["verdict"]:
                w(b["log"]["verdict"])
            w("```\n")

    w(f"### Per implementation, primary build `{prim['name']}`\n")
    if not stats:
        w(f"`aiednn_stats` lines: {NA}.\n")
    else:
        profd = {p[0] for p in prof}
        w("From the `aiednn_stats` lines. execs counts every call of an implementation, including calls made "
          "from inside a composite (for example the inner GEMMs of `gemm+sdpa8`). total = the whole execute, host "
          "packing included; in_entry = time inside the instances' generated `AIEK_ENTRY` calls (the launch "
          "functions of section 4).\n")
        w("| implementation | execs | launches | total ms | in_entry ms | ms / launch | note |")
        w("|---|---:|---:|---:|---:|---:|---|")
        for impl, ex, la, tot, ine in stats:
            note = ""
            if impl.startswith("ref:") and impl not in profd:
                note = "reference re-run for the GEMM check (`VERIFY_GEMM`), not a network layer"
            elif la == 0 and not impl.startswith("ref:"):
                note = "composite / host step; its AIE launches are counted under the inner implementation"
            per = f"{ine / la:.3f}" if la else "-"
            w(f"| `{impl}` | {ex} | {la} | {tot:.2f} | {ine:.2f} | {per} | {note} |")
        w("")
    if prof:
        w("Per-op time of the timed run (`PROF` lines, one row per top-level implementation):\n")
        w("| implementation | ops | ms |")
        w("|---|---:|---:|")
        for impl, n, ms in prof:
            w(f"| `{impl}` | {n} | {ms:.2f} |")
        w(f"| **sum** | **{sum(p[1] for p in prof)}** | **{sum(p[2] for p in prof):.2f}** |")
        w("")
    w("[All models](models.md)")
    return "\n".join(P) + "\n", missing


def index(all_builds):
    P = ["# aiednn eager mode: benchmark models\n",
         "One page per model with the code at every layer of eager mode (PyTorch model, generated C app, eager "
         "plan, generated instance entry, kernel template) and the VEK280 board results (galerina13). Generated "
         "by [`gen_pages.py`](gen_pages.py) from the builds in `worklocal/aiednn/models/`; " + WARM_NOTE + ".\n",
         "Time = `time_ms` of the timed run in ms, then the run's verdict; `-` = not built.\n",
         "| model | primitives | instances (8x4) | 4x4 baseline | 4x4 warm | 8x4 baseline | 8x4 warm |",
         "|---|---:|---:|---|---|---|---|"]
    for model in MODELS:
        builds = all_builds[model]
        prim = pick_primary(builds)
        cells = []
        for b in builds:
            if not b["exists"]:
                cells.append("-")
            elif b.get("log") and b["log"]["model"]:
                m = b["log"]["model"]
                bad = "" if m.get("gemm_bad") == "0" else f", gemm_bad={m.get('gemm_bad')}"
                cells.append(f"{fmt_ms(m.get('time_ms'))} {verdict(b)}{bad}")
            else:
                cells.append(f"{NA} ({verdict(b)})")
        nprim, ninst = NA, NA
        if prim:
            hdr = read(prim["app"])
            mm = re.search(r"(\d+) primitives", hdr.splitlines()[0]) if hdr else None
            nprim = mm.group(1) if mm else NA
            ninst = len(prim["plan"]["instances"]) if prim.get("plan") else NA
        P.append(f"| [{model}]({model}.md) | {nprim} | {ninst} | " + " | ".join(cells) + " |")
    P.append("")
    P.append("Regenerate: `python3 example/aiednn/eager/models/gen_pages.py` (from the repo root).")
    return "\n".join(P) + "\n"


def main():
    zoo_src = read(ZOO)
    all_builds, written, missing = {}, [], {}
    for model in MODELS:
        builds = [load_build(model, mesh, mode, pat) for mesh, mode, pat in VARIANTS]
        all_builds[model] = builds
        text, miss = page(model, builds, zoo_src)
        (HERE / f"{model}.md").write_text(text)
        written.append(f"{model}.md")
        if miss:
            missing[model] = miss
    (HERE / "models.md").write_text(index(all_builds))
    written.append("models.md")
    for f in written:
        print(f"wrote {(HERE / f).relative_to(REPO)} ({len((HERE / f).read_text().splitlines())} lines)")
    for model, miss in missing.items():
        print(f"missing for {model}: {', '.join(miss)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
