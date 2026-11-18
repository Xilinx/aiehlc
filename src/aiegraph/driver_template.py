"""Driver template (unet-m3 S6): the application's C glue in the generated ``graph_driver.c``.

A model build declares its groups (``emod.offload_group``) and, once, the C text the driver
needs around them -- step / kernel declarations, profiling glue, enter / exit hooks -- with
``emod.driver_template(...)``. aiegraph places that text at fixed, checked anchors of every
driver ``arm_build`` generates for the workdir; the application never edits the C itself.

Anchors (each must occur exactly once, else ``DriverTemplateError``):
  * ``decls``        -- before ``static uint8_t kernel_ws[`` (after the group-step declarations);
  * ``expand_enter`` / ``expand_exit`` -- around the loop of ``graph_bf16_expand``;
  * ``run_enter``    -- first statement of ``graph_run``.
A driver that already carries the template text is refused (applied twice = error).
The pass runs before any pass registered after it (e.g. ``keep_layout``).
"""
from pathlib import Path

KWS = "static uint8_t kernel_ws["
EXPAND_LOOP = "    for (uint64_t i = 0; i < n; ++i) d[i] = (uint32_t)src[i] << 16;\n}\n"
EXPAND_HEAD = "static void graph_bf16_expand(float *dst, const uint16_t *src, uint64_t n) {\n"
RUN_HEAD = "float *graph_run(void) {\n"


class DriverTemplateError(ValueError):
    pass


def _once(text: str, anchor: str, what: str) -> int:
    n = text.count(anchor)
    if n != 1:
        raise DriverTemplateError(f"driver template: {what} anchor {anchor.strip()!r} found {n} times (need 1)")
    return text.index(anchor)


def apply_text(text: str, decls: str = "", expand_enter: str = "", expand_exit: str = "",
               run_enter: str = "") -> str:
    """The template applied to one driver's text (pure function, x86-testable)."""
    for name, part in (("decls", decls), ("expand_enter", expand_enter), ("expand_exit", expand_exit),
                       ("run_enter", run_enter)):
        if part and part.strip() and part in text:
            raise DriverTemplateError(f"driver template: {name} already in the driver (applied twice?)")
    if decls:
        i = _once(text, KWS, "decls")
        text = text[:i] + decls + text[i:]
    if expand_enter or expand_exit:
        h = _once(text, EXPAND_HEAD, "graph_bf16_expand")
        j = text.find(EXPAND_LOOP, h)
        if j < 0 or text.count(EXPAND_LOOP) != 1:
            raise DriverTemplateError("driver template: graph_bf16_expand loop not found exactly once "
                                      "(AIEGRAPH_WDTYPE=bf16 driver needed for expand hooks)")
        text = text[:j] + expand_enter + EXPAND_LOOP[:-2] + expand_exit + "}\n" + text[j + len(EXPAND_LOOP):]
    if run_enter:
        r = _once(text, RUN_HEAD, "graph_run") + len(RUN_HEAD)
        text = text[:r] + run_enter + text[r:]
    return text


class DriverTemplatePass:
    name = "driver_template"

    def __init__(self, decls="", expand_enter="", expand_exit="", run_enter=""):
        self.parts = dict(decls=decls, expand_enter=expand_enter, expand_exit=expand_exit, run_enter=run_enter)
        self.last = None

    def __call__(self, driver_c, graph_json=None, weights_bin=None, c_source=None):
        p = Path(driver_c)
        p.write_text(apply_text(p.read_text(), **self.parts))
        self.last = {k: len(v) for k, v in self.parts.items()}
        return self.last


def register(workdir, **parts) -> DriverTemplatePass:
    """Apply the template to every driver ``arm_build`` generates for ``workdir``."""
    from . import _env
    p = DriverTemplatePass(**parts)
    _env.frontend("arm_build").register_driver_pass(workdir, p.name, p)
    return p
