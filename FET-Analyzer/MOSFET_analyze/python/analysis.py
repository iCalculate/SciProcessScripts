"""Extensible analysis layer for the B1500 plotter.

This module is the seam the project will grow along.  Today it ships a couple of
lightweight FET analyses; tomorrow you add more by writing one function and
decorating it with :func:`register`.  The GUI discovers everything in
:data:`REGISTRY` automatically — it builds a checkbox per analysis and overlays
whatever annotations the analysis returns, with **no GUI code changes required**.

Contract for an analysis function::

    @register(Analysis(key="...", label="...", applies_to={"transfer"}))
    def my_analysis(curve, ctx) -> list[Annotation]:
        ...

* ``curve``  : a :class:`b1500_io.Curve` (named columns: ``Vg``, ``Id`` ...).
* ``ctx``    : :class:`AnalysisContext` — the measurement kind, x column, the
               plotted y column, and which y-axis the curve sits on.
* returns    : a list of :class:`Annotation` describing points/lines/text to
               draw.  Returning ``[]`` means "nothing to show for this curve".

Analyses must be pure and defensive: never raise on noisy data, just return ``[]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np

# Avoid a hard import cycle; only used for type hints.
try:  # pragma: no cover
    from b1500_io import Curve
except Exception:  # pragma: no cover
    Curve = object  # type: ignore


# --------------------------------------------------------------------------- #
# Result + descriptor types
# --------------------------------------------------------------------------- #

@dataclass
class Annotation:
    """A drawable analysis result. The plotter renders whichever fields are set."""

    kind: str                       # "point" | "vline" | "hline" | "text" | "line"
    x: Optional[float] = None
    y: Optional[float] = None
    x2: Optional[float] = None      # for "line": second endpoint
    y2: Optional[float] = None
    text: str = ""
    color: Optional[str] = None     # default: inherit the curve's colour
    on_right: bool = False          # draw against the right y-axis


@dataclass
class AnalysisContext:
    kind: str                       # MeasurementKind value
    x_name: str
    y_name: str                     # column currently plotted for this curve
    on_right: bool = False
    extras: Dict[str, object] = field(default_factory=dict)


@dataclass
class Analysis:
    key: str
    label: str                      # shown in the GUI checkbox
    func: Optional[Callable] = None
    applies_to: Set[str] = field(default_factory=set)   # empty -> any kind
    description: str = ""

    def supports(self, kind: str) -> bool:
        return not self.applies_to or kind in self.applies_to


REGISTRY: "Dict[str, Analysis]" = {}


def register(meta: Analysis):
    """Decorator: attach an analysis function and add it to :data:`REGISTRY`."""
    def _wrap(fn: Callable) -> Callable:
        meta.func = fn
        REGISTRY[meta.key] = meta
        return fn
    return _wrap


def analyses_for(kind: str) -> List[Analysis]:
    return [a for a in REGISTRY.values() if a.supports(kind)]


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #

def _finite_xy(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


# --------------------------------------------------------------------------- #
# Built-in analyses (also serve as worked examples of the plugin pattern)
# --------------------------------------------------------------------------- #

@register(Analysis(
    key="onoff",
    label="On/Off ratio (mark Ion, Ioff)",
    applies_to={"transfer"},
    description="Mark the max and min |Id| on a transfer curve and report Ion/Ioff.",
))
def _on_off_ratio(curve, ctx: AnalysisContext) -> List[Annotation]:
    x = curve.get(ctx.x_name)
    y = curve.get(ctx.y_name)
    if x is None or y is None:
        return []
    x, y = _finite_xy(np.asarray(x, float), np.abs(np.asarray(y, float)))
    y = y[y > 0]
    if y.size < 3:
        return []
    ion, ioff = float(np.max(y)), float(np.median(np.sort(y)[:max(3, y.size // 20)]))
    if ioff <= 0:
        return []
    i_on = int(np.argmax(np.abs(np.asarray(curve.get(ctx.y_name), float))))
    out = [Annotation("point", x=float(x[i_on]) if i_on < x.size else float(x[-1]),
                      y=ion, on_right=ctx.on_right)]
    ratio = ion / ioff
    out.append(Annotation("text", x=float(x[i_on]) if i_on < x.size else float(x[-1]),
                          y=ion, text=f"  I$_{{on}}$/I$_{{off}}$ = {ratio:.1e}",
                          on_right=ctx.on_right))
    return out


@register(Analysis(
    key="gm_peak",
    label="Peak transconductance (mark gm,max)",
    applies_to={"transfer"},
    description="Mark the gate voltage of maximum dId/dVg.",
))
def _peak_gm(curve, ctx: AnalysisContext) -> List[Annotation]:
    # Prefer an instrument-provided gm column; else differentiate numerically.
    gm = curve.get("gm")
    x = curve.get(ctx.x_name)
    y = curve.get(ctx.y_name)
    if x is None or y is None:
        return []
    x = np.asarray(x, float)
    if gm is not None:
        gm = np.asarray(gm, float)
    else:
        yy = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(yy)
        if m.sum() < 4:
            return []
        gm = np.full_like(x, np.nan)
        gm[m] = np.gradient(yy[m], x[m])
    gm_abs = np.abs(gm)
    if not np.any(np.isfinite(gm_abs)):
        return []
    i = int(np.nanargmax(gm_abs))
    yv = curve.get(ctx.y_name)
    yval = float(np.asarray(yv, float)[i]) if yv is not None else None
    return [
        Annotation("vline", x=float(x[i]), text="", on_right=ctx.on_right,
                   color="0.5"),
        Annotation("point", x=float(x[i]), y=yval, on_right=ctx.on_right),
        Annotation("text", x=float(x[i]), y=yval,
                   text=f"  V$_g$={x[i]:.2g} V", on_right=ctx.on_right),
    ]
