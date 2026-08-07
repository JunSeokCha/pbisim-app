"""Shared plot-axis controls for the pbisim-app results views.

One small contract used by both plotting backends:

    opts = plot_axis_controls("sim_bact", default_y="Log")   # renders the UI
    apply_axis_mpl(ax, opts)        # matplotlib
    apply_axis_plotly(fig, opts)    # plotly

`plot_axis_controls` renders a compact "Plot options" expander (X/Y scale +
optional axis limits) and returns a plain dict; the two `apply_*` helpers are
pure (no Streamlit) so they can be unit-tested. Blank limit = autoscale.
"""

from __future__ import annotations

import math
import re

import streamlit as st

SCALES = ["Linear", "Log"]


def _pair(text):
    """Parse a 'min, max' axis-range box into a (lo, hi) tuple.

    Comma- or space-separated; either side may be blank for autoscale on that
    side. A single value (ambiguous — min or max?) or unparseable text autoscales
    both. So '1, 1e9' -> (1.0, 1e9); '1,' -> (1.0, None); ', 1e9' -> (None, 1e9);
    '' -> (None, None).
    """
    text = (text or "").strip()
    if not text:
        return (None, None)
    parts = re.split(r"[,\s]+", text)
    if len(parts) < 2:
        return (None, None)  # single value is ambiguous — ignore

    def _num(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    return (_num(parts[0]), _num(parts[1]))


def plot_axis_controls(key_prefix, *, default_x="Linear", default_y="Log",
                       label="Plot options"):
    """Compact axis controls (scale + limits) for a plot, returned as an options
    dict for apply_axis_mpl/apply_axis_plotly.

    Rendered behind a *keyed* checkbox rather than an expander: a keyed widget's
    state persists across reruns, so typing a limit and pressing Enter no longer
    collapses the panel. Limits are one 'min, max' field per axis (both together)
    rather than four separate boxes.
    """
    defaults = {"x_scale": default_x, "y_scale": default_y,
                "xlim": (None, None), "ylim": (None, None)}
    if not st.checkbox(label, key=f"{key_prefix}_show"):
        return defaults
    c1, c2 = st.columns(2)
    x_scale = c1.selectbox("X axis", SCALES, index=SCALES.index(default_x),
                           key=f"{key_prefix}_xs")
    y_scale = c2.selectbox("Y axis", SCALES, index=SCALES.index(default_y),
                           key=f"{key_prefix}_ys")
    c3, c4 = st.columns(2)
    xr = c3.text_input("X limits (min, max)", key=f"{key_prefix}_xr",
                       placeholder="auto — e.g. 0, 48")
    yr = c4.text_input("Y limits (min, max)", key=f"{key_prefix}_yr",
                       placeholder="auto — e.g. 1, 1e9")
    return {"x_scale": x_scale, "y_scale": y_scale, "xlim": _pair(xr), "ylim": _pair(yr)}


def apply_axis_mpl(ax, opts):
    """Apply scale + limits to a matplotlib Axes. Scale is set explicitly (so it
    overrides a prior ``semilogy``); partial limits autoscale the unset side."""
    ax.set_xscale("log" if opts["x_scale"] == "Log" else "linear")
    ax.set_yscale("log" if opts["y_scale"] == "Log" else "linear")
    xlo, xhi = opts["xlim"]
    ylo, yhi = opts["ylim"]
    if xlo is not None or xhi is not None:
        ax.set_xlim(left=xlo, right=xhi)
    if ylo is not None or yhi is not None:
        ax.set_ylim(bottom=ylo, top=yhi)
    return ax


def _plotly_range(axis_type, lo, hi):
    """Plotly needs BOTH bounds, and a log axis takes log10-space values.
    Returns None (autorange) unless both bounds are valid for the axis type."""
    if lo is None or hi is None:
        return None
    if axis_type == "log":
        if lo <= 0 or hi <= 0:
            return None
        return [math.log10(lo), math.log10(hi)]
    return [lo, hi]


def apply_axis_plotly(fig, opts):
    """Apply scale + limits to a plotly Figure (all axes)."""
    xt = "log" if opts["x_scale"] == "Log" else "linear"
    yt = "log" if opts["y_scale"] == "Log" else "linear"
    fig.update_xaxes(type=xt)
    fig.update_yaxes(type=yt)
    xr = _plotly_range(xt, *opts["xlim"])
    yr = _plotly_range(yt, *opts["ylim"])
    if xr is not None:
        fig.update_xaxes(range=xr)
    if yr is not None:
        fig.update_yaxes(range=yr)
    return fig


# ── Matplotlib theme (for the AI Assistant's agent-generated figures) ──────────
# The app's own charts are Plotly; the AI Assistant still renders agent-written
# matplotlib via st.pyplot. Rather than touch that generated code, we apply a
# global rcParams theme so any figure it produces matches the app's palette.
import os as _os

import matplotlib as _mpl
from matplotlib import font_manager as _fm
from cycler import cycler as _cycler

_FONT_DIR = _os.path.join(_os.path.dirname(__file__), "static", "fonts")
_PLEX_TTFS = (
    "IBMPlexSans-Regular.ttf", "IBMPlexSans-Medium.ttf",
    "IBMPlexSans-SemiBold.ttf", "IBMPlexSans-Bold.ttf",
)
_FONTS_REGISTERED = False


def _register_plex_fonts():
    """Register the bundled IBM Plex Sans TTFs with matplotlib, once per process.
    matplotlib cannot read the .woff2 used for the web CSS, so TTF copies live
    alongside them. Returns True if 'IBM Plex Sans' is available afterwards."""
    global _FONTS_REGISTERED
    if not _FONTS_REGISTERED:
        for fn in _PLEX_TTFS:
            p = _os.path.join(_FONT_DIR, fn)
            if _os.path.exists(p):
                try:
                    _fm.fontManager.addfont(p)
                except Exception:
                    pass
        _FONTS_REGISTERED = True
    return any(f.name == "IBM Plex Sans" for f in _fm.fontManager.ttflist)


def apply_mpl_theme():
    """Apply the app's design system to matplotlib via rcParams, so agent-generated
    figures (rendered by the AI Assistant via st.pyplot) match the Plotly charts.
    Call once at startup — takes effect purely through global rcParams; no plotting
    code needs to change. Falls back to system sans if IBM Plex isn't registered."""
    has_plex = _register_plex_fonts()
    family = (["IBM Plex Sans"] if has_plex else []) + ["DejaVu Sans", "Arial", "sans-serif"]
    _mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": family,
        "text.color": "#16211f",
        "axes.labelcolor": "#16211f",
        "axes.titlecolor": "#16211f",
        "axes.edgecolor": "#d3dbd8",
        "axes.linewidth": 0.8,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e4e8e6",
        "grid.linewidth": 0.8,
        "xtick.color": "#66756f",
        "ytick.color": "#66756f",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": _cycler(color=["#0d7a68", "#5457a6", "#c1873a"]),
    })
    # Tick-label colour is a separate rcParam on matplotlib >= 3.4.
    for _k in ("xtick.labelcolor", "ytick.labelcolor"):
        if _k in _mpl.rcParams:
            _mpl.rcParams[_k] = "#66756f"


# ── Plottable-series registry + selector (approach A: per-category checkboxes) ──
# A single source of truth for "what can be plotted" from a SimulationResult, so
# every module offers the same entity choices and adding an engine state surfaces
# it automatically. Derived from result.state_names + named aggregates.
import numpy as _np
from dataclasses import dataclass as _dataclass


@_dataclass
class Series:
    key: str                # stable id, unique within a result
    label: str              # human-readable legend label
    category: str           # Bacteria | Phage | Immune | Nutrient | Antibiotic | OD
    getter: object          # callable(result) -> 1D array
    default: bool = False    # on by default (defaults reproduce the pre-registry view)
    log: bool = True         # density-like (floored for a log axis)
    dash: str = "solid"      # solid | dash | dot | dashdot
    width: float = 2.0
    color: str = None


def _sum_states(result, keys):
    """Sum a set of named states, skipping any that don't exist. Zeros if none."""
    tot = None
    for k in keys:
        try:
            v = _np.asarray(result.get(k), dtype=float)
        except Exception:
            continue
        tot = v if tot is None else tot + v
    return tot if tot is not None else _np.zeros_like(result.time, dtype=float)


# Bacterial "basis" definitions for outcome metrics and sweep overlays. CFU counts only the
# CULTURABLE compartments (active B + dormant D) — infected (I) and hibernating (H) cells do
# not form colonies on a plate. "Active only" is B alone (e.g. a short assay that misses
# persisters). Keys are user-facing labels; values are prefixes for result.sum_prefixes(*v).
BACTERIAL_BASES = {
    "CFU (culturable: B+D)": ("B", "D"),
    "Total incl. infected (B+D+I+H)": ("B", "D", "I", "H"),
    "Active only (B)": ("B",),
}
CFU_BASIS_LABEL = "CFU (culturable: B+D)"
CFU_PREFIXES = ("B", "D")


def bacterial_total(result, basis_label=CFU_BASIS_LABEL):
    """Total bacteria for a chosen basis label (defaults to culturable CFU = B+D)."""
    prefixes = BACTERIAL_BASES.get(basis_label, CFU_PREFIXES)
    return result.sum_prefixes(*prefixes)


def build_series(result, *, config, strains, phages, antibiotics, builder_mode):
    """Enumerate the plottable series for a result, grouped by category.

    Defaults reproduce the previous fixed view (Total viable + per-strain active +
    per-strain dormant; per-phage free + blood), so nothing regresses; the extra
    compartments (infected I, hibernating H, total free phage) are opt-in.
    """
    n_latent = getattr(config, "n_latent", 1)
    n_depth = getattr(config, "n_depth", 1)
    n_phages = getattr(config, "n_phages", 0)
    out = []

    # ── Bacteria ── aggregate views. CFU = only the CULTURABLE compartments (active B +
    # dormant D); infected (I) and hibernating (H) cells don't form colonies on a plate, so
    # they are excluded from CFU and shown only in the "total incl. infected" series.
    out.append(Series("cfu", "CFU (culturable: B+D)", "Bacteria",
                      lambda r: r.sum_prefixes("B", "D"),
                      default=True, log=True, dash="solid", width=3.0, color="#16211f"))
    out.append(Series("total_viable", "Total incl. infected (B+D+I+H)", "Bacteria",
                      lambda r: r.sum_prefixes("B", "D", "I", "H"),
                      default=False, log=True, dash="solid", width=1.5, color="#6b7770"))
    out.append(Series("total_active", "Active only (B)", "Bacteria",
                      lambda r: r.sum_prefixes("B"),
                      default=False, log=True, dash="solid", width=1.5, color="#0d7a68"))

    if builder_mode == "Binary Genotypes (BRG)":
        import itertools
        combs = list(itertools.product([0, 1], repeat=len(phages) + len(antibiotics)))
        for j, comb in enumerate(combs):
            if len(antibiotics) == 0:
                lbl = "".join(map(str, comb))
            else:
                p_lbl = "".join(map(str, comb[:len(phages)])) if len(phages) > 0 else ""
                a_lbl = "".join(map(str, comb[len(phages):]))
                lbl = f"phi{p_lbl}_abx{a_lbl}" if len(phages) > 0 else f"abx{a_lbl}"
            out.append(Series(f"B{j}", lbl, "Bacteria",
                              (lambda r, j=j: r.get(f"B{j}")),
                              default=True, log=True, dash="dash"))
    else:
        for j in range(len(strains)):
            name = strains[j].get("name", f"Strain {j}")
            out.append(Series(f"B{j}", f"{name} (active)", "Bacteria",
                              (lambda r, j=j: r.get(f"B{j}")),
                              default=True, log=True, dash="dash"))
            if strains[j].get("dormancy_enabled", False):
                d_keys = [f"D{q}_{j}" for q in range(n_depth)]
                out.append(Series(f"D_{j}", f"{name} (dormant)", "Bacteria",
                                  (lambda r, ks=d_keys: _sum_states(r, ks)),
                                  default=True, log=True, dash="dot"))
                if n_phages > 0:
                    h_keys = [f"H{q}_{l}_{j}_{m}" for q in range(n_depth)
                              for l in range(n_latent) for m in range(n_phages)]
                    out.append(Series(f"H_{j}", f"{name} (hibernating)", "Bacteria",
                                      (lambda r, ks=h_keys: _sum_states(r, ks)),
                                      default=False, log=True, dash="dashdot"))
            if n_phages > 0:
                i_keys = [f"I{l}_{j}_{m}" for l in range(n_latent) for m in range(n_phages)]
                out.append(Series(f"I_{j}", f"{name} (infected)", "Bacteria",
                                  (lambda r, ks=i_keys: _sum_states(r, ks)),
                                  default=False, log=True, dash="dashdot"))

    # ── Phage ──
    if n_phages > 0:
        out.append(Series("P_total", "Total free phage", "Phage",
                          lambda r: r.sum_prefixes("P"),
                          default=False, log=True, dash="solid", width=1.5, color="#8a9591"))
        for j in range(len(phages)):
            name = phages[j].get("name", f"Phage {j}")
            out.append(Series(f"P{j}", f"{name} (infection site)", "Phage",
                              (lambda r, j=j: r.get(f"P{j}")), default=True, log=True))
            if phages[j].get("pk_mode", "None") != "None":
                Vc = phages[j].get("Vc", 5000.0)
                out.append(Series(f"Pc{j}", f"{name} (central/blood conc.)", "Phage",
                                  (lambda r, j=j, Vc=Vc: r.get(f"Pc{j}") / Vc),
                                  default=True, log=True, dash="dash"))
                # 2-compartment: peripheral tissue amount (no volume in the engine).
                if float(phages[j].get("k12", 0.0)) > 0:
                    out.append(Series(f"Pp{j}", f"{name} (peripheral tissue, amount)", "Phage",
                                      (lambda r, j=j: r.get(f"Pp{j}")),
                                      default=False, log=True, dash="dot"))
    return out


def series_selector(key_prefix, series, *, ncol=3):
    """Category checkbox picker (keyed → state persists across reruns). Returns the
    chosen Series list. Renders nothing/[] for an empty category."""
    if not series:
        return []
    chosen = []
    cols = st.columns(min(ncol, len(series)))
    for i, s in enumerate(series):
        on = cols[i % len(cols)].checkbox(s.label, value=s.default, key=f"{key_prefix}_{s.key}")
        if on:
            chosen.append(s)
    return chosen


def plot_series(result, chosen, opts, *, title, y_label, height=440):
    """Build a single-axis Plotly figure from the chosen Series and apply the axis
    controls. All series in one call share a unit (one tab = one unit family)."""
    import plotly.graph_objects as go
    t = result.time
    floor = opts.get("y_scale") == "Log"
    fig = go.Figure()
    for s in chosen:
        y = _np.asarray(s.getter(result), dtype=float)
        if floor and s.log:
            y = _np.maximum(y, 1.0)
        line = dict(width=s.width, dash=s.dash)
        if s.color:
            line["color"] = s.color
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=s.label, line=line))
    fig.update_layout(title=title, xaxis_title="Time (hours)", yaxis_title=y_label,
                      template="plotly_white", height=height, margin=dict(t=48, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0))
    apply_axis_plotly(fig, opts)
    return fig


def plot_sweep_traces(traces, key_prefix, *, title_suffix="across runs"):
    """Multi-run 'trace' plot: a selectbox picks ONE metric to show across every run
    (overlaying N metrics x M runs would be unreadable — see approach A, phase 2).

    ``traces`` is an ordered dict ``{metric_label: (scale, [(t, y, run_label), ...])}``
    where scale is "log" or "linear". Renders nothing for an empty dict.
    """
    import plotly.graph_objects as go
    if not traces:
        return
    name = st.selectbox("Trace", list(traces), key=f"{key_prefix}_metric")
    scale, series = traces[name]
    is_log = scale == "log"
    st.markdown(f"#### {name} {title_suffix}")
    fig = go.Figure()
    for t_arr, y_arr, lbl in series:
        y = _np.maximum(y_arr, 1.0) if is_log else y_arr
        fig.add_trace(go.Scatter(x=t_arr, y=y, mode="lines", name=lbl))
    fig.update_layout(xaxis_title="Time (hours)", yaxis_title=name, template="plotly_white",
                      height=440, margin=dict(t=20, b=40))
    apply_axis_plotly(fig, plot_axis_controls(f"{key_prefix}_axis",
                                              default_y="Log" if is_log else "Linear"))
    st.plotly_chart(fig, width="stretch")
