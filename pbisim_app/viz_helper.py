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

import streamlit as st

SCALES = ["Linear", "Log"]


def _num(v):
    """Parse a limit text box: blank/invalid -> None (autoscale)."""
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def plot_axis_controls(key_prefix, *, default_x="Linear", default_y="Log",
                       label="Plot options", expanded=False):
    """Render the axis-control expander and return an options dict.

    Note: these are view preferences rendered per page; like other Streamlit
    widgets they reset to defaults if you navigate away and back.
    """
    with st.expander(label, expanded=expanded):
        c1, c2 = st.columns(2)
        with c1:
            x_scale = st.selectbox("X axis", SCALES, index=SCALES.index(default_x),
                                   key=f"{key_prefix}_xs")
        with c2:
            y_scale = st.selectbox("Y axis", SCALES, index=SCALES.index(default_y),
                                   key=f"{key_prefix}_ys")
        st.caption("Axis limits — leave blank to autoscale.")
        c3, c4, c5, c6 = st.columns(4)
        with c3:
            xmin = st.text_input("X min", key=f"{key_prefix}_xmin")
        with c4:
            xmax = st.text_input("X max", key=f"{key_prefix}_xmax")
        with c5:
            ymin = st.text_input("Y min", key=f"{key_prefix}_ymin")
        with c6:
            ymax = st.text_input("Y max", key=f"{key_prefix}_ymax")
    return {
        "x_scale": x_scale,
        "y_scale": y_scale,
        "xlim": (_num(xmin), _num(xmax)),
        "ylim": (_num(ymin), _num(ymax)),
    }


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
