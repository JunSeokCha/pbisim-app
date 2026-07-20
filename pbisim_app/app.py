"""
app.py — Streamlit UI for the pbisim simulation engine.
Provides a visual, interactive simulation builder, a presets browser,
an AI assistant, and a clinical trials cohort simulator.
"""

from __future__ import annotations

# Dump a Python traceback to stderr on a native crash (SIGSEGV/SIGABRT) so segfaults
# surface in the server logs as an actionable stack, not a bare "exited with status 139".
import faulthandler
faulthandler.enable()

import copy
import dataclasses as _dc
import io
import json
import os
import re

# Force the non-interactive backend BEFORE importing pyplot. On a headless server
# (e.g. the Render container) a GUI backend used from Streamlit's script thread
# segfaults (exit 139); Agg is thread-safe and display-free.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Precision fix for float number inputs ─────────────────────────────────────
# Streamlit's number_input infers a display format from the step (default "%.2f"
# for floats), so a value like 0.001 rounds to 0.00 on entry — losing precision
# for small rates (death, dormancy, resuscitation, decay, …). Inject a compact
# full-precision format ("%g") for any *float* input that doesn't set one. This
# auto-skips integer inputs (their value is an int) and any input that already
# specifies `format=` (e.g. scientific "%.1e").
_orig_number_input = st.number_input


def _number_input_precise(label, *args, **kwargs):
    if "format" not in kwargs and isinstance(kwargs.get("value"), float):
        kwargs["format"] = "%g"
    return _orig_number_input(label, *args, **kwargs)


st.number_input = _number_input_precise

from pbisim import (
    ModelBuilder,
    PBIModel,
    solve_ode,
    DoseSchedule,
    DoseEvent,
    time_to_clearance,
    time_to_log_reduction,
    stationary_phase_ic,
)
from pbisim.strains import StrainDefinition, StrainSet
from pbisim.strains.genotypes import BinaryResistanceGenotypes, BacterialStrain, PhageStrain, Antibiotic
from pbisim.pk.antibiotic import AntibioticDefinition, AntibioticSensitivity
from pbisim.trial.clinical import TreatmentArm

from pbisim_app.agent import SimulationAgent
from pbisim_app.executor import execute_code
from pbisim_app.fit_helper import (
    OBSERVABLES,
    predicted_observable,
    normalize_fit_dataframe,
    apply_row_filters,
    aggregate_observations,
    fit_residual,
    STRAIN_TUNABLES,
    STRAIN_DORMANCY_TUNABLES,
    PHAGE_TUNABLES,
    PHAGE_OPTIONAL_TUNABLES,
    ADSORPTION_PHAGE_KEYS,
    entity_param_key,
)
from pbisim_app.trial_helper import (
    IIV_PARAMETERS,
    run_trial_simulation,
    plot_kaplan_meier_plotly,
    plot_metric_distributions_plotly,
    plot_pkpd_trajectories_plotly,
    build_regimen_doses,
)
from pbisim_app.sweep_helper import (
    get_sweep_parameters,
    apply_sweep_parameter,
    parse_comma_separated_series,
    pad_vectors,
)


# ── Dose defaults per target compartment ───────────────────────────────────────
# 1e8 only makes sense for phage (PFU); antibiotics are dosed in mg and nutrient
# in resource units, so give each target a plausible default and unit label.
DOSE_AMOUNT_DEFAULTS = {"phage": 1e8, "antibiotic": 10.0, "nutrient": 1.0}
DOSE_AMOUNT_LABELS = {
    "phage": "Amount (PFU)",
    "antibiotic": "Amount (mg)",
    "nutrient": "Amount (resource units)",
}


def render_regimen_config(prefix, items, target, default_amount, unit_label,
                          default_on=True, initial=None):
    """Render dose-regimen widgets for one agent and return a config dict.

    ``initial`` (an existing config dict) pre-fills the widgets so an arm can be edited
    in place. Returns ``{"on": False}`` when the agent is not included in this arm, else
    ``{"on": True, "index", "amount", "start", "repeat", "interval", "n"}``.
    """
    init = initial or {}
    on = st.checkbox(f"Include {target}", value=bool(init.get("on", default_on)), key=f"{prefix}_on")
    if not on:
        return {"on": False}
    idx = int(init.get("index", 0))
    if len(items) > 1:
        idx = st.selectbox(
            f"{target.title()} target", range(len(items)),
            index=min(idx, len(items) - 1),
            format_func=lambda i: items[i].get("name", f"{target} {i}"),
            key=f"{prefix}_idx",
        )
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input(unit_label, min_value=0.0,
                                 value=float(init.get("amount", default_amount)),
                                 format="%.1e", key=f"{prefix}_amt")
    with c2:
        start = st.number_input("Start (h)", min_value=0.0,
                                value=float(init.get("start", 0.0)), step=1.0,
                                key=f"{prefix}_start")
    repeat = st.checkbox("Repeat regimen (qX h × N)", value=bool(init.get("repeat", False)),
                         key=f"{prefix}_rep")
    interval, n_doses = float(init.get("interval", 8.0)), int(init.get("n", 4))
    if repeat:
        c3, c4 = st.columns(2)
        with c3:
            interval = st.number_input("Interval (h)", min_value=0.5, value=interval,
                                       step=1.0, key=f"{prefix}_int")
        with c4:
            n_doses = st.number_input("Doses", min_value=1, value=n_doses, step=1,
                                      key=f"{prefix}_n")
    return {"on": True, "index": int(idx), "amount": float(amount),
            "start": float(start), "repeat": bool(repeat),
            "interval": float(interval), "n": int(n_doses)}


def render_iiv_config(prefix, initial=None):
    """Render the inter-individual-variability (IIV) form and return an iiv dict
    ``{path, dist_type, params, mode}``. ``initial`` pre-fills the widgets for editing."""
    init = initial or {}
    names = list(IIV_PARAMETERS.keys())
    cur_name = next((n for n, p in IIV_PARAMETERS.items() if p == init.get("path")), names[0])
    param_display = st.selectbox("Select Parameter", names,
                                 index=names.index(cur_name) if cur_name in names else 0,
                                 key=f"{prefix}_param")
    dists = ["LogNormal", "Normal", "Uniform"]
    cur_dist = init.get("dist_type", "LogNormal")
    dist_choice = st.selectbox("Distribution Type", dists,
                               index=dists.index(cur_dist) if cur_dist in dists else 0,
                               key=f"{prefix}_dist")
    ip = init.get("params", {})
    c1, c2 = st.columns(2)
    params = {}
    if dist_choice == "LogNormal":
        with c1:
            params["cv"] = st.number_input("CV (coefficient of variation)",
                                           value=float(ip.get("cv", 0.25)), min_value=0.01,
                                           key=f"{prefix}_cv")
        mode = "multiplicative"
    elif dist_choice == "Normal":
        with c1:
            params["mean"] = st.number_input("Mean", value=float(ip.get("mean", 0.0)),
                                             key=f"{prefix}_mean")
        with c2:
            params["sd"] = st.number_input("SD (standard deviation)",
                                           value=float(ip.get("sd", 0.1)), min_value=0.01,
                                           key=f"{prefix}_sd")
        mode = "additive"
    else:
        with c1:
            params["lo"] = st.number_input("Lower Bound", value=float(ip.get("lo", 0.5)),
                                           key=f"{prefix}_lo")
        with c2:
            params["hi"] = st.number_input("Upper Bound", value=float(ip.get("hi", 1.5)),
                                           key=f"{prefix}_hi")
        mode = "replace"
    return {"path": IIV_PARAMETERS[param_display], "dist_type": dist_choice,
            "params": params, "mode": mode}


def render_mutation_graph_editor(strains, key_prefix):
    """Edit the named mutation-transition graph (shared `int_transitions`).

    Each entry is {"from": strain_name, "to": strain_name, "rate": mu}. Works for
    any number of strains, so it lifts the 2^m restriction of the per-locus shortcut.
    """
    transitions = st.session_state.get("int_transitions", [])
    names = [s["name"] for s in strains]
    for idx, tr in enumerate(list(transitions)):
        c1, c2, c3, c4 = st.columns([3, 3, 3, 1])
        with c1:
            tr["from"] = st.selectbox("From", names,
                                      index=names.index(tr["from"]) if tr.get("from") in names else 0,
                                      key=f"{key_prefix}_src_{idx}")
        with c2:
            tr["to"] = st.selectbox("To", names,
                                    index=names.index(tr["to"]) if tr.get("to") in names else 0,
                                    key=f"{key_prefix}_dest_{idx}")
        with c3:
            tr["rate"] = st.number_input("Rate (mu)", value=float(tr.get("rate", 1e-7)),
                                         format="%.2e", key=f"{key_prefix}_rate_{idx}")
        with c4:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"{key_prefix}_del_{idx}"):
                transitions.pop(idx)
                st.session_state.int_transitions = transitions
                st.rerun()
    if st.button("➕ Add transition", key=f"{key_prefix}_add"):
        transitions.append({"from": names[0] if names else "", "to": names[0] if names else "", "rate": 1e-7})
        st.session_state.int_transitions = transitions
        st.rerun()


def mutation_matrix_from_transitions(transitions, strains):
    """Build the (n,n) mass-conserving mutation matrix from a named transition graph.

    Convention (matches pbisim): M[dest, origin] = rate origin→dest; the diagonal
    M[o, o] = -(sum of outflows from o). Returns None if there are no valid edges.
    """
    n = len(strains)
    name_to_idx = {s["name"]: i for i, s in enumerate(strains)}
    M = np.zeros((n, n))
    any_edge = False
    for tr in transitions:
        o = name_to_idx.get(tr.get("from"))
        d = name_to_idx.get(tr.get("to"))
        r = float(tr.get("rate", 0.0))
        if o is not None and d is not None and o != d and r > 0:
            M[d, o] += r
            M[o, o] -= r
            any_edge = True
    return M if any_edge else None


# ── Cached calibration data processing ────────────────────────────────────────
# The Calibration page re-runs on every widget interaction; without caching it would
# re-parse the CSV and re-filter/normalise/aggregate the whole dataset each time.
# @st.cache_data memoises by content, so these recompute only when inputs change.
@st.cache_data(show_spinner=False)
def read_uploaded_csv(file):
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def calibration_processed(raw, filters_key, time_col, value_col, observable,
                          group_cols, moi_col, stat, band):
    """Filter → normalise → aggregate, cached. Keys must be hashable (tuples)."""
    filtered = apply_row_filters(raw, {c: list(v) for c, v in filters_key})
    long, conds = normalize_fit_dataframe(filtered, time_col, value_col, observable,
                                          list(group_cols), moi_col)
    agg = aggregate_observations(long, stat=stat, band=band)
    return filtered, long, conds, agg


def arm_dose_events(arm):
    """Build the DoseEvent list for a treatment-arm config from its regimens."""
    doses = []
    for key, target in (("phage", "phage"), ("abx", "antibiotic")):
        c = arm.get(key)
        if c and c.get("on"):
            doses += build_regimen_doses(
                target, c["index"], c["amount"], c["start"],
                c["repeat"], c["interval"], c["n"],
            )
    return doses


def arm_regimen_summary(arm):
    """One-line human summary of an arm's dosing."""
    parts = []
    for key, label in (("phage", "P"), ("abx", "A")):
        c = arm.get(key)
        if c and c.get("on"):
            reg = f"q{c['interval']:g}h×{c['n']}" if c.get("repeat") else "single"
            parts.append(f"{label} {c['amount']:.1e} @ {c['start']:g}h ({reg})")
    return ", ".join(parts) if parts else "no doses (control-like)"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="pbisim — Phage-Bacteria Simulation Control Center",
    page_icon="🦠",
    layout="wide",
)


# ── Custom CSS for Premium Aesthetics ─────────────────────────────────────────
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

theme_mode = st.session_state["theme_mode"]

FONT_FACE_CSS = """
@font-face{font-family:'IBM Plex Sans';font-style:normal;font-weight:400;font-display:swap;src:url('app/static/fonts/IBMPlexSans-Regular.woff2') format('woff2');}
@font-face{font-family:'IBM Plex Sans';font-style:normal;font-weight:500;font-display:swap;src:url('app/static/fonts/IBMPlexSans-Medium.woff2') format('woff2');}
@font-face{font-family:'IBM Plex Sans';font-style:normal;font-weight:600;font-display:swap;src:url('app/static/fonts/IBMPlexSans-SemiBold.woff2') format('woff2');}
@font-face{font-family:'IBM Plex Mono';font-style:normal;font-weight:400;font-display:swap;src:url('app/static/fonts/IBMPlexMono-Regular.woff2') format('woff2');}
@font-face{font-family:'IBM Plex Mono';font-style:normal;font-weight:500;font-display:swap;src:url('app/static/fonts/IBMPlexMono-Medium.woff2') format('woff2');}
"""

if theme_mode == "Light":
    css_content = """
    :root{
      --canvas:#faf9f5; --panel:#f6f8f7; --card:#ffffff; --border:#e4e8e6;
      --hair:#eef1f0; --ink:#16211f; --muted:#66756f; --label:#93a09b;
      --teal:#0d7a68; --teal-d:#0a6355; --teal-tint:#eef3f1; --accent:#5457a6;
    }

    html, body, [class*="css"], .stApp, button, input, select, textarea, [data-baseweb] {
      font-family:'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }

    .stApp p, .stApp span, .stApp li, .stApp label, .stApp label p,
    .stApp [data-testid="stWidgetLabel"] p { color: var(--ink); }
    input, select, textarea, [data-baseweb="select"] div { color: var(--ink); }

    /* Headings: solid ink, no gradient */
    h1, h2, h3, h4 {
      color: var(--ink) !important; font-weight:600; letter-spacing:-0.01em;
      -webkit-text-fill-color: var(--ink); background: none;
    }
    h1 { font-size:1.7rem; } h2 { font-size:1.25rem; } h3 { font-size:1.05rem; }

    /* Mono for data / values / code */
    .metric-value, .mono, code, kbd, pre, [data-testid="stMetricValue"] {
      font-family:'IBM Plex Mono', ui-monospace, monospace;
    }

    .section-label { font-size:11px; text-transform:uppercase; letter-spacing:0.13em;
      color: var(--label); font-weight:600; }

    .card { background: var(--card); border:1px solid var(--border); border-radius:6px;
      padding:18px 20px; margin-bottom:16px; box-shadow:none; backdrop-filter:none; }
    .card h4 { margin-top:0; color: var(--ink); font-weight:600;
      border-bottom:1px solid var(--hair); padding-bottom:8px; }

    .metric-container { text-align:left; background: var(--card); padding:14px 16px;
      border:1px solid var(--border); border-radius:6px; }
    .metric-label { font-size:10.5px; color: var(--label); font-weight:600;
      text-transform:uppercase; letter-spacing:0.13em; }
    .metric-value { font-size:1.5rem; font-weight:600; color: var(--ink); margin-top:6px; }
    .metric-sub { font-size:11px; color: var(--label); margin-top:3px; }

    /* Buttons: flat teal, no gradient/lift */
    div.stButton > button {
      background: var(--teal); color:#fff; border:1px solid var(--teal);
      border-radius:6px; padding:8px 20px; font-weight:600; box-shadow:none;
      transition: background .15s ease;
    }
    div.stButton > button:hover { background: var(--teal-d); border-color: var(--teal-d);
      transform:none; box-shadow:none; }

    .preset-card { background: var(--panel); border:1px solid var(--border);
      border-radius:6px; padding:16px; height:100%; transition:border-color .15s ease; }
    .preset-card:hover { border-color: var(--teal); }

    section[data-testid="stSidebar"] { background: var(--panel);
      border-right:1px solid var(--border); }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label { color: var(--ink); }

    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span { color: var(--muted);
      font-weight:600; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: var(--teal); }
    [data-baseweb="tab-highlight"] { background-color: var(--teal); }

    .info-banner { background: var(--teal-tint); border-left:3px solid var(--teal);
      padding:12px 16px; border-radius:4px; margin-bottom:20px; color: var(--ink); }
    """
else:
    css_content = """
    :root{
      --canvas:#111a17; --panel:#17211d; --card:#1b2723; --border:#2c3a35;
      --hair:#26332f; --ink:#e9efec; --muted:#9aa8a2; --label:#7c8a85;
      --teal:#159b83; --teal-d:#12876f; --teal-tint:#16261f; --accent:#8184b8;
    }

    html, body, [class*="css"], .stApp, button, input, select, textarea, [data-baseweb] {
      font-family:'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp { background: var(--canvas); color: var(--ink); }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }

    .stApp p, .stApp span, .stApp li, .stApp label, .stApp label p,
    .stApp [data-testid="stWidgetLabel"] p { color: var(--ink); }
    input, select, textarea, [data-baseweb="select"] div { color: var(--ink); }

    /* Headings: solid ink, no gradient */
    h1, h2, h3, h4 {
      color: var(--ink) !important; font-weight:600; letter-spacing:-0.01em;
      -webkit-text-fill-color: var(--ink); background: none;
    }
    h1 { font-size:1.7rem; } h2 { font-size:1.25rem; } h3 { font-size:1.05rem; }

    /* Mono for data / values / code */
    .metric-value, .mono, code, kbd, pre, [data-testid="stMetricValue"] {
      font-family:'IBM Plex Mono', ui-monospace, monospace;
    }

    .section-label { font-size:11px; text-transform:uppercase; letter-spacing:0.13em;
      color: var(--label); font-weight:600; }

    .card { background: var(--card); border:1px solid var(--border); border-radius:6px;
      padding:18px 20px; margin-bottom:16px; box-shadow:none; backdrop-filter:none; }
    .card h4 { margin-top:0; color: var(--ink); font-weight:600;
      border-bottom:1px solid var(--hair); padding-bottom:8px; }

    .metric-container { text-align:left; background: var(--card); padding:14px 16px;
      border:1px solid var(--border); border-radius:6px; }
    .metric-label { font-size:10.5px; color: var(--label); font-weight:600;
      text-transform:uppercase; letter-spacing:0.13em; }
    .metric-value { font-size:1.5rem; font-weight:600; color: var(--ink); margin-top:6px; }
    .metric-sub { font-size:11px; color: var(--label); margin-top:3px; }

    /* Buttons: flat teal, no gradient/lift */
    div.stButton > button {
      background: var(--teal); color:#fff; border:1px solid var(--teal);
      border-radius:6px; padding:8px 20px; font-weight:600; box-shadow:none;
      transition: background .15s ease;
    }
    div.stButton > button:hover { background: var(--teal-d); border-color: var(--teal-d);
      transform:none; box-shadow:none; }

    .preset-card { background: var(--panel); border:1px solid var(--border);
      border-radius:6px; padding:16px; height:100%; transition:border-color .15s ease; }
    .preset-card:hover { border-color: var(--teal); }

    section[data-testid="stSidebar"] { background: var(--panel);
      border-right:1px solid var(--border); }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label { color: var(--ink); }

    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span { color: var(--muted);
      font-weight:600; }
    button[data-baseweb="tab"][aria-selected="true"] p { color: var(--teal); }
    [data-baseweb="tab-highlight"] { background-color: var(--teal); }

    .info-banner { background: var(--teal-tint); border-left:3px solid var(--teal);
      padding:12px 16px; border-radius:4px; margin-bottom:20px; color: var(--ink); }
    """

st.markdown(f"<style>{FONT_FACE_CSS}{css_content}</style>", unsafe_allow_html=True)


# ── State Initialization ──────────────────────────────────────────────────────
def _init_app_state():
    if "agent" not in st.session_state:
        st.session_state.agent = SimulationAgent()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Interactive Simulator"
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None
    if "simulation_config" not in st.session_state:
        st.session_state.simulation_config = None
    if "int_builder_mode" not in st.session_state:
        st.session_state.int_builder_mode = "Direct (ModelBuilder)"
    if "api_models_list" not in st.session_state:
        st.session_state.api_models_list = []
    
    # Custom builders states
    if "int_transitions" not in st.session_state:
        st.session_state.int_transitions = []
    if "int_brg_initial_B" not in st.session_state:
        st.session_state.int_brg_initial_B = {}
        
    # Clinical trial states
    if "trial_iiv_inputs" not in st.session_state:
        st.session_state.trial_iiv_inputs = []
    if "trial_arms" not in st.session_state:
        st.session_state.trial_arms = []
    if "trial_result" not in st.session_state:
        st.session_state.trial_result = None
    if "user_scenarios" not in st.session_state:
        # name -> {"annotation": str, "schema_version": int, "state": {...}}
        st.session_state.user_scenarios = {}
    if "parts_library" not in st.session_state:
        # {category: {name: {"source","annotation","reference_host?","params"}}}
        st.session_state.parts_library = {"bacteria": {}, "phages": {}, "antibiotics": {}}
    if "fit_dataset" not in st.session_state:
        # {"long": DataFrame[time,arm,observable,value], "conditions": {arm: {"moi": float}}}
        st.session_state.fit_dataset = None


_init_app_state()


def _next_uid(prefix: str) -> str:
    """A session-stable unique id so per-row widget keys (dose / trial-arm editors)
    survive reorder/delete instead of being re-bound to a different row by list index."""
    st.session_state["_uid_counter"] = st.session_state.get("_uid_counter", 0) + 1
    return f"{prefix}{st.session_state['_uid_counter']}"


def _carry_prerun_debris(ic, kwargs):
    """Carry the OD/debris that accumulated during the stationary-phase pre-run into the
    treatment model — when the user opts to INHERIT it (default) rather than wash the dead
    cells out before treatment. No-op when debris isn't tracked (ic.Debris is None) or the
    'inherit' checkbox is unticked (washout)."""
    if st.session_state.get("int_prerun_inherit_debris", True) and getattr(ic, "Debris", None) is not None:
        kwargs["initial_Debris"] = ic.Debris


def _safe_od(result, total_bacteria):
    """OD trajectory that never raises. Uses the debris-aware ``get_od()`` when the debris
    ODE is in the config; otherwise falls back to biomass / conversion factor WITHOUT
    touching the Debris state (accessing it raises ``KeyError: Debris state not found``
    when debris isn't enabled). Guards against a session debris-flag vs. result mismatch."""
    factor = st.session_state.get("int_od_to_cfu_conversion_factor", 2e8) or 1.0
    try:
        return result.get_od()
    except Exception:
        return np.asarray(total_bacteria, dtype=float) / factor


def load_preset_to_state(params: dict):
    """Deep copy preset parameters into st.session_state variables."""
    # 0. Clear old simulation results to prevent dimension mismatch crashes
    st.session_state.simulation_result = None
    st.session_state.simulation_config = None
    st.session_state.int_builder_mode = "Direct (ModelBuilder)"

    # 1. Clear old per-mode config + widget keys to prevent collisions and so a reset
    #    actually clears BRG / StrainSet (not just the Direct builder). Covers the Direct
    #    widget keys, the BRG data + widgets (int_brg_*, brg_*), StrainSet (int_transitions,
    #    ss_*, trans_*, direct_*), and the mode/signal selector widgets (widget_*).
    _clear_prefixes = (
        "strain_", "phage_", "abx_", "dose_", "ads_",
        "int_brg_", "brg_", "ss_", "str_", "direct_", "trans_", "widget_",
    )
    keys_to_clear = [
        k for k in st.session_state.keys()
        if any(k.startswith(p) for p in _clear_prefixes) or k == "int_transitions"
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)

    # 2. Global settings
    st.session_state["int_t_end"] = params.get("t_end", 48.0)
    st.session_state["int_dt"] = params.get("dt", 0.25)
    st.session_state["int_extinction_threshold"] = params.get("extinction_threshold", 1.0)
    st.session_state["int_extinction_check_interval"] = params.get("extinction_check_interval", 0.0)
    st.session_state["int_n_latent"] = params.get("n_latent", 5)
    st.session_state["int_solver_method"] = params.get("solver_method", "BDF")
    st.session_state["int_track_nutrients"] = params.get("track_nutrients", True)
    st.session_state["int_initial_S"] = params.get("initial_S", 1.0)
    st.session_state["int_monod_constant"] = params.get("monod_constant", 0.3)
    st.session_state["int_recycle_fraction"] = params.get("recycle_fraction", 0.0)
    st.session_state["int_s_in"] = params.get("s_in", 0.0)
    st.session_state["int_s_out"] = params.get("s_out", 0.0)
    st.session_state["int_carrying_capacity"] = params.get("carrying_capacity", 1e9)
    # Growth signal: prefer an explicit growth_function; else derive from the legacy
    # track_nutrients flag (True → Monod nutrient growth, False → logistic density).
    _gf = params.get("growth_function")
    if _gf not in ("monod_growth", "logistic_growth", "constant_growth", "monod_logistic_growth"):
        _gf = "monod_growth" if params.get("track_nutrients", True) else "logistic_growth"
    st.session_state["int_growth_function"] = _gf
    st.session_state["int_track_nutrients"] = _gf in ("monod_growth", "monod_logistic_growth")
    st.session_state["int_death_function"] = params.get("death_function_name", "constant_death")
    st.session_state["int_density_total_cells"] = params.get("density_signal_uses_total_cells", False)
    st.session_state["int_superinfection"] = params.get("allow_superinfection", False)
    st.session_state["int_t_prerun"] = params.get("t_prerun", 0.0)
    st.session_state["int_prerun_inherit_debris"] = params.get("prerun_inherit_debris", True)

    # 3. Immunity settings
    st.session_state["int_immunity_enabled"] = params.get("immunity_enabled", False)
    st.session_state["int_innate_kill_rate"] = params.get("innate_kill_rate", 1e7)
    st.session_state["int_innate_kill50"] = params.get("innate_kill50", 1e5)
    st.session_state["int_innate_max"] = params.get("innate_max", 1e7)
    # backward compat: old presets stored "adaptive_decay_rate"
    st.session_state["int_innate_decay_rate"] = params.get(
        "innate_decay_rate", params.get("adaptive_decay_rate", 0.1)
    )
    st.session_state["int_imm_kill_rate_D"] = params.get("innate_kill_rate_D", 0.0)
    # translate legacy "adaptive" (invented by scaffold, not a pbisim module) → "innate"
    _raw_module = params.get("immune_module", "innate")
    st.session_state["int_immune_module"] = "innate" if _raw_module == "adaptive" else _raw_module
    # new fields (missing from older presets → sensible defaults)
    st.session_state["int_imm_stim_rate"] = params.get("imm_stim_rate", 0.1)
    st.session_state["int_imm_stim50"] = params.get("imm_stim50", 1e6)
    st.session_state["int_imm_initial"] = params.get("imm_initial", 0.0)

    # 4. OD & Debris settings
    st.session_state["int_debris_enabled"] = params.get("debris_enabled", False)
    st.session_state["int_debris_u"] = params.get("debris_u", 0.4)
    st.session_state["int_debris_v"] = params.get("debris_v", 0.2)
    st.session_state["int_debris_kdis"] = params.get("debris_kdis", 0.01)
    st.session_state["int_od_to_cfu_conversion_factor"] = params.get("od_to_cfu_conversion_factor", 2e8)

    # 5. Strains list
    strains_list = []
    for i, s in enumerate(params.get("strains", [])):
        strains_list.append(
            {
                "name": s.get("name", f"Strain {i}"),
                "initial_B": s.get("initial_B", 1e7),
                "growth_rate": s.get("growth_rate", 1.2),
                "bacteria_to_resource_ratio": s.get("bacteria_to_resource_ratio", 1e9),
                "death_rate_B": s.get("death_rate_B", 0.0),
                "death_rate_D": s.get("death_rate_D", 0.0),
                "dormancy_enabled": s.get("dormancy_enabled", False),
                "dormancy_depth": s.get("dormancy_depth", 1),
                "dormancy_rate": s.get("dormancy_rate", 0.001),
                "resuscitation_rate": s.get("resuscitation_rate", 0.1),
                "dormancy_diffusion_rate": s.get("dormancy_diffusion_rate", 0.05),
                "dormancy_signal": s.get("dormancy_signal", "nutrient"),
                "resuscitation_signal": s.get("resuscitation_signal", "nutrient"),
                "initial_D": s.get("initial_D", 0.0),
            }
        )
    st.session_state["int_strains"] = strains_list

    # 6. Phages list
    phages_list = []
    for i, p in enumerate(params.get("phages", [])):
        phages_list.append(
            {
                "name": p.get("name", f"Phage {i}"),
                "initial_P": p.get("initial_P", 1e6),
                "burst_sizes": p.get("burst_sizes", 50.0),
                "latent_periods": p.get("latent_periods", 0.5),
                "phage_decay_rates": p.get("phage_decay_rates", 0.1),
                "pk_mode": p.get("pk_mode", "None"),
                "Vc": p.get("Vc", 5000.0),
                "k_elim": p.get("k_elim", 0.2),
                "k_in": p.get("k_in", 0.1),
                "k_out": p.get("k_out", 0.05),
                "Vi": p.get("Vi", 10.0),
                # MM clearances
                "Km_elim": p.get("Km_elim", 0.0),
                "phage_decay_Km": p.get("phage_decay_Km", 0.0),
                # Pseudolysogeny
                "hibernation_rate_s": p.get("hibernation_rate_s", 0.0),
                "hibernation_rate_r": p.get("hibernation_rate_r", 0.0),
                "lytic_resumption_rate_s": p.get("lytic_resumption_rate_s", 0.0),
                "lytic_resumption_rate_r": p.get("lytic_resumption_rate_r", 0.0),
                # BRG specific
                "adsorption_s": p.get("adsorption_s", p.get("adsorption_rates")[0] if isinstance(p.get("adsorption_rates"), (list, np.ndarray)) else (p.get("adsorption_rates") if p.get("adsorption_rates") is not None else 5e-8)),
                "adsorption_r": p.get("adsorption_r", 0.0),
                "fitness_cost": p.get("fitness_cost", 0.05),
                "mu": p.get("mu", 1e-7),
            }
        )
    st.session_state["int_phages"] = phages_list

    # 7. Adsorption matrices (restore from list in presets if any)
    # Default: WT strain (s_idx=0) gets 1e-8; resistant strains get 0.
    for s_idx in range(len(strains_list)):
        for p_idx in range(len(phages_list)):
            p_orig = params.get("phages", [])[p_idx]
            ads = p_orig.get("adsorption_rates", None)
            ads_dorm = p_orig.get("adsorption_rates_dormant", 0.0)

            # Resolve lists or scalars; None → per-strain default
            if isinstance(ads, list):
                val_ads = ads[s_idx]
            elif ads is not None:
                val_ads = ads
            else:
                val_ads = 1e-8 if s_idx == 0 else 0.0
            val_ads_dorm = ads_dorm[s_idx] if isinstance(ads_dorm, list) else ads_dorm

            st.session_state[f"ads_{s_idx}_{p_idx}"] = float(val_ads)
            st.session_state[f"ads_dorm_{s_idx}_{p_idx}"] = float(val_ads_dorm)

    # 8. Antibiotics list
    abx_list = []
    for i, a in enumerate(params.get("antibiotics", [])):
        abx_list.append(
            {
                "name": a.get("name", f"Drug {i}"),
                "Vc": a.get("Vc", 250.0),
                "k_elim": a.get("k_elim", 0.3),
                "k12": a.get("k12", 0.0),
                "k21": a.get("k21", 0.0),
                "emax": a.get("emax", 3.0),
                "ec50": a.get("ec50", 0.2),
                "hill": a.get("hill", 1.5),
                "f_lyse": a.get("f_lyse", 0.0),
                "inoculum_effect_constant": a.get("inoculum_effect_constant", 0.0),
                "Km_elim": a.get("Km_elim", 0.0),
                # BRG specific
                "emax_r": a.get("emax_r", a.get("emax", 3.0) * 0.1),
                "ec50_r": a.get("ec50_r", a.get("ec50", 0.2) * 10.0),
                "fitness_cost": a.get("fitness_cost", 0.05),
                "mu": a.get("mu", 1e-7),
            }
        )
    st.session_state["int_antibiotics"] = abx_list

    # 9. Doses
    doses_list = []
    for d in params.get("doses", []):
        doses_list.append(
            {
                "time": d.get("time", 0.0),
                "amount": d.get("amount", 1e9),
                "target_type": d.get("target_type", "phage"),
                "target_idx": d.get("target_idx", 0),
                "route": d.get("route", "bolus"),
                "duration": d.get("duration", 0.0),
            }
        )
    st.session_state["int_doses"] = doses_list


def configure_summary(config: dict) -> str:
    """One-line human summary of an AI-supplied simulator configuration (no Streamlit)."""
    strains = config.get("strains") or []
    bits = [f"{len(strains)} strain(s)"]
    for key, label in (("phages", "phage"), ("antibiotics", "antibiotic"), ("doses", "dose event")):
        n = len(config.get(key) or [])
        if n:
            bits.append(f"{n} {label}(s)")
    extras = []
    if any(s.get("dormancy_enabled") for s in strains):
        extras.append("dormancy")
    if config.get("immunity_enabled"):
        extras.append("immunity")
    if config.get("debris_enabled"):
        extras.append("OD/debris")
    tail = f"; t_end={config.get('t_end', 48.0)} h"
    if extras:
        tail += ", " + ", ".join(extras)
    return ", ".join(bits) + tail


def apply_ai_configuration(config: dict) -> str:
    """Handler for the assistant's ``configure_simulator`` tool: populate the Interactive
    Simulator from the model's structured config via load_preset_to_state, then set the
    chosen builder mode (Direct / BRG / StrainSet). Returns a summary for the model, or an
    ``ERROR ...`` string (which the model reads as a signal to fall back to run_pbisim_code)."""
    try:
        strains = config.get("strains") or []
        if not strains:
            return "ERROR: configuration needs at least one strain."
        mode = (config.get("builder_mode") or "direct").lower()

        # Shared entities + globals (this also resets to Direct and clears mode-specific keys).
        load_preset_to_state(config)

        if mode == "brg":
            base = strains[0]
            st.session_state["int_builder_mode"] = "Binary Genotypes (BRG)"
            st.session_state["int_brg_base_growth"] = base.get("growth_rate", 1.2)
            st.session_state["int_brg_base_ratio"] = base.get("bacteria_to_resource_ratio", 1e9)
            _dorm = bool(base.get("dormancy_enabled", False))
            st.session_state["int_brg_dormancy_enabled"] = _dorm
            st.session_state["int_brg_dorm_rate"] = base.get("dormancy_rate", 0.001)
            st.session_state["int_brg_resus_rate"] = base.get("resuscitation_rate", 0.1)
            st.session_state["int_brg_diff_rate"] = base.get("dormancy_diffusion_rate", 0.05)
            st.session_state["int_brg_death_rate_B"] = base.get("death_rate_B", 0.0)
            st.session_state["int_brg_death_rate_D"] = base.get("death_rate_D", 0.0)
            st.session_state["int_brg_use_eq_ic"] = bool(config.get("equilibrium_ic", False))
            st.session_state["int_brg_eq_total_B"] = config.get("total_bacteria", 1e7)
            n_phg, n_abx = len(config.get("phages") or []), len(config.get("antibiotics") or [])
            summary = (f"Binary Genotypes (BRG): base strain '{base.get('name', 'WT')}' with "
                       f"{n_phg} phage locus/loci + {n_abx} antibiotic locus/loci "
                       f"({2 ** (n_phg + n_abx)} genotypes)")
        elif mode == "strainset":
            st.session_state["int_builder_mode"] = "Custom Strains & Graph (StrainSet)"
            edges = [
                {"from": g.get("from", ""), "to": g.get("to", ""), "rate": g.get("rate", 0.0)}
                for g in (config.get("mutation_graph") or [])
                if g.get("from") and g.get("to")
            ]
            st.session_state["int_transitions"] = edges
            summary = (f"Custom Strains (StrainSet): {len(strains)} named strain(s), "
                       f"{len(edges)} mutation edge(s)")
        else:  # direct
            st.session_state["int_builder_mode"] = "Direct (ModelBuilder)"
            summary = configure_summary(config)

        return f"Configured the Interactive Simulator — {summary}; t_end={config.get('t_end', 48.0)} h."
    except Exception as e:
        return f"ERROR applying configuration: {e}"


def summarize_current_results(_inp=None) -> str:
    """Handler for the assistant's ``get_simulation_summary`` tool: a compact metrics
    summary of the current Interactive Simulator result, for the model to interpret."""
    res = st.session_state.get("simulation_result")
    cfg = st.session_state.get("simulation_config")
    if res is None or cfg is None:
        return ("No simulation results are available yet — ask the user to run a simulation "
                "in the Interactive Simulator (or set one up and run it) first.")
    try:
        t = np.asarray(res.time, dtype=float)
        total = np.asarray(res.sum_prefixes("B", "D", "I", "H"), dtype=float)
        lines = [f"Current simulation ({st.session_state.get('int_builder_mode', '?')}), "
                 f"t = {t[0]:.1f}–{t[-1]:.1f} h."]
        lines.append(
            f"Total bacteria (CFU/mL): start {total[0]:.2e}, end {total[-1]:.2e}, "
            f"nadir {total.min():.2e}, peak {total.max():.2e} "
            f"({'net decline' if total[-1] < total[0] else 'net growth/regrowth'})."
        )
        try:
            from pbisim import time_to_clearance, time_to_log_reduction
            thr = st.session_state.get("int_extinction_threshold", 1.0) or 1.0
            tc = time_to_clearance(res, threshold=thr)
            t2 = time_to_log_reduction(res, n_logs=2.0)
            lines.append(
                f"Time to clearance (<{thr:g}): {('%.1f h' % tc) if tc is not None else 'NOT cleared'}. "
                f"Time to 2-log reduction: {('%.1f h' % t2) if t2 is not None else 'not reached'}."
            )
        except Exception:
            pass
        # per-strain final active bacteria + non-WT fraction
        n = int(getattr(cfg, "n_bacteria", 0) or 0)
        names = [s.get("name", f"Strain {i}") for i, s in enumerate(st.session_state.get("int_strains", []))]
        finals = []
        for i in range(n):
            try:
                bi = float(np.asarray(res.get(f"B{i}"))[-1])
            except Exception:
                bi = float("nan")
            finals.append((names[i] if i < len(names) else f"Strain {i}", bi))
        if finals:
            lines.append("Final active bacteria per strain: "
                         + ", ".join(f"{nm} {bi:.2e}" for nm, bi in finals) + ".")
            tot_active = sum(b for _, b in finals if b == b)
            if n >= 2 and tot_active > 0:
                res_frac = sum(b for _, b in finals[1:] if b == b) / tot_active
                lines.append(f"Non-first-strain fraction of active bacteria at end: {100 * res_frac:.1f}%.")
        try:
            p = np.asarray(res.sum_prefixes("P"), dtype=float)
            lines.append(f"Free phage (PFU/mL): end {p[-1]:.2e}, peak {p.max():.2e}.")
        except Exception:
            pass
        if st.session_state.get("int_immunity_enabled", False):
            try:
                imm = np.asarray(res.get("Imm"), dtype=float)
                lines.append(f"Immune level: end {imm[-1]:.2e}, peak {imm.max():.2e}.")
            except Exception:
                pass
        if st.session_state.get("int_debris_enabled", False):
            try:
                od = np.asarray(res.get_od(), dtype=float)
                lines.append(f"Optical density: end {od[-1]:.3g}, peak {od.max():.3g}.")
            except Exception:
                pass
        return "\n".join(lines)
    except Exception as e:
        return f"Could not summarize the results: {e}"


# ── Scenario snapshots (Tier 1: full-config save / load / export / import) ─────
# A "scenario" is everything needed to reproduce a simulation: the builder mode,
# strains / phages / antibiotics, pairwise adsorption, dosing, nutrient, immune,
# debris, solver, prerun, and the trial design. We snapshot the *input* session
# keys directly (rather than maintaining an inverse of load_preset_to_state), so
# new parameters are captured automatically and every builder mode is covered.
SCENARIO_SCHEMA_VERSION = 1

# Non-int_ session keys that are still scenario *data* (not widgets).
_SCENARIO_EXTRA_DATA_KEYS = ("direct_phg_res_rates", "trial_arms", "trial_iiv_inputs")
# Pairwise adsorption data keys written by the phage-config widgets.
_ADS_DATA_RE = re.compile(r"^ads_(dorm_)?\d+_\d+$")
# Widget-key prefixes to clear on load so widgets re-read the restored values
# (Streamlit keeps a widget's value under its key and would otherwise override
# the freshly-loaded data).
_SCENARIO_WIDGET_PREFIXES = (
    "int_", "str_", "phg_", "ss_", "brg_", "abx_", "ads_", "dose_",
    "rep_dose", "single_dose", "new_arm", "trial_phg", "trial_abx",
    "widget_builder_mode",
)


def _is_scenario_data_key(k: str) -> bool:
    return (
        k.startswith("int_")
        or k in _SCENARIO_EXTRA_DATA_KEYS
        or bool(_ADS_DATA_RE.match(k))
    )


def _json_safe(obj):
    """Coerce numpy scalars/arrays to plain Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)   # last resort so export never crashes on an odd object


def dump_state_to_scenario() -> dict:
    """Snapshot the current input configuration as a JSON-safe scenario state."""
    return {
        k: _json_safe(copy.deepcopy(st.session_state[k]))
        for k in list(st.session_state.keys())
        if _is_scenario_data_key(k)
    }


def load_scenario_to_state(state: dict) -> None:
    """Restore a scenario snapshot, clearing stale widget state first."""
    # Clear both the data keys we are about to overwrite and the widget keys that
    # would otherwise shadow them.
    for k in list(st.session_state.keys()):
        if k in _SCENARIO_EXTRA_DATA_KEYS or _ADS_DATA_RE.match(k) or any(
            k.startswith(p) for p in _SCENARIO_WIDGET_PREFIXES
        ):
            st.session_state.pop(k, None)
    for k, v in state.items():
        st.session_state[k] = copy.deepcopy(v)
    # Invalidate stale outputs
    st.session_state.simulation_result = None
    st.session_state.simulation_config = None
    st.session_state.trial_result = None


def export_scenarios_json(scenarios: dict) -> str:
    """Serialise the whole scenario library to a portable JSON string."""
    return json.dumps(
        {"schema_version": SCENARIO_SCHEMA_VERSION, "scenarios": _json_safe(scenarios)},
        indent=2,
    )


def import_scenarios_json(text: str) -> dict:
    """Parse an exported library; returns the {name: scenario} mapping.

    Tolerates both the wrapped ({"scenarios": {...}}) and bare ({name: ...})
    forms. Raises ValueError on malformed input.
    """
    data = json.loads(text)
    if isinstance(data, dict) and "scenarios" in data:
        scenarios = data["scenarios"]
    else:
        scenarios = data
    if not isinstance(scenarios, dict):
        raise ValueError("Not a scenario library (expected a JSON object).")
    for name, sc in scenarios.items():
        if not isinstance(sc, dict) or "state" not in sc:
            raise ValueError(f"Scenario '{name}' is missing its 'state'.")
    return scenarios


# ── Parts library (Tier 2: composable bacteria / phages / antibiotics) ─────────
# A "part" is one reusable entity (a bacterium, a phage, or an antibiotic) — its
# parameter dict plus provenance metadata. Parts compose into scenarios: loading a
# part appends its dict to the shared entity list every module reads
# (int_strains / int_phages / int_antibiotics). Phage kinetics (burst/latent/
# adsorption) are phage×host properties, not phage-intrinsic, so phage parts carry
# a `reference_host` tag and a soft "verify for this strain" flag on mismatch.
PARTS_SCHEMA_VERSION = 1

# category -> (shared session entity-list key, max count in the UI, human label)
PART_CATEGORIES = {
    "bacteria":    {"key": "int_strains",     "max": 10, "label": "Bacteria"},
    "phages":      {"key": "int_phages",      "max": 10, "label": "Phages"},
    "antibiotics": {"key": "int_antibiotics", "max": 6, "label": "Antibiotics"},
}
PART_SOURCES = ["educated guess", "literature", "pbisim-fit", "experimental"]


def empty_parts_library() -> dict:
    return {cat: {} for cat in PART_CATEGORIES}


def export_parts_json(library: dict) -> str:
    return json.dumps(
        {"schema_version": PARTS_SCHEMA_VERSION, "parts": _json_safe(library)},
        indent=2,
    )


def import_parts_json(text: str) -> dict:
    """Parse an exported parts library into the {category: {name: part}} mapping."""
    data = json.loads(text)
    parts = data.get("parts", data) if isinstance(data, dict) else None
    if not isinstance(parts, dict):
        raise ValueError("Not a parts library (expected a JSON object).")
    out = empty_parts_library()
    for cat, entries in parts.items():
        if cat not in PART_CATEGORIES:
            continue  # ignore unknown categories rather than fail the whole import
        if not isinstance(entries, dict):
            raise ValueError(f"Parts category '{cat}' must be an object.")
        for name, part in entries.items():
            if not isinstance(part, dict) or "params" not in part:
                raise ValueError(f"Part '{cat}/{name}' is missing its 'params'.")
            out[cat][name] = part
    return out


# Entity-widget key prefixes (NOT data keys). Cleared when a part is appended so the
# strain/phage/antibiotic widgets re-read from the (updated) int_* data lists rather
# than shadowing them with stale per-index widget values.
_ENTITY_WIDGET_PREFIXES = (
    "str_", "phg_", "ss_", "brg_", "abx_", "ads_input_", "ads_dorm_input_",
)


def clear_entity_widgets() -> None:
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) for p in _ENTITY_WIDGET_PREFIXES):
            st.session_state.pop(k, None)


# Minimal sensible starting configuration (one WT strain + one phage, Monod
# nutrients). Used to populate a fresh session and the "Reset" action. Kept in the
# app (not tied to the pbisim tutorials, which may change independently).
DEFAULT_SCENARIO = {
    "t_end": 48.0,
    "dt": 0.25,
    "extinction_threshold": 1.0,
    "solver_method": "BDF",
    "track_nutrients": True,
    "initial_S": 1.0,
    "monod_constant": 0.3,
    "recycle_fraction": 0.0,
    "s_in": 0.0,
    "s_out": 0.0,
    "immunity_enabled": False,
    "debris_enabled": False,
    "strains": [
        {
            "name": "Strain 0 (WT)",
            "initial_B": 1e7,
            "growth_rate": 1.2,
            "bacteria_to_resource_ratio": 1e9,
            "death_rate_B": 0.0,
            "dormancy_enabled": False,
        }
    ],
    "phages": [
        {
            "name": "Phage 0",
            "initial_P": 1e6,
            "adsorption_rates": 1e-8,
            "adsorption_rates_dormant": 0.0,
            "burst_sizes": 50.0,
            "latent_periods": 0.5,
            "phage_decay_rates": 0.1,
            "pk_mode": "None",
        }
    ],
    "antibiotics": [],
    "doses": [],
}


if "int_strains" not in st.session_state:
    load_preset_to_state(DEFAULT_SCENARIO)


# Dormancy / resuscitation entry-signal options — must match pbisim's
# _DORMANCY_SIGNALS / _RESUSCITATION_SIGNALS keys exactly.
SIGNAL_OPTIONS = ["constant", "nutrient", "density", "nutrient+density"]

# Growth-signal options → pbisim growth function name + whether nutrients are tracked.
GROWTH_SIGNALS = {
    "nutrient (Monod)":            ("monod_growth", True),
    "nutrient + density":          ("monod_logistic_growth", True),
    "density (logistic)":          ("logistic_growth", False),
    "constant (unlimited)":        ("constant_growth", False),
}

# Death-signal options → pbisim death function name. constant_death (default) is the
# flat rate d that the app used all along; nutrient = starvation d·(1−S/(Ks+S));
# density = crowding d·min(1, ΣB/K). (No nutrient+density death function exists.)
DEATH_SIGNALS = {
    "constant":                     "constant_death",
    "nutrient (starvation)":        "nutrient_dependent_death",
    "density (crowding)":           "density_dependent_death",
    "nutrient + density":           "nutrient_and_density_death",
}


def canonical_signal(v):
    """Normalise a stored dormancy/resuscitation signal to a pbisim-recognised key.

    Translates the app's legacy ``'nutrient_and_density'`` to the engine's
    ``'nutrient+density'`` (the mismatch that made that option raise a ValueError).
    """
    v = v or "nutrient"
    if v in ("nutrient_and_density", "nutrient+density"):
        return "nutrient+density"
    return v if v in SIGNAL_OPTIONS else "nutrient"


def compat_dormancy_signal(sig, track_nutrients):
    """Coerce a nutrient-based dormancy/resuscitation signal to a nutrient-independent
    one when nutrients are not tracked (S frozen) — otherwise the engine refuses to
    build (a nutrient signal can't read a frozen S). Returns ``(signal, coerced?)``.
    """
    if track_nutrients:
        return sig, False
    if sig == "nutrient":
        return "constant", True
    if sig == "nutrient+density":
        return "density", True
    return sig, False  # constant / density are already compatible


def growth_nutrient_kwargs():
    """Growth function + nutrient config for the selected growth signal.

    Returns a dict of ModelConfig fields (growth_function, track_nutrients, and the
    relevant monod_constant / carrying_capacity / recycle / s_in / s_out). Works both
    for the Direct builder (split into with_growth_function + with_nutrient) and for
    BRG / StrainSet ``to_config(**extra_config_kwargs)`` (forwarded to ModelConfig).
    """
    from pbisim import monod_growth, logistic_growth, constant_growth, monod_logistic_growth
    fns = {"monod_growth": monod_growth, "logistic_growth": logistic_growth,
           "constant_growth": constant_growth, "monod_logistic_growth": monod_logistic_growth}
    name = st.session_state.get("int_growth_function", "monod_growth")
    fn = fns.get(name, monod_growth)
    nutrient_based = name in ("monod_growth", "monod_logistic_growth")
    needs_K = name in ("logistic_growth", "monod_logistic_growth")
    # monod_constant + recycle_fraction are always supplied — StrainSet.to_config
    # requires them (they are simply unused by non-nutrient growth functions).
    kw = {
        "growth_function": fn,
        "track_nutrients": nutrient_based,
        "monod_constant": st.session_state.get("int_monod_constant", 0.3),
        "recycle_fraction": st.session_state.get("int_recycle_fraction", 0.0),
        # density signals (dormancy/resuscitation/death) count active B, or all cell
        # states (B+I+D+H) when this is on. Forwarded to with_nutrient (Direct) and
        # to_config (BRG/StrainSet), and captured by the repro recorder automatically.
        "density_signal_uses_total_cells": st.session_state.get("int_density_total_cells", False),
    }
    if nutrient_based:
        kw["s_in"] = st.session_state.get("int_s_in", 0.0)
        kw["s_out"] = st.session_state.get("int_s_out", 0.0)
    if needs_K:
        kw["carrying_capacity"] = st.session_state.get("int_carrying_capacity", 1e9)
    return kw


def dormancy_signal_functions(dsig, rsig):
    """Map dormancy/resuscitation signal strings to pbisim function objects.

    BRG / StrainSet ``to_config`` take function objects (``dormancy_function`` /
    ``resuscitation_function``) rather than the Direct builder's signal strings.
    (Imported from the transitions module — ``nutrient_and_density_resuscitation``
    isn't re-exported at the pbisim top level.)
    """
    from pbisim.dormancy.transitions import (
        nutrient_dependent_dormancy, constant_dormancy, density_dependent_dormancy,
        nutrient_and_density_dormancy, nutrient_dependent_resuscitation,
        constant_resuscitation, density_dependent_resuscitation,
        nutrient_and_density_resuscitation,
    )
    _D = {"constant": constant_dormancy, "nutrient": nutrient_dependent_dormancy,
          "density": density_dependent_dormancy, "nutrient+density": nutrient_and_density_dormancy}
    _R = {"constant": constant_resuscitation, "nutrient": nutrient_dependent_resuscitation,
          "density": density_dependent_resuscitation, "nutrient+density": nutrient_and_density_resuscitation}
    return _D.get(dsig, nutrient_dependent_dormancy), _R.get(rsig, nutrient_dependent_resuscitation)


def mode_dormancy_kwargs(dsig="nutrient", rsig="nutrient", ks=0.0, kdorm=0.0):
    """``to_config`` dormancy kwargs (function objects + Ks / K_dorm) for BRG / StrainSet
    from the selected signal strings, applying the same compatibility coercion as Direct
    mode (a nutrient signal needs nutrient-tracking growth; density needs a threshold).
    Also covers the dormancy-disabled case — the coerced default is nutrient-independent
    when the growth signal freezes S, so the engine's nutrient default can't crash it.
    """
    track = st.session_state.get("int_growth_function", "monod_growth") in ("monod_growth", "monod_logistic_growth")
    ds, _ = compat_dormancy_signal(canonical_signal(dsig), track)
    rs, _ = compat_dormancy_signal(canonical_signal(rsig), track)
    dfn, rfn = dormancy_signal_functions(ds, rs)
    kw = {"dormancy_function": dfn, "resuscitation_function": rfn}
    if ks > 0 and any(s in ("nutrient", "nutrient+density") for s in (ds, rs)):
        kw["dormancy_monod_constant"] = ks
    if any(s in ("density", "nutrient+density") for s in (ds, rs)):
        kw["dormancy_carrying_capacity"] = kdorm if kdorm > 0 else st.session_state.get("int_carrying_capacity", 1e9)
    return kw


def death_signal_function(name):
    """Map a death-function name to the pbisim function object."""
    from pbisim import (constant_death, nutrient_dependent_death,
                        density_dependent_death, nutrient_and_density_death)
    return {"constant_death": constant_death,
            "nutrient_dependent_death": nutrient_dependent_death,
            "density_dependent_death": density_dependent_death,
            "nutrient_and_density_death": nutrient_and_density_death}.get(name, constant_death)


def death_kwargs():
    """death_function (+ carrying_capacity when the density death function needs it) for
    the selected death signal. `constant_death` reproduces the previous behaviour (a flat
    rate d, applied regardless of nutrients)."""
    name = st.session_state.get("int_death_function", "constant_death")
    kw = {"death_function": death_signal_function(name)}
    if name in ("density_dependent_death", "nutrient_and_density_death"):
        kw["carrying_capacity"] = st.session_state.get("int_carrying_capacity", 1e9)
    return kw


class _ReproRecorder:
    """Records the exact builder calls made while constructing the config, so the
    reproduction script is a *byproduct of the real build* rather than a parallel
    re-implementation that can drift.

    Every method executes the real call **and** appends the rendered source line;
    arguments (numpy arrays, pbisim signal functions, PhageStrain / Antibiotic /
    DoseSchedule objects, …) are rendered from the same values the build passes, so
    adding a parameter to the build flows into the script automatically. The rendered
    grammar mirrors the builder the user actually chose (ModelBuilder / BRG / StrainSet).
    """

    def __init__(self):
        self.lines = []            # source lines that construct `cfg`
        self.imports = set()       # names to import `from pbisim import ...`
        self._render_of = {}       # id(obj) -> source string (inline) or variable name

    # ---- value rendering ----------------------------------------------------
    def _fval(self, x):
        x = float(x)
        if np.isinf(x):
            return "np.inf" if x > 0 else "-np.inf"
        if np.isnan(x):
            return "np.nan"
        return repr(x)

    def _pylist(self, lst):
        if isinstance(lst, list):
            return "[" + ", ".join(self._pylist(v) for v in lst) + "]"
        if isinstance(lst, bool):
            return repr(lst)
        if isinstance(lst, float):
            return self._fval(lst)
        return repr(lst)

    def render(self, v):
        if id(v) in self._render_of:
            return self._render_of[id(v)]
        if v is None or isinstance(v, (bool, int, str)):
            return repr(v)
        if isinstance(v, (float, np.floating)):
            return self._fval(v)
        if isinstance(v, np.integer):
            return repr(int(v))
        if isinstance(v, np.bool_):
            return repr(bool(v))
        if isinstance(v, np.ndarray):
            return f"np.array({self._pylist(v.tolist())})"
        if callable(v) and hasattr(v, "__name__"):
            nm = v.__name__
            if nm.isidentifier():
                self.imports.add(nm)
                return nm
            raise ValueError(f"cannot render callable {v!r} (no importable name)")
        if isinstance(v, tuple):
            body = ", ".join(self.render(x) for x in v)
            return f"({body}{',' if len(v) == 1 else ''})"
        if isinstance(v, list):
            return "[" + ", ".join(self.render(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{self.render(k)}: {self.render(val)}" for k, val in v.items()) + "}"
        if _dc.is_dataclass(v) and not isinstance(v, type):
            cls = type(v).__name__
            self.imports.add(cls)
            body = ", ".join(f"{f.name}={self.render(getattr(v, f.name))}"
                             for f in _dc.fields(v) if f.init)
            return f"{cls}({body})"
        raise ValueError(f"cannot render value of type {type(v)}: {v!r}")

    def _args(self, args, kwargs):
        return ", ".join([self.render(a) for a in args]
                         + [f"{k}={self.render(v)}" for k, v in kwargs.items()])

    # ---- construction / calls (execute AND record) --------------------------
    def init(self, var, cls, *args, **kwargs):
        """`var = Cls(...)` — a top-level object assigned to a script variable."""
        obj = cls(*args, **kwargs)
        self.imports.add(cls.__name__)
        self.lines.append(f"{var} = {cls.__name__}({self._args(args, kwargs)})")
        self._render_of[id(obj)] = var
        return obj

    def new(self, cls, *args, **kwargs):
        """A nested object rendered inline (e.g. PhageStrain inside a list literal)."""
        obj = cls(*args, **kwargs)
        self.imports.add(cls.__name__)
        self._render_of[id(obj)] = f"{cls.__name__}({self._args(args, kwargs)})"
        return obj

    def var(self, name, value):
        """`name = <value>` — bind a (possibly nested) value to a script variable."""
        self.lines.append(f"{name} = {self.render(value)}")
        self._render_of[id(value)] = name
        return value

    def call(self, var, obj, method, *args, **kwargs):
        """`var = var.method(...)` — a fluent builder call that returns the builder."""
        result = getattr(obj, method)(*args, **kwargs)
        self.lines.append(f"{var} = {var}.{method}({self._args(args, kwargs)})")
        if result is not None:
            self._render_of[id(result)] = var
        return result

    def mutate(self, var, obj, method, *args, **kwargs):
        """`var.method(...)` — an in-place call that returns None (e.g. add_strain)."""
        getattr(obj, method)(*args, **kwargs)
        self.lines.append(f"{var}.{method}({self._args(args, kwargs)})")

    def classcall(self, var, cls, method, *args, **kwargs):
        """`var = Cls.method(...)` — a classmethod constructor (e.g. from_strains)."""
        obj = getattr(cls, method)(*args, **kwargs)
        self.imports.add(cls.__name__)
        self.lines.append(f"{var} = {cls.__name__}.{method}({self._args(args, kwargs)})")
        self._render_of[id(obj)] = var
        return obj

    def result(self, var, srcvar, obj, method, *args, **kwargs):
        """`var = srcvar.method(...)` — terminal call producing the config."""
        res = getattr(obj, method)(*args, **kwargs)
        self.lines.append(f"{var} = {srcvar}.{method}({self._args(args, kwargs)})")
        self._render_of[id(res)] = var
        return res

    def expr(self, value, source):
        """Register ``value`` so it renders as the given source expression (e.g. a
        ``brg.equilibrium_initial_condition(...)`` call) instead of a literal dump."""
        self._render_of[id(value)] = source
        return value


def build_nominal_config_from_gui():
    """
    Constructs and returns the ModelConfig and corresponding state initial values
    based on the selected builder mode in the GUI.

    As it builds, it records the exact builder calls into a ``_ReproRecorder`` stashed
    at ``st.session_state['_repro_rec']`` — the reproduction script is generated from
    that recording, so it can never drift from what is actually simulated.
    """
    rec = _ReproRecorder()
    st.session_state["_repro_rec"] = rec

    builder_mode = st.session_state.get("int_builder_mode", "Direct (ModelBuilder)")
    strains = st.session_state.get("int_strains", [])
    phages = st.session_state.get("int_phages", [])
    antibiotics = st.session_state.get("int_antibiotics", [])
    doses = st.session_state.get("int_doses", [])

    n_bacteria = len(strains)
    n_phages = len(phages)
    n_latent = int(st.session_state.get("int_n_latent", 5))  # latency compartments (all builders)

    # ── Resolve solver settings ───────────────────────────────────────────────
    track_nutrients = st.session_state.get("int_track_nutrients", True)
    superinfection = st.session_state.get("int_superinfection", False)
    
    # ── Resolve Debris parameters ─────────────────────────────────────────────
    debris_enabled = st.session_state.get("int_debris_enabled", False)
    extra_kwargs = {}
    if debris_enabled:
        extra_kwargs["debris_u"] = st.session_state.get("int_debris_u", 0.4)
        extra_kwargs["debris_v"] = st.session_state.get("int_debris_v", 0.2)
        extra_kwargs["debris_kdis"] = st.session_state.get("int_debris_kdis", 0.01)
        extra_kwargs["od_to_cfu_conversion_factor"] = st.session_state.get("int_od_to_cfu_conversion_factor", 2e8)
        
    # ── Resolve Dose Schedule ─────────────────────────────────────────────────
    dose_events = []
    for d in doses:
        target_type = d["target_type"]
        if target_type == "phage":
            target = "phage"
            target_idx = d["target_idx"]
        elif target_type == "antibiotic":
            target = "antibiotic"
            target_idx = d["target_idx"]
        else:
            target = "nutrient"
            target_idx = 0
            
        event = rec.new(
            DoseEvent,
            time=d["time"],
            amount=d["amount"],
            target=target,
            index=target_idx,
            route=d["route"],
            duration=d.get("duration", 0.0),
        )
        dose_events.append(event)

    if dose_events:
        rec.var("dose_events", dose_events)
        schedule = rec.var("schedule", rec.new(DoseSchedule, dose_events))
    else:
        schedule = None

    # ── BUILDER MODE: Direct (ModelBuilder) ───────────────────────────────────
    if builder_mode == "Direct (ModelBuilder)":
        max_depth = max([s.get("dormancy_depth", 1) for s in strains] if strains else [1])
        builder = rec.init("builder", ModelBuilder, n_bacteria=n_bacteria, n_phages=n_phages, n_latent=n_latent, n_depth=max_depth)

        # Growth rates
        growth_rates = [s["growth_rate"] for s in strains]
        ratios = [s.get("bacteria_to_resource_ratio", 1e9) for s in strains]
        builder = rec.call("builder", builder, "with_growth_rates", growth_rates, bacteria_to_resource_ratio=ratios)

        # Natural death rates
        death_rates_B = [s.get("death_rate_B", 0.0) for s in strains]
        death_rates_D = [s.get("death_rate_D", 0.0) for s in strains]
        _has_active_death = any(db > 0 for db in death_rates_B)
        _dthk = death_kwargs()
        if _has_active_death and "carrying_capacity" in _dthk:  # density death needs K
            builder = rec.call("builder", builder, "with_nutrient", carrying_capacity=_dthk["carrying_capacity"])
        # Always set the death function so the config reflects the chosen signal; the
        # rates are only overridden when > 0 (a None rate means that pathway is off).
        builder = rec.call(
            "builder", builder, "with_death",
            death_rate_B=np.array(death_rates_B) if _has_active_death else None,
            death_rate_D=np.array(death_rates_D) if any(dd > 0 for dd in death_rates_D) else None,
            death_function=_dthk["death_function"],
        )
        
        # Dormancy
        any_dormancy = any(s.get("dormancy_enabled", False) for s in strains)
        if any_dormancy:
            dormancy_rates = [s["dormancy_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
            resus_rates = [s["resuscitation_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
            diff_rates = [s["dormancy_diffusion_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
            enabled_strains = [s for s in strains if s.get("dormancy_enabled", False)]
            ds = canonical_signal(enabled_strains[0]["dormancy_signal"]) if enabled_strains else "nutrient"
            rs = canonical_signal(enabled_strains[0]["resuscitation_signal"]) if enabled_strains else "nutrient"
            _dorm_ks = float(enabled_strains[0].get("dormancy_monod_constant", 0.0)) if enabled_strains else 0.0
            _dorm_kdorm = float(enabled_strains[0].get("dormancy_carrying_capacity", 0.0)) if enabled_strains else 0.0
            # nutrient dormancy signals need S tracked; coerce when it isn't.
            ds, _cd = compat_dormancy_signal(ds, track_nutrients)
            rs, _cr = compat_dormancy_signal(rs, track_nutrients)

            _dorm_kwargs = dict(
                dormancy_rate=np.array(dormancy_rates),
                resuscitation_rate=np.array(resus_rates),
                dormancy_diffusion_rate=np.array(diff_rates),
                dormancy_signal=ds,
                resuscitation_signal=rs,
            )
            if _dorm_ks > 0 and any(sig in ("nutrient", "nutrient+density") for sig in (ds, rs)):
                _dorm_kwargs["dormancy_monod_constant"] = _dorm_ks
            # density-based dormancy/resuscitation needs a density threshold. Use the
            # per-strain dormancy_carrying_capacity when set, else the growth carrying
            # capacity (Monod growth doesn't set one, so supply it explicitly).
            if any(sig in ("density", "nutrient+density") for sig in (ds, rs)):
                _dorm_kwargs["dormancy_carrying_capacity"] = (
                    _dorm_kdorm if _dorm_kdorm > 0 else st.session_state.get("int_carrying_capacity", 1e9))
            builder = rec.call("builder", builder, "with_dormancy", **_dorm_kwargs)
        elif not track_nutrients:
            # No dormancy configured, but a frozen S is incompatible with the default
            # nutrient dormancy function — pin nutrient-independent (constant) functions.
            builder = rec.call(
                "builder", builder, "with_dormancy",
                dormancy_rate=0.0, resuscitation_rate=0.0, dormancy_diffusion_rate=0.0,
                dormancy_signal="constant", resuscitation_signal="constant")
            
        # Phages
        if n_phages > 0:
            adsorption_rates = []
            adsorption_rates_dormant = []
            for s_idx in range(n_bacteria):
                s_ads = []
                s_ads_dorm = []
                for p_idx in range(n_phages):
                    s_ads.append(st.session_state.get(f"ads_{s_idx}_{p_idx}", 1e-8 if s_idx == 0 else 0.0))
                    s_ads_dorm.append(st.session_state.get(f"ads_dorm_{s_idx}_{p_idx}", 0.0))
                adsorption_rates.append(s_ads)
                adsorption_rates_dormant.append(s_ads_dorm)
                
            burst_sizes = np.tile(np.array([p["burst_sizes"] for p in phages]), (n_bacteria, 1))
            latent_periods = np.tile(np.array([p["latent_periods"] for p in phages]), (n_bacteria, 1))
            decay_rates = [p["phage_decay_rates"] for p in phages]
            
            # Phage PK
            has_phage_pk = any(p["pk_mode"] != "None" for p in phages)
            if has_phage_pk:
                from pbisim import PhagePKConfig
                vcs = np.array([p.get("Vc", 5000.0) for p in phages])
                k_elims = np.array([p.get("k_elim", 0.2) if p["pk_mode"] != "None" else 0.0 for p in phages])
                k_ins = np.array([p.get("k_in", 0.1) if p["pk_mode"] != "None" else 0.0 for p in phages])
                k_outs = np.array([p.get("k_out", 0.05) if p["pk_mode"] != "None" else 0.0 for p in phages])
                
                # Nonlinear PK Km
                kms = np.array([p.get("Km_elim", 0.0) if p.get("Km_elim", 0.0) > 0 else np.inf for p in phages])
                
                has_mc = any(p["pk_mode"] == "Mass-Conserving" for p in phages)
                vis = np.array([p.get("Vi", 10.0) if p["pk_mode"] == "Mass-Conserving" else 0.0 for p in phages]) if has_mc else None
                
                pk_config = rec.new(
                    PhagePKConfig,
                    n_phages=n_phages, Vc=vcs, k_elim=k_elims, k_in=k_ins, k_out=k_outs, Vi=vis, Km_elim=kms
                )
                builder = rec.call("builder", builder, "with_phage_pk", pk_config)
                
            # Phage decay nonlinear Km
            phage_decay_Km = np.array([p.get("phage_decay_Km", 0.0) if p.get("phage_decay_Km", 0.0) > 0 else np.inf for p in phages])

            # Dormant-adsorption attenuation with dormancy depth (per phage, broadcast
            # across strains): effective dormant rate = adsorption_dormant * exp(-att * depth).
            attenuation_rate = np.tile(
                np.array([p.get("attenuation_rate", 0.0) for p in phages]), (n_bacteria, 1)
            )

            builder = rec.call(
                "builder", builder, "with_phage_params",
                adsorption_rates=np.array(adsorption_rates),
                adsorption_rates_dormant=np.array(adsorption_rates_dormant),
                burst_sizes=np.array(burst_sizes),
                latent_periods=np.array(latent_periods),
                phage_decay_rates=np.array(decay_rates),
                allow_superinfection=superinfection,
                phage_decay_Km=phage_decay_Km,
                attenuation_rate=attenuation_rate,
            )
            
            # Pseudolysogeny
            any_pseudo = any(p.get("hibernation_rate_s", 0.0) > 0 or p.get("hibernation_rate_r", 0.0) > 0 for p in phages)
            if any_pseudo:
                # build n_bacteria x n_phages matrices
                hib_rates = np.zeros((n_bacteria, n_phages))
                res_rates = np.zeros((n_bacteria, n_phages))
                # For Direct mode, broadcast: WT takes susceptible rates, other strains take resistant rates
                for p_idx in range(n_phages):
                    p = phages[p_idx]
                    hib_rates[0, p_idx] = p.get("hibernation_rate_s", 0.0)
                    res_rates[0, p_idx] = p.get("lytic_resumption_rate_s", 0.0)
                    if n_bacteria > 1:
                        hib_rates[1:, p_idx] = p.get("hibernation_rate_r", 0.0)
                        res_rates[1:, p_idx] = p.get("lytic_resumption_rate_r", 0.0)
                builder = rec.call("builder", builder, "with_pseudolysogeny", hibernation_rate=hib_rates, lytic_resumption_rate=res_rates)
                
        # Mutations. A custom mutation-network graph (any n_bacteria) takes
        # precedence; otherwise fall back to the per-phage-locus shortcut, which
        # pbisim only supports when n_bacteria == 2**n_phages.
        _mut_M = mutation_matrix_from_transitions(st.session_state.get("int_transitions", []), strains)
        if _mut_M is not None:
            builder = rec.call("builder", builder, "with_mutations", mutation_rates=_mut_M)
        elif n_phages > 0 and n_bacteria == 2**n_phages:
            phg_res_rates = st.session_state.get("direct_phg_res_rates", [1e-7] * n_phages)
            builder = rec.call("builder", builder, "with_mutations", phage_resistance_rates=phg_res_rates)
            
        # Antibiotics
        for abx in antibiotics:
            builder = rec.call(
                "builder", builder, "with_antibiotic",
                name=abx["name"],
                k_elim=abx["k_elim"],
                Vc=abx.get("Vc", 1.0),
                k12=abx.get("k12", 0.0),
                k21=abx.get("k21", 0.0),
                emax=abx["emax"],
                ec50=abx["ec50"],
                hill=abx.get("hill", 1.0),
                f_lyse=abx.get("f_lyse", 0.0),
                inoculum_effect_constant=abx.get("inoculum_effect_constant", None) if abx.get("inoculum_effect_constant", 0.0) > 0 else None,
                Km_elim=abx.get("Km_elim", None) if abx.get("Km_elim", 0.0) > 0 else None,
            )

        # Nutrients / growth signal
        _gk = growth_nutrient_kwargs()
        builder = rec.call("builder", builder, "with_growth_function", _gk.pop("growth_function"))
        builder = rec.call("builder", builder, "with_nutrient", **_gk)
            
        # Immunity
        immunity_enabled = st.session_state.get("int_immunity_enabled", False)
        if immunity_enabled:
            kill_rate_D = st.session_state.get("int_imm_kill_rate_D", 0.0)
            builder = rec.call(
                "builder", builder, "with_immunity",
                imm_stim_rate=np.full(n_bacteria, st.session_state.get("int_imm_stim_rate", 0.1)),
                imm_stim50=st.session_state.get("int_imm_stim50", 1e6),
                imm_kill_rate=np.full(n_bacteria, st.session_state.get("int_innate_kill_rate", 1e7)),
                imm_kill50=st.session_state.get("int_innate_kill50", 1e5),
                imm_decay_rate=st.session_state.get("int_innate_decay_rate", 0.1),
                immune_module=st.session_state.get("int_immune_module", "innate"),
                imm_max=st.session_state.get("int_innate_max", 1e7),
                imm_kill_rate_D=np.array([kill_rate_D] * n_bacteria) if kill_rate_D > 0 else None
            )

        if schedule:
            builder = rec.call("builder", builder, "with_dose_schedule", schedule)

        # OD / debris ODE (Direct mode). ModelBuilder.build() takes no kwargs — debris
        # must be configured via with_od_debris(), not passed to build().
        if debris_enabled:
            builder = rec.call(
                "builder", builder, "with_od_debris",
                u=extra_kwargs.get("debris_u", 0.4),
                v=extra_kwargs.get("debris_v", 0.2),
                kdis=extra_kwargs.get("debris_kdis", 0.01),
                od_to_cfu_conversion_factor=extra_kwargs.get("od_to_cfu_conversion_factor", 2e8),
            )

        config = rec.result("cfg", "builder", builder, "build")

        initial_B = np.array([s["initial_B"] for s in strains])
        initial_P = np.array([p["initial_P"] for p in phages])
        initial_S = st.session_state.get("int_initial_S", 1.0) if track_nutrients else 1.0

        model_kwargs = {}
        if immunity_enabled:
            model_kwargs["initial_Imm"] = st.session_state.get("int_imm_initial", 0.0)
            
        # Dormant initial conditions — use per-strain initial_D (default 0).
        # PBIModel accepts shape (n_bacteria,) and distributes evenly across Q layers.
        if any_dormancy:
            ic_D = np.array([s.get("initial_D", 0.0) for s in strains])
            if np.any(ic_D > 0):
                model_kwargs["initial_D"] = ic_D
            
        return config, initial_B, initial_P, initial_S, model_kwargs

    # ── BUILDER MODE: Binary Resistance Genotypes (BRG) ──────────────────────
    elif builder_mode == "Binary Genotypes (BRG)":
        # Build BacterialStrain
        base_growth = st.session_state.get("int_brg_base_growth", 1.2)
        base_ratio = st.session_state.get("int_brg_base_ratio", 1e9)
        dormancy_enabled = st.session_state.get("int_brg_dormancy_enabled", False)
        
        dorm_rate = st.session_state.get("int_brg_dorm_rate", 0.001) if dormancy_enabled else 0.0
        resus_rate = st.session_state.get("int_brg_resus_rate", 0.1) if dormancy_enabled else 0.0
        diff_rate = st.session_state.get("int_brg_diff_rate", 0.05) if dormancy_enabled else 0.0
        
        b = rec.init(
            "bacteria", BacterialStrain,
            base_growth_rate=base_growth,
            bacteria_to_resource_ratio=base_ratio,
            dormancy_rate=dorm_rate,
            resuscitation_rate=resus_rate,
            dormancy_diffusion_rate=diff_rate,
            death_rate_B=st.session_state.get("int_brg_death_rate_B", 0.0) if st.session_state.get("int_brg_death_rate_B", 0.0) > 0 else None,
            death_rate_D=st.session_state.get("int_brg_death_rate_D", 0.0) if dormancy_enabled and st.session_state.get("int_brg_death_rate_D", 0.0) > 0 else None,
        )

        # Build PhageStrains
        phage_strains = []
        for p in phages:
            phage_strains.append(
                rec.new(
                    PhageStrain,
                    name=p["name"],
                    adsorption_s=p.get("adsorption_s", 5e-8),
                    adsorption_r=p.get("adsorption_r", 0.0),
                    burst_size_s=p["burst_sizes"],
                    latent_period_s=p["latent_periods"],
                    decay_rate=p["phage_decay_rates"],
                    fitness_cost=p.get("fitness_cost", 0.0),
                    mu=p.get("mu", 1e-7),
                    hibernation_rate_s=p.get("hibernation_rate_s", None) if p.get("hibernation_rate_s", 0.0) > 0 else None,
                    hibernation_rate_r=p.get("hibernation_rate_r", None) if p.get("hibernation_rate_r", 0.0) > 0 else None,
                    lytic_resumption_rate_s=p.get("lytic_resumption_rate_s", None) if p.get("lytic_resumption_rate_s", 0.0) > 0 else None,
                    lytic_resumption_rate_r=p.get("lytic_resumption_rate_r", None) if p.get("lytic_resumption_rate_r", 0.0) > 0 else None,
                )
            )
        rec.var("phages", phage_strains)

        # Build Antibiotics
        abx_strains = []
        for abx in antibiotics:
            abx_strains.append(
                rec.new(
                    Antibiotic,
                    name=abx["name"],
                    emax_s=abx["emax"],
                    emax_r=abx.get("emax_r", abx["emax"] * 0.1),
                    ec50_s=abx["ec50"],
                    ec50_r=abx.get("ec50_r", abx["ec50"] * 10.0),
                    k_elim=abx["k_elim"],
                    hill=abx.get("hill", 1.0),
                    fitness_cost=abx.get("fitness_cost", 0.0),
                    mu=abx.get("mu", 1e-7),
                    Vc=abx.get("Vc", 1.0),
                    k12=abx.get("k12", 0.0),
                    k21=abx.get("k21", 0.0),
                    inoculum_effect_constant=abx.get("inoculum_effect_constant", None) if abx.get("inoculum_effect_constant", 0.0) > 0 else None,
                )
            )
        if abx_strains:
            rec.var("antibiotics", abx_strains)

        brg = rec.classcall(
            "brg", BinaryResistanceGenotypes, "from_strains",
            phage_strains,
            bacteria=b,
            antibiotics=abx_strains if abx_strains else None,
        )
        
        # Build config — dormancy depth compartments (configurable; 1 when dormancy off)
        max_depth = int(st.session_state.get("int_brg_n_depth", 1)) if dormancy_enabled else 1

        # Resolve Phage PK config
        phage_pk_config = None
        has_phage_pk = any(p["pk_mode"] != "None" for p in phages)
        if has_phage_pk:
            from pbisim import PhagePKConfig
            vcs = np.array([p.get("Vc", 5000.0) for p in phages])
            k_elims = np.array([p.get("k_elim", 0.2) if p["pk_mode"] != "None" else 0.0 for p in phages])
            k_ins = np.array([p.get("k_in", 0.1) if p["pk_mode"] != "None" else 0.0 for p in phages])
            k_outs = np.array([p.get("k_out", 0.05) if p["pk_mode"] != "None" else 0.0 for p in phages])
            kms = np.array([p.get("Km_elim", 0.0) if p.get("Km_elim", 0.0) > 0 else np.inf for p in phages])
            has_mc = any(p["pk_mode"] == "Mass-Conserving" for p in phages)
            vis = np.array([p.get("Vi", 10.0) if p["pk_mode"] == "Mass-Conserving" else 0.0 for p in phages]) if has_mc else None
            
            phage_pk_config = rec.new(
                PhagePKConfig,
                n_phages=n_phages, Vc=vcs, k_elim=k_elims, k_in=k_ins, k_out=k_outs, Vi=vis, Km_elim=kms
            )

        # Expose nonlinear clearances
        extra_kwargs["allow_superinfection"] = superinfection

        # Growth signal + nutrient config (forwarded to ModelConfig via to_config).
        extra_kwargs.update(growth_nutrient_kwargs())
        # Dormancy signal functions (+ Ks / K_dorm) from the BRG selectors.
        extra_kwargs.update(mode_dormancy_kwargs(
            st.session_state.get("int_brg_dorm_signal", "nutrient"),
            st.session_state.get("int_brg_resus_signal", "nutrient"),
            float(st.session_state.get("int_brg_dorm_ks", 0.0)),
            float(st.session_state.get("int_brg_dorm_kdorm", 0.0)),
        ))
        extra_kwargs.update(death_kwargs())  # death signal function (+ K for density death)

        # Dose schedule
        if schedule:
            extra_kwargs["dose_schedule"] = schedule
            
        # Immunity
        immunity_enabled = st.session_state.get("int_immunity_enabled", False)
        if immunity_enabled:
            extra_kwargs["imm_stim_rate"] = np.full(brg.n_strains, st.session_state.get("int_imm_stim_rate", 0.1))
            extra_kwargs["imm_stim50"] = st.session_state.get("int_imm_stim50", 1e6)
            extra_kwargs["imm_kill_rate"] = np.full(brg.n_strains, st.session_state.get("int_innate_kill_rate", 1e7))
            extra_kwargs["imm_kill50"] = st.session_state.get("int_innate_kill50", 1e5)
            extra_kwargs["imm_decay_rate"] = st.session_state.get("int_innate_decay_rate", 0.1)
            extra_kwargs["immune_module"] = st.session_state.get("int_immune_module", "innate")
            extra_kwargs["imm_max"] = st.session_state.get("int_innate_max", 1e7)
            kill_rate_D = st.session_state.get("int_imm_kill_rate_D", 0.0)
            if kill_rate_D > 0:
                extra_kwargs["imm_kill_rate_D"] = np.array([kill_rate_D] * brg.n_strains)

        # Per-phage dormant-adsorption attenuation (broadcast across genotypes).
        if phages:
            extra_kwargs["attenuation_rate"] = np.array(
                [p.get("attenuation_rate", 0.0) for p in phages]
            )

        # Per-phage nonlinear (Michaelis-Menten) phage decay saturation.
        if phages and any(p.get("phage_decay_Km", 0.0) > 0 for p in phages):
            extra_kwargs["phage_decay_Km"] = np.array(
                [p.get("phage_decay_Km", 0.0) if p.get("phage_decay_Km", 0.0) > 0 else np.inf for p in phages]
            )

        config = rec.result(
            "cfg", "brg", brg, "to_config",
            n_latent=n_latent,
            n_depth=max_depth,
            phage_pk_config=phage_pk_config,
            **extra_kwargs
        )

        # Resolve initial densities
        if st.session_state.get("int_brg_use_eq_ic", False):
            total_B = st.session_state.get("int_brg_eq_total_B", 1e7)
            # Record as the derivation call so the repro shows the pbisim function, not
            # just the resulting numbers.
            initial_B = rec.expr(
                brg.equilibrium_initial_condition(total_bacteria=total_B),
                f"brg.equilibrium_initial_condition(total_bacteria={rec.render(total_B)})",
            )
        else:
            initial_B = np.zeros(brg.n_strains)
            saved_init_B = st.session_state.get("int_brg_initial_B", {})
            for idx, lbl in enumerate(brg.strain_labels):
                initial_B[idx] = saved_init_B.get(lbl, 1e7 if idx == 0 else 0.0)
            
        initial_P = np.array([p["initial_P"] for p in phages])
        initial_S = st.session_state.get("int_initial_S", 1.0) if track_nutrients else 1.0
        
        model_kwargs = {}
        if immunity_enabled:
            model_kwargs["initial_Imm"] = st.session_state.get("int_imm_initial", 0.0)

        return config, initial_B, initial_P, initial_S, model_kwargs

    # ── BUILDER MODE: Custom Strains & Graph (StrainSet) ──────────────────────
    else:
        ss = rec.init("ss", StrainSet, n_phages=n_phages)

        # Register antibiotics
        for abx in antibiotics:
            rec.mutate(
                "ss", ss, "add_antibiotic",
                rec.new(
                    AntibioticDefinition,
                    name=abx["name"],
                    k_elim=abx["k_elim"],
                    Vc=abx.get("Vc", 1.0),
                    k12=abx.get("k12", 0.0),
                    k21=abx.get("k21", 0.0),
                    inoculum_effect_constant=abx.get("inoculum_effect_constant", None) if abx.get("inoculum_effect_constant", 0.0) > 0 else None,
                    Km_elim=abx.get("Km_elim", None) if abx.get("Km_elim", 0.0) > 0 else None,
                )
            )
            
        # Add Strains
        for i, s in enumerate(strains):
            # Parse phage parameters
            ads_rates = []
            ads_rates_dorm = []
            bursts = []
            latents = []
            latents_dorm = []
            hibernations = []
            resumptions = []
            for p_idx in range(n_phages):
                ads_rates.append(st.session_state.get(f"ads_{i}_{p_idx}", 1e-8 if i == 0 else 0.0))
                ads_rates_dorm.append(st.session_state.get(f"ads_dorm_{i}_{p_idx}", 0.0))
                p = phages[p_idx]
                bursts.append(p["burst_sizes"])
                latents.append(p["latent_periods"])
                latents_dorm.append(p["latent_periods"]) # nominal copy
                hibernations.append(p.get("hibernation_rate_s", 0.0) if i == 0 else p.get("hibernation_rate_r", 0.0))
                resumptions.append(p.get("lytic_resumption_rate_s", 0.0) if i == 0 else p.get("lytic_resumption_rate_r", 0.0))
                
            # Parse antibiotic sensitivities
            sensitivities = {}
            for abx in antibiotics:
                # default: first strain is susceptible, others are resistant
                emax_val = abx["emax"] if i == 0 else abx["emax"] * 0.1
                ec50_val = abx["ec50"] if i == 0 else abx["ec50"] * 10.0
                sensitivities[abx["name"]] = rec.new(AntibioticSensitivity, emax=emax_val, ec50=ec50_val)

            rec.mutate(
                "ss", ss, "add_strain",
                rec.new(
                    StrainDefinition,
                    name=s["name"],
                    growth_rate=s["growth_rate"],
                    adsorption_rates=np.array(ads_rates),
                    adsorption_rates_dormant=np.array(ads_rates_dorm),
                    burst_sizes=np.array(bursts),
                    latent_periods=np.array(latents),
                    latent_periods_dormant=np.array(latents_dorm),
                    bacteria_to_resource_ratio=s.get("bacteria_to_resource_ratio", 1e9),
                    dormancy_rate=s["dormancy_rate"] if s.get("dormancy_enabled", False) else 0.0,
                    resuscitation_rate=s["resuscitation_rate"] if s.get("dormancy_enabled", False) else 0.0,
                    dormancy_diffusion_rate=s["dormancy_diffusion_rate"] if s.get("dormancy_enabled", False) else 0.0,
                    imm_stim_rate=st.session_state.get("int_imm_stim_rate", 0.1) if st.session_state.get("int_immunity_enabled", False) else 0.0,
                    imm_kill_rate=st.session_state.get("int_innate_kill_rate", 1e7) if st.session_state.get("int_immunity_enabled", False) else 0.0,
                    attenuation_rate=np.array([p.get("attenuation_rate", 0.0) for p in phages]),
                    death_rate_B=s.get("death_rate_B", None) if s.get("death_rate_B", 0.0) > 0 else None,
                    death_rate_D=s.get("death_rate_D", None) if s.get("death_rate_D", 0.0) > 0 else None,
                    hibernation_rate=np.array(hibernations) if any(h > 0 for h in hibernations) else None,
                    lytic_resumption_rate=np.array(resumptions) if any(r > 0 for r in resumptions) else None,
                    antibiotic_sensitivity=sensitivities
                )
            )

        # Build mutation graph
        transitions = st.session_state.get("int_transitions", [])
        graph_dict = {}
        for trans in transitions:
            src = trans["from"]
            dest = trans["to"]
            rate = trans["rate"]
            if src and dest:
                if src not in graph_dict:
                    graph_dict[src] = {}
                graph_dict[src][dest] = rate
        if graph_dict:
            rec.mutate("ss", ss, "set_mutation_graph", graph_dict)
            
        # Phage PK
        phage_pk_config = None
        has_phage_pk = any(p["pk_mode"] != "None" for p in phages)
        if has_phage_pk:
            from pbisim import PhagePKConfig
            vcs = np.array([p.get("Vc", 5000.0) for p in phages])
            k_elims = np.array([p.get("k_elim", 0.2) if p["pk_mode"] != "None" else 0.0 for p in phages])
            k_ins = np.array([p.get("k_in", 0.1) if p["pk_mode"] != "None" else 0.0 for p in phages])
            k_outs = np.array([p.get("k_out", 0.05) if p["pk_mode"] != "None" else 0.0 for p in phages])
            kms = np.array([p.get("Km_elim", 0.0) if p.get("Km_elim", 0.0) > 0 else np.inf for p in phages])
            has_mc = any(p["pk_mode"] == "Mass-Conserving" for p in phages)
            vis = np.array([p.get("Vi", 10.0) if p["pk_mode"] == "Mass-Conserving" else 0.0 for p in phages]) if has_mc else None
            
            phage_pk_config = rec.new(
                PhagePKConfig,
                n_phages=n_phages, Vc=vcs, k_elim=k_elims, k_in=k_ins, k_out=k_outs, Vi=vis, Km_elim=kms
            )

        decay_rates = np.array([p["phage_decay_rates"] for p in phages])
        max_depth = max([s.get("dormancy_depth", 1) for s in strains] if strains else [1])
        
        # Expose extra kwargs
        extra_kwargs["allow_superinfection"] = superinfection

        # Growth signal + nutrient config (forwarded to ModelConfig via to_config).
        extra_kwargs.update(growth_nutrient_kwargs())
        # Dormancy signal functions (+ Ks / K_dorm) from the first dormancy-enabled
        # strain's selectors (the engine dormancy function is model-wide).
        _ss_dorm = [s for s in strains if s.get("dormancy_enabled", False)]
        extra_kwargs.update(mode_dormancy_kwargs(
            _ss_dorm[0].get("dormancy_signal", "nutrient") if _ss_dorm else "nutrient",
            _ss_dorm[0].get("resuscitation_signal", "nutrient") if _ss_dorm else "nutrient",
            float(_ss_dorm[0].get("dormancy_monod_constant", 0.0)) if _ss_dorm else 0.0,
            float(_ss_dorm[0].get("dormancy_carrying_capacity", 0.0)) if _ss_dorm else 0.0,
        ))
        extra_kwargs.update(death_kwargs())  # death signal function (+ K for density death)

        # Dose schedule
        if schedule:
            extra_kwargs["dose_schedule"] = schedule

        # Per-phage nonlinear (Michaelis-Menten) phage decay saturation.
        if phages and any(p.get("phage_decay_Km", 0.0) > 0 for p in phages):
            extra_kwargs["phage_decay_Km"] = np.array(
                [p.get("phage_decay_Km", 0.0) if p.get("phage_decay_Km", 0.0) > 0 else np.inf for p in phages]
            )

        # Immunity defaults
        immunity_enabled = st.session_state.get("int_immunity_enabled", False)

        config = rec.result(
            "cfg", "ss", ss, "to_config",
            n_latent=n_latent,
            n_depth=max_depth,
            phage_decay_rates=decay_rates,
            imm_decay_rate=st.session_state.get("int_innate_decay_rate", 0.1),
            imm_stim50=st.session_state.get("int_imm_stim50", 1e6),
            imm_kill50=st.session_state.get("int_innate_kill50", 1e5),
            phage_pk_config=phage_pk_config,
            immune_module=st.session_state.get("int_immune_module", "innate"),
            imm_max=st.session_state.get("int_innate_max", 1e7),
            **extra_kwargs  # includes growth_function + monod_constant/recycle/etc.
        )

        initial_B = np.array([s["initial_B"] for s in strains])
        initial_P = np.array([p["initial_P"] for p in phages])
        initial_S = st.session_state.get("int_initial_S", 1.0) if track_nutrients else 1.0

        model_kwargs = {}
        if immunity_enabled:
            model_kwargs["initial_Imm"] = st.session_state.get("int_imm_initial", 0.0)
            
        return config, initial_B, initial_P, initial_S, model_kwargs


def run_sim_from_gui_params():
    """Builds and solves the ODE for the nominal patient."""
    config, initial_B, initial_P, initial_S, model_kwargs = build_nominal_config_from_gui()
    
    # ── Pretreatment equilibrate prerun ───────────────────────────────────────
    t_prerun = st.session_state.get("int_t_prerun", 0.0)
    if t_prerun > 0:
        # Equilibrate to stationary phase (no treatment) and carry the FULL final
        # state into the treatment sim: active B, the dormant reservoir D, nutrient
        # S, and immune priming Imm. Previously only B and S were kept — discarding
        # ic.D silently dropped the (usually dominant) dormant population, so longer
        # preruns lost more of the culture until the residual active cells fell below
        # the extinction floor and the treatment plotted as a flat 0 CFU curve.
        # (S0= was also passed here but stationary_phase_ic has no such argument — it
        # was silently forwarded to the solver and ignored.)
        ic = stationary_phase_ic(config, t_prerun=t_prerun, B0=initial_B)
        initial_B = ic.B
        initial_S = max(float(ic.S), 0.0)  # prerun can leave S slightly negative (numerical)
        if ic.D is not None:
            model_kwargs["initial_D"] = ic.D
        if ic.Imm is not None:
            model_kwargs["initial_Imm"] = ic.Imm
        _carry_prerun_debris(ic, model_kwargs)

    model = PBIModel(
        config,
        initial_B=initial_B,
        initial_P=initial_P,
        initial_S=initial_S,
        **model_kwargs
    )
    
    t_end = st.session_state.get("int_t_end", 48.0)
    dt = st.session_state.get("int_dt", 0.25)
    method = st.session_state.get("int_solver_method", "BDF")
    extinction_threshold = st.session_state.get("int_extinction_threshold", 1.0) or None
    extinction_check_interval = st.session_state.get("int_extinction_check_interval", 0.0) or None
    result = solve_ode(model, t_end=t_end, dt=dt, method=method,
                       extinction_threshold=extinction_threshold,
                       extinction_check_interval=extinction_check_interval)
    return result, config


def prerun_collapse_fraction(config, B0, t_prerun):
    """Fraction of the inoculum surviving a stationary-phase pre-run (B + D total).

    Returns ``None`` when there is no pre-run or the pre-run can't be evaluated.
    A small value flags the trap where a natural death rate with no dormancy
    decimates the culture during the pre-run (death keeps acting once nutrients
    exhaust and growth stops), so the treatment — and its CFU/OD curve — starts
    far below the inoculum.
    """
    if not t_prerun or t_prerun <= 0:
        return None
    try:
        ic = stationary_phase_ic(config, t_prerun=t_prerun, B0=B0)
    except Exception:
        return None
    total = float(np.sum(ic.B)) + (float(np.sum(ic.D)) if ic.D is not None else 0.0)
    b0 = float(np.sum(B0))
    return (total / b0) if b0 > 0 else None


def warn_if_prerun_collapses(config, B0):
    """Emit a Streamlit warning if the configured pre-run decimates the culture."""
    t_prerun = st.session_state.get("int_t_prerun", 0.0)
    frac = prerun_collapse_fraction(config, B0, t_prerun)
    if frac is not None and frac < 0.1:
        st.warning(
            f"⚠️ The {t_prerun:g} h pre-run leaves only ~{frac*100:.2g}% of the inoculum "
            "(active + dormant) at treatment start, so the CFU/OD curves scale very low. "
            "A natural death rate with no dormancy makes the culture decline during the "
            "pre-run once nutrients exhaust (death keeps acting after growth stops). "
            "Reduce the pre-run duration or the death rate, or enable dormancy so persisters "
            "survive the pre-run."
        )


def reseed_widget_config(store_key, prefixes):
    """Re-seed widget selections from a persistent store BEFORE the widgets render.

    Streamlit drops a widget's key from session_state whenever the widget is not
    rendered on a rerun, so leaving a page (e.g. a sweep page) and coming back would
    otherwise reset all its controls. Shadowing the keys into a plain dict (which
    survives) and re-seeding from it keeps the configuration alive across navigation.
    """
    store = st.session_state.setdefault(store_key, {})
    for k, v in list(store.items()):
        if any(k.startswith(p) for p in prefixes) and k not in st.session_state:
            try:
                st.session_state[k] = v
            except Exception:
                pass  # non-settable widget (e.g. a button) — skip


def save_widget_config(store_key, prefixes, exclude=()):
    """Shadow the current widget selections (keys matching *prefixes*) into the store."""
    store = st.session_state.setdefault(store_key, {})
    for k in list(st.session_state.keys()):
        if k in exclude:
            continue
        if any(k.startswith(p) for p in prefixes):
            store[k] = st.session_state[k]


def generate_reproduction_code() -> str:
    """Generate a standalone script that reproduces the current simulation.

    The configuration-building portion is emitted from the *same* builder calls the app
    actually makes (recorded into a ``_ReproRecorder`` by ``build_nominal_config_from_gui``),
    so the script mirrors the chosen builder — ``ModelBuilder`` / ``BinaryResistanceGenotypes``
    / ``StrainSet`` — call-for-call and can never drift from what is simulated. Only the
    initial-conditions / solve / plot tail is templated here (the Plotly charts in the app
    are rendered with matplotlib in the script, as noted).
    """
    config, iB, iP, iS, mkw = build_nominal_config_from_gui()
    rec = st.session_state["_repro_rec"]

    t_prerun = st.session_state.get("int_t_prerun", 0.0)

    imports = set(rec.imports) | {"PBIModel", "solve_ode"}
    if t_prerun > 0:
        imports.add("stationary_phase_ic")

    code = [
        "# \u2500\u2500 pbisim Auto-Generated Reproduction Script \u2500\u2500",
        "# Config built via the exact builder calls the app made (no re-derivation).",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from pbisim import " + ", ".join(sorted(imports)),
        "",
        "# 1. Build the model configuration",
        *rec.lines,
        "",
        "# 2. Initial conditions",
        f"initial_B = {rec.render(iB)}",
        f"initial_P = {rec.render(iP)}",
        f"initial_S = {rec.render(iS)}",
    ]

    if t_prerun > 0:
        _inherit_debris = (st.session_state.get("int_prerun_inherit_debris", True)
                           and st.session_state.get("int_debris_enabled", False))
        _debris_arg = ", initial_Debris=ic.Debris" if _inherit_debris else ""
        code.append(f"# Stationary-phase pre-run ({t_prerun} h) starting from the configured inoculum;")
        code.append("# carry the full final state — active B, dormant D, nutrient S, immune Imm"
                     + (", debris." if _inherit_debris else " (dead-cell debris washed out)."))
        code.append(f"ic = stationary_phase_ic(cfg, t_prerun={t_prerun}, B0=initial_B)")
        code.append(
            "model = PBIModel(cfg, initial_B=ic.B, initial_P=initial_P, "
            f"initial_S=max(float(ic.S), 0.0), initial_D=ic.D, initial_Imm=(ic.Imm or 0.0){_debris_arg})"
        )
    else:
        _extra_ic = "".join(f", {k}={rec.render(v)}" for k, v in mkw.items())
        code.append(
            f"model = PBIModel(cfg, initial_B=initial_B, initial_P=initial_P, initial_S=initial_S{_extra_ic})"
        )

    _method = st.session_state.get("int_solver_method", "BDF")
    _thresh = st.session_state.get("int_extinction_threshold", 1.0) or None
    _check_int = st.session_state.get("int_extinction_check_interval", 0.0) or None
    code += [
        "",
        "# 3. Solve",
        f"result = solve_ode(model, t_end={st.session_state.get('int_t_end', 48.0)}, "
        f"dt={st.session_state.get('int_dt', 0.25)}, method='{_method}', "
        f"extinction_threshold={_thresh}, extinction_check_interval={_check_int})",
        "",
        "# 4. Plot trajectories",
        "fig, ax = plt.subplots(figsize=(8, 4))",
        "ax.semilogy(result.time, np.maximum(result.sum_prefixes('B', 'D', 'I', 'H'), 1.0), label='Total Viable')",
        "ax.set(xlabel='Time (h)', ylabel='Density (cells/mL)', title='Simulation Run')",
        "ax.legend()",
        "plt.show()",
    ]

    _script = "\n".join(code)
    st.session_state["_last_repro_code"] = _script  # for tests / debugging
    return _script


def _repro_base_config_block(rec, iB, iP, iS, mk, extra_imports=(), initial_P_override=None):
    """Shared prefix for sweep reproduction scripts: imports + the recorded base-config
    builder calls + initial_B/P/S/model_kwargs statements. ``initial_P_override`` lets the
    caller substitute a modified initial_P array (e.g. swept phages zeroed)."""
    imports = set(rec.imports) | {"PBIModel", "solve_ode"} | set(extra_imports)
    lines = [
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from pbisim import " + ", ".join(sorted(imports)),
        "from pbisim_app.sweep_helper import apply_sweep_parameter",
        "",
        "# 1. Nominal model configuration (exact shadow of the app's builder calls)",
        *rec.lines,
        f"initial_B = {rec.render(iB)}",
        f"initial_P = {rec.render(iP if initial_P_override is None else initial_P_override)}",
        f"initial_S = {rec.render(iS)}",
        f"model_kwargs = {rec.render(mk)}",
    ]
    return imports, lines


def _repro_solve_kwargs():
    _thresh = st.session_state.get("int_extinction_threshold", 1.0) or None
    _check = st.session_state.get("int_extinction_check_interval", 0.0) or None
    return (f"t_end={st.session_state.get('int_t_end', 48.0)}, dt={st.session_state.get('int_dt', 0.25)}, "
            f"method='{st.session_state.get('int_solver_method', 'BDF')}', "
            f"extinction_threshold={_thresh}, extinction_check_interval={_check}")


def _repro_prerun_lines(indent, t_prerun):
    """Emit the per-run pre-run block (carry B/D/S/Imm) at the given indent, or []."""
    if not t_prerun:
        return []
    p = indent
    lines = [
        f"{p}ic = stationary_phase_ic(c_k, t_prerun={t_prerun}, B0=ib_k)",
        f"{p}ib_k, is_k = ic.B, max(float(ic.S), 0.0)",
        f"{p}if ic.D is not None: mk_k['initial_D'] = ic.D",
        f"{p}if ic.Imm is not None: mk_k['initial_Imm'] = ic.Imm",
    ]
    if (st.session_state.get("int_prerun_inherit_debris", True)
            and st.session_state.get("int_debris_enabled", False)):
        lines.append(f"{p}if ic.Debris is not None: mk_k['initial_Debris'] = ic.Debris")
    return lines


def generate_param_sweep_reproduction_code() -> str:
    """Reproduction script for a 1D parameter sweep: the recorded base config plus a loop
    calling the app's own ``apply_sweep_parameter`` per value (no re-derivation)."""
    if st.session_state.get("ps_sweep_type", "1D Sweep") != "1D Sweep":
        return ("# Reproduction code is currently available for 1D parameter sweeps only.\n"
                "# Switch 'Sweep Dimension' to '1D Sweep' to export a script.")
    config, iB, iP, iS, mk = build_nominal_config_from_gui()
    rec = st.session_state["_repro_rec"]

    sweep_params = get_sweep_parameters(
        config, st.session_state.get("int_strains", []),
        st.session_state.get("int_phages", []), st.session_state.get("int_antibiotics", []))
    label = st.session_state.get("p1_sweep_label")
    if not label or label not in sweep_params:
        return "# Configure and run a 1D parameter sweep first."
    meta = sweep_params[label]

    t_prerun = st.session_state.get("int_t_prerun", 0.0)
    imports, code = _repro_base_config_block(
        rec, iB, iP, iS, mk, extra_imports={"stationary_phase_ic"} if t_prerun else set())
    code[0:0] = ["# ── pbisim 1D Parameter-Sweep Reproduction Script ──"]

    _min = st.session_state.get("ps_1d_min")
    _max = st.session_state.get("ps_1d_max")
    _steps = int(st.session_state.get("ps_1d_steps", 5))
    _log = st.session_state.get("ps_1d_spacing", "Linear") == "Logarithmic"

    code += ["", "# 2. Sweep definition", f"meta = {rec.render(meta)}"]
    if _log:
        code.append(f"sweep_values = np.logspace(np.log10({_min}), np.log10({_max}), {_steps})")
    else:
        code.append(f"sweep_values = np.linspace({_min}, {_max}, {_steps})")
    if meta["type"] == "dimension":
        code.append("sweep_values = np.unique(np.clip(np.round(sweep_values).astype(int), 1, None))")

    _label_lit = label.replace('"', "'")
    code += [
        "",
        "# 3. Run the sweep (apply_sweep_parameter is the same function the app uses)",
        "fig, ax = plt.subplots(figsize=(8, 4))",
        "for val in sweep_values:",
        "    c_k, ib_k, ip_k, is_k, mk_k = apply_sweep_parameter(",
        "        val, meta, cfg, initial_B, initial_P, initial_S, model_kwargs)",
        *_repro_prerun_lines("    ", t_prerun),
        "    model = PBIModel(c_k, initial_B=ib_k, initial_P=ip_k, initial_S=is_k, **mk_k)",
        f"    result = solve_ode(model, {_repro_solve_kwargs()})",
        "    total = np.maximum(result.sum_prefixes('B', 'D', 'I', 'H'), 1.0)",
        '    ax.semilogy(result.time, total, label="' + _label_lit + ' = %.2e" % val)',
        f'ax.set(xlabel="Time (h)", ylabel="Total viable (cells/mL)", title="1D sweep: {_label_lit}")',
        "ax.legend(fontsize=7)",
        "plt.show()",
    ]
    _script = "\n".join(code)
    st.session_state["_last_sweep_repro_code"] = _script
    return _script


def generate_dose_sweep_reproduction_code() -> str:
    """Reproduction script for the dose-response sweep: the recorded base config plus a
    loop that rebuilds the per-run dose schedule exactly as the app does (MOI scaling,
    repeat regimens, nominal overrides) and re-solves."""
    phages = st.session_state.get("int_phages", [])
    antibiotics = st.session_state.get("int_antibiotics", [])
    strains = st.session_state.get("int_strains", [])

    # Reconstruct the sweep inputs from the dr_sweep_* widgets (mirrors the page).
    swept_inputs, swept_units, swept_repeat_configs = {}, {}, {}
    for j in range(len(phages)):
        if st.session_state.get(f"dr_sweep_phg_en_{j}"):
            swept_inputs[f"phage_{j}"] = st.session_state.get(f"dr_sweep_phg_series_{j}", "0, 1e3, 1e5, 1e7, 1e9")
            swept_units[f"phage_{j}"] = st.session_state.get(f"dr_sweep_phg_unit_{j}", "PFU (absolute)")
            if st.session_state.get(f"dr_sweep_phg_rep_en_{j}"):
                swept_repeat_configs[f"phage_{j}"] = {
                    "interval": st.session_state.get(f"dr_sweep_phg_rep_int_{j}", 12.0),
                    "count": int(st.session_state.get(f"dr_sweep_phg_rep_count_{j}", 4)),
                    "start": st.session_state.get(f"dr_sweep_phg_rep_start_{j}", 0.0),
                    "route": st.session_state.get(f"dr_sweep_phg_rep_route_{j}", "bolus"),
                    "duration": st.session_state.get(f"dr_sweep_phg_rep_dur_{j}", 0.0),
                }
    for j in range(len(antibiotics)):
        if st.session_state.get(f"dr_sweep_abx_en_{j}"):
            swept_inputs[f"abx_{j}"] = st.session_state.get(f"dr_sweep_abx_series_{j}", "0.5, 1.0, 2.0")
            swept_units[f"abx_{j}"] = "absolute"
            if st.session_state.get(f"dr_sweep_abx_rep_en_{j}"):
                swept_repeat_configs[f"abx_{j}"] = {
                    "interval": st.session_state.get(f"dr_sweep_abx_rep_int_{j}", 12.0),
                    "count": int(st.session_state.get(f"dr_sweep_abx_rep_count_{j}", 4)),
                    "start": st.session_state.get(f"dr_sweep_abx_rep_start_{j}", 0.0),
                    "route": st.session_state.get(f"dr_sweep_abx_rep_route_{j}", "bolus"),
                    "duration": st.session_state.get(f"dr_sweep_abx_rep_dur_{j}", 0.0),
                }
    if not swept_inputs:
        return "# Enable at least one phage/antibiotic dose sweep, then reopen this."

    parsed = {}
    for k, s in swept_inputs.items():
        parsed[k] = parse_comma_separated_series(s)
    padded, _ = pad_vectors(parsed)

    config, iB, iP, iS, mk = build_nominal_config_from_gui()
    rec = st.session_state["_repro_rec"]

    # Swept phages start at zero free phage — the dose delivers them (mirrors the app).
    swept_phage_idx = [int(k.split("_")[1]) for k in padded if k.startswith("phage_")]
    iP_mod = np.array(iP, dtype=float).copy()
    for jj in swept_phage_idx:
        if jj < len(iP_mod):
            iP_mod[jj] = 0.0

    original_doses = list(st.session_state.get("int_doses", []))
    sum_initial_B = float(sum(s["initial_B"] for s in strains))
    t_prerun = st.session_state.get("int_t_prerun", 0.0)

    _extra = {"DoseEvent", "DoseSchedule"}
    if t_prerun:
        _extra.add("stationary_phase_ic")
    imports, code = _repro_base_config_block(
        rec, iB, iP, iS, mk, extra_imports=_extra, initial_P_override=iP_mod)
    code[0:0] = ["# ── pbisim Dose-Response Sweep Reproduction Script ──"]

    code += [
        "",
        "# 2. Sweep definition (padded value series per swept agent)",
        f"padded = {padded!r}",
        f"swept_units = {swept_units!r}",
        f"swept_repeat_configs = {swept_repeat_configs!r}",
        f"original_doses = {original_doses!r}",
        f"sum_initial_B = {sum_initial_B!r}",
        "M = len(next(iter(padded.values())))",
        "",
        "# 3. Run one simulation per dose combination",
        "fig, ax = plt.subplots(figsize=(8, 4))",
        "for k in range(M):",
        "    custom = [nd for nd in original_doses if not (",
        "        (nd['target_type'] == 'phage' and f\"phage_{nd['target_idx']}\" in padded) or",
        "        (nd['target_type'] == 'antibiotic' and f\"abx_{nd['target_idx']}\" in padded))]",
        "    for key, vec in padded.items():",
        "        val = vec[k]",
        "        if swept_units.get(key) == 'MOI (relative to B(0))':",
        "            val = val * sum_initial_B",
        "        tt = 'phage' if key.startswith('phage') else 'antibiotic'",
        "        ti = int(key.split('_')[1])",
        "        if key in swept_repeat_configs:",
        "            rc = swept_repeat_configs[key]",
        "            for r in range(rc['count']):",
        "                custom.append({'time': rc['start'] + r * rc['interval'], 'amount': val,",
        "                               'target_type': tt, 'target_idx': ti, 'route': rc['route'], 'duration': rc['duration']})",
        "        else:",
        "            nom = [d for d in original_doses if d['target_type'] == tt and d['target_idx'] == ti]",
        "            if nom:",
        "                for nd in nom:",
        "                    c = dict(nd); c['amount'] = val; custom.append(c)",
        "            else:",
        "                custom.append({'time': 0.0, 'amount': val, 'target_type': tt,",
        "                               'target_idx': ti, 'route': 'bolus', 'duration': 0.0})",
        "    events = [DoseEvent(time=d['time'], amount=d['amount'],",
        "                        target=('phage' if d['target_type'] == 'phage' else",
        "                                'antibiotic' if d['target_type'] == 'antibiotic' else 'nutrient'),",
        "                        index=d['target_idx'], route=d['route'], duration=d.get('duration', 0.0))",
        "              for d in custom]",
        "    cfg.dose_schedule = DoseSchedule(events) if events else None",
        "    c_k, ib_k, ip_k, is_k, mk_k = cfg, np.array(initial_B, float), np.array(initial_P, float), initial_S, dict(model_kwargs)",
        *_repro_prerun_lines("    ", t_prerun),
        "    model = PBIModel(c_k, initial_B=ib_k, initial_P=ip_k, initial_S=is_k, **mk_k)",
        f"    result = solve_ode(model, {_repro_solve_kwargs()})",
        "    total = np.maximum(result.sum_prefixes('B', 'D', 'I', 'H'), 1.0)",
        '    ax.semilogy(result.time, total, label="run %d" % (k + 1))',
        'ax.set(xlabel="Time (h)", ylabel="Total viable (cells/mL)", title="Dose-response sweep")',
        "ax.legend(fontsize=7)",
        "plt.show()",
    ]
    _script = "\n".join(code)
    st.session_state["_last_sweep_repro_code"] = _script
    return _script


# ── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("pbisim")

    # Apply any pending programmatic navigation (from Load buttons, etc.) BEFORE
    # the radio is instantiated — a keyed widget's value can only be set prior to
    # its creation, and once set it overrides the `index=` default.
    _pending_nav = st.session_state.pop("_nav_to", None)
    if _pending_nav:
        st.session_state.current_page_radio = _pending_nav

    _pages = ["Interactive Simulator", "Dose-Response Sweeps", "Parameter Sweeps", "Clinical Trials & Cohorts", "Calibration", "AI Assistant", "Library"]
    st.session_state.current_page = st.radio(
        "Navigation",
        _pages,
        key="current_page_radio",
        index=_pages.index(st.session_state.current_page),
    )
    st.session_state.current_page = st.session_state.current_page_radio

    st.markdown("---")
    st.markdown("### ⚙️ AI Settings")

    # API key — the masked field is a browser "password" input; re-rendering it on every
    # navigation/rerun makes Chrome repeatedly offer to save/update the password. So once a
    # key is set we do NOT render the field: we show a compact status + a Change button, and
    # only render the input when there is no key (or the user clicks Change). No password
    # field in the DOM during normal use → no repeated Chrome prompt.
    _env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if "api_key" not in st.session_state:
        st.session_state.api_key = _env_key

    if st.session_state.api_key and not st.session_state.get("_editing_api_key", False):
        _src = "from environment" if st.session_state.api_key == _env_key and _env_key else "entered"
        st.caption(f"🔑 Anthropic API key set ({_src})")
        if st.button("Change key", key="change_api_key", width="stretch"):
            st.session_state._editing_api_key = True
            st.rerun()
    else:
        _entered = st.text_input(
            "Anthropic API Key",
            value="",
            type="password",
            key="api_key_field",
            help="Required ONLY for the AI Assistant. Local simulation runs entirely offline. "
                 "Once set, the field is hidden so your browser stops prompting to save it.",
        )
        if _entered:
            st.session_state.api_key = _entered
            st.session_state._editing_api_key = False
            st.rerun()
        if st.session_state.api_key and st.button("Clear key", key="clear_api_key", width="stretch"):
            st.session_state.api_key = ""
            st.session_state._editing_api_key = False
            st.rerun()

    api_key = st.session_state.api_key
    if api_key:
        if st.session_state.agent.client.api_key != api_key:
            st.session_state.agent.client.api_key = api_key
            st.session_state.api_models_list = []
    else:
        if st.session_state.agent.client.api_key != "":
            st.session_state.agent.client.api_key = ""
            st.session_state.api_models_list = []

    # Fetch models dynamically if API key is present and list is empty
    if st.session_state.agent.client.api_key and not st.session_state.api_models_list:
        try:
            models_page = st.session_state.agent.client.models.list()
            fetched_ids = [m.id for m in models_page.data]
            if fetched_ids:
                st.session_state.api_models_list = fetched_ids
        except Exception:
            # Store sentinel to prevent repeated API calls failing in loop
            st.session_state.api_models_list = ["__FAILED__"]

    default_model_ops = [
        "claude-opus-4-8",          # default — strongest for one-shot code generation
        "claude-sonnet-4-6",        # faster / cheaper
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-fable-5",
    ]

    if st.session_state.api_models_list and st.session_state.api_models_list != ["__FAILED__"]:
        model_ops = list(st.session_state.api_models_list)
    else:
        model_ops = list(default_model_ops)

    current_agent_model = st.session_state.agent.model
    is_custom = current_agent_model not in model_ops

    if "Custom Model ID..." not in model_ops:
        model_ops.append("Custom Model ID...")

    try:
        if is_custom:
            idx = model_ops.index("Custom Model ID...")
        else:
            idx = model_ops.index(current_agent_model)
    except ValueError:
        idx = 0

    selected_model = st.selectbox(
        "Claude Model",
        model_ops,
        index=idx,
        help="Choose the Claude model to power the AI Assistant."
    )

    if selected_model == "Custom Model ID...":
        custom_model = st.text_input("Enter Model ID", value=current_agent_model if is_custom else "claude-3-5-haiku-latest")
        st.session_state.agent.model = custom_model
    else:
        st.session_state.agent.model = selected_model

    if st.button("🔍 Test API Key & List Models", key="test_api_key_btn"):
        if not st.session_state.agent.client.api_key:
            st.error("Please enter an Anthropic API Key first in the sidebar.")
        else:
            with st.spinner("Connecting to Anthropic..."):
                try:
                    models_page = st.session_state.agent.client.models.list()
                    model_ids = [m.id for m in models_page.data]
                    st.success("API Key is valid!")
                    st.markdown("**Authorized Models for this Key:**")
                    st.write(model_ids)
                except Exception as e:
                    st.error(f"API Diagnostics Failed: {e}")
                    st.info("💡 Note: If you get a 404 error here, your key is authentic but has no models enabled (often because the Anthropic account is at Tier 0/unfunded). If you get a 401, the key is invalid.")




    show_code = st.toggle("Show generated code", value=True)
    show_assumptions = st.toggle("Show assumptions", value=True)

    st.markdown("---")
    st.markdown("### 🎨 Appearance")
    st.session_state["theme_mode"] = st.selectbox(
        "Theme Mode",
        ["Light", "Dark"],
        index=["Light", "Dark"].index(st.session_state.get("theme_mode", "Light")),
        key="theme_mode_selectbox"
    )

    st.markdown("---")
    if st.button("🔄 Reset Environment"):
        st.session_state.agent.reset()
        st.session_state.history.clear()
        st.session_state.simulation_result = None
        st.session_state.simulation_config = None
        st.session_state.trial_result = None
        load_preset_to_state(DEFAULT_SCENARIO)
        st.rerun()


# Flash message carried across a rerun/navigation (e.g. after loading a part).
_flash = st.session_state.pop("_flash", None)
if _flash:
    (st.warning if _flash.get("kind") == "warning" else st.success)(_flash["msg"])


# ── Library Page (Scenarios + Parts) ──────────────────────────────────────────
if st.session_state.current_page == "Library":
    st.title("Library")
    st.caption("Reusable building blocks. **Scenarios** = whole configurations; "
               "**Parts** = individual bacteria / phages / antibiotics you compose.")

    st.markdown("## 💾 Scenarios")
    st.markdown(
        "<div class='info-banner'>💡 A scenario captures your <b>entire</b> configuration. "
        "Loading one configures the <b>Interactive Simulator</b> and applies across all pages "
        "(sweeps, clinical trials). Export to JSON to keep a portable personal library.</div>",
        unsafe_allow_html=True,
    )

    # ── My Scenarios (save / load / export / import full configurations) ──────
    st.caption(
        "Save the **entire current configuration** (builder mode, strains/phages/"
        "antibiotics, dosing, nutrient, immune, solver, prerun, and trial design) as a "
        "reusable scenario. Scenarios live in this browser session — **export to JSON to "
        "keep them** (your portable personal library) and re-import any time."
    )
    _scenarios = st.session_state.user_scenarios

    sc_save, sc_io = st.columns(2)
    with sc_save:
        with st.expander("➕ Save current configuration", expanded=not _scenarios):
            _sc_name = st.text_input("Scenario name", value=f"Scenario {len(_scenarios) + 1}", key="sc_save_name")
            _sc_note = st.text_area(
                "Annotation (optional)", key="sc_save_note",
                placeholder="e.g. PA high-persister + fast-adsorbing phage, immunocompromised host",
            )
            if st.button("💾 Save scenario", key="sc_save_btn", width="stretch"):
                _name = (_sc_name or "").strip()
                if not _name:
                    st.error("Please enter a scenario name.")
                else:
                    _scenarios[_name] = {
                        "annotation": _sc_note or "",
                        "schema_version": SCENARIO_SCHEMA_VERSION,
                        "state": dump_state_to_scenario(),
                    }
                    st.session_state.user_scenarios = _scenarios
                    st.success(f"Saved '{_name}'.")
                    st.rerun()
    with sc_io:
        with st.expander("📤 Export / 📥 Import library", expanded=False):
            st.download_button(
                "📤 Export all scenarios (JSON)",
                data=export_scenarios_json(_scenarios),
                file_name="pbisim_scenarios.json",
                mime="application/json",
                width="stretch",
                disabled=not _scenarios,
            )
            _up = st.file_uploader("📥 Import scenarios (JSON)", type=["json"], key="sc_import")
            if _up is not None and st.button("Merge imported scenarios", key="sc_import_btn"):
                try:
                    imported = import_scenarios_json(_up.getvalue().decode("utf-8"))
                    _scenarios.update(imported)
                    st.session_state.user_scenarios = _scenarios
                    st.success(f"Imported {len(imported)} scenario(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    if _scenarios:
        st.markdown("#### Saved scenarios")
        for _name in list(_scenarios.keys()):
            _sc = _scenarios[_name]
            c_info, c_load, c_del = st.columns([6, 1, 1])
            with c_info:
                _note = _sc.get("annotation", "")
                st.markdown(f"**{_name}**" + (f" — {_note}" if _note else ""))
            with c_load:
                if st.button("Load", key=f"sc_load_{_name}"):
                    load_scenario_to_state(_sc["state"])
                    st.session_state._nav_to = "Interactive Simulator"
                    st.success(f"Loaded '{_name}'.")
                    st.rerun()
            with c_del:
                if st.button("🗑️", key=f"sc_del_{_name}"):
                    _scenarios.pop(_name, None)
                    st.session_state.user_scenarios = _scenarios
                    st.rerun()
    else:
        st.info("No saved scenarios yet — configure a simulation, then save it above.")

    # ── 🧬 Parts (composable building blocks) ────────────────────────────────
    st.markdown("---")
    st.markdown("## 🧬 Parts")
    st.caption(
        "Save individual **bacteria / phages / antibiotics** as reusable parts and compose "
        "them into any configuration. Loading a part adds it to the current strains / phages "
        "/ antibiotics (shared across all pages). Phage kinetics (burst / latent / adsorption) "
        "depend on the host, so phage parts record the **reference host** they were "
        "characterised against and warn if you reuse them elsewhere."
    )
    _lib = st.session_state.parts_library

    with st.expander("📤 Export / 📥 Import parts library (JSON)"):
        _has_parts = any(_lib[c] for c in PART_CATEGORIES)
        st.download_button(
            "📤 Export parts (JSON)", data=export_parts_json(_lib),
            file_name="pbisim_parts.json", mime="application/json",
            width="stretch", disabled=not _has_parts,
        )
        _pup = st.file_uploader("📥 Import parts (JSON)", type=["json"], key="parts_import")
        if _pup is not None and st.button("Merge imported parts", key="parts_import_btn"):
            try:
                _imported = import_parts_json(_pup.getvalue().decode("utf-8"))
                for _c in PART_CATEGORIES:
                    _lib[_c].update(_imported.get(_c, {}))
                st.session_state.parts_library = _lib
                st.success("Parts imported.")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

    _part_tabs = st.tabs([PART_CATEGORIES[c]["label"] for c in PART_CATEGORIES])
    for _tab, _cat in zip(_part_tabs, PART_CATEGORIES):
        with _tab:
            _meta = PART_CATEGORIES[_cat]
            _singular = _meta["label"][:-1].lower() if _meta["label"].endswith("s") else _meta["label"].lower()
            _entities = st.session_state.get(_meta["key"], [])
            _store = _lib[_cat]

            with st.expander(f"➕ Save a current {_singular} as a part", expanded=not _store):
                if not _entities:
                    st.info(f"No {_meta['label'].lower()} configured yet — set one up in the Interactive Simulator first.")
                else:
                    _names = [f"{i}: {e.get('name', _singular)}" for i, e in enumerate(_entities)]
                    _pick_label = st.selectbox(f"Which {_singular}?", _names, key=f"part_pick_{_cat}")
                    _pick = _names.index(_pick_label) if _pick_label in _names else 0
                    _pname = st.text_input("Part name", value=_entities[_pick].get("name", _singular), key=f"part_name_{_cat}")
                    _psrc = st.selectbox("Source (provenance)", PART_SOURCES, key=f"part_src_{_cat}")
                    _pnote = st.text_area(
                        "Annotation", key=f"part_note_{_cat}",
                        placeholder="e.g. PA clinical isolate; high persister fraction",
                    )
                    _pref = ""
                    if _cat == "phages":
                        _sn = [s.get("name", "") for s in st.session_state.get("int_strains", [])]
                        _pref = st.selectbox(
                            "Reference host (bacterium it was characterised against)",
                            _sn + ["(unspecified)"], key=f"part_refhost_{_cat}",
                            help="Burst/latent/adsorption are phage×host properties — record the host so reuse elsewhere is flagged.",
                        )
                        _pref = "" if _pref == "(unspecified)" else _pref
                    if st.button("💾 Save part", key=f"part_save_{_cat}", width="stretch"):
                        _nm = (_pname or "").strip()
                        if not _nm:
                            st.error("Please enter a part name.")
                        else:
                            _entry = {
                                "source": _psrc,
                                "annotation": _pnote or "",
                                "params": _json_safe(copy.deepcopy(_entities[_pick])),
                            }
                            if _cat == "phages":
                                _entry["reference_host"] = _pref
                            _store[_nm] = _entry
                            st.session_state.parts_library = _lib
                            st.success(f"Saved {_singular} part '{_nm}'.")
                            st.rerun()

            if not _store:
                st.caption("No saved parts yet.")
            for _pn in list(_store.keys()):
                _p = _store[_pn]
                _ci, _cl, _cd = st.columns([6, 1, 1])
                with _ci:
                    _bits = [f"**{_pn}**", f"`{_p.get('source', '?')}`"]
                    if _cat == "phages" and _p.get("reference_host"):
                        _bits.append(f"· host *{_p['reference_host']}*")
                    if _p.get("annotation"):
                        _bits.append("— " + _p["annotation"])
                    st.markdown(" ".join(_bits))
                with _cl:
                    if st.button("Load", key=f"part_load_{_cat}_{_pn}"):
                        _cur = list(st.session_state.get(_meta["key"], []))
                        if len(_cur) >= _meta["max"]:
                            st.warning(f"At most {_meta['max']} {_meta['label'].lower()} are supported — remove one first.")
                        else:
                            _cur.append(copy.deepcopy(_p["params"]))
                            st.session_state[_meta["key"]] = _cur
                            clear_entity_widgets()
                            _kind, _msg = "success", f"Added '{_pn}' to the configuration."
                            if _cat == "phages" and _p.get("reference_host"):
                                _sn = [s.get("name", "") for s in st.session_state.get("int_strains", [])]
                                if _p["reference_host"] not in _sn:
                                    _kind = "warning"
                                    _msg = (f"Added '{_pn}', but it was characterised against "
                                            f"'{_p['reference_host']}', which isn't among your current "
                                            "strains — verify burst/latent/adsorption for your host.")
                            st.session_state._flash = {"kind": _kind, "msg": _msg}
                            st.session_state._nav_to = "Interactive Simulator"
                            st.rerun()
                with _cd:
                    if st.button("🗑️", key=f"part_del_{_cat}_{_pn}"):
                        _store.pop(_pn, None)
                        st.session_state.parts_library = _lib
                        st.rerun()


# ── Calibration Page (Phase A: data upload + overlay + fit metric) ────────────
elif st.session_state.current_page == "Calibration":
    st.title("Calibration — data overlay")
    st.caption(
        "Upload experimental data and overlay the **current model's** prediction (configured in "
        "the Interactive Simulator) on the observations. Tune parameters there to match; a "
        "manual-tuning panel and the pbisim-fit hand-off come next."
    )

    # Re-seed the Calibration widgets from a persistent config BEFORE they render.
    # Streamlit drops a widget's key from session_state whenever the widget isn't
    # rendered on a rerun, so navigating to the Simulator (to change the model) and
    # back would otherwise reset the filters / grouping / statistics. `fit_config`
    # is a plain (non-widget) key, so it survives.
    # Buttons and the file-uploader can't be re-seeded via session_state, so they
    # are never persisted; everything else (filters/grouping/statistics/overlay
    # selections) is.
    # Buttons + the file-uploader must never be shadowed into fit_config: re-seeding a
    # button's value pre-sets it, which makes the later st.button() raise. (Text/number
    # widgets are fine to persist.)
    _FIT_NOPERSIST = {"fit_csv", "fit_config", "fit_dataset", "fit_overlay", "fit_clear",
                      "fit_load", "fit_save_scenario"}
    _fcfg = st.session_state.setdefault("fit_config", {})
    for _wk, _wv in list(_fcfg.items()):
        if _wk in _FIT_NOPERSIST:
            _fcfg.pop(_wk, None)  # scrub any stale non-persistable key
            continue
        if _wk not in st.session_state:
            try:
                st.session_state[_wk] = _wv
            except Exception:
                pass  # widget type refuses assignment (e.g. a button) — skip it

    # ── 1. Upload + column mapping ───────────────────────────────────────────
    st.markdown("### 1 · Upload data")
    _up = st.file_uploader("Experimental data (CSV)", type=["csv"], key="fit_csv")
    if _up is not None:
        try:
            _raw = read_uploaded_csv(_up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            _raw = None
        if _raw is not None:
            st.dataframe(_raw.head(8), width="stretch")
            _cols = list(_raw.columns)
            _low = [c.lower() for c in _cols]

            def _guess(cands, default=0):
                for c in cands:
                    if c in _low:
                        return _low.index(c)
                return default

            _canonical = all(k in _low for k in ("time", "arm", "observable", "value"))
            if _canonical:
                st.success("Detected pbisim-fit long format (time, arm, observable, value).")
            with st.expander("🔧 Column mapping", expanded=not _canonical):
                _tc = st.selectbox("Time column", _cols, index=_guess(["time"]))
                _vc = st.selectbox("Value (measurement) column", _cols, index=_guess(["value", "dv"]))
                _obs_from_col = st.checkbox("Observable is in a column", value=("observable" in _low))
                if _obs_from_col:
                    _obs = st.selectbox("Observable column", _cols, index=_guess(["observable"]))
                else:
                    _obs = st.selectbox("Observable type (fixed for all rows)", list(OBSERVABLES),
                                        format_func=lambda k: OBSERVABLES[k]["label"])
                _default_arms = [c for c in _cols if c.lower() in ("phage", "moi", "arm", "experi", "experi_num")]
                _ac = st.multiselect("Arm-defining column(s)", _cols, default=_default_arms or ([_cols[0]] if _cols else []))
                _mc = st.selectbox("Phage-dose / MOI column (optional — drives the simulated dose per arm)",
                                   ["(none)"] + _cols, index=(1 + _guess(["moi", "dose_phage"])) if ("moi" in _low or "dose_phage" in _low) else 0)
                _mc = None if _mc == "(none)" else _mc
            if st.button("📥 Load dataset", key="fit_load", width="stretch"):
                st.session_state.fit_dataset = {
                    "raw": _raw, "time": _tc, "value": _vc, "observable": _obs,
                    "arm_cols": _ac, "moi": _mc,
                }
                st.success(f"Loaded {len(_raw)} rows. Configure grouping / filters / statistics below.")
                st.rerun()

    # ── 2. Filter · group · statistics · overlay ─────────────────────────────
    _ds = st.session_state.get("fit_dataset")
    if not _ds:
        st.info("Upload a dataset above to begin.")
    else:
        _raw = _ds["raw"]
        _cols = list(_raw.columns)
        _tc, _vc, _obs, _mc = _ds["time"], _ds["value"], _ds["observable"], _ds["moi"]

        # -- Filters --------------------------------------------------------
        st.markdown("### 2 · Filter rows")
        _filter_cols = st.multiselect(
            "Filter on column(s) (leave a value list empty = include all)", _cols,
            default=[], key="fit_filter_cols",
        )
        _filters = {}
        for _fc in _filter_cols:
            _uniques = sorted(_raw[_fc].dropna().astype(str).unique().tolist())
            _filters[_fc] = st.multiselect(f"Include {_fc} =", _uniques, default=[], key=f"fit_filter_{_fc}")

        # -- Grouping + statistic -------------------------------------------
        st.markdown("### 3 · Grouping & statistics")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            _group_cols = st.multiselect("Grouping variables (define the curves/arms)", _cols,
                                         default=_ds["arm_cols"], key="fit_group_cols")
        with gc2:
            _stat = st.selectbox("Statistic over replicates", ["Raw points", "Mean", "Median"], key="fit_stat")
        with gc3:
            _band_choice = st.selectbox("Percentile band", ["None", "10–90", "25–75", "5–95"],
                                        index=0, disabled=(_stat == "Raw points"), key="fit_band")
        _stat_key = {"Raw points": "raw", "Mean": "mean", "Median": "median"}[_stat]
        _band = None if (_band_choice == "None" or _stat_key == "raw") else tuple(int(x) for x in _band_choice.split("–"))

        # Filter → normalise → aggregate (cached; recomputes only when inputs change)
        try:
            _filters_key = tuple((c, tuple(v)) for c, v in _filters.items())
            _filtered, _long, _conds, _agg = calibration_processed(
                _raw, _filters_key, _tc, _vc, _obs, tuple(_group_cols), _mc, _stat_key, _band)
        except Exception as e:
            st.error(f"Could not build the grouped dataset: {e}")
            _filtered, _long, _agg = _raw, None, None
        st.caption(f"{len(_filtered)} / {len(_raw)} rows after filtering.")

        if _long is not None and len(_long):
            _arms = sorted(_long["arm"].unique())
            _obs_keys = sorted(_long["observable"].unique())

            st.markdown("### 4 · Overlay")
            st.caption(f"{len(_arms)} group(s) · pick which to overlay against the current model.")
            oc1, oc2 = st.columns(2)
            with oc1:
                _sel_arms = st.multiselect("Groups to overlay", _arms, default=_arms[:min(4, len(_arms))], key="fit_arms")
            with oc2:
                _obs_key = st.selectbox("Observable", _obs_keys,
                                        format_func=lambda k: OBSERVABLES.get(k, {}).get("label", k), key="fit_obs")
            _spec = OBSERVABLES.get(_obs_key, {"log": True, "link": None, "label": _obs_key, "prefixes": ("B", "D", "I", "H")})
            _link_val = None
            # When the OD/debris module is on, OD comes from the model's debris-inclusive
            # get_od() (using od_to_cfu_conversion_factor, edited in the tuning panel's
            # Global & structural section) rather than the simple biomass/link scaling.
            _use_model_od = _obs_key == "od" and st.session_state.get("int_debris_enabled", False)
            lc1, lc2 = st.columns(2)
            if _spec.get("link") and not _use_model_od:
                _pname, _op, _default = _spec["link"]
                with lc1:
                    _link_val = st.number_input(f"Link parameter · {_pname}", value=float(_default), format="%.3e",
                                                key=f"fit_link_{_obs_key}",
                                                help="Scales model state → signal (OD = biomass / od_to_cfu; "
                                                     "luminescence = active biomass × rlu_per_cell). Tunable below / future fit param.")
            elif _use_model_od:
                with lc1:
                    st.caption("OD uses the **debris module** (`get_od`, includes lysed-cell debris). "
                               "Tune `od_to_cfu` and the debris rates in *Global & structural* below.")
            with lc2:
                _t_end_fit = st.number_input("Overlay duration (h)", value=float(np.ceil(_long["time"].max())), step=1.0, key="fit_tend")

            # ── 5. Manual parameter tuning (Phase B) ─────────────────────────
            # Edit the model's ACTUAL parameter values (absolute, per entity — like
            # the Interactive Simulator), not multipliers. These widgets read from
            # and write to the shared int_strains / int_phages dicts, so edits ARE
            # the live model: no separate "apply" step, and they're savable as Parts.
            # The widgets are seeded from the dict each render (value=), so they stay
            # in sync with edits made on the Simulator page.
            _tstrains = st.session_state.get("int_strains", [])
            _tphages = st.session_state.get("int_phages", [])
            with st.expander("🎛 Manual parameter tuning", expanded=False):
                st.caption("Edit the model's real parameter values, then re-overlay. Changes update the live "
                           "Interactive-Simulator model directly (no separate apply step) and can be saved as "
                           "a Scenario or as Parts in the Library.")

                # ── Global & structural parameters ───────────────────────────
                st.markdown("**Global & structural**")
                _track_nut = st.session_state.get("int_track_nutrients", True)
                gk1, gk2, gk3 = st.columns(3)
                with gk1:
                    st.session_state["int_n_latent"] = int(st.number_input(
                        "Latent compartments (L)", min_value=1, max_value=50,
                        value=int(st.session_state.get("int_n_latent", 5)), step=1, key="fit_edit_n_latent",
                        help="Number of phage latent (eclipse) stages — Erlang shape of the latent period."))
                with gk2:
                    st.session_state["int_carrying_capacity"] = st.number_input(
                        "Carrying capacity (K)", value=float(st.session_state.get("int_carrying_capacity", 1e9)),
                        format="%.3e", key="fit_edit_K")
                with gk3:
                    st.session_state["int_monod_constant"] = st.number_input(
                        "Monod constant (Ks)", value=float(st.session_state.get("int_monod_constant", 0.3)),
                        format="%g", key="fit_edit_Ks")
                if _track_nut:
                    nk1, nk2, nk3, nk4 = st.columns(4)
                    with nk1:
                        st.session_state["int_initial_S"] = st.number_input(
                            "Initial nutrient (S₀)", value=float(st.session_state.get("int_initial_S", 1.0)),
                            format="%g", key="fit_edit_S0")
                    with nk2:
                        st.session_state["int_recycle_fraction"] = st.number_input(
                            "Recycle fraction", value=float(st.session_state.get("int_recycle_fraction", 0.0)),
                            format="%g", key="fit_edit_recycle")
                    with nk3:
                        st.session_state["int_s_in"] = st.number_input(
                            "Nutrient inflow (s_in)", value=float(st.session_state.get("int_s_in", 0.0)),
                            format="%g", key="fit_edit_s_in")
                    with nk4:
                        st.session_state["int_s_out"] = st.number_input(
                            "Nutrient washout (s_out)", value=float(st.session_state.get("int_s_out", 0.0)),
                            format="%g", key="fit_edit_s_out")
                else:
                    st.caption("Nutrient tracking is off (constant/logistic growth) — S₀/recycle/inflow/washout "
                               "are inactive. Enable it in the Interactive Simulator to fit them.")
                if st.session_state.get("int_debris_enabled", False):
                    st.markdown("*OD / debris module*")
                    dk1, dk2, dk3, dk4 = st.columns(4)
                    with dk1:
                        st.session_state["int_od_to_cfu_conversion_factor"] = st.number_input(
                            "od_to_cfu", value=float(st.session_state.get("int_od_to_cfu_conversion_factor", 2e8)),
                            format="%.3e", key="fit_edit_od2cfu",
                            help="CFU per OD unit: OD = (biomass + debris) / od_to_cfu.")
                    with dk2:
                        st.session_state["int_debris_u"] = st.number_input(
                            "Debris yield · deaths (u)", value=float(st.session_state.get("int_debris_u", 0.4)),
                            format="%g", key="fit_edit_debris_u")
                    with dk3:
                        st.session_state["int_debris_v"] = st.number_input(
                            "Debris yield · lysis (v)", value=float(st.session_state.get("int_debris_v", 0.2)),
                            format="%g", key="fit_edit_debris_v")
                    with dk4:
                        st.session_state["int_debris_kdis"] = st.number_input(
                            "Debris dissolution (k_dis)", value=float(st.session_state.get("int_debris_kdis", 0.01)),
                            format="%g", key="fit_edit_debris_kdis")

                # Bacterial parameters. IMPORTANT: in Binary-Genotypes (BRG) mode the
                # strain kinetics live on `int_brg_base_*` session keys (a single WT base
                # from which the genotypes are derived), NOT the per-strain dicts — and
                # initial_B comes from the equilibrium IC / per-genotype table. So the
                # per-strain-dict editors below (correct for Direct / Custom-Strains)
                # would be silently ignored in BRG. Edit the right storage per mode.
                _is_brg = st.session_state.get("int_builder_mode", "").startswith("Binary")
                if _is_brg:
                    st.markdown("**Base strain (WT) — genotypes derived**")
                    _bc = st.columns(3)
                    with _bc[0]:
                        st.session_state["int_brg_base_growth"] = st.number_input(
                            "Growth rate (1/h)", value=float(st.session_state.get("int_brg_base_growth", 1.2)),
                            format="%g", key="fit_edit_brg_growth")
                    with _bc[1]:
                        st.session_state["int_brg_base_ratio"] = st.number_input(
                            "Bacteria/resource", value=float(st.session_state.get("int_brg_base_ratio", 1e9)),
                            format="%.2e", key="fit_edit_brg_ratio")
                    with _bc[2]:
                        st.session_state["int_brg_death_rate_B"] = st.number_input(
                            "Natural death (1/h)", value=float(st.session_state.get("int_brg_death_rate_B", 0.0)),
                            format="%g", key="fit_edit_brg_death")
                    if st.session_state.get("int_brg_use_eq_ic", False):
                        st.session_state["int_brg_eq_total_B"] = st.number_input(
                            "Total bacteria (equilibrium IC)",
                            value=float(st.session_state.get("int_brg_eq_total_B", 1e7)),
                            format="%.3e", key="fit_edit_brg_eqtotal",
                            help="With the equilibrium initial condition on, per-genotype B₀ is derived "
                                 "from this total (and fitness cost) — individual strain B₀ is not used.")
                    else:
                        st.caption("Per-genotype initial counts are set in the BRG phage-loci table on the "
                                   "Interactive Simulator.")
                _tune_strains = [] if _is_brg else _tstrains
                if _tune_strains:
                    st.markdown("**Bacterial strains**")
                for _si, _s in enumerate(_tune_strains):
                    st.markdown(f"*{_s.get('name', f'Strain {_si}')}*")
                    _scols = st.columns(len(STRAIN_TUNABLES))
                    for _sc, _knob in zip(_scols, STRAIN_TUNABLES):
                        with _sc:
                            _s[_knob["key"]] = st.number_input(
                                _knob["label"], value=float(_s.get(_knob["key"], _knob["default"]) or 0.0),
                                format=_knob["fmt"], key=f"fit_edit_s_{_knob['key']}_{_si}")
                    # dormancy kinetics + depth compartments — only when enabled
                    if _s.get("dormancy_enabled"):
                        _dcols = st.columns(len(STRAIN_DORMANCY_TUNABLES) + 1)
                        with _dcols[0]:
                            _s["dormancy_depth"] = int(st.number_input(
                                "Depth layers (Q)", min_value=1, max_value=10,
                                value=int(_s.get("dormancy_depth", 1)), step=1,
                                key=f"fit_edit_s_dormancy_depth_{_si}",
                                help="Number of dormancy-depth compartments (max across strains sets n_depth)."))
                        for _dc, _knob in zip(_dcols[1:], STRAIN_DORMANCY_TUNABLES):
                            with _dc:
                                _s[_knob["key"]] = st.number_input(
                                    _knob["label"], value=float(_s.get(_knob["key"], _knob["default"]) or 0.0),
                                    format=_knob["fmt"], key=f"fit_edit_s_{_knob['key']}_{_si}")

                # Adsorption is a strain×phage property; its storage is builder-mode
                # specific. Direct / Custom-Strains keep it in the pairwise
                # ads_{strain}_{phage} session keys (edited per pair here); Binary-
                # Genotypes keeps it on the phage dict as adsorption_s.
                _ads_pairwise = not st.session_state.get("int_builder_mode", "").startswith("Binary")

                if _tphages:
                    st.markdown("**Phages**")
                for _pj, _p in enumerate(_tphages):
                    st.markdown(f"*{_p.get('name', f'Phage {_pj}')}*")
                    _pcols = st.columns(len(PHAGE_TUNABLES))
                    for _pc, _knob in zip(_pcols, PHAGE_TUNABLES):
                        with _pc:
                            _p[_knob["key"]] = st.number_input(
                                _knob["label"], value=float(_p.get(_knob["key"], _knob["default"]) or 0.0),
                                format=_knob["fmt"], key=f"fit_edit_p_{_knob['key']}_{_pj}")
                    # Mutation rate / fitness cost — only in Binary-Genotypes, the only
                    # mode that reads them from the phage dict. (Direct-mode phage dicts
                    # may carry a stale `mu`, but Direct/Custom-Strains take mutation from
                    # the strain→strain graph edited on the Simulator, so editing it here
                    # would be a silent no-op.)
                    _opt = PHAGE_OPTIONAL_TUNABLES if _is_brg else []
                    if _opt:
                        _ocols = st.columns(len(_opt))
                        for _oc, _knob in zip(_ocols, _opt):
                            with _oc:
                                _p[_knob["key"]] = st.number_input(
                                    _knob["label"], value=float(_p.get(_knob["key"], _knob["default"]) or 0.0),
                                    format=_knob["fmt"], key=f"fit_edit_p_{_knob['key']}_{_pj}")
                    # adsorption inputs (per strain in pairwise modes: active + dormant)
                    if _ads_pairwise and _tstrains:
                        _acols = st.columns(len(_tstrains))
                        for _si, _s in enumerate(_tstrains):
                            _adk = f"ads_{_si}_{_pj}"
                            with _acols[_si]:
                                st.session_state[_adk] = st.number_input(
                                    f"Adsorption → {_s.get('name', f'Strain {_si}')}",
                                    value=float(st.session_state.get(_adk, 1e-8 if _si == 0 else 0.0)),
                                    format="%.3e", key=f"fit_edit_ads_{_si}_{_pj}")
                            # dormant-cell adsorption for strains with dormancy on
                            if _s.get("dormancy_enabled"):
                                _addk = f"ads_dorm_{_si}_{_pj}"
                                with _acols[_si]:
                                    st.session_state[_addk] = st.number_input(
                                        f"Dormant ads → {_s.get('name', f'Strain {_si}')}",
                                        value=float(st.session_state.get(_addk, 0.0)),
                                        format="%.3e", key=f"fit_edit_adsdorm_{_si}_{_pj}")
                    elif not _ads_pairwise:
                        _adk = entity_param_key(_p, ADSORPTION_PHAGE_KEYS)
                        _p[_adk] = st.number_input(
                            "Adsorption (adsorption_s)", value=float(_p.get(_adk, 5e-8) or 0.0),
                            format="%.3e", key=f"fit_edit_adss_{_pj}")
                        if "adsorption_r" in _p:
                            _p["adsorption_r"] = st.number_input(
                                "Adsorption resistant (adsorption_r)", value=float(_p.get("adsorption_r", 0.0) or 0.0),
                                format="%.3e", key=f"fit_edit_adsr_{_pj}")
                st.caption("Tip: B₀ may be overridden by an equilibrium/pre-run initial condition in some builder "
                           "modes; the phage inoculum in the overlay comes from each group's MOI × B₀.")

            # Compute the overlay only when the button is clicked; store the plot
            # data in session_state so the visualization stays alive across page
            # navigation (and reruns) until it is explicitly re-run or the dataset
            # is cleared.
            if st.button("🔬 Overlay model on data", key="fit_overlay", width="stretch"):
                try:
                    _config, _iB, _iP, _iS, _mk = build_nominal_config_from_gui()
                    _B0 = float(np.sum(_iB))
                    _method = st.session_state.get("int_solver_method", "BDF")
                    _thr = st.session_state.get("int_extinction_threshold", 1.0) or None
                    _series, _metrics = [], []
                    for _arm in _sel_arms:
                        _moi = float(_conds.get(_arm, {}).get("moi", 0.0))
                        _armP = np.zeros(len(_iP))
                        if len(_iP):
                            _armP[0] = _moi * _B0
                        _m = PBIModel(_config, initial_B=_iB, initial_P=_armP, initial_S=_iS, **_mk)
                        _r = solve_ode(_m, t_end=_t_end_fit, dt=0.25, method=_method, extinction_threshold=_thr)
                        _pred = predicted_observable(_r, _obs_key, _link_val, use_model_od=_use_model_od)
                        _d = _agg[(_agg["arm"] == _arm) & (_agg["observable"] == _obs_key)].sort_values("time")
                        _has_band = _band is not None and _d["lo"].notna().any()
                        _series.append({
                            "label": _arm,
                            "time": np.asarray(_r.time),
                            "pred": np.asarray(_pred),
                            "obs_time": _d["time"].to_numpy(),
                            "obs_value": _d["value"].to_numpy(),
                            "obs_lo": _d["lo"].to_numpy() if _has_band else None,
                            "obs_hi": _d["hi"].to_numpy() if _has_band else None,
                            "is_raw": _stat_key == "raw",
                        })
                        _metrics.append({"group": _arm, "MOI": _moi, "n_points": len(_d),
                                         "RMSE": fit_residual(_r.time, _pred, _d["time"].values, _d["value"].values, _spec.get("log", False))})
                    _stat_label = _stat if _stat_key != "raw" else "raw points"
                    st.session_state["calib_overlay_result"] = {
                        "series": _series,
                        "metrics": _metrics,
                        "log": bool(_spec.get("log")),
                        "ylabel": _spec.get("label", _obs_key),
                        "stat_label": _stat_label,
                        "title": (f"Model (lines) vs observations ({_stat_label}"
                                  + (f" + {_band_choice} band)" if _band else ")")),
                    }
                except Exception as e:
                    st.session_state["calib_overlay_result"] = None
                    st.error(f"Overlay failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # Render the (persisted) overlay result if one exists.
            _ovr = st.session_state.get("calib_overlay_result")
            if _ovr:
                _fig, _ax = plt.subplots(figsize=(10, 5))
                _palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                for _i, _s in enumerate(_ovr["series"]):
                    _color = _palette[_i % len(_palette)]
                    _yp = np.maximum(_s["pred"], 1e-30) if _ovr["log"] else _s["pred"]
                    _ax.plot(_s["time"], _yp, color=_color, lw=2, label=f"{_s['label']} (model)")
                    if _s["is_raw"]:
                        _ax.scatter(_s["obs_time"], _s["obs_value"], color=_color, s=14, alpha=0.45)
                    else:
                        _ax.scatter(_s["obs_time"], _s["obs_value"], color=_color, s=18, marker="o",
                                    edgecolor="k", linewidth=0.3, zorder=3)
                        if _s["obs_lo"] is not None:
                            _ax.fill_between(_s["obs_time"], _s["obs_lo"], _s["obs_hi"], color=_color, alpha=0.18, linewidth=0)
                if _ovr["log"]:
                    _ax.set_yscale("log")
                _ax.set_xlabel("Time (h)")
                _ax.set_ylabel(_ovr["ylabel"])
                _ax.legend(fontsize=8, ncol=2)
                _ax.set_title(_ovr["title"])
                _ax.grid(True, alpha=0.15)
                st.pyplot(_fig)
                plt.close(_fig)
                st.markdown("#### Fit quality (RMSE" + (" on log₁₀" if _ovr["log"] else "") +
                            f", vs {_ovr['stat_label']})")
                st.dataframe(pd.DataFrame(_ovr["metrics"]), width="stretch", hide_index=True)
                st.caption("Edit the parameter values above and re-overlay to improve the fit. "
                           "Edits update the live model directly and can be saved as Parts in the Library.")

        # ── 6. Save the calibrated model ─────────────────────────────────────
        if _ds:
            st.markdown("### 6 · Save the calibrated model")
            st.caption("Parameter edits in the tuning panel are **already applied** to the live "
                       "Interactive-Simulator model. Save the whole calibrated configuration as a "
                       "Scenario to reload it later (from the Library page), or save individual "
                       "strains/phages as Parts in the Library.")
            _cs1, _cs2 = st.columns([3, 2])
            with _cs1:
                _cal_name = st.text_input("Scenario name", value="calibrated", key="fit_save_name")
            with _cs2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("💾 Save calibrated config as Scenario", key="fit_save_scenario", width="stretch"):
                    _nm = (_cal_name or "").strip()
                    if not _nm:
                        st.error("Enter a scenario name.")
                    else:
                        _scen = st.session_state.user_scenarios
                        _scen[_nm] = {
                            "annotation": "Saved from Calibration",
                            "schema_version": SCENARIO_SCHEMA_VERSION,
                            "state": dump_state_to_scenario(),
                        }
                        st.session_state.user_scenarios = _scen
                        st.success(f"Saved scenario '{_nm}'. Reload it from the Library page.")

        if st.button("🗑️ Clear dataset", key="fit_clear"):
            st.session_state.fit_dataset = None
            st.session_state.fit_config = {}
            st.session_state.calib_overlay_result = None
            st.rerun()

    # Save the current Calibration widget selections to the persistent config so they
    # survive navigation (see the re-seed block at the top of this page). The
    # parameter-tuning widgets (fit_edit_*) are excluded: they mirror the live
    # int_strains/int_phages dicts (authoritative + already persistent), so caching
    # and re-seeding them would let a stale copy override edits made elsewhere.
    for _wk in list(st.session_state.keys()):
        if (_wk.startswith("fit_") and _wk not in _FIT_NOPERSIST
                and not _wk.startswith("fit_edit_")):
            st.session_state.fit_config[_wk] = st.session_state[_wk]


# ── AI Simulation Assistant Page ──────────────────────────────────────────────
elif st.session_state.current_page == "AI Assistant":
    st.title("AI Simulation Assistant")
    st.caption("Instruct Claude to design, simulate, and analyze phage therapy setups using natural language.")

    # Check key
    if not st.session_state.agent.client.api_key:
        st.warning("⚠️ Please enter your Anthropic API Key in the sidebar to use the AI Assistant.")

    # Chat UI
    for turn in st.session_state.history:
        role, val = turn
        if role == "user":
            st.chat_message("user").markdown(val)
        else:
            exec_result, agent_resp = val
            with st.chat_message("assistant"):
                st.markdown(agent_resp.narrative)
                if getattr(agent_resp, "assumptions", "") and show_assumptions:
                    with st.expander("👁️ Model Assumptions"):
                        st.markdown(agent_resp.assumptions)
                if agent_resp.code and show_code:
                    with st.expander("🐍 Generated python code"):
                        st.code(agent_resp.code, language="python")

                # Show figures and outputs
                if exec_result:
                    if exec_result.success:
                        for fig in exec_result.figures:
                            st.pyplot(fig)
                            plt.close(fig)
                        if exec_result.stdout:
                            with st.expander("📄 Print outputs"):
                                st.text(exec_result.stdout)
                    else:
                        st.error("Execution failed:")
                        st.code(exec_result.error, language="text")

    # input
    if prompt := st.chat_input("Ask a question, or describe a simulation to run (e.g. 'what's a realistic adsorption rate?' or 'simulate 1 strain + 1 phage, burst 50')..."):
        st.chat_message("user").markdown(prompt)

        # Agentic generation: the model decides whether the request is a question (answer
        # directly) or a simulation (write + run + self-correct code via run_pbisim_code),
        # and only runs code when a simulation is actually asked for.
        run = None
        try:
            with st.spinner("Claude is thinking..."):
                run = st.session_state.agent.generate(
                    prompt, execute_code,
                    configure=apply_ai_configuration,
                    summarize=summarize_current_results,
                )
        except Exception as e:
            st.error(f"❌ AI Assistant Error: {e}")
            st.info("💡 If you are getting a 401 Authentication Error, please verify that your Anthropic API key is correct, active, and has remaining usage credits.")

        if run is not None:
            exec_result = run.result
            with st.chat_message("assistant"):
                st.markdown(run.narrative or "_(no explanation returned)_")
                if run.code and show_code:
                    _n = run.tool_calls
                    with st.expander(f"🐍 Generated python code · {_n} execution{'s' if _n != 1 else ''}"):
                        st.code(run.code, language="python")

                if exec_result is not None:
                    if exec_result.success:
                        for fig in exec_result.figures:
                            st.pyplot(fig)
                            plt.close(fig)
                        if exec_result.stdout:
                            with st.expander("📄 Print outputs"):
                                st.text(exec_result.stdout)
                    else:
                        st.error("The code still failed after self-correction:")
                        st.code(exec_result.error or "", language="text")

                # The assistant populated the Interactive Simulator's widgets — offer to open it.
                if getattr(run, "configured", False):
                    st.success("✅ I've set up the **Interactive Simulator** with this configuration — open it to review, tweak, and run.")
                    if st.button("▶ Open in Interactive Simulator", key=f"nav_sim_{len(st.session_state.history)}", width="stretch"):
                        st.session_state["_nav_to"] = "Interactive Simulator"
                        st.rerun()

            # Release every figure the tool loop created (the model may have run code several
            # times; only the displayed set was closed above). Unclosed matplotlib figures
            # accumulate in the global registry and are a real memory leak on the small
            # Render container — a prime cause of the app being OOM-killed mid-chat.
            plt.close("all")

            st.session_state.history.append(("assistant", (exec_result, run)))
            # Keep a generous on-page chat log; only drop the oldest turns in very long
            # sessions (bounds stored figures/results without truncating normal use).
            st.session_state.history = st.session_state.history[-40:]



# ── Clinical Trials & Cohorts Page ────────────────────────────────────────────
elif st.session_state.current_page == "Clinical Trials & Cohorts":
    st.title("Clinical Trials & Cohort Simulator")
    st.caption("Generate a virtual population (VPOP), apply statistical variability (IIV), and run matching parallel arms.")
    
    st.markdown(
        "<div class='info-banner'>🧬 Virtual Cohort simulations use the current biological model configured "
        "in the <b>Interactive Simulator</b> tab as the baseline 'nominal patient'. Change parameters there first.</div>",
        unsafe_allow_html=True,
    )
    
    t_cols = st.columns([1, 2])
    
    with t_cols[0]:
        st.markdown("### 📊 Trial Settings")
        trial_patients = st.number_input("Cohort Size (N)", min_value=10, max_value=500, value=50, step=10)
        trial_seed = st.number_input("Cohort RNG Seed", value=42)
        trial_t_end = st.number_input("Trial Duration (hours)", min_value=12.0, max_value=336.0, value=72.0, step=12.0)
        trial_dt = st.number_input("Solver output step (dt)", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
        trial_n_jobs = st.slider("Parallel workers (n_jobs)", min_value=1, max_value=16, value=1, help="Parallel patient simulation via joblib (loky). Keep at 1 on small/shared hosts (e.g. the free Render tier) — forked worker processes can segfault or OOM there; raise it on a beefier machine.")
        
        st.markdown("### 💉 Treatment Arms")
        st.caption("Define any number of arms (e.g. low-dose vs high-dose), each with its own phage / antibiotic regimen. The Control arm never receives doses.")
        trial_include_control = st.checkbox("Include Control arm (no doses)", value=True)

        _tphages = st.session_state.get("int_phages", [])
        _tabx = st.session_state.get("int_antibiotics", [])
        trial_arms = st.session_state.get("trial_arms", [])

        if not _tphages and not _tabx:
            st.info("Configure at least one phage or antibiotic in the Interactive Simulator to define dosed arms.")

        # Existing arms — editable in place
        for _ai, _arm in enumerate(list(trial_arms)):
            _arm.setdefault("_id", _next_uid("arm"))   # stable key across reorder/delete
            _aid = _arm["_id"]
            _lc, _dc = st.columns([6, 1])
            with _lc:
                st.markdown(f"**{_arm['name']}** — {arm_regimen_summary(_arm)}")
            with _dc:
                if st.button("🗑️", key=f"del_arm_{_aid}"):
                    trial_arms.pop(_ai)
                    st.session_state.trial_arms = trial_arms
                    st.rerun()
            with st.expander(f"✏️ Edit '{_arm['name']}'", expanded=False):
                _en = st.text_input("Arm name", value=_arm["name"], key=f"edit_arm_name_{_aid}")
                _ep, _ea = {"on": False}, {"on": False}
                if _tphages:
                    st.markdown("**Phage dosing**")
                    _ep = render_regimen_config(f"edit_arm_p_{_aid}", _tphages, "phage",
                                                1e9, "Amount (PFU)", initial=_arm.get("phage"))
                if _tabx:
                    st.markdown("**Antibiotic dosing**")
                    _ea = render_regimen_config(f"edit_arm_a_{_aid}", _tabx, "antibiotic",
                                                10.0, "Amount (mg)", initial=_arm.get("abx"))
                if st.button("💾 Save changes", key=f"save_arm_{_aid}"):
                    _arm["name"], _arm["phage"], _arm["abx"] = _en, _ep, _ea
                    st.session_state.trial_arms = trial_arms
                    st.rerun()

        # Add-arm form
        with st.expander("➕ Add treatment arm", expanded=not trial_arms):
            _new_name = st.text_input("Arm name", value=f"Arm {len(trial_arms) + 1}", key="new_arm_name")
            _pcfg, _acfg = {"on": False}, {"on": False}
            if _tphages:
                st.markdown("**Phage dosing**")
                _pcfg = render_regimen_config("new_arm_p", _tphages, "phage", 1e9, "Amount (PFU)", default_on=True)
            if _tabx:
                st.markdown("**Antibiotic dosing**")
                _acfg = render_regimen_config("new_arm_a", _tabx, "antibiotic", 10.0, "Amount (mg)", default_on=not _tphages)
            if st.button("➕ Add arm", key="add_arm_btn"):
                trial_arms.append({"name": _new_name, "phage": _pcfg, "abx": _acfg})
                st.session_state.trial_arms = trial_arms
                st.rerun()

        st.markdown("### 🧬 Parameter Variability (IIV)")
        
        # Active IIVs — editable in place
        trial_iivs = st.session_state.get("trial_iiv_inputs", [])

        for idx, iiv in enumerate(trial_iivs):
            iiv.setdefault("_id", _next_uid("iiv"))   # stable key across reorder/delete
            _iid = iiv["_id"]
            _pname = next((n for n, p in IIV_PARAMETERS.items() if p == iiv["path"]), iiv["path"])
            col_p, col_act = st.columns([6, 1])
            with col_p:
                st.markdown(f"**{_pname}** — {iiv['dist_type']} {iiv['params']} ({iiv['mode']})")
            with col_act:
                if st.button("🗑️", key=f"del_iiv_{_iid}"):
                    trial_iivs.pop(idx)
                    st.session_state.trial_iiv_inputs = trial_iivs
                    st.rerun()
            with st.expander("✏️ Edit", expanded=False):
                _edited = render_iiv_config(f"edit_iiv_{_iid}", initial=iiv)
                if st.button("💾 Save changes", key=f"save_iiv_{_iid}"):
                    _edited["_id"] = _iid
                    trial_iivs[idx] = _edited
                    st.session_state.trial_iiv_inputs = trial_iivs
                    st.rerun()

        # Add IIV form
        with st.expander("➕ Add Parameter Variability"):
            _new_iiv = render_iiv_config("new_iiv")
            if st.button("Add Parameter IIV"):
                trial_iivs.append(_new_iiv)
                st.session_state.trial_iiv_inputs = trial_iivs
                st.success("Added parameter variability.")
                st.rerun()

        # Run Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Run Parallel Clinical Trial", width="stretch"):
            with st.spinner("Generating cohort populations & simulating treatment arms..."):
                try:
                    # 1. Compile nominal base config
                    base_cfg, init_B, init_P, init_S, model_kwargs = build_nominal_config_from_gui()

                    # Phage is the *intervention*, delivered per-arm via each arm's
                    # dose schedule (Treatment Arms builder). In this crossover design
                    # all arms share initial_conditions and differ only by dose_schedule,
                    # so every arm starts with zero free phage and receives only its own
                    # configured doses — keeping the Control arm a true no-treatment arm.
                    init_P = np.asarray(init_P, dtype=float)
                    base_P = np.zeros_like(init_P)

                    from pbisim.trial.population import InitialConditions
                    base_cfg.initial_conditions = InitialConditions(
                        B=init_B,
                        P=base_P,
                        S=init_S,
                        D=model_kwargs.get("initial_D", None),
                        Imm=model_kwargs.get("initial_Imm", None),
                    )

                    # 2. Assemble arms from the Treatment Arms builder
                    arms = []
                    _used_names = set()

                    def _add_arm(name, doses):
                        # ClinicalTrial requires unique arm names
                        base = name.strip() or "Arm"
                        uniq, k = base, 2
                        while uniq in _used_names:
                            uniq = f"{base} ({k})"; k += 1
                        _used_names.add(uniq)
                        arms.append(TreatmentArm(name=uniq, dose_schedule=DoseSchedule(list(doses))))

                    if trial_include_control:
                        _add_arm("Control", [])

                    for _arm in trial_arms:
                        _doses = arm_dose_events(_arm)
                        if not _doses:
                            st.warning(f"Arm '{_arm['name']}' has no doses — it will behave like the Control arm.")
                        _add_arm(_arm["name"], _doses)

                    if not arms:
                        st.error("Add at least one treatment arm (or enable the Control arm) to run.")
                    else:
                        pretreatment_hours = st.session_state.get("int_t_prerun", 0.0)
                        
                        trial_result = run_trial_simulation(
                            base_cfg,
                            trial_iivs,
                            arms,
                            n_patients=int(trial_patients),
                            t_end=trial_t_end,
                            dt=trial_dt,
                            seed=int(trial_seed),
                            pretreatment_hours=pretreatment_hours,
                            n_jobs=int(trial_n_jobs),
                            base_initial_B=init_B,
                            base_initial_P=base_P,
                            base_initial_S=init_S,
                            inherit_debris=st.session_state.get("int_prerun_inherit_debris", True),
                            **model_kwargs
                        )
                        st.session_state.trial_result = trial_result
                        st.success("Clinical Trial cohort simulation completed successfully!")
                except Exception as e:
                    st.error(f"Trial Execution Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    
    with t_cols[1]:
        st.markdown("### 📊 Outcomes & Visualization")
        
        if st.session_state.trial_result is None:
            st.info("Run the clinical trial simulation on the left panel to display outcomes.")
        else:
            result = st.session_state.trial_result
            
            # Outcome selection
            _endpoint_labels = {
                "tte": "Time-to-Eradication (TTE)",
                "tt2lr": "Time-to-2-Log-Reduction (TT2LR)",
            }
            _metric_labels = {
                "max_log_reduction": "Maximum Log Reduction",
                "log_reduction_final": "Log Reduction (baseline → last obs)",
                "bacterial_auc": "Bacterial AUC",
                "nadir_count": "Nadir Count",
            }
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                endpoint_choice = st.selectbox(
                    "Survival Endpoint (time-to-event)", ["tte", "tt2lr"], index=0,
                    format_func=lambda x: _endpoint_labels[x],
                )
            with c_v2:
                metric_choice = st.selectbox(
                    "Distribution Metric", list(_metric_labels), index=0,
                    format_func=lambda x: _metric_labels[x],
                )
                
            clearance_threshold = st.session_state.get("int_extinction_threshold", 100.0)
            
            # Raw PKPD time trajectories (CFU + PFU) per arm
            st.markdown("#### 🧫 PK/PD trajectories (median & IQR per arm)")
            fig_cfu = plot_pkpd_trajectories_plotly(
                result, prefixes=("B", "D", "I", "H"),
                title="Total Bacteria (CFU/mL)", y_label="log₁₀ CFU/mL",
            )
            st.plotly_chart(fig_cfu, width="stretch")
            fig_pfu = plot_pkpd_trajectories_plotly(
                result, prefixes=("P",),
                title="Free Phage (PFU/mL)", y_label="log₁₀ PFU/mL",
            )
            st.plotly_chart(fig_pfu, width="stretch")

            # Step survival plot
            st.markdown("#### ⏳ Step-Survival (Kaplan-Meier)")
            fig_km = plot_kaplan_meier_plotly(result, endpoint=endpoint_choice, t_end=trial_t_end, threshold=clearance_threshold, n_logs=2.0)
            st.plotly_chart(fig_km, width="stretch")

            # Metric distributions
            st.markdown("#### 📦 Distribution of outcomes")
            fig_dist = plot_metric_distributions_plotly(result, metric=metric_choice)
            st.plotly_chart(fig_dist, width="stretch")
            
            # Data Exports
            st.markdown("---")
            st.markdown("### 📥 Cohort Data Exports")
            
            cx1, cx2 = st.columns(2)
            with cx1:
                # Outcome Dataframe
                out_df = result.outcome_dataframe(endpoint=endpoint_choice, t_end=trial_t_end, threshold=clearance_threshold)
                csv_out = out_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Survival Outcomes DataFrame (CSV)",
                    data=csv_out,
                    file_name="pbisim_survival_outcomes.csv",
                    mime="text/csv",
                    width="stretch"
                )
            with cx2:
                # NLME Dataframe for pharmacometric models
                outputs_spec = {"DV_B": ("B", "D", "I", "H")}
                obs_times = np.linspace(0, trial_t_end, 10)
                nlme_df = result.nlme_dataframe(outputs_spec, times=obs_times)
                csv_nlme = nlme_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Pharmacometrics (NLME) DataFrame (CSV)",
                    data=csv_nlme,
                    file_name="pbisim_nlme_cohort.csv",
                    mime="text/csv",
                    width="stretch"
                )


# ── Dose-Response Sweeps Page ──────────────────────────────────────────────────
elif st.session_state.current_page == "Dose-Response Sweeps":
    st.title("Dose-Response Simulator")
    st.caption("Perform multi-drug dose-response sweeps with MOI scaling, vector padding, and raw time-series visualization.")

    # Keep the sweep controls alive across navigation (re-seed before they render).
    reseed_widget_config("dr_sweep_config", ("dr_sweep_",))

    strains = st.session_state.get("int_strains", [])
    phages = st.session_state.get("int_phages", [])
    antibiotics = st.session_state.get("int_antibiotics", [])
    theme_mode = st.session_state.get("theme_mode", "Light")

    st.markdown(
        "<div class='info-banner'>💊 Sweeps are based on the biological model currently configured in the "
        "<b>Interactive Simulator</b>. Change biological parameters (e.g. growth rates, adsorption) there first.</div>",
        unsafe_allow_html=True,
    )

    # 1. Sweep configurations
    swept_inputs = {}
    swept_units = {}
    swept_repeat_configs = {}

    col_setup, col_run = st.columns([1, 2])

    with col_setup:
        st.markdown("### ⚙️ Configure Sweeps")
        
        # Phages
        for j, p in enumerate(phages):
            st.markdown(f"#### Phage {j}: {p['name']}")
            do_sweep = st.checkbox(f"Sweep Phage {j}", key=f"dr_sweep_phg_en_{j}", value=False)
            if do_sweep:
                series_str = st.text_input(
                    "Dose series (comma-separated)",
                    value="0, 1e3, 1e5, 1e7, 1e9",
                    key=f"dr_sweep_phg_series_{j}"
                )
                unit = st.selectbox(
                    "Dose Unit",
                    ["PFU (absolute)", "MOI (relative to B(0))"],
                    key=f"dr_sweep_phg_unit_{j}"
                )
                swept_inputs[f"phage_{j}"] = series_str
                swept_units[f"phage_{j}"] = unit
                
                rep_en = st.checkbox("Configure as custom repeat dosing regimen", key=f"dr_sweep_phg_rep_en_{j}", value=False)
                if rep_en:
                    r_interval = st.number_input("Interval (hours)", min_value=1.0, value=12.0, key=f"dr_sweep_phg_rep_int_{j}")
                    r_count = st.number_input("Number of repeats", min_value=1, value=4, key=f"dr_sweep_phg_rep_count_{j}")
                    r_start = st.number_input("Start time (hours)", min_value=0.0, value=0.0, key=f"dr_sweep_phg_rep_start_{j}")
                    r_route = st.selectbox("Route", ["bolus", "infusion"], key=f"dr_sweep_phg_rep_route_{j}")
                    r_dur = 0.0
                    if r_route == "infusion":
                        r_dur = st.number_input("Duration (hours)", min_value=0.1, value=2.0, key=f"dr_sweep_phg_rep_dur_{j}")
                    swept_repeat_configs[f"phage_{j}"] = {
                        "interval": r_interval,
                        "count": int(r_count),
                        "start": r_start,
                        "route": r_route,
                        "duration": r_dur
                    }

        # Antibiotics
        for j, a in enumerate(antibiotics):
            st.markdown(f"#### Antibiotic {j}: {a['name']}")
            do_sweep = st.checkbox(f"Sweep Antibiotic {j}", key=f"dr_sweep_abx_en_{j}", value=False)
            if do_sweep:
                series_str = st.text_input(
                    "Dose series (comma-separated)",
                    value="0.5, 1.0, 2.0",
                    key=f"dr_sweep_abx_series_{j}"
                )
                swept_inputs[f"abx_{j}"] = series_str
                swept_units[f"abx_{j}"] = "absolute"
                
                rep_en = st.checkbox("Configure as custom repeat dosing regimen", key=f"dr_sweep_abx_rep_en_{j}", value=False)
                if rep_en:
                    r_interval = st.number_input("Interval (hours)", min_value=1.0, value=12.0, key=f"dr_sweep_abx_rep_int_{j}")
                    r_count = st.number_input("Number of repeats", min_value=1, value=4, key=f"dr_sweep_abx_rep_count_{j}")
                    r_start = st.number_input("Start time (hours)", min_value=0.0, value=0.0, key=f"dr_sweep_abx_rep_start_{j}")
                    r_route = st.selectbox("Route", ["bolus", "infusion"], key=f"dr_sweep_abx_rep_route_{j}")
                    r_dur = 0.0
                    if r_route == "infusion":
                        r_dur = st.number_input("Duration (hours)", min_value=0.1, value=2.0, key=f"dr_sweep_abx_rep_dur_{j}")
                    swept_repeat_configs[f"abx_{j}"] = {
                        "interval": r_interval,
                        "count": int(r_count),
                        "start": r_start,
                        "route": r_route,
                        "duration": r_dur
                    }

        st.markdown("<br>", unsafe_allow_html=True)
        run_sweep = st.button("🚀 Run Dose-Response Sweep", width="stretch")

    with col_run:
        st.markdown("### 📊 Sweep Results")
        if run_sweep:
            # Parse vectors
            parsed_vectors = {}
            for k, val_str in swept_inputs.items():
                try:
                    vals = parse_comma_separated_series(val_str)
                    if not vals:
                        st.error(f"Sweep vector for '{k}' cannot be empty.")
                        st.stop()
                    parsed_vectors[k] = vals
                except ValueError:
                    st.error(f"Invalid comma-separated values for '{k}'. Please check formatting.")
                    st.stop()

            if not parsed_vectors:
                st.error("Please select at least one drug/phage to sweep.")
            else:
                # Perform padding
                padded, warnings = pad_vectors(parsed_vectors)
                for w in warnings:
                    st.warning(f"⚠️ {w}")

                # Determine number of runs M
                first_key = list(padded.keys())[0]
                M = len(padded[first_key])
                
                # Baseline initial_B
                sum_initial_B = sum(s["initial_B"] for s in strains)

                # Warn once if the pre-run decimates the culture (death w/o dormancy).
                _pc_cfg, _pc_B0, *_ = build_nominal_config_from_gui()
                warn_if_prerun_collapses(_pc_cfg, _pc_B0)

                # Save original doses
                original_doses = list(st.session_state.get("int_doses", []))

                # When a phage's DOSE is swept, its baseline initial_P inoculum must not
                # leak in: otherwise a swept value of 0 still starts with the configured
                # P0 (default 1e6) and suppresses the bacteria. Zero the swept phages'
                # initial_P for the duration of the sweep — the dose delivers the phage —
                # and restore it afterwards. (Antibiotics have no such baseline: they are
                # delivered purely via dose events.)
                swept_phage_idx = [int(k.split("_")[1]) for k in padded if k.startswith("phage_")]
                original_initial_P = {j: phages[j]["initial_P"] for j in swept_phage_idx if j < len(phages)}
                for j in original_initial_P:
                    phages[j]["initial_P"] = 0.0

                runs_outcomes = []
                trajectories = [] # list of (time, viable_b, label)
                od_trajectories = [] # list of (time, od, label) — only if OD/debris enabled
                _od_enabled = st.session_state.get("int_debris_enabled", False)

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    for k_idx in range(M):
                        status_text.text(f"Running simulation {k_idx+1} of {M}...")
                        
                        # Generate custom doses for this run
                        custom_doses = []
                        
                        # Add non-swept nominal doses
                        for nd in original_doses:
                            is_swept = False
                            if nd["target_type"] == "phage" and f"phage_{nd['target_idx']}" in padded:
                                is_swept = True
                            elif nd["target_type"] == "antibiotic" and f"abx_{nd['target_idx']}" in padded:
                                is_swept = True
                            
                            if not is_swept:
                                custom_doses.append(nd)

                        # Add swept doses
                        swept_label_parts = []
                        for key, vec in padded.items():
                            val = vec[k_idx]
                            
                            # Scale MOI
                            if swept_units.get(key) == "MOI (relative to B(0))":
                                val_absolute = val * sum_initial_B
                                swept_label_parts.append(f"{key}: {val} MOI ({val_absolute:.1e})")
                                val = val_absolute
                            else:
                                swept_label_parts.append(f"{key}: {val:.1e}")

                            # Check if repeat regimen or nominal dose overrides
                            if key in swept_repeat_configs:
                                rcfg = swept_repeat_configs[key]
                                target_type = "phage" if key.startswith("phage") else "antibiotic"
                                target_idx = int(key.split("_")[1])
                                for r in range(rcfg["count"]):
                                    custom_doses.append({
                                        "time": rcfg["start"] + r * rcfg["interval"],
                                        "amount": val,
                                        "target_type": target_type,
                                        "target_idx": target_idx,
                                        "route": rcfg["route"],
                                        "duration": rcfg["duration"]
                                    })
                            else:
                                # Overwrite nominal doses
                                target_type = "phage" if key.startswith("phage") else "antibiotic"
                                target_idx = int(key.split("_")[1])
                                target_nominal_doses = [d for d in original_doses if d["target_type"] == target_type and d["target_idx"] == target_idx]
                                if target_nominal_doses:
                                    for nd in target_nominal_doses:
                                        nd_copy = dict(nd)
                                        nd_copy["amount"] = val
                                        custom_doses.append(nd_copy)
                                else:
                                    # Create single dose event at t=0
                                    custom_doses.append({
                                        "time": 0.0,
                                        "amount": val,
                                        "target_type": target_type,
                                        "target_idx": target_idx,
                                        "route": "bolus",
                                        "duration": 0.0
                                    })

                        st.session_state.int_doses = custom_doses
                        
                        # Run
                        result, config = run_sim_from_gui_params()
                        
                        # Compute metrics
                        total_bacteria = result.sum_prefixes("B", "D", "I", "H")
                        nadir_val = np.min(total_bacteria)

                        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                        auc_val = _trapz(total_bacteria, result.time)

                        t_clear = time_to_clearance(
                            result,
                            threshold=st.session_state.get("int_extinction_threshold", 1.0),
                        )
                        t_log_red = time_to_log_reduction(result, n_logs=2.0)
                        
                        run_label = ", ".join(swept_label_parts)
                        runs_outcomes.append({
                            "Run Index": k_idx + 1,
                            "Swept Doses": run_label,
                            "Nadir (cells/mL)": nadir_val,
                            "AUC (cells·h/mL)": auc_val,
                            "Clearance Time (h)": t_clear if t_clear is not None else np.nan,
                            "2-Log Red Time (h)": t_log_red if t_log_red is not None else np.nan
                        })
                        
                        trajectories.append((result.time, total_bacteria, f"Run {k_idx + 1}: {run_label}"))
                        if _od_enabled:
                            _od = (_safe_od(result, total_bacteria))
                            od_trajectories.append((result.time, _od, f"Run {k_idx + 1}: {run_label}"))
                        progress_bar.progress((k_idx + 1) / M)
                        
                    status_text.text("Sweep completed!")
                finally:
                    # Restore original doses + swept phages' baseline inoculum
                    st.session_state.int_doses = original_doses
                    for j, v in original_initial_P.items():
                        phages[j]["initial_P"] = v

                # Persist results so they survive navigation (rendered below, until
                # the sweep is re-run).
                st.session_state.dr_sweep_result = {
                    "summary": runs_outcomes,
                    "trajectories": [(np.asarray(t), np.asarray(b), lbl) for t, b, lbl in trajectories],
                    "od_trajectories": [(np.asarray(t), np.asarray(o), lbl) for t, o, lbl in od_trajectories],
                }

        # Render the (persisted) sweep result if one exists.
        _dr = st.session_state.get("dr_sweep_result")
        if _dr:
            import plotly.graph_objects as go
            df_summary = pd.DataFrame(_dr["summary"])
            st.markdown("#### Summary of Runs")
            st.dataframe(
                df_summary.style.format({
                    "Nadir (cells/mL)": "{:.2e}",
                    "AUC (cells·h/mL)": "{:.2e}",
                    "Clearance Time (h)": "{:.1f}",
                    "2-Log Red Time (h)": "{:.1f}"
                }),
                width="stretch"
            )

            st.markdown("#### Raw Simulation Trajectories (Viable Bacteria)")
            fig_traj = go.Figure()
            for t_arr, b_arr, legend_lbl in _dr["trajectories"]:
                fig_traj.add_trace(go.Scatter(x=t_arr, y=np.maximum(b_arr, 1.0), mode='lines', name=legend_lbl))
            fig_traj.update_layout(
                xaxis_title="Time (hours)", yaxis_title="Total Viable Bacteria (CFU/mL)",
                yaxis_type="log", template="plotly_white" if theme_mode == "Light" else "plotly_dark")
            st.plotly_chart(fig_traj, width="stretch")

            if _dr["od_trajectories"]:
                st.markdown("#### Raw Simulation Trajectories (Optical Density)")
                fig_od = go.Figure()
                for t_arr, od_arr, legend_lbl in _dr["od_trajectories"]:
                    fig_od.add_trace(go.Scatter(x=t_arr, y=od_arr, mode='lines', name=legend_lbl))
                fig_od.update_layout(
                    xaxis_title="Time (hours)", yaxis_title="Optical density (AU)",
                    template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                st.plotly_chart(fig_od, width="stretch")

            st.markdown("#### Outcome Metrics vs Run Index")
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Scatter(x=df_summary["Run Index"], y=df_summary["AUC (cells·h/mL)"], mode="lines+markers", name="Bacterial AUC", yaxis="y1"))
            fig_metrics.add_trace(go.Scatter(x=df_summary["Run Index"], y=df_summary["Nadir (cells/mL)"], mode="lines+markers", name="Nadir", yaxis="y2"))
            fig_metrics.update_layout(
                title="Bacterial Efficacy Metrics Across Sweep Runs",
                xaxis=dict(title="Run Index"),
                yaxis=dict(title="AUC (cells·h/mL)", type="log"),
                yaxis2=dict(title="Nadir (cells/mL)", type="log", overlaying="y", side="right"),
                template="plotly_white" if theme_mode == "Light" else "plotly_dark")
            st.plotly_chart(fig_metrics, width="stretch")
        else:
            st.info("Configure the sweep on the left and click **Run Dose-Response Sweep** to view results.")

    with st.expander("🐍 View Python Reproduction Code"):
        st.caption("Standalone script that reproduces this sweep — the recorded base "
                   "model plus a loop rebuilding the per-run dose schedule exactly as the app does.")
        try:
            st.code(generate_dose_sweep_reproduction_code(), language="python")
        except Exception as _e:
            st.warning(f"Reproduction code unavailable: {_e}")

    # Persist the sweep controls so they survive navigation (see reseed above).
    save_widget_config("dr_sweep_config", ("dr_sweep_",))

# ── Parameter Sweeps Page ──────────────────────────────────────────────────────
elif st.session_state.current_page == "Parameter Sweeps":
    st.title("Model Parameter Sweeps")
    st.caption("Sweep any model parameter in 1D or 2D and visualize cellular trajectories and outcome heatmaps.")

    # Keep the sweep controls alive across navigation (re-seed before they render).
    reseed_widget_config("param_sweep_config", ("p1_", "p2_", "ps_", "pc_"))

    # Build nominal configuration
    try:
        nominal_config, initial_B, initial_P, initial_S, model_kwargs = build_nominal_config_from_gui()
    except Exception as e:
        st.error(f"Nominal configuration build failed. Please configure it in the Interactive Simulator first. Error: {e}")
        st.stop()

    strains_gui = st.session_state.get("int_strains", [])
    phages_gui = st.session_state.get("int_phages", [])
    antibiotics_gui = st.session_state.get("int_antibiotics", [])
    theme_mode = st.session_state.get("theme_mode", "Light")

    st.markdown(
        "<div class='info-banner'>⚙️ Sweeps are based on the biological model currently configured in the "
        "<b>Interactive Simulator</b>. Toggle 1D vs 2D sweep type, select parameters, and run.</div>",
        unsafe_allow_html=True,
    )

    sweep_type = st.radio("Sweep Dimension", ["1D Sweep", "2D Sweep", "Coupled (linked)"], horizontal=True, key="ps_sweep_type")

    sweep_params = get_sweep_parameters(nominal_config, strains_gui, phages_gui, antibiotics_gui)
    param_labels = sorted(list(sweep_params.keys()))

    col_setup, col_run = st.columns([1, 2])

    with col_setup:
        st.markdown("### ⚙️ Configure Parameters")
        
        if sweep_type == "1D Sweep":
            param1_label = st.selectbox("Select Parameter", param_labels, key="p1_sweep_label")
            meta1 = sweep_params[param1_label]
            # Re-autoscale the range widgets when the swept parameter changes (their
            # value= default depends on the nominal value; a persisted key would pin it).
            if st.session_state.get("_ps_1d_last_param") != param1_label:
                for _k in ("ps_1d_min", "ps_1d_max", "ps_1d_steps", "ps_1d_spacing"):
                    st.session_state.pop(_k, None)
                    st.session_state.setdefault("param_sweep_config", {}).pop(_k, None)
                st.session_state["_ps_1d_last_param"] = param1_label

            # Default values
            default_val = 1e-9
            if meta1["type"] == "scalar":
                default_val = getattr(nominal_config, meta1["field"])
                if default_val is None:  # e.g. dormancy_monod_constant when inheriting
                    default_val = meta1.get("default", 1.0)
            elif meta1["type"] == "dimension":
                default_val = getattr(nominal_config, meta1["field"])
            elif meta1["type"] == "array1d":
                default_val = getattr(nominal_config, meta1["field"])[meta1["index"]]
            elif meta1["type"] == "array1d_or_none":
                arr = getattr(nominal_config, meta1["field"])
                default_val = arr[meta1["index"]] if arr is not None else 0.0
            elif meta1["type"] in ("array1d_broadcast", "array1d_broadcast_or_none"):
                arr = getattr(nominal_config, meta1["field"])
                default_val = float(arr[0]) if arr is not None and len(arr) else 0.0
            elif meta1["type"] == "initial_B_broadcast":
                default_val = float(initial_B[0]) if len(initial_B) else 1e7
            elif meta1["type"] == "array2d":
                default_val = getattr(nominal_config, meta1["field"])[meta1["index_row"], meta1["index_col"]]
            elif meta1["type"] == "pk_array1d":
                pk_config = nominal_config.phage_pk_config or nominal_config.pk_config
                default_val = getattr(pk_config, meta1["field"])[meta1["index"]]
            elif meta1["type"] == "pd_array2d":
                default_val = getattr(nominal_config.pd_config, meta1["field"])[meta1["index_row"], meta1["index_col"]]
            elif meta1["type"] == "initial_B":
                default_val = initial_B[meta1["index"]]
            elif meta1["type"] == "initial_P":
                default_val = initial_P[meta1["index"]]
            elif meta1["type"] == "initial_S":
                default_val = initial_S

            st.caption(f"Nominal Value: `{default_val:.2e}`" if isinstance(default_val, (int, float)) else f"Nominal Value: `{default_val}`")
            
            _is_dim = meta1["type"] == "dimension"
            c1, c2, c3 = st.columns(3)
            if _is_dim:
                # integer compartment count: integer bounds ≥ 1
                _dv = max(1, int(round(default_val)))
                with c1:
                    min_val = st.number_input("Min Value", min_value=1, value=1, step=1, key="ps_1d_min")
                with c2:
                    max_val = st.number_input("Max Value", min_value=1, value=max(_dv + 3, 5), step=1, key="ps_1d_max")
                with c3:
                    steps = st.number_input("Steps", min_value=2, max_value=25, value=5, key="ps_1d_steps")
            else:
                with c1:
                    min_val = st.number_input("Min Value", value=float(default_val * 0.1) if default_val > 0 else 0.0, format="%.2e", key="ps_1d_min")
                with c2:
                    max_val = st.number_input("Max Value", value=float(default_val * 10.0) if default_val > 0 else 1.0, format="%.2e", key="ps_1d_max")
                with c3:
                    steps = st.number_input("Steps", min_value=2, max_value=25, value=5, key="ps_1d_steps")

            spacing = st.selectbox("Spacing", ["Linear", "Logarithmic"], key="ps_1d_spacing")
            run_sweep = st.button("🚀 Run 1D Sweep", width="stretch")

        elif sweep_type == "2D Sweep":
            param1_label = st.selectbox("Select Parameter 1 (X-axis)", param_labels, key="p1_sweep_label")
            meta1 = sweep_params[param1_label]
            
            c1, c2, c3 = st.columns(3)
            with c1:
                min_val = st.number_input("P1 Min Value", value=1e-10, format="%.2e", key="p1_min")
            with c2:
                max_val = st.number_input("P1 Max Value", value=1e-6, format="%.2e", key="p1_max")
            with c3:
                steps = st.number_input("P1 Steps", min_value=2, max_value=10, value=3, key="p1_steps")
            spacing = st.selectbox("P1 Spacing", ["Linear", "Logarithmic"], key="p1_spacing")
            
            st.markdown("---")
            param2_label = st.selectbox("Select Parameter 2 (Y-axis)", param_labels, key="p2_sweep_label")
            meta2 = sweep_params[param2_label]
            
            c4, c5, c6 = st.columns(3)
            with c4:
                min_val2 = st.number_input("P2 Min Value", value=10.0, format="%.2e", key="p2_min")
            with c5:
                max_val2 = st.number_input("P2 Max Value", value=200.0, format="%.2e", key="p2_max")
            with c6:
                steps2 = st.number_input("P2 Steps", min_value=2, max_value=10, value=3, key="p2_steps")
            spacing2 = st.selectbox("P2 Spacing", ["Linear", "Logarithmic"], key="p2_spacing")

            run_sweep = st.button("🚀 Run 2D Sweep", width="stretch")

        else:  # Coupled (linked) sweep
            st.caption(
                "Sweep several parameters **together**: pick the parameters and give each a "
                "value series of the **same length**. At step *k*, value[k] of every parameter "
                "is applied at once. Use the **(ALL strains)** parameters to share one value "
                "across strains. Example — (dormancy rate, resuscitation rate) = "
                "(0, 1), (0.5, 0.5), (1, 0)."
            )
            coupled_labels = st.multiselect("Parameters to sweep together", param_labels, key="pc_labels")
            for _ci, _lbl in enumerate(coupled_labels):
                _meta = sweep_params[_lbl]
                st.text_input(
                    f"Values for '{_lbl}' (comma-separated)",
                    key=f"pc_series_{_ci}",
                    placeholder="e.g. 0, 0.5, 1",
                )
            run_sweep = st.button("🚀 Run Coupled Sweep", width="stretch")

    with col_run:
        st.markdown("### 📊 Sweep Results")
        if run_sweep:
            # Warn once if the pre-run decimates the culture (death w/o dormancy).
            warn_if_prerun_collapses(nominal_config, initial_B)
            progress_bar = st.progress(0)
            status_text = st.empty()

            if sweep_type == "1D Sweep":
                # Compute sweep values
                if spacing == "Logarithmic":
                    if min_val <= 0 or max_val <= 0:
                        st.error("Logarithmic spacing requires positive bounds.")
                        st.stop()
                    sweep_values = np.logspace(np.log10(min_val), np.log10(max_val), int(steps))
                else:
                    sweep_values = np.linspace(min_val, max_val, int(steps))
                # Compartment counts (n_depth / n_latent) are integers >= 1 — sweep unique
                # integers so the results aren't fractional/duplicated.
                if meta1["type"] == "dimension":
                    sweep_values = np.unique(np.clip(np.round(sweep_values).astype(int), 1, None))
                    st.caption("Integer parameter — swept over unique integer values ≥ 1: "
                               f"{list(sweep_values)}")

                runs_outcomes = []
                trajectories = [] # (time, viable_b, label)
                od_trajectories = [] # (time, od, label) — only if OD/debris enabled
                _od_enabled = st.session_state.get("int_debris_enabled", False)

                for idx, val in enumerate(sweep_values):
                    status_text.text(f"Running simulation {idx+1} of {len(sweep_values)} (Value: {val:.2e})...")
                    
                    # Apply parameter
                    c_k, ib_k, ip_k, is_k, mk_k = apply_sweep_parameter(
                        val, meta1, nominal_config, initial_B, initial_P, initial_S, model_kwargs
                    )

                    # equilibrate pre-treatment prerun — carry the full stationary
                    # state (B, D, S, Imm), not just B/S (see run_sim_from_gui_params).
                    t_prerun = st.session_state.get("int_t_prerun", 0.0)
                    if t_prerun > 0:
                        ic = stationary_phase_ic(c_k, t_prerun=t_prerun, B0=ib_k)
                        ib_k = ic.B
                        is_k = max(float(ic.S), 0.0)
                        if ic.D is not None:
                            mk_k["initial_D"] = ic.D
                        if ic.Imm is not None:
                            mk_k["initial_Imm"] = ic.Imm
                        _carry_prerun_debris(ic, mk_k)

                    model = PBIModel(c_k, initial_B=ib_k, initial_P=ip_k, initial_S=is_k, **mk_k)
                    result = solve_ode(model, t_end=st.session_state.get("int_t_end", 48.0), dt=st.session_state.get("int_dt", 0.25), method=st.session_state.get("int_solver_method", "BDF"), extinction_threshold=st.session_state.get("int_extinction_threshold", 1.0) or None, extinction_check_interval=st.session_state.get("int_extinction_check_interval", 0.0) or None)

                    # Compute metrics
                    total_bacteria = result.sum_prefixes("B", "D", "I", "H")
                    nadir_val = np.min(total_bacteria)

                    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                    auc_val = _trapz(total_bacteria, result.time)

                    t_clear = time_to_clearance(
                        result,
                        threshold=st.session_state.get("int_extinction_threshold", 1.0),
                    )
                    t_log_red = time_to_log_reduction(result, n_logs=2.0)

                    runs_outcomes.append({
                        param1_label: val,
                        "Nadir (cells/mL)": nadir_val,
                        "AUC (cells·h/mL)": auc_val,
                        "Clearance Time (h)": t_clear if t_clear is not None else np.nan,
                        "2-Log Red Time (h)": t_log_red if t_log_red is not None else np.nan
                    })
                    _lbl = f"{param1_label} = {val:.2e}"
                    trajectories.append((result.time, total_bacteria, _lbl))
                    if _od_enabled:
                        _od = (_safe_od(result, total_bacteria))
                        od_trajectories.append((result.time, _od, _lbl))
                    progress_bar.progress((idx + 1) / len(sweep_values))

                status_text.text("Sweep completed!")
                st.session_state.param_sweep_result = {
                    "type": "1D",
                    "param1_label": param1_label,
                    "spacing": spacing,
                    "summary": runs_outcomes,
                    "trajectories": [(np.asarray(t), np.asarray(b), lbl) for t, b, lbl in trajectories],
                    "od_trajectories": [(np.asarray(t), np.asarray(o), lbl) for t, o, lbl in od_trajectories],
                }

            elif sweep_type == "Coupled (linked)":
                # Parse each selected parameter's value series; all must be equal length.
                labels = st.session_state.get("pc_labels", [])
                if not labels:
                    st.error("Select at least one parameter to sweep together.")
                    st.stop()
                series = {}
                for _ci, lbl in enumerate(labels):
                    vals = parse_comma_separated_series(st.session_state.get(f"pc_series_{_ci}", ""))
                    if not vals:
                        st.error(f"Provide a value series for '{lbl}'.")
                        st.stop()
                    series[lbl] = vals
                M = len(next(iter(series.values())))
                if any(len(v) != M for v in series.values()):
                    st.error("All value series must have the same number of points.")
                    st.stop()

                runs_outcomes = []
                trajectories = []
                od_trajectories = []
                _od_enabled = st.session_state.get("int_debris_enabled", False)
                for k in range(M):
                    status_text.text(f"Running simulation {k+1} of {M}...")
                    c_k, ib_k, ip_k, is_k, mk_k = nominal_config, initial_B, initial_P, initial_S, model_kwargs
                    for lbl in labels:
                        c_k, ib_k, ip_k, is_k, mk_k = apply_sweep_parameter(
                            series[lbl][k], sweep_params[lbl], c_k, ib_k, ip_k, is_k, mk_k)
                    t_prerun = st.session_state.get("int_t_prerun", 0.0)
                    if t_prerun > 0:
                        ic = stationary_phase_ic(c_k, t_prerun=t_prerun, B0=ib_k)
                        ib_k = ic.B
                        is_k = max(float(ic.S), 0.0)
                        if ic.D is not None:
                            mk_k["initial_D"] = ic.D
                        if ic.Imm is not None:
                            mk_k["initial_Imm"] = ic.Imm
                        _carry_prerun_debris(ic, mk_k)
                    model = PBIModel(c_k, initial_B=ib_k, initial_P=ip_k, initial_S=is_k, **mk_k)
                    result = solve_ode(model, t_end=st.session_state.get("int_t_end", 48.0), dt=st.session_state.get("int_dt", 0.25), method=st.session_state.get("int_solver_method", "BDF"), extinction_threshold=st.session_state.get("int_extinction_threshold", 1.0) or None, extinction_check_interval=st.session_state.get("int_extinction_check_interval", 0.0) or None)
                    total_bacteria = result.sum_prefixes("B", "D", "I", "H")
                    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                    t_clear = time_to_clearance(result, threshold=st.session_state.get("int_extinction_threshold", 1.0))
                    _row = {lbl: series[lbl][k] for lbl in labels}
                    _row.update({
                        "Nadir (cells/mL)": np.min(total_bacteria),
                        "AUC (cells·h/mL)": _trapz(total_bacteria, result.time),
                        "Clearance Time (h)": t_clear if t_clear is not None else np.nan,
                        "2-Log Red Time (h)": time_to_log_reduction(result, n_logs=2.0) or np.nan,
                    })
                    runs_outcomes.append(_row)
                    _lbl_txt = ", ".join(f"{s.rsplit(' - ',1)[0].split('(')[0].strip()}={series[s][k]:.2g}" for s in labels)
                    _lbl = f"Step {k+1}: {_lbl_txt}"
                    trajectories.append((result.time, total_bacteria, _lbl))
                    if _od_enabled:
                        _od = (_safe_od(result, total_bacteria))
                        od_trajectories.append((result.time, _od, _lbl))
                    progress_bar.progress((k + 1) / M)

                status_text.text("Sweep completed!")
                st.session_state.param_sweep_result = {
                    "type": "coupled",
                    "labels": list(labels),
                    "summary": runs_outcomes,
                    "trajectories": [(np.asarray(t), np.asarray(b), lbl) for t, b, lbl in trajectories],
                    "od_trajectories": [(np.asarray(t), np.asarray(o), lbl) for t, o, lbl in od_trajectories],
                }

            else:
                # 2D Sweep
                if spacing == "Logarithmic":
                    if min_val <= 0 or max_val <= 0:
                        st.error("Logarithmic spacing requires positive bounds.")
                        st.stop()
                    sweep_values1 = np.logspace(np.log10(min_val), np.log10(max_val), int(steps))
                else:
                    sweep_values1 = np.linspace(min_val, max_val, int(steps))

                if spacing2 == "Logarithmic":
                    if min_val2 <= 0 or max_val2 <= 0:
                        st.error("Logarithmic spacing requires positive bounds.")
                        st.stop()
                    sweep_values2 = np.logspace(np.log10(min_val2), np.log10(max_val2), int(steps2))
                else:
                    sweep_values2 = np.linspace(min_val2, max_val2, int(steps2))

                total_sims = len(sweep_values1) * len(sweep_values2)
                sim_idx = 0

                grid_auc = np.zeros((len(sweep_values2), len(sweep_values1)))
                grid_nadir = np.zeros((len(sweep_values2), len(sweep_values1)))
                grid_clear = np.zeros((len(sweep_values2), len(sweep_values1)))

                for i2, val2 in enumerate(sweep_values2):
                    for i1, val1 in enumerate(sweep_values1):
                        status_text.text(f"Running simulation {sim_idx+1} of {total_sims}...")
                        
                        # Apply both parameters
                        c_k, ib_k, ip_k, is_k, mk_k = apply_sweep_parameter(
                            val1, meta1, nominal_config, initial_B, initial_P, initial_S, model_kwargs
                        )
                        c_k, ib_k, ip_k, is_k, mk_k = apply_sweep_parameter(
                            val2, meta2, c_k, ib_k, ip_k, is_k, mk_k
                        )

                        # equilibrate — carry the full stationary state (B, D, S, Imm).
                        t_prerun = st.session_state.get("int_t_prerun", 0.0)
                        if t_prerun > 0:
                            ic = stationary_phase_ic(c_k, t_prerun=t_prerun, B0=ib_k)
                            ib_k = ic.B
                            is_k = max(float(ic.S), 0.0)
                            if ic.D is not None:
                                mk_k["initial_D"] = ic.D
                            if ic.Imm is not None:
                                mk_k["initial_Imm"] = ic.Imm
                            _carry_prerun_debris(ic, mk_k)

                        model = PBIModel(c_k, initial_B=ib_k, initial_P=ip_k, initial_S=is_k, **mk_k)
                        result = solve_ode(model, t_end=st.session_state.get("int_t_end", 48.0), dt=st.session_state.get("int_dt", 0.25), method=st.session_state.get("int_solver_method", "BDF"), extinction_threshold=st.session_state.get("int_extinction_threshold", 1.0) or None, extinction_check_interval=st.session_state.get("int_extinction_check_interval", 0.0) or None)

                        total_bacteria = result.sum_prefixes("B", "D", "I", "H")
                        nadir_val = np.min(total_bacteria)

                        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                        auc_val = _trapz(total_bacteria, result.time)

                        t_clear = time_to_clearance(
                            result,
                            threshold=st.session_state.get("int_extinction_threshold", 1.0),
                        )
                        
                        grid_auc[i2, i1] = auc_val
                        grid_nadir[i2, i1] = nadir_val
                        grid_clear[i2, i1] = t_clear if t_clear is not None else st.session_state.get("int_t_end", 48.0)

                        sim_idx += 1
                        progress_bar.progress(sim_idx / total_sims)

                status_text.text("Sweep completed!")
                st.session_state.param_sweep_result = {
                    "type": "2D",
                    "param1_label": param1_label, "param2_label": param2_label,
                    "spacing": spacing, "spacing2": spacing2,
                    "sweep_values1": np.asarray(sweep_values1), "sweep_values2": np.asarray(sweep_values2),
                    "grid_auc": grid_auc, "grid_nadir": grid_nadir, "grid_clear": grid_clear,
                }

        # Render the (persisted) parameter-sweep result if one exists.
        _ps = st.session_state.get("param_sweep_result")
        if _ps:
            import plotly.graph_objects as go
            if _ps["type"] == "1D":
                _p1 = _ps["param1_label"]
                df_summary = pd.DataFrame(_ps["summary"])
                st.markdown("#### Summary of Runs")
                st.dataframe(
                    df_summary.style.format({
                        _p1: "{:.2e}", "Nadir (cells/mL)": "{:.2e}", "AUC (cells·h/mL)": "{:.2e}",
                        "Clearance Time (h)": "{:.1f}", "2-Log Red Time (h)": "{:.1f}"
                    }),
                    width="stretch")
                st.markdown("#### Raw Simulation Trajectories (Viable Bacteria)")
                fig_traj = go.Figure()
                for t_arr, b_arr, legend_lbl in _ps["trajectories"]:
                    fig_traj.add_trace(go.Scatter(x=t_arr, y=np.maximum(b_arr, 1.0), mode='lines', name=legend_lbl))
                fig_traj.update_layout(
                    xaxis_title="Time (hours)", yaxis_title="Total Viable Bacteria (CFU/mL)",
                    yaxis_type="log", template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                st.plotly_chart(fig_traj, width="stretch")
                if _ps.get("od_trajectories"):
                    st.markdown("#### Raw Simulation Trajectories (Optical Density)")
                    fig_od = go.Figure()
                    for t_arr, od_arr, legend_lbl in _ps["od_trajectories"]:
                        fig_od.add_trace(go.Scatter(x=t_arr, y=od_arr, mode='lines', name=legend_lbl))
                    fig_od.update_layout(
                        xaxis_title="Time (hours)", yaxis_title="Optical density (AU)",
                        template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                    st.plotly_chart(fig_od, width="stretch")
                st.markdown("#### Outcome Metrics vs Parameter Value")
                fig_metric = go.Figure()
                fig_metric.add_trace(go.Scatter(x=df_summary[_p1], y=df_summary["AUC (cells·h/mL)"], mode="lines+markers", name="Bacterial AUC", yaxis="y1"))
                fig_metric.add_trace(go.Scatter(x=df_summary[_p1], y=df_summary["Nadir (cells/mL)"], mode="lines+markers", name="Nadir", yaxis="y2"))
                fig_metric.update_layout(
                    xaxis=dict(title=_p1, type="log" if _ps["spacing"] == "Logarithmic" else "linear"),
                    yaxis=dict(title="AUC (cells·h/mL)", type="log"),
                    yaxis2=dict(title="Nadir (cells/mL)", type="log", overlaying="y", side="right"),
                    template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                st.plotly_chart(fig_metric, width="stretch")
            elif _ps["type"] == "coupled":
                _labels = _ps["labels"]
                df_summary = pd.DataFrame(_ps["summary"])
                st.markdown("#### Summary of Runs (linked parameters)")
                _fmt = {c: "{:.2e}" for c in _labels}
                _fmt.update({"Nadir (cells/mL)": "{:.2e}", "AUC (cells·h/mL)": "{:.2e}",
                             "Clearance Time (h)": "{:.1f}", "2-Log Red Time (h)": "{:.1f}"})
                st.dataframe(df_summary.style.format(_fmt), width="stretch")
                st.markdown("#### Raw Simulation Trajectories (Viable Bacteria)")
                fig_traj = go.Figure()
                for t_arr, b_arr, legend_lbl in _ps["trajectories"]:
                    fig_traj.add_trace(go.Scatter(x=t_arr, y=np.maximum(b_arr, 1.0), mode='lines', name=legend_lbl))
                fig_traj.update_layout(
                    xaxis_title="Time (hours)", yaxis_title="Total Viable Bacteria (CFU/mL)",
                    yaxis_type="log", template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                st.plotly_chart(fig_traj, width="stretch")
                if _ps.get("od_trajectories"):
                    st.markdown("#### Raw Simulation Trajectories (Optical Density)")
                    fig_od = go.Figure()
                    for t_arr, od_arr, legend_lbl in _ps["od_trajectories"]:
                        fig_od.add_trace(go.Scatter(x=t_arr, y=od_arr, mode='lines', name=legend_lbl))
                    fig_od.update_layout(
                        xaxis_title="Time (hours)", yaxis_title="Optical density (AU)",
                        template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                    st.plotly_chart(fig_od, width="stretch")
                st.markdown("#### Outcome Metrics vs Step Index")
                _step = list(range(1, len(df_summary) + 1))
                fig_metric = go.Figure()
                fig_metric.add_trace(go.Scatter(x=_step, y=df_summary["AUC (cells·h/mL)"], mode="lines+markers", name="Bacterial AUC", yaxis="y1"))
                fig_metric.add_trace(go.Scatter(x=_step, y=df_summary["Nadir (cells/mL)"], mode="lines+markers", name="Nadir", yaxis="y2"))
                fig_metric.update_layout(
                    xaxis=dict(title="Step index"),
                    yaxis=dict(title="AUC (cells·h/mL)", type="log"),
                    yaxis2=dict(title="Nadir (cells/mL)", type="log", overlaying="y", side="right"),
                    template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                st.plotly_chart(fig_metric, width="stretch")
            else:
                _p1, _p2 = _ps["param1_label"], _ps["param2_label"]
                _xt = "log" if _ps["spacing"] == "Logarithmic" else "linear"
                _yt = "log" if _ps["spacing2"] == "Logarithmic" else "linear"
                st.markdown("#### Outcome Heatmaps (2D Sweep)")
                h1, h2 = st.columns(2)
                with h1:
                    fig_auc = go.Figure(data=go.Contour(z=_ps["grid_auc"], x=_ps["sweep_values1"], y=_ps["sweep_values2"], colorscale="Viridis", colorbar=dict(title="AUC")))
                    fig_auc.update_layout(title="Bacterial AUC Heatmap", xaxis=dict(title=_p1, type=_xt), yaxis=dict(title=_p2, type=_yt), template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                    st.plotly_chart(fig_auc, width="stretch")
                with h2:
                    fig_nadir = go.Figure(data=go.Contour(z=_ps["grid_nadir"], x=_ps["sweep_values1"], y=_ps["sweep_values2"], colorscale="Magma", colorbar=dict(title="Nadir")))
                    fig_nadir.update_layout(title="Bacterial Nadir Heatmap", xaxis=dict(title=_p1, type=_xt), yaxis=dict(title=_p2, type=_yt), template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                    st.plotly_chart(fig_nadir, width="stretch")
        else:
            st.info("Configure parameters and click **Run Sweep** to start the analysis.")

    with st.expander("🐍 View Python Reproduction Code"):
        st.caption("Standalone script that reproduces this sweep — the recorded base model "
                   "plus a loop calling the app's own apply_sweep_parameter per value. "
                   "(1D sweeps only.)")
        try:
            st.code(generate_param_sweep_reproduction_code(), language="python")
        except Exception as _e:
            st.warning(f"Reproduction code unavailable: {_e}")

    # Persist the sweep controls so they survive navigation (see reseed above).
    save_widget_config("param_sweep_config", ("p1_", "p2_", "ps_", "pc_"))

# ── Interactive Simulator Page ────────────────────────────────────────────────
elif st.session_state.current_page == "Interactive Simulator":
    st.title("Interactive Simulation Builder")
    st.caption("Configure custom variables, build mathematical parameters, and solve the ODE.")

    st.markdown(
        "<div class='info-banner'>🛠️ Configure your biological layers using "
        "the tabs below, then scroll to the bottom and click <b>Run Simulation</b>!</div>",
        unsafe_allow_html=True,
    )

    # 1. Retrieve current lists from state
    strains = st.session_state.get("int_strains", [])
    phages = st.session_state.get("int_phages", [])
    antibiotics = st.session_state.get("int_antibiotics", [])
    doses = st.session_state.get("int_doses", [])
    track_nutrients = st.session_state.get("int_track_nutrients", True)

    # 2. Main tabs for parameters configuration
    config_tabs = st.tabs(
        [
            "🧫 Strains & Phages",
            "🧪 Antibiotics & Immunity",
            "📅 Environment & Dosing",
            "⚙️ Solver Settings",
        ]
    )

    # ──── Tab 1: Strains & Phages ─────────────────────────────────────────────
    with config_tabs[0]:
        # Builder Mode Selector
        builder_mode = st.selectbox(
            "Bacterial Population Builder Mode",
            ["Direct (ModelBuilder)", "Binary Genotypes (BRG)", "Custom Strains & Graph (StrainSet)"],
            index=["Direct (ModelBuilder)", "Binary Genotypes (BRG)", "Custom Strains & Graph (StrainSet)"].index(
                st.session_state.get("int_builder_mode", "Direct (ModelBuilder)")
            ),
            key="widget_builder_mode"
        )
        if builder_mode != st.session_state.get("int_builder_mode", "Direct (ModelBuilder)"):
            st.session_state["int_builder_mode"] = builder_mode
            st.session_state.simulation_result = None
            st.session_state.simulation_config = None
            st.rerun()

        # ── Growth model (signal function + its half-saturation / carrying capacity) ──
        # Kept here in the model builder (not the Environment tab) since it defines the
        # growth kinetics. The nutrient environment (S0 / recycle / inflow / washout)
        # stays in Environment & Dosing.
        _gm1, _gm2 = st.columns([2, 1])
        with _gm1:
            _gs_cur_fn = st.session_state.get("int_growth_function", "monod_growth")
            _gs_labels = list(GROWTH_SIGNALS.keys())
            _gs_cur_label = next((L for L, (fn, _) in GROWTH_SIGNALS.items() if fn == _gs_cur_fn), _gs_labels[0])
            _gs_choice = st.selectbox(
                "Growth signal function", _gs_labels, index=_gs_labels.index(_gs_cur_label),
                help="How the per-strain growth rate is modulated:  constant = unlimited;  "
                     "nutrient = Monod S/(Ks+S);  density = logistic (1−ΣB/K);  "
                     "nutrient+density = Monod × logistic.",
            )
        _gs_fn, _gs_track = GROWTH_SIGNALS[_gs_choice]
        st.session_state["int_growth_function"] = _gs_fn
        st.session_state["int_track_nutrients"] = _gs_track
        with _gm2:
            if _gs_fn in ("monod_growth", "monod_logistic_growth"):
                st.session_state["int_monod_constant"] = st.number_input(
                    "Monod constant (Ks)", value=float(st.session_state.get("int_monod_constant", 0.3)),
                    step=0.05, help="Nutrient half-saturation for Monod growth S/(Ks+S).")
            if _gs_fn in ("logistic_growth", "monod_logistic_growth"):
                st.session_state["int_carrying_capacity"] = st.number_input(
                    "Carrying capacity (K)", value=float(st.session_state.get("int_carrying_capacity", 1e9)),
                    format="%.1e", help="Density ceiling for logistic growth (1 − ΣB/K).")

        # Death signal (model-wide) — modulates the per-strain natural death rate dB.
        _dth_cur_fn = st.session_state.get("int_death_function", "constant_death")
        _dth_labels = list(DEATH_SIGNALS.keys())
        _dth_cur_label = next((L for L, fn in DEATH_SIGNALS.items() if fn == _dth_cur_fn), _dth_labels[0])
        _dth_choice = st.selectbox(
            "Death signal function", _dth_labels, index=_dth_labels.index(_dth_cur_label),
            help="How the per-strain natural death rate dB is modulated:  constant = flat rate d "
                 "(the previous behaviour);  nutrient = starvation d·(1−S/(Ks+S)) (rises as nutrients "
                 "deplete);  density = crowding d·min(1, ΣB/K). Note: starvation death only separates "
                 "stationary from death phase when nutrients persist at a low plateau (recycling).")
        st.session_state["int_death_function"] = DEATH_SIGNALS[_dth_choice]

        # What "density" counts for ALL density signals (dormancy / resuscitation / death).
        st.session_state["int_density_total_cells"] = st.checkbox(
            "Density signals count all cell states (B + I + D + H)",
            value=bool(st.session_state.get("int_density_total_cells", False)),
            key="widget_density_total_cells",
            help="Off (default): density-dependent signals use ACTIVE bacteria only, "
                 "min(1, ΣB/K). On: they use TOTAL cell density including infected (I), "
                 "dormant (D) and hibernating (H) cells — so all compartments count toward "
                 "crowding. Applies to every density / nutrient+density dormancy, "
                 "resuscitation and death signal.")

        st.markdown("---")
        col1, col2 = st.columns(2)

        # ── DIRECT MODE ──
        if builder_mode == "Direct (ModelBuilder)":
            with col1:
                st.markdown("### 🧫 Bacterial Strains")

                n_strains = st.number_input(
                    "Number of strains", min_value=1, max_value=10, value=len(strains)
                )
                if n_strains != len(strains):
                    st.session_state.simulation_result = None
                    st.session_state.simulation_config = None
                # Adjust list size
                if n_strains > len(strains):
                    for i in range(len(strains), n_strains):
                        strains.append(
                            {
                                "name": f"Strain {i}",
                                "initial_B": 1e7 if i == 0 else 0.0,
                                "initial_D": 0.0,
                                "growth_rate": 1.2,
                                "bacteria_to_resource_ratio": 1e9,
                                "death_rate_B": 0.0,
                                "death_rate_D": 0.0,
                                "dormancy_enabled": False,
                                "dormancy_depth": 1,
                                "dormancy_rate": 0.001,
                                "resuscitation_rate": 0.1,
                                "dormancy_diffusion_rate": 0.05,
                                "dormancy_signal": "nutrient",
                                "resuscitation_signal": "nutrient",
                            }
                        )
                elif n_strains < len(strains):
                    strains = strains[:n_strains]
                st.session_state["int_strains"] = strains

                # Inputs per strain
                for i in range(n_strains):
                    with st.expander(f"Strain {i}: {strains[i]['name']}", expanded=True):
                        strains[i]["name"] = st.text_input(
                            "Name", value=strains[i]["name"], key=f"str_name_{i}"
                        )
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            strains[i]["initial_B"] = st.number_input(
                                "Initial Density (B0)",
                                value=float(strains[i]["initial_B"]),
                                format="%.1e",
                                key=f"str_init_{i}",
                            )
                        with cc2:
                            strains[i]["growth_rate"] = st.number_input(
                                "Growth rate (r)",
                                value=float(strains[i]["growth_rate"]),
                                step=0.1,
                                key=f"str_growth_{i}",
                            )
                        strains[i]["bacteria_to_resource_ratio"] = st.number_input(
                            "Bacteria-to-resource ratio",
                            value=float(strains[i].get("bacteria_to_resource_ratio", 1e9)),
                            format="%.2e",
                            key=f"str_ratio_{i}",
                            help="Bacteria produced per unit resource consumed (yield). Governs how fast "
                                 "growth depletes the substrate under nutrient-limited (Monod) growth.",
                        )
                        strains[i]["death_rate_B"] = st.number_input(
                            "Natural death rate (dB)",
                            value=float(strains[i].get("death_rate_B", 0.0)),
                            step=0.01,
                            key=f"str_death_{i}",
                        )
                        strains[i]["dormancy_enabled"] = st.checkbox(
                            "Enable Dormancy",
                            value=strains[i].get("dormancy_enabled", False),
                            key=f"str_dorm_en_{i}",
                        )
                        if strains[i]["dormancy_enabled"]:
                            cd1, cd2 = st.columns(2)
                            with cd1:
                                strains[i]["dormancy_depth"] = st.number_input(
                                    "Depth layers (Q)",
                                    min_value=1,
                                    max_value=10,
                                    value=int(strains[i].get("dormancy_depth", 1)),
                                    key=f"str_depth_{i}",
                                )
                            with cd2:
                                strains[i]["dormancy_rate"] = st.number_input(
                                    "Dormancy rate (sleep)",
                                    value=float(strains[i].get("dormancy_rate", 0.001)),
                                    step=0.05,
                                    key=f"str_sleep_{i}",
                                )
                            cd3, cd4 = st.columns(2)
                            with cd3:
                                strains[i]["resuscitation_rate"] = st.number_input(
                                    "Resuscitation rate (wake)",
                                    value=float(strains[i].get("resuscitation_rate", 0.1)),
                                    step=0.05,
                                    key=f"str_wake_{i}",
                                )
                            with cd4:
                                strains[i]["death_rate_D"] = st.number_input(
                                    "Dormant death rate (dD)",
                                    value=float(strains[i].get("death_rate_D", 0.0)),
                                    step=0.01,
                                    key=f"str_death_d_{i}",
                                )
                            strains[i]["dormancy_diffusion_rate"] = st.number_input(
                                "Depth diffusion rate",
                                value=float(strains[i].get("dormancy_diffusion_rate", 0.05)),
                                step=0.01,
                                key=f"str_diff_{i}",
                            )
                            strains[i]["dormancy_signal"] = st.selectbox(
                                "Dormancy Signal",
                                SIGNAL_OPTIONS,
                                index=SIGNAL_OPTIONS.index(canonical_signal(strains[i].get("dormancy_signal"))),
                                key=f"str_dsig_{i}",
                                help="Entry-rate modulation: constant, nutrient-scarcity (Monod), "
                                     "density (quorum), or nutrient×density.",
                            )
                            strains[i]["resuscitation_signal"] = st.selectbox(
                                "Resuscitation Signal",
                                SIGNAL_OPTIONS,
                                index=SIGNAL_OPTIONS.index(canonical_signal(strains[i].get("resuscitation_signal"))),
                                key=f"str_rsig_{i}",
                            )
                            _dsig_i = canonical_signal(strains[i].get("dormancy_signal"))
                            _rsig_i = canonical_signal(strains[i].get("resuscitation_signal"))
                            if "nutrient" in (_dsig_i, _rsig_i) or "nutrient+density" in (_dsig_i, _rsig_i):
                                # Inherit the growth Monod constant by default (pbisim
                                # convention: dormancy_monod_constant=None → monod_constant);
                                # the user only changes it to decouple the two.
                                _grow_ks = float(st.session_state.get("int_monod_constant", 0.3))
                                strains[i]["dormancy_monod_constant"] = st.number_input(
                                    "Dormancy nutrient half-saturation (Ks)",
                                    value=float(strains[i].get("dormancy_monod_constant", _grow_ks)),
                                    min_value=0.0, format="%g", key=f"str_dks_{i}",
                                    help="Half-saturation for the nutrient dormancy signal. Defaults to the "
                                         "growth Monod constant (inherited); change it to decouple the two. "
                                         "0 also inherits the growth Monod constant.",
                                )
                            if _dsig_i in ("density", "nutrient+density") or _rsig_i in ("density", "nutrient+density"):
                                strains[i]["dormancy_carrying_capacity"] = st.number_input(
                                    "Dormancy density threshold (K_dorm)",
                                    value=float(strains[i].get("dormancy_carrying_capacity", 1e8)),
                                    min_value=0.0, format="%.2e", key=f"str_dcc_{i}",
                                    help="Density threshold (CFU/mL) for the density dormancy signal "
                                         "(rate ∝ ΣB/K_dorm). Default 1e8. Set 0 to inherit the growth "
                                         "carrying capacity instead.",
                                )
                            strains[i]["initial_D"] = st.number_input(
                                "Initial dormant density (D0)",
                                value=float(strains[i].get("initial_D", 0.0)),
                                min_value=0.0,
                                format="%.1e",
                                key=f"str_init_d_{i}",
                                help="Dormant cells/mL at t=0. Defaults to 0. Distributed evenly across Q depth layers.",
                            )

            with col2:
                st.markdown("### 🧬 Phage Strains")

                n_phages = st.number_input(
                    "Number of phages", min_value=0, max_value=10, value=len(phages)
                )
                if n_phages != len(phages):
                    st.session_state.simulation_result = None
                    st.session_state.simulation_config = None
                # Adjust list size
                if n_phages > len(phages):
                    for i in range(len(phages), n_phages):
                        phages.append(
                            {
                                "name": f"Phage {i}",
                                "initial_P": 1e6,
                                "adsorption_rates": 1e-8,
                                "adsorption_rates_dormant": 0.0,
                                "burst_sizes": 50.0,
                                "latent_periods": 0.5,
                                "phage_decay_rates": 0.1,
                                "pk_mode": "None",
                                "Vc": 5000.0,
                                "k_elim": 0.2,
                                "k_in": 0.1,
                                "k_out": 0.05,
                                "Vi": 10.0,
                                "Km_elim": 0.0,
                                "phage_decay_Km": 0.0,
                                "hibernation_rate_s": 0.0,
                                "hibernation_rate_r": 0.0,
                                "lytic_resumption_rate_s": 0.0,
                                "lytic_resumption_rate_r": 0.0,
                            }
                        )
                elif n_phages < len(phages):
                    phages = phages[:n_phages]
                st.session_state["int_phages"] = phages

                # Inputs per phage
                for i in range(n_phages):
                    with st.expander(f"Phage {i}: {phages[i]['name']}", expanded=True):
                        phages[i]["name"] = st.text_input(
                            "Name", value=phages[i]["name"], key=f"phg_name_{i}"
                        )
                        cc3, cc4 = st.columns(2)
                        with cc3:
                            phages[i]["initial_P"] = st.number_input(
                                "Initial Density (P0)",
                                value=float(phages[i]["initial_P"]),
                                format="%.1e",
                                key=f"phg_init_{i}",
                            )
                        with cc4:
                            phages[i]["burst_sizes"] = st.number_input(
                                "Burst size (Y)",
                                value=float(phages[i]["burst_sizes"]),
                                step=10.0,
                                key=f"phg_burst_{i}",
                            )
                        phages[i]["latent_periods"] = st.number_input(
                            "Latent period (hours)",
                            value=float(phages[i]["latent_periods"]),
                            step=0.1,
                            key=f"phg_latent_{i}",
                        )
                        phages[i]["phage_decay_rates"] = st.number_input(
                            "Phage decay rate (m)",
                            value=float(phages[i]["phage_decay_rates"]),
                            step=0.05,
                            key=f"phg_decay_{i}",
                        )

                        # Cross-resistance adsorption rates
                        st.markdown("**Adsorption Rates**")
                        for s_idx in range(n_strains):
                            ads_key = f"ads_{s_idx}_{i}"
                            ads_dorm_key = f"ads_dorm_{s_idx}_{i}"
                            # suscept ads
                            st.session_state[ads_key] = st.number_input(
                                f"Adsorption to {strains[s_idx]['name']} (mL/h)",
                                value=float(st.session_state.get(ads_key, 1e-8 if s_idx == 0 else 0.0)),
                                format="%.1e",
                                key=f"ads_input_{s_idx}_{i}",
                            )
                            # dormant ads
                            st.session_state[ads_dorm_key] = st.number_input(
                                f"Adsorption to dormant {strains[s_idx]['name']} (mL/h)",
                                value=float(st.session_state.get(ads_dorm_key, 0.0)),
                                format="%.1e",
                                key=f"ads_dorm_input_{s_idx}_{i}",
                            )

                        # Advanced biological and PK options
                        st.markdown("**Advanced Phage Kinetics**")
                        phages[i]["phage_decay_Km"] = st.number_input(
                            "Decay Km (Michaelis-Menten)",
                            value=float(phages[i].get("phage_decay_Km", 0.0)),
                            format="%.1e",
                            key=f"phg_decay_km_{i}",
                            help="Phage decay saturation. Set to 0 to disable."
                        )
                        phages[i]["attenuation_rate"] = st.number_input(
                            "Dormant adsorption attenuation (per depth layer)",
                            value=float(phages[i].get("attenuation_rate", 0.0)),
                            min_value=0.0,
                            step=0.1,
                            key=f"phg_atten_{i}",
                            help=(
                                "Exponential decay of adsorption to dormant cells with dormancy "
                                "depth: effective rate = adsorption_dormant × exp(−attenuation × "
                                "depth layer). 0 = phage penetrates all dormant layers equally."
                            ),
                        )

                        st.markdown("**Pseudolysogeny & Hibernation**")
                        phages[i]["hibernation_rate_s"] = st.number_input("Susceptible I->H rate", value=float(phages[i].get("hibernation_rate_s", 0.0)), step=0.05, key=f"phg_hib_s_{i}")
                        phages[i]["hibernation_rate_r"] = st.number_input("Resistant I->H rate", value=float(phages[i].get("hibernation_rate_r", 0.0)), step=0.05, key=f"phg_hib_r_{i}")
                        phages[i]["lytic_resumption_rate_s"] = st.number_input("Susceptible extra H->I rate", value=float(phages[i].get("lytic_resumption_rate_s", 0.0)), step=0.05, key=f"phg_res_s_{i}")
                        phages[i]["lytic_resumption_rate_r"] = st.number_input("Resistant extra H->I rate", value=float(phages[i].get("lytic_resumption_rate_r", 0.0)), step=0.05, key=f"phg_res_r_{i}")

                        st.markdown("**Pharmacokinetics (PK)**")
                        phages[i]["pk_mode"] = st.selectbox(
                            "Phage PK Mode",
                            ["None", "Effect Compartment", "Mass-Conserving"],
                            index=["None", "Effect Compartment", "Mass-Conserving"].index(phages[i].get("pk_mode", "None")),
                            key=f"phg_pk_{i}",
                        )
                        if phages[i]["pk_mode"] != "None":
                            pk1, pk2 = st.columns(2)
                            with pk1:
                                phages[i]["Vc"] = st.number_input("Central volume (Vc mL)", value=float(phages[i].get("Vc", 5000.0)), key=f"phg_vc_{i}")
                                phages[i]["k_elim"] = st.number_input("Elimination rate (k_elim h^-1)", value=float(phages[i].get("k_elim", 0.2)), key=f"phg_kelim_{i}")
                            with pk2:
                                phages[i]["k_in"] = st.number_input("Inflow rate (k_in h^-1)", value=float(phages[i].get("k_in", 0.1)), key=f"phg_kin_{i}")
                                phages[i]["k_out"] = st.number_input("Outflow rate (k_out h^-1)", value=float(phages[i].get("k_out", 0.05)), key=f"phg_kout_{i}")
                            
                            phages[i]["Km_elim"] = st.number_input(
                                "Elimination Km",
                                value=float(phages[i].get("Km_elim", 0.0)),
                                format="%.1e",
                                key=f"phg_km_elim_{i}",
                                help="Central elimination saturation. Set to 0 to disable."
                            )
                            if phages[i]["pk_mode"] == "Mass-Conserving":
                                phages[i]["Vi"] = st.number_input("Infection Site Volume (Vi mL)", value=float(phages[i].get("Vi", 10.0)), key=f"phg_vi_{i}")

                if n_phages > 0:
                    st.markdown("---")
                    st.markdown("### 🧬 Bacterial Mutations (WT → R)")
                    if n_strains == 2**n_phages:
                        st.caption("Per-phage-locus shortcut (binary-genotype layout, n_strains = 2^n_phages):")
                        phg_res_rates = []
                        _prev_mu = st.session_state.get("direct_phg_res_rates", [])
                        for j in range(n_phages):
                            # Seed from the persisted direct_phg_res_rates (a scenario data
                            # key) so a value of 0 survives navigation. The widget key
                            # direct_mu_{j} is dropped when the page isn't rendered, so
                            # value= must come from a key that actually persists — the old
                            # code read `direct_phg_mu_{j}`, which was never written, so it
                            # reverted to 1e-7 on return.
                            res_rate = st.number_input(
                                f"Mutation rate to {phages[j]['name']} resistance (mu)",
                                value=float(_prev_mu[j]) if j < len(_prev_mu) else 1e-7,
                                format="%.1e",
                                key=f"direct_mu_{j}"
                            )
                            phg_res_rates.append(res_rate)
                        st.session_state["direct_phg_res_rates"] = phg_res_rates

                    with st.expander("🔄 Custom mutation network (any number of strains)",
                                     expanded=(n_strains != 2**n_phages)):
                        st.caption(
                            "Define arbitrary strain→strain mutation transitions. Works for any "
                            "strain count (lifts the 2^n_phages requirement). **If any transition "
                            "is added here it overrides the per-locus shortcut above.**"
                        )
                        render_mutation_graph_editor(strains, key_prefix="dir_trans")

        # ── BINARY RESISTANCE GENOTYPES (BRG) ──
        elif builder_mode == "Binary Genotypes (BRG)":
            with col1:
                st.markdown("### 🧫 Base Bacteria (WT)")
                st.session_state["int_brg_base_growth"] = st.number_input(
                    "Base growth rate (r)", value=float(st.session_state.get("int_brg_base_growth", 1.2)), step=0.1
                )
                st.session_state["int_brg_base_ratio"] = st.number_input(
                    "Resource consumption ratio", value=float(st.session_state.get("int_brg_base_ratio", 1e9)), format="%.1e"
                )
                st.session_state["int_brg_death_rate_B"] = st.number_input(
                    "Natural death rate (dB)", value=float(st.session_state.get("int_brg_death_rate_B", 0.0)), step=0.01
                )
                st.session_state["int_brg_dormancy_enabled"] = st.checkbox(
                    "Enable Dormancy", value=st.session_state.get("int_brg_dormancy_enabled", False)
                )
                if st.session_state["int_brg_dormancy_enabled"]:
                    st.session_state["int_brg_dorm_rate"] = st.number_input(
                        "Dormancy rate (sleep)", value=float(st.session_state.get("int_brg_dorm_rate", 0.001)), step=0.05
                    )
                    st.session_state["int_brg_resus_rate"] = st.number_input(
                        "Resuscitation rate (wake)", value=float(st.session_state.get("int_brg_resus_rate", 0.1)), step=0.05
                    )
                    st.session_state["int_brg_diff_rate"] = st.number_input(
                        "Depth diffusion rate", value=float(st.session_state.get("int_brg_diff_rate", 0.05)), step=0.01
                    )
                    st.session_state["int_brg_n_depth"] = st.number_input(
                        "Dormancy depth layers (Q)",
                        min_value=1, max_value=10,
                        value=int(st.session_state.get("int_brg_n_depth", 1)),
                        help="Number of dormancy-depth compartments for all genotypes.",
                    )
                    st.session_state["int_brg_death_rate_D"] = st.number_input(
                        "Dormant death rate (dD)", value=float(st.session_state.get("int_brg_death_rate_D", 0.0)), step=0.01
                    )
                    _bds = canonical_signal(st.session_state.get("int_brg_dorm_signal", "nutrient"))
                    _brs = canonical_signal(st.session_state.get("int_brg_resus_signal", "nutrient"))
                    st.session_state["int_brg_dorm_signal"] = st.selectbox(
                        "Dormancy signal", SIGNAL_OPTIONS, index=SIGNAL_OPTIONS.index(_bds),
                        key="widget_brg_dorm_signal",
                        help="Entry-rate modulation: constant, nutrient-scarcity (Monod), density (quorum), or nutrient×density.")
                    st.session_state["int_brg_resus_signal"] = st.selectbox(
                        "Resuscitation signal", SIGNAL_OPTIONS, index=SIGNAL_OPTIONS.index(_brs),
                        key="widget_brg_resus_signal")
                    _bds2 = canonical_signal(st.session_state["int_brg_dorm_signal"])
                    _brs2 = canonical_signal(st.session_state["int_brg_resus_signal"])
                    if "nutrient" in (_bds2, _brs2) or "nutrient+density" in (_bds2, _brs2):
                        st.session_state["int_brg_dorm_ks"] = st.number_input(
                            "Dormancy nutrient half-saturation (Ks)",
                            value=float(st.session_state.get("int_brg_dorm_ks", st.session_state.get("int_monod_constant", 0.3))),
                            min_value=0.0, format="%g",
                            help="Defaults to the growth Monod constant (inherited); change to decouple.")
                    if _bds2 in ("density", "nutrient+density") or _brs2 in ("density", "nutrient+density"):
                        st.session_state["int_brg_dorm_kdorm"] = st.number_input(
                            "Dormancy density threshold (K_dorm)",
                            value=float(st.session_state.get("int_brg_dorm_kdorm", 1e8)),
                            min_value=0.0, format="%.2e",
                            help="Density threshold (CFU/mL) for the density dormancy signal. 0 inherits growth K.")

                # Renders the loci count
                st.markdown("---")
                st.markdown("### 🧬 Phage Loci")
                n_phg_loci = st.number_input("Number of phage species (loci)", min_value=1, max_value=10, value=max(len(phages), 1))
                if n_phg_loci != len(phages):
                    phages = phages[:n_phg_loci]
                    while len(phages) < n_phg_loci:
                        phages.append({
                            "name": f"Phage {len(phages)}",
                            "initial_P": 1e6,
                            "burst_sizes": 50.0,
                            "latent_periods": 0.5,
                            "phage_decay_rates": 0.1,
                            "pk_mode": "None",
                            "Vc": 5000.0, "k_elim": 0.2, "k_in": 0.1, "k_out": 0.05, "Vi": 10.0,
                            "adsorption_s": 5e-8,
                            "adsorption_r": 0.0,
                            "fitness_cost": 0.05,
                            "mu": 1e-7,
                        })
                    st.session_state["int_phages"] = phages
                    
                for idx in range(n_phg_loci):
                    with st.expander(f"Phage Locus {idx}: {phages[idx]['name']}", expanded=True):
                        phages[idx]["name"] = st.text_input("Locus name", value=phages[idx]["name"], key=f"brg_phg_name_{idx}")
                        phages[idx]["initial_P"] = st.number_input("Initial count (P0)", value=float(phages[idx]["initial_P"]), format="%.1e", key=f"brg_phg_init_{idx}")
                        phages[idx]["adsorption_s"] = st.number_input("Adsorption WT (adsorption_s)", value=float(phages[idx].get("adsorption_s", 5e-8)), format="%.2e", key=f"brg_phg_ads_s_{idx}")
                        phages[idx]["adsorption_r"] = st.number_input("Adsorption Res (adsorption_r)", value=float(phages[idx].get("adsorption_r", 0.0)), format="%.2e", key=f"brg_phg_ads_r_{idx}")
                        phages[idx]["burst_sizes"] = st.number_input("Burst size", value=float(phages[idx]["burst_sizes"]), step=10.0, key=f"brg_phg_burst_{idx}")
                        phages[idx]["latent_periods"] = st.number_input("Latent period (h)", value=float(phages[idx]["latent_periods"]), step=0.1, key=f"brg_phg_latent_{idx}")
                        phages[idx]["phage_decay_rates"] = st.number_input("Phage decay rate", value=float(phages[idx]["phage_decay_rates"]), step=0.05, key=f"brg_phg_decay_{idx}")
                        phages[idx]["fitness_cost"] = st.number_input(
                            "Resistance fitness cost",
                            value=float(phages[idx].get("fitness_cost", 0.05)), step=0.01,
                            key=f"brg_phg_fit_{idx}",
                            help="Fractional growth-rate penalty of phage-resistant genotypes "
                                 "(resistant growth = base × (1 − cost)). Drives the equilibrium "
                                 "initial condition — with cost 0 the resistant mutants are neutral "
                                 "and dominate the pre-treatment equilibrium.",
                        )
                        phages[idx]["mu"] = st.number_input("Mutation rate (mu)", value=float(phages[idx].get("mu", 1e-7)), format="%.1e", key=f"brg_phg_mu_{idx}")
                        phages[idx]["attenuation_rate"] = st.number_input(
                            "Dormant adsorption attenuation (per depth layer)",
                            value=float(phages[idx].get("attenuation_rate", 0.0)),
                            min_value=0.0, step=0.1, key=f"brg_phg_atten_{idx}",
                            help="Exponential decay of dormant-cell adsorption with dormancy depth (0 = none).",
                        )

                        st.markdown("**Advanced Phage Kinetics**")
                        phages[idx]["phage_decay_Km"] = st.number_input(
                            "Decay Km (Michaelis-Menten)",
                            value=float(phages[idx].get("phage_decay_Km", 0.0)),
                            format="%.1e",
                            key=f"brg_phg_decay_km_{idx}",
                            help="Phage decay saturation. Set to 0 to disable.",
                        )

                        st.markdown("**Pseudolysogeny & Hibernation**")
                        phages[idx]["hibernation_rate_s"] = st.number_input("Susceptible I->H rate", value=float(phages[idx].get("hibernation_rate_s", 0.0)), step=0.05, key=f"brg_phg_hib_s_{idx}")
                        phages[idx]["hibernation_rate_r"] = st.number_input("Resistant I->H rate", value=float(phages[idx].get("hibernation_rate_r", 0.0)), step=0.05, key=f"brg_phg_hib_r_{idx}")
                        phages[idx]["lytic_resumption_rate_s"] = st.number_input("Susceptible extra H->I rate", value=float(phages[idx].get("lytic_resumption_rate_s", 0.0)), step=0.05, key=f"brg_phg_res_s_{idx}")
                        phages[idx]["lytic_resumption_rate_r"] = st.number_input("Resistant extra H->I rate", value=float(phages[idx].get("lytic_resumption_rate_r", 0.0)), step=0.05, key=f"brg_phg_res_r_{idx}")

                        st.markdown("**Pharmacokinetics (PK)**")
                        phages[idx]["pk_mode"] = st.selectbox(
                            "Phage PK Mode",
                            ["None", "Effect Compartment", "Mass-Conserving"],
                            index=["None", "Effect Compartment", "Mass-Conserving"].index(phages[idx].get("pk_mode", "None")),
                            key=f"brg_phg_pkmode_{idx}",
                        )
                        if phages[idx]["pk_mode"] != "None":
                            bpk1, bpk2 = st.columns(2)
                            with bpk1:
                                phages[idx]["Vc"] = st.number_input("Central volume (Vc mL)", value=float(phages[idx].get("Vc", 5000.0)), key=f"brg_phg_vc_{idx}")
                                phages[idx]["k_elim"] = st.number_input("Elimination rate (k_elim h^-1)", value=float(phages[idx].get("k_elim", 0.2)), key=f"brg_phg_kelim_{idx}")
                            with bpk2:
                                phages[idx]["k_in"] = st.number_input("Inflow rate (k_in h^-1)", value=float(phages[idx].get("k_in", 0.1)), key=f"brg_phg_kin_{idx}")
                                phages[idx]["k_out"] = st.number_input("Outflow rate (k_out h^-1)", value=float(phages[idx].get("k_out", 0.05)), key=f"brg_phg_kout_{idx}")
                            phages[idx]["Km_elim"] = st.number_input(
                                "Elimination Km",
                                value=float(phages[idx].get("Km_elim", 0.0)),
                                format="%.1e",
                                key=f"brg_phg_km_elim_{idx}",
                                help="Central elimination saturation. Set to 0 to disable.",
                            )
                            if phages[idx]["pk_mode"] == "Mass-Conserving":
                                phages[idx]["Vi"] = st.number_input("Infection Site Volume (Vi mL)", value=float(phages[idx].get("Vi", 10.0)), key=f"brg_phg_vi_{idx}")

            with col2:
                st.markdown("### 🧮 Auto-generated Genotypes")
                # Show list of 2^(m+a) genotypes and initial conditions inputs
                import itertools
                n_abx = len(antibiotics)
                combs = list(itertools.product([0, 1], repeat=n_phg_loci + n_abx))
                st.caption(f"Based on {n_phg_loci} phages and {n_abx} antibiotics, there are {len(combs)} genotypes:")
                
                st.session_state["int_brg_use_eq_ic"] = st.checkbox(
                    "Use equilibrium initial condition",
                    value=st.session_state.get("int_brg_use_eq_ic", False),
                    help="Compute B0 per genotype from replicator-dynamics equilibrium (BRG growth rates + mutation matrix). Overrides per-genotype inputs.",
                )
                if st.session_state["int_brg_use_eq_ic"]:
                    st.session_state["int_brg_eq_total_B"] = st.number_input(
                        "Total bacteria at t=0 (cells/mL)",
                        value=float(st.session_state.get("int_brg_eq_total_B", 1e7)),
                        format="%.1e",
                        key="brg_eq_total_B",
                    )
                    st.caption("Per-genotype B0 computed from `brg.equilibrium_initial_condition()` at run time.")
                else:
                    brg_initial_B = st.session_state.get("int_brg_initial_B", {})
                    for idx, comb in enumerate(combs):
                        if n_abx == 0:
                            lbl = "".join(map(str, comb))
                        else:
                            p_lbl = "".join(map(str, comb[:n_phg_loci])) if n_phg_loci > 0 else ""
                            a_lbl = "".join(map(str, comb[n_phg_loci:]))
                            if n_phg_loci > 0:
                                lbl = f"phi{p_lbl}_abx{a_lbl}"
                            else:
                                lbl = f"abx{a_lbl}"
                        brg_initial_B[lbl] = st.number_input(
                            f"Initial count for genotype {lbl}",
                            value=float(brg_initial_B.get(lbl, 1e7 if idx == 0 else 0.0)),
                            format="%.1e",
                            key=f"brg_init_B_{lbl}"
                        )
                    st.session_state.int_brg_initial_B = brg_initial_B

        # ── CUSTOM STRAINS & MUTATION GRAPH (StrainSet) ──
        elif builder_mode == "Custom Strains & Graph (StrainSet)":
            with col1:
                st.markdown("### 🧫 Custom Bacterial Strains")
                
                n_strains = st.number_input("Number of custom strains", min_value=1, max_value=10, value=max(len(strains), 1))
                if n_strains != len(strains):
                    strains = strains[:n_strains]
                    while len(strains) < n_strains:
                        strains.append({
                            "name": f"Strain {len(strains)}",
                            "initial_B": 1e7,
                            "growth_rate": 1.2,
                            "bacteria_to_resource_ratio": 1e9,
                            "death_rate_B": 0.0,
                            "death_rate_D": 0.0,
                            "dormancy_enabled": False,
                            "dormancy_rate": 0.001, "resuscitation_rate": 0.1, "dormancy_diffusion_rate": 0.05
                        })
                    st.session_state["int_strains"] = strains
                    
                # Renders expanders per strain
                for i in range(n_strains):
                    with st.expander(f"Strain {i}: {strains[i]['name']}", expanded=True):
                        strains[i]["name"] = st.text_input("Strain name", value=strains[i]["name"], key=f"ss_str_name_{i}")
                        strains[i]["initial_B"] = st.number_input("Initial count (B0)", value=float(strains[i]["initial_B"]), format="%.1e", key=f"ss_str_init_{i}")
                        strains[i]["growth_rate"] = st.number_input("Growth rate (r)", value=float(strains[i]["growth_rate"]), step=0.1, key=f"ss_str_growth_{i}")
                        strains[i]["bacteria_to_resource_ratio"] = st.number_input(
                            "Bacteria-to-resource ratio", value=float(strains[i].get("bacteria_to_resource_ratio", 1e9)),
                            format="%.2e", key=f"ss_str_ratio_{i}",
                            help="Bacteria produced per unit resource consumed (yield). Governs how fast "
                                 "growth depletes the substrate under nutrient-limited (Monod) growth.")
                        strains[i]["death_rate_B"] = st.number_input("Natural death rate (dB)", value=float(strains[i].get("death_rate_B", 0.0)), step=0.01, key=f"ss_str_death_{i}")
                        
                        strains[i]["dormancy_enabled"] = st.checkbox("Enable Dormancy", value=strains[i].get("dormancy_enabled", False), key=f"ss_str_dorm_{i}")
                        if strains[i]["dormancy_enabled"]:
                            strains[i]["dormancy_depth"] = st.number_input("Depth layers (Q)", min_value=1, max_value=10, value=int(strains[i].get("dormancy_depth", 1)), key=f"ss_str_depth_{i}", help="Number of dormancy-depth compartments (max across strains sets the model n_depth).")
                            strains[i]["dormancy_rate"] = st.number_input("Dormancy rate", value=float(strains[i].get("dormancy_rate", 0.001)), key=f"ss_str_sleep_{i}")
                            strains[i]["resuscitation_rate"] = st.number_input("Resuscitation rate", value=float(strains[i].get("resuscitation_rate", 0.1)), key=f"ss_str_wake_{i}")
                            strains[i]["dormancy_diffusion_rate"] = st.number_input("Depth diffusion", value=float(strains[i].get("dormancy_diffusion_rate", 0.05)), key=f"ss_str_diff_{i}")
                            strains[i]["death_rate_D"] = st.number_input("Dormant death rate (dD)", value=float(strains[i].get("death_rate_D", 0.0)), step=0.01, key=f"ss_str_death_d_{i}")
                            _sds = canonical_signal(strains[i].get("dormancy_signal", "nutrient"))
                            _srs = canonical_signal(strains[i].get("resuscitation_signal", "nutrient"))
                            strains[i]["dormancy_signal"] = st.selectbox(
                                "Dormancy signal", SIGNAL_OPTIONS, index=SIGNAL_OPTIONS.index(_sds), key=f"ss_str_dsig_{i}",
                                help="Model-wide dormancy entry signal (taken from the first dormancy-enabled strain).")
                            strains[i]["resuscitation_signal"] = st.selectbox(
                                "Resuscitation signal", SIGNAL_OPTIONS, index=SIGNAL_OPTIONS.index(_srs), key=f"ss_str_rsig_{i}")
                            _sds2 = canonical_signal(strains[i]["dormancy_signal"])
                            _srs2 = canonical_signal(strains[i]["resuscitation_signal"])
                            if "nutrient" in (_sds2, _srs2) or "nutrient+density" in (_sds2, _srs2):
                                strains[i]["dormancy_monod_constant"] = st.number_input(
                                    "Dormancy nutrient half-saturation (Ks)",
                                    value=float(strains[i].get("dormancy_monod_constant", st.session_state.get("int_monod_constant", 0.3))),
                                    min_value=0.0, format="%g", key=f"ss_str_dks_{i}",
                                    help="Defaults to the growth Monod constant (inherited); change to decouple.")
                            if _sds2 in ("density", "nutrient+density") or _srs2 in ("density", "nutrient+density"):
                                strains[i]["dormancy_carrying_capacity"] = st.number_input(
                                    "Dormancy density threshold (K_dorm)",
                                    value=float(strains[i].get("dormancy_carrying_capacity", 1e8)),
                                    min_value=0.0, format="%.2e", key=f"ss_str_dcc_{i}",
                                    help="Density threshold (CFU/mL). 0 inherits growth K.")

                        if len(phages) > 0:
                            st.markdown("**Phage Adsorption Rates**")
                            for p_idx in range(len(phages)):
                                p_name = phages[p_idx]["name"]
                                st.session_state[f"ads_{i}_{p_idx}"] = st.number_input(
                                    f"Adsorption of {p_name} (mL/h)",
                                    value=float(st.session_state.get(f"ads_{i}_{p_idx}", 1e-8 if i == 0 else 0.0)),
                                    format="%.1e",
                                    key=f"ss_ads_input_{i}_{p_idx}"
                                )
                                if strains[i]["dormancy_enabled"]:
                                    st.session_state[f"ads_dorm_{i}_{p_idx}"] = st.number_input(
                                        f"Dormant adsorption of {p_name} (mL/h)",
                                        value=float(st.session_state.get(f"ads_dorm_{i}_{p_idx}", 0.0)),
                                        format="%.1e",
                                        key=f"ss_ads_dorm_input_{i}_{p_idx}"
                                    )

                # 🔄 Transitions graph editor
                st.markdown("#### 🔄 Mutation Graph (Transitions)")
                transitions = st.session_state.get("int_transitions", [])
                
                for idx, trans in enumerate(transitions):
                    c_src, c_dest, c_rate, c_del = st.columns([3, 3, 3, 1])
                    with c_src:
                        src_ops = [s["name"] for s in strains]
                        trans["from"] = st.selectbox(f"From", src_ops, index=src_ops.index(trans["from"]) if trans["from"] in src_ops else 0, key=f"trans_src_{idx}")
                    with c_dest:
                        dest_ops = [s["name"] for s in strains]
                        trans["to"] = st.selectbox(f"To", dest_ops, index=dest_ops.index(trans["to"]) if trans["to"] in dest_ops else 0, key=f"trans_dest_{idx}")
                    with c_rate:
                        trans["rate"] = st.number_input(f"Rate", value=float(trans["rate"]), format="%.2e", key=f"trans_rate_{idx}")
                    with c_del:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🗑️", key=f"trans_del_{idx}"):
                            transitions.pop(idx)
                            st.session_state.int_transitions = transitions
                            st.rerun()
                            
                if st.button("➕ Add Mutation Transition"):
                    transitions.append({"from": strains[0]["name"] if strains else "", "to": strains[0]["name"] if strains else "", "rate": 1e-7})
                    st.session_state.int_transitions = transitions
                    st.rerun()

            with col2:
                st.markdown("### 🧬 Phage Strains")
                n_phages = st.number_input("Number of phages", min_value=0, max_value=10, value=len(phages))
                if n_phages != len(phages):
                    phages = phages[:n_phages]
                    while len(phages) < n_phages:
                        phages.append({
                            "name": f"Phage {len(phages)}",
                            "initial_P": 1e6,
                            "burst_sizes": 50.0,
                            "latent_periods": 0.5,
                            "phage_decay_rates": 0.1,
                            "pk_mode": "None",
                            "Vc": 5000.0, "k_elim": 0.2, "k_in": 0.1, "k_out": 0.05, "Vi": 10.0,
                        })
                    st.session_state["int_phages"] = phages
                    
                for idx in range(n_phages):
                    with st.expander(f"Phage {idx}: {phages[idx]['name']}", expanded=True):
                        phages[idx]["name"] = st.text_input("Phage name", value=phages[idx]["name"], key=f"ss_phg_name_{idx}")
                        phages[idx]["initial_P"] = st.number_input("Initial count (P0)", value=float(phages[idx]["initial_P"]), format="%.1e", key=f"ss_phg_init_{idx}")
                        phages[idx]["burst_sizes"] = st.number_input("Burst size (Y)", value=float(phages[idx].get("burst_sizes", 50.0)), step=10.0, key=f"ss_phg_burst_{idx}")
                        phages[idx]["latent_periods"] = st.number_input("Latent period (h)", value=float(phages[idx].get("latent_periods", 0.5)), step=0.1, key=f"ss_phg_latent_{idx}")
                        phages[idx]["phage_decay_rates"] = st.number_input("Phage decay rate", value=float(phages[idx]["phage_decay_rates"]), step=0.05, key=f"ss_phg_decay_{idx}")
                        phages[idx]["attenuation_rate"] = st.number_input(
                            "Dormant adsorption attenuation (per depth layer)",
                            value=float(phages[idx].get("attenuation_rate", 0.0)),
                            min_value=0.0, step=0.1, key=f"ss_phg_atten_{idx}",
                            help="Exponential decay of dormant-cell adsorption with dormancy depth (0 = none).",
                        )

                        st.markdown("**Advanced Phage Kinetics**")
                        phages[idx]["phage_decay_Km"] = st.number_input(
                            "Decay Km (Michaelis-Menten)",
                            value=float(phages[idx].get("phage_decay_Km", 0.0)),
                            format="%.1e",
                            key=f"ss_phg_decay_km_{idx}",
                            help="Phage decay saturation. Set to 0 to disable.",
                        )

                        st.markdown("**Pseudolysogeny & Hibernation**")
                        phages[idx]["hibernation_rate_s"] = st.number_input("Susceptible I->H rate", value=float(phages[idx].get("hibernation_rate_s", 0.0)), step=0.05, key=f"ss_phg_hib_s_{idx}")
                        phages[idx]["hibernation_rate_r"] = st.number_input("Resistant I->H rate", value=float(phages[idx].get("hibernation_rate_r", 0.0)), step=0.05, key=f"ss_phg_hib_r_{idx}")
                        phages[idx]["lytic_resumption_rate_s"] = st.number_input("Susceptible extra H->I rate", value=float(phages[idx].get("lytic_resumption_rate_s", 0.0)), step=0.05, key=f"ss_phg_res_s_{idx}")
                        phages[idx]["lytic_resumption_rate_r"] = st.number_input("Resistant extra H->I rate", value=float(phages[idx].get("lytic_resumption_rate_r", 0.0)), step=0.05, key=f"ss_phg_res_r_{idx}")

                        st.markdown("**Pharmacokinetics (PK)**")
                        phages[idx]["pk_mode"] = st.selectbox(
                            "Phage PK Mode",
                            ["None", "Effect Compartment", "Mass-Conserving"],
                            index=["None", "Effect Compartment", "Mass-Conserving"].index(phages[idx].get("pk_mode", "None")),
                            key=f"ss_phg_pkmode_{idx}",
                        )
                        if phages[idx]["pk_mode"] != "None":
                            spk1, spk2 = st.columns(2)
                            with spk1:
                                phages[idx]["Vc"] = st.number_input("Central volume (Vc mL)", value=float(phages[idx].get("Vc", 5000.0)), key=f"ss_phg_vc_{idx}")
                                phages[idx]["k_elim"] = st.number_input("Elimination rate (k_elim h^-1)", value=float(phages[idx].get("k_elim", 0.2)), key=f"ss_phg_kelim_{idx}")
                            with spk2:
                                phages[idx]["k_in"] = st.number_input("Inflow rate (k_in h^-1)", value=float(phages[idx].get("k_in", 0.1)), key=f"ss_phg_kin_{idx}")
                                phages[idx]["k_out"] = st.number_input("Outflow rate (k_out h^-1)", value=float(phages[idx].get("k_out", 0.05)), key=f"ss_phg_kout_{idx}")
                            phages[idx]["Km_elim"] = st.number_input(
                                "Elimination Km",
                                value=float(phages[idx].get("Km_elim", 0.0)),
                                format="%.1e",
                                key=f"ss_phg_km_elim_{idx}",
                                help="Central elimination saturation. Set to 0 to disable.",
                            )
                            if phages[idx]["pk_mode"] == "Mass-Conserving":
                                phages[idx]["Vi"] = st.number_input("Infection Site Volume (Vi mL)", value=float(phages[idx].get("Vi", 10.0)), key=f"ss_phg_vi_{idx}")

    # ──── Tab 2: Antibiotics & Immunity ───────────────────────────────────────
    with config_tabs[1]:
        col1, col2 = st.columns(2)

        # Antibiotics
        with col1:
            st.markdown("### 💊 Antibiotics")

            n_abx = st.number_input(
                "Number of antibiotics", min_value=0, max_value=6, value=len(antibiotics)
            )
            if n_abx != len(antibiotics):
                st.session_state.simulation_result = None
                st.session_state.simulation_config = None
            # Adjust list size
            if n_abx > len(antibiotics):
                for i in range(len(antibiotics), n_abx):
                    antibiotics.append(
                        {
                            "name": f"Drug {i}",
                            "Vc": 250.0,
                            "k_elim": 0.3,
                            "k12": 0.0,
                            "k21": 0.0,
                            "emax": 3.0,
                            "ec50": 0.2,
                            "hill": 1.5,
                            "f_lyse": 0.0,
                            "inoculum_effect_constant": 0.0,
                            "Km_elim": 0.0,
                            "emax_r": 0.3,
                            "ec50_r": 2.0,
                            "fitness_cost": 0.05,
                            "mu": 1e-7,
                        }
                    )
            elif n_abx < len(antibiotics):
                antibiotics = antibiotics[:n_abx]
            st.session_state["int_antibiotics"] = antibiotics

            # Inputs per antibiotic
            for i in range(n_abx):
                with st.expander(
                    f"Antibiotic {i}: {antibiotics[i]['name']}", expanded=True
                ):
                    antibiotics[i]["name"] = st.text_input(
                        "Name", value=antibiotics[i]["name"], key=f"abx_name_{i}"
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        antibiotics[i]["Vc"] = st.number_input(
                            "Volume (Vc L)",
                            value=float(antibiotics[i].get("Vc", 250.0)),
                            key=f"abx_vc_{i}",
                        )
                        antibiotics[i]["k_elim"] = st.number_input(
                            "Clearance (k_elim h^-1)",
                            value=float(antibiotics[i]["k_elim"]),
                            step=0.05,
                            key=f"abx_kelim_{i}",
                        )
                    with c2:
                        antibiotics[i]["k12"] = st.number_input(
                            "Transfer central->peripheral (k12)",
                            value=float(antibiotics[i].get("k12", 0.0)),
                            step=0.05,
                            key=f"abx_k12_{i}",
                        )
                        antibiotics[i]["k21"] = st.number_input(
                            "Transfer peripheral->central (k21)",
                            value=float(antibiotics[i].get("k21", 0.0)),
                            step=0.05,
                            key=f"abx_k21_{i}",
                        )

                    st.markdown("**Pharmacodynamics (PD)**")
                    
                    if builder_mode == "Binary Genotypes (BRG)":
                        # Susceptible vs Resistant inputs
                        antibiotics[i]["emax"] = st.number_input("Susceptible Max Efficacy (Emax_s)", value=float(antibiotics[i].get("emax", 3.0)), key=f"abx_emax_s_{i}")
                        antibiotics[i]["emax_r"] = st.number_input("Resistant Max Efficacy (Emax_r)", value=float(antibiotics[i].get("emax_r", 0.3)), key=f"abx_emax_r_{i}")
                        
                        antibiotics[i]["ec50"] = st.number_input("Susceptible EC50_s", value=float(antibiotics[i].get("ec50", 0.2)), key=f"abx_ec50_s_{i}")
                        antibiotics[i]["ec50_r"] = st.number_input("Resistant EC50_r", value=float(antibiotics[i].get("ec50_r", 2.0)), key=f"abx_ec50_r_{i}")
                        
                        antibiotics[i]["fitness_cost"] = st.number_input(
                            "Resistance fitness cost",
                            value=float(antibiotics[i].get("fitness_cost", 0.05)), step=0.01,
                            key=f"abx_fit_{i}",
                            help="Fractional growth-rate penalty of antibiotic-resistant genotypes. "
                                 "Drives the equilibrium initial condition; cost 0 → resistant "
                                 "mutants dominate the pre-treatment equilibrium.",
                        )
                        antibiotics[i]["mu"] = st.number_input("Mutation rate (mu)", value=float(antibiotics[i].get("mu", 1e-7)), format="%.1e", key=f"abx_mu_{i}")
                    else:
                        # Direct parameters
                        c3, c4 = st.columns(2)
                        with c3:
                            antibiotics[i]["emax"] = st.number_input(
                                "Max efficacy (Emax)",
                                value=float(antibiotics[i]["emax"]),
                                step=0.5,
                                key=f"abx_emax_{i}",
                            )
                        with c4:
                            antibiotics[i]["ec50"] = st.number_input(
                                "Half-max conc (EC50)",
                                value=float(antibiotics[i]["ec50"]),
                                step=0.05,
                                key=f"abx_ec50_{i}",
                            )
                            
                    antibiotics[i]["hill"] = st.number_input(
                        "Hill coefficient (H)",
                        value=float(antibiotics[i].get("hill", 1.5)),
                        step=0.1,
                        key=f"abx_hill_{i}",
                    )
                    
                    # Advanced PD inoculum + lytic
                    st.markdown("**Advanced PD features**")
                    antibiotics[i]["f_lyse"] = st.slider(
                        "Bacteriolytic Fraction (f_lyse)",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(antibiotics[i].get("f_lyse", 0.0)),
                        step=0.1,
                        key=f"abx_flyse_{i}",
                        help="0.0 = non-lytic (e.g. Cipro); 1.0 = fully lytic (e.g. beta-lactams). Affects OD debris."
                    )
                    antibiotics[i]["inoculum_effect_constant"] = st.number_input(
                        "Inoculum Effect Constant (K_inoc)",
                        value=float(antibiotics[i].get("inoculum_effect_constant", 0.0)),
                        format="%.1e",
                        key=f"abx_inoc_{i}",
                        help="Bacterial density (cells/mL) at which antibiotic EC50 doubles. Set to 0 to disable."
                    )
                    
                    antibiotics[i]["Km_elim"] = st.number_input(
                        "Nonlinear elimination Km",
                        value=float(antibiotics[i].get("Km_elim", 0.0)),
                        format="%.1e",
                        key=f"abx_km_elim_{i}",
                        help="Michaelis-Menten central elimination saturation. Set to 0 to disable."
                    )

        # Host Immunity
        with col2:
            st.markdown("### 🛡️ Host Immunity")

            st.session_state["int_immunity_enabled"] = st.checkbox(
                "Enable Immune System Module",
                value=st.session_state.get("int_immunity_enabled", False),
            )

            if st.session_state["int_immunity_enabled"]:
                _valid_modules = ["innate", "hill"]
                _cur_module = st.session_state.get("int_immune_module", "innate")
                if _cur_module not in _valid_modules:
                    _cur_module = "innate"
                st.session_state["int_immune_module"] = st.selectbox(
                    "Immune module",
                    _valid_modules,
                    index=_valid_modules.index(_cur_module),
                    help=(
                        "innate: Imm is integrated (dImm/dt = stim_rate·B/(stim50+B) "
                        "− decay·Imm) and killing ∝ imm_kill_rate·Imm.  |  "
                        "hill: Imm is frozen; killing = imm_max·B/(imm_kill50+B) directly "
                        "(imm_stim_rate / imm_kill_rate / imm_decay_rate / initial_Imm unused)."
                    ),
                )
                _module = st.session_state["int_immune_module"]

                if _module == "innate":
                    imm_col1, imm_col2 = st.columns(2)
                    with imm_col1:
                        st.session_state["int_imm_stim_rate"] = st.number_input(
                            "Stimulation rate (imm_stim_rate)",
                            value=float(st.session_state.get("int_imm_stim_rate", 0.1)),
                            format="%.2e",
                            help="Rate at which each bacterium recruits immune effectors.",
                        )
                        st.session_state["int_innate_kill_rate"] = st.number_input(
                            "Kill rate coefficient (imm_kill_rate)",
                            value=float(st.session_state.get("int_innate_kill_rate", 1e7)),
                            format="%.1e",
                            help="Per-bacterium immune killing coefficient.",
                        )
                        st.session_state["int_innate_decay_rate"] = st.number_input(
                            "Effector decay rate (imm_decay_rate)",
                            value=float(st.session_state.get("int_innate_decay_rate", 0.1)),
                            step=0.01,
                        )
                    with imm_col2:
                        st.session_state["int_imm_stim50"] = st.number_input(
                            "Stimulation half-sat. (imm_stim50)",
                            value=float(st.session_state.get("int_imm_stim50", 1e6)),
                            format="%.1e",
                            help="Bacterial density at half-max immune stimulation.",
                        )
                        st.session_state["int_innate_kill50"] = st.number_input(
                            "Killing half-sat. (imm_kill50)",
                            value=float(st.session_state.get("int_innate_kill50", 1e5)),
                            format="%.1e",
                            help="Bacterial density at half-max immune killing.",
                        )
                        st.session_state["int_imm_initial"] = st.number_input(
                            "Initial immune density (initial_Imm)",
                            value=float(st.session_state.get("int_imm_initial", 0.0)),
                            format="%.1e",
                            help="Starting immune effector level. Typically 0 — grows from bacterial stimulation.",
                        )
                else:  # hill
                    st.caption(
                        "**Hill module:** immune killing = `imm_max · B / (imm_kill50 + B_total)`. "
                        "The `Imm` state is *frozen* (not integrated), so `imm_stim_rate`, "
                        "`imm_kill_rate`, `imm_decay_rate` and `initial_Imm` have no effect here — "
                        "only the two parameters below (plus `imm_kill_rate_D`) apply."
                    )
                    imm_col1, imm_col2 = st.columns(2)
                    with imm_col1:
                        st.session_state["int_innate_max"] = st.number_input(
                            "Max clearance rate (imm_max)",
                            value=float(st.session_state.get("int_innate_max", 1e7)),
                            format="%.1e",
                            help=(
                                "Maximum immune clearance flux (cells/mL/h) at saturating bacterial "
                                "density: killing = imm_max·B/(imm_kill50+B). Set comparable to the "
                                "bacterial growth flux (~growth_rate × bacterial load)."
                            ),
                        )
                    with imm_col2:
                        st.session_state["int_innate_kill50"] = st.number_input(
                            "Killing half-sat. (imm_kill50)",
                            value=float(st.session_state.get("int_innate_kill50", 1e5)),
                            format="%.1e",
                            help="Bacterial density at half-max immune killing.",
                        )

                st.session_state["int_imm_kill_rate_D"] = st.number_input(
                    "Kill rate for dormant/hibernating cells (imm_kill_rate_D)",
                    value=float(st.session_state.get("int_imm_kill_rate_D", 0.0)),
                    format="%.1e",
                    help="Set > 0 to allow immune clearance of dormant compartments.",
                )

                # Immune-refuge warning: dormant cells are immune-privileged unless
                # imm_kill_rate_D > 0. With dormancy on, a resistant/persister
                # population can hide in the dormant compartment — the immune system
                # neither kills it (imm_kill_rate_D=0) nor is stimulated by it
                # (dormant cells are excluded from the immune signal). The infection
                # then never clears even though immunity is "on", which is a common
                # source of confusion.
                _dormancy_on = any(
                    s.get("dormancy_enabled", False)
                    for s in st.session_state.get("int_strains", [])
                )
                if _dormancy_on and st.session_state["int_imm_kill_rate_D"] <= 0:
                    st.warning(
                        "⚠️ **Dormancy + immunity:** dormant/hibernating cells are "
                        "immune-privileged while `imm_kill_rate_D = 0` — the immune "
                        "system will not kill them and they do not stimulate it. A "
                        "phage-resistant (or persister) population can survive in the "
                        "dormant reservoir, so the infection may never clear and the "
                        "resistant fraction can stay near 100% even with immunity "
                        "enabled. Set `imm_kill_rate_D > 0` above to let immunity "
                        "clear dormant cells."
                    )

    # ──── Tab 3: Environment & Dosing ─────────────────────────────────────────
    with config_tabs[2]:
        col1, col2 = st.columns(2)

        # Environment & Debris
        with col1:
            st.markdown("### 🍎 Nutrient environment")
            st.caption("The **growth signal function** (and its Monod constant / carrying capacity) "
                       "is set under **Strains & Phages → Growth model**. These are the medium/reactor "
                       "conditions for nutrient-tracking growth.")
            if st.session_state.get("int_track_nutrients", True):
                st.session_state["int_initial_S"] = st.number_input(
                    "Initial Resource Substrate (S0)",
                    value=float(st.session_state.get("int_initial_S", 1.0)), step=0.1,
                )
                st.session_state["int_recycle_fraction"] = st.number_input(
                    "Nutrient recycling fraction",
                    value=float(st.session_state.get("int_recycle_fraction", 0.0)),
                    min_value=0.0, max_value=1.0, step=0.1,
                )
                st.session_state["int_s_in"] = st.number_input(
                    "Continuous medium inflow (s_in)",
                    value=float(st.session_state.get("int_s_in", 0.0)), step=0.1,
                )
                st.session_state["int_s_out"] = st.number_input(
                    "Continuous Washout dilution (s_out)",
                    value=float(st.session_state.get("int_s_out", 0.0)), step=0.05,
                )
            else:
                st.info("The selected growth signal is nutrient-independent (constant / density), "
                        "so nutrient substrate dynamics are inactive.")

            st.markdown("### 🎚️ Optical Density (OD) & Debris")
            st.session_state["int_debris_enabled"] = st.checkbox(
                "Track Bacteriolytic Cell Debris",
                value=st.session_state.get("int_debris_enabled", False),
            )

            if st.session_state["int_debris_enabled"]:
                st.session_state["int_debris_u"] = st.number_input(
                    "Scattering weight for intact dead cells (u)",
                    value=float(st.session_state.get("int_debris_u", 0.4)),
                    step=0.1,
                )
                st.session_state["int_debris_v"] = st.number_input(
                    "Scattering weight for lysed cell fragments (v)",
                    value=float(st.session_state.get("int_debris_v", 0.2)),
                    step=0.1,
                )
                st.session_state["int_debris_kdis"] = st.number_input(
                    "Debris dissolution rate (k_dis)",
                    value=float(st.session_state.get("int_debris_kdis", 0.01)),
                    step=0.05,
                )
                st.session_state["int_od_to_cfu_conversion_factor"] = st.number_input(
                    "OD-to-CFU conversion factor",
                    value=float(st.session_state.get("int_od_to_cfu_conversion_factor", 2e8)),
                    format="%.1e",
                )

        # Dosing Schedule
        with col2:
            st.markdown("### 📅 Dosing Schedule & Regimens")

            sub_col1, sub_col2 = st.columns(2)

            with sub_col1:
                st.markdown("#### Active Dosing Events")
                if doses:
                    st.caption("Edit time / amount / route inline; 🗑️ removes a row. "
                               "To change the target, delete and re-add.")
                _routes = ["bolus", "infusion"]
                for idx, dose in enumerate(doses):
                    dose.setdefault("_id", _next_uid("dose"))  # stable key across reorder/delete
                    _did = dose["_id"]
                    c_t1, c_t2, c_t3, c_t4 = st.columns([2, 3, 3, 1])
                    with c_t1:
                        dose["time"] = st.number_input(
                            "Time (h)", min_value=0.0, value=float(dose.get("time", 0.0)),
                            step=1.0, key=f"dose_time_{_did}", label_visibility="collapsed")
                    with c_t2:
                        dose["amount"] = st.number_input(
                            f"Amount → {dose['target_type']} #{dose['target_idx']}",
                            min_value=0.0, value=float(dose.get("amount", 0.0)), format="%.2e",
                            key=f"dose_amt_{_did}", label_visibility="collapsed")
                        st.caption(f"→ {dose['target_type']} #{dose['target_idx']}")
                    with c_t3:
                        dose["route"] = st.selectbox(
                            "Route", _routes, index=_routes.index(dose.get("route", "bolus")),
                            key=f"dose_route_{_did}", label_visibility="collapsed")
                        if dose["route"] == "infusion":
                            dose["duration"] = st.number_input(
                                "Infusion duration (h)", min_value=0.1,
                                value=float(dose.get("duration") or 2.0), step=0.5,
                                key=f"dose_dur_{_did}")
                    with c_t4:
                        if st.button("🗑️", key=f"del_dose_{_did}"):
                            doses.pop(idx)
                            st.session_state.int_doses = doses
                            st.rerun()
                st.session_state.int_doses = doses  # persist inline edits

                st.markdown("#### Add Single Dose Event")
                with st.expander("➕ Define Single Dosing Event"):
                    # Build target options
                    target_ops = ["phage"]
                    if len(antibiotics) > 0:
                        target_ops.append("antibiotic")
                    target_ops.append("nutrient")

                    d_type = st.selectbox("Target Compartment", target_ops)

                    d_time = st.number_input("Time (hours)", min_value=0.0, value=0.0, step=1.0)
                    # Target-appropriate default amount (1e8 makes sense only for phage).
                    # Keyed by target so switching target suggests a sensible default.
                    d_amount = st.number_input(
                        DOSE_AMOUNT_LABELS[d_type],
                        min_value=0.0,
                        value=DOSE_AMOUNT_DEFAULTS[d_type],
                        format="%.1e",
                        key=f"single_dose_amount_{d_type}",
                        help="Phage in PFU (e.g. 1e8); antibiotic in mg (e.g. 10); nutrient in resource units.",
                    )

                    d_idx = 0
                    if d_type == "phage" and len(phages) > 1:
                        d_idx = st.selectbox("Phage Target index", list(range(len(phages))))
                    elif d_type == "antibiotic" and len(antibiotics) > 1:
                        d_idx = st.selectbox("Antibiotic Target index", list(range(len(antibiotics))))

                    d_route = st.selectbox("Administration Route", ["bolus", "infusion"])
                    d_dur = 0.0
                    if d_route == "infusion":
                        d_dur = st.number_input("Infusion Duration (hours)", min_value=0.1, value=2.0, step=0.5)

                    if st.button("➕ Add Dose Event"):
                        doses.append(
                            {
                                "time": d_time,
                                "amount": d_amount,
                                "target_type": d_type,
                                "target_idx": d_idx,
                                "route": d_route,
                                "duration": d_dur,
                            }
                        )
                        st.session_state.int_doses = doses
                        st.success("Dose event appended successfully!")
                        st.rerun()

            with sub_col2:
                st.markdown("#### Add Repeat Dosing Regimen")
                with st.expander("➕ Define Repeat Dosing Regimen"):
                    # Target options
                    target_ops_rep = ["phage"]
                    if len(antibiotics) > 0:
                        target_ops_rep.append("antibiotic")
                    target_ops_rep.append("nutrient")
                    r_type = st.selectbox("Target Compartment", target_ops_rep, key="rep_dose_type")

                    r_amount = st.number_input(
                        DOSE_AMOUNT_LABELS[r_type],
                        min_value=0.0,
                        value=DOSE_AMOUNT_DEFAULTS[r_type],
                        format="%.1e",
                        key=f"rep_dose_amount_{r_type}",
                        help="Phage in PFU (e.g. 1e8); antibiotic in mg (e.g. 10); nutrient in resource units.",
                    )
                    r_interval = st.number_input("Interdose Interval (hours)", min_value=1.0, value=12.0, step=1.0, key="rep_dose_interval")
                    r_count = st.number_input("Number of Repeats", min_value=1, value=4, step=1, key="rep_dose_count")
                    r_start = st.number_input("Start Time (hours)", min_value=0.0, value=0.0, step=1.0, key="rep_dose_start")

                    r_idx = 0
                    if r_type == "phage" and len(phages) > 1:
                        r_idx = st.selectbox("Phage Target index", list(range(len(phages))), key="rep_dose_phage_idx")
                    elif r_type == "antibiotic" and len(antibiotics) > 1:
                        r_idx = st.selectbox("Antibiotic Target index", list(range(len(antibiotics))), key="rep_dose_abx_idx")

                    r_route = st.selectbox("Administration Route", ["bolus", "infusion"], key="rep_dose_route")
                    r_dur = 0.0
                    if r_route == "infusion":
                        r_dur = st.number_input("Infusion Duration (hours)", min_value=0.1, value=2.0, step=0.5, key="rep_dose_duration")

                    if st.button("➕ Add Repeat Regimen", key="rep_dose_add_btn"):
                        for k in range(int(r_count)):
                            doses.append({
                                "time": r_start + k * r_interval,
                                "amount": r_amount,
                                "target_type": r_type,
                                "target_idx": r_idx,
                                "route": r_route,
                                "duration": r_dur,
                            })
                        st.session_state.int_doses = doses
                        st.success(f"Added {r_count} repeat dose events successfully!")
                        st.rerun()


    # ──── Tab 4: Solver Settings ──────────────────────────────────────────────
    with config_tabs[3]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🕒 Solver Time parameters")
            st.session_state["int_t_end"] = st.number_input(
                "Simulation end time (hours)",
                value=float(st.session_state.get("int_t_end", 48.0)),
                step=12.0,
            )
            st.session_state["int_dt"] = st.number_input(
                "ODE output interval (dt)",
                value=float(st.session_state.get("int_dt", 0.25)),
                step=0.05,
            )

            st.markdown("### 🧬 Model structure")
            st.session_state["int_n_latent"] = st.number_input(
                "Latency compartments (n_latent)",
                min_value=1, max_value=20,
                value=int(st.session_state.get("int_n_latent", 5)),
                help="Number of latent-infection stages (Erlang-distributed latent period). "
                     "Applies to all builder modes (Direct / BRG / Custom Strains).",
            )

            st.markdown("### 🧩 Advanced solver options")
            st.session_state["int_superinfection"] = st.checkbox(
                "Allow Phage Superinfection",
                value=st.session_state.get("int_superinfection", False),
                help="If active, phages will adsorb to latently-infected cells without producing extra burst."
            )
            
            st.session_state["int_t_prerun"] = st.number_input(
                "Pre-treatment prerun pre-growth (hours)",
                value=float(st.session_state.get("int_t_prerun", 0.0)),
                step=4.0,
                help="Let the bacteria/nutrient system equilibrate without treatments before t=0. Set to 0 to disable."
            )
            if st.session_state.get("int_t_prerun", 0.0) > 0 and st.session_state.get("int_debris_enabled", False):
                st.session_state["int_prerun_inherit_debris"] = st.checkbox(
                    "Inherit bacterial debris from the pre-run",
                    value=bool(st.session_state.get("int_prerun_inherit_debris", True)),
                    key="widget_prerun_inherit_debris",
                    help="On (default): the OD/debris that built up during the pre-run carries into "
                         "treatment (t=0 starts with a realistic dead-cell background). Off: the dead "
                         "cells are washed out — treatment starts with zero debris.")

        with col2:
            st.markdown("### 🎛️ ODE solver specifics")
            st.session_state["int_extinction_threshold"] = st.number_input(
                "Absorbing Extinction threshold",
                value=float(st.session_state.get("int_extinction_threshold", 1.0)),
                step=1.0,
                help="If density falls below this threshold, it is locked to 0 to prevent numerical recovery.",
            )
            st.session_state["int_extinction_check_interval"] = st.number_input(
                "Extinction check interval (hours)",
                value=float(st.session_state.get("int_extinction_check_interval", 0.0)),
                min_value=0.0,
                step=1.0,
                help=(
                    "Apply the extinction threshold every this many hours (not just at "
                    "dose events). Zeroes any sub-threshold strain before it can regrow "
                    "from a below-threshold pool. Set to 0 to check only at dose boundaries. "
                    "Ignored when the extinction threshold is 0."
                ),
            )
            st.session_state["int_solver_method"] = st.selectbox(
                "ODE Solver integration method",
                ["BDF", "Radau", "LSODA"],
                index=["BDF", "Radau", "LSODA"].index(
                    st.session_state.get("int_solver_method", "BDF")
                ),
            )
            st.session_state["int_phage_noise_floor"] = st.number_input(
                "Phage noise floor suppression",
                value=float(st.session_state.get("int_phage_noise_floor", 1e-9)),
                help="Suppress floating-point round-off of non-dosed phages. Set <= 0 to disable.",
            )

    # ──── Run Button ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Run Simulation", width="stretch"):
        with st.spinner("Assembling model equations & integrating..."):
            try:
                _pc_cfg, _pc_B0, *_ = build_nominal_config_from_gui()
                warn_if_prerun_collapses(_pc_cfg, _pc_B0)
                result, config = run_sim_from_gui_params()
                st.session_state.simulation_result = result
                st.session_state.simulation_config = config
                st.success("Simulation finished successfully!")
            except Exception as e:
                st.error(f"Solver Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ──── Outputs / Results Section ───────────────────────────────────────────
    if st.session_state.simulation_result is not None:
        result = st.session_state.simulation_result
        config = st.session_state.simulation_config

        st.markdown("## 📊 Simulation Results")

        # 1. Calculate Metrics
        total_bacteria = result.sum_prefixes("B", "D", "I", "H")
        nadir_val = np.min(total_bacteria)
        nadir_time = result.time[np.argmin(total_bacteria)]

        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        auc_val = _trapz(total_bacteria, result.time)

        # Clearance & Log Reduction
        t_clear = time_to_clearance(
            result,
            threshold=st.session_state.get("int_extinction_threshold", 1.0),
        )
        t_log_red = time_to_log_reduction(result, n_logs=2.0)

        # Render Metrics in Columns
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Bacterial Nadir</div>
                    <div class="metric-value">{nadir_val:.2e}</div>
                    <div class="metric-sub">cells/mL at t={nadir_time:.1f}h</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Bacterial AUC</div>
                    <div class="metric-value">{auc_val:.1e}</div>
                    <div class="metric-sub">cells·h/mL</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            clear_lbl = f"{t_clear:.1f}h" if t_clear is not None else "Never"
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Time to Clearance</div>
                    <div class="metric-value">{clear_lbl}</div>
                    <div class="metric-sub">threshold={st.session_state.get('int_extinction_threshold', 1.0)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col4:
            log_lbl = f"{t_log_red:.1f}h" if t_log_red is not None else "Never"
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">2-Log Reduction Time</div>
                    <div class="metric-value">{log_lbl}</div>
                    <div class="metric-sub">from baseline B(0)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col5:
            # show final resistant fraction
            total_active = result.sum_prefixes("B")
            final_res_frac = (total_active[-1] - result.get("B0")[-1]) / total_active[-1] if (total_active[-1] > 0 and len(strains) > 1) else 0.0
            st.markdown(
                f"""
                <div class="metric-container">
                    <div class="metric-label">Resistant fraction</div>
                    <div class="metric-value">{final_res_frac*100:.1f}%</div>
                    <div class="metric-sub">at t_end={result.time[-1]}h</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Plot Tabs
        plot_tabs = st.tabs(
            [
                "📈 Bacterial Dynamics",
                "🧬 Phage Dynamics",
                "🍎 Nutrients & OD",
                "💊 Antibiotics & Immunity",
            ]
        )

        t = result.time

        # Bacterial Dynamics Plot
        with plot_tabs[0]:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.semilogy(
                t, np.maximum(total_bacteria, 1.0), "k-", lw=3, label="Total Viable"
            )
            
            # Map genotype labels if BRG
            if builder_mode == "Binary Genotypes (BRG)":
                import itertools
                combs = list(itertools.product([0, 1], repeat=len(phages) + len(antibiotics)))
                labels = []
                for comb in combs:
                    if len(antibiotics) == 0:
                        lbl = "".join(map(str, comb))
                    else:
                        p_lbl = "".join(map(str, comb[:len(phages)])) if len(phages) > 0 else ""
                        a_lbl = "".join(map(str, comb[len(phages):]))
                        lbl = f"phi{p_lbl}_abx{a_lbl}" if len(phages) > 0 else f"abx{a_lbl}"
                    labels.append(lbl)
                    
                for j in range(len(labels)):
                    ax.semilogy(
                        t, np.maximum(result.get(f"B{j}"), 1.0), "--", label=labels[j]
                    )
            else:
                for j in range(len(strains)):
                    name = strains[j]["name"]
                    ax.semilogy(
                        t, np.maximum(result.get(f"B{j}"), 1.0), "--", label=f"{name} (Active)"
                    )
                    if strains[j].get("dormancy_enabled", False):
                        D_total = np.zeros_like(t)
                        for q in range(strains[j].get("dormancy_depth", 1)):
                            D_total += result.get(f"D{q}_{j}")
                        ax.semilogy(
                            t, np.maximum(D_total, 1.0), ":", label=f"{name} (Dormant)"
                        )
            ax.set(
                xlabel="Time (hours)",
                ylabel="Density (cells/mL)",
                title="Bacterial Population Trajectories",
            )
            ax.legend(fontsize=9, loc="lower left")
            ax.grid(True, which="both", ls="-", alpha=0.1)
            st.pyplot(fig)
            plt.close(fig)

        # Phage Dynamics Plot
        with plot_tabs[1]:
            if len(phages) > 0:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                for j in range(len(phages)):
                    name = phages[j]["name"]
                    ax.semilogy(
                        t,
                        np.maximum(result.get(f"P{j}"), 1.0),
                        "-",
                        label=f"{name} (Infection Site)",
                    )
                    # blood Pc
                    if phages[j]["pk_mode"] != "None":
                        Vc = phages[j].get("Vc", 5000.0)
                        ax.semilogy(
                            t,
                            np.maximum(result.get(f"Pc{j}") / Vc, 1.0),
                            "--",
                            label=f"{name} (Blood Conc)",
                        )
                ax.set(
                    xlabel="Time (hours)",
                    ylabel="Density (phages/mL)",
                    title="Phage Population Trajectories",
                )
                ax.legend(fontsize=9, loc="lower left")
                ax.grid(True, which="both", ls="-", alpha=0.1)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No phages were configured in this simulation.")

        # Nutrients & OD Plot
        with plot_tabs[2]:
            col_nut1, col_nut2 = st.columns(2)
            with col_nut1:
                if track_nutrients:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(t, result.get("S"), "C1-", lw=2, label="Substrate (S)")
                    ax.set(
                        xlabel="Time (hours)",
                        ylabel="Substrate concentration",
                        title="Nutrient Resource Depletion",
                    )
                    ax.legend()
                    ax.grid(True, alpha=0.15)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Nutrient tracking is disabled (constant/logistic growth).")

            with col_nut2:
                if st.session_state.get("int_debris_enabled", False):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(
                        t,
                        _safe_od(result, total_bacteria),
                        "C2-",
                        lw=2.5,
                        label="OD (AU)",
                    )
                    cfu_od = total_bacteria / st.session_state.get(
                        "int_od_to_cfu_conversion_factor", 2e8
                    )
                    ax.plot(
                        t,
                        cfu_od,
                        "g--",
                        alpha=0.6,
                        label="Live-only OD",
                    )
                    ax.set(
                        xlabel="Time (hours)",
                        ylabel="Optical Density (AU)",
                        title="Simulated Optical Density (Live + Debris)",
                    )
                    ax.legend()
                    ax.grid(True, alpha=0.15)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(
                        "Bacterial debris & Optical Density (OD) tracking was not enabled."
                    )

        # Antibiotics & Host Immunity Plot
        with plot_tabs[3]:
            abx_present = len(antibiotics) > 0
            imm_present = st.session_state.get("int_immunity_enabled", False)

            if abx_present or imm_present:
                fig, ax1 = plt.subplots(figsize=(10, 4.5))

                if abx_present:
                    color = "#3b82f6"
                    ax1.set_xlabel("Time (hours)")
                    ax1.set_ylabel("Antibiotic Concentration", color=color)
                    for j, abx in enumerate(antibiotics):
                        Vc = abx.get("Vc", 1.0)
                        conc = result.get(f"Ac{j}") / Vc
                        ax1.plot(
                            t,
                            conc,
                            color=color,
                            lw=2,
                            label=f"{abx['name']} (Blood)",
                        )
                        if abx.get("k12", 0.0) > 0:
                            ax1.plot(
                                t,
                                result.get(f"Ap{j}") / Vc,
                                color=color,
                                ls="--",
                                alpha=0.7,
                                label=f"{abx['name']} (Peripheral)",
                            )
                    ax1.tick_params(axis="y", labelcolor=color)

                if imm_present:
                    # When an antibiotic is present, put Imm on a twin y-axis;
                    # otherwise plot it directly on ax1 (the displayed figure).
                    # NOTE: previously this used plt.subplots(...)[1] in the no-abx
                    # case, which drew Imm onto a throwaway figure while st.pyplot(fig)
                    # showed the empty original — so the graph appeared blank whenever
                    # immunity was enabled without an antibiotic.
                    ax2 = ax1.twinx() if abx_present else ax1
                    color = "#ec4899"
                    ax2.set_ylabel("Immune Effector Cells (Imm)", color=color)
                    ax2.plot(
                        t,
                        result.get("Imm"),
                        color=color,
                        lw=2,
                        label="Immune Effector",
                    )
                    ax2.tick_params(axis="y", labelcolor=color)
                    if not abx_present:
                        ax2.set_xlabel("Time (hours)")

                plt.title("Pharmacokinetics & Host Defense Dynamics")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No antibiotics or immune modules were configured.")

        # 3. Export Code & Data
        st.markdown("### 📤 Export & Reproducibility")

        c_down1, c_down2 = st.columns(2)
        with c_down1:
            # Generate CSV data
            df_export = pd.DataFrame({"time": t})
            for col in result.state_names:
                df_export[col] = result.get(col)
            df_export["CFU_total"] = total_bacteria

            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()

            st.download_button(
                "📥 Download Simulation Trajectories (CSV)",
                data=csv_str,
                file_name="pbisim_simulation_results.csv",
                mime="text/csv",
                width="stretch",
            )

        with c_down2:
            rep_code = generate_reproduction_code()
            st.download_button(
                "📥 Download Python Script",
                data=rep_code,
                file_name="pbisim_run.py",
                mime="text/x-python",
                width="stretch",
            )

        if show_code:
            with st.expander("🐍 View Python Reproduction Code"):
                st.code(rep_code, language="python")
