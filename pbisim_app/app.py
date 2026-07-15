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
                          default_on=True):
    """Render dose-regimen widgets for one agent and return a config dict.

    Returns ``{"on": False}`` when the agent is not included in this arm, else
    ``{"on": True, "index", "amount", "start", "repeat", "interval", "n"}``.
    """
    on = st.checkbox(f"Include {target}", value=default_on, key=f"{prefix}_on")
    if not on:
        return {"on": False}
    idx = 0
    if len(items) > 1:
        idx = st.selectbox(
            f"{target.title()} target", range(len(items)),
            format_func=lambda i: items[i].get("name", f"{target} {i}"),
            key=f"{prefix}_idx",
        )
    c1, c2 = st.columns(2)
    with c1:
        amount = st.number_input(unit_label, min_value=0.0, value=default_amount,
                                 format="%.1e", key=f"{prefix}_amt")
    with c2:
        start = st.number_input("Start (h)", min_value=0.0, value=0.0, step=1.0,
                                key=f"{prefix}_start")
    repeat = st.checkbox("Repeat regimen (qX h × N)", value=False, key=f"{prefix}_rep")
    interval, n_doses = 8.0, 1
    if repeat:
        c3, c4 = st.columns(2)
        with c3:
            interval = st.number_input("Interval (h)", min_value=0.5, value=8.0,
                                       step=1.0, key=f"{prefix}_int")
        with c4:
            n_doses = st.number_input("Doses", min_value=1, value=4, step=1,
                                      key=f"{prefix}_n")
    return {"on": True, "index": int(idx), "amount": float(amount),
            "start": float(start), "repeat": bool(repeat),
            "interval": float(interval), "n": int(n_doses)}


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

if theme_mode == "Light":
    css_content = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        color: #0f172a !important;
    }
    
    /* Force light colors on text labels */
    .stApp p, .stApp span, .stApp li, .stApp label, .stApp label p, .stApp [data-testid="stWidgetLabel"] p {
        color: #1e293b !important;
    }
    
    input, select, textarea, [data-baseweb="select"] div {
        color: #0f172a !important;
    }
    
    h1, h2, h3 {
        background: linear-gradient(90deg, #059669 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        backdrop-filter: blur(12px);
    }
    
    .card h4 {
        margin-top: 0;
        color: #0f172a !important;
        font-weight: 700;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        padding-bottom: 8px;
    }
    
    .metric-container {
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.05);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #475569 !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #059669 !important;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b !important;
        margin-top: 2px;
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3) !important;
    }
    
    .preset-card {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        transition: border 0.2s ease;
    }
    .preset-card:hover {
        border-color: rgba(37, 99, 235, 0.5);
    }
    
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: #1e293b !important;
    }
    
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    .info-banner {
        background: rgba(37, 99, 235, 0.08);
        border-left: 4px solid #2563eb;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        color: #1e293b !important;
    }
    """
else:
    css_content = """
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc !important;
    }
    
    .stApp p, .stApp span, .stApp li, .stApp label, .stApp label p, .stApp [data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
    }
    
    input, select, textarea, [data-baseweb="select"] div {
        color: #f8fafc !important;
    }
    
    h1, h2, h3 {
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .card {
        background: rgba(30, 41, 59, 0.45);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    
    .card h4 {
        margin-top: 0;
        color: #f1f5f9 !important;
        font-weight: 700;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }
    
    .metric-container {
        text-align: center;
        background: rgba(15, 23, 42, 0.4);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8 !important;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #10b981 !important;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b !important;
        margin-top: 2px;
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3) !important;
    }
    
    .preset-card {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        transition: border 0.2s ease;
    }
    .preset-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
    }
    
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    .info-banner {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    """

st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


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


def load_preset_to_state(params: dict):
    """Deep copy preset parameters into st.session_state variables."""
    # 0. Clear old simulation results to prevent dimension mismatch crashes
    st.session_state.simulation_result = None
    st.session_state.simulation_config = None
    st.session_state.int_builder_mode = "Direct (ModelBuilder)"

    # 1. Clear any old strain/phage/abx/dose keys to prevent widget collision
    keys_to_clear = [
        k
        for k in st.session_state.keys()
        if any(
            k.startswith(prefix)
            for prefix in ("strain_", "phage_", "abx_", "dose_", "ads_")
        )
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
    st.session_state["int_growth_function"] = params.get("growth_function", "monod_growth")
    st.session_state["int_superinfection"] = params.get("allow_superinfection", False)
    st.session_state["int_t_prerun"] = params.get("t_prerun", 0.0)

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
    st.session_state["int_debris_u"] = params.get("debris_u", 1.0)
    st.session_state["int_debris_v"] = params.get("debris_v", 0.5)
    st.session_state["int_debris_kdis"] = params.get("debris_kdis", 0.1)
    st.session_state["int_od_to_cfu_conversion_factor"] = params.get("od_to_cfu_conversion_factor", 1.0)

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
                "dormancy_depth": s.get("dormancy_depth", 3),
                "dormancy_rate": s.get("dormancy_rate", 0.2),
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
    return obj


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
        {"schema_version": SCENARIO_SCHEMA_VERSION, "scenarios": scenarios},
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
    "bacteria":    {"key": "int_strains",     "max": 4, "label": "Bacteria"},
    "phages":      {"key": "int_phages",      "max": 3, "label": "Phages"},
    "antibiotics": {"key": "int_antibiotics", "max": 2, "label": "Antibiotics"},
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


def build_nominal_config_from_gui():
    """
    Constructs and returns the ModelConfig and corresponding state initial values
    based on the selected builder mode in the GUI.
    """
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
        extra_kwargs["debris_u"] = st.session_state.get("int_debris_u", 1.0)
        extra_kwargs["debris_v"] = st.session_state.get("int_debris_v", 0.5)
        extra_kwargs["debris_kdis"] = st.session_state.get("int_debris_kdis", 0.1)
        extra_kwargs["od_to_cfu_conversion_factor"] = st.session_state.get("int_od_to_cfu_conversion_factor", 1.0)
        
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
            
        event = DoseEvent(
            time=d["time"],
            amount=d["amount"],
            target=target,
            index=target_idx,
            route=d["route"],
            duration=d.get("duration", 0.0)
        )
        dose_events.append(event)
        
    schedule = DoseSchedule(dose_events) if dose_events else None

    # ── BUILDER MODE: Direct (ModelBuilder) ───────────────────────────────────
    if builder_mode == "Direct (ModelBuilder)":
        max_depth = max([s.get("dormancy_depth", 1) for s in strains] if strains else [1])
        builder = ModelBuilder(n_bacteria=n_bacteria, n_phages=n_phages, n_latent=n_latent, n_depth=max_depth)
        
        # Growth rates
        growth_rates = [s["growth_rate"] for s in strains]
        ratios = [s.get("bacteria_to_resource_ratio", 1e9) for s in strains]
        builder = builder.with_growth_rates(growth_rates, bacteria_to_resource_ratio=ratios)
        
        # Natural death rates
        death_rates_B = [s.get("death_rate_B", 0.0) for s in strains]
        death_rates_D = [s.get("death_rate_D", 0.0) for s in strains]
        if any(db > 0 for db in death_rates_B) or any(dd > 0 for dd in death_rates_D):
            builder = builder.with_death(
                death_rate_B=np.array(death_rates_B) if any(db > 0 for db in death_rates_B) else None,
                death_rate_D=np.array(death_rates_D) if any(dd > 0 for dd in death_rates_D) else None,
            )
        
        # Dormancy
        any_dormancy = any(s.get("dormancy_enabled", False) for s in strains)
        if any_dormancy:
            dormancy_rates = [s["dormancy_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
            resus_rates = [s["resuscitation_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
            diff_rates = [s["dormancy_diffusion_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
            enabled_strains = [s for s in strains if s.get("dormancy_enabled", False)]
            ds = enabled_strains[0]["dormancy_signal"] if enabled_strains else "nutrient"
            rs = enabled_strains[0]["resuscitation_signal"] if enabled_strains else "nutrient"
            
            builder = builder.with_dormancy(
                dormancy_rate=np.array(dormancy_rates),
                resuscitation_rate=np.array(resus_rates),
                dormancy_diffusion_rate=np.array(diff_rates),
                dormancy_signal=ds,
                resuscitation_signal=rs
            )
            
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
                
                pk_config = PhagePKConfig(
                    n_phages=n_phages, Vc=vcs, k_elim=k_elims, k_in=k_ins, k_out=k_outs, Vi=vis, Km_elim=kms
                )
                builder = builder.with_phage_pk(pk_config)
                
            # Phage decay nonlinear Km
            phage_decay_Km = np.array([p.get("phage_decay_Km", 0.0) if p.get("phage_decay_Km", 0.0) > 0 else np.inf for p in phages])

            # Dormant-adsorption attenuation with dormancy depth (per phage, broadcast
            # across strains): effective dormant rate = adsorption_dormant * exp(-att * depth).
            attenuation_rate = np.tile(
                np.array([p.get("attenuation_rate", 0.0) for p in phages]), (n_bacteria, 1)
            )

            builder = builder.with_phage_params(
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
                builder = builder.with_pseudolysogeny(hibernation_rate=hib_rates, lytic_resumption_rate=res_rates)
                
        # Mutations. A custom mutation-network graph (any n_bacteria) takes
        # precedence; otherwise fall back to the per-phage-locus shortcut, which
        # pbisim only supports when n_bacteria == 2**n_phages.
        _mut_M = mutation_matrix_from_transitions(st.session_state.get("int_transitions", []), strains)
        if _mut_M is not None:
            builder = builder.with_mutations(mutation_rates=_mut_M)
        elif n_phages > 0 and n_bacteria == 2**n_phages:
            phg_res_rates = st.session_state.get("direct_phg_res_rates", [1e-7] * n_phages)
            builder = builder.with_mutations(phage_resistance_rates=phg_res_rates)
            
        # Antibiotics
        for abx in antibiotics:
            builder = builder.with_antibiotic(
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
            
        # Nutrients
        if not track_nutrients:
            from pbisim import logistic_growth
            builder = builder.with_growth_function(logistic_growth)
            builder = builder.with_nutrient(
                track_nutrients=False,
                carrying_capacity=st.session_state.get("int_carrying_capacity", 1e9)
            )
        else:
            builder = builder.with_nutrient(
                track_nutrients=True,
                monod_constant=st.session_state.get("int_monod_constant", 0.3),
                recycle_fraction=st.session_state.get("int_recycle_fraction", 0.0),
                s_in=st.session_state.get("int_s_in", 0.0),
                s_out=st.session_state.get("int_s_out", 0.0)
            )
            
        # Immunity
        immunity_enabled = st.session_state.get("int_immunity_enabled", False)
        if immunity_enabled:
            kill_rate_D = st.session_state.get("int_imm_kill_rate_D", 0.0)
            builder = builder.with_immunity(
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
            builder = builder.with_dose_schedule(schedule)

        # OD / debris ODE (Direct mode). ModelBuilder.build() takes no kwargs — debris
        # must be configured via with_od_debris(), not passed to build().
        if debris_enabled:
            builder = builder.with_od_debris(
                u=extra_kwargs.get("debris_u", 1.0),
                v=extra_kwargs.get("debris_v", 0.5),
                kdis=extra_kwargs.get("debris_kdis", 0.1),
                od_to_cfu_conversion_factor=extra_kwargs.get("od_to_cfu_conversion_factor", 1.0),
            )

        config = builder.build()

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
        
        dorm_rate = st.session_state.get("int_brg_dorm_rate", 0.2) if dormancy_enabled else 0.0
        resus_rate = st.session_state.get("int_brg_resus_rate", 0.1) if dormancy_enabled else 0.0
        diff_rate = st.session_state.get("int_brg_diff_rate", 0.05) if dormancy_enabled else 0.0
        
        b = BacterialStrain(
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
                PhageStrain(
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
            
        # Build Antibiotics
        abx_strains = []
        for abx in antibiotics:
            abx_strains.append(
                Antibiotic(
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
            
        brg = BinaryResistanceGenotypes.from_strains(
            phage_strains,
            bacteria=b,
            antibiotics=abx_strains if abx_strains else None
        )
        
        # Build config — dormancy depth compartments (configurable; 1 when dormancy off)
        max_depth = int(st.session_state.get("int_brg_n_depth", 3)) if dormancy_enabled else 1

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
            
            phage_pk_config = PhagePKConfig(
                n_phages=n_phages, Vc=vcs, k_elim=k_elims, k_in=k_ins, k_out=k_outs, Vi=vis, Km_elim=kms
            )
            
        # Expose nonlinear clearances
        extra_kwargs["allow_superinfection"] = superinfection
        
        # Add solver-specified non-Monod growth if needed
        if not track_nutrients:
            from pbisim import logistic_growth
            extra_kwargs["growth_function"] = logistic_growth
            extra_kwargs["carrying_capacity"] = st.session_state.get("int_carrying_capacity", 1e9)
            extra_kwargs["track_nutrients"] = False
        else:
            extra_kwargs["monod_constant"] = st.session_state.get("int_monod_constant", 0.3)
            extra_kwargs["recycle_fraction"] = st.session_state.get("int_recycle_fraction", 0.0)
            
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

        config = brg.to_config(
            n_latent=n_latent,
            n_depth=max_depth,
            phage_pk_config=phage_pk_config,
            **extra_kwargs
        )
        
        # Resolve initial densities
        if st.session_state.get("int_brg_use_eq_ic", False):
            total_B = st.session_state.get("int_brg_eq_total_B", 1e7)
            initial_B = brg.equilibrium_initial_condition(total_bacteria=total_B)
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
        ss = StrainSet(n_phages=n_phages)
        
        # Register antibiotics
        for abx in antibiotics:
            ss.add_antibiotic(
                AntibioticDefinition(
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
                sensitivities[abx["name"]] = AntibioticSensitivity(emax=emax_val, ec50=ec50_val)
                
            ss.add_strain(
                StrainDefinition(
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
            ss.set_mutation_graph(graph_dict)
            
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
            
            phage_pk_config = PhagePKConfig(
                n_phages=n_phages, Vc=vcs, k_elim=k_elims, k_in=k_ins, k_out=k_outs, Vi=vis, Km_elim=kms
            )
            
        decay_rates = np.array([p["phage_decay_rates"] for p in phages])
        max_depth = max([s.get("dormancy_depth", 1) for s in strains] if strains else [1])
        
        # Expose extra kwargs
        extra_kwargs["allow_superinfection"] = superinfection
        
        if not track_nutrients:
            from pbisim import logistic_growth
            extra_kwargs["growth_function"] = logistic_growth
            extra_kwargs["carrying_capacity"] = st.session_state.get("int_carrying_capacity", 1e9)
            extra_kwargs["track_nutrients"] = False
            
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

        config = ss.to_config(
            n_latent=n_latent,
            n_depth=max_depth,
            phage_decay_rates=decay_rates,
            imm_decay_rate=st.session_state.get("int_innate_decay_rate", 0.1),
            imm_stim50=st.session_state.get("int_imm_stim50", 1e6),
            imm_kill50=st.session_state.get("int_innate_kill50", 1e5),
            monod_constant=st.session_state.get("int_monod_constant", 0.3),
            recycle_fraction=st.session_state.get("int_recycle_fraction", 0.0),
            phage_pk_config=phage_pk_config,
            immune_module=st.session_state.get("int_immune_module", "innate"),
            imm_max=st.session_state.get("int_innate_max", 1e7),
            **extra_kwargs
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


def generate_reproduction_code() -> str:
    """Generates complete Python script code corresponding to the current state."""
    builder_mode = st.session_state.get("int_builder_mode", "Direct (ModelBuilder)")
    strains = st.session_state.get("int_strains", [])
    phages = st.session_state.get("int_phages", [])
    antibiotics = st.session_state.get("int_antibiotics", [])
    doses = st.session_state.get("int_doses", [])
    _n_latent = int(st.session_state.get("int_n_latent", 5))

    code = []
    code.append("# ── pbisim Auto-Generated Reproduction Script ──")
    code.append("import numpy as np")
    code.append("import matplotlib.pyplot as plt")
    code.append("from pbisim import PBIModel, solve_ode, DoseSchedule, DoseEvent, stationary_phase_ic")
    
    # ──── DIRECT ────
    if builder_mode == "Direct (ModelBuilder)":
        code.append("from pbisim import ModelBuilder")
        code.append("")
        code.append("# 1. Build Model Configuration")
        max_depth = max([s.get("dormancy_depth", 1) for s in strains] if strains else [1])
        code.append(f"builder = ModelBuilder(n_bacteria={len(strains)}, n_phages={len(phages)}, n_latent={_n_latent}, n_depth={max_depth})")
        
        # growth
        rates = [s["growth_rate"] for s in strains]
        ratios = [s.get("bacteria_to_resource_ratio", 1e9) for s in strains]
        code.append(f"builder = builder.with_growth_rates({rates}, bacteria_to_resource_ratio={ratios})")
        
        # natural death
        death_rates_B = [s.get("death_rate_B", 0.0) for s in strains]
        death_rates_D = [s.get("death_rate_D", 0.0) for s in strains]
        if any(db > 0 for db in death_rates_B) or any(dd > 0 for dd in death_rates_D):
            code.append(f"builder = builder.with_death(death_rate_B=np.array({death_rates_B}), death_rate_D=np.array({death_rates_D}))")
            
        # dormancy
        any_dorm = any(s.get("dormancy_enabled", False) for s in strains)
        if any_dorm:
            d_rates = [s["dormancy_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
            r_rates = [s["resuscitation_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
            diff_rates = [s["dormancy_diffusion_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
            code.append(f"builder = builder.with_dormancy(dormancy_rate=np.array({d_rates}), resuscitation_rate=np.array({r_rates}), dormancy_diffusion_rate=np.array({diff_rates}))")
            
        # phages
        if phages:
            ads = []
            ads_dorm = []
            for s_idx in range(len(strains)):
                s_ads = [st.session_state.get(f"ads_{s_idx}_{p_idx}", 1e-8 if s_idx == 0 else 0.0) for p_idx in range(len(phages))]
                s_ads_dorm = [st.session_state.get(f"ads_dorm_{s_idx}_{p_idx}", 0.0) for p_idx in range(len(phages))]
                ads.append(s_ads)
                ads_dorm.append(s_ads_dorm)
            code.append(f"builder = builder.with_phage_params(")
            code.append(f"    adsorption_rates=np.array({ads}),")
            code.append(f"    adsorption_rates_dormant=np.array({ads_dorm}),")
            _burst_2d = [[p['burst_sizes'] for p in phages]] * len(strains)
            _latent_2d = [[p['latent_periods'] for p in phages]] * len(strains)
            code.append(f"    burst_sizes=np.array({_burst_2d}),")
            code.append(f"    latent_periods=np.array({_latent_2d}),")
            code.append(f"    phage_decay_rates=np.array({[p['phage_decay_rates'] for p in phages]}),")
            _atten_2d = [[p.get('attenuation_rate', 0.0) for p in phages]] * len(strains)
            code.append(f"    attenuation_rate=np.array({_atten_2d}),")
            code.append(f"    allow_superinfection={st.session_state.get('int_superinfection', False)}")
            code.append(f")")
            
        # antibiotics
        for abx in antibiotics:
            code.append(f"builder = builder.with_antibiotic(")
            code.append(f"    name='{abx['name']}', k_elim={abx['k_elim']}, Vc={abx['Vc']},")
            code.append(f"    emax={abx['emax']}, ec50={abx['ec50']}, hill={abx['hill']},")
            code.append(f"    f_lyse={abx['f_lyse']}, inoculum_effect_constant={abx['inoculum_effect_constant'] or None}")
            code.append(f")")

        # mutations — custom graph takes precedence over the per-locus shortcut
        _repro_M = mutation_matrix_from_transitions(st.session_state.get("int_transitions", []), strains)
        if _repro_M is not None:
            code.append(f"builder = builder.with_mutations(mutation_rates=np.array({_repro_M.tolist()}))")
        elif phages and len(strains) == 2**len(phages):
            phg_res_rates = st.session_state.get("direct_phg_res_rates", [1e-7] * len(phages))
            code.append(f"builder = builder.with_mutations(phage_resistance_rates={phg_res_rates})")

        # nutrients
        _track_nutrients = st.session_state.get("int_track_nutrients", True)
        if not _track_nutrients:
            code.append(f"builder = builder.with_nutrient(track_nutrients=False, carrying_capacity={st.session_state.get('int_carrying_capacity', 1e9)})")
        else:
            code.append(f"builder = builder.with_nutrient(track_nutrients=True, monod_constant={st.session_state.get('int_monod_constant', 0.3)}, recycle_fraction={st.session_state.get('int_recycle_fraction', 0.0)}, s_in={st.session_state.get('int_s_in', 0.0)}, s_out={st.session_state.get('int_s_out', 0.0)})")

        # immunity
        if st.session_state.get("int_immunity_enabled", False):
            _kD = st.session_state.get("int_imm_kill_rate_D", 0.0)
            _kD_arg = f", imm_kill_rate_D=np.array([{_kD}] * {len(strains)})" if _kD > 0 else ""
            _n = len(strains)
            code.append(
                f"builder = builder.with_immunity("
                f"imm_stim_rate=np.full({_n}, {st.session_state.get('int_imm_stim_rate', 1.0)}), "
                f"imm_stim50={st.session_state.get('int_imm_stim50', 1e6)}, "
                f"imm_kill_rate=np.full({_n}, {st.session_state.get('int_innate_kill_rate', 1e7)}), "
                f"imm_kill50={st.session_state.get('int_innate_kill50', 1e8)}, "
                f"imm_decay_rate={st.session_state.get('int_innate_decay_rate', 0.05)}, "
                f"immune_module='{st.session_state.get('int_immune_module', 'innate')}', "
                f"imm_max={st.session_state.get('int_innate_max', 1e7)}"
                f"{_kD_arg})"
            )

        # OD / debris ODE
        if st.session_state.get("int_debris_enabled", False):
            code.append(
                f"builder = builder.with_od_debris("
                f"u={st.session_state.get('int_debris_u', 1.0)}, "
                f"v={st.session_state.get('int_debris_v', 0.5)}, "
                f"kdis={st.session_state.get('int_debris_kdis', 0.1)}, "
                f"od_to_cfu_conversion_factor={st.session_state.get('int_od_to_cfu_conversion_factor', 1.0)})"
            )

    # ──── BRG ────
    elif builder_mode == "Binary Genotypes (BRG)":
        code.append("from pbisim.strains.genotypes import BinaryResistanceGenotypes, BacterialStrain, PhageStrain, Antibiotic")
        code.append("")
        code.append("# 1. Build Locus Strains")
        code.append("bacteria = BacterialStrain(")
        code.append(f"    base_growth_rate={st.session_state.get('int_brg_base_growth', 1.2)},")
        code.append(f"    bacteria_to_resource_ratio={st.session_state.get('int_brg_base_ratio', 1e9)},")
        code.append(f"    dormancy_rate={st.session_state.get('int_brg_dorm_rate', 0.2) if st.session_state.get('int_brg_dormancy_enabled', False) else 0.0},")
        code.append(f"    resuscitation_rate={st.session_state.get('int_brg_resus_rate', 0.1) if st.session_state.get('int_brg_dormancy_enabled', False) else 0.0},")
        code.append(f"    dormancy_diffusion_rate={st.session_state.get('int_brg_diff_rate', 0.05) if st.session_state.get('int_brg_dormancy_enabled', False) else 0.0},")
        code.append(f"    death_rate_B={st.session_state.get('int_brg_death_rate_B', 0.0) or None},")
        code.append(f"    death_rate_D={st.session_state.get('int_brg_death_rate_D', 0.0) or None if st.session_state.get('int_brg_dormancy_enabled', False) else None}")
        code.append(")")
        code.append("phages = [")
        for p in phages:
            code.append(f"    PhageStrain(name='{p['name']}', adsorption_s={p.get('adsorption_s', 5e-8)}, adsorption_r={p.get('adsorption_r', 0.0)}, burst_size_s={p['burst_sizes']}, latent_period_s={p['latent_periods']}, decay_rate={p['phage_decay_rates']}, fitness_cost={p.get('fitness_cost', 0.0)}, mu={p.get('mu', 1e-7)}),")
        code.append("]")
        code.append("antibiotics = [")
        for abx in antibiotics:
            code.append(f"    Antibiotic(name='{abx['name']}', emax_s={abx['emax']}, ec50_s={abx['ec50']}, emax_r={abx.get('emax_r', abx['emax']*0.1)}, ec50_r={abx.get('ec50_r', abx['ec50']*10.0)}, k_elim={abx['k_elim']}, fitness_cost={abx.get('fitness_cost', 0.0)}, mu={abx.get('mu', 1e-7)}),")
        code.append("]")
        code.append("")
        code.append("brg = BinaryResistanceGenotypes.from_strains(phages, bacteria=bacteria, antibiotics=antibiotics or None)")
        _brg_n_depth = int(st.session_state.get("int_brg_n_depth", 3)) if st.session_state.get("int_brg_dormancy_enabled", False) else 1
        if st.session_state.get("int_brg_use_eq_ic", False):
            _eq_total = st.session_state.get("int_brg_eq_total_B", 1e7)
            code.append(f"initial_B = brg.equilibrium_initial_condition(total_bacteria={_eq_total})")
        else:
            import itertools as _it
            _n_abx_r = len(antibiotics)
            _combs_r = list(_it.product([0, 1], repeat=len(phages) + _n_abx_r))
            _saved = st.session_state.get("int_brg_initial_B", {})
            _ic_brg = []
            for _idx, _comb in enumerate(_combs_r):
                if _n_abx_r == 0:
                    _lbl = "".join(map(str, _comb))
                else:
                    _p = "".join(map(str, _comb[:len(phages)])) if phages else ""
                    _a = "".join(map(str, _comb[len(phages):]))
                    _lbl = f"phi{_p}_abx{_a}" if phages else f"abx{_a}"
                _ic_brg.append(_saved.get(_lbl, 1e7 if _idx == 0 else 0.0))
            code.append(f"initial_B = np.array({_ic_brg})")

        # BRG immunity repro — pass as kwargs to brg.to_config(...)
        _brg_imm_args = ""
        if st.session_state.get("int_immunity_enabled", False):
            _n_brg = 2 ** (len(phages) + len(antibiotics))
            _kD = st.session_state.get("int_imm_kill_rate_D", 0.0)
            _kD_arg = f", imm_kill_rate_D=np.array([{_kD}] * {_n_brg})" if _kD > 0 else ""
            _mod = st.session_state.get("int_immune_module", "innate")
            _brg_imm_args = (
                f", imm_stim_rate=np.full({_n_brg}, {st.session_state.get('int_imm_stim_rate', 1.0)})"
                f", imm_stim50={st.session_state.get('int_imm_stim50', 1e6)}"
                f", imm_kill_rate=np.full({_n_brg}, {st.session_state.get('int_innate_kill_rate', 1e7)})"
                f", imm_kill50={st.session_state.get('int_innate_kill50', 1e8)}"
                f", imm_decay_rate={st.session_state.get('int_innate_decay_rate', 0.05)}"
                f", immune_module='{_mod}'"
                f", imm_max={st.session_state.get('int_innate_max', 1e7)}"
                f"{_kD_arg}"
            )
        _brg_atten = [p.get('attenuation_rate', 0.0) for p in phages]
        _brg_atten_arg = f", attenuation_rate=np.array({_brg_atten})" if phages else ""
        code.append(f"cfg = brg.to_config(n_latent={_n_latent}, n_depth={_brg_n_depth}{_brg_atten_arg}{_brg_imm_args})")

    # ──── STRAINSET ────
    else:
        code.append("from pbisim.strains import StrainDefinition, StrainSet")
        code.append("from pbisim.pk.antibiotic import AntibioticDefinition, AntibioticSensitivity")
        code.append("")
        code.append(f"ss = StrainSet(n_phages={len(phages)})")
        for abx in antibiotics:
            code.append(f"ss.add_antibiotic(AntibioticDefinition('{abx['name']}', k_elim={abx['k_elim']}))")
        for i, s in enumerate(strains):
            ads_rates = [st.session_state.get(f"ads_{i}_{p_idx}", 1e-8 if i == 0 else 0.0) for p_idx in range(len(phages))]
            ads_dorm = [st.session_state.get(f"ads_dorm_{i}_{p_idx}", 0.0) for p_idx in range(len(phages))]
            bursts = [p["burst_sizes"] for p in phages]
            latents = [p["latent_periods"] for p in phages]
            dorm_enabled = s.get("dormancy_enabled", False)
            dorm_rate = s["dormancy_rate"] if dorm_enabled else 0.0
            resus_rate = s["resuscitation_rate"] if dorm_enabled else 0.0
            diff_rate = s["dormancy_diffusion_rate"] if dorm_enabled else 0.0
            db_val = s.get("death_rate_B", 0.0)
            dd_val = s.get("death_rate_D", 0.0)
            code.append(f"        ss.add_strain(StrainDefinition(")
            code.append(f"            name='{s['name']}', growth_rate={s['growth_rate']},")
            code.append(f"            adsorption_rates=np.array({ads_rates}),")
            code.append(f"            adsorption_rates_dormant=np.array({ads_dorm}),")
            code.append(f"            burst_sizes=np.array({bursts}),")
            code.append(f"            latent_periods=np.array({latents}),")
            code.append(f"            latent_periods_dormant=np.array({latents}),")
            code.append(f"            bacteria_to_resource_ratio={s.get('bacteria_to_resource_ratio', 1e9)},")
            code.append(f"            dormancy_rate={dorm_rate}, resuscitation_rate={resus_rate}, dormancy_diffusion_rate={diff_rate},")
            code.append(f"            death_rate_B={db_val if db_val > 0 else None}, death_rate_D={dd_val if dd_val > 0 else None},")
            _imm_on = st.session_state.get("int_immunity_enabled", False)
            _stim_r = st.session_state.get("int_imm_stim_rate", 0.1) if _imm_on else 0.0
            _kill_r = st.session_state.get("int_innate_kill_rate", 1e7) if _imm_on else 0.0
            code.append(f"            imm_stim_rate={_stim_r}, imm_kill_rate={_kill_r}, attenuation_rate=np.array({[p.get('attenuation_rate', 0.0) for p in phages]}),")
            if antibiotics:
                code.append(f"            antibiotic_sensitivity={{")
                for abx in antibiotics:
                    emax_val = abx["emax"] if i == 0 else abx["emax"] * 0.1
                    ec50_val = abx["ec50"] if i == 0 else abx["ec50"] * 10.0
                    code.append(f"                '{abx['name']}': AntibioticSensitivity(emax={emax_val}, ec50={ec50_val}),")
                code.append(f"            }}")
            code.append(f"        ))")
            
        transitions = st.session_state.get("int_transitions", [])
        graph_dict = {t["from"]: {t["to"]: t["rate"]} for t in transitions if t["from"] and t["to"]}
        code.append(f"ss.set_mutation_graph({graph_dict})")
        _ss_decay = [p["phage_decay_rates"] for p in phages]
        _ss_max_depth = max((s.get("dormancy_depth", 3) for s in strains if s.get("dormancy_enabled", False)), default=1)
        _ss_imm_on = st.session_state.get("int_immunity_enabled", False)
        _ss_imm_args = ""
        if _ss_imm_on:
            _ss_imm_args = (
                f", imm_decay_rate={st.session_state.get('int_innate_decay_rate', 0.05)}"
                f", imm_stim50={st.session_state.get('int_imm_stim50', 1e6)}"
                f", imm_kill50={st.session_state.get('int_innate_kill50', 1e8)}"
                f", immune_module='{st.session_state.get('int_immune_module', 'innate')}'"
                f", imm_max={st.session_state.get('int_innate_max', 1e7)}"
            )
        code.append(f"cfg = ss.to_config(n_latent={_n_latent}, n_depth={_ss_max_depth}, phage_decay_rates=np.array({_ss_decay}){_ss_imm_args})")

    # ──── Dosing Schedule ────
    if doses:
        code.append("")
        code.append("# 2. Dosing Schedule")
        code.append("dose_events = [")
        for d in doses:
            code.append(f"    DoseEvent(time={d['time']}, amount={d['amount']}, target='{d['target_type']}', index={d['target_idx']}, route='{d['route']}', duration={d.get('duration', 0.0)}),")
        code.append("]")
        if builder_mode == "Direct (ModelBuilder)":
            code.append("builder = builder.with_dose_schedule(DoseSchedule(dose_events))")
            code.append("cfg = builder.build()")
        else:
            code.append("# Add schedule to brg/ss config manually if required")
    elif builder_mode == "Direct (ModelBuilder)":
        code.append("cfg = builder.build()")

    code.append("")
    code.append("# 3. Initialize Model and Solve")
    
    # Prerun stationary phase
    t_prerun = st.session_state.get("int_t_prerun", 0.0)
    # Determine initial_B representation for the reproduction code
    if builder_mode == "Direct (ModelBuilder)":
        _ic_B_repr = f"np.array({[s['initial_B'] for s in strains]})"
    elif builder_mode == "Binary Genotypes (BRG)":
        _ic_B_repr = "initial_B"   # already emitted above in the BRG block
    else:
        _ic_B_repr = f"np.array({[s['initial_B'] for s in strains]})"
    _ic_P = [p["initial_P"] for p in phages] if phages else [1e6]
    _ic_S = st.session_state.get("int_initial_S", 1.0) if st.session_state.get("int_track_nutrients", True) else 1.0
    _imm_enabled = st.session_state.get("int_immunity_enabled", False)
    _imm_str = f", initial_Imm={st.session_state.get('int_imm_initial', 0.0)}" if _imm_enabled else ""
    _ic_D_vals = [s.get("initial_D", 0.0) for s in strains] if builder_mode == "Direct (ModelBuilder)" else []
    _ic_D_str = f", initial_D=np.array({_ic_D_vals})" if any(v > 0 for v in _ic_D_vals) else ""
    if t_prerun > 0:
        code.append(f"# Run stationary phase equilibration prerun for {t_prerun} hours")
        code.append(f"ic = stationary_phase_ic(cfg, t_prerun={t_prerun})")
        code.append("# Carry the full stationary state — active B, dormant D, nutrient S, immune Imm")
        code.append(f"model = PBIModel(cfg, initial_B=ic.B, initial_P=np.array({_ic_P}), initial_S=max(float(ic.S), 0.0), initial_D=ic.D, initial_Imm=(ic.Imm or 0.0))")
    else:
        code.append(f"model = PBIModel(cfg, initial_B={_ic_B_repr}, initial_P=np.array({_ic_P}), initial_S={_ic_S}{_imm_str}{_ic_D_str})")

    _method = st.session_state.get("int_solver_method", "BDF")
    _thresh = st.session_state.get("int_extinction_threshold", 1.0) or None
    _check_int = st.session_state.get("int_extinction_check_interval", 0.0) or None
    code.append(f"result = solve_ode(model, t_end={st.session_state.get('int_t_end', 48.0)}, dt={st.session_state.get('int_dt', 0.25)}, method='{_method}', extinction_threshold={_thresh}, extinction_check_interval={_check_int})")
    
    code.append("")
    code.append("# 4. Plot trajectories")
    code.append("fig, ax = plt.subplots(figsize=(8, 4))")
    code.append("ax.semilogy(result.time, np.maximum(result.sum_prefixes('B', 'D', 'I', 'H'), 1.0), label='Total Viable')")
    code.append("ax.set(xlabel='Time (h)', ylabel='Density (cells/mL)', title='Simulation Run')")
    code.append("ax.legend()")
    code.append("plt.show()")
    
    return "\n".join(code)


# ── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🦠 pbisim App")

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

    api_key = st.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Required ONLY for the AI Assistant. Local simulation runs entirely offline.",
    )
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
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-8",
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
    st.title("📚 Library")
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
            if st.button("💾 Save scenario", key="sc_save_btn", use_container_width=True):
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
                use_container_width=True,
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
            use_container_width=True, disabled=not _has_parts,
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
                    if st.button("💾 Save part", key=f"part_save_{_cat}", use_container_width=True):
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
    st.title("📐 Calibration — data overlay")
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
            st.dataframe(_raw.head(8), use_container_width=True)
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
            if st.button("📥 Load dataset", key="fit_load", use_container_width=True):
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
                            "od_to_cfu", value=float(st.session_state.get("int_od_to_cfu_conversion_factor", 1e9)),
                            format="%.3e", key="fit_edit_od2cfu",
                            help="CFU per OD unit: OD = (biomass + debris) / od_to_cfu.")
                    with dk2:
                        st.session_state["int_debris_u"] = st.number_input(
                            "Debris yield · deaths (u)", value=float(st.session_state.get("int_debris_u", 1.0)),
                            format="%g", key="fit_edit_debris_u")
                    with dk3:
                        st.session_state["int_debris_v"] = st.number_input(
                            "Debris yield · lysis (v)", value=float(st.session_state.get("int_debris_v", 0.5)),
                            format="%g", key="fit_edit_debris_v")
                    with dk4:
                        st.session_state["int_debris_kdis"] = st.number_input(
                            "Debris dissolution (k_dis)", value=float(st.session_state.get("int_debris_kdis", 0.1)),
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
                                value=int(_s.get("dormancy_depth", 3)), step=1,
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
            if st.button("🔬 Overlay model on data", key="fit_overlay", use_container_width=True):
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
                st.dataframe(pd.DataFrame(_ovr["metrics"]), use_container_width=True, hide_index=True)
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
                if st.button("💾 Save calibrated config as Scenario", key="fit_save_scenario", use_container_width=True):
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
    st.title("💬 AI Simulation Assistant")
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
                if agent_resp.assumptions and show_assumptions:
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
    if prompt := st.chat_input("Ex: simulate 1 wild-type strain and 1 phage with burst size 50 and adsorption 1e-8..."):
        st.chat_message("user").markdown(prompt)

        # Call agent
        agent_resp = None
        initial_history_len = len(st.session_state.agent.history)
        try:
            with st.spinner("Claude is thinking..."):
                agent_resp = st.session_state.agent.ask(prompt)
        except Exception as e:
            st.error(f"❌ AI Assistant Error: {e}")
            st.info("💡 If you are getting a 401 Authentication Error, please verify that your Anthropic API key is correct, active, and has remaining usage credits.")

        if agent_resp is not None:
            exec_result = None
            if agent_resp.code:
                with st.spinner("Executing simulation..."):
                    exec_result = execute_code(agent_resp.code)

                # Self-healing loop
                max_retries = 3
                retry_count = 0
                while not exec_result.success and retry_count < max_retries:
                    retry_count += 1
                    with st.spinner(f"Code execution failed. Attempting self-healing (attempt {retry_count}/{max_retries})..."):
                        healing_prompt = (
                            f"The generated code failed to execute with the following error:\n"
                            f"```text\n{exec_result.error}\n```\n"
                            f"Please correct the code to resolve this error. Ensure you only output the corrected Python code block and necessary narrative/assumptions."
                        )
                        try:
                            agent_resp = st.session_state.agent.ask(healing_prompt)
                            exec_result = execute_code(agent_resp.code)
                        except Exception as e:
                            st.warning(f"Self-healing attempt {retry_count} failed to query the agent: {e}")
                            break

                # Clean up history if execution failed
                if not exec_result.success:
                    del st.session_state.agent.history[initial_history_len:]

            # Render response
            with st.chat_message("assistant"):
                st.markdown(agent_resp.narrative)
                if agent_resp.assumptions and show_assumptions:
                    with st.expander("👁️ Model Assumptions"):
                        st.markdown(agent_resp.assumptions)
                if agent_resp.code and show_code:
                    with st.expander("🐍 Generated python code"):
                        st.code(agent_resp.code, language="python")

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

            st.session_state.history.append(("assistant", (exec_result, agent_resp)))



# ── Clinical Trials & Cohorts Page ────────────────────────────────────────────
elif st.session_state.current_page == "Clinical Trials & Cohorts":
    st.title("👥 Clinical Trials & cohort Simulator")
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

        # Existing arms
        for _ai, _arm in enumerate(list(trial_arms)):
            _lc, _dc = st.columns([6, 1])
            with _lc:
                st.markdown(f"**{_arm['name']}** — {arm_regimen_summary(_arm)}")
            with _dc:
                if st.button("🗑️", key=f"del_arm_{_ai}"):
                    trial_arms.pop(_ai)
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
        
        # Display active IIVs
        trial_iivs = st.session_state.get("trial_iiv_inputs", [])
        
        for idx, iiv in enumerate(trial_iivs):
            col_p, col_d, col_m, col_act = st.columns([3, 3, 2, 1])
            with col_p:
                st.text(f"Path: {iiv['path']}")
            with col_d:
                st.text(f"Dist: {iiv['dist_type']} ({iiv['params']})")
            with col_m:
                st.text(f"Mode: {iiv['mode']}")
            with col_act:
                if st.button("🗑️", key=f"del_iiv_{idx}"):
                    trial_iivs.pop(idx)
                    st.rerun()
                    
        # Add IIV form
        with st.expander("➕ Add Parameter Variability"):
            param_display = st.selectbox("Select Parameter", list(IIV_PARAMETERS.keys()))
            dist_choice = st.selectbox("Distribution Type", ["LogNormal", "Normal", "Uniform"])
            
            c_p1, c_p2 = st.columns(2)
            dist_params = {}
            if dist_choice == "LogNormal":
                with c_p1:
                    cv = st.number_input("CV (coefficient of variation)", value=0.25, min_value=0.01)
                    dist_params["cv"] = cv
                mode = "multiplicative"
            elif dist_choice == "Normal":
                with c_p1:
                    mean = st.number_input("Mean", value=0.0)
                    dist_params["mean"] = mean
                with c_p2:
                    sd = st.number_input("SD (standard deviation)", value=0.1, min_value=0.01)
                    dist_params["sd"] = sd
                mode = "additive"
            else:
                with c_p1:
                    lo = st.number_input("Lower Bound", value=0.5)
                    dist_params["lo"] = lo
                with c_p2:
                    hi = st.number_input("Upper Bound", value=1.5)
                    dist_params["hi"] = hi
                mode = "replace"
                
            if st.button("Add Parameter IIV"):
                trial_iivs.append({
                    "path": IIV_PARAMETERS[param_display],
                    "dist_type": dist_choice,
                    "params": dist_params,
                    "mode": mode
                })
                st.session_state.trial_iiv_inputs = trial_iivs
                st.success(f"Added variability to {param_display}!")
                st.rerun()

        # Run Button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Run Parallel Clinical Trial", use_container_width=True):
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
            st.plotly_chart(fig_cfu, use_container_width=True)
            fig_pfu = plot_pkpd_trajectories_plotly(
                result, prefixes=("P",),
                title="Free Phage (PFU/mL)", y_label="log₁₀ PFU/mL",
            )
            st.plotly_chart(fig_pfu, use_container_width=True)

            # Step survival plot
            st.markdown("#### ⏳ Step-Survival (Kaplan-Meier)")
            fig_km = plot_kaplan_meier_plotly(result, endpoint=endpoint_choice, t_end=trial_t_end, threshold=clearance_threshold, n_logs=2.0)
            st.plotly_chart(fig_km, use_container_width=True)

            # Metric distributions
            st.markdown("#### 📦 Distribution of outcomes")
            fig_dist = plot_metric_distributions_plotly(result, metric=metric_choice)
            st.plotly_chart(fig_dist, use_container_width=True)
            
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
                    use_container_width=True
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
                    use_container_width=True
                )


# ── Dose-Response Sweeps Page ──────────────────────────────────────────────────
elif st.session_state.current_page == "Dose-Response Sweeps":
    st.title("📈 Dose-Response Simulator")
    st.caption("Perform multi-drug dose-response sweeps with MOI scaling, vector padding, and raw time-series visualization.")

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
                    value="1e3, 1e5, 1e7, 1e9",
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
        run_sweep = st.button("🚀 Run Dose-Response Sweep", use_container_width=True)

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
                            _od = (result.get_od() if hasattr(result, "get_od")
                                   else (total_bacteria + result.get("Debris"))
                                   / st.session_state.get("int_od_to_cfu_conversion_factor", 1.0))
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
                use_container_width=True
            )

            st.markdown("#### Raw Simulation Trajectories (Viable Bacteria)")
            fig_traj = go.Figure()
            for t_arr, b_arr, legend_lbl in _dr["trajectories"]:
                fig_traj.add_trace(go.Scatter(x=t_arr, y=np.maximum(b_arr, 1.0), mode='lines', name=legend_lbl))
            fig_traj.update_layout(
                xaxis_title="Time (hours)", yaxis_title="Total Viable Bacteria (CFU/mL)",
                yaxis_type="log", template="plotly_white" if theme_mode == "Light" else "plotly_dark")
            st.plotly_chart(fig_traj, use_container_width=True)

            if _dr["od_trajectories"]:
                st.markdown("#### Raw Simulation Trajectories (Optical Density)")
                fig_od = go.Figure()
                for t_arr, od_arr, legend_lbl in _dr["od_trajectories"]:
                    fig_od.add_trace(go.Scatter(x=t_arr, y=od_arr, mode='lines', name=legend_lbl))
                fig_od.update_layout(
                    xaxis_title="Time (hours)", yaxis_title="Optical density (AU)",
                    template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                st.plotly_chart(fig_od, use_container_width=True)

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
            st.plotly_chart(fig_metrics, use_container_width=True)
        else:
            st.info("Configure the sweep on the left and click **Run Dose-Response Sweep** to view results.")

# ── Parameter Sweeps Page ──────────────────────────────────────────────────────
elif st.session_state.current_page == "Parameter Sweeps":
    st.title("📊 Model Parameter Sweeps")
    st.caption("Sweep any model parameter in 1D or 2D and visualize cellular trajectories and outcome heatmaps.")

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

    sweep_type = st.radio("Sweep Dimension", ["1D Sweep", "2D Sweep"], horizontal=True)

    sweep_params = get_sweep_parameters(nominal_config, strains_gui, phages_gui, antibiotics_gui)
    param_labels = sorted(list(sweep_params.keys()))

    col_setup, col_run = st.columns([1, 2])

    with col_setup:
        st.markdown("### ⚙️ Configure Parameters")
        
        if sweep_type == "1D Sweep":
            param1_label = st.selectbox("Select Parameter", param_labels, key="p1_sweep_label")
            meta1 = sweep_params[param1_label]
            
            # Default values
            default_val = 1e-9
            if meta1["type"] == "scalar":
                default_val = getattr(nominal_config, meta1["field"])
            elif meta1["type"] == "dimension":
                default_val = getattr(nominal_config, meta1["field"])
            elif meta1["type"] == "array1d":
                default_val = getattr(nominal_config, meta1["field"])[meta1["index"]]
            elif meta1["type"] == "array1d_or_none":
                arr = getattr(nominal_config, meta1["field"])
                default_val = arr[meta1["index"]] if arr is not None else 0.0
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
            
            c1, c2, c3 = st.columns(3)
            with c1:
                min_val = st.number_input("Min Value", value=float(default_val * 0.1) if default_val > 0 else 0.0, format="%.2e")
            with c2:
                max_val = st.number_input("Max Value", value=float(default_val * 10.0) if default_val > 0 else 1.0, format="%.2e")
            with c3:
                steps = st.number_input("Steps", min_value=2, max_value=25, value=5)

            spacing = st.selectbox("Spacing", ["Linear", "Logarithmic"])
            run_sweep = st.button("🚀 Run 1D Sweep", use_container_width=True)

        else:
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

            run_sweep = st.button("🚀 Run 2D Sweep", use_container_width=True)

    with col_run:
        st.markdown("### 📊 Sweep Results")
        if run_sweep:
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

                runs_outcomes = []
                trajectories = [] # (time, viable_b, label)

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
                    trajectories.append((result.time, total_bacteria, f"{param1_label} = {val:.2e}"))
                    progress_bar.progress((idx + 1) / len(sweep_values))

                status_text.text("Sweep completed!")
                st.session_state.param_sweep_result = {
                    "type": "1D",
                    "param1_label": param1_label,
                    "spacing": spacing,
                    "summary": runs_outcomes,
                    "trajectories": [(np.asarray(t), np.asarray(b), lbl) for t, b, lbl in trajectories],
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
                    use_container_width=True)
                st.markdown("#### Raw Simulation Trajectories (Viable Bacteria)")
                fig_traj = go.Figure()
                for t_arr, b_arr, legend_lbl in _ps["trajectories"]:
                    fig_traj.add_trace(go.Scatter(x=t_arr, y=np.maximum(b_arr, 1.0), mode='lines', name=legend_lbl))
                fig_traj.update_layout(
                    xaxis_title="Time (hours)", yaxis_title="Total Viable Bacteria (CFU/mL)",
                    yaxis_type="log", template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                st.plotly_chart(fig_traj, use_container_width=True)
                st.markdown("#### Outcome Metrics vs Parameter Value")
                fig_metric = go.Figure()
                fig_metric.add_trace(go.Scatter(x=df_summary[_p1], y=df_summary["AUC (cells·h/mL)"], mode="lines+markers", name="Bacterial AUC", yaxis="y1"))
                fig_metric.add_trace(go.Scatter(x=df_summary[_p1], y=df_summary["Nadir (cells/mL)"], mode="lines+markers", name="Nadir", yaxis="y2"))
                fig_metric.update_layout(
                    xaxis=dict(title=_p1, type="log" if _ps["spacing"] == "Logarithmic" else "linear"),
                    yaxis=dict(title="AUC (cells·h/mL)", type="log"),
                    yaxis2=dict(title="Nadir (cells/mL)", type="log", overlaying="y", side="right"),
                    template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                st.plotly_chart(fig_metric, use_container_width=True)
            else:
                _p1, _p2 = _ps["param1_label"], _ps["param2_label"]
                _xt = "log" if _ps["spacing"] == "Logarithmic" else "linear"
                _yt = "log" if _ps["spacing2"] == "Logarithmic" else "linear"
                st.markdown("#### Outcome Heatmaps (2D Sweep)")
                h1, h2 = st.columns(2)
                with h1:
                    fig_auc = go.Figure(data=go.Contour(z=_ps["grid_auc"], x=_ps["sweep_values1"], y=_ps["sweep_values2"], colorscale="Viridis", colorbar=dict(title="AUC")))
                    fig_auc.update_layout(title="Bacterial AUC Heatmap", xaxis=dict(title=_p1, type=_xt), yaxis=dict(title=_p2, type=_yt), template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                    st.plotly_chart(fig_auc, use_container_width=True)
                with h2:
                    fig_nadir = go.Figure(data=go.Contour(z=_ps["grid_nadir"], x=_ps["sweep_values1"], y=_ps["sweep_values2"], colorscale="Magma", colorbar=dict(title="Nadir")))
                    fig_nadir.update_layout(title="Bacterial Nadir Heatmap", xaxis=dict(title=_p1, type=_xt), yaxis=dict(title=_p2, type=_yt), template="plotly_white" if theme_mode == "Light" else "plotly_dark")
                    st.plotly_chart(fig_nadir, use_container_width=True)
        else:
            st.info("Configure parameters and click **Run Sweep** to start the analysis.")

# ── Interactive Simulator Page ────────────────────────────────────────────────
elif st.session_state.current_page == "Interactive Simulator":
    st.title("🦠 Interactive Simulation Builder")
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
            
        st.markdown("---")
        col1, col2 = st.columns(2)

        # ── DIRECT MODE ──
        if builder_mode == "Direct (ModelBuilder)":
            with col1:
                st.markdown("### 🧫 Bacterial Strains")

                n_strains = st.number_input(
                    "Number of strains", min_value=1, max_value=4, value=len(strains)
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
                                "dormancy_depth": 3,
                                "dormancy_rate": 0.2,
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
                                    value=int(strains[i].get("dormancy_depth", 3)),
                                    key=f"str_depth_{i}",
                                )
                            with cd2:
                                strains[i]["dormancy_rate"] = st.number_input(
                                    "Dormancy rate (sleep)",
                                    value=float(strains[i].get("dormancy_rate", 0.2)),
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
                                ["nutrient", "density", "nutrient_and_density"],
                                index=["nutrient", "density", "nutrient_and_density"].index(strains[i].get("dormancy_signal", "nutrient")),
                                key=f"str_dsig_{i}",
                            )
                            strains[i]["resuscitation_signal"] = st.selectbox(
                                "Resuscitation Signal",
                                ["nutrient", "density", "nutrient_and_density"],
                                index=["nutrient", "density", "nutrient_and_density"].index(strains[i].get("resuscitation_signal", "nutrient")),
                                key=f"str_rsig_{i}",
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
                    "Number of phages", min_value=0, max_value=3, value=len(phages)
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
                        for j in range(n_phages):
                            res_rate = st.number_input(
                                f"Mutation rate to {phages[j]['name']} resistance (mu)",
                                value=float(st.session_state.get(f"direct_phg_mu_{j}", 1e-7)),
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
                        "Dormancy rate (sleep)", value=float(st.session_state.get("int_brg_dorm_rate", 0.2)), step=0.05
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
                        value=int(st.session_state.get("int_brg_n_depth", 3)),
                        help="Number of dormancy-depth compartments for all genotypes.",
                    )
                    st.session_state["int_brg_death_rate_D"] = st.number_input(
                        "Dormant death rate (dD)", value=float(st.session_state.get("int_brg_death_rate_D", 0.0)), step=0.01
                    )
                    
                # Renders the loci count
                st.markdown("---")
                st.markdown("### 🧬 Phage Loci")
                n_phg_loci = st.number_input("Number of phage species (loci)", min_value=1, max_value=3, value=max(len(phages), 1))
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
                
                n_strains = st.number_input("Number of custom strains", min_value=1, max_value=4, value=max(len(strains), 1))
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
                            "dormancy_rate": 0.2, "resuscitation_rate": 0.1, "dormancy_diffusion_rate": 0.05
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
                            strains[i]["dormancy_depth"] = st.number_input("Depth layers (Q)", min_value=1, max_value=10, value=int(strains[i].get("dormancy_depth", 3)), key=f"ss_str_depth_{i}", help="Number of dormancy-depth compartments (max across strains sets the model n_depth).")
                            strains[i]["dormancy_rate"] = st.number_input("Dormancy rate", value=float(strains[i].get("dormancy_rate", 0.2)), key=f"ss_str_sleep_{i}")
                            strains[i]["resuscitation_rate"] = st.number_input("Resuscitation rate", value=float(strains[i].get("resuscitation_rate", 0.1)), key=f"ss_str_wake_{i}")
                            strains[i]["dormancy_diffusion_rate"] = st.number_input("Depth diffusion", value=float(strains[i].get("dormancy_diffusion_rate", 0.05)), key=f"ss_str_diff_{i}")
                            strains[i]["death_rate_D"] = st.number_input("Dormant death rate (dD)", value=float(strains[i].get("death_rate_D", 0.0)), step=0.01, key=f"ss_str_death_d_{i}")

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
                n_phages = st.number_input("Number of phages", min_value=0, max_value=3, value=len(phages))
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
                "Number of antibiotics", min_value=0, max_value=2, value=len(antibiotics)
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
            st.markdown("### 🍎 Nutrients / Substrate")
            st.session_state["int_track_nutrients"] = st.checkbox(
                "Track Nutrients (Monod Kinetics)",
                value=st.session_state.get("int_track_nutrients", True),
            )

            if st.session_state["int_track_nutrients"]:
                st.session_state["int_initial_S"] = st.number_input(
                    "Initial Resource Substrate (S0)",
                    value=float(st.session_state.get("int_initial_S", 1.0)),
                    step=0.1,
                )
                st.session_state["int_monod_constant"] = st.number_input(
                    "Monod Half-saturation Constant (Ks)",
                    value=float(st.session_state.get("int_monod_constant", 0.3)),
                    step=0.05,
                )
                st.session_state["int_recycle_fraction"] = st.number_input(
                    "Nutrient recycling fraction",
                    value=float(st.session_state.get("int_recycle_fraction", 0.0)),
                    min_value=0.0,
                    max_value=1.0,
                    step=0.1,
                )
                st.session_state["int_s_in"] = st.number_input(
                    "Continuous medium inflow (s_in)",
                    value=float(st.session_state.get("int_s_in", 0.0)),
                    step=0.1,
                )
                st.session_state["int_s_out"] = st.number_input(
                    "Continuous Washout dilution (s_out)",
                    value=float(st.session_state.get("int_s_out", 0.0)),
                    step=0.05,
                )
            else:
                st.session_state["int_carrying_capacity"] = st.number_input(
                    "Logistic Carrying Capacity (K)",
                    value=float(st.session_state.get("int_carrying_capacity", 1e9)),
                    format="%.1e",
                )

            st.markdown("### 🎚️ Optical Density (OD) & Debris")
            st.session_state["int_debris_enabled"] = st.checkbox(
                "Track Bacteriolytic Cell Debris",
                value=st.session_state.get("int_debris_enabled", False),
            )

            if st.session_state["int_debris_enabled"]:
                st.session_state["int_debris_u"] = st.number_input(
                    "Scattering weight for intact dead cells (u)",
                    value=float(st.session_state.get("int_debris_u", 1.0)),
                    step=0.1,
                )
                st.session_state["int_debris_v"] = st.number_input(
                    "Scattering weight for lysed cell fragments (v)",
                    value=float(st.session_state.get("int_debris_v", 0.5)),
                    step=0.1,
                )
                st.session_state["int_debris_kdis"] = st.number_input(
                    "Debris dissolution rate (k_dis)",
                    value=float(st.session_state.get("int_debris_kdis", 0.1)),
                    step=0.05,
                )
                st.session_state["int_od_to_cfu_conversion_factor"] = st.number_input(
                    "OD-to-CFU conversion factor",
                    value=float(st.session_state.get("int_od_to_cfu_conversion_factor", 1e9)),
                    format="%.1e",
                )

        # Dosing Schedule
        with col2:
            st.markdown("### 📅 Dosing Schedule & Regimens")

            sub_col1, sub_col2 = st.columns(2)

            with sub_col1:
                st.markdown("#### Active Dosing Events")
                # Display active doses
                for idx, dose in enumerate(doses):
                    c_t1, c_t2, c_t3, c_t4 = st.columns([2, 3, 2, 1])
                    with c_t1:
                        st.text(f"t={dose['time']}h")
                    with c_t2:
                        tgt_lbl = f"{dose['target_type']} (Index {dose['target_idx']})"
                        st.text(f"{dose['amount']:.1e} to {tgt_lbl}")
                    with c_t3:
                        st.text(f"Route: {dose['route']}")
                    with c_t4:
                        if st.button("🗑️", key=f"del_dose_{idx}"):
                            doses.pop(idx)
                            st.session_state.int_doses = doses
                            st.rerun()

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
    if st.button("🚀 Run Simulation", use_container_width=True):
        with st.spinner("Assembling model equations & integrating..."):
            try:
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
                        for q in range(strains[j].get("dormancy_depth", 3)):
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
                        result.get_od() if hasattr(result, "get_od") else (total_bacteria + result.get("Debris")) / st.session_state.get("int_od_to_cfu_conversion_factor", 1.0),
                        "C2-",
                        lw=2.5,
                        label="OD (AU)",
                    )
                    cfu_od = total_bacteria / st.session_state.get(
                        "int_od_to_cfu_conversion_factor", 1.0
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
                use_container_width=True,
            )

        with c_down2:
            rep_code = generate_reproduction_code()
            st.download_button(
                "📥 Download Python Script",
                data=rep_code,
                file_name="pbisim_run.py",
                mime="text/x-python",
                use_container_width=True,
            )

        if show_code:
            with st.expander("🐍 View Python Reproduction Code"):
                st.code(rep_code, language="python")
