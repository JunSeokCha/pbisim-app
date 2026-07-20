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
import time

# Force the non-interactive backend BEFORE importing pyplot. On a headless server
# (e.g. the Render container) a GUI backend used from Streamlit's script thread
# segfaults (exit 139); Agg is thread-safe and display-free.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st






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
from pbisim_app.viz_helper import plot_axis_controls, apply_axis_plotly
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

from pbisim_app.common import *  # helpers + constants (Phase-1 refactor)




















# ── Page config ───────────────────────────────────────────────────────────────
_FAVICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCAzMiAzMic+PHJlY3Qgd2lkdGg9JzMyJyBoZWlnaHQ9JzMyJyByeD0nNicgZmlsbD0nIzBkN2E2OCcvPjx0ZXh0IHg9JzE2JyB5PScyMycgZm9udC1mYW1pbHk9J21vbm9zcGFjZScgZm9udC1zaXplPScyMicgZm9udC13ZWlnaHQ9JzYwMCcgZmlsbD0nI2ZmZmZmZicgdGV4dC1hbmNob3I9J21pZGRsZSc+JiM5NjY7PC90ZXh0Pjwvc3ZnPg=="
st.set_page_config(
    page_title="pbisim — Phage-Bacteria Simulation Control Center",
    page_icon=_FAVICON,
    layout="wide",
)


# ── Custom CSS for Premium Aesthetics ─────────────────────────────────────────
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

theme_mode = st.session_state["theme_mode"]

# App styling — loaded once from static/styles.css (light theme; dark deferred).
with open(os.path.join(os.path.dirname(__file__), "static", "styles.css"), encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)




_init_app_state()
















































if "int_strains" not in st.session_state:
    load_preset_to_state(DEFAULT_SCENARIO)














































# ── Sidebar Settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin:2px 0 8px'>"
        "<div style='width:30px;height:30px;border-radius:6px;background:var(--teal);color:#fff;"
        "display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:600;"
        "font-family:IBM Plex Mono,monospace'>&#966;</div>"
        "<div><div style='font-size:19px;font-weight:600;line-height:1;color:var(--ink)'>pbisim</div>"
        "<div class='section-label' style='font-size:9.5px;margin-top:4px'>PHAGE-BACTERIA SIM</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

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
    with st.expander("AI & model settings", expanded=False):

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
            st.caption(f"Anthropic API key set ({_src})")
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

        if st.button("Test API Key & List Models", key="test_api_key_btn"):
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
                        st.info("Note: If you get a 404 error here, your key is authentic but has no models enabled (often because the Anthropic account is at Tier 0/unfunded). If you get a 401, the key is invalid.")

    # Dark mode is deferred: the redesign targets the (light-only) mockup, and the
    # dark CSS branch still has contrast issues. Force light and hide the toggle so
    # nobody lands on the broken dark state; the dark CSS is kept dormant for a
    # future one-shot dark pass. Re-expose the selectbox here to bring it back.
    st.session_state["theme_mode"] = "Light"

    st.markdown("---")
    if st.button("Reset Environment"):
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
    from pbisim_app.views import library
    library.render()
elif st.session_state.current_page == "Calibration":
    from pbisim_app.views import calibration
    calibration.render()
elif st.session_state.current_page == "AI Assistant":
    from pbisim_app.views import assistant
    assistant.render()
elif st.session_state.current_page == "Clinical Trials & Cohorts":
    from pbisim_app.views import trials
    trials.render()
elif st.session_state.current_page == "Dose-Response Sweeps":
    from pbisim_app.views import dose_response
    dose_response.render()
elif st.session_state.current_page == "Parameter Sweeps":
    from pbisim_app.views import param_sweeps
    param_sweeps.render()
elif st.session_state.current_page == "Interactive Simulator":
    from pbisim_app.views import simulator
    simulator.render()
