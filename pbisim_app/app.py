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

# Theme matplotlib globally so the AI Assistant's agent-generated figures match the
# app's Plotly palette/typography (rcParams only — no change to generated code).
from pbisim_app.viz_helper import apply_mpl_theme
apply_mpl_theme()

# Optional shared-credential gate (active only when APP_PASSWORD[_HASH] is set in the
# environment). Blocks everything below until signed in; a no-op locally.
from pbisim_app.auth import require_login, sign_out_control
require_login()


_init_app_state()
















































if "int_strains" not in st.session_state:
    load_preset_to_state(DEFAULT_SCENARIO)

# Capture the pristine default Model once (organism/kinetics only) — prebuilt demo
# models are materialised as overrides on top of this snapshot.
if "_default_model_state" not in st.session_state:
    st.session_state["_default_model_state"] = dump_model()














































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

    _pages = ["Interactive Simulator", "Dose-Response Sweeps", "Parameter Sweeps", "Clinical Trials & Cohorts", "Calibration", "AI Assistant", "Library", "Help"]
    # Scripting is an opt-in power-user page (arbitrary Python; research sandbox). Hidden
    # unless PBISIM_ENABLE_SCRIPTING is set — see scripting_enabled().
    if scripting_enabled():
        _pages.append("Scripting")
    # Guard the radio index: if the previously-selected page is no longer present (e.g.
    # the scripting flag was turned off), fall back to the first page. A keyed radio's
    # stored value overrides index=, so drop it too when it points at a missing option.
    if st.session_state.get("current_page_radio") not in _pages:
        st.session_state.pop("current_page_radio", None)
    _cur = st.session_state.current_page if st.session_state.current_page in _pages else _pages[0]
    st.session_state.current_page = st.radio(
        "Navigation",
        _pages,
        key="current_page_radio",
        index=_pages.index(_cur),
    )
    st.session_state.current_page = st.session_state.current_page_radio

    sign_out_control()

    # ── Models ────────────────────────────────────────────────────────────────
    # The active Model = the organism/kinetics config the builder reflects. Saved &
    # demo models are frozen; downstream tasks (sweeps, trials, fitting) run against
    # a chosen Model, so the live builder can't silently contaminate them.
    st.markdown("---")
    st.markdown("<div class='section-label' style='margin-bottom:4px'>MODELS</div>",
                unsafe_allow_html=True)
    # Apply a pending programmatic model activation (from Save) BEFORE the selectbox
    # is instantiated, so its keyed value can be set (and won't revert the choice).
    _pend_model = st.session_state.pop("_pending_active_model", None)
    if _pend_model is not None:
        st.session_state.active_model = _pend_model
        st.session_state.sidebar_model_pick = _pend_model
    _mopts = model_options()
    if st.session_state.get("sidebar_model_pick") not in _mopts:
        st.session_state.pop("sidebar_model_pick", None)
    _active = st.session_state.active_model if st.session_state.active_model in _mopts else WORKING_DRAFT_LABEL
    _msel = st.selectbox("Active model", _mopts, index=_mopts.index(_active),
                         key="sidebar_model_pick", label_visibility="collapsed")
    _mdesc = None
    _demos = {d["name"]: d for d in DEMO_MODELS}
    if _msel in _demos:
        _mdesc = _demos[_msel]["description"]
    elif _msel in st.session_state.user_models:
        _mdesc = st.session_state.user_models[_msel].get("description")
    if _msel == WORKING_DRAFT_LABEL:
        st.caption("Live builder state — edits here flow to any task using this option.")
    elif _mdesc:
        st.caption(_mdesc)
    if _msel != st.session_state.active_model:
        # user switched models → load the chosen one into the builder
        if _msel != WORKING_DRAFT_LABEL:
            _snap = resolve_model_snapshot(_msel)
            if _snap is not None:
                apply_model_to_state(_snap)
        st.session_state.active_model = _msel
        st.session_state["_flash"] = {"kind": "success",
                                      "msg": f"Model '{_msel}' loaded into the builder."}
        # Stay on the current page — switching models from Calibration/Sweeps/etc.
        # shouldn't yank the user to the Interactive Simulator.
        st.rerun()
    with st.expander("Save current builder as a Model", expanded=False):
        _mname = st.text_input("Model name", key="save_model_name",
                               placeholder="e.g. E. coli + T4 (lit.)")
        _mdescr = st.text_input("Description (optional)", key="save_model_desc")
        if st.button("Save model", key="save_model_btn", width="stretch"):
            _nm = (_mname or "").strip()
            if not _nm:
                st.error("Enter a model name.")
            elif _nm in (list(_demos) + [WORKING_DRAFT_LABEL]):
                st.error("That name is reserved — choose another.")
            else:
                _um = st.session_state.user_models
                _um[_nm] = {"description": (_mdescr or "").strip(), "source": "user",
                            "schema_version": MODEL_SCHEMA_VERSION, "state": dump_model()}
                st.session_state.user_models = _um
                st.session_state["_pending_active_model"] = _nm
                st.session_state["_flash"] = {"kind": "success", "msg": f"Saved model '{_nm}'."}
                st.rerun()

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
elif st.session_state.current_page == "Scripting" and scripting_enabled():
    from pbisim_app.views import scripting
    scripting.render()
elif st.session_state.current_page == "Help":
    from pbisim_app.views import help as help_view
    help_view.render()
elif st.session_state.current_page == "Interactive Simulator":
    from pbisim_app.views import simulator
    simulator.render()
