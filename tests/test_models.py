"""Model registry: a saved/demo Model (organism + kinetics only) is the unit that
downstream tasks run against, decoupled from the live builder widgets.

Core guarantees:
- the Model partition EXCLUDES dosing / trial / analysis keys;
- selecting a demo/saved Model in the sidebar loads it into the builder;
- saving keeps the new model active (no revert);
- a fit against a FROZEN model is immune to live-builder contamination and leaves
  the builder untouched (the bug that motivated this whole design).
"""

from __future__ import annotations

import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

from pbisim_app.common import _is_model_data_key, DEMO_MODELS, WORKING_DRAFT_LABEL

APP = "pbisim_app/app.py"


def _free_targets(at, paths):
    """Set role='Free' for the given parameter paths in the unified fit table (AppTest
    can't click data_editor cells, so we edit the stable source df directly)."""
    tdf = at.session_state["fit_targets_df"].copy()
    for i, r in tdf.iterrows():
        if r["path"] in paths:
            tdf.at[i, "role"] = "Free"
    at.session_state["fit_targets_df"] = tdf


def _derive_targets(at, path_expr):
    """Set role='Derived' + an expression for each {path: expr} — mappings now live on
    the parameter rows, not a separate map table."""
    tdf = at.session_state["fit_targets_df"].copy()
    for i, r in tdf.iterrows():
        if r["path"] in path_expr:
            tdf.at[i, "role"] = "Derived"
            tdf.at[i, "expression"] = path_expr[r["path"]]
    at.session_state["fit_targets_df"] = tdf


def test_model_partition_excludes_analysis_keys():
    assert _is_model_data_key("int_strains")
    assert _is_model_data_key("int_phages")
    assert _is_model_data_key("int_monod_constant")
    assert _is_model_data_key("ads_0_0")
    # analysis-only keys are NOT part of a Model
    assert not _is_model_data_key("int_doses")
    assert not _is_model_data_key("trial_arms")
    assert not _is_model_data_key("trial_iiv_inputs")
    # non-model keys
    assert not _is_model_data_key("fit_arms")
    assert not _is_model_data_key("current_page")


def test_demo_models_have_name_and_description():
    assert DEMO_MODELS
    for d in DEMO_MODELS:
        assert d["name"] and d["description"] and isinstance(d["overrides"], dict)


def test_default_snapshot_and_registry_init():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert at.session_state["active_model"] == WORKING_DRAFT_LABEL
    snap = at.session_state["_default_model_state"]
    assert "int_strains" in snap
    assert "int_doses" not in snap and "trial_arms" not in snap
    assert at.session_state["user_models"] == {}


def test_sidebar_select_demo_loads_into_builder():
    at = AppTest.from_file(APP, default_timeout=150)
    at.run()
    at.session_state["sidebar_model_pick"] = "Growth calibration (Monod)"
    at.run()
    assert at.session_state["active_model"] == "Growth calibration (Monod)"
    s0 = at.session_state["int_strains"][0]
    assert s0["growth_rate"] == 1.2
    assert s0["bacteria_to_resource_ratio"] == 1e8
    assert len(at.exception) == 0


def test_save_model_keeps_it_active_and_excludes_doses():
    at = AppTest.from_file(APP, default_timeout=150)
    at.run()
    at.session_state["save_model_name"] = "My model"
    at.run()
    [b for b in at.button if b.key == "save_model_btn"][0].click().run()
    assert "My model" in at.session_state["user_models"]
    assert at.session_state["active_model"] == "My model"        # no revert
    state = at.session_state["user_models"]["My model"]["state"]
    assert "int_doses" not in state and "trial_arms" not in state
    assert len(at.exception) == 0


def test_model_snapshot_renders_on_simulator_and_pages():
    """The click-to-view model-config snapshot renders a full sectioned summary of the
    live draft (Simulator) and of a frozen Model (a page_model_selector page), so the
    whole config is visible without tab-hunting."""
    at = AppTest.from_file(APP, default_timeout=150)
    at.run()
    # Simulator — live draft
    at.session_state["current_page_radio"] = "Interactive Simulator"
    at.run()
    at.session_state["sim_show_cfg"] = True
    at.run()
    md = " ".join(m.value for m in at.markdown)
    assert "Builder mode:" in md
    assert all(s in md for s in ("Growth & nutrient environment", "Phage", "Solver & structure"))
    assert len(at.dataframe) >= 5           # one table per section
    assert len(at.exception) == 0

    # A page that selects a frozen Model — snapshot of that model
    a = AppTest.from_file(APP, default_timeout=150)
    a.run()
    a.session_state["current_page_radio"] = "Parameter Sweeps"
    a.run()
    a.session_state["psweep_model_sel"] = "Two-strain resistance (WT + resistant)"
    a.session_state["psweep_show_cfg"] = True
    a.run()
    md2 = " ".join(m.value for m in a.markdown)
    assert "Builder mode:" in md2 and "Growth & nutrient environment" in md2
    assert len(a.exception) == 0


def test_estimate_b0_mode_runs_end_to_end():
    """Selecting the 'Estimate (shared)' B0 source runs a fit that estimates B0 via
    free_initial_conditions — the fitted config carries a fit_initial_cfu."""
    import numpy as np
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Growth calibration (Monod)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu"]
    at.session_state["fit_b0_mode"] = "Estimate (shared)"
    at.run()
    _free_targets(at, {"growth_rates[0]"})
    at.session_state["fit_nls_restarts"] = 1
    at.session_state["fit_nls_maxnfev"] = 120
    at.run()
    [b for b in at.button if b.key == "fit_run_nls"][0].click().run()
    assert len(at.exception) == 0
    fc = at.session_state["calib_fitted_config"]
    _ic = getattr(fc, "fit_initial_cfu", None)
    assert _ic is not None and float(np.atleast_1d(_ic).ravel()[0]) > 0


def test_fit_draws_fitted_overlay_and_apply_persists():
    """After a fit: (a) the overlay auto-redraws with the fitted curves, and (b)
    'Apply' persists the fitted values into the model (the manual-tuning widgets must
    NOT clobber them on the next rerun)."""
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Growth calibration (Monod)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu", "od"]
    at.run()  # build the target table for this model
    _free_targets(at, {"growth_rates[0]", "bacteria_to_resource_ratio[0]"})
    at.session_state["fit_nls_restarts"] = 2
    at.session_state["fit_nls_maxnfev"] = 200
    at.run()
    [b for b in at.button if b.key == "fit_run_nls"][0].click().run()
    # (a) the overlay is the fitted one
    ovr = at.session_state["calib_overlay_result"]
    assert ovr and "Fitted model" in ovr["title"]
    mapv = at.session_state["calib_fit_result"]["map"]["free0"]  # freed growth
    # (b) apply persists into the builder (not reverted by fit_edit_* widgets)
    [b for b in at.button if b.key == "fit_apply_map"][0].click().run()
    assert abs(at.session_state["int_strains"][0]["growth_rate"] - mapv) < 1e-6
    assert len(at.exception) == 0


def test_reparameterization_theta_mapping():
    """The theta/mapping panel: define thetas and bind targets to expressions of them
    (growth_rates[1] = theta1 * (1 - theta2)); the fit reports the thetas and the
    fitted config honours the mapping."""
    import numpy as np
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.session_state["sidebar_model_pick"] = "Two-strain resistance (WT + resistant)"
    at.run()
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Two-strain resistance (WT + resistant)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu"]
    at.run()
    _free_targets(at, {"bacteria_to_resource_ratio[0]"})
    # Numeric cells are TextColumn (scientific-notation friendly) → pass strings.
    at.session_state["fit_thetas_df"] = pd.DataFrame([
        {"name": "theta1", "lower": "0.1", "upper": "3.0", "log": False, "initial": "1.0"},
        {"name": "theta2", "lower": "0.0", "upper": "0.9", "log": False, "initial": "0.1"}])
    _derive_targets(at, {"growth_rates[0]": "theta1",
                         "growth_rates[1]": "theta1*(1-theta2)"})
    at.session_state["fit_nls_restarts"] = 2
    at.session_state["fit_nls_maxnfev"] = 250
    at.run()
    [b for b in at.button if b.key == "fit_run_nls"][0].click().run()
    assert len(at.exception) == 0
    keys = {p["key"] for p in at.session_state["calib_fit_result"]["params"]}
    assert "theta1" in keys and "theta2" in keys
    g = np.atleast_1d(np.asarray(at.session_state["calib_fitted_config"].growth_rates))
    cost = at.session_state["calib_fit_result"]["map"]["theta2"]
    assert abs(g[1] - g[0] * (1.0 - cost)) < 1e-6      # mapping honoured


def test_model_switch_stays_on_current_page():
    """Switching models from another page (e.g. Calibration) must NOT yank the user to
    the Interactive Simulator — it just loads the model into the builder."""
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["sidebar_model_pick"] = "Growth calibration (Monod)"
    at.run()
    assert at.session_state["active_model"] == "Growth calibration (Monod)"
    assert at.session_state["current_page"] == "Calibration"      # stayed put
    assert len(at.exception) == 0


def test_apply_writes_estimated_initial_cfu():
    """Estimating fit_initial_cfu must update the model's initial_B on Apply, so a
    re-run / the simulator matches the fit (previously it stayed at the old B0)."""
    at = AppTest.from_file(APP, default_timeout=250)
    at.run()
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Growth calibration (Monod)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu"]
    at.run()
    _free_targets(at, {"growth_rates[0]", "fit_initial_cfu"})
    at.session_state["fit_nls_restarts"] = 1
    at.session_state["fit_nls_maxnfev"] = 150
    at.run()
    [b for b in at.button if b.key == "fit_run_nls"][0].click().run()
    est_b0 = at.session_state["calib_fitted_config"].fit_initial_cfu
    assert est_b0 and est_b0 > 0
    [b for b in at.button if b.key == "fit_apply_map"][0].click().run()
    applied = sum(float(s["initial_B"]) for s in at.session_state["int_strains"])
    assert abs(applied - float(est_b0)) / float(est_b0) < 1e-6     # B0 updated to the estimate
    assert len(at.exception) == 0


def test_fitted_overlay_uses_estimated_initial_cfu():
    """The auto-drawn fitted overlay must use the ESTIMATED B0 (fit_initial_cfu), not
    the model's — the per-arm condition b0 would otherwise override it."""
    at = AppTest.from_file(APP, default_timeout=250)
    at.run()
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Growth calibration (Monod)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu"]
    at.session_state["int_strains"][0]["initial_B"] = 1e9   # deliberately wrong (data ~5e6)
    at.run()
    tdf = at.session_state["fit_targets_df"].copy()
    for i, r in tdf.iterrows():
        if r["path"] in ("growth_rates[0]", "fit_initial_cfu"):
            tdf.at[i, "role"] = "Free"
            if r["path"] == "fit_initial_cfu":
                tdf.at[i, "lower"] = "1e3"; tdf.at[i, "upper"] = "1e11"
            else:
                tdf.at[i, "lower"] = "0.1"; tdf.at[i, "upper"] = "3.0"
    at.session_state["fit_targets_df"] = tdf
    at.session_state["fit_nls_restarts"] = 1
    at.session_state["fit_nls_maxnfev"] = 200
    at.run()
    [b for b in at.button if b.key == "fit_run_nls"][0].click().run()
    est = at.session_state["calib_fitted_config"].fit_initial_cfu
    b0_used = [m["B₀"] for m in at.session_state["calib_overlay_result"]["metrics"]]
    assert all(abs(b - est) / est < 1e-6 for b in b0_used)     # overlay used the estimate
    assert at.session_state["calib_overlay_result"]["combined"] < 0.3   # so it matches data


def test_fit_spec_text_applies_to_tables_and_fits():
    """The text spec applies to the fix/free tables (two-way) and the resulting fit
    runs — a theta with a prior + a mapping, entered as text."""
    at = AppTest.from_file(APP, default_timeout=250)
    at.run()
    at.session_state["sidebar_model_pick"] = "Two-strain resistance (WT + resistant)"
    at.run()
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Two-strain resistance (WT + resistant)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu"]
    at.run()
    at.session_state["_spec_pending"] = (
        "theta g bounds=0.1..3.0 prior=1.2,0.3\n"
        "map growth_rates[0] = g\nmap growth_rates[1] = g")
    at.run()
    [b for b in at.button if b.key == "fit_spec_to_tables"][0].click().run()
    assert not any(m.value for m in at.error)                 # parsed cleanly
    assert list(at.session_state["fit_thetas_df"]["name"]) == ["g"]
    _tdf = at.session_state["fit_targets_df"]
    assert (_tdf["role"] == "Derived").sum() == 2             # mappings on the target rows
    at.session_state["fit_nls_restarts"] = 1
    at.session_state["fit_nls_maxnfev"] = 150
    at.run()
    [b for b in at.button if b.key == "fit_run_nls"][0].click().run()
    assert len(at.exception) == 0
    assert "g" in {p["key"] for p in at.session_state["calib_fit_result"]["params"]}


def test_share_helper_ties_rows_to_one_theta():
    """The one-click Share helper sets the picked parameter rows to role='Derived' with
    a common θ and appends that θ to the θ table — the unified-table way to share."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["sidebar_model_pick"] = "Two-strain resistance (WT + resistant)"
    at.run()
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Two-strain resistance (WT + resistant)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu"]
    at.run()  # build the target table for this model
    gl = [l for l in at.session_state["fit_targets_df"]["parameter"] if "Growth rate" in l][:2]
    assert len(gl) == 2
    at.session_state["fit_share_pick"] = gl
    [b for b in at.button if b.key == "fit_share_go"][0].click().run()
    tdf = at.session_state["fit_targets_df"]
    derived = tdf[tdf["role"] == "Derived"]
    # both growth rows are now Derived, tied to one auto-named θ
    assert set(derived["parameter"]) == set(gl)
    _exprs = set(derived["expression"])
    assert len(_exprs) == 1
    theta = _exprs.pop()
    assert theta in list(at.session_state["fit_thetas_df"]["name"])
    # a Derived row draws everything from its θ → its value/bounds/prior cells are blanked
    for _c in ("value", "lower", "upper", "prior μ", "prior σ"):
        assert all(str(v).strip() == "" for v in derived[_c])
    assert len(at.exception) == 0


def test_unbounded_params_run_single_start():
    """Blank bounds mean UNCONSTRAINED (not blocked, not a silent [0,1]). An unbounded
    parameter forces a single start (multi-start needs finite bounds) but still fits
    from its current value / theta initial."""
    import numpy as np
    at = AppTest.from_file(APP, default_timeout=250)
    at.run()
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Growth calibration (Monod)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu"]
    at.run()
    # free growth with BOTH bounds blank (fully unconstrained); ratio one-sided (lower only)
    tdf = at.session_state["fit_targets_df"].copy()
    for i, r in tdf.iterrows():
        if r["path"] == "growth_rates[0]":
            tdf.at[i, "role"] = "Free"
        if r["path"] == "bacteria_to_resource_ratio[0]":
            tdf.at[i, "role"] = "Free"
            tdf.at[i, "lower"] = "1e6"          # string cell (TextColumn); upper blank → unbounded above
    at.session_state["fit_targets_df"] = tdf
    at.session_state["fit_nls_restarts"] = 3   # will be capped to 1 internally
    at.session_state["fit_nls_maxnfev"] = 200
    at.run()
    [b for b in at.button if b.key == "fit_run_nls"][0].click().run()
    assert len(at.exception) == 0
    m = at.session_state["calib_fit_result"]["map"]
    assert 0.9 < m["free0"] < 1.5              # unbounded growth recovered from its start


def test_invalid_theta_range_blocks_fit():
    """A theta with lower ≥ upper (both finite) is a genuine error and blocks the fit —
    but this is NOT the omitted-bounds case (which is allowed = unconstrained)."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["sidebar_model_pick"] = "Two-strain resistance (WT + resistant)"
    at.run()
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Two-strain resistance (WT + resistant)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu"]
    at.run()
    at.session_state["fit_thetas_df"] = pd.DataFrame(
        [{"name": "g", "lower": "5.0", "upper": "2.0", "log": False, "initial": "1.0"}])  # lower > upper
    _derive_targets(at, {"growth_rates[0]": "g"})
    at.run()
    [b for b in at.button if b.key == "fit_run_nls"][0].click().run()
    assert "calib_fit_result" not in at.session_state or at.session_state["calib_fit_result"] is None
    assert any("lower ≥ upper" in (m.value or "") for m in at.error)


def test_frozen_model_fit_ignores_contaminated_builder():
    """The headline decoupling test: contaminate the live builder, fit against a
    frozen demo model, and confirm (a) truth is still recovered and (b) the live
    builder is left untouched."""
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.session_state["int_strains"][0]["growth_rate"] = 0.1   # would pin the fit
    df = pd.read_csv("pbisim_app/examples/tutorial_synthetic_brg.csv")
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "time", "value": "value", "observable": "observable",
        "arm_cols": ["arm"], "moi": None}
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_model_sel"] = "Growth calibration (Monod)"
    at.session_state["fit_arms"] = ["control"]
    at.session_state["fit_obs_sel"] = ["cfu", "od"]
    at.run()
    _free_targets(at, {"growth_rates[0]", "bacteria_to_resource_ratio[0]"})
    at.session_state["fit_nls_restarts"] = 2
    at.session_state["fit_nls_maxnfev"] = 200
    at.run()
    [b for b in at.button if b.key == "fit_run_nls"][0].click().run()
    assert len(at.exception) == 0
    m = at.session_state["calib_fit_result"]["map"]
    assert 0.9 < m["free0"] < 1.5           # freed growth recovered despite bad builder
    # live builder untouched by the frozen build
    assert at.session_state["int_strains"][0]["growth_rate"] == 0.1
