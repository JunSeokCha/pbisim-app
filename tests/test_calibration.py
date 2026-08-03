"""Calibration page: dataset-config persistence across page navigation (via AppTest).

Streamlit drops a widget's key from session_state whenever that widget is not
rendered on a rerun. Navigating from the Calibration page to the Simulator (to
change the model) and back must NOT reset the filter / grouping / statistics
selections — they are shadowed into the plain `fit_config` key and re-seeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from streamlit.testing.v1 import AppTest

APP = "pbisim_app/app.py"


def _synthetic_dataset():
    """A tiny long-format OD dataset with two arms and replicates."""
    rows = []
    for phage in ("MXP1", "MXP2"):
        for moi in (0.1, 1.0):
            for t in (0.0, 2.0, 4.0):
                for rep in range(2):
                    rows.append({"PHAGE": phage, "MOI": moi, "TIME": t,
                                 "DV": 0.05 * (t + 1) + 0.001 * rep})
    return pd.DataFrame(rows)


def test_calibration_config_survives_navigation():
    at = AppTest.from_file(APP, default_timeout=150)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _synthetic_dataset(), "time": "TIME", "value": "DV",
        "observable": "od", "arm_cols": ["PHAGE", "MOI"], "moi": "MOI",
    }

    # configure grouping + statistics on the Calibration page
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_group_cols"] = ["MOI"]
    at.session_state["fit_stat"] = "Median"
    at.session_state["fit_band"] = "25–75"
    at.run()  # render -> shadow into fit_config
    assert at.session_state["fit_config"]["fit_stat"] == "Median"

    # leave to change the model, then come back
    at.session_state["current_page_radio"] = "Interactive Simulator"
    at.run()
    at.session_state["current_page_radio"] = "Calibration"
    at.run()

    # selections must be restored, not reset to defaults
    assert at.session_state["fit_stat"] == "Median"
    assert at.session_state["fit_group_cols"] == ["MOI"]
    assert at.session_state["fit_band"] == "25–75"
    assert len(at.exception) == 0


def test_calibration_loads_dataset_with_nonregistry_observable():
    """A dataset whose observable column holds custom labels (e.g. 'colony_count',
    'od600') must still render the Calibration page — the observation-model selector
    only applies to registry observables and must not KeyError on the rest."""
    rows = [{"ARM": "A", "TIME": t, "DV": 1e7, "obs": lbl}
            for lbl in ("cfu", "colony_count", "od600")
            for t in (0.0, 2.0, 4.0)]
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": pd.DataFrame(rows), "time": "TIME", "value": "DV",
        "observable": "obs", "arm_cols": ["ARM"], "moi": None,
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    assert len(at.exception) == 0, at.exception


def test_calibration_covariate_panel_renders():
    """With a numeric grouping column (a covariate), the Calibration page renders the
    covariate-effects panel and offers that column as an available covariate — no errors."""
    rows = [{"ARM": "A", "temp": 30.0, "TIME": t, "DV": 1e7} for t in (0.0, 2.0, 4.0)]
    rows += [{"ARM": "B", "temp": 37.0, "TIME": t, "DV": 5e6} for t in (0.0, 2.0, 4.0)]
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": pd.DataFrame(rows), "time": "TIME", "value": "DV",
        "observable": "cfu", "arm_cols": ["ARM", "temp"], "moi": None,
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    assert len(at.exception) == 0, at.exception


def test_manual_tuning_edits_model_directly():
    """Phase B: the manual-tuning panel now renders the SAME model builder as the
    Interactive Simulator (render_model_builder), so editing an absolute value there
    updates the live model dict directly — no separate apply step."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _synthetic_dataset(), "time": "TIME", "value": "DV",
        "observable": "od", "arm_cols": ["PHAGE", "MOI"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    # reveal the embedded builder (Direct mode by default)
    at.session_state["fit_show_builder"] = True
    at.run()

    # set an absolute burst size on the builder's phage input (phg_burst_0), exactly
    # like editing it on the Interactive Simulator page
    at.session_state["phg_burst_0"] = 137.0
    at.run()
    # the live model dict reflects the edit directly
    assert at.session_state["int_phages"][0]["burst_sizes"] == 137.0

    # the overlay runs against the edited model
    [b for b in at.button if b.key == "fit_overlay"][0].click().run()
    assert len(at.exception) == 0


def test_calibration_embeds_full_builder_all_modes():
    """The manual-tuning panel renders the SAME builder as the Interactive Simulator,
    so every builder mode and its complete parameter set is present on the Calibration
    page — closing the reported gap where dormancy entry/resuscitation rates were
    missing. Direct-mode dormancy exposes the per-strain entry + resuscitation inputs;
    BRG and StrainSet render without error."""
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _synthetic_dataset(), "time": "TIME", "value": "DV",
        "observable": "od", "arm_cols": ["PHAGE", "MOI"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.session_state["fit_show_builder"] = True
    at.session_state["int_builder_mode"] = "Direct (ModelBuilder)"
    at.run()
    # tick the builder's dormancy checkbox on strain 0 (like a user) → the builder shows
    # entry (str_sleep) + resuscitation (str_wake) rates, which the old curated panel
    # omitted in some modes
    at.session_state["str_dorm_en_0"] = True
    at.run()
    keys = {n.key for n in at.number_input if n.key}
    assert "str_sleep_0" in keys and "str_wake_0" in keys, sorted(k for k in keys if "str_" in k)
    assert len(at.exception) == 0
    # switching builder mode on the Calibration page renders each mode cleanly
    for _mode in ("Binary Genotypes (BRG)", "Custom Strains & Graph (StrainSet)"):
        at.session_state["widget_builder_mode"] = _mode
        at.run()
        assert len(at.exception) == 0, (_mode, at.exception)
        assert at.session_state["int_builder_mode"] == _mode


def test_per_arm_b0_defaults_to_first_data_point():
    """pbisim-fit parity: each arm's B₀ defaults to its OWN first data observation (CFU
    verbatim), anchored to the data — NOT to the builder's B₀ (which only supplies the
    strain/genotype ratio and is renormalised away)."""
    df = pd.DataFrame([
        {"PHAGE": ph, "MOI": 1.0, "TIME": t, "DV": v}
        for ph in ("MXP1", "MXP2")
        for (t, v) in ((0.0, 4.2e6), (2.0, 1.0e6), (4.0, 5.0e5))
    ])
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    # deliberately-different builder B₀ — must NOT be what the overlay uses
    at.session_state["int_strains"][0]["initial_B"] = 9e9
    at.session_state["fit_dataset"] = {
        "raw": df, "time": "TIME", "value": "DV",
        "observable": "cfu", "arm_cols": ["PHAGE"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    # Per-arm mode uses editable widgets seeded from the first data point (first-obs
    # mode shows the same value as a live caption, not a widget).
    at.session_state["fit_b0_mode"] = "Per-arm values"
    at.run()
    b0s = [n.value for n in at.number_input if n.key and n.key.startswith("fit_cond_b0_")]
    assert b0s, "no per-arm B₀ inputs rendered"
    assert all(abs(float(v) - 4.2e6) < 1.0 for v in b0s), b0s   # first CFU, not the builder's 9e9
    assert len(at.exception) == 0


def test_calibration_can_enable_debris_for_od_fitting():
    """The OD/debris module can be toggled ON from the Calibration page itself (its
    checkbox lives in the Simulator's Environment tab, which the embedded builder does
    not include) — otherwise OD fitting via debris would be unreachable here."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _synthetic_dataset(), "time": "TIME", "value": "DV",
        "observable": "od", "arm_cols": ["PHAGE", "MOI"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.session_state["fit_show_builder"] = True
    at.run()
    at.session_state["fit_edit_debris_enabled"] = True
    at.run()
    assert at.session_state["int_debris_enabled"] is True
    # the dormant-OD-weight input becomes available once debris is on
    keys = {n.key for n in at.number_input if n.key}
    assert "fit_edit_dorm_od" in keys
    assert len(at.exception) == 0


def test_calibration_builder_uses_ratio_inoculum_labels():
    """Calibration renders the builder in ratio mode — per-strain B0 becomes a relative
    'ratio' and the absolute phage P0 input is hidden — while the Simulator keeps
    absolute magnitudes (inoculum_mode default)."""
    sim = AppTest.from_file(APP, default_timeout=120)
    sim.run()
    sim_labels = {n.label for n in sim.number_input}
    assert "Initial Density (B0)" in sim_labels          # Simulator = magnitude
    assert "Initial Density (P0)" in sim_labels

    cal = AppTest.from_file(APP, default_timeout=150)
    cal.run()
    cal.session_state["fit_dataset"] = {
        "raw": _synthetic_dataset(), "time": "TIME", "value": "DV",
        "observable": "od", "arm_cols": ["PHAGE", "MOI"], "moi": "MOI",
    }
    cal.session_state["current_page_radio"] = "Calibration"
    cal.session_state["fit_show_builder"] = True
    cal.run()
    cal_labels = {n.label for n in cal.number_input}
    assert "Initial ratio (relative)" in cal_labels      # Calibration = ratio
    assert "Initial Density (B0)" not in cal_labels
    assert "Initial Density (P0)" not in cal_labels       # phage P0 hidden (dose-driven)
    assert len(cal.exception) == 0


def _nonmem_dataset():
    """A NONMEM/Monolix-style long dataset: dose rows (EVID=1) interleaved with CFU
    observations; the observable column names the dose target for dose rows."""
    rows = []
    for arm in ("A", "B"):
        rows.append({"ARM": arm, "TIME": 0.0, "OBS": "bacteria", "DV": np.nan, "AMT": 5e6, "EVID": 1})
        rows.append({"ARM": arm, "TIME": 0.0, "OBS": "phage", "DV": np.nan, "AMT": 1e8, "EVID": 1})
        for t in (0.0, 2.0, 4.0):
            rows.append({"ARM": arm, "TIME": t, "OBS": "cfu", "DV": 1e6 / (t + 1),
                         "AMT": np.nan, "EVID": 0})
    return pd.DataFrame(rows)


def test_nonmem_dose_rows_imported_and_gate_manual_fields():
    """NONMEM dose rows are imported and, for the arms/targets they cover, replace the
    manual per-arm B₀ / MOI inputs (which are hidden in favour of a 'data dose' caption)."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _nonmem_dataset(), "time": "TIME", "value": "DV", "observable": "OBS",
        "arm_cols": ["ARM"], "moi": None, "dose_unit": "pfu",
        "evid": "EVID", "amount": "AMT", "unit_col": None,
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    _caps = [c.value for c in at.caption]
    assert any("Imported dose records" in c for c in _caps)
    keys = {n.key for n in at.number_input if n.key}
    # both arms carry data doses → the manual per-arm B₀ and MOI widgets are gated away
    assert not any(k.startswith("fit_cond_moi_") for k in keys)
    assert not any(k.startswith("fit_cond_b0_") for k in keys)
    assert len(at.exception) == 0


def test_overlay_persists_across_navigation():
    """The overlay visualization stays alive after navigating away and back,
    until it is explicitly re-run (item 1)."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _synthetic_dataset(), "time": "TIME", "value": "DV",
        "observable": "od", "arm_cols": ["PHAGE", "MOI"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    [b for b in at.button if b.key == "fit_overlay"][0].click().run()
    assert "calib_overlay_result" in at.session_state and at.session_state["calib_overlay_result"]
    _fitq = lambda a: any("Fit quality" in m.value for m in a.markdown)
    assert _fitq(at)  # shown right after overlay

    # leave and return without re-clicking — plot must still be there
    at.session_state["current_page_radio"] = "Interactive Simulator"
    at.run()
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    assert _fitq(at)
    assert len(at.exception) == 0


def test_globals_and_debris_and_save_scenario():
    """Calibration exposes global/structural + OD-debris params, uses the
    debris-inclusive OD when the module is on, and can save the calibrated
    config as a Scenario (items in the 2026-07-15 request)."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["int_debris_enabled"] = True
    at.session_state["int_od_to_cfu_conversion_factor"] = 1e9
    at.session_state["fit_dataset"] = {
        "raw": _synthetic_dataset(), "time": "TIME", "value": "DV",
        "observable": "od", "arm_cols": ["PHAGE", "MOI"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    # the global/structural + debris inputs live under the model-builder toggle
    at.session_state["fit_show_builder"] = True
    at.run()

    keys = {n.key for n in at.number_input if n.key}
    for k in ("fit_edit_n_latent", "fit_edit_S0", "fit_edit_recycle",
              "fit_edit_debris_u", "fit_edit_dorm_od"):
        assert k in keys
    # the single od_to_cfu now lives in §4 Overlay (drives B₀, overlay, and debris)
    assert "fit_link_od" in keys
    # the OD/debris block is still shown
    assert any("debris module" in m.value for m in at.markdown)

    [b for b in at.button if b.key == "fit_overlay"][0].click().run()
    assert at.session_state["calib_overlay_result"]

    at.session_state["fit_save_name"] = "calib_test"
    at.run()
    [b for b in at.button if b.key == "fit_save_scenario"][0].click().run()
    assert "calib_test" in at.session_state["user_scenarios"]
    assert len(at.exception) == 0


def _multi_observable_dataset():
    """A long-format dataset with TWO observables (CFU + OD) per arm — a single
    'observable' column, as pbisim-fit expects for joint multi-dataset fitting."""
    rows = []
    for phage in ("MXP1", "MXP2"):
        for t in (0.0, 2.0, 4.0):
            rows.append({"PHAGE": phage, "MOI": 1.0, "TIME": t, "OBS": "cfu", "DV": 1e7 / (t + 1)})
            rows.append({"PHAGE": phage, "MOI": 1.0, "TIME": t, "OBS": "od", "DV": 0.05 * (t + 1)})
    return pd.DataFrame(rows)


def test_multi_observable_overlay_small_multiples():
    """A CFU+OD dataset produces one overlay panel per observable, per-observable
    RMSE/R², and a combined objective — all from one simulation per arm."""
    at = AppTest.from_file(APP, default_timeout=220)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _multi_observable_dataset(), "time": "TIME", "value": "DV",
        "observable": "OBS", "arm_cols": ["PHAGE"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    [b for b in at.button if b.key == "fit_overlay"][0].click().run()
    assert len(at.exception) == 0, at.exception
    ovr = at.session_state["calib_overlay_result"]
    obs = {p["obs"] for p in ovr["panels"]}
    assert obs == {"cfu", "od"}, obs                 # one panel per observable
    assert np.isfinite(ovr["combined"])              # combined objective computed
    blob = " ".join(m.value for m in at.markdown)
    assert "Combined objective J" in blob
    # the observables multiselect defaulted to both present observables
    ms = [m for m in at.multiselect if m.key == "fit_obs_sel"][0]
    assert set(ms.value) == {"cfu", "od"}


def test_overlay_exposes_residuals_and_model_comparison():
    """The overlay result carries the pooled residual vector, and the AIC/BIC panel
    snapshots candidate models and ranks them."""
    at = AppTest.from_file(APP, default_timeout=220)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _multi_observable_dataset(), "time": "TIME", "value": "DV",
        "observable": "OBS", "arm_cols": ["PHAGE"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    [b for b in at.button if b.key == "fit_overlay"][0].click().run()
    ovr = at.session_state["calib_overlay_result"]
    assert ovr["residuals"] and ovr["n_resid"] == len(ovr["residuals"])  # pooled residuals surfaced

    # snapshot two candidates (different free-param counts) and rank them
    at.session_state["fit_cmp_k"] = 3
    [b for b in at.button if b.key == "fit_cmp_add"][0].click().run()
    at.session_state["fit_cmp_k"] = 6
    [b for b in at.button if b.key == "fit_cmp_add"][0].click().run()
    assert len(at.exception) == 0, at.exception
    cmp = at.session_state["fit_model_comparison"]
    assert len(cmp) == 2 and {c["k"] for c in cmp} == {3, 6}


def _two_arm_cfu_dataset():
    """Two CFU arms (no phage) so a per-arm pre-run is the only thing that differs."""
    rows = []
    for phage in ("MXP1", "MXP2"):
        for t in (0.0, 2.0, 4.0):
            rows.append({"PHAGE": phage, "MOI": 0.0, "TIME": t, "OBS": "cfu", "DV": 1e7})
    return pd.DataFrame(rows)


def test_per_arm_prerun_condition_changes_trajectory():
    """A per-arm pre-run (stationary phase) changes only that arm's model
    trajectory, so log-phase and stationary-phase data can be fit together."""
    at = AppTest.from_file(APP, default_timeout=220)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _two_arm_cfu_dataset(), "time": "TIME", "value": "DV",
        "observable": "OBS", "arm_cols": ["PHAGE"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    at.session_state["fit_cond_prerun_MXP1"] = 12.0   # MXP1 stationary; MXP2 log-phase
    at.run()
    [b for b in at.button if b.key == "fit_overlay"][0].click().run()
    assert len(at.exception) == 0, at.exception
    ovr = at.session_state["calib_overlay_result"]
    cfu = [p for p in ovr["panels"] if p["obs"] == "cfu"][0]
    ser = {s["label"]: np.asarray(s["pred"], dtype=float) for s in cfu["series"]}
    assert not np.allclose(ser["MXP1"], ser["MXP2"]), "pre-run did not change the arm"
    assert any(m.get("pre-run (h)") == 12.0 for m in ovr["metrics"])


def test_export_fit_spec_available_after_data():
    """The 'Export fit specification' hand-off (pbisim-fit) appears once data is loaded."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _multi_observable_dataset(), "time": "TIME", "value": "DV",
        "observable": "OBS", "arm_cols": ["PHAGE"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()
    assert len(at.exception) == 0, at.exception
    assert any("Export fit specification" in (m.value or "") for m in at.markdown)
    # the spec built successfully (not the "select a group first" error fallback)
    assert not any("Select at least one group" in (m.value or "") for m in at.markdown)
