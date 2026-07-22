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


def test_manual_tuning_edits_model_directly():
    """Phase B: editing an absolute parameter value updates the live model dict."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _synthetic_dataset(), "time": "TIME", "value": "DV",
        "observable": "od", "arm_cols": ["PHAGE", "MOI"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()

    # set an absolute burst size on the phage tuning input, exactly like ModelBuilder
    at.session_state["fit_edit_p_burst_sizes_0"] = 137.0
    at.run()
    # the live model dict reflects the edit directly — no separate apply step
    assert at.session_state["int_phages"][0]["burst_sizes"] == 137.0
    # and it is NOT shadowed into fit_config (dict stays authoritative)
    assert "fit_edit_p_burst_sizes_0" not in at.session_state["fit_config"]

    # the overlay runs against the edited model
    [b for b in at.button if b.key == "fit_overlay"][0].click().run()
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

    keys = {n.key for n in at.number_input if n.key}
    for k in ("fit_edit_n_latent", "fit_edit_S0", "fit_edit_recycle",
              "fit_edit_od2cfu", "fit_edit_debris_u"):
        assert k in keys
    # with debris on, the simple biomass/link input is replaced by a note
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
