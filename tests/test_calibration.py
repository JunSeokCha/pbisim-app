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


def test_manual_tuning_applies_to_model():
    """Phase B: a tuning multiplier bakes into the live model's GUI dicts."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.session_state["fit_dataset"] = {
        "raw": _synthetic_dataset(), "time": "TIME", "value": "DV",
        "observable": "od", "arm_cols": ["PHAGE", "MOI"], "moi": "MOI",
    }
    at.session_state["current_page_radio"] = "Calibration"
    at.run()

    before = at.session_state["int_phages"][0]["burst_sizes"]
    at.session_state["fit_tune_burst"] = 3.0
    at.run()
    [b for b in at.button if b.key == "fit_tune_apply"][0].click().run()

    # the multiplier is baked into the model, then cleared back to unity
    assert at.session_state["int_phages"][0]["burst_sizes"] == before * 3.0
    assert "fit_tune_burst" not in at.session_state or at.session_state["fit_tune_burst"] == 1.0
    assert len(at.exception) == 0
