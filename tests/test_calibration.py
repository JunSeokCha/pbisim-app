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
