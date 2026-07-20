"""Regression tests for the visual redesign (Pass B structure).

These assert the *structure* the redesign introduces (results header bar,
outcome badge, metric tiles) renders without error — not pixels.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_results_header_and_peak_phage_tile():
    """After a run, the Interactive Simulator shows a results header with a
    solver/runtime meta line, an outcome badge, and a Peak Phage Titre tile."""
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=180)
    at.run()
    [b for b in at.button if "Run Simulation" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception

    blob = " ".join(m.value for m in at.markdown)
    assert "Simulation results" in blob
    assert "Peak Phage Titre" in blob
    assert any(w in blob for w in ("Suppressed", "Cleared", "Regrowth", "Uncontrolled")), blob
    # runtime was captured for the meta line (SafeSessionState: no .get())
    assert "sim_runtime" in at.session_state and at.session_state["sim_runtime"] is not None


def test_run_button_is_primary():
    """The main Run action is a primary button (visual hierarchy)."""
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=120)
    at.run()
    runs = [b for b in at.button if "Run Simulation" in (b.label or "")]
    assert runs and runs[0].proto.type == "primary"
