"""The Help page renders (no exceptions) and surfaces the bundled user guide."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from streamlit.testing.v1 import AppTest

APP = "pbisim_app/app.py"


def test_help_page_renders():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert "Help" in [o for r in at.radio for o in r.options]  # in the nav
    at.session_state["current_page_radio"] = "Help"
    at.run()
    assert len(at.exception) == 0, at.exception
    heads = " ".join(m.value for m in at.markdown)
    assert "Quick start" in heads
    assert "What each page does" in heads
    # the full USER_GUIDE.md is bundled and embedded
    assert any("pbisim-app User Guide" in m.value for m in at.markdown)


def test_help_page_title():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.session_state["current_page_radio"] = "Help"
    at.run()
    assert any("Help & User Guide" in (t.value or "") for t in at.title)
