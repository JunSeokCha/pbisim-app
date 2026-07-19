"""Phase 2 integration: the assistant's configure_simulator tool populates the app's
Interactive Simulator (real session state), and offers to open it.

Drives the real app with a fake agent whose generate() calls the real configure handler
(apply_ai_configuration) with a canned config — so no API is needed, but the actual
load_preset_to_state path runs against real st.session_state.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from streamlit.testing.v1 import AppTest
from pbisim_app.agent import AgentRun


_CONFIG = {
    "strains": [
        {"name": "WT", "growth_rate": 1.2, "initial_B": 1e7},
        {"name": "resistant", "growth_rate": 1.1, "initial_B": 10},
    ],
    "phages": [{"name": "P0", "burst_sizes": 50, "latent_periods": 0.5,
                "adsorption_rates": [1e-8, 0.0]}],
    "t_end": 24.0,
}


class _ConfiguringAgent:
    """Fake agent: its generate() invokes the real configure handler and returns a
    configured AgentRun (no API)."""
    def __init__(self):
        self.client = SimpleNamespace(api_key="x")
        self.model = "claude-opus-4-8"
        self.history = []
        self.last_usage = None

    def generate(self, prompt, execute, configure=None, max_tool_calls=6):
        summary = configure(_CONFIG)
        return AgentRun(f"Set it up — {summary}", "", None, False, 0, (),
                        configured=not summary.upper().startswith("ERROR"))


def test_configure_tool_populates_interactive_simulator():
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=120)
    at.run()

    # Land on the AI Assistant page with a fake, key-bearing agent so the chat renders.
    at.session_state["current_page_radio"] = "AI Assistant"
    at.session_state["api_key"] = "x"
    at.session_state["api_models_list"] = ["claude-opus-4-8"]
    at.session_state.agent = _ConfiguringAgent()
    at.run()

    # Submit a chat message → the fake agent applies the config to real session state.
    at.chat_input[0].set_value("set up a 2-strain phage model").run()
    assert len(at.exception) == 0, at.exception

    ss = at.session_state
    assert ss["int_builder_mode"] == "Direct (ModelBuilder)"
    assert [s["name"] for s in ss["int_strains"]] == ["WT", "resistant"]
    assert len(ss["int_phages"]) == 1 and ss["int_phages"][0]["name"] == "P0"
    # per-strain adsorption written to the pairwise keys the widgets read
    assert ss["ads_0_0"] == 1e-8 and ss["ads_1_0"] == 0.0
    assert ss["int_t_end"] == 24.0

    # the "Open in Interactive Simulator" button is offered
    assert any("Open in Interactive Simulator" in (b.label or "") for b in at.button)
