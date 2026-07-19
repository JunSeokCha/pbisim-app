"""Phase 2 integration: the assistant's configure_simulator tool populates the app's
Interactive Simulator (real session state) across ALL builder modes, and the configured
state actually builds + runs.

Drives the real app with a fake agent whose generate() calls the real configure handler
(apply_ai_configuration) with a canned config — so no API is needed, but the actual
load_preset_to_state + mode setup runs against real st.session_state.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace
from streamlit.testing.v1 import AppTest
from pbisim_app.agent import AgentRun


class _ConfiguringAgent:
    """Fake agent: generate() invokes the real configure handler with a fixed config."""
    def __init__(self, config):
        self.config = config
        self.client = SimpleNamespace(api_key="x")
        self.model = "claude-opus-4-8"
        self.history = []
        self.last_usage = None

    def generate(self, prompt, execute, configure=None, summarize=None, max_tool_calls=6):
        summary = configure(self.config)
        return AgentRun(f"Set it up — {summary}", "", None, False, 0, (),
                        configured=not summary.upper().startswith("ERROR"))


def _configure(config):
    """Return an AppTest that has applied `config` via the assistant's configure handler."""
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=150)
    at.run()
    at.session_state["current_page_radio"] = "AI Assistant"
    at.session_state["api_key"] = "x"
    at.session_state["api_models_list"] = ["claude-opus-4-8"]
    at.session_state.agent = _ConfiguringAgent(config)
    at.run()
    at.chat_input[0].set_value("set it up").run()
    assert len(at.exception) == 0, at.exception
    return at


def _run_simulator(at):
    """Navigate to the Interactive Simulator and click Run; return the AppTest."""
    at.session_state["current_page_radio"] = "Interactive Simulator"
    at.run()
    [b for b in at.button if "Run Simulation" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception
    return at


def test_configure_direct_mode():
    at = _configure({
        "builder_mode": "direct",
        "strains": [{"name": "WT", "growth_rate": 1.2, "initial_B": 1e7},
                    {"name": "resistant", "growth_rate": 1.1, "initial_B": 10}],
        "phages": [{"name": "P0", "burst_sizes": 50, "adsorption_rates": [1e-8, 0.0]}],
        "t_end": 24.0,
    })
    ss = at.session_state
    assert ss["int_builder_mode"] == "Direct (ModelBuilder)"
    assert [s["name"] for s in ss["int_strains"]] == ["WT", "resistant"]
    assert ss["ads_0_0"] == 1e-8 and ss["ads_1_0"] == 0.0
    assert ss["int_t_end"] == 24.0
    assert any("Open in Interactive Simulator" in (b.label or "") for b in at.button)
    _run_simulator(at)
    assert at.session_state["simulation_config"] is not None


def test_configure_brg_mode_builds():
    at = _configure({
        "builder_mode": "brg",
        "strains": [{"name": "base", "growth_rate": 1.2}],
        "phages": [{"name": "phi", "burst_sizes": 50, "latent_periods": 0.5,
                    "adsorption_s": 1e-8, "adsorption_r": 0.0, "fitness_cost": 0.05, "mu": 1e-7}],
        "antibiotics": [{"name": "cipro", "emax": 3.0, "ec50": 0.2, "emax_r": 0.3,
                         "ec50_r": 2.0, "k_elim": 0.3}],
        "t_end": 24.0,
    })
    ss = at.session_state
    assert ss["int_builder_mode"] == "Binary Genotypes (BRG)"
    assert ss["int_brg_base_growth"] == 1.2
    assert len(ss["int_phages"]) == 1 and len(ss["int_antibiotics"]) == 1
    _run_simulator(at)   # BRG genotype lattice actually builds + solves
    assert at.session_state["simulation_config"] is not None


def test_configure_strainset_mode_builds():
    at = _configure({
        "builder_mode": "strainset",
        "strains": [{"name": "WT", "growth_rate": 1.2, "initial_B": 1e7},
                    {"name": "mutant", "growth_rate": 1.1, "initial_B": 10}],
        "phages": [{"name": "P0", "burst_sizes": 50, "adsorption_rates": [1e-8, 0.0]}],
        "mutation_graph": [{"from": "WT", "to": "mutant", "rate": 1e-7}],
        "t_end": 24.0,
    })
    ss = at.session_state
    assert ss["int_builder_mode"] == "Custom Strains & Graph (StrainSet)"
    assert ss["int_transitions"] == [{"from": "WT", "to": "mutant", "rate": 1e-7}]
    _run_simulator(at)   # StrainSet + mutation graph actually builds + solves
    assert at.session_state["simulation_config"] is not None


class _SummarizingAgent:
    """Fake agent: generate() calls the real summarize handler and returns its text."""
    def __init__(self):
        self.client = SimpleNamespace(api_key="x")
        self.model = "claude-opus-4-8"
        self.history = []
        self.last_usage = None

    def generate(self, prompt, execute, configure=None, summarize=None, max_tool_calls=6):
        text = summarize({})
        return AgentRun(f"Here is what happened:\n{text}", "", None, False, 0, ())


def test_summary_reads_current_results():
    """Phase 3: after a sim runs, get_simulation_summary returns real metrics for the model."""
    at = _configure({
        "builder_mode": "direct",
        "strains": [{"name": "WT", "growth_rate": 1.2, "initial_B": 1e7}],
        "phages": [{"name": "P0", "burst_sizes": 80, "adsorption_rates": [5e-8]},],
        "doses": [{"time": 0.0, "amount": 1e8, "target_type": "phage", "target_idx": 0}],
        "t_end": 24.0,
    })
    _run_simulator(at)   # produces simulation_result
    assert at.session_state["simulation_result"] is not None

    at.session_state["current_page_radio"] = "AI Assistant"
    at.session_state.agent = _SummarizingAgent()
    at.run()
    at.chat_input[0].set_value("interpret the results").run()
    assert len(at.exception) == 0, at.exception

    # the rendered answer contains the real summary metrics
    md = " ".join((m.value or "") for m in at.markdown)
    assert "Total bacteria" in md
    assert "clearance" in md.lower()
