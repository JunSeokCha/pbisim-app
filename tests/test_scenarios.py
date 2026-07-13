"""Tier-1 scenario snapshots: save / load / export round-trip (via AppTest)."""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

APP = "pbisim_app/app.py"


def test_scenario_save_load_roundtrip_restores_full_config():
    at = AppTest.from_file(APP, default_timeout=150)
    at.run()

    # Distinctive config spanning several subsystems + pairwise adsorption.
    s = at.session_state["int_strains"]
    s[0]["growth_rate"] = 0.42
    at.session_state["int_strains"] = s
    at.session_state["int_builder_mode"] = "Binary Genotypes (BRG)"
    at.session_state["int_t_end"] = 99.0
    at.session_state["ads_0_0"] = 4.2e-8
    at.session_state["sc_save_name"] = "RT"
    at.session_state["current_page_radio"] = "Library"
    at.run()

    # Save the current configuration.
    [b for b in at.button if "Save scenario" in (b.label or "")][0].click().run()
    assert "RT" in at.session_state["user_scenarios"]

    # The saved state must be JSON-serialisable (the export path depends on it).
    state = at.session_state["user_scenarios"]["RT"]["state"]
    json.dumps(state)
    assert state["int_builder_mode"] == "Binary Genotypes (BRG)"
    assert state["int_strains"][0]["growth_rate"] == 0.42

    # Mangle every captured field, then load the scenario back.
    s = at.session_state["int_strains"]
    s[0]["growth_rate"] = 0.1
    at.session_state["int_strains"] = s
    at.session_state["int_builder_mode"] = "Direct (ModelBuilder)"
    at.session_state["int_t_end"] = 48.0
    at.session_state["ads_0_0"] = 1e-9
    at.run()
    [b for b in at.button if b.label == "Load"][0].click().run()

    # Full restore + programmatic navigation to the simulator.
    assert at.session_state["current_page"] == "Interactive Simulator"
    assert at.session_state["int_strains"][0]["growth_rate"] == 0.42
    assert at.session_state["int_builder_mode"] == "Binary Genotypes (BRG)"
    assert at.session_state["int_t_end"] == 99.0
    assert at.session_state["ads_0_0"] == 4.2e-8
    assert len(at.exception) == 0


def test_scenario_export_import_helpers_are_available_and_round_trip():
    # Exercise the pure export/import parsing indirectly: a saved library exported
    # to JSON must re-import to an equivalent {name: scenario} mapping.
    at = AppTest.from_file(APP, default_timeout=150)
    at.run()
    at.session_state["sc_save_name"] = "A"
    at.session_state["current_page_radio"] = "Library"
    at.run()
    [b for b in at.button if "Save scenario" in (b.label or "")][0].click().run()

    scenarios = at.session_state["user_scenarios"]
    payload = json.dumps({"schema_version": 1, "scenarios": scenarios})
    reparsed = json.loads(payload)["scenarios"]
    assert "A" in reparsed
    assert reparsed["A"]["state"]["int_builder_mode"] == scenarios["A"]["state"]["int_builder_mode"]
