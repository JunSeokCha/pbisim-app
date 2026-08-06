"""Tier-2 parts library: save / load / export (via AppTest)."""

from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest
from pathlib import Path as _Path

APP = str(_Path(__file__).resolve().parents[1] / "pbisim_app" / "app.py")


def _library_page():
    at = AppTest.from_file(APP, default_timeout=150)
    at.run()
    at.session_state["current_page_radio"] = "Library"
    at.run()
    return at


def test_save_and_load_bacterium_part_appends_entity():
    at = _library_page()
    at.session_state["part_name_bacteria"] = "PA"
    [b for b in at.button if b.key == "part_save_bacteria"][0].click().run()
    assert "PA" in at.session_state["parts_library"]["bacteria"]
    # the saved params match the current strain
    assert at.session_state["parts_library"]["bacteria"]["PA"]["params"]["name"] == \
        at.session_state["int_strains"][0]["name"]

    before = len(at.session_state["int_strains"])
    [b for b in at.button if b.key == "part_load_bacteria_PA"][0].click().run()
    # loading a part appends the entity to the shared config and navigates to the simulator
    assert len(at.session_state["int_strains"]) == before + 1
    assert at.session_state["current_page"] == "Interactive Simulator"
    assert len(at.exception) == 0


def test_phage_part_records_reference_host_and_library_exports():
    at = _library_page()
    at.session_state["part_name_phages"] = "phi"
    [b for b in at.button if b.key == "part_save_phages"][0].click().run()

    part = at.session_state["parts_library"]["phages"]["phi"]
    assert "params" in part
    assert "reference_host" in part  # phage parts are host-tagged
    assert part["source"] in ["educated guess", "literature", "pbisim-fit", "experimental"]

    # whole library is JSON-serialisable / re-importable
    lib = at.session_state["parts_library"]
    payload = json.dumps({"schema_version": 1, "parts": lib})
    reparsed = json.loads(payload)["parts"]
    assert "phi" in reparsed["phages"]
    assert reparsed["phages"]["phi"]["params"] == part["params"]
