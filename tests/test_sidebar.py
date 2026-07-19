"""Sidebar behavior — the API-key field must not linger as a password input.

Re-rendering a masked (password) field on every navigation makes Chrome repeatedly
offer to save the password. The app hides the field once a key is set; this test locks
that in.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from streamlit.testing.v1 import AppTest


def _api_key_field_shown(at):
    return any((ti.label or "") == "Anthropic API Key" for ti in at.text_input)


def test_api_key_field_hidden_once_set():
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=90)
    at.run()

    # With a key in place, the password field must NOT be rendered (so the browser has
    # no field to prompt about) — a compact status + Change button is shown instead.
    at.session_state["api_key"] = "sk-test-123"
    at.session_state["_editing_api_key"] = False
    at.run()
    assert not _api_key_field_shown(at)
    assert any("Change" in (b.label or "") for b in at.button)


def test_change_button_reveals_field():
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=90)
    at.run()
    at.session_state["api_key"] = "sk-test-123"
    at.session_state["_editing_api_key"] = False
    at.run()
    assert not _api_key_field_shown(at)

    # Clicking Change re-opens the input so the user can enter a new key.
    [b for b in at.button if "Change" in (b.label or "")][0].click().run()
    assert _api_key_field_shown(at)
