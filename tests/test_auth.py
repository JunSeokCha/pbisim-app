"""The optional shared-credential gate (pbisim_app/auth.py).

Active only when APP_PASSWORD / APP_PASSWORD_HASH is set in the environment;
a no-op otherwise (so every other test runs unauthenticated).
"""

from __future__ import annotations

import hashlib

from streamlit.testing.v1 import AppTest

APP = "pbisim_app/app.py"


def test_gate_off_when_unconfigured(monkeypatch):
    """No credential in the environment -> the app runs open (local dev)."""
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("APP_PASSWORD_HASH", raising=False)
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert len(at.exception) == 0
    assert any(r.label == "Navigation" for r in at.radio)  # sidebar rendered


def test_gate_blocks_until_correct_password(monkeypatch):
    """With APP_PASSWORD set, the app shows only the sign-in form until the right
    password is entered; a wrong one is rejected."""
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    monkeypatch.delenv("APP_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("APP_USERNAME", raising=False)

    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    # blocked: no navigation, a password field is shown
    assert not any(r.label == "Navigation" for r in at.radio)
    pw = [t for t in at.text_input if t.label == "Password"]
    assert pw, "password field missing"

    # wrong password -> still blocked, error shown
    pw[0].set_value("nope")
    [b for b in at.button if "Sign in" in (b.label or "")][0].click().run()
    assert not any(r.label == "Navigation" for r in at.radio)
    assert any("Incorrect" in (e.value or "") for e in at.error)

    # correct password -> app unlocks
    [t for t in at.text_input if t.label == "Password"][0].set_value("s3cret")
    [b for b in at.button if "Sign in" in (b.label or "")][0].click().run()
    assert any(r.label == "Navigation" for r in at.radio)
    assert at.session_state["_authenticated"] is True


def test_password_hash_and_username(monkeypatch):
    """APP_PASSWORD_HASH (SHA-256) + APP_USERNAME are both honoured."""
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setenv("APP_PASSWORD_HASH", hashlib.sha256(b"pw123").hexdigest())
    monkeypatch.setenv("APP_USERNAME", "atty")
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    [t for t in at.text_input if t.label == "Username"][0].set_value("atty")
    [t for t in at.text_input if t.label == "Password"][0].set_value("pw123")
    [b for b in at.button if "Sign in" in (b.label or "")][0].click().run()
    assert any(r.label == "Navigation" for r in at.radio)
