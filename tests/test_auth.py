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


# ── persistent-login cookie (signed, expiring) ──
def test_auth_cookie_token_sign_and_verify(monkeypatch):
    """The cookie token is HMAC-signed + self-expiring: a fresh one verifies, an expired
    or tampered one doesn't, and rotating the password (the signing secret) invalidates it."""
    from pbisim_app import auth
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    monkeypatch.delenv("APP_AUTH_SECRET", raising=False)
    monkeypatch.delenv("APP_PASSWORD_HASH", raising=False)
    monkeypatch.setenv("APP_AUTH_TTL_HOURS", "24")
    now = 1_000_000.0
    tok = auth._make_token(now)
    assert auth._token_valid(tok, now)                     # fresh
    assert auth._token_valid(tok, now + 23 * 3600)         # within TTL
    assert not auth._token_valid(tok, now + 25 * 3600)     # expired
    assert not auth._token_valid("garbage", now)           # malformed
    assert not auth._token_valid(tok[:-1] + ("0" if tok[-1] != "0" else "1"), now)  # bad signature
    monkeypatch.setenv("APP_PASSWORD", "rotated")          # secret changed
    assert not auth._token_valid(tok, now)


def test_valid_cookie_restores_login_without_form(monkeypatch):
    """A valid signed cookie logs the browser straight in — no password form (the fix for
    being bounced to login after a reconnect / restart)."""
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    monkeypatch.delenv("APP_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("APP_USERNAME", raising=False)
    from pbisim_app import auth
    monkeypatch.setattr(auth, "_read_cookie", lambda: auth._make_token(auth.time.time()))
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert at.session_state["_authenticated"] is True
    assert any(r.label == "Navigation" for r in at.radio)          # app rendered, no gate
    assert not [t for t in at.text_input if t.label == "Password"]  # no sign-in form


def test_invalid_cookie_still_prompts(monkeypatch):
    """A tampered/foreign cookie does NOT unlock the app."""
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    monkeypatch.delenv("APP_USERNAME", raising=False)
    from pbisim_app import auth
    monkeypatch.setattr(auth, "_read_cookie", lambda: "9999999999.deadbeef")
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not any(r.label == "Navigation" for r in at.radio)
    assert [t for t in at.text_input if t.label == "Password"]


def test_sign_out_suppresses_cookie_relogin(monkeypatch):
    """Signing out must not be immediately undone by the still-present request cookie
    (st.context.cookies is fixed for the connection) — the session suppresses auto-login."""
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    monkeypatch.delenv("APP_USERNAME", raising=False)
    from pbisim_app import auth
    monkeypatch.setattr(auth, "_read_cookie", lambda: auth._make_token(auth.time.time()))
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert at.session_state["_authenticated"] is True              # logged in via cookie
    [b for b in at.button if b.key == "_sign_out"][0].click().run()
    at.run()   # let the gate's st.stop() render settle (AppTest applies it next run)
    assert at.session_state["_authenticated"] is False            # signed out
    assert at.session_state["_cookie_suppressed"] is True         # and not silently re-logged in
    assert not any(r.label == "Navigation" for r in at.radio)     # gate blocks again
