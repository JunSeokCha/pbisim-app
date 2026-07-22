"""Lightweight shared-credential gate for pbisim-app.

Purpose: keep the deployed app off the open internet when a URL is shared with a
small number of trusted people (e.g. a patent attorney). It is a basic password
gate, NOT enterprise SSO — adequate to stop casual/public access, not a substitute
for real isolation of the AI code-execution sandbox.

Configuration (set as environment variables on the host, e.g. Render — never
commit these):

    APP_PASSWORD        the shared password (plain), OR
    APP_PASSWORD_HASH   its SHA-256 hex digest (preferred; the password never
                        lives in the environment in cleartext)
    APP_USERNAME        optional; when set, the username must also match

When neither APP_PASSWORD nor APP_PASSWORD_HASH is set, the gate is DISABLED and
the app runs open (local development). So: set APP_PASSWORD[_HASH] on Render to
lock the deployment; leave it unset locally.
"""

from __future__ import annotations

import hashlib
import hmac
import os

import streamlit as st


def login_configured() -> bool:
    """True when a credential is configured in the environment (gate is active)."""
    return bool(os.environ.get("APP_PASSWORD") or os.environ.get("APP_PASSWORD_HASH"))


def _password_ok(password: str) -> bool:
    exp_hash = os.environ.get("APP_PASSWORD_HASH", "").strip().lower()
    if exp_hash:
        got = hashlib.sha256((password or "").encode("utf-8")).hexdigest()
        return hmac.compare_digest(exp_hash, got)
    exp_plain = os.environ.get("APP_PASSWORD", "")
    return hmac.compare_digest(exp_plain, password or "")


def _credentials_ok(username: str, password: str) -> bool:
    exp_user = os.environ.get("APP_USERNAME")
    user_ok = True if not exp_user else hmac.compare_digest(exp_user, username or "")
    return user_ok and _password_ok(password)


def require_login() -> None:
    """Block the app behind a sign-in form until authenticated. No-op when the
    gate isn't configured. Call once, early in app.py (after the page config/CSS,
    before session init / sidebar / page dispatch)."""
    if not login_configured():
        return
    if st.session_state.get("_authenticated"):
        return

    _wants_user = bool(os.environ.get("APP_USERNAME"))
    st.markdown(
        "<div style='max-width:400px;margin:10vh auto 0;text-align:center'>"
        "<div style='width:44px;height:44px;border-radius:10px;background:var(--teal);"
        "color:#fff;display:flex;align-items:center;justify-content:center;font-size:22px;"
        "font-weight:600;font-family:IBM Plex Mono,monospace;margin:0 auto 14px'>&#966;</div>"
        "<div style='font-size:1.3rem;font-weight:600;color:var(--ink)'>pbisim</div>"
        "<div class='section-label' style='margin:4px 0 18px'>Sign in to continue</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _c = st.columns([1, 2, 1])[1]
    with _c:
        with st.form("login_form"):
            _u = st.text_input("Username") if _wants_user else ""
            _p = st.text_input("Password", type="password")
            _ok = st.form_submit_button("Sign in", type="primary", width="stretch")
        if _ok:
            if _credentials_ok(_u, _p):
                st.session_state["_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect credentials.")
    st.stop()


def sign_out_control() -> None:
    """Render a small Sign-out control (only when the gate is active and signed in).
    Call inside the sidebar."""
    if login_configured() and st.session_state.get("_authenticated"):
        if st.button("Sign out", key="_sign_out", width="stretch"):
            st.session_state["_authenticated"] = False
            st.rerun()
