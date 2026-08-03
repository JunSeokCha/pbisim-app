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
    APP_AUTH_TTL_HOURS  optional; "stay logged in" lifetime of the auth cookie
                        (default 168 = 7 days)
    APP_AUTH_SECRET     optional; overrides the cookie-signing secret (by default it
                        is derived from the configured credential, so rotating the
                        password invalidates every outstanding cookie)

When neither APP_PASSWORD nor APP_PASSWORD_HASH is set, the gate is DISABLED and
the app runs open (local development). So: set APP_PASSWORD[_HASH] on Render to
lock the deployment; leave it unset locally.

Persistence: on a successful sign-in a **signed, expiring cookie** is stored in the
browser, and on each fresh session it is read back server-side (``st.context.cookies``)
to restore the login. This keeps the user signed in across the WebSocket reconnects,
laptop-sleep, and server restarts/redeploys that otherwise wipe the in-memory
``st.session_state`` gate and bounce them to the password page. The cookie only proves
"this browser passed the password", signed with a server secret so it can't be forged;
it is not a session/identity token. Requires the ``streamlit-cookies-controller``
component to SET the cookie — absent it, the gate degrades to the old session-only
behaviour (still correct, just not persistent).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time

import streamlit as st

_COOKIE_NAME = "pbisim_auth"


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


# ── persistent-login cookie (signed, expiring) ────────────────────────────────
def _auth_secret() -> bytes:
    """HMAC key for the auth cookie. Derived from the configured credential so that
    rotating the password invalidates every outstanding cookie; overridable with
    APP_AUTH_SECRET."""
    base = (os.environ.get("APP_AUTH_SECRET")
            or os.environ.get("APP_PASSWORD_HASH")
            or os.environ.get("APP_PASSWORD", ""))
    return hashlib.sha256(("pbisim-auth-v1|" + base).encode("utf-8")).digest()


def _ttl_seconds() -> int:
    try:
        hours = float(os.environ.get("APP_AUTH_TTL_HOURS", "168"))  # default 7 days
    except ValueError:
        hours = 168.0
    return max(1, int(hours * 3600))


def _make_token(now: float) -> str:
    """A cookie value ``<expiry_epoch>.<hmac>`` that proves the password was passed and
    self-expires. Only the holder of the server secret can produce a valid signature."""
    exp = str(int(now) + _ttl_seconds())
    sig = hmac.new(_auth_secret(), exp.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def _token_valid(token: str, now: float) -> bool:
    """True iff *token* is well-formed, unexpired, and correctly signed for the current
    secret (constant-time signature compare)."""
    if not token or "." not in token:
        return False
    exp_s, _, sig = token.partition(".")
    try:
        exp = int(exp_s)
    except ValueError:
        return False
    if now >= exp:
        return False
    good = hmac.new(_auth_secret(), exp_s.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(good, sig)


def _read_cookie() -> str:
    """Read the auth cookie from the current request, server-side. Unlike a JS cookie
    component's ``get`` (which isn't available until a frontend round-trip — a login-form
    flash), ``st.context.cookies`` is populated immediately from the connection's request,
    which is exactly what we need to bootstrap a fresh session after a reconnect/restart."""
    try:
        return st.context.cookies.get(_COOKIE_NAME, "") or ""
    except Exception:
        return ""


def _cookie_controller():
    """The cookie component used only to SET/REMOVE the cookie (reads go via
    ``st.context``). Returns None when the component isn't installed."""
    try:
        from streamlit_cookies_controller import CookieController
        return CookieController(key="pbisim_auth_cookie")
    except Exception:
        return None


def _persist_cookie(token: str) -> None:
    ctrl = _cookie_controller()
    if ctrl is None:
        return
    try:
        ctrl.set(_COOKIE_NAME, token, max_age=_ttl_seconds(), same_site="lax")
    except Exception:
        pass


def _forget_cookie() -> None:
    ctrl = _cookie_controller()
    if ctrl is None:
        return
    try:
        ctrl.remove(_COOKIE_NAME, same_site="lax")
    except Exception:
        pass


def require_login() -> None:
    """Block the app behind a sign-in form until authenticated. No-op when the
    gate isn't configured. Call once, early in app.py (after the page config/CSS,
    before session init / sidebar / page dispatch)."""
    if not login_configured():
        return

    now = time.time()
    # (a) A just-completed sign-in queued a cookie to persist. Write it here, on a clean
    #     render (the login handler can't set it and st.rerun() in the same run — the
    #     rerun would abort the component before its JS runs).
    _pending = st.session_state.pop("_pending_cookie", None)
    if _pending:
        _persist_cookie(_pending)
    # (b) A just-completed sign-out queued a cookie removal. Delete it AND suppress the
    #     auto-login below for the rest of this session: st.context.cookies is fixed for
    #     the connection's lifetime, so the (now-stale) request cookie would otherwise log
    #     us straight back in until the next reconnect.
    if st.session_state.pop("_forget_cookie", False):
        _forget_cookie()
        st.session_state["_cookie_suppressed"] = True

    if st.session_state.get("_authenticated"):
        return

    # (c) Persistent login: restore the session from a valid signed cookie.
    if not st.session_state.get("_cookie_suppressed"):
        _tok = _read_cookie()
        if _tok and _token_valid(_tok, now):
            st.session_state["_authenticated"] = True
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
                # Persist across reconnects/restarts: queue the signed cookie; it's
                # written on the next (clean) render — see require_login step (a).
                st.session_state["_pending_cookie"] = _make_token(time.time())
                st.session_state.pop("_cookie_suppressed", None)
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
            # Drop the persistent cookie too, else the next render restores the login.
            st.session_state["_forget_cookie"] = True
            st.rerun()
