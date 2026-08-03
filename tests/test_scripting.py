"""Scripting page: executor kernel semantics + the default-off env-flag gate."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from pbisim_app.executor import execute_code, new_namespace

APP = "pbisim_app/app.py"


# ── executor: notebook 'kernel' vs the AI-assistant fresh namespace ──
def test_namespace_persists_across_cells():
    """A shared namespace carries variables across execute_code calls (cell state)."""
    ns = new_namespace()
    r1 = execute_code("a = 7", namespace=ns)
    r2 = execute_code("print(a * 6)", namespace=ns)
    assert r1.success and r2.success and r2.stdout.strip() == "42"


def test_fresh_namespace_is_isolated():
    """Default (namespace=None) is a fresh namespace per call — the AI-assistant path,
    unchanged: nothing leaks between independent executions."""
    execute_code("b = 123")
    r = execute_code("print(b)")
    assert not r.success and "NameError" in r.error


def test_restart_kernel_clears_state():
    ns = new_namespace()
    execute_code("kept = 1", namespace=ns)
    ns2 = new_namespace()          # Restart kernel = a brand-new namespace
    r = execute_code("print(kept)", namespace=ns2)
    assert not r.success and "NameError" in r.error


# ── app gate: hidden by default, shown only when PBISIM_ENABLE_SCRIPTING is set ──
def _nav(at):
    return [r for r in at.radio if r.label == "Navigation"][0]


def test_scripting_hidden_by_default(monkeypatch):
    monkeypatch.delenv("PBISIM_ENABLE_SCRIPTING", raising=False)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert "Scripting" not in list(_nav(at).options)
    assert len(at.exception) == 0


def test_scripting_page_renders_and_runs_when_enabled(monkeypatch):
    monkeypatch.setenv("PBISIM_ENABLE_SCRIPTING", "1")
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert "Scripting" in list(_nav(at).options)
    at.session_state["current_page_radio"] = "Scripting"
    at.run()
    assert len(at.exception) == 0
    assert any("Run all" in (b.label or "") for b in at.button)
    # Run the seeded demo cell end-to-end (page → executor → output).
    [b for b in at.button if "Run all" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0
    assert "script_outputs" in at.session_state          # SafeSessionState has no .get()
    _outs = at.session_state["script_outputs"]
    assert 0 in _outs and _outs[0]["success"], _outs


def _goto_scripting(monkeypatch, timeout=200):
    monkeypatch.setenv("PBISIM_ENABLE_SCRIPTING", "1")
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    at.session_state["current_page_radio"] = "Scripting"
    at.run()
    return at


# Cell source lives in the plain ``script_src_{cid}`` session key regardless of which
# editor renders it (code_editor component or the text_area fallback), so drive it there.
def test_add_cell_preserves_existing_source(monkeypatch):
    """Regression: adding a cell must NOT wipe other cells' code (the earlier st.rerun
    bug purged not-yet-rendered widget keys)."""
    at = _goto_scripting(monkeypatch, timeout=240)
    at.session_state["script_src_0"] = "keep_me = 111"
    at.run()
    [b for b in at.button if "Add cell" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0
    assert at.session_state["script_src_0"] == "keep_me = 111"   # source survived the add
    assert 1 in at.session_state["script_cell_ids"]              # new cell exists
    # the new cell shares the kernel — Run all executes cell 0 (defines keep_me) then cell 1
    at.session_state["script_src_1"] = "print(keep_me)"
    at.run()
    [b for b in at.button if "Run all" in (b.label or "")][0].click().run()
    _o = at.session_state["script_outputs"][1]
    assert _o["success"] and "111" in _o["stdout"]


def test_delete_cell_removes_only_that_cell(monkeypatch):
    at = _goto_scripting(monkeypatch)
    # one cell → Delete is disabled (always keep at least one)
    assert [b for b in at.button if b.key == "script_del_0"][0].disabled is True
    [b for b in at.button if "Add cell" in (b.label or "")][0].click().run()
    [b for b in at.button if b.key == "script_del_0"][0].click().run()
    assert at.session_state["script_cell_ids"] == [1] and len(at.exception) == 0


def test_editor_response_runs_once_per_submit(monkeypatch):
    """_apply_editor_response stores the buffer on any real event, runs only on a NEW
    submit id (so an unrelated rerun echoing the last 'submit' doesn't re-execute), and
    ignores a no-event response (empty type) so it can't clobber the stored source."""
    from pbisim_app.views import scripting
    store = {"script_src_9": "old"}
    monkeypatch.setattr(scripting.st, "session_state", store, raising=False)

    # no-event response (headless / first render): source untouched, no run
    assert scripting._apply_editor_response(9, {"type": "", "text": "", "id": ""}) is False
    assert store["script_src_9"] == "old"

    # a blur event stores the edited buffer but does not run
    assert scripting._apply_editor_response(9, {"type": "blur", "text": "x=1", "id": "a1"}) is False
    assert store["script_src_9"] == "x=1"

    # a submit runs once; the same submit echoed on a later rerun does NOT re-run
    assert scripting._apply_editor_response(9, {"type": "submit", "text": "x=2", "id": "s1"}) is True
    assert store["script_src_9"] == "x=2"
    assert scripting._apply_editor_response(9, {"type": "submit", "text": "x=2", "id": "s1"}) is False
    # a new submit (new id) runs again
    assert scripting._apply_editor_response(9, {"type": "submit", "text": "x=3", "id": "s2"}) is True


def test_textarea_fallback_when_editor_missing(monkeypatch):
    """With the code-editor component unavailable, cells fall back to st.text_area and
    still run end-to-end (so the base deploy without the [scripting] extra works)."""
    from pbisim_app.views import scripting
    monkeypatch.setattr(scripting, "_HAS_CODE_EDITOR", False)
    at = _goto_scripting(monkeypatch, timeout=240)
    assert any(t.key == "script_src_0" for t in at.text_area)   # fallback textarea present
    at.session_state["script_src_0"] = "print(6 * 7)"
    at.run()
    [b for b in at.button if b.key == "script_run_0"][0].click().run()
    _o = at.session_state["script_outputs"][0]
    assert _o["success"] and "42" in _o["stdout"] and len(at.exception) == 0
