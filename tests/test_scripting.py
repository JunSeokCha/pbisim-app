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


def _src(at, key):
    return [t for t in at.text_area if t.key == key][0]


def test_add_cell_preserves_existing_source(monkeypatch):
    """Regression: adding a cell must NOT wipe other cells' code (the earlier st.rerun
    bug purged not-yet-rendered widget keys)."""
    at = _goto_scripting(monkeypatch, timeout=240)
    _src(at, "script_src_0").set_value("keep_me = 111").run()
    [b for b in at.button if "Add cell" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0
    assert _src(at, "script_src_0").value == "keep_me = 111"   # source survived the add
    assert any(t.key == "script_src_1" for t in at.text_area)  # new cell exists
    # the new cell shares the kernel — Run all executes cell 0 (defines keep_me) then cell 1
    _src(at, "script_src_1").set_value("print(keep_me)").run()
    [b for b in at.button if "Run all" in (b.label or "")][0].click().run()
    _o = at.session_state["script_outputs"][1]
    assert _o["success"] and "111" in _o["stdout"]


def test_delete_cell_removes_only_that_cell(monkeypatch):
    at = _goto_scripting(monkeypatch)
    # one cell → Delete is disabled (always keep at least one)
    assert [b for b in at.button if b.key == "script_del_0"][0].disabled is True
    [b for b in at.button if "Add cell" in (b.label or "")][0].click().run()
    [b for b in at.button if b.key == "script_del_0"][0].click().run()
    keys = sorted(t.key for t in at.text_area if t.key.startswith("script_src_"))
    assert keys == ["script_src_1"] and len(at.exception) == 0
