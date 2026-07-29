"""Automatic pass/fail checks for eval cases.

A *check* is a callable ``(code: str, result: ExecutionResult) -> (ok: bool, msg: str)``.
It sees the generated code (a string) and the outcome of running it in the app's real
sandbox (``pbisim_app.executor.ExecutionResult`` — success flag, figures, stdout, error).
``msg`` explains a failure (ignored when ``ok`` is True). Each factory returns a named
callable so reports can list which checks failed.
"""

from __future__ import annotations

import re


def runs_ok():
    """The generated code executed in the sandbox without raising."""
    def _c(code, result):
        return bool(result.success), ("" if result.success
                                      else f"execution failed:\n{result.error.strip().splitlines()[-1] if result.error else ''}")
    _c.__name__ = "runs_ok"
    return _c


def has_figure(n: int = 1):
    """At least ``n`` matplotlib figures were produced (the assistant plotted something)."""
    def _c(code, result):
        got = len(result.figures)
        return got >= n, ("" if got >= n else f"expected >={n} figure(s), got {got}")
    _c.__name__ = f"has_figure(>={n})"
    return _c


def stdout_contains(substr: str):
    """Printed output contains ``substr`` (case-insensitive)."""
    def _c(code, result):
        ok = substr.lower() in (result.stdout or "").lower()
        return ok, ("" if ok else f"stdout missing {substr!r}")
    _c.__name__ = f"stdout_contains({substr!r})"
    return _c


def stdout_contains_any(substrs):
    """Printed output contains at least one of ``substrs`` (case-insensitive)."""
    subs = tuple(substrs)
    def _c(code, result):
        low = (result.stdout or "").lower()
        ok = any(s.lower() in low for s in subs)
        return ok, ("" if ok else f"stdout contains none of {subs}")
    _c.__name__ = f"stdout_contains_any({subs})"
    return _c


def answer_contains_any(substrs):
    """The assistant's TEXT answer mentions at least one of ``substrs`` (case-insensitive).

    For chat/no-code cases the runner surfaces the model's narrative into ``result.stdout``,
    so this scores the prose answer for the expected mechanistic concepts (e.g. a synergy
    explanation naming "resistance" or "collateral sensitivity"). Keep the option list broad
    — accept any reasonable phrasing of the right idea."""
    subs = tuple(substrs)
    def _c(code, result):
        low = (result.stdout or "").lower()
        ok = any(s.lower() in low for s in subs)
        return ok, ("" if ok else f"answer mentions none of {subs}")
    _c.__name__ = f"answer_contains_any({subs})"
    return _c


def no_code_run():
    """The assistant answered WITHOUT generating/running a simulation (chat / interpret
    intent) — i.e. it routed to a text answer, not the code path."""
    def _c(code, result):
        ok = not (code or "").strip()
        return ok, ("" if ok else "expected a text answer, but a simulation was run")
    _c.__name__ = "no_code_run"
    return _c


def code_contains(substr: str):
    """Generated code contains the literal ``substr``."""
    def _c(code, result):
        ok = substr in (code or "")
        return ok, ("" if ok else f"code missing {substr!r}")
    _c.__name__ = f"code_contains({substr!r})"
    return _c


def code_absent(substr: str):
    """Generated code does NOT contain ``substr`` (guard against an anti-pattern)."""
    def _c(code, result):
        ok = substr not in (code or "")
        return ok, ("" if ok else f"code should not contain {substr!r}")
    _c.__name__ = f"code_absent({substr!r})"
    return _c


def code_matches(pattern: str):
    """Generated code matches the regular expression ``pattern``."""
    rx = re.compile(pattern, re.DOTALL)
    def _c(code, result):
        ok = bool(rx.search(code or ""))
        return ok, ("" if ok else f"code does not match /{pattern}/")
    _c.__name__ = f"code_matches({pattern!r})"
    return _c


def uses_cfu_sum_prefixes():
    """Bacterial totals are computed via ``result.sum_prefixes(...)`` (CFU = culturable
    ``('B','D')``; total live load = ``('B','D','I','H')``) rather than a bare ``get('B')``
    (ECOSYSTEM.md §3.2). The check just requires a ``sum_prefixes(`` aggregation."""
    def _c(code, result):
        ok = "sum_prefixes(" in (code or "")
        return ok, ("" if ok else "bacterial total should use result.sum_prefixes(...)")
    _c.__name__ = "uses_cfu_sum_prefixes"
    return _c


def run_checks(case_checks, code, result):
    """Run all checks for a case; return list of (name, ok, msg) and the overall pass."""
    rows = []
    for chk in case_checks:
        try:
            ok, msg = chk(code, result)
        except Exception as e:  # a buggy check must not crash the run
            ok, msg = False, f"check raised: {e}"
        rows.append((getattr(chk, "__name__", "check"), bool(ok), msg))
    passed = all(ok for _, ok, _ in rows)
    return rows, passed
