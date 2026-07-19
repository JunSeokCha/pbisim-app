"""Offline tests for the eval harness (no API calls).

Covers the checkers, the case set's well-formedness, and the full run_case loop
(generate → execute → self-healing retry → checks) driven by a FAKE agent returning
canned code, so the harness plumbing is CI-verified without touching the Claude API.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from pbisim_app.executor import ExecutionResult, execute_code


@pytest.fixture(autouse=True)
def _close_figures():
    # matplotlib figures are global; the app and CLI close them after each run, so the
    # sandbox's new-figure detection starts clean. Mirror that between tests.
    plt.close("all")
    yield
    plt.close("all")
from evals import checks
from evals.cases import CASES, case_by_id
from evals.runner import run_case, summarize, CaseResult


# ── checkers ────────────────────────────────────────────────────────────────────
_OK = ExecutionResult(success=True, figures=["fig"], stdout="time to clearance: 12 h", error="")
_FAIL = ExecutionResult(success=False, figures=[], stdout="", error="NameError: x")


def test_runs_ok():
    assert checks.runs_ok()("code", _OK)[0] is True
    assert checks.runs_ok()("code", _FAIL)[0] is False


def test_has_figure():
    assert checks.has_figure(1)("c", _OK)[0] is True
    assert checks.has_figure(2)("c", _OK)[0] is False


def test_stdout_contains():
    assert checks.stdout_contains("clearance")("c", _OK)[0] is True
    assert checks.stdout_contains("nope")("c", _OK)[0] is False


def test_code_contains_and_absent():
    assert checks.code_contains("solve_ode")("x = solve_ode()", _OK)[0] is True
    assert checks.code_absent("eval(")("x = solve_ode()", _OK)[0] is True
    assert checks.code_absent("solve")("x = solve_ode()", _OK)[0] is False


def test_uses_cfu_sum_prefixes():
    good = "cfu = result.sum_prefixes('B','D','I','H')"
    assert checks.uses_cfu_sum_prefixes()(good, _OK)[0] is True
    assert checks.uses_cfu_sum_prefixes()("cfu = result.get('B0')", _OK)[0] is False


def test_run_checks_aggregates():
    rows, passed = checks.run_checks([checks.runs_ok(), checks.has_figure()], "c", _OK)
    assert passed is True and len(rows) == 2
    rows, passed = checks.run_checks([checks.runs_ok(), checks.has_figure()], "c", _FAIL)
    assert passed is False


def test_check_that_raises_is_caught():
    def bad(code, result):
        raise RuntimeError("boom")
    rows, passed = checks.run_checks([bad], "c", _OK)
    assert passed is False and "raised" in rows[0][2]


# ── case set well-formedness ────────────────────────────────────────────────────
def test_cases_wellformed():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    assert len(CASES) >= 12
    for c in CASES:
        assert c.prompt.strip()
        assert c.checks, f"{c.id} has no checks"
        assert all(callable(chk) for chk in c.checks)
    assert case_by_id(CASES[0].id) is CASES[0]


# ── run_case loop, driven by a fake agent (no API) ──────────────────────────────
from pbisim_app.agent import AgentRun


class _FakeAgent:
    """Simulates SimulationAgent.generate's tool loop offline: runs the queued code
    strings (repeating the last one) until one succeeds or the tool budget is spent."""
    def __init__(self, codes):
        self._codes = list(codes)
        self.calls = 0
        self.last_usage = None

    def generate(self, prompt, execute, max_tool_calls=6):
        last_code, last_result, n, transcript = "", None, 0, []
        for _ in range(max_tool_calls):
            code = self._codes[min(self.calls, len(self._codes) - 1)]
            self.calls += 1
            n += 1
            last_code = code
            last_result = execute(code) if code else _fake_no_code()
            transcript.append({"code": code, "success": bool(last_result.success),
                               "error": last_result.error, "figures": len(last_result.figures)})
            if last_result.success:
                break
        ok = bool(last_result and last_result.success)
        return AgentRun("narrative", last_code, last_result, ok, n, tuple(transcript))


def _fake_no_code():
    return ExecutionResult(success=False, figures=[], stdout="", error="no code")


_GOOD_CODE = (
    "import numpy as np\n"
    "from pbisim.builder import ModelBuilder\n"
    "from pbisim.core.model import PBIModel\n"
    "from pbisim.core.solver import solve_ode\n"
    "import matplotlib.pyplot as plt\n"
    "cfg = ModelBuilder(n_bacteria=1, n_phages=0).with_growth_rates(1.2).build()\n"
    "m = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([]))\n"
    "r = solve_ode(m, t_end=10.0, dt=1.0)\n"
    "plt.plot(r.time, r.sum_prefixes('B','D','I','H'))\n"
    "print('done')\n"
)
_BAD_CODE = "raise ValueError('boom')\n"


def _simple_case():
    return CASES[0].__class__(
        "unit", "prompt",
        [checks.runs_ok(), checks.has_figure(), checks.uses_cfu_sum_prefixes()])


def test_run_case_one_shot():
    res = run_case(_simple_case(), _FakeAgent([_GOOD_CODE]), execute_code, clock=lambda: 0.0)
    assert res.passed and res.one_shot and res.attempts == 1
    assert res.first_ok and not res.self_corrected


def test_run_case_recovers_after_retry():
    agent = _FakeAgent([_BAD_CODE, _GOOD_CODE])   # fails once, then fixes it
    res = run_case(_simple_case(), agent, execute_code, max_retries=3, clock=lambda: 0.0)
    assert res.passed and not res.one_shot and res.attempts == 2
    # the true one-shot metric: first execution failed, model self-corrected
    assert res.first_ok is False and res.self_corrected is True


def test_run_case_exhausts_retries():
    agent = _FakeAgent([_BAD_CODE])   # always bad
    res = run_case(_simple_case(), agent, execute_code, max_retries=2, clock=lambda: 0.0)
    assert not res.passed and res.attempts == 3  # 1 + 2 retries
    assert "runs_ok" in res.failed_checks


def test_run_case_no_code():
    res = run_case(_simple_case(), _FakeAgent([""]), execute_code, max_retries=1, clock=lambda: 0.0)
    assert not res.passed


def test_summarize():
    rows = [("runs_ok", True, "")]
    results = [
        CaseResult("a", True, True, 1, 2.0, rows, first_ok=True),       # first try
        CaseResult("b", True, False, 2, 4.0, rows, first_ok=False),     # self-corrected
        CaseResult("c", False, False, 4, 6.0, rows, first_ok=False),    # failed
    ]
    s = summarize(results)
    assert s["n_cases"] == 3
    assert round(s["one_shot_pct"], 1) == 33.3
    assert round(s["first_pass_pct"], 1) == 33.3      # only "a" ran clean first
    assert round(s["self_corrected_pct"], 1) == 33.3  # only "b" recovered
    assert round(s["overall_pct"], 1) == 66.7
    assert s["mean_attempts"] == (1 + 2 + 4) / 3


# ── the REAL agent.generate tool loop, with a mocked Claude client (no API) ──

from pbisim_app.agent import SimulationAgent


class _Blk:
    def __init__(self, type, text=None, id=None, inp=None):
        self.type, self.text, self.id, self.input = type, text, id, inp


class _MsgResp:
    def __init__(self, content):
        self.content = content
        self.usage = None


class _FakeMessages:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def create(self, **kw):
        r = self._responses[self.calls]
        self.calls += 1
        return r


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_generate_tool_loop_self_corrects():
    """generate() must run tool code, feed the traceback back, self-correct, and return
    the working code + its result — all within one turn."""
    agent = SimulationAgent(api_key="x")
    agent.client = _FakeClient([
        _MsgResp([_Blk("tool_use", id="t1", inp={"code": _BAD_CODE})]),    # first: fails
        _MsgResp([_Blk("tool_use", id="t2", inp={"code": _GOOD_CODE})]),   # fixed after seeing error
        _MsgResp([_Blk("text", text="Here is the simulation result.")]),   # final answer, no tool
    ])
    run = agent.generate("simulate something", execute_code)
    assert run.success is True
    assert run.tool_calls == 2          # ran code twice (self-corrected once)
    assert "sum_prefixes" in run.code   # kept the good code
    assert "result" in run.narrative.lower()
    assert run.transcript[0]["success"] is False and run.transcript[1]["success"] is True


def test_generate_respects_tool_budget():
    """If the model never fixes the code, generate stops at max_tool_calls and reports
    the failing result (no infinite loop)."""
    agent = SimulationAgent(api_key="x")
    agent.client = _FakeClient([
        _MsgResp([_Blk("tool_use", id="t1", inp={"code": _BAD_CODE})]),
        _MsgResp([_Blk("tool_use", id="t2", inp={"code": _BAD_CODE})]),
    ])
    run = agent.generate("simulate", execute_code, max_tool_calls=2)
    assert run.success is False
    assert run.tool_calls == 2


def test_stdout_contains_any():
    r = ExecutionResult(success=True, figures=["f"], stdout="2-log reduction at 3.1 h", error="")
    assert checks.stdout_contains_any(["clear", "reduction"])("c", r)[0] is True
    assert checks.stdout_contains_any(["clear", "nope"])("c", r)[0] is False
