"""Core eval runner — the deterministic, unit-tested part.

``run_case`` drives one case through the *same* generate → execute → self-healing-retry
loop the app uses (app.py), then applies the case's checks. The agent and the execute
function are injected so the whole loop can be exercised offline with a fake agent
(tests/test_evals.py); the live measurement in run_eval.py passes the real
``SimulationAgent`` and ``executor.execute_code``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from evals.checks import run_checks


def _no_code_result():
    from pbisim_app.executor import ExecutionResult
    return ExecutionResult(success=False, figures=[], stdout="",
                           error="the model returned no python code block")


def _usage_dict(usage):
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
    }


@dataclass
class CaseResult:
    id: str
    passed: bool
    one_shot: bool          # passed AND the model ran code exactly once (tool_calls==1)
    attempts: int           # number of code executions (tool_calls)
    latency_s: float
    check_rows: list        # list of (name, ok, msg)
    code: str = ""
    error: str = ""
    usage: dict = field(default_factory=dict)
    first_ok: bool = True   # did the FIRST execution succeed (the true one-shot analog)
    transcript: tuple = ()  # per-execution {code, success, error, figures}
    intent: str = "code"    # "code" | "chat" — segments the metrics

    @property
    def failed_checks(self):
        return [name for name, ok, _ in self.check_rows if not ok]

    @property
    def self_corrected(self):
        """The model's first execution failed but it fixed it in-turn."""
        return self.passed and not self.first_ok


def run_case(case, agent, execute, max_retries: int = 3, clock=time.perf_counter) -> CaseResult:
    """Run one EvalCase through the agent's tool-use loop and score it.

    ``agent.generate(prompt, execute, max_tool_calls)`` runs the code-and-self-correct
    loop and returns an AgentRun (``.code``, ``.result``, ``.tool_calls``). ``attempts``
    is the number of code executions (1 = solved without self-correction). ``max_retries``
    maps to ``max_tool_calls = max_retries + 1`` to match the old loop's budget.
    """
    t0 = clock()
    run = agent.generate(case.prompt, execute, max_tool_calls=max_retries + 1)
    latency = clock() - t0

    result = run.result if run.result is not None else _no_code_result()
    rows, passed = run_checks(case.checks, run.code, result)
    attempts = max(1, run.tool_calls)
    transcript = getattr(run, "transcript", ()) or ()
    first_ok = transcript[0]["success"] if transcript else result.success
    return CaseResult(
        id=case.id,
        passed=passed,
        one_shot=passed and attempts == 1,
        attempts=attempts,
        latency_s=latency,
        check_rows=rows,
        code=run.code,
        error=result.error,
        usage=_usage_dict(getattr(agent, "last_usage", None)),
        first_ok=bool(first_ok),
        transcript=transcript,
        intent=getattr(case, "intent", "code"),
    )


def summarize(results) -> dict:
    """Aggregate metrics across CaseResults."""
    n = len(results) or 1
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0
    in_tok = sum((r.usage.get("input_tokens") or 0) for r in results)
    out_tok = sum((r.usage.get("output_tokens") or 0) for r in results)
    cache_read = sum((r.usage.get("cache_read_input_tokens") or 0) for r in results)
    # Segment by intent: first-pass/self-corrected only make sense for code cases; chat
    # cases (Q&A) are scored purely on passing their checks (answered without running code).
    code = [r for r in results if r.intent == "code"]
    chat = [r for r in results if r.intent == "chat"]
    nc = len(code) or 1
    return {
        "n_cases": len(results),
        "n_code": len(code),
        "n_chat": len(chat),
        "one_shot_pct": 100.0 * sum(r.one_shot for r in code) / nc,
        "first_pass_pct": 100.0 * sum(r.first_ok for r in code) / nc,      # code: first execution clean
        "self_corrected_pct": 100.0 * sum(r.self_corrected for r in code) / nc,
        "chat_pass_pct": (100.0 * sum(r.passed for r in chat) / len(chat)) if chat else None,
        "overall_pct": 100.0 * sum(r.passed for r in results) / n,
        "mean_attempts": _mean([r.attempts for r in results]),
        "mean_latency_s": _mean([r.latency_s for r in results]),
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "total_cache_read_tokens": cache_read,
    }
