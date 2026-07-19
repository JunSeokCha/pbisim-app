"""Live eval runner for the pbisim-app AI assistant — CALLS THE REAL CLAUDE API.

This is intentionally NOT part of the pytest suite: it costs tokens and needs an API
key. Run it on demand to measure the assistant's one-shot success rate, retry rate and
latency, e.g. after changing the model, system prompt, or the generation loop.

    conda activate pbisim
    export ANTHROPIC_API_KEY=sk-...
    python -m evals.run_eval                 # all cases, default (Opus) model
    python -m evals.run_eval --limit 3       # smoke test on the first 3 cases
    python -m evals.run_eval --model claude-sonnet-4-6 --json out.json
    python -m evals.run_eval --list          # list cases without calling the API

The offline unit tests in tests/test_evals.py cover the harness logic (checkers +
loop + aggregation) with a fake agent, so this script's plumbing is CI-verified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from evals.cases import CASES
from evals.runner import run_case, summarize


def _select(cases, ids, tag, limit):
    if ids:
        wanted = set(ids.split(","))
        cases = [c for c in cases if c.id in wanted]
    if tag:
        cases = [c for c in cases if tag in c.tags]
    if limit:
        cases = cases[:limit]
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description="Live eval of the pbisim-app AI assistant.")
    ap.add_argument("--model", default=None, help="Claude model id (default: agent's default).")
    ap.add_argument("--max-retries", type=int, default=3, help="Self-healing retries per case.")
    ap.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    ap.add_argument("--ids", default=None, help="Comma-separated case ids to run.")
    ap.add_argument("--tag", default=None, help="Only run cases with this tag.")
    ap.add_argument("--json", dest="json_path", default=None, help="Write a JSON report here.")
    ap.add_argument("--dump", dest="dump_dir", default=None,
                    help="Write each case's final generated code + error to this directory (for diagnosis).")
    ap.add_argument("--list", action="store_true", help="List selected cases and exit.")
    args = ap.parse_args(argv)

    cases = _select(CASES, args.ids, args.tag, args.limit)

    if args.list:
        for c in cases:
            print(f"{c.id:22s} [{','.join(c.tags)}]  {c.prompt[:70]}...")
        print(f"\n{len(cases)} case(s).")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY to run the live eval.", file=sys.stderr)
        return 2

    # Import here so --list / --help work without the app/engine importable.
    from pbisim_app.agent import SimulationAgent, _MODEL
    from pbisim_app.executor import execute_code
    import matplotlib.pyplot as plt

    model = args.model or _MODEL
    est = len(cases) * (1 + args.max_retries)
    print(f"Running {len(cases)} case(s) on {model} "
          f"(up to ~{est} API calls, {args.max_retries} retries each).\n")

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    results = []
    for i, case in enumerate(cases, 1):
        agent = SimulationAgent(model=model)   # fresh history per case
        res = run_case(case, agent, execute_code, max_retries=args.max_retries)
        results.append(res)
        plt.close("all")  # release figures produced during execution
        if args.dump_dir:
            # Full per-execution transcript, so we can see WHY a case ran code twice
            # (a real first-attempt error vs. the model voluntarily re-running).
            with open(os.path.join(args.dump_dir, f"{case.id}.py"), "w") as fh:
                fh.write(f"# case: {case.id}  passed={res.passed}  first_ok={res.first_ok}  "
                         f"executions={res.attempts}\n# prompt: {case.prompt}\n")
                fh.write(f"# failed checks: {res.failed_checks}\n\n{res.code}\n")
            for k, step in enumerate(res.transcript, 1):
                if not step["success"]:
                    with open(os.path.join(args.dump_dir, f"{case.id}.exec{k}.error.txt"), "w") as fh:
                        fh.write(f"# execution {k} failed:\n{step['error']}\n\n# code was:\n{step['code']}")
        # code: ✓ first try · ⟳ self-corrected · ✗ failed ; chat: 💬 answered · ✗ failed
        if getattr(res, "intent", "code") == "chat":
            flag = "💬" if res.passed else "✗"
        else:
            flag = "✓" if res.first_ok and res.passed else ("⟳" if res.passed else "✗")
        fails = ("  fail: " + ", ".join(res.failed_checks)) if res.failed_checks else ""
        print(f"[{i:2d}/{len(cases)}] {flag} {case.id:22s} [{getattr(res, 'intent', 'code')}] "
              f"exec={res.attempts} {res.latency_s:5.1f}s{fails}")

    summary = summarize(results)
    _code = [r for r in results if r.intent == "code"]
    _chat = [r for r in results if r.intent == "chat"]
    print("\n" + "=" * 66)
    print(f"code cases ({summary['n_code']}):")
    print(f"  first-pass success : {summary['first_pass_pct']:5.1f}%  "
          f"({sum(r.first_ok for r in _code)}/{summary['n_code']})   <- true one-shot")
    print(f"  self-corrected     : {summary['self_corrected_pct']:5.1f}%  (first run failed, fixed in-turn)")
    if _chat:
        print(f"chat cases ({summary['n_chat']}):")
        print(f"  answered w/o code  : {summary['chat_pass_pct']:5.1f}%  "
              f"({sum(r.passed for r in _chat)}/{summary['n_chat']})   <- routing")
    print(f"overall success    : {summary['overall_pct']:5.1f}%  "
          f"({sum(r.passed for r in results)}/{summary['n_cases']})")
    print(f"mean executions    : {summary['mean_attempts']:.2f}")
    print(f"mean latency       : {summary['mean_latency_s']:.1f}s")
    print(f"tokens (in/out)    : {summary['total_input_tokens']}/{summary['total_output_tokens']} "
          f"(cache-read {summary['total_cache_read_tokens']})")
    print("=" * 66)

    if args.json_path:
        payload = {
            "model": model,
            "summary": summary,
            "cases": [
                {"id": r.id, "intent": r.intent, "passed": r.passed, "one_shot": r.one_shot,
                 "first_ok": r.first_ok, "self_corrected": r.self_corrected,
                 "attempts": r.attempts, "latency_s": r.latency_s,
                 "failed_checks": r.failed_checks, "usage": r.usage}
                for r in results
            ],
        }
        with open(args.json_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
