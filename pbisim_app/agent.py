"""
agent.py — Claude API integration for pbisim-app.

Translates natural-language simulation requests into pbisim Python code,
executes the code in a sandboxed namespace, and returns the results.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import NamedTuple

import anthropic


# ── Load system prompt ────────────────────────────────────────────────────────
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

# ── Claude model ──────────────────────────────────────────────────────────────
# Default to the strongest code model — one-shot accuracy matters more here than
# per-call cost, since a wrong first guess triggers a full extra round-trip. The
# sidebar dropdown lets the user switch to a faster/cheaper model when desired.
_MODEL = "claude-opus-4-8"


class AgentResponse(NamedTuple):
    """Structured response from the simulation agent."""
    code: str            # generated Python code
    narrative: str       # plain-English interpretation
    assumptions: str     # bullet list of assumptions
    raw_text: str        # full model response (for debugging)


class AgentRun(NamedTuple):
    """Result of an agentic (tool-using) generation.

    Unlike AgentResponse (single blind generation), this is produced by the model
    running its own code via the ``run_pbisim_code`` tool and correcting itself within
    one turn. ``result`` is the ExecutionResult of the last code the model ran (holds
    the figures/stdout to display); ``tool_calls`` counts how many times it ran code.
    """
    narrative: str       # the model's final explanation (after the code worked)
    code: str            # the last code the model executed
    result: object       # executor.ExecutionResult of that code (figures, stdout, error)
    success: bool        # did the final executed code run cleanly
    tool_calls: int      # number of run_pbisim_code calls (1 = ran code once)
    transcript: tuple = ()  # per-run ({code, success, error, figures}); [0] = the FIRST execution


# Tool the model uses to run and iterate on its code inside a single turn.
_RUN_TOOL = {
    "name": "run_pbisim_code",
    "description": (
        "Execute Python code in the pbisim sandbox and return its stdout, the number of "
        "matplotlib figures it created, and the full traceback if it raised. Use this to "
        "TEST and DEBUG your simulation code before answering: call it, read the result, "
        "and if it errored or produced no figure, fix the code and call it again. The "
        "sandbox already has the full pbisim API, numpy (np) and matplotlib (plt) loaded. "
        "When the code runs cleanly and produces the requested plot, stop calling the tool "
        "and give your final plain-English explanation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Complete, self-contained pbisim Python code."}
        },
        "required": ["code"],
    },
}

_TOOL_INSTRUCTION = (
    "You have a run_pbisim_code tool. Always use it to run your simulation code and "
    "verify it executes without error and produces the requested plot BEFORE giving your "
    "final answer. If it fails, read the traceback, fix the code, and run it again "
    "(up to a few attempts). Your LAST run_pbisim_code call must be the COMPLETE final "
    "script that produces the requested plot (and any printed values) — do not follow a "
    "working script with a separate diagnostic-only run, or the plot will be lost. "
    "Only after it succeeds, reply with a concise explanation of the result. Do not paste "
    "the code in your final message — the app captures it from your last tool call."
)


class SimulationAgent:
    """
    Wraps the Claude API to translate natural-language queries into
    pbisim simulation code.

    Parameters
    ----------
    api_key : str or None
        Anthropic API key.  If None, reads from ``ANTHROPIC_API_KEY`` env var.
    model : str
        Claude model identifier.
    max_tokens : int
        Maximum tokens in the response.
    history : list
        Conversation history (for multi-turn refinement).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _MODEL,
        max_tokens: int = 4096,
    ) -> None:
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens
        self.history: list[dict] = []

    def ask(self, user_message: str) -> AgentResponse:
        """
        Send a simulation request and return structured response.

        Parameters
        ----------
        user_message : str
            Natural-language simulation request from the user.

        Returns
        -------
        AgentResponse
            Contains generated code, narrative, and assumptions.
        """
        self.history.append({"role": "user", "content": user_message})
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # Send the long, static API reference as a cached content block. It is
                # re-used across every turn and every self-healing retry within the
                # 5-minute cache window, cutting latency and cost substantially.
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=self.history,
            )
        except Exception:
            self.history.pop()  # Keep history clean
            raise

        raw_text = response.content[0].text
        self.history.append({"role": "assistant", "content": raw_text})
        # Expose token usage of the most recent call (used by the eval harness for
        # cost/latency reporting; harmless for the app).
        self.last_usage = getattr(response, "usage", None)

        return _parse_response(raw_text)

    def generate(self, user_message: str, execute, max_tool_calls: int = 6) -> AgentRun:
        """Agentic generation: the model writes code, runs it via the ``run_pbisim_code``
        tool (``execute`` — typically ``executor.execute_code``), reads the result, and
        self-corrects within this single turn until the code runs and plots, then gives a
        final explanation. Returns an :class:`AgentRun` with the last executed code, its
        ExecutionResult (figures/stdout), and how many times it ran code.

        ``execute`` maps ``code:str -> ExecutionResult`` (``.success/.figures/.stdout/.error``).
        """
        _entry_len = len(self.history)   # roll back to here if this turn fails
        self.history.append({"role": "user", "content": user_message})
        system = [
            {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": _TOOL_INSTRUCTION},
        ]

        last_code, last_result, tool_calls, transcript = "", None, 0, []
        results_hist = []
        try:
            for _ in range(max_tool_calls):
                response = self.client.messages.create(
                    model=self.model, max_tokens=self.max_tokens,
                    system=system, tools=[_RUN_TOOL], messages=self.history,
                )
                self.last_usage = getattr(response, "usage", None)
                self.history.append({"role": "assistant", "content": response.content})

                tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
                if not tool_uses:
                    narrative = "".join(getattr(b, "text", "") for b in response.content
                                        if getattr(b, "type", None) == "text").strip()
                    display = _pick_display(last_result, results_hist)
                    ok = bool(display and display.success)
                    return AgentRun(narrative, last_code, display, ok, tool_calls, tuple(transcript))

                tool_results = []
                for tu in tool_uses:
                    code = (tu.input or {}).get("code", "")
                    result = execute(code) if code else _no_code_execution_result()
                    last_code, last_result, tool_calls = code, result, tool_calls + 1
                    results_hist.append(result)
                    transcript.append({"code": code, "success": bool(result.success),
                                       "error": result.error, "figures": len(result.figures)})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": _format_tool_result(result),
                        "is_error": not result.success,
                    })
                self.history.append({"role": "user", "content": tool_results})

            # Ran out of tool budget — return the best (last) attempt.
            display = _pick_display(last_result, results_hist)
            ok = bool(display and display.success)
            return AgentRun(
                "Reached the maximum number of code-execution attempts."
                + ("" if ok else " The last attempt still errored."),
                last_code, display, ok, tool_calls, tuple(transcript),
            )
        except Exception:
            # Roll the whole turn back — leaving a partial tool exchange (an assistant
            # tool_use with no matching tool_result) would break the next API call.
            del self.history[_entry_len:]
            raise

    def reset(self) -> None:
        """Clear conversation history (start a new simulation session)."""
        self.history.clear()


def _pick_display(last_result, history):
    """Which execution's figures/stdout to surface. Normally the last run, but if it
    produced no figure (e.g. the model ran a follow-up diagnostic that didn't re-plot),
    fall back to the most recent successful run that DID make a figure, so the user (and
    the eval) still see the plot."""
    if last_result is not None and getattr(last_result, "figures", None):
        return last_result
    for r in reversed(history):
        if r.success and r.figures:
            return r
    return last_result


def _no_code_execution_result():
    from pbisim_app.executor import ExecutionResult
    return ExecutionResult(success=False, figures=[], stdout="",
                           error="the tool call contained no code")


def _format_tool_result(result) -> str:
    """Compact textual feedback the model reads after running code."""
    parts = [f"figures_created: {len(result.figures)}"]
    out = (result.stdout or "").strip()
    if out:
        parts.append("stdout:\n" + out[:2000])
    if result.success:
        parts.append("status: SUCCESS")
    else:
        err = (result.error or "").strip()
        parts.append("status: ERROR (fix the code and run it again)\n" + err[-2500:])
    return "\n".join(parts)


# Any fenced block: captures an optional language tag and the body.
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_PBISIM_TOKENS = ("solve_ode", "ModelBuilder", "PBIModel", "BinaryResistanceGenotypes",
                  "StrainSet", "DoseSchedule", "import ", "plt.", "np.")


def _extract_code(text: str) -> str:
    """Pull the Python code out of a model response, robustly.

    Tolerates ```py / ```Python / bare ``` fences and multiple blocks: prefers
    Python-tagged (or untagged) blocks, and among those picks the one that most
    looks like a pbisim script (falling back to the longest). Returns "" if none.
    """
    blocks = [(lang.lower(), body) for lang, body in _FENCE_RE.findall(text)]
    if not blocks:
        return ""
    py = [b for lang, b in blocks if lang in ("python", "py", "python3", "")]
    candidates = py or [b for _, b in blocks]

    def score(body: str) -> tuple:
        hits = sum(tok in body for tok in _PBISIM_TOKENS)
        return (hits, len(body))

    return textwrap.dedent(max(candidates, key=score)).strip()


def _parse_response(text: str) -> AgentResponse:
    """Extract code block, narrative, and assumptions from model response."""
    code = _extract_code(text)

    # Remove ALL fenced blocks for narrative extraction (any language).
    text_no_code = _FENCE_RE.sub("", text).strip()

    # Extract assumptions bullet list (after "assumption" keyword)
    assumptions_match = re.search(
        r"(?i)(assumptions?.*?)(?:\n\n|\Z)", text_no_code, re.DOTALL
    )
    assumptions = assumptions_match.group(1).strip() if assumptions_match else ""

    # Narrative: everything that isn't the assumptions block
    narrative = text_no_code.replace(assumptions, "").strip()

    return AgentResponse(
        code=code,
        narrative=narrative,
        assumptions=assumptions,
        raw_text=text,
    )
