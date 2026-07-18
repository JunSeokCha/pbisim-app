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

    def reset(self) -> None:
        """Clear conversation history (start a new simulation session)."""
        self.history.clear()


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
