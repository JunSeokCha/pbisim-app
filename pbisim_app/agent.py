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
_MODEL = "claude-sonnet-4-6"   # update when newer versions are available
# For highest code-generation quality, use "claude-opus-4-8" instead.


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

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=self.history,
        )

        raw_text = response.content[0].text
        self.history.append({"role": "assistant", "content": raw_text})

        return _parse_response(raw_text)

    def reset(self) -> None:
        """Clear conversation history (start a new simulation session)."""
        self.history.clear()


def _parse_response(text: str) -> AgentResponse:
    """Extract code block, narrative, and assumptions from model response."""
    # Extract Python code block
    code_match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    code = textwrap.dedent(code_match.group(1)).strip() if code_match else ""

    # Remove the code block for narrative extraction
    text_no_code = re.sub(r"```python.*?```", "", text, flags=re.DOTALL).strip()

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
