"""Tests for the agentic self-correction loop (SimulationAgent.generate).

The old app-level "self-healing" retry loop was replaced by an in-turn tool-use loop:
the model runs its code via the run_pbisim_code tool, sees the traceback, and fixes it
within a single turn. These tests mock the Claude client (no API) and drive that loop.
"""

import pytest
from unittest.mock import MagicMock

from pbisim_app.agent import SimulationAgent
from pbisim_app.executor import ExecutionResult


def _blk(btype, text=None, id=None, inp=None):
    b = MagicMock()
    b.type, b.text, b.id, b.input = btype, text, id, inp
    return b


def _resp(content):
    r = MagicMock()
    r.content = content
    r.usage = None
    return r


def _execute(code):
    """Fake sandbox: code containing GOOD succeeds, anything else raises."""
    if "GOOD" in code:
        return ExecutionResult(success=True, figures=["fig"], stdout="ok", error="")
    return ExecutionResult(success=False, figures=[], stdout="", error="Boom traceback")


def test_generate_self_corrects_within_one_turn():
    """A failing first attempt is fixed after the model sees the traceback; generate
    returns the working code + a full, well-formed history for the next turn."""
    agent = SimulationAgent(api_key="mock-key")
    agent.client.messages.create = MagicMock(side_effect=[
        _resp([_blk("tool_use", id="t1", inp={"code": "BAD"})]),
        _resp([_blk("tool_use", id="t2", inp={"code": "GOOD"})]),
        _resp([_blk("text", text="Here is the result.")]),
    ])

    run = agent.generate("simulate something", _execute)

    assert run.success is True
    assert run.tool_calls == 2          # ran once, corrected, ran again
    assert run.code == "GOOD"
    # user, assistant(tool_use), tool_result, assistant(tool_use), tool_result, assistant(text)
    assert len(agent.history) == 6
    assert agent.history[0] == {"role": "user", "content": "simulate something"}


def test_generate_respects_tool_budget():
    """If the model never fixes the code, generate stops at max_tool_calls (no infinite
    loop) and reports the failing result."""
    agent = SimulationAgent(api_key="mock-key")
    agent.client.messages.create = MagicMock(side_effect=[
        _resp([_blk("tool_use", id="t1", inp={"code": "BAD"})]),
        _resp([_blk("tool_use", id="t2", inp={"code": "BAD"})]),
    ])
    run = agent.generate("simulate", _execute, max_tool_calls=2)
    assert run.success is False
    assert run.tool_calls == 2


def test_generate_rolls_back_history_on_api_error():
    """An exception mid-turn must leave history exactly as it was before the turn —
    never a dangling tool_use without its tool_result."""
    agent = SimulationAgent(api_key="mock-key")
    agent.history.append({"role": "user", "content": "earlier turn"})
    agent.history.append({"role": "assistant", "content": "earlier reply"})
    before = list(agent.history)

    # Fails on the 2nd API call — after a tool_use/tool_result pair is already in history.
    agent.client.messages.create = MagicMock(side_effect=[
        _resp([_blk("tool_use", id="t1", inp={"code": "BAD"})]),
        RuntimeError("api down"),
    ])
    with pytest.raises(RuntimeError):
        agent.generate("new prompt", _execute)

    assert agent.history == before   # whole turn rolled back
