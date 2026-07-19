"""Tests for the agentic self-correction loop (SimulationAgent.generate).

The old app-level "self-healing" retry loop was replaced by an in-turn tool-use loop:
the model runs its code via the run_pbisim_code tool, sees the traceback, and fixes it
within a single turn. These tests mock the Claude client (no API) and drive that loop.
"""

import pytest
from unittest.mock import MagicMock

from pbisim_app.agent import SimulationAgent
from pbisim_app.executor import ExecutionResult


def _blk(btype, text=None, id=None, inp=None, name="run_pbisim_code"):
    b = MagicMock()
    b.type, b.text, b.id, b.input, b.name = btype, text, id, inp, name
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
    assert run.transcript[0]["success"] is False and run.transcript[1]["success"] is True


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


def test_generate_recovers_figure_from_earlier_run():
    """If the model's LAST execution makes no figure (e.g. a follow-up print-only run),
    generate must surface the figure from the earlier run that did plot."""
    def execute(code):
        if "PLOT" in code:
            return ExecutionResult(success=True, figures=["fig"], stdout="", error="")
        return ExecutionResult(success=True, figures=[], stdout="value=1", error="")

    agent = SimulationAgent(api_key="mock-key")
    agent.client.messages.create = MagicMock(side_effect=[
        _resp([_blk("tool_use", id="t1", inp={"code": "PLOT the sim"})]),   # makes the figure
        _resp([_blk("tool_use", id="t2", inp={"code": "print stuff"})]),    # no figure
        _resp([_blk("text", text="Done.")]),
    ])
    run = agent.generate("plot it", execute)
    assert run.success is True
    assert len(run.result.figures) == 1   # figure recovered from the earlier run


def test_generate_answers_question_without_running_code():
    """Intent routing (Phase 1): a question the model answers in text — with no tool_use —
    must return the answer and run NO code (result None, tool_calls 0)."""
    agent = SimulationAgent(api_key="mock-key")
    agent.client.messages.create = MagicMock(side_effect=[
        _resp([_blk("text", text="A realistic lytic-phage adsorption rate is ~1e-8 mL/PFU/h.")]),
    ])
    run = agent.generate("what's a realistic adsorption rate?", _execute)
    assert run.tool_calls == 0
    assert run.result is None      # nothing was simulated
    assert run.code == ""
    assert "adsorption" in run.narrative.lower()


def test_generate_routes_to_configure_tool():
    """Phase 2: when the model calls configure_simulator, generate invokes the handler and
    marks the run configured — without running any code."""
    applied = {}
    def configure(cfg):
        applied.update(cfg)
        return "Configured the Interactive Simulator (Direct mode): 2 strain(s); t_end=24.0 h"

    agent = SimulationAgent(api_key="mock-key")
    agent.client.messages.create = MagicMock(side_effect=[
        _resp([_blk("tool_use", id="c1", name="configure_simulator",
                    inp={"strains": [{"name": "WT"}, {"name": "R"}]})]),
        _resp([_blk("text", text="I've set up a 2-strain model in the simulator.")]),
    ])
    run = agent.generate("set up a 2-strain model", _execute, configure=configure)

    assert run.configured is True
    assert run.tool_calls == 0      # no code executed
    assert run.result is None
    assert applied.get("strains")   # the handler received the config
    assert "set up" in run.narrative.lower()


def test_configure_error_lets_model_fall_back_to_code():
    """If configure_simulator reports ERROR, the run isn't marked configured and the model
    can proceed to run_pbisim_code (generality preserved)."""
    def configure(cfg):
        return "ERROR: configuration needs at least one strain."

    agent = SimulationAgent(api_key="mock-key")
    agent.client.messages.create = MagicMock(side_effect=[
        _resp([_blk("tool_use", id="c1", name="configure_simulator", inp={})]),
        _resp([_blk("tool_use", id="t1", name="run_pbisim_code", inp={"code": "GOOD"})]),
        _resp([_blk("text", text="Done via code.")]),
    ])
    run = agent.generate("do something", _execute, configure=configure)
    assert run.configured is False
    assert run.tool_calls == 1 and run.success is True


def test_generate_routes_to_summary_tool():
    """Phase 3: to interpret results the model calls get_simulation_summary, reads the
    metrics, and answers — running no code and configuring nothing."""
    called = {}
    def summarize(inp):
        called["yes"] = True
        return "Total bacteria: start 1e7, end 5e2. Time to clearance: 12.0 h."

    agent = SimulationAgent(api_key="mock-key")
    agent.client.messages.create = MagicMock(side_effect=[
        _resp([_blk("tool_use", id="s1", name="get_simulation_summary", inp={})]),
        _resp([_blk("text", text="Your infection cleared by ~12 h — the phage drove it down.")]),
    ])
    run = agent.generate("why did it clear?", _execute, summarize=summarize)
    assert called.get("yes")
    assert run.tool_calls == 0 and run.result is None and run.configured is False
    assert "cleared" in run.narrative.lower()


def test_generate_uses_api_lookup_then_codes():
    """The built-in pbisim_api_lookup tool grounds the model before it writes code."""
    agent = SimulationAgent(api_key="mock-key")
    agent.client.messages.create = MagicMock(side_effect=[
        _resp([_blk("tool_use", id="l1", name="pbisim_api_lookup", inp={"name": "ModelBuilder.with_phage_params"})]),
        _resp([_blk("tool_use", id="t1", name="run_pbisim_code", inp={"code": "GOOD"})]),
        _resp([_blk("text", text="Done.")]),
    ])
    run = agent.generate("simulate carefully", _execute)
    assert run.success is True and run.tool_calls == 1


def test_trim_history_bounds_conversation():
    """Long chats must not grow the API history without limit (memory/cost on the host)."""
    agent = SimulationAgent(api_key="mock-key")
    for k in range(12):   # 12 turns, some with a tool exchange
        agent.history.append({"role": "user", "content": f"turn {k}"})
        agent.history.append({"role": "assistant", "content": [_blk("tool_use", id=f"t{k}", inp={"code": "GOOD"})]})
        agent.history.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{k}", "content": "ok"}]})
    agent._trim_history(max_turns=8)
    user_prompts = [m for m in agent.history if m["role"] == "user" and isinstance(m["content"], str)]
    assert len(user_prompts) == 8
    assert agent.history[0] == {"role": "user", "content": "turn 4"}   # oldest kept, at a clean boundary
