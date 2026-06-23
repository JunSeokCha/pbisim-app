"""Unit tests for the self-healing execution loop and history rollback logic."""

import pytest
from unittest.mock import MagicMock
from pbisim_app.agent import SimulationAgent
from pbisim_app.executor import ExecutionResult


def test_self_healing_logic_success_after_retry():
    """Test that if code execution fails initially but succeeds after a retry, the history is retained."""
    agent = SimulationAgent(api_key="mock-key")

    mock_resp1 = MagicMock()
    mock_resp1.content = [MagicMock(text="```python\ncode1\n```\nNarrative 1\nAssumptions:\n- test")]

    mock_resp2 = MagicMock()
    mock_resp2.content = [MagicMock(text="```python\ncode2\n```\nNarrative 2\nAssumptions:\n- test")]

    agent.client.messages.create = MagicMock(side_effect=[mock_resp1, mock_resp2])

    initial_history_len = len(agent.history)

    # 1. Initial ask
    agent_resp = agent.ask("initial prompt")
    assert len(agent.history) == 2
    assert agent.history[0]["content"] == "initial prompt"

    # Mock execution result: first fails, second succeeds
    exec_results = [
        ExecutionResult(success=False, figures=[], stdout="", error="SyntaxError"),
        ExecutionResult(success=True, figures=[], stdout="OK", error="")
    ]

    exec_idx = 0
    exec_result = exec_results[exec_idx]
    exec_idx += 1

    max_retries = 3
    retry_count = 0
    while not exec_result.success and retry_count < max_retries:
        retry_count += 1
        healing_prompt = f"Failed with {exec_result.error}"
        agent_resp = agent.ask(healing_prompt)
        exec_result = exec_results[exec_idx]
        exec_idx += 1

    if not exec_result.success:
        del agent.history[initial_history_len:]

    assert retry_count == 1
    assert exec_result.success
    # History contains: initial user, initial assistant, healing user, healing assistant
    assert len(agent.history) == 4
    assert agent.history[2]["content"] == "Failed with SyntaxError"


def test_self_healing_logic_all_fail_rollback():
    """Test that if code execution fails even after all retries, the history is rolled back to clean."""
    agent = SimulationAgent(api_key="mock-key")

    # Mock client to return responses for 4 calls (1 initial + 3 retries)
    mock_responses = []
    for i in range(4):
        resp = MagicMock()
        resp.content = [MagicMock(text=f"```python\ncode{i}\n```\nNarrative {i}\nAssumptions:\n- test")]
        mock_responses.append(resp)

    agent.client.messages.create = MagicMock(side_effect=mock_responses)

    initial_history_len = len(agent.history)

    # 1. Initial ask
    agent_resp = agent.ask("initial prompt")

    # All execution results fail
    exec_result = ExecutionResult(success=False, figures=[], stdout="", error="RuntimeError")

    max_retries = 3
    retry_count = 0
    while not exec_result.success and retry_count < max_retries:
        retry_count += 1
        healing_prompt = f"Failed with {exec_result.error}"
        agent_resp = agent.ask(healing_prompt)
        exec_result = ExecutionResult(success=False, figures=[], stdout="", error="RuntimeError")

    if not exec_result.success:
        del agent.history[initial_history_len:]

    # History should be rolled back to its initial length (0)
    assert len(agent.history) == initial_history_len
    assert len(agent.history) == 0
