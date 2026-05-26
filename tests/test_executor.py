"""Tests for the code execution sandbox."""

import sys
import pytest

sys.path.insert(0, r"C:\Users\pipte\Projects\pbisim\pbisim")

from pbisim_app.executor import execute_code, ExecutionResult


class TestExecuteCode:

    def test_simple_print(self):
        r = execute_code('print("hello pbisim")')
        assert r.success
        assert "hello pbisim" in r.stdout

    def test_numpy_available(self):
        r = execute_code('import numpy as np; print(np.sum([1, 2, 3]))')
        assert r.success
        assert "6" in r.stdout

    def test_preloaded_modelbuilder(self):
        """ModelBuilder should be available without explicit import."""
        r = execute_code(
            'cfg = ModelBuilder(n_bacteria=1, n_phages=0)'
            '.with_growth_rates(1.2).build()\n'
            'print(cfg.n_bacteria)'
        )
        assert r.success, r.error
        assert "1" in r.stdout

    def test_full_simulation(self):
        code = """
import numpy as np
from pbisim.builder import ModelBuilder
from pbisim.core.model import PBIModel
from pbisim.core.solver import solve_ode

cfg = ModelBuilder(n_bacteria=1, n_phages=0).with_growth_rates(1.0).build()
model = PBIModel(cfg, initial_B=np.array([1e6]), initial_P=np.zeros(0))
res = solve_ode(model, t_end=5.0, dt=1.0)
print("ok", round(res.sum_prefixes("B")[-1]))
"""
        r = execute_code(code)
        assert r.success, r.error
        assert "ok" in r.stdout

    def test_figure_captured(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
ax.set_title("captured")
"""
        r = execute_code(code)
        assert r.success, r.error
        assert len(r.figures) == 1

    def test_open_is_blocked(self):
        r = execute_code('open("test.txt", "w")')
        assert not r.success
        assert "NameError" in r.error or "open" in r.error

    def test_syntax_error_handled(self):
        r = execute_code('def foo(:\n    pass')
        assert not r.success
        assert r.error != ""

    def test_runtime_error_captured(self):
        r = execute_code('x = 1 / 0')
        assert not r.success
        assert "ZeroDivisionError" in r.error

    def test_returns_execution_result(self):
        r = execute_code('pass')
        assert isinstance(r, ExecutionResult)

    def test_stdout_captured_not_printed(self, capsys):
        execute_code('print("captured output")')
        # Nothing should leak to the real stdout
        captured = capsys.readouterr()
        assert "captured output" not in captured.out
