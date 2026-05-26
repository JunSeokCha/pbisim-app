"""
executor.py — Safe code execution sandbox for generated pbisim code.

The generated Python code from the agent is executed in an isolated
namespace that has access to pbisim, numpy, and matplotlib but not to
system utilities, file I/O, or network calls.
"""

from __future__ import annotations

import io
import traceback
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class ExecutionResult(NamedTuple):
    """Output from a code execution run."""
    success: bool
    figures: list           # list of matplotlib.figure.Figure objects
    stdout: str             # captured print() output
    error: str              # traceback string if failed, else ""


# ── Allowed namespace (everything generated code needs) ───────────────────────
_SAFE_NAMESPACE_BASE = {
    "np": np,
    "numpy": np,
    "plt": plt,
    "matplotlib": matplotlib,
    "__builtins__": {
        # safe builtins only
        "print": print,
        "range": range,
        "len": len,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "isinstance": isinstance,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "reversed": reversed,
        "__import__": _guarded_import,
    },
}


def _guarded_import(name, *args, **kwargs):
    """Allow only whitelisted module imports inside generated code."""
    _ALLOWED_MODULES = {
        "numpy", "np", "scipy", "matplotlib", "matplotlib.pyplot",
        "pbisim", "pbisim.builder", "pbisim.core", "pbisim.core.model",
        "pbisim.core.solver", "pbisim.core.state", "pbisim.pk",
        "pbisim.pk.dosing", "pbisim.pk.antibiotic", "pbisim.pk.phage_pk",
        "pbisim.pd", "pbisim.pd.models", "pbisim.strains",
        "pbisim.strains.builder", "pbisim.strains.genotypes",
        "pbisim.trial", "pbisim.trial.runner", "pbisim.trial.population",
        "pbisim.trial.metrics", "pbisim.analysis",
    }
    top = name.split(".")[0]
    if name in _ALLOWED_MODULES or top in {"numpy", "scipy", "matplotlib", "pbisim"}:
        import importlib
        return importlib.import_module(name)
    raise ImportError(
        f"Import of '{name}' is not allowed in generated simulation code."
    )


def execute_code(code: str) -> ExecutionResult:
    """
    Execute generated pbisim code in a restricted namespace.

    Figures created inside the code are captured automatically.
    print() output is captured to a string.

    Parameters
    ----------
    code : str
        Python source code to execute.

    Returns
    -------
    ExecutionResult
        Contains success flag, figures, stdout, and error message.
    """
    import sys

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_out = io.StringIO()

    # Track figures created during execution
    before_figs = set(map(id, map(plt.figure, plt.get_fignums())))

    namespace = dict(_SAFE_NAMESPACE_BASE)

    try:
        exec(compile(code, "<pbisim-agent>", "exec"), namespace)
        success = True
        error = ""
    except Exception:
        success = False
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    stdout = captured_out.getvalue()

    # Collect new figures
    after_fig_nums = plt.get_fignums()
    figures = []
    for num in after_fig_nums:
        fig = plt.figure(num)
        if id(fig) not in before_figs:
            figures.append(fig)

    return ExecutionResult(
        success=success,
        figures=figures,
        stdout=stdout,
        error=error,
    )
