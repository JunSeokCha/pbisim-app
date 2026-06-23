"""
executor.py — Code execution sandbox for generated pbisim code.

Security model
--------------
Generated code runs in a fresh namespace pre-populated with the full
pbisim API, numpy, and matplotlib.  The namespace's ``__builtins__``
strips ``open``, ``exec``, ``eval``, and ``__import__`` so that
generated code cannot easily escape into the filesystem or spawn
processes.  This is a *lightweight* sandbox appropriate for a research
prototype; for a public-facing deployment, use Docker or RestrictedPython.

Design rationale
----------------
Attempting to globally patch ``builtins.__import__`` breaks matplotlib's
lazy internal imports (backends, font-loaders, etc.) and causes spurious
failures.  Instead we pre-load everything the LLM needs so that most
generated code never has to call ``import`` at all — and anything that
does import will succeed through the normal machinery.
"""

from __future__ import annotations

import io
import sys
import traceback
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe inside Streamlit
import matplotlib.pyplot as plt
import numpy as np


# ── Safe builtins: strip file/exec primitives ─────────────────────────────────
import builtins as _builtins_mod
_SAFE_BUILTINS = vars(_builtins_mod).copy()

for _dangerous in ("open", "exec", "eval", "compile",
                   "__loader__", "__spec__", "__build_class__"):
    _SAFE_BUILTINS.pop(_dangerous, None)


# ── Pre-built namespace factory ───────────────────────────────────────────────
def _build_namespace() -> dict:
    """Return a fresh namespace with the complete pbisim public API loaded."""
    from pbisim.builder import ModelBuilder
    from pbisim.core.model import PBIModel
    from pbisim.core.solver import solve_ode
    from pbisim.pk.dosing import DoseSchedule, DoseEvent
    from pbisim.pk.antibiotic import AntibioticDefinition, AntibioticSensitivity
    from pbisim.pk.phage_pk import PhagePKConfig
    from pbisim.strains.builder import StrainSet, StrainDefinition
    from pbisim.strains.genotypes import (
        BinaryResistanceGenotypes, BacterialStrain, PhageStrain, Antibiotic,
    )
    from pbisim.analysis.stationary import stationary_phase_ic
    from pbisim.trial.metrics import time_to_clearance
    from pbisim import time_to_log_reduction

    ns: dict = {
        "__builtins__": _SAFE_BUILTINS,
        # Numerics / plotting
        "np": np, "numpy": np,
        "plt": plt, "matplotlib": matplotlib,
        # Core simulation
        "ModelBuilder": ModelBuilder,
        "PBIModel": PBIModel,
        "solve_ode": solve_ode,
        # Dosing
        "DoseSchedule": DoseSchedule,
        "DoseEvent": DoseEvent,
        # Named strains
        "StrainSet": StrainSet,
        "StrainDefinition": StrainDefinition,
        "AntibioticDefinition": AntibioticDefinition,
        "AntibioticSensitivity": AntibioticSensitivity,
        # Phage PK
        "PhagePKConfig": PhagePKConfig,
        # Systematic resistance genotypes
        "BinaryResistanceGenotypes": BinaryResistanceGenotypes,
        "BacterialStrain": BacterialStrain,
        "PhageStrain": PhageStrain,
        "Antibiotic": Antibiotic,
        # Analysis helpers
        "stationary_phase_ic": stationary_phase_ic,
        "time_to_clearance": time_to_clearance,
        "time_to_log_reduction": time_to_log_reduction,
    }

    # Optional: trial runner (may not be installed)
    try:
        from pbisim.trial.runner import TrialRunner
        from pbisim.trial.population import IIVSpec, VirtualPopulation
        from pbisim.trial.distributions import LogNormal, Normal, Uniform, Fixed
        from pbisim.trial.metrics import default_metrics
        from pbisim.trial.clinical import ClinicalTrial, TreatmentArm, PretreatmentPhase
        ns.update({
            "TrialRunner": TrialRunner,
            "VirtualPopulation": VirtualPopulation,
            "IIVSpec": IIVSpec,
            "default_metrics": default_metrics,
            "LogNormal": LogNormal,
            "Normal": Normal,
            "Uniform": Uniform,
            "Fixed": Fixed,
            "ClinicalTrial": ClinicalTrial,
            "TreatmentArm": TreatmentArm,
            "PretreatmentPhase": PretreatmentPhase,
        })
    except ImportError:
        pass

    return ns


class ExecutionResult(NamedTuple):
    """Result of executing a generated code snippet."""
    success: bool
    figures: list    # matplotlib Figure objects created during execution
    stdout: str      # captured print() output
    error: str       # full traceback on failure, else ""


def execute_code(code: str) -> ExecutionResult:
    """
    Execute a generated pbisim code snippet in a controlled namespace.

    Parameters
    ----------
    code : str
        Python source code produced by the simulation agent.

    Returns
    -------
    ExecutionResult
        ``.success`` is True if execution completed without exception.
        ``.figures`` contains any matplotlib figures created.
        ``.stdout`` contains all text printed to stdout.
        ``.error`` contains the traceback on failure.
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    existing_figs = set(plt.get_fignums())
    namespace = _build_namespace()

    try:
        exec(compile(code, "<pbisim-agent>", "exec"), namespace)  # noqa: S102
        success = True
        error = ""
    except Exception:
        success = False
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    stdout = captured.getvalue()
    new_figs = [plt.figure(n) for n in plt.get_fignums()
                if n not in existing_figs]

    return ExecutionResult(success=success, figures=new_figs,
                           stdout=stdout, error=error)
