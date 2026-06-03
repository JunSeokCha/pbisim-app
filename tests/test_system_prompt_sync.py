"""
test_system_prompt_sync.py — guard against drift between the system prompt and
the real pbisim API.

`prompts/system_prompt.md` hand-documents pbisim signatures so the agent
generates valid code. If pbisim renames or removes any of that surface, the
prompt silently goes stale and the agent emits broken code. These tests fail
loudly when that happens, so the prompt (and executor namespace) can be updated
in lockstep.

If a test here fails after a pbisim upgrade, update BOTH this file and
`prompts/system_prompt.md` (and `pbisim_app/executor.py` if the namespace
changed).
"""

import inspect

import pytest


# ── 1. Every import path the prompt + executor rely on must resolve ───────────

def test_prompt_import_paths_resolve():
    """The deep import paths quoted in system_prompt.md still exist."""
    from pbisim.builder import ModelBuilder            # §1, §7
    from pbisim.core.model import PBIModel             # §3, §7
    from pbisim.core.solver import solve_ode           # §4, §7
    from pbisim.pk.dosing import DoseSchedule, DoseEvent          # §6
    from pbisim.strains.builder import StrainSet, StrainDefinition  # §8
    from pbisim.pk.antibiotic import (                 # §8
        AntibioticDefinition,
        AntibioticSensitivity,
    )
    from pbisim.strains.genotypes import BinaryResistanceGenotypes  # §8


# ── 2. ModelBuilder constructor + fluent methods named in §1–§2 ───────────────

def _params(func):
    return set(inspect.signature(func).parameters)


def test_modelbuilder_constructor_signature():
    from pbisim.builder import ModelBuilder
    params = _params(ModelBuilder.__init__)
    for name in ("n_bacteria", "n_phages", "n_latent", "n_depth"):
        assert name in params, f"ModelBuilder.__init__ lost parameter '{name}'"


@pytest.mark.parametrize(
    "method",
    [
        "with_growth_rates",
        "with_phage_params",
        "with_mutations",
        "with_nutrient",
        "with_antibiotic",
        "build",
    ],
)
def test_modelbuilder_methods_exist(method):
    from pbisim.builder import ModelBuilder
    assert callable(getattr(ModelBuilder, method, None)), (
        f"system_prompt.md uses ModelBuilder.{method}() but it no longer exists"
    )


def test_with_antibiotic_signature():
    from pbisim.builder import ModelBuilder
    params = _params(ModelBuilder.with_antibiotic)
    for name in ("name", "k_elim", "Vc", "emax", "ec50", "hill"):
        assert name in params, f"with_antibiotic lost parameter '{name}'"


# ── 3. PBIModel / solve_ode signatures named in §3–§4 ─────────────────────────

def test_pbimodel_signature():
    from pbisim.core.model import PBIModel
    params = _params(PBIModel.__init__)
    for name in ("config", "initial_B", "initial_P", "initial_S"):
        assert name in params, f"PBIModel.__init__ lost parameter '{name}'"


def test_solve_ode_signature():
    from pbisim.core.solver import solve_ode
    params = _params(solve_ode)
    for name in ("t_end", "dt"):
        assert name in params, f"solve_ode lost parameter '{name}'"


# ── 4. SimulationResult accessors named in §5 ─────────────────────────────────

def test_simulation_result_accessors():
    """`result.get(...)` and `result.sum_prefixes(...)` are the §5 contract."""
    from pbisim.core.solver import SimulationResult
    for method in ("get", "sum_prefixes"):
        assert callable(getattr(SimulationResult, method, None)), (
            f"system_prompt.md uses result.{method}() but it no longer exists"
        )
