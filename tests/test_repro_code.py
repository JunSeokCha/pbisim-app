"""The auto-generated reproduction script must faithfully reproduce the simulation.

Regression guard for a real drift: the signal *functions* (growth / dormancy /
resuscitation / death) selected in the UI were applied by the build path but were
silently omitted from the "View Python Reproduction Code" output, so the generated
script ran a different model (the engine defaults) than the app.

The app stashes the generated script into ``st.session_state["_last_repro_code"]``.
These tests drive the app (AppTest), read that script, *execute* it, and assert the
resulting ``cfg`` carries the chosen signal functions — and that they agree with the
app's own build path (``simulation_config``).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pytest
from streamlit.testing.v1 import AppTest


def _sel(at, label):
    return [s for s in at.selectbox if label in (s.label or "")][0]


def _run(growth_label, death_label, dorm_signal=None):
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=200)
    at.run()
    _sel(at, "Growth signal function").set_value(growth_label)
    _sel(at, "Death signal function").set_value(death_label)
    if dorm_signal is not None:
        # keyed Direct-mode dormancy widgets: seeding session_state before the next
        # render makes the keyed widgets adopt the value.
        at.session_state["str_dorm_en_0"] = True
        at.session_state["str_dsig_0"] = dorm_signal
        at.session_state["str_rsig_0"] = dorm_signal
    at.run()
    [b for b in at.button if "Run Simulation" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception
    built = at.session_state["simulation_config"]
    code = at.session_state["_last_repro_code"]
    ns = {}
    exec(compile(code, "<repro>", "exec"), ns)
    return built, ns["cfg"], code


def test_repro_reproduces_growth_and_death_signals():
    """Non-default growth (density/logistic) + death (nutrient) signals must round-trip
    through the generated script and match the app's build path."""
    built, cfg, code = _run("density (logistic)", "nutrient (starvation)")
    assert cfg.growth_function.__name__ == "logistic_growth" == built.growth_function.__name__
    assert cfg.death_function.__name__ == "nutrient_dependent_death" == built.death_function.__name__


def test_repro_reproduces_dormancy_signals():
    """Enabling dormancy with a nutrient signal must emit the dormancy/resuscitation
    functions (not leave them at the engine default) and match the build path."""
    built, cfg, code = _run("nutrient (Monod)", "constant", dorm_signal="nutrient")
    assert "dormancy_signal=" in code  # the string the app used to omit entirely
    assert cfg.dormancy_function.__name__ == "nutrient_dependent_dormancy"
    assert cfg.resuscitation_function.__name__ == "nutrient_dependent_resuscitation"
    assert cfg.dormancy_function.__name__ == built.dormancy_function.__name__
    assert cfg.resuscitation_function.__name__ == built.resuscitation_function.__name__


@pytest.mark.parametrize("mode", [
    "Binary Genotypes (BRG)",
    "Custom Strains & Graph (StrainSet)",
])
def test_repro_brg_and_strainset_signal_functions(mode):
    """BRG / StrainSet pass the signal functions to ``to_config`` as function objects —
    the script must import and forward them, execute cleanly, and match the build path."""
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=200)
    at.run()
    _sel(at, "Bacterial Population Builder Mode").set_value(mode)
    at.run()
    _sel(at, "Growth signal function").set_value("nutrient (Monod)")
    _sel(at, "Death signal function").set_value("nutrient (starvation)")
    at.run()
    [b for b in at.button if "Run Simulation" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception

    built = at.session_state["simulation_config"]
    ns = {}
    exec(compile(at.session_state["_last_repro_code"], "<repro>", "exec"), ns)
    cfg = ns["cfg"]
    for attr in ("growth_function", "dormancy_function", "resuscitation_function", "death_function"):
        assert getattr(cfg, attr).__name__ == getattr(built, attr).__name__, attr
    assert cfg.death_function.__name__ == "nutrient_dependent_death"


# ── full-config parity: the script must reproduce EVERY field, not just signals ──

import dataclasses as _dc
import numpy as _np


def _assert_val_equal(a, b, path):
    if a is None or b is None:
        assert a is None and b is None, f"{path}: {a!r} != {b!r}"
        return
    if callable(a) or callable(b):
        assert getattr(a, "__name__", a) == getattr(b, "__name__", b), f"{path}: {a} != {b}"
        return
    if isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray):
        aa, bb = _np.asarray(a, dtype=float), _np.asarray(b, dtype=float)
        assert aa.shape == bb.shape, f"{path}: shape {aa.shape} != {bb.shape}"
        assert _np.allclose(aa, bb, equal_nan=True), f"{path}: arrays differ"
        return
    if _dc.is_dataclass(a) and not isinstance(a, type):
        _assert_cfg_equal(a, b, path)
        return
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_val_equal(x, y, f"{path}[{i}]")
        return
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: dict keys {set(a)} != {set(b)}"
        for k in a:
            _assert_val_equal(a[k], b[k], f"{path}[{k!r}]")
        return
    if isinstance(a, float):
        assert a == b or _np.isclose(a, b, equal_nan=True), f"{path}: {a} != {b}"
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


def _assert_cfg_equal(a, b, path="cfg"):
    fa = [f.name for f in _dc.fields(a)]
    fb = [f.name for f in _dc.fields(b)]
    assert fa == fb, f"{path}: field sets differ"
    for name in fa:
        # initial_conditions is a run-time carrier, not part of the built config here
        if name == "initial_conditions":
            continue
        _assert_val_equal(getattr(a, name), getattr(b, name), f"{path}.{name}")


_PHAGE_DOSE = [{"time": 1.0, "amount": 1e8, "target_type": "phage",
                "target_idx": 0, "route": "bolus", "duration": 0.0}]


@pytest.mark.parametrize("mode", [
    "Direct (ModelBuilder)",
    "Binary Genotypes (BRG)",
    "Custom Strains & Graph (StrainSet)",
])
def test_repro_full_config_parity_with_dosing(mode):
    """Every field of the script's config must equal the app's built config — including
    a t=1 phage bolus (the dosing that BRG/StrainSet repro used to drop entirely)."""
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=220)
    at.run()
    if mode != "Direct (ModelBuilder)":
        _sel(at, "Bacterial Population Builder Mode").set_value(mode)
        at.run()
    at.session_state["int_doses"] = [dict(d) for d in _PHAGE_DOSE]
    _sel(at, "Growth signal function").set_value("nutrient (Monod)")
    _sel(at, "Death signal function").set_value("nutrient (starvation)")
    at.run()
    [b for b in at.button if "Run Simulation" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception

    built = at.session_state["simulation_config"]
    ns = {}
    exec(compile(at.session_state["_last_repro_code"], "<repro>", "exec"), ns)
    _assert_cfg_equal(built, ns["cfg"])
    # the dose actually made it into the reproduced config
    assert ns["cfg"].dose_schedule is not None
    assert len(ns["cfg"].dose_schedule.events) == 1
