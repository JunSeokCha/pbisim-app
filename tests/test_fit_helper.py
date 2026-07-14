"""Phase-A calibration helpers: observable registry, ingestion, overlay/fit math."""

from __future__ import annotations

import numpy as np
import pandas as pd

import dataclasses

from pbisim_app.fit_helper import (
    OBSERVABLES,
    predicted_observable,
    normalize_fit_dataframe,
    apply_row_filters,
    aggregate_observations,
    fit_residual,
    TUNING_KNOBS,
    apply_tuning_to_config,
    bake_tuning_into_entities,
)


@dataclasses.dataclass
class _FakeConfig:
    """Minimal stand-in for a pbisim ModelConfig (only the tunable fields)."""
    growth_rates: object = None
    adsorption_rates: object = None
    burst_sizes: object = None
    latent_periods: object = None
    phage_decay_rates: object = None


class _FakeResult:
    """Minimal stand-in exposing sum_prefixes, like a pbisim SimulationResult."""
    def __init__(self, series):
        self._series = series  # dict: prefix -> array

    def sum_prefixes(self, *prefixes):
        return sum(self._series[p] for p in prefixes)


def test_observable_registry_covers_the_four_signals():
    assert set(OBSERVABLES) == {"cfu", "pfu", "od", "lum"}
    # OD divides by a link, luminescence multiplies, CFU/PFU are identity
    assert OBSERVABLES["od"]["link"][1] == "div"
    assert OBSERVABLES["lum"]["link"][1] == "mul"
    assert OBSERVABLES["cfu"]["link"] is None
    # luminescence tracks active biomass only (dormant cells are dark)
    assert OBSERVABLES["lum"]["prefixes"] == ("B",)


def test_predicted_observable_links():
    r = _FakeResult({"B": np.array([1e8]), "D": np.array([1e7]),
                     "I": np.array([0.0]), "H": np.array([0.0]), "P": np.array([1e9])})
    # CFU = B+D+I+H
    assert predicted_observable(r, "cfu") == 1.1e8
    # PFU = P
    assert predicted_observable(r, "pfu") == 1e9
    # OD = total biomass / od_to_cfu
    assert np.isclose(predicted_observable(r, "od", 1e9), 1.1e8 / 1e9)
    # luminescence = active biomass (B only) * rlu_per_cell
    assert np.isclose(predicted_observable(r, "lum", 2.0), 1e8 * 2.0)


def test_normalize_monolix_format():
    df = pd.DataFrame({
        "ID": [1, 1, 2, 2],
        "TIME": [0, 1, 0, 1],
        "DV": [0.02, 0.5, 0.03, 0.1],
        "MOI": [0.0, 0.0, 1.0, 1.0],
        "PHAGE": ["MXP1001", "MXP1001", "MXP1001", "MXP1001"],
    })
    long, conds = normalize_fit_dataframe(df, "TIME", "DV", "od", ["PHAGE", "MOI"], moi_col="MOI")
    assert list(long.columns) == ["time", "arm", "observable", "value"] or set(long.columns) == {"time", "value", "arm", "observable"}
    assert set(long["arm"]) == {"MXP1001 | 0.0", "MXP1001 | 1.0"}
    assert (long["observable"] == "od").all()
    assert conds["MXP1001 | 1.0"]["moi"] == 1.0
    assert conds["MXP1001 | 0.0"]["moi"] == 0.0


def test_apply_row_filters_include_only():
    df = pd.DataFrame({"PHAGE": ["A", "A", "B", "C"], "v": [1, 2, 3, 4]})
    # empty allow-list on a column = no restriction
    assert len(apply_row_filters(df, {"PHAGE": []})) == 4
    # restrict to A and B
    out = apply_row_filters(df, {"PHAGE": ["A", "B"]})
    assert set(out["PHAGE"]) == {"A", "B"} and len(out) == 3
    # numeric-as-string comparison works
    df2 = pd.DataFrame({"MOI": [0.0, 1.0, 1.0], "v": [1, 2, 3]})
    assert len(apply_row_filters(df2, {"MOI": ["1.0"]})) == 2


def test_aggregate_observations_mean_median_band():
    # two replicates per (arm, time) for arm 'X', observable 'od'
    long = pd.DataFrame({
        "arm": ["X", "X", "X", "X"],
        "observable": ["od", "od", "od", "od"],
        "time": [0, 0, 1, 1],
        "value": [1.0, 3.0, 10.0, 30.0],
    })
    # raw returns every point
    assert len(aggregate_observations(long, stat="raw")) == 4
    # mean collapses replicates per (arm, time)
    m = aggregate_observations(long, stat="mean").sort_values("time")
    assert list(m["value"]) == [2.0, 20.0]
    # median with a percentile band
    md = aggregate_observations(long, stat="median", band=(25, 75)).sort_values("time")
    assert list(md["value"]) == [2.0, 20.0]
    assert md["lo"].notna().all() and (md["hi"] >= md["lo"]).all()


def test_apply_tuning_to_config_scales_only_nonunit_knobs():
    cfg = _FakeConfig(
        growth_rates=np.array([1.0, 2.0]),
        adsorption_rates=np.array([[1e-8]]),
        burst_sizes=np.array([[50.0]]),
        latent_periods=np.array([[0.5]]),
        phage_decay_rates=np.array([0.1]),
    )
    tuned = apply_tuning_to_config(cfg, {"growth": 2.0, "decay": 0.5})
    assert np.allclose(tuned.growth_rates, [2.0, 4.0])   # scaled ×2
    assert np.allclose(tuned.phage_decay_rates, [0.05])  # scaled ×0.5
    # untouched knobs are unchanged
    assert np.allclose(tuned.burst_sizes, [[50.0]])
    # original is not mutated
    assert np.allclose(cfg.growth_rates, [1.0, 2.0])
    # a no-op tuning returns the config unchanged (same object)
    assert apply_tuning_to_config(cfg, {"growth": 1.0}) is cfg


def test_bake_tuning_into_entities_scales_gui_dicts_in_place():
    strains = [{"name": "PA", "growth_rate": 1.2}]
    phages = [{"name": "phi", "adsorption_s": 1e-8, "burst_sizes": 50.0,
               "latent_periods": 0.5, "phage_decay_rates": 0.1}]
    n = bake_tuning_into_entities(strains, phages,
                                  {"growth": 2.0, "adsorption": 10.0, "burst": 1.0})
    assert n == 2  # growth_rate + adsorption_s changed; burst unchanged
    assert strains[0]["growth_rate"] == 2.4
    assert phages[0]["adsorption_s"] == 1e-7
    assert phages[0]["burst_sizes"] == 50.0  # ×1.0 no-op
    # every knob key resolves to a real scope
    assert {k["scope"] for k in TUNING_KNOBS} == {"bacteria", "phages"}


def test_fit_residual_zero_and_positive():
    t = np.array([0.0, 1.0, 2.0])
    model = np.array([1.0, 10.0, 100.0])
    # exact match on log scale -> 0
    assert fit_residual(t, model, t, model, log_scale=True) == 0.0
    # a mismatch is positive and finite
    r = fit_residual(t, model, t, np.array([1.0, 1.0, 1.0]), log_scale=True)
    assert r > 0 and np.isfinite(r)
