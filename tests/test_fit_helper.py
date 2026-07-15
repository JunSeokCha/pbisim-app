"""Phase-A calibration helpers: observable registry, ingestion, overlay/fit math."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pbisim_app.fit_helper import (
    OBSERVABLES,
    predicted_observable,
    normalize_fit_dataframe,
    apply_row_filters,
    aggregate_observations,
    fit_residual,
    STRAIN_TUNABLES,
    STRAIN_DORMANCY_TUNABLES,
    PHAGE_TUNABLES,
    ADSORPTION_PHAGE_KEYS,
    entity_param_key,
)


class _FakeResult:
    """Minimal stand-in exposing sum_prefixes, like a pbisim SimulationResult."""
    def __init__(self, series, od=None):
        self._series = series  # dict: prefix -> array
        self._od = od

    def sum_prefixes(self, *prefixes):
        return sum(self._series[p] for p in prefixes)

    def get_od(self):
        return self._od


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


def test_predicted_od_uses_model_debris_when_requested():
    # simple OD = biomass / od_to_cfu; debris-aware OD comes from get_od()
    r = _FakeResult({"B": np.array([1e8]), "D": np.array([0.0]),
                     "I": np.array([0.0]), "H": np.array([0.0])},
                    od=np.array([0.42]))
    assert np.isclose(predicted_observable(r, "od", 1e9), 1e8 / 1e9)          # simple
    assert np.isclose(predicted_observable(r, "od", 1e9, use_model_od=True), 0.42)  # debris-aware
    # non-OD observables ignore the flag
    r2 = _FakeResult({"P": np.array([1e9])}, od=np.array([9.9]))
    assert predicted_observable(r2, "pfu", use_model_od=True) == 1e9


def test_entity_param_key_is_builder_mode_aware():
    # adsorption is stored under different phage-dict keys per builder mode
    brg_phage = {"adsorption_s": 5e-8, "burst_sizes": 50.0}
    legacy_phage = {"adsorption_rates": 1e-8, "burst_sizes": 50.0}
    assert entity_param_key(brg_phage, ADSORPTION_PHAGE_KEYS) == "adsorption_s"
    assert entity_param_key(legacy_phage, ADSORPTION_PHAGE_KEYS) == "adsorption_rates"
    # falls back to the first candidate when the entity has neither yet
    assert entity_param_key({}, ADSORPTION_PHAGE_KEYS) == ADSORPTION_PHAGE_KEYS[0]


def test_tunable_registries_cover_the_fit_parameters():
    strain_keys = {k["key"] for k in STRAIN_TUNABLES}
    assert {"growth_rate", "bacteria_to_resource_ratio", "death_rate_B", "initial_B"} <= strain_keys
    phage_keys = {k["key"] for k in PHAGE_TUNABLES}
    assert {"burst_sizes", "latent_periods", "phage_decay_rates", "phage_decay_Km", "attenuation_rate"} <= phage_keys
    # adsorption is handled separately (builder-mode specific storage)
    assert "adsorption_s" in ADSORPTION_PHAGE_KEYS


def test_fit_residual_zero_and_positive():
    t = np.array([0.0, 1.0, 2.0])
    model = np.array([1.0, 10.0, 100.0])
    # exact match on log scale -> 0
    assert fit_residual(t, model, t, model, log_scale=True) == 0.0
    # a mismatch is positive and finite
    r = fit_residual(t, model, t, np.array([1.0, 1.0, 1.0]), log_scale=True)
    assert r > 0 and np.isfinite(r)
