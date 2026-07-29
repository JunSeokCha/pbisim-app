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
                     "I": np.array([5e7]), "H": np.array([2e7]), "P": np.array([1e9])})
    # CFU = culturable only (B+D); infected I / hibernating H are NOT counted.
    assert predicted_observable(r, "cfu") == 1.1e8
    # PFU = P
    assert predicted_observable(r, "pfu") == 1e9
    # OD = all physically-present biomass (B+D+I+H) / od_to_cfu
    assert np.isclose(predicted_observable(r, "od", 1e9), 1.8e8 / 1e9)
    # luminescence = active biomass (B only) * rlu_per_cell
    assert np.isclose(predicted_observable(r, "lum", 2.0), 1e8 * 2.0)


def test_observation_model_override():
    from pbisim_app.fit_helper import obs_prefixes, OBS_COMPARTMENTS
    r = _FakeResult({"B": np.array([1e8]), "D": np.array([1e7]),
                     "I": np.array([5e7]), "H": np.array([2e7])})
    # default CFU = B+D
    assert OBSERVABLES["cfu"]["prefixes"] == ("B", "D")
    assert obs_prefixes("cfu", None) == ("B", "D")
    # a user observation model can add I/H (total live load) or restrict to B (active only)
    assert obs_prefixes("cfu", {"cfu": ("B", "D", "I", "H")}) == ("B", "D", "I", "H")
    assert predicted_observable(r, "cfu", prefixes=("B", "D", "I", "H")) == 1.8e8
    assert predicted_observable(r, "cfu", prefixes=("B",)) == 1e8
    assert OBS_COMPARTMENTS == ("B", "D", "I", "H")


def test_app_cfu_matches_pbisim_fit_default():
    """The app's default CFU compartments must equal pbisim-fit's fit residual default,
    so the Calibration overlay and the NLS fit agree out of the box."""
    from pbisim_fit.refinement.nls import NLSConfig
    assert tuple(OBSERVABLES["cfu"]["prefixes"]) == tuple(NLSConfig().cfu_compartments)


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


def test_parse_dose_rows_nonmem_format():
    """NONMEM/Monolix dose rows (EVID=1) are parsed into per-arm dose records; the
    observable column names the target compartment, AMT the amount, default units apply."""
    from pbisim_app.fit_helper import parse_dose_rows
    df = pd.DataFrame([
        {"ARM": "A", "TIME": 0.0, "OBS": "cfu",      "DV": 1e6, "AMT": "",  "EVID": 0},
        {"ARM": "A", "TIME": 0.0, "OBS": "bacteria", "DV": "",  "AMT": 5e6, "EVID": 1},
        {"ARM": "A", "TIME": 0.0, "OBS": "phage",    "DV": "",  "AMT": 1e8, "EVID": 1},
        {"ARM": "B", "TIME": 0.0, "OBS": "cfu",      "DV": 2e6, "AMT": "",  "EVID": 0},
    ])
    doses = parse_dose_rows(df, ["ARM"], "EVID", "OBS", "AMT", "TIME")
    assert set(doses) == {"A"}                                   # only arm A has dose rows
    by_t = {d["target"]: d for d in doses["A"]}
    assert by_t["bacteria"]["amount"] == 5e6 and by_t["bacteria"]["unit"] == "cfu"
    assert by_t["phage"]["amount"] == 1e8 and by_t["phage"]["unit"] == "pfu"
    # no EVID mapping → observation-only dataset
    assert parse_dose_rows(df, ["ARM"], None, "OBS", "AMT", "TIME") == {}


def test_fit_residual_zero_and_positive():
    t = np.array([0.0, 1.0, 2.0])
    model = np.array([1.0, 10.0, 100.0])
    # exact match on log scale -> 0
    assert fit_residual(t, model, t, model, log_scale=True) == 0.0
    # a mismatch is positive and finite
    r = fit_residual(t, model, t, np.array([1.0, 1.0, 1.0]), log_scale=True)
    assert r > 0 and np.isfinite(r)


def test_residual_vector_log10_and_floor():
    """log10 residuals: perfect fit → zeros; both sides clipped to the floor."""
    import numpy as np
    from pbisim_app.fit_helper import residual_vector_log10
    r = residual_vector_log10([0, 1, 2], [1e7, 1e6, 1e5], [0, 1, 2], [1e7, 1e6, 1e5], 1.0)
    assert np.allclose(r, 0.0)
    # both pred and obs below the floor (10^1=10) → clipped equal → zero residual
    r2 = residual_vector_log10([0, 1], [1.0, 2.0], [0, 1], [3.0, 4.0], 1.0)
    assert np.allclose(r2, 0.0)


def test_build_fit_spec_maps_to_pbisim_fit():
    """build_fit_spec produces a pbisim-fit-shaped dataset (arms w/ observable arrays,
    MOI dose, pretreatment_h/inoculum) + NLSConfig floors + warm-start; JSON-safe."""
    import json
    import numpy as np
    import pandas as pd
    from pbisim_app.fit_helper import build_fit_spec
    agg = pd.DataFrame([
        {"arm": "A", "observable": "cfu", "time": 0.0, "value": 1e7, "lo": np.nan, "hi": np.nan},
        {"arm": "A", "observable": "od", "time": 0.0, "value": 0.05, "lo": np.nan, "hi": np.nan},
        {"arm": "B", "observable": "cfu", "time": 2.0, "value": 1e5, "lo": np.nan, "hi": np.nan},
    ])
    spec = build_fit_spec(agg, ["A", "B"], ["cfu", "od"],
                          {"A": {"b0": 1e7, "prerun": 12.0, "moi": 1.0}, "B": {"prerun": 0.0, "moi": 0.0}},
                          od_to_cfu=8e8, model_params={"growth_rates": [1.2]})
    ds = spec["dataset"]
    assert set(ds["arms"]) == {"A", "B"}
    assert "cfu" in ds["arms"]["A"] and "od" in ds["arms"]["A"]
    assert ds["arms"]["A"]["pretreatment_h"] == 12.0          # stationary phase
    assert ds["arms"]["A"]["doses"][0]["amount"] == "MOI:1"   # MOI dose
    assert spec["nls_cfg"]["obs_keys"] == ["cfu", "od"]
    assert spec["nls_cfg"]["od_floor_log10"] == -2.5
    assert spec["dataset"]["metadata"]["od_to_cfu"] == 8e8
    assert "warm_start" in spec
    json.dumps(spec)   # must be serializable (no NaN/ndarray)
