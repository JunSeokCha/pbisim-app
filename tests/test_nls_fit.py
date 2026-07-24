"""pbisim-fit NLS integration (Calibration §5c).

The nls_fit helpers must (a) offer only free params that fit the model's
dimensions, (b) build a pbisim-fit dataset from the app's aggregated calibration
data, and (c) recover ground-truth parameters from the committed tutorial CSV via
`refine_nls`. The fit is torch-free (lazy `import pbisim_fit`), so these run without
the SBI extras. Skipped if pbisim-fit is not installed in the env.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pbisim_fit")

from pbisim import ModelBuilder
from pbisim.growth.signals import monod_growth

from pbisim_app import nls_fit as nls
from pbisim_app.fit_helper import normalize_fit_dataframe, aggregate_observations

CSV = "pbisim_app/examples/tutorial_synthetic_brg.csv"


def _agg_from_csv():
    df = pd.read_csv(CSV)
    long, _conds = normalize_fit_dataframe(df, "time", "value", "observable", ["arm"], None)
    return aggregate_observations(long, "raw", None)


def _monod_base(ratio=1e9):
    return (ModelBuilder(n_bacteria=1, n_phages=1)
            .with_growth_rates([1.2], bacteria_to_resource_ratio=[ratio])
            .with_growth_function(monod_growth)
            .with_nutrient(monod_constant=0.3)).build()


def test_available_free_params_respects_dimensions():
    cfg = _monod_base()
    labels = [l for l, *_ in nls.available_free_params(cfg)]
    assert "Growth rate — strain 0" in labels
    assert "Burst size — strain 0 × phage 0" in labels     # 1 strain × 1 phage
    assert "Growth rate — strain 1" not in labels          # only 1 strain
    assert "Monod Ks (global)" in labels


def test_build_dataset_dose_unit_pfu_vs_moi():
    """A dose value can be an absolute PFU/mL titre (unit='pfu') or an MOI multiplier
    (unit='moi'); build_dataset emits the right DoseRecord unit."""
    agg = _agg_from_csv()
    cond = {"1e+09 PFU": {"moi": 1e9}}
    ds_pfu = nls.build_dataset(agg, ["1e+09 PFU"], ["cfu"], cond, dose_unit="pfu")
    ds_moi = nls.build_dataset(agg, ["1e+09 PFU"], ["cfu"], cond, dose_unit="moi")
    d_pfu = ds_pfu.arms[0].dose_events[0]
    d_moi = ds_moi.arms[0].dose_events[0]
    assert d_pfu.unit == "pfu" and d_pfu.amount == 1e9   # absolute titre
    assert d_moi.unit == "moi" and d_moi.amount == 1e9   # multiplier of B0


def test_build_dataset_shapes_and_moi_dose():
    agg = _agg_from_csv()
    ds = nls.build_dataset(agg, ["control"], ["cfu", "od"],
                           {"control": {"moi": 2.0}}, od_to_cfu=2e8)
    assert len(ds.arms) == 1
    arm = ds.arms[0]
    assert arm.cfu is not None and arm.od is not None
    assert len(arm.cfu) == len(ds.time)
    # MOI condition became a phage dose
    assert any(d.target == "phage" and d.unit == "moi" and d.amount == 2.0
               for d in arm.dose_events)


def test_estimate_od_to_cfu_recovers_data_ratio():
    agg = _agg_from_csv()
    r = nls.estimate_od_to_cfu(agg, ["control"])
    assert 1.5e8 < r < 2.6e8   # tutorial truth od_to_cfu ~ 2e8


def test_build_param_spec_sharing_reduces_param_count():
    """Reparameterization: sharing ties several paths to ONE estimated value, so the
    spec has fewer parameters than paths."""
    from pbisim_fit.synthetic import reference_config
    cfg = reference_config()
    shared = [{"paths": ["growth_rates[0]", "growth_rates[1]"], "lo": 0.1, "hi": 3.0, "log": False}]
    _c, spec = nls.build_param_spec(cfg, [], shared_groups=shared)
    assert spec.n_params == 1                       # 2 paths → 1 shared theta


def test_nls_fit_sharing_recovers_shared_growth():
    """A shared growth theta across the two BRG strains recovers ~truth from control-arm
    CFU, with a single estimated parameter."""
    from pbisim_fit.synthetic import reference_config
    agg = _agg_from_csv()
    cfg = reference_config()
    shared = [{"paths": ["growth_rates[0]", "growth_rates[1]"], "lo": 0.1, "hi": 3.0, "log": False},
              {"paths": ["bacteria_to_resource_ratio[0]", "bacteria_to_resource_ratio[1]"],
               "lo": 1e6, "hi": 1e10, "log": True}]
    ds = nls.build_dataset(agg, ["control"], ["cfu"], {}, od_to_cfu=None)
    fp = nls.run_nls_fit(cfg, [], ds, ["cfu"], shared_groups=shared, n_restarts=3, max_nfev=300)
    assert fp.n_params == 2                         # 4 paths → 2 shared thetas
    m = fp.map()
    assert 0.9 < m["shr0"] < 1.5                    # shared growth ~ 1.2
    assert 5e7 < m["shr1"] < 2e8                    # shared ratio ~ 1e8


def test_build_param_spec_fitness_cost_derives_target():
    """A fitness-cost link binds target = source × (1 − cost); the fitted config's
    target path equals source × (1 − cost_MAP)."""
    from pbisim_fit.synthetic import reference_config
    cfg = reference_config()
    indiv = [("WT growth", "growth_rates[0]", 0.1, 3.0, False)]
    links = [{"source": "growth_rates[0]", "target": "growth_rates[1]",
              "kind": "fitness_cost", "lo": 0.0, "hi": 0.9}]
    _c, spec = nls.build_param_spec(cfg, indiv, fitness_links=links)
    assert spec.n_params == 2                        # WT growth + one cost
    ds = nls.build_dataset(_agg_from_csv(), ["control"], ["cfu"], {}, od_to_cfu=None)
    fp = nls.run_nls_fit(cfg, indiv, ds, ["cfu"], fitness_links=links, n_restarts=2, max_nfev=150)
    m = fp.map()
    fc = fp.to_config()
    g0, g1 = float(np.atleast_1d(fc.growth_rates)[0]), float(np.atleast_1d(fc.growth_rates)[1])
    assert abs(g1 - g0 * (1.0 - m["link0"])) < 1e-6  # derivation actually bound


def test_build_param_spec_scale_link():
    """A scale link binds target = source × factor."""
    from pbisim_fit.synthetic import reference_config
    cfg = reference_config()
    indiv = [("WT ads", "adsorption_rates[0,0]", 1e-10, 1e-7, True)]
    links = [{"source": "adsorption_rates[0,0]", "target": "adsorption_rates[1,0]",
              "kind": "scale", "lo": 0.0, "hi": 1.0}]
    _c, spec = nls.build_param_spec(cfg, indiv, fitness_links=links)
    assert spec.n_params == 2                        # WT adsorption + one scale factor


def test_available_targets_is_comprehensive():
    """The target catalog includes mutation, debris, and the fit-side virtuals
    (fitness cost, initial CFU/PFU, resistant fraction) the user flagged."""
    from pbisim_fit.synthetic import reference_config
    cfg = reference_config()   # 2-strain, debris on, mutation matrix present
    paths = {p for (_l, p, *_r) in nls.available_targets(cfg)}
    assert "growth_rates[0]" in paths and "growth_rates[1]" in paths
    assert "mutation_rates[1,0]" in paths          # mutation now estimable
    assert {"debris_u", "debris_v", "debris_kdis"} <= paths   # debris now estimable
    assert "monod_constant" in paths
    # fit-side virtuals (items 1 & 7)
    assert "fitness_cost" in paths and "init_resistant_fraction" in paths
    assert "fit_initial_cfu" in paths and "fit_initial_pfu" in paths


def test_estimable_fitness_cost_and_initial_cfu():
    """fitness_cost and fit_initial_cfu are freeable via the v2 spec and recover
    sensible values (fit_initial_cfu ≈ the control-arm B₀)."""
    from pbisim_fit.synthetic import reference_config
    cfg = reference_config()
    tgts = [
        {"path": "growth_rates[0]", "free": True, "value": 1.0, "lo": 0.1, "hi": 3.0, "log": False},
        {"path": "fitness_cost", "free": True, "value": 0.0, "lo": 0.0, "hi": 0.9, "log": False},
        {"path": "fit_initial_cfu", "free": True, "value": 5e6, "lo": 1e3, "hi": 1e11, "log": True},
    ]
    ds = nls.build_dataset(_agg_from_csv(), ["control"], ["cfu"], {}, od_to_cfu=None)
    fp = nls.run_nls_fit_v2(cfg, tgts, [], [], ds, ["cfu"], n_restarts=2, max_nfev=200)
    m = fp.map()
    assert 0.9 < m["free0"] < 1.5                    # growth
    assert 1e6 < m["free2"] < 5e7                     # fit_initial_cfu ≈ control B₀ (5e6)


def test_virtual_params_not_set_when_fixed():
    """A fixed fitness_cost must NOT be pinned (setting fitness_cost=0 would wipe a
    BRG's baked-in resistant growth) — it acts only when freed."""
    from pbisim_fit.synthetic import reference_config
    import numpy as np
    cfg = reference_config()
    g_before = np.array(cfg.growth_rates, dtype=float).copy()
    tgts = [{"path": "growth_rates[0]", "free": True, "value": 1.0, "lo": 0.1, "hi": 3.0, "log": False},
            {"path": "fitness_cost", "free": False, "value": 0.0, "lo": 0.0, "hi": 1.0, "log": False}]
    out_cfg, _spec = nls.build_param_spec_v2(cfg, tgts, [], [])
    # fitness_cost must be unset on the built config (so _apply_fitness_cost is a no-op)
    assert getattr(out_cfg, "fitness_cost", None) is None


def test_validate_expr_guards_bad_input():
    ok, _ = nls.validate_expr("theta1*(1-theta2)", ["theta1", "theta2"])
    assert ok
    bad, _ = nls.validate_expr("theta1 + nope", ["theta1"])         # unknown name
    assert not bad
    bad2, _ = nls.validate_expr("__import__('os')", ["theta1"])     # no calls/attrs
    assert not bad2


def test_build_param_spec_v2_free_and_mapping():
    """v2 spec: a freed target plus a theta/mapping produce the right parameter count
    and bind the mapped path."""
    from pbisim_fit.synthetic import reference_config
    cfg = reference_config()
    targets = [{"path": p, "free": (p == "bacteria_to_resource_ratio[0]"), "value": v,
                "lo": lo, "hi": hi, "log": log}
               for (lab, p, v, lo, hi, log) in nls.available_targets(cfg)]
    thetas = [{"name": "theta1", "lo": 0.1, "hi": 3.0, "log": False, "initial": 1.0},
              {"name": "theta2", "lo": 0.0, "hi": 0.9, "log": False, "initial": 0.1}]
    mappings = [{"path": "growth_rates[0]", "expr": "theta1"},
                {"path": "growth_rates[1]", "expr": "theta1*(1-theta2)"}]
    _c, spec = nls.build_param_spec_v2(cfg, targets, thetas, mappings)
    assert spec.n_params == 3         # 1 free target + theta1 + theta2
    ds = nls.build_dataset(_agg_from_csv(), ["control"], ["cfu"], {}, od_to_cfu=None)
    fp = nls.run_nls_fit_v2(cfg, targets, thetas, mappings, ds, ["cfu"], n_restarts=2, max_nfev=200)
    m = fp.map()
    g = np.atleast_1d(np.asarray(fp.to_config().growth_rates))
    assert abs(g[1] - g[0] * (1.0 - m["theta2"])) < 1e-6   # mapping bound
    assert 0.9 < m["theta1"] < 1.5


def test_unbounded_and_one_sided_bounds():
    """Blank bounds → ±inf; run_nls_fit_v2 caps to a single start and still recovers."""
    import numpy as np
    from pbisim import ModelBuilder
    from pbisim.growth.signals import monod_growth
    cfg = (ModelBuilder(n_bacteria=1, n_phages=1)
           .with_growth_rates([1.2], bacteria_to_resource_ratio=[1e8])
           .with_growth_function(monod_growth).with_nutrient(monod_constant=0.3)).build()
    # growth fully unbounded (both inf); ratio one-sided (lower finite, upper inf)
    targets = [
        {"path": "growth_rates[0]", "free": True, "value": 1.0,
         "lo": float("-inf"), "hi": float("inf"), "log": False},
        {"path": "bacteria_to_resource_ratio[0]", "free": True, "value": 1e8,
         "lo": 1e6, "hi": float("inf"), "log": True},
    ]
    assert nls.has_unbounded(targets, [])
    ds = nls.build_dataset(_agg_from_csv(), ["control"], ["cfu"], {}, od_to_cfu=None)
    fp = nls.run_nls_fit_v2(cfg, targets, [], [], ds, ["cfu"], n_restarts=5, max_nfev=200)
    m = fp.map()
    assert 0.9 < m["free0"] < 1.5
    assert 5e7 < m["free1"] < 2e8


def test_prior_regularizes_map_estimate():
    """A Gaussian prior (prior_mu/prior_sd) turns the NLS into a MAP estimate: a tight
    prior pulls the estimate toward its mean, away from the data-only value (~1.2)."""
    from pbisim_fit.synthetic import reference_config
    cfg = reference_config()
    ds = nls.build_dataset(_agg_from_csv(), ["control"], ["cfu"], {}, od_to_cfu=None)

    def _fit(prior):
        t = [{"path": "growth_rates[0]", "free": True, "value": 1.0,
              "lo": 0.1, "hi": 3.0, "log": False}]
        if prior:
            t[0]["prior_mu"], t[0]["prior_sd"] = prior
        return nls.run_nls_fit_v2(cfg, t, [], [], ds, ["cfu"], n_restarts=2, max_nfev=200).map()["free0"]

    assert 1.0 < _fit(None) < 1.4              # data-only ≈ 1.2
    assert abs(_fit((0.6, 0.03)) - 0.6) < 0.05  # tight prior at 0.6 dominates
    assert abs(_fit((2.0, 0.03)) - 2.0) < 0.05  # tight prior at 2.0 dominates


def test_nls_fit_recovers_growth_from_tutorial_csv():
    """The headline check: CSV → refine_nls recovers the control-arm growth rate
    and yield (Monod base config, CFU + OD, few restarts)."""
    agg = _agg_from_csv()
    cfg = _monod_base()
    free = nls.available_free_params(cfg)
    by = {l: (l, p, lo, hi, g) for l, p, lo, hi, g in free}
    picks = [by["Growth rate — strain 0"], by["Bacteria/resource ratio — strain 0"]]
    od_link = nls.estimate_od_to_cfu(agg, ["control"])
    ds = nls.build_dataset(agg, ["control"], ["cfu", "od"], {}, od_to_cfu=od_link)
    fp = nls.run_nls_fit(cfg, picks, ds, ["cfu", "od"],
                         od_to_cfu=od_link, n_restarts=2, max_nfev=200)
    m = fp.map()
    assert 0.9 < m["growth_rates[0]"] < 1.5              # truth 1.2
    assert 5e7 < m["bacteria_to_resource_ratio[0]"] < 2e8   # truth 1e8
