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


def test_build_dataset_emits_bacteria_dose_additive_b0():
    """Additive-B0 model: a KNOWN inoculum (b0_is_dose) becomes a t=0 bacteria dose;
    a pre-run arm carries it as pretreatment_inoculum instead; first-observation mode
    (b0_is_dose=False) records neither, so pbisim-fit falls back to cfu[0]."""
    agg = _agg_from_csv()
    ds = nls.build_dataset(agg, ["control"], ["cfu"],
                           {"control": {"b0": 5e6, "b0_is_dose": True, "prerun": 0.0}})
    bd = [d for d in ds.arms[0].dose_events if d.target == "bacteria"]
    assert len(bd) == 1 and bd[0].amount == 5e6 and bd[0].unit == "cfu"
    assert ds.arms[0].pretreatment_inoculum is None

    ds_pr = nls.build_dataset(agg, ["control"], ["cfu"],
                              {"control": {"b0": 5e6, "b0_is_dose": True, "prerun": 24.0}})
    assert not [d for d in ds_pr.arms[0].dose_events if d.target == "bacteria"]
    assert ds_pr.arms[0].pretreatment_inoculum == 5e6

    ds_fo = nls.build_dataset(agg, ["control"], ["cfu"],
                              {"control": {"b0": 1e6, "b0_is_dose": False, "prerun": 0.0}})
    assert not [d for d in ds_fo.arms[0].dose_events if d.target == "bacteria"]
    assert ds_fo.arms[0].pretreatment_inoculum is None


def test_build_dataset_uses_imported_dose_records():
    """Imported NONMEM/Monolix dose rows (arm_doses) are emitted verbatim and override
    the manual per-arm fields for the targets they specify (inoculum + phage)."""
    agg = _agg_from_csv()
    arm_doses = {"control": [
        {"time": 0.0, "target": "bacteria", "amount": 3e6, "unit": "cfu"},
        {"time": 0.0, "target": "phage", "amount": 1e8, "unit": "pfu"}]}
    cond = {"control": {"moi": 5.0, "b0": 9e9, "b0_is_dose": True, "prerun": 0.0}}
    ds = nls.build_dataset(agg, ["control"], ["cfu"], cond, arm_doses=arm_doses)
    doses = ds.arms[0].dose_events
    bac = [d for d in doses if d.target == "bacteria"]
    ph = [d for d in doses if d.target == "phage"]
    assert len(bac) == 1 and bac[0].amount == 3e6           # data dose, not the manual 9e9
    assert len(ph) == 1 and ph[0].amount == 1e8 and ph[0].unit == "pfu"  # not the manual moi=5


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


def test_estimate_b0_wires_free_initial_conditions():
    """The B0-source 'Estimate' modes add an estimated additive-B0 parameter via
    free_initial_conditions; a role-table fit_initial_cfu target is skipped so the two
    don't double-wire."""
    from pbisim_fit.synthetic import reference_config
    cfg = reference_config()
    ds = nls.build_dataset(_agg_from_csv(), ["control"], ["cfu"], {}, od_to_cfu=None)
    tg = [{"path": "growth_rates[0]", "free": True, "value": 1.0,
           "lo": 0.1, "hi": 3.0, "log": False, "prior_mu": None, "prior_sd": None}]
    _c0, spec0 = nls.build_param_spec_v2(cfg, tg)
    _c1, spec1 = nls.build_param_spec_v2(cfg, tg, dataset=ds, estimate_b0="shared")
    assert spec1.n_params == spec0.n_params + 1          # one shared B0 θ added
    # a fit_initial_cfu free target is dropped when estimating (no double-wire)
    tg2 = tg + [{"path": "fit_initial_cfu", "free": True, "value": 1e6,
                 "lo": 1e3, "hi": 1e11, "log": True, "prior_mu": None, "prior_sd": None}]
    _c2, spec2 = nls.build_param_spec_v2(cfg, tg2, dataset=ds, estimate_b0="shared")
    assert spec2.n_params == spec1.n_params


def test_estimate_b0_fit_runs_and_sets_initial_cfu():
    """A fit with estimate_b0='shared' completes and the fitted config carries an
    estimated fit_initial_cfu (the additive B0 offset)."""
    from pbisim_fit.synthetic import reference_config
    cfg = reference_config()
    ds = nls.build_dataset(_agg_from_csv(), ["control"], ["cfu"], {}, od_to_cfu=None)
    tg = [{"path": "growth_rates[0]", "free": True, "value": 1.0,
           "lo": 0.1, "hi": 3.0, "log": False, "prior_mu": None, "prior_sd": None}]
    fp = nls.run_nls_fit_v2(cfg, tg, [], [], ds, ["cfu"],
                            n_restarts=1, max_nfev=100, estimate_b0="shared")
    fc = fp.to_config()
    _ic = getattr(fc, "fit_initial_cfu", None)
    assert _ic is not None and float(np.atleast_1d(_ic).ravel()[0]) > 0


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


def test_fit_spec_dsl_parse_and_roundtrip():
    """The statement-based DSL parses into the table schemas, validates paths, and
    round-trips (serialize → parse gives the same thetas/maps/freed)."""
    from pbisim_fit.synthetic import reference_config
    cat = nls.available_targets(reference_config(), initial_cfu=5e6)
    spec = (
        "theta g bounds=0.1..3.0 prior=1.2,0.3\n"
        "theta cost bounds=0..0.9 init=0.05\n"
        "map growth_rates[0] = g\n"
        "map growth_rates[1] = g * (1 - cost)\n"
        "free bacteria_to_resource_ratio[0] bounds=1e6..1e10 log\n"
        "fix death_rate_B[0] = 0.0\n"
    )
    tdf, thdf, errs = nls.parse_fit_spec(spec, cat)
    assert not errs
    assert list(thdf["name"]) == ["g", "cost"]
    # mappings now live on the target rows as role='Derived' + an expression
    _derived = {r["path"]: r["expression"] for _, r in tdf.iterrows() if r["role"] == "Derived"}
    assert set(_derived) == {"growth_rates[0]", "growth_rates[1]"}
    assert "bacteria_to_resource_ratio[0]" in [r["path"] for _, r in tdf.iterrows() if r["role"] == "Free"]

    # unknown path is reported, not silently dropped
    _t, _th, e2 = nls.parse_fit_spec("free not_a_param bounds=0..1", cat)
    assert any("unknown parameter" in x for x in e2)

    # serialize the structures back and re-parse — thetas/maps preserved
    targets = [{"path": r["path"], "free": (r["role"] == "Free"),
                "value": float(r["value"]) if str(r["value"]).strip() else 0.0,
                "lo": (float(r["lower"]) if str(r["lower"]).strip() else 0.0),
                "hi": (float(r["upper"]) if str(r["upper"]).strip() else float("inf")),
                "log": bool(r["log"]), "prior_mu": None, "prior_sd": None}
               for _, r in tdf.iterrows()]
    thetas = [{"name": r["name"], "lo": float(r["lower"]), "hi": float(r["upper"]),
               "log": bool(r["log"]), "initial": (float(r["initial"]) if str(r["initial"]).strip() else None),
               "prior_mu": None, "prior_sd": None} for _, r in thdf.iterrows()]
    maps = [{"path": p, "expr": e} for p, e in _derived.items()]
    txt = nls.serialize_fit_spec(targets, thetas, maps, cat)
    _t2, _th2, e3 = nls.parse_fit_spec(txt, cat)
    assert not e3
    _derived2 = {r["path"] for _, r in _t2.iterrows() if r["role"] == "Derived"}
    assert list(_th2["name"]) == ["g", "cost"] and len(_derived2) == 2


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
