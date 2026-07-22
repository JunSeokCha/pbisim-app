"""
test_sweeps.py — Unit tests for the sweep mapping, applying, and padding logic.
"""

from __future__ import annotations

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from pbisim import ModelBuilder
from pbisim_app.sweep_helper import (
    get_sweep_parameters,
    apply_sweep_parameter,
    parse_comma_separated_series,
    pad_vectors,
)

APP = "pbisim_app/app.py"

def test_parse_comma_separated_series():
    """Verify parsing handles spaces, scientific notation, and float values."""
    assert parse_comma_separated_series("") == []
    assert parse_comma_separated_series("  ") == []
    assert parse_comma_separated_series("1, 2.5, 3e6,1e-2") == [1.0, 2.5, 3000000.0, 0.01]


def test_pad_vectors():
    """Verify vectors are padded to the longest, using the last value, and warnings are raised."""
    vecs = {
        "Phage 0": [1e5, 1e6],
        "Abx 0": [1.0],
        "Phage 1": [] # Empty is not padded or swept, it's ignored
    }
    
    padded, warnings = pad_vectors(vecs)
    assert len(padded) == 2
    assert padded["Phage 0"] == [1e5, 1e6]
    assert padded["Abx 0"] == [1.0, 1.0]
    assert "Phage 1" not in padded
    assert len(warnings) == 1
    assert "Abx 0" in warnings[0]


def test_get_sweep_parameters():
    """Verify that get_sweep_parameters generates a valid mapping dictionary."""
    builder = ModelBuilder(n_bacteria=2, n_phages=1, n_latent=5, n_depth=2)
    builder = builder.with_growth_rates([1.2, 1.1])
    config = builder.build()
    
    params = get_sweep_parameters(config)
    assert "Monod Constant (Ks)" in params
    assert "Dormancy Depth Layers (Q)" in params
    assert "Phage Latent Stages (L)" in params
    assert "Growth Rate - Strain 0" in params
    assert "Phage Decay Rate - Phage 0" in params
    assert "Adsorption - Phage 0 on Strain 0" in params
    assert "Initial Density (B0) - Strain 0" in params


def test_apply_sweep_parameter():
    """Verify applying a sweep parameter mutates the configuration correctly."""
    builder = ModelBuilder(n_bacteria=2, n_phages=1, n_latent=5, n_depth=2)
    builder = builder.with_growth_rates([1.2, 1.1])
    builder = builder.with_dormancy(
        dormancy_rate=np.array([0.2, 0.2]),
        resuscitation_rate=np.array([0.1, 0.1]),
        dormancy_diffusion_rate=np.array([0.05, 0.05])
    )
    config = builder.build()
    
    initial_B = np.array([1e7, 1e7])
    initial_P = np.array([1e6])
    initial_S = 1.0
    
    init_D = np.zeros((2, 2))
    init_D[:, 0] = 1e6
    init_D[:, 1] = 1e5
    model_kwargs = {"initial_D": init_D}
    
    # 1. Scalar sweep
    params = get_sweep_parameters(config)
    meta = params["Monod Constant (Ks)"]
    c2, ib2, ip2, is2, mk2 = apply_sweep_parameter(0.9, meta, config, initial_B, initial_P, initial_S, model_kwargs)
    assert c2.monod_constant == 0.9
    
    # 2. 1D Array sweep
    meta = params["Growth Rate - Strain 0"]
    c3, ib3, ip3, is3, mk3 = apply_sweep_parameter(1.8, meta, config, initial_B, initial_P, initial_S, model_kwargs)
    assert c3.growth_rates[0] == 1.8
    assert c3.growth_rates[1] == 1.1 # unchanged
    
    # 3. 2D Array sweep
    meta = params["Adsorption - Phage 0 on Strain 1"]
    c4, ib4, ip4, is4, mk4 = apply_sweep_parameter(5e-8, meta, config, initial_B, initial_P, initial_S, model_kwargs)
    assert c4.adsorption_rates[1, 0] == 5e-8
    
    # 4. Dimension n_depth sweep (resizes initial_D)
    meta = params["Dormancy Depth Layers (Q)"]
    c5, ib5, ip5, is5, mk5 = apply_sweep_parameter(5, meta, config, initial_B, initial_P, initial_S, model_kwargs)
    assert c5.n_depth == 5
    assert mk5["initial_D"].shape == (5, 2)
    # Total dormant cells should be conserved (2.2e6)
    assert np.allclose(np.sum(mk5["initial_D"]), 2.2e6)


def test_dose_response_zero_phage_dose_does_not_suppress():
    """A swept phage dose of 0 must mean no phage — the baseline initial_P
    inoculum must not leak in and suppress the bacteria (regression)."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["current_page_radio"] = "Dose-Response Sweeps"
    at.run()
    at.session_state["dr_sweep_phg_en_0"] = True
    at.run()
    at.session_state["dr_sweep_phg_series_0"] = "0, 1e8"
    at.session_state["dr_sweep_phg_unit_0"] = "PFU (absolute)"
    at.run()
    [b for b in at.button if "Run Dose-Response" in (b.label or "")][0].click().run()

    assert len(at.exception) == 0
    df = [d.value for d in at.dataframe][0]
    nadir = {row["Swept Doses"]: float(row["Nadir (cells/mL)"]) for _, row in df.iterrows()}
    zero_nadir = nadir["phage_0: 0.0e+00"]
    high_nadir = nadir["phage_0: 1.0e+08"]
    # dose 0 leaves bacteria near their initial density; the big dose eradicates
    assert zero_nadir > 1e6
    assert high_nadir < zero_nadir

    # the phage's baseline inoculum is restored after the sweep
    assert at.session_state["int_phages"][0]["initial_P"] > 0

def test_dose_response_shows_od_trajectories_when_enabled():
    """OD trajectories are plotted in the dose sweep only when the OD/debris
    module is enabled; the default phage series is 1e3..1e9."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["int_debris_enabled"] = True
    at.session_state["int_od_to_cfu_conversion_factor"] = 1e9
    at.session_state["current_page_radio"] = "Dose-Response Sweeps"
    at.run()
    at.session_state["dr_sweep_phg_en_0"] = True
    at.run()
    assert at.session_state["dr_sweep_phg_series_0"] == "0, 1e3, 1e5, 1e7, 1e9"
    at.session_state["dr_sweep_phg_unit_0"] = "PFU (absolute)"
    at.run()
    [b for b in at.button if "Run Dose-Response" in (b.label or "")][0].click().run()

    assert len(at.exception) == 0
    # The trajectory metric is chosen from a "Trace" selectbox; OD is offered only
    # because the debris module is enabled.
    trace = [s for s in at.selectbox if s.label == "Trace"]
    assert trace, "Trace selectbox missing"
    opts = list(trace[0].options)
    assert "Total viable bacteria (CFU/mL)" in opts
    assert "Total free phage (PFU/mL)" in opts
    assert "Optical density (AU)" in opts


def test_sweep_results_survive_navigation():
    """Dose-response and parameter sweep results persist across page navigation
    (rendered from session_state, until re-run)."""
    at = AppTest.from_file(APP, default_timeout=220)
    at.run()
    # dose-response
    at.session_state["current_page_radio"] = "Dose-Response Sweeps"
    at.run()
    at.session_state["dr_sweep_phg_en_0"] = True
    at.run()
    at.session_state["dr_sweep_phg_series_0"] = "1e3, 1e7"
    at.session_state["dr_sweep_phg_unit_0"] = "PFU (absolute)"
    at.run()
    [b for b in at.button if "Run Dose-Response" in (b.label or "")][0].click().run()
    assert at.session_state["dr_sweep_result"]
    at.session_state["current_page_radio"] = "Interactive Simulator"
    at.run()
    at.session_state["current_page_radio"] = "Dose-Response Sweeps"
    at.run()
    assert any("Summary of Runs" in m.value for m in at.markdown)
    assert len(at.exception) == 0

    # parameter sweep (1D)
    at.session_state["current_page_radio"] = "Parameter Sweeps"
    at.run()
    [b for b in at.button if "Run 1D Sweep" in (b.label or "")][0].click().run()
    assert at.session_state["param_sweep_result"]
    at.session_state["current_page_radio"] = "Interactive Simulator"
    at.run()
    at.session_state["current_page_radio"] = "Parameter Sweeps"
    at.run()
    assert any("Summary of Runs" in m.value for m in at.markdown)
    assert len(at.exception) == 0


def test_prerun_collapse_warning():
    """A pre-run with a death rate and no dormancy decimates the culture; the
    sweep warns that the CFU/OD curves start very low (item: OD scales low)."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["int_strains"][0]["death_rate_B"] = 0.5
    at.session_state["int_t_prerun"] = 24.0
    at.session_state["current_page_radio"] = "Dose-Response Sweeps"
    at.run()
    at.session_state["dr_sweep_phg_en_0"] = True
    at.run()
    at.session_state["dr_sweep_phg_series_0"] = "0"
    at.session_state["dr_sweep_phg_unit_0"] = "PFU (absolute)"
    at.run()
    [b for b in at.button if "Run Dose-Response" in (b.label or "")][0].click().run()
    assert any("pre-run leaves only" in w.value for w in at.warning)
    assert len(at.exception) == 0


def test_sweep_configs_survive_navigation():
    """The sweep CONTROLS (not just results) survive navigation, and the
    dose-response phage series defaults to include a zero-dose control."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    # dose-response: default series includes the 0 control
    at.session_state["current_page_radio"] = "Dose-Response Sweeps"
    at.run()
    at.session_state["dr_sweep_phg_en_0"] = True
    at.run()
    assert [t for t in at.text_input if t.key == "dr_sweep_phg_series_0"][0].value == "0, 1e3, 1e5, 1e7, 1e9"
    at.session_state["dr_sweep_phg_series_0"] = "0, 5e6"
    at.session_state["dr_sweep_phg_unit_0"] = "MOI (relative to B(0))"
    at.run()
    at.session_state["current_page_radio"] = "Interactive Simulator"
    at.run()
    at.session_state["current_page_radio"] = "Dose-Response Sweeps"
    at.run()
    assert at.session_state["dr_sweep_phg_en_0"] is True
    assert at.session_state["dr_sweep_phg_series_0"] == "0, 5e6"
    assert at.session_state["dr_sweep_phg_unit_0"] == "MOI (relative to B(0))"

    # parameter sweep: 1D/2D choice survives navigation
    at.session_state["current_page_radio"] = "Parameter Sweeps"
    at.run()
    at.session_state["ps_sweep_type"] = "2D Sweep"
    at.run()
    at.session_state["current_page_radio"] = "Interactive Simulator"
    at.run()
    at.session_state["current_page_radio"] = "Parameter Sweeps"
    at.run()
    assert at.session_state["ps_sweep_type"] == "2D Sweep"
    assert len(at.exception) == 0


def test_broadcast_sweep_parameters_apply_to_all():
    """`(ALL strains)` sweep parameters set the same value across every strain."""
    import numpy as np
    b = ModelBuilder(n_bacteria=2, n_phages=1, n_latent=5, n_depth=2).with_growth_rates([1.2, 1.1])
    cfg = b.build()
    params = get_sweep_parameters(cfg)
    assert "Growth Rate (ALL strains)" in params
    c2, ib2, *_ = apply_sweep_parameter(0.7, params["Growth Rate (ALL strains)"],
                                        cfg, np.array([1e7, 1e7]), np.array([1e6]), 1.0, {})
    assert list(c2.growth_rates) == [0.7, 0.7]
    # broadcast initial B0
    _, ib3, *_ = apply_sweep_parameter(5e6, params["Initial Density B0 (ALL strains)"],
                                       cfg, np.array([1e7, 0.0]), np.array([1e6]), 1.0, {})
    assert list(ib3) == [5e6, 5e6]


def test_coupled_sweep_runs_and_persists():
    """A coupled (linked) sweep applies several parameters together per step and
    survives navigation."""
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["current_page_radio"] = "Parameter Sweeps"
    at.run()
    at.session_state["ps_sweep_type"] = "Coupled (linked)"
    at.run()
    opts = [m for m in at.multiselect if m.key == "pc_labels"][0].options
    g = [o for o in opts if o.startswith("Growth Rate - ")][0]
    d = [o for o in opts if o.startswith("Phage Decay Rate - ")][0]
    at.session_state["pc_labels"] = [g, d]
    at.run()
    at.session_state["pc_series_0"] = "1.0, 1.2, 1.5"
    at.session_state["pc_series_1"] = "0.1, 0.2, 0.3"
    at.run()
    [b for b in at.button if "Run Coupled" in (b.label or "")][0].click().run()
    res = at.session_state["param_sweep_result"]
    assert res["type"] == "coupled" and len(res["summary"]) == 3
    assert len(at.exception) == 0
    # mismatched lengths are rejected
    at.session_state["pc_series_1"] = "0.1, 0.2"
    at.run()
    [b for b in at.button if "Run Coupled" in (b.label or "")][0].click().run()
    assert any("same number of points" in e.value for e in at.error)

    # persists across navigation
    at.session_state["current_page_radio"] = "Interactive Simulator"
    at.run()
    at.session_state["current_page_radio"] = "Parameter Sweeps"
    at.run()
    assert at.session_state["ps_sweep_type"] == "Coupled (linked)"


def test_parameter_sweep_shows_od_trajectories_when_enabled():
    """The parameter sweep (1D + coupled) plots OD trajectories when the OD/debris
    module is enabled, and omits them otherwise."""
    at = AppTest.from_file(APP, default_timeout=220)
    at.run()
    at.session_state["int_debris_enabled"] = True
    at.session_state["int_od_to_cfu_conversion_factor"] = 1e9
    at.session_state["current_page_radio"] = "Parameter Sweeps"
    at.run()
    [b for b in at.button if "Run 1D Sweep" in (b.label or "")][0].click().run()
    _tr = [s for s in at.selectbox if s.key == "ps1d_traj_metric"]
    assert _tr and "Optical density (AU)" in list(_tr[0].options)
    assert len(at.exception) == 0

    # coupled sweep also shows OD
    at.session_state["ps_sweep_type"] = "Coupled (linked)"
    at.run()
    g = [o for o in [m for m in at.multiselect if m.key == "pc_labels"][0].options
         if o.startswith("Growth Rate - ")][0]
    at.session_state["pc_labels"] = [g]
    at.run()
    at.session_state["pc_series_0"] = "1.0, 1.2, 1.5"
    at.run()
    [b for b in at.button if "Run Coupled" in (b.label or "")][0].click().run()
    _tr = [s for s in at.selectbox if s.key == "pscoupled_traj_metric"]
    assert _tr and "Optical density (AU)" in list(_tr[0].options)

    # no OD chart when the module is off
    at2 = AppTest.from_file(APP, default_timeout=220)
    at2.run()
    at2.session_state["current_page_radio"] = "Parameter Sweeps"
    at2.run()
    [b for b in at2.button if "Run 1D Sweep" in (b.label or "")][0].click().run()
    _tr = [s for s in at2.selectbox if s.key == "ps1d_traj_metric"]
    assert _tr and "Optical density (AU)" not in list(_tr[0].options)
    assert len(at2.exception) == 0


def test_dimension_sweep_clamps_to_int_ge_1():
    """Sweeping a dimension parameter (n_depth / n_latent) never produces 0 or a
    fractional compartment count — it rounds and clamps to an integer >= 1 (a
    fractional/sub-1 value previously crashed the model with an IndexError)."""
    b = ModelBuilder(n_bacteria=1, n_phages=1, n_latent=5, n_depth=2).with_growth_rates([1.2])
    b = b.with_dormancy(dormancy_rate=0.2, resuscitation_rate=0.1, dormancy_diffusion_rate=0.05)
    cfg = b.build()
    params = get_sweep_parameters(cfg)
    mk = {"initial_D": np.zeros((2, 1))}
    for label, field in [("Dormancy Depth Layers (Q)", "n_depth"),
                         ("Phage Latent Stages (L)", "n_latent")]:
        for val, expect in [(0.1, 1), (0.9, 1), (2.4, 2), (2.6, 3)]:
            c, *_ = apply_sweep_parameter(val, params[label], cfg,
                                          np.array([1e7]), np.array([1e6]), 1.0, mk)
            assert getattr(c, field) == expect, f"{field} {val} -> {getattr(c, field)} != {expect}"


def test_sweep_parameters_cover_dormancy_immune_debris():
    """The previously-missing parameters are now sweepable: dormancy thresholds,
    dormant-death / diffusion / immune broadcasts, per-strain immune rates, and
    dormant latent / attenuation per pair."""
    b = ModelBuilder(n_bacteria=2, n_phages=1, n_latent=5, n_depth=2).with_growth_rates([1.2, 1.1])
    b = b.with_dormancy(dormancy_rate=0.2, resuscitation_rate=0.1, dormancy_diffusion_rate=0.05)
    b = b.with_nutrient(monod_constant=0.3, carrying_capacity=1e9)
    cfg = b.build()
    p = get_sweep_parameters(cfg)
    for label in [
        "Dormancy Nutrient Half-saturation (Ks_dorm)",
        "Dormancy Density Threshold (K_dorm)",
        "Immune Stim 50 (K_stim)",
        "Dormant Death Rate dD (ALL strains)",
        "Dormancy Diffusion Rate (ALL strains)",
        "Immune Kill Rate Dormant (ALL strains)",
        "Immune Stim Rate (ALL strains)",
        "Immune Kill Rate (ALL strains)",
        "Immune Stim Rate - Strain 0",
        "Immune Kill Rate (Active) - Strain 0",
        "Dormant Latent Period - Phage 0 on Strain 0",
        "Dormant Adsorption Attenuation - Phage 0 on Strain 0",
    ]:
        assert label in p, f"missing sweep parameter: {label}"

    # a None-valued dormancy threshold can still be swept (safety-net: set the field)
    c, *_ = apply_sweep_parameter(0.05, p["Dormancy Nutrient Half-saturation (Ks_dorm)"],
                                  cfg, np.array([1e7, 1e7]), np.array([1e6]), 1.0, {})
    assert c.dormancy_monod_constant == 0.05


def test_get_od_raises_without_debris_module():
    """Documents the failure the app's _safe_od guards against: result.get_od() (and the
    Debris state) raise when the debris ODE wasn't enabled, so OD-trajectory code must
    catch it rather than assume get_od works."""
    import numpy as np
    import pytest
    from pbisim.builder import ModelBuilder
    from pbisim.core.model import PBIModel
    from pbisim.core.solver import solve_ode

    cfg = ModelBuilder(n_bacteria=1, n_phages=0).with_growth_rates(1.2).build()  # no debris
    m = PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([]), initial_S=1.0)
    r = solve_ode(m, t_end=5.0, dt=1.0)
    with pytest.raises(Exception):
        r.get_od()


def test_prerun_duration_is_sweepable():
    """The pre-run duration can be swept as a 1D parameter (0 h = no pre-run);
    each point runs stationary_phase_ic with that duration."""
    at = AppTest.from_file(APP, default_timeout=220)
    at.run()
    at.session_state["current_page_radio"] = "Parameter Sweeps"
    at.run()
    at.session_state["p1_sweep_label"] = "Pre-run duration (hours)"
    at.run()
    at.session_state["ps_1d_min"] = 0.0
    at.session_state["ps_1d_max"] = 24.0
    at.session_state["ps_1d_steps"] = 3
    at.session_state["ps_1d_spacing"] = "Linear"
    at.run()
    [b for b in at.button if "Run 1D Sweep" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception
    res = at.session_state["param_sweep_result"]
    assert res and len(res["trajectories"]) == 3


def test_sweep_exposes_mutation_and_pseudolysogeny():
    """Mutation rate (mass-conserving) and pseudolysogeny (hibernation / lytic
    resumption) are now sweepable; phage_decay_Km appears when set."""
    import numpy as np
    from pbisim import ModelBuilder
    from pbisim_app.sweep_helper import get_sweep_parameters, apply_sweep_parameter
    b = (ModelBuilder(n_bacteria=2, n_phages=1).with_growth_rates([1.2, 1.1])
         .with_pseudolysogeny(hibernation_rate=np.array([[0.1], [0.1]]),
                              lytic_resumption_rate=np.array([[0.05], [0.05]])))
    cfg = b.build()
    p = get_sweep_parameters(cfg, strains=[{"name": "WT"}, {"name": "Mut"}], phages=[{"name": "P0"}])
    assert any(k.startswith("Mutation Rate - ") for k in p)
    assert any("Hibernation Rate" in k for k in p)
    assert any("Lytic Resumption Rate" in k for k in p)
    mk = [k for k in p if k.startswith("Mutation Rate - WT")][0]
    c, *_ = apply_sweep_parameter(1e-6, p[mk], cfg, np.array([1e7, 10.0]), np.array([1e6]), 1.0, {})
    assert c.mutation_rates[1, 0] == 1e-6           # WT(0) → Mut(1) rate set
    assert np.allclose(c.mutation_rates.sum(axis=0), 0.0), "mutation not mass-conserving"
