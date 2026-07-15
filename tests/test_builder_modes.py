"""Tests for pbisim-app builder modes (BRG, StrainSet) and cohort trials."""

from __future__ import annotations

import numpy as np
from pbisim import PBIModel, solve_ode
from pbisim.strains import StrainDefinition, StrainSet
from pbisim.strains.genotypes import BinaryResistanceGenotypes, BacterialStrain, PhageStrain, Antibiotic


def test_brg_build_and_run():
    """Verify BinaryResistanceGenotypes constructs config and runs successfully."""
    # 1. Setup BRG
    b = BacterialStrain(base_growth_rate=1.2)
    p = [PhageStrain(name="Phage0", burst_size_s=50.0)]
    abx = [Antibiotic(name="Drug0", emax_s=3.0, ec50_s=0.2, emax_r=0.1, ec50_r=2.0)]
    
    brg = BinaryResistanceGenotypes.from_strains(p, bacteria=b, antibiotics=abx)
    config = brg.to_config(n_latent=5, n_depth=1)
    
    # 2 phages/abx loci -> 4 genotypes: '00', '10', '01', '11'
    initial_B = np.array([1e7, 10.0, 10.0, 0.0])
    initial_P = np.array([1e6])
    initial_S = 1.0
    
    model = PBIModel(config, initial_B=initial_B, initial_P=initial_P, initial_S=initial_S)
    result = solve_ode(model, t_end=24.0, dt=1.0)
    assert result is not None
    assert len(result.time) > 0


def test_strainset_build_and_run():
    """Verify StrainSet with graph mutation dictionary constructs config and runs successfully."""
    # 1. Setup StrainSet
    ss = StrainSet(n_phages=1)
    from pbisim.pk.antibiotic import AntibioticDefinition, AntibioticSensitivity
    ss.add_antibiotic(AntibioticDefinition("cipro", k_elim=0.5))
    
    ss.add_strain(StrainDefinition(
        name="WT",
        growth_rate=1.2,
        adsorption_rates=np.array([1e-7]),
        adsorption_rates_dormant=np.array([0.0]),
        burst_sizes=np.array([50.0]),
        latent_periods=np.array([0.5]),
        latent_periods_dormant=np.array([0.5]),
        bacteria_to_resource_ratio=1e9,
        dormancy_rate=0.0,
        resuscitation_rate=0.0,
        dormancy_diffusion_rate=0.0,
        imm_stim_rate=0.0,
        imm_kill_rate=0.0,
        attenuation_rate=np.zeros(1),
        antibiotic_sensitivity={"cipro": AntibioticSensitivity(emax=5.0, ec50=0.5)}
    ))
    
    ss.add_strain(StrainDefinition(
        name="R_cipro",
        growth_rate=1.1,
        adsorption_rates=np.array([1e-7]),
        adsorption_rates_dormant=np.array([0.0]),
        burst_sizes=np.array([50.0]),
        latent_periods=np.array([0.5]),
        latent_periods_dormant=np.array([0.5]),
        bacteria_to_resource_ratio=1e9,
        dormancy_rate=0.0,
        resuscitation_rate=0.0,
        dormancy_diffusion_rate=0.0,
        imm_stim_rate=0.0,
        imm_kill_rate=0.0,
        attenuation_rate=np.zeros(1),
        antibiotic_sensitivity={"cipro": AntibioticSensitivity(emax=0.5, ec50=5.0)}
    ))
    
    ss.set_mutation_graph({"WT": {"R_cipro": 1e-7}})
    config = ss.to_config(
        n_latent=3,
        n_depth=1,
        phage_decay_rates=np.array([0.03]),
        imm_decay_rate=0.1,
        imm_stim50=1e6,
        imm_kill50=1e6,
        monod_constant=1.0,
        recycle_fraction=0.5
    )
    
    initial_B = np.array([1e7, 10.0])
    initial_P = np.array([1e6])
    initial_S = 1.0
    
    model = PBIModel(config, initial_B=initial_B, initial_P=initial_P, initial_S=initial_S)
    result = solve_ode(model, t_end=24.0, dt=1.0)
    assert result is not None
    assert len(result.time) > 0


def test_clinical_trial_with_pretreatment():
    """Verify that a ClinicalTrial running with a PretreatmentPhase succeeds with InitialConditions attached."""
    from pbisim.trial.clinical import TreatmentArm
    from pbisim.trial.population import InitialConditions
    from pbisim.builder import ModelBuilder
    from pbisim_app.trial_helper import run_trial_simulation
    
    cfg = ModelBuilder(n_bacteria=1, n_phages=0).with_growth_rates(1.0).build()
    
    init_B = np.array([1e6])
    init_P = np.zeros(0)
    init_S = 1.0
    
    cfg.initial_conditions = InitialConditions(
        B=init_B,
        P=init_P,
        S=init_S
    )
    
    iiv_inputs = [
        {"path": "growth_rates", "dist_type": "LogNormal", "params": {"cv": 0.2}, "mode": "multiplicative"}
    ]
    
    arms = [TreatmentArm(name="Control", dose_schedule=None)]
    
    res = run_trial_simulation(
        cfg,
        iiv_inputs,
        arms,
        n_patients=2,
        t_end=5.0,
        dt=1.0,
        seed=42,
        pretreatment_hours=2.0,
        n_jobs=1,
        base_initial_B=init_B,
        base_initial_P=init_P,
        base_initial_S=init_S
    )
    
    assert res is not None
    assert "Control" in res.arm_names


def test_trial_pretreatment_carries_dormant_reservoir():
    """A trial PretreatmentPhase must carry its dormant reservoir into treatment.

    Same class of bug as the interactive-simulator pre-run: PretreatmentPhase replaces
    the patient config's initial_conditions with the full stationary-phase state, but
    the model factory previously took initial_D/initial_Imm from the GUI base kwargs,
    discarding the (dominant) dormant population. A long pretreatment on a dormancy
    model must still start the treatment near the stationary carrying capacity.
    """
    from pbisim.trial.clinical import TreatmentArm
    from pbisim.trial.population import InitialConditions
    from pbisim.builder import ModelBuilder
    from pbisim_app.trial_helper import run_trial_simulation

    b = ModelBuilder(n_bacteria=1, n_phages=0, n_latent=5, n_depth=3)
    b = b.with_growth_rates([1.2], bacteria_to_resource_ratio=[1e9])
    b = b.with_dormancy(dormancy_rate=np.array([0.2]), resuscitation_rate=np.array([0.1]),
                        dormancy_diffusion_rate=np.array([0.05]))
    b = b.with_nutrient(track_nutrients=True, monod_constant=0.3)
    cfg = b.build()

    init_B = np.array([1e7])
    init_P = np.zeros(0)
    init_S = 1.0
    cfg.initial_conditions = InitialConditions(B=init_B, P=init_P, S=init_S)

    res = run_trial_simulation(
        cfg, [], [TreatmentArm(name="Control", dose_schedule=None)],
        n_patients=2, t_end=12.0, dt=0.5, seed=1,
        pretreatment_hours=48.0,  # long prerun -> population mostly dormant
        n_jobs=1, base_initial_B=init_B, base_initial_P=init_P, base_initial_S=init_S,
    )

    for r in res["Control"].results:
        assert r is not None
        total = r.sum_prefixes("B", "D", "I", "H")
        assert total[0] > 1e8, "treatment must start from the full stationary population"


def test_trial_control_arm_has_no_phage():
    """Regression: the phage inoculum must not leak into non-phage arms.

    Crossover trials share initial conditions across arms and differ only by
    dose schedule, so seeding the phage via ``initial_P`` would contaminate the
    Control (and Antibiotic-Only) arms and eradicate the bacteria there, making
    the control indistinguishable from phage therapy.  The app instead starts
    every arm at zero free phage and delivers the inoculum as a t=0 bolus only
    in phage-containing arms.  This test locks in that invariant: Control keeps
    phage at zero and bacteria grow, while the phage arm eradicates them.
    """
    from pbisim.trial.clinical import TreatmentArm
    from pbisim.trial.population import InitialConditions
    from pbisim.builder import ModelBuilder
    from pbisim.pk.dosing import DoseEvent, DoseSchedule
    from pbisim_app.trial_helper import run_trial_simulation

    builder = ModelBuilder(n_bacteria=1, n_phages=1, n_latent=5, n_depth=1)
    builder = builder.with_growth_rates([0.6], bacteria_to_resource_ratio=[1e9])
    builder = builder.with_phage_params(
        adsorption_rates=np.array([[1e-8]]),
        adsorption_rates_dormant=np.array([[0.0]]),
        burst_sizes=np.array([[50.0]]),
        latent_periods=np.array([[0.5]]),
        phage_decay_rates=np.array([0.1]),
    )
    builder = builder.with_nutrient(track_nutrients=True, monod_constant=0.3)
    base_cfg = builder.build()

    init_B = np.array([1e7])
    init_P = np.array([1e6])
    init_S = 1.0

    # Replicate the fixed app arm-assembly: zero the shared phage baseline and
    # deliver the inoculum only as a t=0 phage bolus in the phage arm.
    base_P = np.zeros_like(init_P)
    phage_inoculum_doses = [
        DoseEvent(time=0.0, amount=float(init_P[i]), target="phage",
                  index=i, route="bolus", duration=0.0)
        for i in range(len(init_P)) if float(init_P[i]) > 0.0
    ]
    base_cfg.initial_conditions = InitialConditions(B=init_B, P=base_P, S=init_S)

    arms = [
        TreatmentArm(name="Control", dose_schedule=DoseSchedule([])),
        TreatmentArm(name="Phage-Only", dose_schedule=DoseSchedule(phage_inoculum_doses)),
    ]

    res = run_trial_simulation(
        base_cfg,
        [],  # no IIV — deterministic
        arms,
        n_patients=3,
        t_end=48.0,
        dt=0.5,
        seed=1,
        pretreatment_hours=0.0,
        n_jobs=1,
        base_initial_B=init_B,
        base_initial_P=base_P,
        base_initial_S=init_S,
    )

    # Control: no phage present at any time, bacteria grow (no eradication).
    control = res["Control"]
    for r in control.results:
        assert r is not None
        assert np.max(r.sum_prefixes("P")) < 1.0, "phage leaked into the Control arm"
        total_b = r.sum_prefixes("B", "D", "I", "H")
        assert total_b[-1] > total_b[0], "Control bacteria should grow untreated"

    # Phage-Only: phage delivered, bacteria eradicated.
    phage_arm = res["Phage-Only"]
    for r in phage_arm.results:
        assert r is not None
        assert np.max(r.sum_prefixes("P")) > init_P[0], "phage inoculum not delivered"
        total_b = r.sum_prefixes("B", "D", "I", "H")
        assert total_b[-1] < 1.0, "phage arm should eradicate bacteria"


def test_dormancy_creates_immune_refuge():
    """Dormant cells are immune-privileged unless imm_kill_rate_D > 0.

    This locks in the mechanism behind the app's dormancy+immunity warning: with
    dormancy enabled, a resistant strain survives in the dormant reservoir that
    immunity neither kills (imm_kill_rate_D=0) nor is stimulated by, so the
    infection never clears — whereas imm_kill_rate_D > 0 lets immunity clear it.
    """
    from pbisim.builder import ModelBuilder

    def build(kill_rate_D):
        b = ModelBuilder(n_bacteria=2, n_phages=1, n_latent=5, n_depth=3)
        b = b.with_growth_rates([1.2, 1.2], bacteria_to_resource_ratio=[1e9, 1e9])
        b = b.with_dormancy(
            dormancy_rate=np.array([0.2, 0.2]),
            resuscitation_rate=np.array([0.1, 0.1]),
            dormancy_diffusion_rate=np.array([0.05, 0.05]),
        )
        b = b.with_phage_params(
            adsorption_rates=np.array([[1e-8], [0.0]]),
            adsorption_rates_dormant=np.array([[0.0], [0.0]]),
            burst_sizes=np.array([[50.0], [50.0]]),
            latent_periods=np.array([[0.5], [0.5]]),
            phage_decay_rates=np.array([0.1]),
        )
        b = b.with_mutations(phage_resistance_rates=[1e-7])
        b = b.with_nutrient(track_nutrients=True, monod_constant=0.3)
        b = b.with_immunity(
            imm_stim_rate=np.full(2, 1.0), imm_stim50=1e6,
            imm_kill_rate=np.full(2, 1e7), imm_kill50=1e8,
            imm_decay_rate=0.05, immune_module="innate",
            imm_kill_rate_D=(np.array([kill_rate_D, kill_rate_D])
                             if kill_rate_D > 0 else None),
        )
        m = PBIModel(b.build(), initial_B=np.array([1e7, 0.0]),
                     initial_P=np.array([1e6]), initial_S=1.0, initial_Imm=0.0)
        return solve_ode(m, t_end=48.0, dt=0.5, method="BDF", extinction_threshold=1.0)

    refuge = build(0.0).sum_prefixes("B", "D", "I", "H")[-1]
    cleared = build(1e7).sum_prefixes("B", "D", "I", "H")[-1]

    # With no dormant killing, a large reservoir persists (>1e7); enabling it
    # drops the burden by orders of magnitude.
    assert refuge > 1e7, "dormant reservoir should persist when imm_kill_rate_D=0"
    assert cleared < refuge / 100.0, "imm_kill_rate_D>0 should clear the reservoir"


def test_hill_immune_module_active_at_app_defaults():
    """The hill module must actually kill bacteria at the app's default parameters.

    Hill killing is imm_max·B/(imm_kill50+B); with the app defaults (imm_max=1e7,
    imm_kill50=1e5) it should control an otherwise-unchecked bloom. Also verifies the
    engine ignores the innate-only fields in hill mode (imm_stim_rate/imm_kill_rate/
    imm_decay_rate), which is why the UI hides them — changing them must not move the
    hill result.
    """
    from pbisim.builder import ModelBuilder

    def build(immune, **imm):
        b = ModelBuilder(n_bacteria=1, n_phages=0, n_latent=5, n_depth=1)
        b = b.with_growth_rates([1.2], bacteria_to_resource_ratio=[1e9])
        b = b.with_nutrient(track_nutrients=True, monod_constant=0.3)
        if immune:
            b = b.with_immunity(immune_module="hill", **imm)
        m = PBIModel(b.build(), initial_B=np.array([1e7]),
                     initial_P=np.zeros(0), initial_S=1.0)
        return solve_ode(m, t_end=48.0, dt=0.5, method="BDF",
                         extinction_threshold=1.0).sum_prefixes("B", "D", "I", "H")

    defaults = dict(imm_max=1e7, imm_kill50=1e5, imm_stim_rate=np.full(1, 0.1),
                    imm_kill_rate=np.full(1, 1e7), imm_decay_rate=0.1, imm_stim50=1e6)

    off = build(False)[-1]
    on = build(True, **defaults)[-1]
    assert off > 1e8, "control should bloom to carrying capacity"
    assert on < off / 1000.0, "hill immunity must control the bloom at app defaults"

    # Innate-only fields are inert in hill mode: perturbing them by orders of
    # magnitude must not change the outcome.
    inert = {**defaults, "imm_stim_rate": np.full(1, 1e3),
             "imm_kill_rate": np.full(1, 1e13), "imm_decay_rate": 5.0}
    assert np.isclose(build(True, **inert)[-1], on, atol=1.0), \
        "hill output must not depend on imm_stim_rate/imm_kill_rate/imm_decay_rate"


def test_prerun_carries_dormant_reservoir():
    """A long stationary-phase prerun must not collapse the treatment population.

    Guards the run_sim_from_gui_params prerun fix: stationary_phase_ic returns the
    active B *and* the dormant reservoir D (which dominates at stationary phase).
    Keeping only B (the old behaviour) discards most of the culture, so a longer
    prerun starts the treatment with fewer and fewer cells until it reads as ~0.
    Carrying ic.D forward must preserve the full population regardless of prerun
    length.
    """
    from pbisim.builder import ModelBuilder
    from pbisim.analysis import stationary_phase_ic

    b = ModelBuilder(n_bacteria=1, n_phages=0, n_latent=5, n_depth=3)
    b = b.with_growth_rates([1.2], bacteria_to_resource_ratio=[1e9])
    b = b.with_dormancy(dormancy_rate=np.array([0.2]), resuscitation_rate=np.array([0.1]),
                        dormancy_diffusion_rate=np.array([0.05]))
    b = b.with_nutrient(track_nutrients=True, monod_constant=0.3)
    cfg = b.build()

    def treat(t_prerun, keep_dormant):
        ic = stationary_phase_ic(cfg, t_prerun=t_prerun, B0=np.array([1e7]))
        kw = {}
        if keep_dormant and ic.D is not None:
            kw["initial_D"] = ic.D
        m = PBIModel(cfg, initial_B=ic.B, initial_P=np.zeros(0),
                     initial_S=max(float(ic.S), 0.0), **kw)
        return solve_ode(m, t_end=24.0, dt=0.5, method="BDF",
                         extinction_threshold=1.0).sum_prefixes("B", "D", "I", "H")

    dropped = treat(48.0, keep_dormant=False)   # old behaviour
    kept = treat(48.0, keep_dormant=True)        # fixed behaviour
    assert dropped.max() < 1e7, "dropping ic.D should lose most of the stationary culture"
    assert kept.max() > 1e8, "carrying ic.D must preserve the stationary population"
    assert kept.max() > 100 * dropped.max(), "fix must retain orders of magnitude more biomass"


def test_bacteria_to_resource_ratio_editable_in_all_modes():
    """bacteria_to_resource_ratio has an editable widget in Direct + StrainSet
    (it was previously read-only there); pseudolysogeny is now editable in
    BRG + StrainSet."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=200)
    at.run()
    keys = lambda a: {n.key for n in a.number_input if n.key}
    assert "str_ratio_0" in keys(at)  # Direct

    at.session_state["widget_builder_mode"] = "Custom Strains & Graph (StrainSet)"
    at.run(); at.run()
    k = keys(at)
    assert "ss_str_ratio_0" in k
    assert all(f"ss_phg_{s}_0" in k for s in ("hib_s", "hib_r", "res_s", "res_r"))

    at.session_state["widget_builder_mode"] = "Binary Genotypes (BRG)"
    at.run(); at.run()
    k = keys(at)
    assert all(f"brg_phg_{s}_0" in k for s in ("hib_s", "hib_r", "res_s", "res_r"))
    assert len(at.exception) == 0
