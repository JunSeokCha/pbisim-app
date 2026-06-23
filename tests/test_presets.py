"""Tests for preset configurations to ensure they are valid and buildable in pbisim."""

from __future__ import annotations

import numpy as np
import pytest
from pbisim import ModelBuilder, PBIModel, solve_ode, DoseSchedule, DoseEvent
from pbisim_app.presets import TUTORIALS


def test_presets_structure():
    """Verify that all presets have the required keys and types."""
    assert len(TUTORIALS) == 13
    for p in TUTORIALS:
        assert "id" in p
        assert "name" in p
        assert "description" in p
        assert "type" in p
        assert p["type"] in ("single", "script")

        if p["type"] == "single":
            assert "parameters" in p
            params = p["parameters"]
            assert "strains" in params
            assert "phages" in params
            assert "antibiotics" in params
            assert "doses" in params
            assert "t_end" in params
            assert "dt" in params
        else:
            assert "script_code" in p


@pytest.mark.parametrize(
    "preset", [p for p in TUTORIALS if p["type"] == "single"]
)
def test_preset_builds_successfully(preset):
    """Verify that each single-patient preset successfully constructs a ModelConfig."""
    params = preset["parameters"]

    strains = params["strains"]
    phages = params["phages"]
    antibiotics = params["antibiotics"]
    doses = params["doses"]

    n_bacteria = len(strains)
    n_phages = len(phages)

    # Max depth
    max_depth = max([s.get("dormancy_depth", 1) for s in strains] if strains else [1])

    builder = ModelBuilder(n_bacteria=n_bacteria, n_phages=n_phages, n_latent=5, n_depth=max_depth)

    # Growth rates
    growth_rates = [s["growth_rate"] for s in strains]
    ratios = [s.get("bacteria_to_resource_ratio", 1e9) for s in strains]
    builder = builder.with_growth_rates(growth_rates, bacteria_to_resource_ratio=ratios)

    # Dormancy
    any_dormancy = any(s.get("dormancy_enabled", False) for s in strains)
    if any_dormancy:
        dormancy_rates = [s["dormancy_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
        resus_rates = [s["resuscitation_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
        diff_rates = [s["dormancy_diffusion_rate"] if s.get("dormancy_enabled", False) else 0.0 for s in strains]
        enabled_strains = [s for s in strains if s.get("dormancy_enabled", False)]
        ds = enabled_strains[0]["dormancy_signal"] if enabled_strains else "nutrient"
        rs = enabled_strains[0]["resuscitation_signal"] if enabled_strains else "nutrient"

        builder = builder.with_dormancy(
            dormancy_rate=np.array(dormancy_rates),
            resuscitation_rate=np.array(resus_rates),
            dormancy_diffusion_rate=np.array(diff_rates),
            dormancy_signal=ds,
            resuscitation_signal=rs
        )

    # Phages
    if n_phages > 0:
        # Build adsorption rates
        adsorption_rates = []
        adsorption_rates_dormant = []
        for s_idx in range(n_bacteria):
            s_ads = []
            s_ads_dorm = []
            for p_idx in range(n_phages):
                p_orig = phages[p_idx]
                ads = p_orig.get("adsorption_rates", 2e-9)
                ads_dorm = p_orig.get("adsorption_rates_dormant", 0.0)

                # Resolve lists or arrays
                val_ads = ads[s_idx] if isinstance(ads, list) else ads
                val_ads_dorm = ads_dorm[s_idx] if isinstance(ads_dorm, list) else ads_dorm

                s_ads.append(val_ads)
                s_ads_dorm.append(val_ads_dorm)

            adsorption_rates.append(s_ads)
            adsorption_rates_dormant.append(s_ads_dorm)

        burst_sizes = np.tile(np.array([p["burst_sizes"] for p in phages]), (n_bacteria, 1))
        latent_periods = np.tile(np.array([p["latent_periods"] for p in phages]), (n_bacteria, 1))
        decay_rates = [p["phage_decay_rates"] for p in phages]

        # Check if any phage has PK
        has_phage_pk = any(p["pk_mode"] != "None" for p in phages)
        pk_config = None
        if has_phage_pk:
            from pbisim import PhagePKConfig
            vcs = np.array([p.get("Vc", 5000.0) for p in phages])
            k_elims = np.array([p.get("k_elim", 0.2) if p["pk_mode"] != "None" else 0.0 for p in phages])
            k_ins = np.array([p.get("k_in", 0.1) if p["pk_mode"] != "None" else 0.0 for p in phages])
            k_outs = np.array([p.get("k_out", 0.05) if p["pk_mode"] != "None" else 0.0 for p in phages])

            has_mc = any(p["pk_mode"] == "Mass-Conserving" for p in phages)
            if has_mc:
                vis = np.array([p.get("Vi", 10.0) if p["pk_mode"] == "Mass-Conserving" else 0.0 for p in phages])
            else:
                vis = None

            pk_config = PhagePKConfig(
                n_phages=n_phages,
                Vc=vcs,
                k_elim=k_elims,
                k_in=k_ins,
                k_out=k_outs,
                Vi=vis
            )

        builder = builder.with_phage_params(
            adsorption_rates=np.array(adsorption_rates),
            adsorption_rates_dormant=np.array(adsorption_rates_dormant),
            burst_sizes=np.array(burst_sizes),
            latent_periods=np.array(latent_periods),
            phage_decay_rates=np.array(decay_rates)
        )
        if pk_config is not None:
            builder = builder.with_phage_pk(pk_config)

    # Antibiotics
    for abx in antibiotics:
        builder = builder.with_antibiotic(
            name=abx["name"],
            k_elim=abx["k_elim"],
            Vc=abx.get("Vc", 1.0),
            k12=abx.get("k12", 0.0),
            k21=abx.get("k21", 0.0),
            emax=abx["emax"],
            ec50=abx["ec50"],
            hill=abx.get("hill", 1.0),
            f_lyse=abx.get("f_lyse", 0.0),
            inoculum_effect_constant=abx.get("inoculum_effect_constant", None) if abx.get("inoculum_effect_constant", 0.0) > 0 else None
        )

    # Nutrients
    track_nutrients = params.get("track_nutrients", True)
    if not track_nutrients:
        from pbisim import logistic_growth
        builder = builder.with_growth_function(logistic_growth)
        builder = builder.with_nutrient(
            track_nutrients=False,
            carrying_capacity=params.get("carrying_capacity", 1e9)
        )
    else:
        builder = builder.with_nutrient(
            track_nutrients=True,
            monod_constant=params.get("monod_constant", 0.3),
            recycle_fraction=params.get("recycle_fraction", 0.0),
            s_in=params.get("s_in", 0.0),
            s_out=params.get("s_out", 0.0)
        )

    # Immunity
    immunity_enabled = params.get("immunity_enabled", False)
    if immunity_enabled:
        kill_rate_D = params.get("innate_kill_rate_D", 0.0)
        builder = builder.with_immunity(
            imm_kill_rate=params.get("innate_kill_rate", 1e7),
            imm_kill50=params.get("innate_kill50", 1e8),
            imm_decay_rate=params.get("innate_decay_rate", 0.05),
            immune_module=params.get("immune_module", "innate"),
            imm_max=params.get("innate_max", 1e7),
            imm_kill_rate_D=np.array([kill_rate_D] * n_bacteria) if kill_rate_D > 0 else None
        )

    # OD / Debris
    debris_enabled = params.get("debris_enabled", False)
    if debris_enabled:
        builder = builder.with_od_debris(
            u=params.get("debris_u", 1.0),
            v=params.get("debris_v", 0.5),
            kdis=params.get("debris_kdis", 0.1),
            od_to_cfu_conversion_factor=params.get("od_to_cfu_conversion_factor", 1.0)
        )

    # Dosing Schedule
    dose_events = []
    for d in doses:
        target_type = d["target_type"]
        if target_type == "phage":
            target = "phage"
            target_idx = d["target_idx"]
        elif target_type == "antibiotic":
            target = "antibiotic"
            target_idx = d["target_idx"]
        else:
            target = "nutrient"
            target_idx = 0

        event = DoseEvent(
            time=d["time"],
            amount=d["amount"],
            target=target,
            index=target_idx,
            route=d["route"],
            duration=d.get("duration", 0.0)
        )
        dose_events.append(event)

    if dose_events:
        builder = builder.with_dose_schedule(DoseSchedule(dose_events))

    config = builder.build()
    assert config is not None
