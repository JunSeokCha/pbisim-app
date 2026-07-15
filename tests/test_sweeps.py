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
    assert at.session_state["dr_sweep_phg_series_0"] == "1e3, 1e5, 1e7, 1e9"
    at.session_state["dr_sweep_phg_unit_0"] = "PFU (absolute)"
    at.run()
    [b for b in at.button if "Run Dose-Response" in (b.label or "")][0].click().run()

    assert len(at.exception) == 0
    headers = [m.value for m in at.markdown]
    assert any("Viable Bacteria" in h for h in headers)   # CFU chart
    assert any("Optical Density" in h for h in headers)   # OD chart
