"""Regression tests for the visual redesign (Pass B structure).

These assert the *structure* the redesign introduces (results header bar,
outcome badge, metric tiles) renders without error — not pixels.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest
from pathlib import Path as _Path

APP = str(_Path(__file__).resolve().parents[1] / "pbisim_app" / "app.py")


def test_results_header_and_peak_phage_tile():
    """After a run, the Interactive Simulator shows a results header with a
    solver/runtime meta line, an outcome badge, and a Peak Phage Titre tile."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    [b for b in at.button if "Run Simulation" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception

    blob = " ".join(m.value for m in at.markdown)
    assert "Simulation results" in blob
    assert "Peak Phage Titre" in blob
    assert any(w in blob for w in ("Suppressed", "Cleared", "Regrowth", "Uncontrolled")), blob
    # runtime was captured for the meta line (SafeSessionState: no .get())
    assert "sim_runtime" in at.session_state and at.session_state["sim_runtime"] is not None


def test_run_button_is_primary():
    """The main Run action is a primary button (visual hierarchy)."""
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    runs = [b for b in at.button if "Run Simulation" in (b.label or "")]
    assert runs and runs[0].proto.type == "primary"


def test_plot_axis_controls_render_after_run():
    """The 'Plot options' axis-control toggle appears once results exist."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    [b for b in at.button if "Run Simulation" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception
    labels = [c.label for c in at.checkbox]
    assert any("Plot options" in (l or "") for l in labels), labels


def test_axis_limit_pair_parsing():
    """The 'min, max' box parses full and partial ranges; a lone value autoscales."""
    from pbisim_app.viz_helper import _pair
    assert _pair("1, 1e9") == (1.0, 1e9)
    assert _pair("0,48") == (0.0, 48.0)
    assert _pair("1,") == (1.0, None)          # partial (mpl autoscales the top)
    assert _pair(", 1e9") == (None, 1e9)
    assert _pair("") == (None, None)
    assert _pair("5") == (None, None)          # ambiguous single value -> ignore


def test_viz_helper_apply_functions():
    """apply_axis_mpl / apply_axis_plotly set scale + limits (log10-space for plotly)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from pbisim_app.viz_helper import apply_axis_mpl, apply_axis_plotly

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [10, 100, 1000])
    apply_axis_mpl(ax, {"x_scale": "Linear", "y_scale": "Log",
                        "xlim": (None, None), "ylim": (1.0, None)})
    assert ax.get_yscale() == "log" and ax.get_xscale() == "linear"
    assert ax.get_ylim()[0] == 1.0
    # picking Linear overrides a prior log scale
    apply_axis_mpl(ax, {"x_scale": "Linear", "y_scale": "Linear",
                        "xlim": (None, None), "ylim": (None, None)})
    assert ax.get_yscale() == "linear"

    f = go.Figure()
    apply_axis_plotly(f, {"x_scale": "Linear", "y_scale": "Log",
                          "xlim": (None, None), "ylim": (1.0, 1e4)})
    assert f.layout.yaxis.type == "log"
    assert abs(f.layout.yaxis.range[1] - 4.0) < 1e-9  # log10(1e4)
    # non-positive bound on a log axis -> autorange, no crash
    f2 = go.Figure()
    apply_axis_plotly(f2, {"x_scale": "Linear", "y_scale": "Log",
                           "xlim": (None, None), "ylim": (0.0, 1e4)})
    assert f2.layout.yaxis.range is None


def test_entity_series_selector_renders():
    """After a run, the Bacterial tab offers per-compartment series checkboxes
    (registry-driven), with the defaults reproducing the previous view."""
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    [b for b in at.button if "Run Simulation" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception
    labels = [c.label for c in at.checkbox]
    # the CFU aggregate + per-strain active series are offered; CFU on by default
    assert "CFU (culturable: B+D)" in labels, labels
    assert any("(active)" in (l or "") for l in labels), labels
    tv = [c for c in at.checkbox if c.label == "CFU (culturable: B+D)"][0]
    assert tv.value is True


def test_build_series_registry():
    """build_series enumerates compartments with sane defaults (active/dormant on,
    infected/hibernating off) and every getter returns a finite trajectory."""
    import numpy as np
    from pbisim import ModelBuilder, PBIModel, solve_ode
    from pbisim_app.viz_helper import build_series
    b = (ModelBuilder(n_bacteria=2, n_phages=1, n_latent=2, n_depth=2)
         .with_growth_rates([1.2, 1.1])
         .with_dormancy(dormancy_rate=np.array([0.1, 0.1]),
                        resuscitation_rate=np.array([0.05, 0.05]),
                        dormancy_diffusion_rate=np.array([0.02, 0.02])))
    cfg = b.build()
    r = solve_ode(PBIModel(cfg, initial_B=np.array([1e7, 10.0]),
                           initial_P=np.array([1e6]), initial_S=1.0), t_end=5, dt=1.0)
    strains = [{"name": "WT", "dormancy_enabled": True}, {"name": "Mut", "dormancy_enabled": True}]
    S = build_series(r, config=cfg, strains=strains, phages=[{"name": "P0", "pk_mode": "None"}],
                     antibiotics=[], builder_mode="Direct (ModelBuilder)")
    by_key = {s.key: s for s in S}
    # CFU (B+D) is the default aggregate; total-incl-infected and active-only are opt-in.
    assert by_key["cfu"].default and by_key["B0"].default and by_key["D_0"].default
    assert not by_key["total_viable"].default and not by_key["total_active"].default
    assert not by_key["I_0"].default and not by_key["H_0"].default
    # CFU excludes I/H; the "total incl. infected" series includes them.
    import numpy as _np
    assert _np.allclose(by_key["cfu"].getter(r), r.sum_prefixes("B", "D"))
    assert _np.allclose(by_key["total_viable"].getter(r), r.sum_prefixes("B", "D", "I", "H"))
    for s in S:
        assert np.isfinite(np.asarray(s.getter(r), dtype=float)).all()


def test_phage_pk_central_and_peripheral_series_and_helpers():
    """With phage PK on, build_series exposes the CENTRAL (blood conc. = Pc/Vc, default on)
    and — for a 2-compartment model (k12 > 0) — the PERIPHERAL (Pp amount) series; the
    common helpers compute the central concentration and peripheral amount."""
    import numpy as np
    from pbisim import ModelBuilder, PBIModel, solve_ode, PhagePKConfig
    from pbisim_app.viz_helper import build_series
    from pbisim_app.common import (central_phage_total, peripheral_phage_total,
                                    phage_uses_two_compartment, phage_pk_enabled)
    pk = PhagePKConfig(n_phages=1, Vc=np.array([5000.0]), k_elim=np.array([0.2]),
                       k_in=np.array([0.1]), k_out=np.array([0.05]),
                       k12=np.array([0.3]), k21=np.array([0.15]))   # 2-compartment
    cfg = (ModelBuilder(n_bacteria=1, n_phages=1).with_growth_rates([0.8])
           .with_phage_params(burst_sizes=[[80]], latent_periods=[[0.5]],
                              adsorption_rates=[[1e-8]]).with_phage_pk(pk)).build()
    r = solve_ode(PBIModel(cfg, initial_B=np.array([1e7]), initial_P=np.array([0.0]),
                           initial_Pc=np.array([1e9])), t_end=12, dt=0.5)
    phages = [{"name": "P0", "pk_mode": "Effect Compartment", "Vc": 5000.0,
               "k12": 0.3, "k21": 0.15}]
    S = {s.key: s for s in build_series(r, config=cfg, strains=[{"name": "WT"}],
                                        phages=phages, antibiotics=[],
                                        builder_mode="Direct (ModelBuilder)")}
    assert "P0" in S and "Pc0" in S and "Pp0" in S          # site + central + peripheral
    assert S["P0"].default and S["Pc0"].default             # site + central on by default
    assert not S["Pp0"].default                             # peripheral opt-in
    assert np.allclose(S["Pc0"].getter(r), r.get("Pc0") / 5000.0)   # central = Pc/Vc
    # common helpers
    assert phage_pk_enabled(phages) and phage_uses_two_compartment(phages)
    assert float(np.max(central_phage_total(r, phages))) > 0
    assert float(np.max(peripheral_phage_total(r, phages))) > 0
    # 1-compartment phage → no peripheral series / zero peripheral
    phages1 = [{"name": "P0", "pk_mode": "Effect Compartment", "Vc": 5000.0}]
    S1 = {s.key for s in build_series(r, config=cfg, strains=[{"name": "WT"}],
                                      phages=phages1, antibiotics=[],
                                      builder_mode="Direct (ModelBuilder)")}
    assert "Pc0" in S1 and "Pp0" not in S1
    assert not phage_uses_two_compartment(phages1)
