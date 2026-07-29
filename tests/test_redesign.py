"""Regression tests for the visual redesign (Pass B structure).

These assert the *structure* the redesign introduces (results header bar,
outcome badge, metric tiles) renders without error — not pixels.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_results_header_and_peak_phage_tile():
    """After a run, the Interactive Simulator shows a results header with a
    solver/runtime meta line, an outcome badge, and a Peak Phage Titre tile."""
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=180)
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
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=120)
    at.run()
    runs = [b for b in at.button if "Run Simulation" in (b.label or "")]
    assert runs and runs[0].proto.type == "primary"


def test_plot_axis_controls_render_after_run():
    """The 'Plot options' axis-control toggle appears once results exist."""
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=180)
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
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=180)
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
