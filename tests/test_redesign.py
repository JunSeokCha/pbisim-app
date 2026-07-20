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
    """The 'Plot options' axis controls appear once results exist."""
    at = AppTest.from_file("pbisim_app/app.py", default_timeout=180)
    at.run()
    [b for b in at.button if "Run Simulation" in (b.label or "")][0].click().run()
    assert len(at.exception) == 0, at.exception
    labels = [e.label for e in at.expander]
    assert any("Plot options" in (l or "") for l in labels), labels


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
