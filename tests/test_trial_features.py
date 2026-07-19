"""Tests for the clinical-trial UI features: dose regimens, distribution metrics,
and PK/PD trajectory plots (trial_helper)."""

from __future__ import annotations

import numpy as np

from pbisim import ModelBuilder
from pbisim.trial.clinical import TreatmentArm
from pbisim.trial.population import InitialConditions
from pbisim.pk.dosing import DoseSchedule

from pbisim_app.trial_helper import (
    build_regimen_doses,
    trial_metric_fns,
    max_log_reduction,
    log_reduction_final,
    run_trial_simulation,
    plot_pkpd_trajectories_plotly,
)


def test_build_regimen_doses_single_and_repeat():
    single = build_regimen_doses("phage", 0, 1e9, 2.0, is_repeat=False,
                                 interval=8.0, n_doses=4)
    assert len(single) == 1
    assert single[0].time == 2.0 and single[0].target == "phage"

    repeat = build_regimen_doses("antibiotic", 1, 10.0, 0.0, is_repeat=True,
                                 interval=8.0, n_doses=3)
    assert [d.time for d in repeat] == [0.0, 8.0, 16.0]
    assert all(d.target == "antibiotic" and d.index == 1 for d in repeat)

    # zero amount => no doses
    assert build_regimen_doses("phage", 0, 0.0, 0.0, True, 8.0, 3) == []


def _phage_trial(t_end=48.0):
    b = ModelBuilder(n_bacteria=1, n_phages=1, n_latent=5, n_depth=1).with_growth_rates([1.0])
    b = b.with_phage_params(
        adsorption_rates=np.array([[1e-8]]), adsorption_rates_dormant=np.array([[0.0]]),
        burst_sizes=np.array([[50.0]]), latent_periods=np.array([[0.5]]),
        phage_decay_rates=np.array([0.1]),
    )
    b = b.with_nutrient(track_nutrients=True, monod_constant=0.3)
    cfg = b.build()
    cfg.initial_conditions = InitialConditions(B=np.array([1e7]), P=np.zeros(1), S=1.0)
    doses = build_regimen_doses("phage", 0, 1e9, 0.0, False, 8.0, 1)
    arms = [
        TreatmentArm("Control", dose_schedule=DoseSchedule([])),
        TreatmentArm("Phage", dose_schedule=DoseSchedule(doses)),
    ]
    return run_trial_simulation(
        cfg, [{"path": "growth_rates", "dist_type": "LogNormal",
               "params": {"cv": 0.2}, "mode": "multiplicative"}],
        arms, n_patients=4, t_end=t_end, dt=0.5, seed=1, pretreatment_hours=0.0,
        n_jobs=1, base_initial_B=np.zeros(1), base_initial_P=np.zeros(1), base_initial_S=1.0,
    )


def test_distribution_metrics_present_and_sane():
    assert "max_log_reduction" in trial_metric_fns()
    assert "log_reduction_final" in trial_metric_fns()

    res = _phage_trial()
    m = res["Phage"].metrics
    assert "max_log_reduction" in m.columns and "log_reduction_final" in m.columns
    # phage clears a 1e7 inoculum -> ~7-log max reduction
    assert m["max_log_reduction"].min() > 5.0
    # max reduction is always >= final reduction (nadir <= last point)
    assert (m["max_log_reduction"] >= m["log_reduction_final"] - 1e-6).all()


def test_pkpd_trajectory_plot_has_traces():
    res = _phage_trial()
    fig_cfu = plot_pkpd_trajectories_plotly(res, prefixes=("B", "D", "I", "H"))
    fig_pfu = plot_pkpd_trajectories_plotly(res, prefixes=("P",))
    # 2 arms x (median + IQR band) = 4 traces each
    assert len(fig_cfu.data) == 4
    assert len(fig_pfu.data) == 4


def test_metric_functions_directly():
    res = _phage_trial()
    r = res["Phage"].results[0]
    assert r is not None
    assert max_log_reduction(r) > 5.0
    assert np.isfinite(log_reduction_final(r))


def test_multiple_dose_arms_produce_distinct_outcomes():
    """Low-dose vs high-dose phage arms must give measurably different exposure/kill.

    This is the multi-arm trial feature: arbitrary named arms, each with its own dose
    schedule, run over the same cohort and compared.
    """
    b = ModelBuilder(n_bacteria=1, n_phages=1, n_latent=5, n_depth=1).with_growth_rates([1.2])
    b = b.with_phage_params(
        adsorption_rates=np.array([[3e-9]]), adsorption_rates_dormant=np.array([[0.0]]),
        burst_sizes=np.array([[30.0]]), latent_periods=np.array([[0.6]]),
        phage_decay_rates=np.array([0.3]),
    )
    b = b.with_nutrient(track_nutrients=True, monod_constant=0.3)
    cfg = b.build()
    cfg.initial_conditions = InitialConditions(B=np.array([1e7]), P=np.zeros(1), S=1.0)

    low = build_regimen_doses("phage", 0, 1e6, 0.0, False, 8.0, 1)
    high = build_regimen_doses("phage", 0, 1e10, 0.0, False, 8.0, 1)
    arms = [
        TreatmentArm("Control", dose_schedule=DoseSchedule([])),
        TreatmentArm("Low dose", dose_schedule=DoseSchedule(low)),
        TreatmentArm("High dose", dose_schedule=DoseSchedule(high)),
    ]
    res = run_trial_simulation(
        cfg, [], arms, n_patients=3, t_end=48.0, dt=0.5, seed=1, pretreatment_hours=0.0,
        n_jobs=1, base_initial_B=np.zeros(1), base_initial_P=np.zeros(1), base_initial_S=1.0,
    )
    assert res.arm_names == ["Control", "Low dose", "High dose"]

    def auc(name):
        return float(res[name].metrics["bacterial_auc"].median())

    # Control (no phage) grows unchecked -> far larger bacterial burden than either dose
    assert auc("Control") > auc("High dose") * 10
    # Higher dose clears faster -> lower bacterial AUC than the low dose
    assert auc("High dose") < auc("Low dose")
    # Peak phage exposure scales with dose
    hi_p = res["High dose"].get_trajectories("P")[1].max()
    lo_p = res["Low dose"].get_trajectories("P")[1].max()
    assert hi_p > lo_p


def test_dose_record_is_editable_inline():
    """Dose rows are now editable in place (time/amount/route) with a stable id, so an
    edit sticks and delete doesn't rebind widgets to the wrong row."""
    import matplotlib
    matplotlib.use("Agg")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("pbisim_app/app.py", default_timeout=120)
    at.run()
    at.session_state["int_doses"] = [
        {"time": 1.0, "amount": 1e8, "target_type": "phage", "target_idx": 0,
         "route": "bolus", "duration": 0.0}
    ]
    at.run()   # renders the Interactive Simulator incl. the Environment & Dosing tab
    assert len(at.exception) == 0, at.exception
    assert "_id" in at.session_state["int_doses"][0]   # stable id assigned

    times = [w for w in at.number_input if (w.label or "") == "Time (h)"]
    assert times, "an editable dose time input rendered"
    times[0].set_value(7.0).run()
    assert at.session_state["int_doses"][0]["time"] == 7.0   # edit persisted


def test_trial_arm_is_editable_inline():
    """Trial arms get an edit expander (rename + reconfigure regimen), keyed by a stable id."""
    import matplotlib
    matplotlib.use("Agg")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("pbisim_app/app.py", default_timeout=120)
    at.run()
    at.session_state["trial_arms"] = [
        {"name": "Low dose",
         "phage": {"on": True, "index": 0, "amount": 1e8, "start": 0.0,
                   "repeat": False, "interval": 8.0, "n": 1},
         "abx": {"on": False}}
    ]
    at.session_state["current_page_radio"] = "Clinical Trials & Cohorts"
    at.run()
    assert len(at.exception) == 0, at.exception
    assert "_id" in at.session_state["trial_arms"][0]
    assert any((w.label or "") == "Arm name" for w in at.text_input)   # edit form rendered
