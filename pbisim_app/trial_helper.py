"""
trial_helper.py — Cohort simulation, IIVSpec mapping, and Plotly visualization.
"""

from __future__ import annotations

import math
from typing import Callable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from pbisim.trial.clinical import ClinicalTrial, TreatmentArm, PretreatmentPhase
from pbisim.trial.population import IIVSpec, VirtualPopulation, InitialConditions
from pbisim.trial.distributions import LogNormal, Normal, Uniform, Fixed
from pbisim.trial.runner import default_metrics
from pbisim import PBIModel

_CFU = ("B", "D", "I", "H")


def _total_cfu(result):
    """Total bacterial CFU trajectory (all compartments), floored at 1.0 for log."""
    return np.maximum(result.sum_prefixes(*_CFU), 1.0)


def max_log_reduction(result) -> float:
    """Largest log10 drop from the baseline (t=0) CFU to the trajectory nadir."""
    cfu = _total_cfu(result)
    return float(np.log10(cfu[0]) - np.log10(cfu.min()))


def log_reduction_final(result) -> float:
    """log10 reduction from baseline (t=0) to the last observation time point.

    Positive = net reduction; negative = net regrowth above baseline.
    """
    cfu = _total_cfu(result)
    return float(np.log10(cfu[0]) - np.log10(cfu[-1]))


def build_regimen_doses(target, index, amount, start, is_repeat, interval, n_doses,
                        route="bolus", duration=0.0):
    """Return a list of DoseEvent for a single or repeated (qX h × N) regimen.

    ``is_repeat=False`` yields a single dose at *start*; ``is_repeat=True`` yields
    *n_doses* doses spaced *interval* hours apart. An amount ≤ 0 yields no doses.
    """
    from pbisim.pk.dosing import DoseEvent
    if amount <= 0:
        return []
    n = int(n_doses) if is_repeat else 1
    step = float(interval) if is_repeat else 0.0
    return [
        DoseEvent(time=float(start) + k * step, amount=float(amount), target=target,
                  index=int(index), route=route, duration=float(duration))
        for k in range(n)
    ]


def trial_metric_fns() -> dict:
    """default_metrics plus the app's max/final log-reduction distribution metrics."""
    fns = dict(default_metrics())
    fns["max_log_reduction"] = max_log_reduction
    fns["log_reduction_final"] = log_reduction_final
    return fns


# Map parameter displays to actual dotted paths
IIV_PARAMETERS = {
    "Bacterial Growth Rate": "growth_rates",
    "Bacterial Dormancy Rate": "dormancy_rate",
    "Bacterial Resuscitation Rate": "resuscitation_rate",
    "Bacterial Resource Consumption Ratio": "bacteria_to_resource_ratio",
    "Monod Constant (Ks)": "monod_constant",
    "Antibiotic Central volume (Vc)": "pk_config.Vc",
    "Antibiotic Clearance (k_elim)": "pk_config.k_elim",
    "Phage Adsorption Rate": "adsorption_rates",
    "Phage Burst Size": "burst_sizes",
    "Phage Latent Period": "latent_periods",
    "Phage Volume (Vc)": "phage_pk_config.Vc",
    "Phage Clearance (k_elim)": "phage_pk_config.k_elim",
    "Initial Bacterial Density": "ic.B",
    "Initial Phage Density": "ic.P",
    "Initial Nutrient Substrate": "ic.S",
}


def build_distribution(dist_type: str, params: dict):
    """Factory to construct pbisim IIV Distribution objects from UI inputs."""
    if dist_type == "LogNormal":
        cv = params.get("cv", 0.25)
        return LogNormal(cv=cv)
    elif dist_type == "Normal":
        mean = params.get("mean", 1.0)
        sd = params.get("sd", 0.1)
        return Normal(mean=mean, sd=sd)
    elif dist_type == "Uniform":
        lo = params.get("lo", 0.5)
        hi = params.get("hi", 1.5)
        return Uniform(lo=lo, hi=hi)
    else:
        value = params.get("value", 1.0)
        return Fixed(value=value)


def create_model_factory(
    base_initial_B: np.ndarray,
    base_initial_P: np.ndarray,
    base_initial_S: float,
    **base_kwargs
) -> Callable[[ModelConfig], PBIModel]:
    """
    Constructs a model factory closure that generates patient-specific PBIModel instances.
    """
    def factory(config):
        ic = config.initial_conditions

        # Resolve bacterial density
        if ic is not None and ic.B is not None:
            init_B = ic.B
        else:
            init_B = base_initial_B

        # Resolve phage density
        if ic is not None and ic.P is not None:
            init_P = ic.P
        else:
            init_P = base_initial_P

        # Resolve substrate density
        if ic is not None and ic.S is not None:
            init_S = float(ic.S)
        else:
            init_S = base_initial_S
        init_S = max(init_S, 0.0)  # a PretreatmentPhase can leave S slightly negative

        # Extract additional keyword arguments
        kwargs = {}
        for k, v in base_kwargs.items():
            if v is not None:
                kwargs[k] = v

        # Prefer the config's initial conditions for the dormant reservoir (D) and
        # immune priming (Imm). When a PretreatmentPhase runs, it replaces
        # config.initial_conditions with the full stationary-phase state; taking D/Imm
        # from the GUI base kwargs instead would discard the (usually dominant) dormant
        # population and immune priming, collapsing the treatment population — the same
        # bug fixed for the interactive simulator's pre-run.
        if ic is not None:
            if getattr(ic, "D", None) is not None:
                kwargs["initial_D"] = ic.D
            if getattr(ic, "Imm", None) is not None:
                kwargs["initial_Imm"] = ic.Imm

        return PBIModel(config, initial_B=init_B, initial_P=init_P, initial_S=init_S, **kwargs)
    return factory


def run_trial_simulation(
    base_config,
    iiv_inputs: list[dict],
    arms_list: list[TreatmentArm],
    n_patients: int,
    t_end: float,
    dt: float,
    seed: int,
    pretreatment_hours: float,
    n_jobs: int,
    base_initial_B: np.ndarray,
    base_initial_P: np.ndarray,
    base_initial_S: float,
    **base_kwargs
):
    """
    Assembles IIVSpec, VirtualPopulation, and ClinicalTrial, then executes parallel trial.
    """
    # 1. Build IIV specification dictionary
    spec_dict = {}
    for entry in iiv_inputs:
        path = entry["path"]
        dist = build_distribution(entry["dist_type"], entry["params"])
        mode = entry["mode"]
        spec_dict[path] = (dist, mode)
        
    iiv_spec = IIVSpec(spec_dict)
    
    # 2. Build model factory
    model_factory = create_model_factory(
        base_initial_B=base_initial_B,
        base_initial_P=base_initial_P,
        base_initial_S=base_initial_S,
        **base_kwargs
    )
    
    # 3. Setup PretreatmentPhase if requested
    pretreatment = None
    if pretreatment_hours > 0:
        pretreatment = PretreatmentPhase(t_prerun=pretreatment_hours, dt=dt)
        
    # 4. Build and run ClinicalTrial
    trial = ClinicalTrial(
        base_config=base_config,
        iiv_spec=iiv_spec,
        n_patients=n_patients,
        arms=arms_list,
        model_factory=model_factory,
        t_end=t_end,
        dt=dt,
        seed=seed,
        pretreatment=pretreatment,
        n_jobs=n_jobs,
        metric_fns=trial_metric_fns(),
        on_error="skip"
    )
    
    return trial.run()


def plot_kaplan_meier_plotly(result, endpoint="tte", t_end=72.0, threshold=100.0, n_logs=2.0):
    """
    Generates an interactive Plotly step-survival curve for time-to-event outcomes.
    """
    # Retrieve outcome dataframe
    df = result.outcome_dataframe(endpoint=endpoint, t_end=t_end, threshold=threshold, n_logs=n_logs)
    if df.empty:
        return go.Figure()
        
    fig = go.Figure()
    
    # Group by arm
    for arm_name, group in df.groupby("arm"):
        group = group.sort_values("duration")
        times = [0.0]
        survival = [1.0]
        
        n_total = len(group)
        n_surviving = n_total
        
        for idx, row in group.iterrows():
            t = row["duration"]
            is_event = row["event"]
            
            # Step point (before event)
            times.append(t)
            survival.append(n_surviving / n_total)
            
            if is_event:
                n_surviving -= 1
                
            # Step point (after event)
            times.append(t)
            survival.append(n_surviving / n_total)
            
        # Censoring endpoint
        times.append(t_end)
        survival.append(n_surviving / n_total)
        
        fig.add_trace(go.Scatter(
            x=times,
            y=survival,
            mode="lines",
            name=arm_name,
            line=dict(shape="hv", width=3)
        ))
        
    fig.update_layout(
        title=f"Kaplan-Meier: {endpoint.upper()} Survival (N={len(df)//len(result.arm_names)} per arm)",
        xaxis_title="Time (hours)",
        yaxis_title="Proportion Remaining",
        yaxis_range=[-0.05, 1.05],
        xaxis_range=[-0.5, t_end + 0.5],
        template="plotly_dark",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a #rrggbb hex string to an rgba(...) string with the given alpha."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(99,110,250,{alpha})"
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def plot_pkpd_trajectories_plotly(
    result,
    prefixes=("B", "D", "I", "H"),
    title="Total Bacteria (CFU/mL)",
    y_label="log₁₀ CFU/mL",
    log10=True,
    floor=1.0,
):
    """Per-arm median trajectory (+ IQR band) for a set of state prefixes.

    Use ``prefixes=("B","D","I","H")`` for total bacteria (CFU) and
    ``prefixes=("P",)`` for free phage (PFU).
    """
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    for i, arm_name in enumerate(result.arm_names):
        tr = result[arm_name]
        try:
            time, traj = tr.get_trajectories(*prefixes)
        except (ValueError, KeyError):
            continue
        if log10:
            traj = np.log10(np.maximum(traj, float(floor)))

        median = np.nanmedian(traj, axis=0)
        lo = np.nanpercentile(traj, 25, axis=0)
        hi = np.nanpercentile(traj, 75, axis=0)
        color = palette[i % len(palette)]

        # IQR band (25–75th percentile)
        fig.add_trace(go.Scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([hi, lo[::-1]]),
            fill="toself", fillcolor=_hex_to_rgba(color, 0.18),
            line=dict(width=0), hoverinfo="skip", showlegend=False,
            name=f"{arm_name} IQR",
        ))
        # Median line
        fig.add_trace(go.Scatter(
            x=time, y=median, mode="lines", name=arm_name,
            line=dict(color=color, width=3),
        ))

    fig.update_layout(
        title=f"{title} — median & IQR per arm",
        xaxis_title="Time (hours)",
        yaxis_title=y_label,
        template="plotly_dark",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_metric_distributions_plotly(result, metric="bacterial_auc"):
    """
    Generates an interactive Plotly box plot comparing patient-level outcome metrics.
    """
    arm_dfs = []
    for arm_name in result.arm_names:
        arm_res = result[arm_name]
        if arm_res is not None and arm_res.metrics is not None:
            df = arm_res.metrics.copy()
            df["arm"] = arm_name
            arm_dfs.append(df)
            
    if not arm_dfs:
        return go.Figure()
        
    df_all = pd.concat(arm_dfs, ignore_index=True)
    
    if metric not in df_all.columns:
        return go.Figure()
        
    # Clean label
    y_label = metric.replace('_', ' ').title()
    
    fig = px.box(
        df_all,
        x="arm",
        y=metric,
        color="arm",
        points="all",
        template="plotly_dark",
        title=f"Cohort Outcome Distribution: {y_label}",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Treatment Arm",
        yaxis_title=y_label,
    )
    return fig
