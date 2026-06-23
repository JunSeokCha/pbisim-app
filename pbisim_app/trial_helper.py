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
from pbisim import PBIModel


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
            
        # Extract additional keyword arguments
        kwargs = {}
        for k, v in base_kwargs.items():
            if v is not None:
                kwargs[k] = v
                
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
