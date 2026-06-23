"""
sweep_helper.py — Utility functions for mapping, applying, and executing parameter sweeps
and dose-response simulations in the pbisim Streamlit app.
"""

from __future__ import annotations

import dataclasses
import numpy as np
from pbisim import PBIModel, solve_ode, stationary_phase_ic

def get_sweep_parameters(config, strains=None, phages=None, antibiotics=None) -> dict:
    """
    Dynamically constructs a dictionary of sweepable parameters based on the current model configuration.
    Maps user-friendly labels to target fields, types, and indices.
    """
    params = {}

    # 1. Global scalars (always available)
    params["Monod Constant (Ks)"] = {
        "type": "scalar",
        "field": "monod_constant",
    }
    params["Carrying Capacity (K)"] = {
        "type": "scalar",
        "field": "carrying_capacity",
    }
    params["Recycle Fraction"] = {
        "type": "scalar",
        "field": "recycle_fraction",
    }
    params["Medium Inflow (s_in)"] = {
        "type": "scalar",
        "field": "s_in",
    }
    params["Medium Washout (s_out)"] = {
        "type": "scalar",
        "field": "s_out",
    }
    params["Immune Decay Rate"] = {
        "type": "scalar",
        "field": "imm_decay_rate",
    }
    params["Immune Kill 50 (K_kill)"] = {
        "type": "scalar",
        "field": "imm_kill50",
    }
    params["Immune Capacity (Imm_max)"] = {
        "type": "scalar",
        "field": "imm_max",
    }
    params["OD-to-CFU Conversion Factor"] = {
        "type": "scalar",
        "field": "od_to_cfu_conversion_factor",
    }
    params["Debris Dissolution Rate (k_dis)"] = {
        "type": "scalar",
        "field": "debris_kdis",
    }
    
    # Dimensions (Q & L)
    params["Dormancy Depth Layers (Q)"] = {
        "type": "dimension",
        "field": "n_depth",
    }
    params["Phage Latent Stages (L)"] = {
        "type": "dimension",
        "field": "n_latent",
    }

    # 2. Strain-specific parameters
    for i in range(config.n_bacteria):
        strain_name = f"Strain {i}"
        if strains and i < len(strains):
            strain_name = f"Strain {i} ({strains[i]['name']})"
        elif hasattr(config, "strain_labels") and config.strain_labels and i < len(config.strain_labels):
            strain_name = f"Strain {i} ({config.strain_labels[i]})"

        params[f"Growth Rate - {strain_name}"] = {
            "type": "array1d",
            "field": "growth_rates",
            "index": i,
        }
        params[f"Bacteria-Resource Ratio - {strain_name}"] = {
            "type": "array1d",
            "field": "bacteria_to_resource_ratio",
            "index": i,
        }
        params[f"Dormancy Rate (sleep) - {strain_name}"] = {
            "type": "array1d",
            "field": "dormancy_rate",
            "index": i,
        }
        params[f"Resuscitation Rate (wake) - {strain_name}"] = {
            "type": "array1d",
            "field": "resuscitation_rate",
            "index": i,
        }
        params[f"Dormancy Diffusion Rate - {strain_name}"] = {
            "type": "array1d",
            "field": "dormancy_diffusion_rate",
            "index": i,
        }
        params[f"Natural Death Rate (dB) - {strain_name}"] = {
            "type": "array1d_or_none",
            "field": "death_rate_B",
            "index": i,
        }
        params[f"Dormant Death Rate (dD) - {strain_name}"] = {
            "type": "array1d_or_none",
            "field": "death_rate_D",
            "index": i,
        }
        params[f"Immune Kill Rate (Dormant) - {strain_name}"] = {
            "type": "array1d_or_none",
            "field": "imm_kill_rate_D",
            "index": i,
        }

    # 3. Phage-specific parameters
    for j in range(config.n_phages):
        phage_name = f"Phage {j}"
        if phages and j < len(phages):
            phage_name = f"Phage {j} ({phages[j]['name']})"

        params[f"Phage Decay Rate - {phage_name}"] = {
            "type": "array1d",
            "field": "phage_decay_rates",
            "index": j,
        }

        # 2D arrays: adsorption, burst, latent (bacteria x phage)
        for i in range(config.n_bacteria):
            strain_name = f"Strain {i}"
            if strains and i < len(strains):
                strain_name = strains[i]['name']
            elif hasattr(config, "strain_labels") and config.strain_labels and i < len(config.strain_labels):
                strain_name = config.strain_labels[i]

            params[f"Adsorption - {phage_name} on {strain_name}"] = {
                "type": "array2d",
                "field": "adsorption_rates",
                "index_row": i,
                "index_col": j,
            }
            params[f"Dormant Adsorption - {phage_name} on {strain_name}"] = {
                "type": "array2d",
                "field": "adsorption_rates_dormant",
                "index_row": i,
                "index_col": j,
            }
            params[f"Burst Size - {phage_name} on {strain_name}"] = {
                "type": "array2d",
                "field": "burst_sizes",
                "index_row": i,
                "index_col": j,
            }
            params[f"Latent Period - {phage_name} on {strain_name}"] = {
                "type": "array2d",
                "field": "latent_periods",
                "index_row": i,
                "index_col": j,
            }

    # 4. Antibiotics-specific parameters
    if config.n_antibiotics > 0 and config.pk_config is not None:
        for j in range(config.n_antibiotics):
            abx_name = f"Antibiotic {j}"
            if antibiotics and j < len(antibiotics):
                abx_name = f"Antibiotic {j} ({antibiotics[j]['name']})"

            params[f"Clearance (k_elim) - {abx_name}"] = {
                "type": "pk_array1d",
                "field": "k_elim",
                "index": j,
            }
            params[f"Volume (Vc) - {abx_name}"] = {
                "type": "pk_array1d",
                "field": "Vc",
                "index": j,
            }

            if config.pd_config is not None:
                for i in range(config.n_bacteria):
                    strain_name = f"Strain {i}"
                    if strains and i < len(strains):
                        strain_name = strains[i]['name']
                    elif hasattr(config, "strain_labels") and config.strain_labels and i < len(config.strain_labels):
                        strain_name = config.strain_labels[i]

                    params[f"Emax - {abx_name} on {strain_name}"] = {
                        "type": "pd_array2d",
                        "field": "abx_emax",
                        "index_row": i,
                        "index_col": j,
                    }
                    params[f"EC50 - {abx_name} on {strain_name}"] = {
                        "type": "pd_array2d",
                        "field": "abx_ec50",
                        "index_row": i,
                        "index_col": j,
                    }
                    params[f"Hill (H) - {abx_name} on {strain_name}"] = {
                        "type": "pd_array2d",
                        "field": "abx_hill",
                        "index_row": i,
                        "index_col": j,
                    }

    # 5. Initial conditions
    for i in range(config.n_bacteria):
        strain_name = f"Strain {i}"
        if strains and i < len(strains):
            strain_name = f"Strain {i} ({strains[i]['name']})"
        elif hasattr(config, "strain_labels") and config.strain_labels and i < len(config.strain_labels):
            strain_name = f"Strain {i} ({config.strain_labels[i]})"

        params[f"Initial Density (B0) - {strain_name}"] = {
            "type": "initial_B",
            "index": i,
        }

    for j in range(config.n_phages):
        phage_name = f"Phage {j}"
        if phages and j < len(phages):
            phage_name = f"Phage {j} ({phages[j]['name']})"

        params[f"Initial Density (P0) - {phage_name}"] = {
            "type": "initial_P",
            "index": j,
        }

    params["Initial Resource Substrate (S0)"] = {
        "type": "initial_S",
    }

    return params


def apply_sweep_parameter(val: float, meta: dict, config, initial_B, initial_P, initial_S, model_kwargs) -> tuple:
    """
    Applies a sweep value to the model configuration or initial conditions.
    Returns a new tuple of (config, initial_B, initial_P, initial_S, model_kwargs).
    Resizes initial_D if n_depth changes.
    """
    # Clone everything to prevent side-effects
    config = dataclasses.replace(config)
    initial_B = np.copy(initial_B)
    initial_P = np.copy(initial_P)
    model_kwargs = dict(model_kwargs)
    if "initial_D" in model_kwargs and model_kwargs["initial_D"] is not None:
        model_kwargs["initial_D"] = np.copy(model_kwargs["initial_D"])

    param_type = meta["type"]

    if param_type == "scalar":
        setattr(config, meta["field"], val)

    elif param_type == "dimension":
        field = meta["field"]
        if field == "n_depth":
            new_depth = int(val)
            config = dataclasses.replace(config, n_depth=new_depth)
            # Resize initial_D if present
            if "initial_D" in model_kwargs and model_kwargs["initial_D"] is not None:
                init_D = model_kwargs["initial_D"]
                total_D = np.sum(init_D, axis=0) # shape (n_bacteria,)
                new_init_D = np.zeros((new_depth, config.n_bacteria))
                for q in range(new_depth):
                    new_init_D[q, :] = total_D / new_depth
                model_kwargs["initial_D"] = new_init_D
        elif field == "n_latent":
            config = dataclasses.replace(config, n_latent=int(val))

    elif param_type == "array1d":
        arr = np.copy(getattr(config, meta["field"]))
        arr[meta["index"]] = val
        setattr(config, meta["field"], arr)

    elif param_type == "array1d_or_none":
        arr = getattr(config, meta["field"])
        if arr is None:
            arr = np.zeros(config.n_bacteria)
        else:
            arr = np.copy(arr)
        arr[meta["index"]] = val
        setattr(config, meta["field"], arr)

    elif param_type == "array2d":
        arr = np.copy(getattr(config, meta["field"]))
        arr[meta["index_row"], meta["index_col"]] = val
        setattr(config, meta["field"], arr)

    elif param_type == "pk_array1d":
        # pk_array1d always targets the antibiotic PKConfig, not PhagePKConfig.
        # Using phage_pk_config here when both are set would corrupt phage PK params.
        pk_config = dataclasses.replace(config.pk_config)
        arr = np.copy(getattr(pk_config, meta["field"]))
        arr[meta["index"]] = val
        setattr(pk_config, meta["field"], arr)
        config = dataclasses.replace(config, pk_config=pk_config)

    elif param_type == "pd_array2d":
        pd_config = dataclasses.replace(config.pd_config)
        arr = np.copy(getattr(pd_config, meta["field"]))
        arr[meta["index_row"], meta["index_col"]] = val
        setattr(pd_config, meta["field"], arr)
        config = dataclasses.replace(config, pd_config=pd_config)

    elif param_type == "initial_B":
        initial_B[meta["index"]] = val

    elif param_type == "initial_P":
        initial_P[meta["index"]] = val

    elif param_type == "initial_S":
        initial_S = val

    return config, initial_B, initial_P, initial_S, model_kwargs


def parse_comma_separated_series(text: str) -> list[float]:
    """
    Parses a comma-separated string of values into a list of floats.
    Supports linear, scientific notation, and spaces.
    """
    if not text.strip():
        return []
    parts = text.split(",")
    vals = []
    for p in parts:
        p_clean = p.strip()
        if p_clean:
            vals.append(float(p_clean))
    return vals


def pad_vectors(vectors: dict[str, list[float]]) -> tuple[dict[str, list[float]], list[str]]:
    """
    Pads all non-empty lists in vectors to the maximum length of any list.
    Padding is done by repeating the last value in the list.
    Returns the padded vectors and a list of warning messages detailing which lists were padded.
    """
    non_empty = {k: v for k, v in vectors.items() if len(v) > 0}
    if not non_empty:
        return {}, []

    max_len = max(len(v) for v in non_empty.values())
    padded_vectors = {}
    warnings = []

    for k, v in non_empty.items():
        if len(v) < max_len:
            pad_val = v[-1]
            padded_v = list(v) + [pad_val] * (max_len - len(v))
            padded_vectors[k] = padded_v
            warnings.append(f"Dose series for '{k}' was padded from length {len(v)} to {max_len} using the last value {pad_val:.1e}.")
        else:
            padded_vectors[k] = list(v)

    return padded_vectors, warnings
