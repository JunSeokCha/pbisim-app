"""fit_helper.py — data ingestion, observable registry, and overlay/fit math for
the Calibration page (Phase A) and the future pbisim-fit integration.

Kept free of Streamlit so it is unit-testable and reusable. The canonical dataset
representation is the pbisim-fit long format: columns ``time, arm, observable,
value`` (plus a per-arm condition map with the phage MOI), so an ingested dataset
feeds pbisim-fit directly when that integration lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Observable registry ────────────────────────────────────────────────────────
# Each observable declares which model compartments it reflects and the "link" that
# turns model state into the measured signal. Adding a new signal (fluorescence, a
# reporter, …) is one entry — the overlay/fit machinery is unchanged.
#   link = None                      -> signal = model quantity        (CFU, PFU)
#   link = (param, "div", default)   -> signal = model quantity / param (OD)
#   link = (param, "mul", default)   -> signal = model quantity * param (luminescence)
OBSERVABLES = {
    "cfu": {"label": "CFU/mL",               "prefixes": ("B", "D", "I", "H"), "log": True,  "link": None},
    "pfu": {"label": "PFU/mL",               "prefixes": ("P",),               "log": True,  "link": None},
    "od":  {"label": "Optical density (OD)", "prefixes": ("B", "D", "I", "H"), "log": False, "link": ("od_to_cfu", "div", 1e9)},
    "lum": {"label": "Luminescence (RLU)",   "prefixes": ("B",),               "log": True,  "link": ("rlu_per_cell", "mul", 1.0)},
}


def predicted_observable(result, obs_key, link_value=None, use_model_od=False):
    """Predicted measured signal from a SimulationResult for the given observable.

    ``result`` only needs a ``sum_prefixes(*prefixes)`` method, so this works for any
    pbisim ``SimulationResult``.

    When ``use_model_od`` is set and the observable is OD, the debris-inclusive OD
    from the model (``result.get_od()``, which folds in lysed-cell debris and uses the
    model's own ``od_to_cfu_conversion_factor``) is used instead of the simple
    biomass/link scaling — so an enabled OD/debris module propagates into the overlay.
    """
    if obs_key == "od" and use_model_od and hasattr(result, "get_od"):
        return result.get_od()
    spec = OBSERVABLES[obs_key]
    qty = result.sum_prefixes(*spec["prefixes"])
    link = spec.get("link")
    if link is None:
        return qty
    _, op, default = link
    lv = float(link_value) if link_value is not None else float(default)
    return qty / lv if op == "div" else qty * lv


def _join_columns(df, cols):
    """Vectorised ' | '-join of the string form of *cols* (fast; avoids row-wise agg)."""
    if not cols:
        return pd.Series(["all"] * len(df), index=df.index)
    series = df[cols[0]].astype(str)
    for c in cols[1:]:
        series = series + " | " + df[c].astype(str)
    return series


def normalize_fit_dataframe(df, time_col, value_col, observable, arm_cols, moi_col=None):
    """Normalise an uploaded dataframe to the canonical long format.

    Returns ``(long_df[time, arm, observable, value], conditions{arm: {"moi": float}})``.
    ``observable`` is either a registry key (fixed for all rows) or a column name.
    ``arm`` is the ``" | "``-joined combination of *arm_cols* (e.g. ``"MXP1001 | 0.1"``).
    """
    arm = _join_columns(df, arm_cols)  # computed once, reused for conditions
    out = pd.DataFrame({
        "time": pd.to_numeric(df[time_col], errors="coerce"),
        "value": pd.to_numeric(df[value_col], errors="coerce"),
        "arm": arm.values,
        "observable": (observable if observable in OBSERVABLES
                       else df[observable].astype(str).str.lower().values),
    })
    out = out.dropna(subset=["time", "value"]).reset_index(drop=True)

    conditions = {a: {"moi": 0.0} for a in out["arm"].unique()}
    if moi_col and moi_col in df.columns:
        moi = pd.to_numeric(df[moi_col], errors="coerce")
        first = moi.groupby(arm).first()
        for a, v in first.items():
            if pd.notna(v):
                conditions[a] = {"moi": float(v)}
    return out, conditions


def apply_row_filters(df, filters):
    """Keep rows whose column values are in the allowed set for every filtered column.

    ``filters`` maps ``column -> iterable of allowed values`` (compared as strings).
    An empty/absent allow-list for a column means "no restriction" on that column.
    """
    mask = pd.Series(True, index=df.index)
    for col, allowed in (filters or {}).items():
        if allowed and col in df.columns:
            mask &= df[col].astype(str).isin({str(v) for v in allowed})
    return df[mask]


def aggregate_observations(long_df, stat="raw", band=None):
    """Aggregate replicate observations per (arm, observable, time).

    Parameters
    ----------
    stat : "raw" | "mean" | "median"
        ``"raw"`` returns every point unchanged; otherwise replicates are collapsed
        to their mean or median per (arm, observable, time).
    band : (lo_pct, hi_pct) or None
        Percentile band (e.g. ``(25, 75)``) to compute alongside the central value.

    Returns a DataFrame with columns ``arm, observable, time, value, lo, hi`` where
    ``lo``/``hi`` are NaN when no band is requested (or stat="raw").
    """
    if stat == "raw":
        out = long_df[["arm", "observable", "time", "value"]].copy()
        out["lo"] = np.nan
        out["hi"] = np.nan
        return out
    grp = long_df.groupby(["arm", "observable", "time"])["value"]
    central = (grp.mean() if stat == "mean" else grp.median()).rename("value").reset_index()
    if band:
        lo = grp.quantile(band[0] / 100.0).rename("lo").reset_index()
        hi = grp.quantile(band[1] / 100.0).rename("hi").reset_index()
        central = central.merge(lo, on=["arm", "observable", "time"]).merge(hi, on=["arm", "observable", "time"])
    else:
        central["lo"] = np.nan
        central["hi"] = np.nan
    return central


# ── Manual parameter tuning (Phase B) ───────────────────────────────────────────
# The tuning panel edits the model's *actual* parameter values (like the Interactive
# Simulator), so each knob names the GUI-dict key it edits per entity. Adsorption is
# handled separately in the UI because its storage is builder-mode specific (Direct
# and Custom-Strains keep it in the pairwise ``ads_{strain}_{phage}`` session keys;
# Binary-Genotypes keeps it on the phage dict as ``adsorption_s``).
STRAIN_TUNABLES = [
    {"key": "growth_rate",                "label": "Growth rate (1/h)",   "fmt": "%g",   "default": 1.2},
    {"key": "bacteria_to_resource_ratio", "label": "Bacteria/resource",   "fmt": "%.2e", "default": 1e9},
    {"key": "death_rate_B",               "label": "Natural death (1/h)", "fmt": "%g",   "default": 0.0},
    {"key": "initial_B",                  "label": "Initial density B₀",  "fmt": "%.3e", "default": 1e7},
]
# Shown only for a strain with dormancy enabled.
STRAIN_DORMANCY_TUNABLES = [
    {"key": "dormancy_rate",           "label": "Dormancy rate (1/h)",     "fmt": "%g", "default": 0.2},
    {"key": "resuscitation_rate",      "label": "Resuscitation (1/h)",     "fmt": "%g", "default": 0.1},
    {"key": "dormancy_diffusion_rate", "label": "Depth diffusion (1/h)",   "fmt": "%g", "default": 0.05},
    {"key": "death_rate_D",            "label": "Dormant death (1/h)",     "fmt": "%g", "default": 0.0},
]
PHAGE_TUNABLES = [
    {"key": "burst_sizes",       "label": "Burst size",         "fmt": "%g",   "default": 50.0},
    {"key": "latent_periods",    "label": "Latent period (h)",  "fmt": "%g",   "default": 0.5},
    {"key": "phage_decay_rates", "label": "Phage decay (1/h)",  "fmt": "%g",   "default": 0.1},
    {"key": "phage_decay_Km",    "label": "Decay Km",           "fmt": "%.1e", "default": 0.0},
    {"key": "attenuation_rate",  "label": "Dormant attenuation","fmt": "%g",   "default": 0.0},
]
# adsorption_s is the phage-dict key used in Binary-Genotypes mode.
ADSORPTION_PHAGE_KEYS = ("adsorption_s", "adsorption_rates")


def entity_param_key(entity, candidate_keys):
    """Return the key an entity actually stores a parameter under (first match).

    Falls back to the first candidate so a fresh number-input seeds/writes a valid
    key even on an entity that doesn't have it yet.
    """
    for k in candidate_keys:
        if k in entity:
            return k
    return candidate_keys[0]


def fit_residual(model_time, model_signal, data_time, data_value, log_scale):
    """RMSE between the model (interpolated to the data times) and observations.

    On ``log_scale`` the residual is taken on log10 (with a floor), appropriate for
    CFU/PFU spanning orders of magnitude.
    """
    pred = np.interp(np.asarray(data_time, float), np.asarray(model_time, float),
                     np.asarray(model_signal, float))
    obs = np.asarray(data_value, float)
    if log_scale:
        pred = np.log10(np.maximum(pred, 1e-30))
        obs = np.log10(np.maximum(obs, 1e-30))
    diff = pred - obs
    diff = diff[np.isfinite(diff)]
    return float(np.sqrt(np.mean(diff ** 2))) if len(diff) else float("nan")
