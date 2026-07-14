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


def predicted_observable(result, obs_key, link_value=None):
    """Predicted measured signal from a SimulationResult for the given observable.

    ``result`` only needs a ``sum_prefixes(*prefixes)`` method, so this works for any
    pbisim ``SimulationResult``.
    """
    spec = OBSERVABLES[obs_key]
    qty = result.sum_prefixes(*spec["prefixes"])
    link = spec.get("link")
    if link is None:
        return qty
    _, op, default = link
    lv = float(link_value) if link_value is not None else float(default)
    return qty / lv if op == "div" else qty * lv


def normalize_fit_dataframe(df, time_col, value_col, observable, arm_cols, moi_col=None):
    """Normalise an uploaded dataframe to the canonical long format.

    Returns ``(long_df[time, arm, observable, value], conditions{arm: {"moi": float}})``.
    ``observable`` is either a registry key (fixed for all rows) or a column name.
    ``arm`` is the ``" | "``-joined combination of *arm_cols* (e.g. ``"MXP1001 | 0.1"``).
    """
    out = pd.DataFrame()
    out["time"] = pd.to_numeric(df[time_col], errors="coerce")
    out["value"] = pd.to_numeric(df[value_col], errors="coerce")
    if arm_cols:
        out["arm"] = df[arm_cols].astype(str).agg(" | ".join, axis=1)
    else:
        out["arm"] = "all"
    if observable in OBSERVABLES:
        out["observable"] = observable
    else:
        out["observable"] = df[observable].astype(str).str.lower()
    out = out.dropna(subset=["time", "value"]).reset_index(drop=True)

    conditions = {arm: {"moi": 0.0} for arm in out["arm"].unique()}
    if moi_col and moi_col in df.columns:
        arm_series = df[arm_cols].astype(str).agg(" | ".join, axis=1) if arm_cols else pd.Series(["all"] * len(df))
        moi = pd.to_numeric(df[moi_col], errors="coerce")
        for arm, g in moi.groupby(arm_series):
            vals = g.dropna()
            if len(vals):
                conditions[arm] = {"moi": float(vals.iloc[0])}
    return out, conditions


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
