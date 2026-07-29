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
# `log` is the PLOT-axis default (log for CFU/PFU, linear for OD). `floor_log10` is
# the detection floor used by the OBJECTIVE, which — matching pbisim-fit's NLS — is
# always computed in log10 space with these per-observable floors (values are clipped
# to 10**floor before log10). pbisim-fit: floor_log10=1.0 (CFU/PFU), -2.5 (OD).
# CFU counts only CULTURABLE cells — active (B) + dormant (D). Infected (I) and
# hibernating (H) cells don't form colonies on a plate, so they are excluded from CFU.
# OD (turbidity) still reflects all physically-present cells (B,D,I,H) + debris. The
# `prefixes` here are the DEFAULT compartment set; the Calibration page can override the
# set per observable (the "observation model"), and the same set is threaded to the fit.
OBSERVABLES = {
    "cfu": {"label": "CFU/mL",               "prefixes": ("B", "D"),           "log": True,  "link": None,                        "floor_log10": 1.0},
    "pfu": {"label": "PFU/mL",               "prefixes": ("P",),               "log": True,  "link": None,                        "floor_log10": 1.0},
    "od":  {"label": "Optical density (OD)", "prefixes": ("B", "D", "I", "H"), "log": False, "link": ("od_to_cfu", "div", 1e9),   "floor_log10": -2.5},
    "lum": {"label": "Luminescence (RLU)",   "prefixes": ("B",),               "log": True,  "link": ("rlu_per_cell", "mul", 1.0), "floor_log10": 1.0},
}

# Bacterial compartments the user can toggle in the observation model (debris is folded
# into OD via the model's get_od()). Order = display order.
OBS_COMPARTMENTS = ("B", "D", "I", "H")


def obs_prefixes(obs_key, overrides=None):
    """The compartment prefixes for an observable, honoring a per-observable override
    (the user-defined observation model). Falls back to the registry default; returns
    ``()`` for an observable not in the registry (a custom label in the data)."""
    if overrides and obs_key in overrides and overrides[obs_key]:
        return tuple(overrides[obs_key])
    spec = OBSERVABLES.get(obs_key)
    return tuple(spec["prefixes"]) if spec else ()


def predicted_observable(result, obs_key, link_value=None, use_model_od=False, prefixes=None):
    """Predicted measured signal from a SimulationResult for the given observable.

    ``result`` only needs a ``sum_prefixes(*prefixes)`` method, so this works for any
    pbisim ``SimulationResult``. ``prefixes`` overrides the compartment set (the user's
    observation model); when None the registry default is used.

    When ``use_model_od`` is set and the observable is OD, the debris-inclusive OD
    from the model (``result.get_od()``, which folds in lysed-cell debris and uses the
    model's own ``od_to_cfu_conversion_factor``) is used instead of the simple
    biomass/link scaling — so an enabled OD/debris module propagates into the overlay.
    """
    if obs_key == "od" and use_model_od and hasattr(result, "get_od"):
        return result.get_od()
    spec = OBSERVABLES[obs_key]
    px = tuple(prefixes) if prefixes else spec["prefixes"]
    qty = result.sum_prefixes(*px)
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


# Default dose unit per target (used when the dataset has no unit column). The set of
# valid dose targets is sourced from pbisim-fit (below) so it can't drift.
_DEFAULT_DOSE_UNIT = {"phage": "pfu", "bacteria": "cfu", "antibiotic": "mg", "nutrient": "mg"}


def dose_targets():
    """The valid dose-target compartments — pbisim-fit's `DOSE_TARGETS` (single source
    of truth), with a static fallback if pbisim-fit isn't importable."""
    try:
        from pbisim_fit import DOSE_TARGETS as _t
        return tuple(_t)
    except Exception:  # pragma: no cover - pbisim-fit is a declared dep
        return ("phage", "bacteria", "antibiotic", "nutrient")


# Back-compat module constant (kept in sync with pbisim-fit at import time).
DOSE_TARGETS = dose_targets()


def parse_dose_rows(df, arm_cols, evid_col, observable_col, amount_col, time_col, unit_col=None):
    """Per-arm dose records from NONMEM/Monolix dose rows (``dose_event``/EVID == 1).

    The obs/dose split and the canonical event model are delegated to pbisim-fit's
    ``EventTable`` (single source of truth — no hand-rolled EVID handling); the app only
    supplies the column→canonical mapping. The dose's target compartment is the value of
    the *observable* column on that row (bacteria / phage / antibiotic / nutrient — it
    double-duties as CMT for dose rows). Returns ``{arm: [{"time","target","amount",
    "unit"}]}``; ``{}`` when no EVID or amount column is mapped (observation-only)."""
    if (not evid_col or evid_col not in df.columns
            or not amount_col or amount_col not in df.columns):
        return {}
    from pbisim_fit import EventTable
    _targets = dose_targets()
    canon = pd.DataFrame({
        "arm": _join_columns(df, list(arm_cols)).values,
        "time": pd.to_numeric(df[time_col], errors="coerce").values,
        "observable": df[observable_col].astype(str).str.strip().str.lower().values,
        "amount": pd.to_numeric(df[amount_col], errors="coerce").values,
        "dose_event": pd.to_numeric(df[evid_col], errors="coerce").fillna(0).values,
        "unit": (df[unit_col].astype(str).values if unit_col and unit_col in df.columns
                 else [""] * len(df)),
    })
    et = EventTable(events=canon)   # canonicalises + separates obs/dose rows
    out = {}
    for _, r in et.doses.iterrows():
        tgt = str(r["observable"]).strip().lower()
        amt = pd.to_numeric(r["amount"], errors="coerce")
        if tgt not in _targets or not np.isfinite(amt):
            continue
        unit = str(r["unit"]).strip().lower() or _DEFAULT_DOSE_UNIT.get(tgt, "")
        t = pd.to_numeric(r["time"], errors="coerce")
        out.setdefault(str(r["arm"]), []).append({
            "time": float(t) if np.isfinite(t) else 0.0,
            "target": tgt, "amount": float(amt), "unit": unit})
    return out


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
# NOTE: the hand-curated *_TUNABLES lists + entity_param_key that used to drive the
# Calibration manual-tuning panel were removed (2026-07-25). That panel now renders the
# SAME builder as the Interactive Simulator via pbisim_app.views.simulator.
# render_model_builder(), so every parameter and builder mode is covered automatically
# and the two can never drift.


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


def residual_vector_log10(model_time, model_signal, data_time, data_value, floor_log10):
    """log10-space residuals (model interpolated to the data times), with a detection
    floor — the exact form pbisim-fit's NLS minimises. Both prediction and observation
    are clipped to ``10**floor_log10`` before log10, so all observables are compared on
    a common, order-of-magnitude scale. Returns the finite residual array (poolable
    across observables/arms to form the joint objective)."""
    pred = np.interp(np.asarray(data_time, float), np.asarray(model_time, float),
                     np.asarray(model_signal, float))
    obs = np.asarray(data_value, float)
    floor = 10.0 ** float(floor_log10)
    pred = np.log10(np.maximum(pred, floor))
    obs = np.log10(np.maximum(obs, floor))
    diff = pred - obs
    return diff[np.isfinite(diff)]


def config_param_snapshot(config):
    """JSON-able snapshot of the current model's fittable biological parameters — the
    warm-start reference for pbisim-fit (its FreeParamSpec path mapping is a later
    phase; this captures the current *values*)."""
    def _j(v):
        if v is None:
            return None
        a = np.asarray(v)
        return a.tolist() if a.ndim else float(a)
    fields = ["growth_rates", "bacteria_to_resource_ratio", "monod_constant", "carrying_capacity",
              "adsorption_rates", "adsorption_rates_dormant", "burst_sizes", "latent_periods",
              "latent_periods_dormant", "phage_decay_rates", "death_rate_B", "death_rate_D",
              "dormancy_rate", "resuscitation_rate", "dormancy_diffusion_rate", "mutation_rates",
              "od_to_cfu_conversion_factor"]
    snap = {}
    for f in fields:
        v = getattr(config, f, None)
        if v is not None:
            snap[f] = _j(v)
    return snap


def build_fit_spec(agg, sel_arms, sel_obs, arm_cond, *, od_to_cfu=None,
                   model_params=None, notes=None, dose_unit="moi"):
    """Assemble a pbisim-fit *fit specification* from the calibration state.

    Returns a JSON-serializable dict whose ``dataset`` maps onto
    ``pbisim_fit.ExperimentalDataset.from_dict`` (arms carry their observable arrays
    aligned to a shared ``time`` grid — ``None`` marks unmeasured points, i.e. NaN —
    plus the per-arm growth-phase/inoculum conditions and MOI dose), and whose
    ``nls_cfg`` maps onto ``pbisim_fit.NLSConfig`` (obs_keys + detection floors).
    ``warm_start`` holds the current parameter values.
    """
    sel_obs = list(sel_obs)
    m = agg[agg["arm"].isin(sel_arms) & agg["observable"].isin(sel_obs)]
    times = sorted(float(t) for t in m["time"].unique())
    tindex = {t: i for i, t in enumerate(times)}

    arms_out = {}
    for arm in sel_arms:
        cond = arm_cond.get(arm, {})
        entry, has_data = {}, False
        for ok in sel_obs:
            col = [None] * len(times)
            d = m[(m["arm"] == arm) & (m["observable"] == ok)]
            for _, r in d.iterrows():
                col[tindex[float(r["time"])]] = float(r["value"])
                has_data = True
            if any(v is not None for v in col):
                entry[ok] = col
        if not has_data:
            continue
        _dose = float(cond.get("moi", 0.0) or 0.0)   # value in `dose_unit`
        if _dose > 0:
            _amt = f"{_dose:g}" if dose_unit == "pfu" else f"MOI:{_dose:g}"
            entry["doses"] = [{"time": 0.0, "amount": _amt, "unit": dose_unit, "target": "phage"}]
        pr = float(cond.get("prerun", 0.0) or 0.0)
        if pr > 0:
            entry["pretreatment_h"] = pr           # stationary pre-grow (TreatmentRecord)
        b0 = cond.get("b0")
        if b0:
            entry["pretreatment_inoculum"] = float(b0)
        arms_out[arm] = entry

    data_type = "od_assay" if set(sel_obs) == {"od"} else "time_kill"
    metadata = {"od_to_cfu_cv": 0.30}
    if od_to_cfu is not None:
        metadata["od_to_cfu"] = float(od_to_cfu)
    if notes:
        metadata["notes"] = notes

    nls_cfg = {"obs_keys": sel_obs}
    _floor_key = {"cfu": "floor_log10", "od": "od_floor_log10",
                  "pfu": "pfu_floor_log10", "phage_blood": "phage_blood_floor_log10"}
    for ok in sel_obs:
        fl = OBSERVABLES.get(ok, {}).get("floor_log10")
        if fl is not None and ok in _floor_key:
            nls_cfg[_floor_key[ok]] = fl

    spec = {
        "schema_version": 1,
        "generated_by": "pbisim-app calibration",
        "dataset": {"data_type": data_type, "time": times, "arms": arms_out, "metadata": metadata},
        "nls_cfg": nls_cfg,
    }
    if model_params:
        spec["warm_start"] = model_params
    return spec
