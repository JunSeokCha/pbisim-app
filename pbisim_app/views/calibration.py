"""Rendered by app.py when this page is selected."""
import re
import threading
import time as _time

from pbisim_app.common import *  # noqa: F401,F403


def _pf(s):
    """Parse a possibly-scientific-notation string to float. Blank/invalid → None.
    (data_editor NumberColumn won't accept typed `1e9`; TextColumn + this does.)"""
    if s is None:
        return None
    t = str(s).strip()
    if t == "" or t.lower() in ("nan", "none"):
        return None
    try:
        return float(t)
    except (TypeError, ValueError):
        return None


def _fit_worker(holder):
    """Background NLS fit. Writes results into the plain ``holder`` dict only — it
    never touches ``st`` / ``session_state`` (it runs off the ScriptRunContext), so
    the Streamlit UI stays responsive and the fit can be abandoned via a Stop button."""
    try:
        from pbisim_app import nls_fit as _nls
        fp = _nls.run_nls_fit_v2(
            holder["cfg"], holder["targets"], holder["thetas"], holder["mappings"],
            holder["ds"], holder["obs"], od_to_cfu=holder["od_link"],
            n_restarts=holder["restarts"], max_nfev=holder["maxnfev"])
        holder["fp"] = fp
        holder["status"] = "done"
    except Exception as e:  # noqa: BLE001 — surface any failure to the UI
        holder["error"] = f"{type(e).__name__}: {e}"
        holder["status"] = "error"


def _apply_map_to_state(map_dict):
    """Write pbisim-fit MAP path→value results back into the live model dicts /
    session keys, so a fit updates the Interactive-Simulator model in place.

    Handled paths: growth_rates[i], bacteria_to_resource_ratio[i], death_rate_B[i]
    → int_strains[i]; burst_sizes[i,j], latent_periods[i,j], phage_decay_rates[j]
    → int_phages[j]; adsorption_rates[i,j] → ads_{i}_{j} key (+ phage adsorption_s);
    monod_constant → int_monod_constant.
    """
    strains = st.session_state.get("int_strains", [])
    phages = st.session_state.get("int_phages", [])
    for path, val in map_dict.items():
        val = float(val)
        m1 = re.fullmatch(r"([a-zA-Z_]+)\[(\d+)\]", path)
        m2 = re.fullmatch(r"([a-zA-Z_]+)\[(\d+),(\d+)\]", path)
        if path == "monod_constant":
            st.session_state["int_monod_constant"] = val
        elif m1:
            name, i = m1.group(1), int(m1.group(2))
            if name == "growth_rates" and i < len(strains):
                strains[i]["growth_rate"] = val
            elif name in ("bacteria_to_resource_ratio", "death_rate_B") and i < len(strains):
                strains[i][name] = val
            elif name == "phage_decay_rates" and i < len(phages):
                phages[i]["phage_decay_rates"] = val
        elif m2:
            name, i, j = m2.group(1), int(m2.group(2)), int(m2.group(3))
            if name == "adsorption_rates":
                st.session_state[f"ads_{i}_{j}"] = val
                if j < len(phages):
                    phages[j]["adsorption_rates"] = val
                    phages[j]["adsorption_s"] = val
            elif name in ("burst_sizes", "latent_periods") and j < len(phages):
                phages[j][name] = val
    st.session_state["int_strains"] = strains
    st.session_state["int_phages"] = phages


def _apply_config_to_state(cfg):
    """Write a fitted pbisim ModelConfig's resolved biological parameters back into the
    live builder dicts / session keys. Robust for reparameterized fits (reads final
    values off the config rather than parsing theta names)."""
    strains = st.session_state.get("int_strains", [])
    phages = st.session_state.get("int_phages", [])

    def _a1(x):
        return np.atleast_1d(np.asarray(x, dtype=float)) if x is not None else np.array([])

    _gr, _rr = _a1(getattr(cfg, "growth_rates", None)), _a1(getattr(cfg, "bacteria_to_resource_ratio", None))
    _db = _a1(getattr(cfg, "death_rate_B", None))
    for _i, _s in enumerate(strains):
        if _i < len(_gr): _s["growth_rate"] = float(_gr[_i])
        if _i < len(_rr): _s["bacteria_to_resource_ratio"] = float(_rr[_i])
        if _i < len(_db): _s["death_rate_B"] = float(_db[_i])
    _bs = np.atleast_2d(np.asarray(getattr(cfg, "burst_sizes", [[]]), dtype=float))
    _lp = np.atleast_2d(np.asarray(getattr(cfg, "latent_periods", [[]]), dtype=float))
    _ads = np.atleast_2d(np.asarray(getattr(cfg, "adsorption_rates", [[]]), dtype=float))
    _pdr = _a1(getattr(cfg, "phage_decay_rates", None))
    for _j, _p in enumerate(phages):
        if _bs.shape[1] > _j: _p["burst_sizes"] = float(_bs[0, _j])
        if _lp.shape[1] > _j: _p["latent_periods"] = float(_lp[0, _j])
        if _j < len(_pdr): _p["phage_decay_rates"] = float(_pdr[_j])
        if _ads.shape[1] > _j:
            _p["adsorption_rates"] = float(_ads[0, _j]); _p["adsorption_s"] = float(_ads[0, _j])
    for _i in range(_ads.shape[0]):
        for _j in range(_ads.shape[1]):
            st.session_state[f"ads_{_i}_{_j}"] = float(_ads[_i, _j])
    if getattr(cfg, "monod_constant", None) is not None:
        st.session_state["int_monod_constant"] = float(cfg.monod_constant)
    # Estimated initial conditions (fit_initial_cfu / fit_initial_pfu) → the model's
    # inoculum, so a re-run / the Interactive Simulator matches the fit.
    _ic = getattr(cfg, "fit_initial_cfu", None)
    if _ic is not None and np.isfinite(_ic) and _ic > 0 and strains:
        _tot = sum(float(s.get("initial_B", 0.0)) for s in strains)
        for _s in strains:
            _s["initial_B"] = (float(_s.get("initial_B", 0.0)) * float(_ic) / _tot
                               if _tot > 0 else float(_ic) / len(strains))
    _ip = getattr(cfg, "fit_initial_pfu", None)
    if _ip is not None and np.isfinite(_ip) and _ip > 0 and phages:
        phages[0]["initial_P"] = float(_ip)
    st.session_state["int_strains"] = strains
    st.session_state["int_phages"] = phages


def _compute_overlay(config, iB, iP, iS, mk, ctx):
    """Simulate `config` once per arm and project every selected observable into a
    small-multiples overlay-vs-data result dict (also computes per-obs RMSE/R² and the
    pooled log10 combined objective J). Shared by the manual overlay button and the
    post-fit fitted-curve overlay, so both draw the same way. `ctx` bundles the arm/
    observable selection, aggregated data, and plotting options."""
    _B0 = float(np.sum(iB))
    _panels, _metrics = {}, []
    for _arm in ctx["sel_arms"]:
        _cond = ctx["arm_cond"].get(_arm, {})
        _arm_b0 = float(_cond.get("b0", _B0)) or _B0
        _arm_prerun = float(_cond.get("prerun", 0.0) or 0.0)
        _moi = float(_cond.get("moi", ctx["conds"].get(_arm, {}).get("moi", 0.0)))
        _armB = iB * (_arm_b0 / _B0) if _B0 > 0 else iB
        _armP = np.zeros(len(iP))
        if len(iP):
            # dose value is absolute PFU/mL (pfu) or a multiple of B₀ (moi)
            _armP[0] = _moi if ctx.get("dose_unit") == "pfu" else _moi * _arm_b0
        _mk_arm = dict(mk)
        _iS_arm = iS
        if _arm_prerun > 0:
            _ic = stationary_phase_ic(config, t_prerun=_arm_prerun, B0=_armB)
            _armB = _ic.B
            _iS_arm = max(float(_ic.S), 0.0)
            if _ic.D is not None:
                _mk_arm["initial_D"] = _ic.D
            if _ic.Imm is not None:
                _mk_arm["initial_Imm"] = _ic.Imm
            _carry_prerun_debris(_ic, _mk_arm)
        _m = PBIModel(config, initial_B=_armB, initial_P=_armP, initial_S=_iS_arm, **_mk_arm)
        _r = solve_ode(_m, t_end=ctx["t_end"], dt=0.25, method=ctx["method"],
                       extinction_threshold=ctx["thr"])
        for _ok in ctx["sel_obs"]:
            _sp = OBSERVABLES.get(_ok, {"log": True, "link": None, "label": _ok})
            _umo = (_ok == "od" and ctx["debris_on"])
            _pred = predicted_observable(_r, _ok, ctx["link_vals"].get(_ok), use_model_od=_umo)
            _d = ctx["agg"][(ctx["agg"]["arm"] == _arm) & (ctx["agg"]["observable"] == _ok)].sort_values("time")
            if not len(_d):
                continue
            _has_band = ctx["band"] is not None and _d["lo"].notna().any()
            _pan = _panels.setdefault(_ok, {"obs": _ok, "label": _sp.get("label", _ok),
                                            "log": bool(_sp.get("log")), "series": []})
            _pan["series"].append({
                "label": _arm, "time": np.asarray(_r.time), "pred": np.asarray(_pred),
                "obs_time": _d["time"].to_numpy(), "obs_value": _d["value"].to_numpy(),
                "obs_lo": _d["lo"].to_numpy() if _has_band else None,
                "obs_hi": _d["hi"].to_numpy() if _has_band else None,
                "is_raw": ctx["stat_key"] == "raw",
            })
            _resid = residual_vector_log10(_r.time, _pred, _d["time"].values,
                                           _d["value"].values, _sp.get("floor_log10", 1.0))
            _metrics.append({"observable": _sp.get("label", _ok), "group": _arm,
                             "B₀": _arm_b0, "pre-run (h)": _arm_prerun,
                             ctx.get("dose_label", "MOI"): _moi,
                             "n_points": len(_d),
                             "RMSE (log₁₀)": float(np.sqrt(np.mean(_resid ** 2))) if _resid.size else float("nan")})

    _panel_list, _all_resid = [], []
    for _ok, _pan in _panels.items():
        _floor = 10.0 ** float(OBSERVABLES.get(_ok, {}).get("floor_log10", 1.0))
        _oa, _pa = [], []
        for _s in _pan["series"]:
            _pt = np.interp(_s["obs_time"], _s["time"], _s["pred"])
            _ov = np.asarray(_s["obs_value"], dtype=float)
            _oa.append(np.log10(np.maximum(_ov, _floor)))
            _pa.append(np.log10(np.maximum(_pt, _floor)))
        _oa = np.concatenate(_oa) if _oa else np.array([])
        _pa = np.concatenate(_pa) if _pa else np.array([])
        _msk = np.isfinite(_oa) & np.isfinite(_pa)
        _oa, _pa = _oa[_msk], _pa[_msk]
        _pan["rmse"] = float(np.sqrt(np.mean((_oa - _pa) ** 2))) if _oa.size else float("nan")
        _sst = float(np.sum((_oa - _oa.mean()) ** 2)) if _oa.size else 0.0
        _pan["r2"] = (1.0 - float(np.sum((_oa - _pa) ** 2)) / _sst) if _sst > 0 else float("nan")
        _pan["n"] = int(_oa.size)
        _panel_list.append(_pan)
        if _oa.size:
            _all_resid.append(_oa - _pa)

    _all = np.concatenate(_all_resid) if _all_resid else np.array([])
    _stat_label = ctx["stat"] if ctx["stat_key"] != "raw" else "raw points"
    return {
        "panels": _panel_list, "metrics": _metrics,
        "combined": float(np.sqrt(np.mean(_all ** 2))) if _all.size else float("nan"),
        "stat_label": _stat_label,
        "title": ctx.get("title") or (f"Model vs observations ({_stat_label}"
                 + (f" + {ctx['band_choice']} band)" if ctx["band"] else ")")),
    }


def render():
    theme_mode = st.session_state.get("theme_mode", "Light")
    st.title("Calibration — data overlay")
    st.caption(
        "Upload experimental data and overlay the **current model's** prediction (configured in "
        "the Interactive Simulator) on the observations. Tune parameters there to match; a "
        "manual-tuning panel and the pbisim-fit hand-off come next."
    )

    # Re-seed the Calibration widgets from a persistent config BEFORE they render.
    # Streamlit drops a widget's key from session_state whenever the widget isn't
    # rendered on a rerun, so navigating to the Simulator (to change the model) and
    # back would otherwise reset the filters / grouping / statistics. `fit_config`
    # is a plain (non-widget) key, so it survives.
    # Buttons and the file-uploader can't be re-seeded via session_state, so they
    # are never persisted; everything else (filters/grouping/statistics/overlay
    # selections) is.
    # Buttons + the file-uploader must never be shadowed into fit_config: re-seeding a
    # button's value pre-sets it, which makes the later st.button() raise. (Text/number
    # widgets are fine to persist.)
    _FIT_NOPERSIST = {"fit_csv", "fit_config", "fit_dataset", "fit_overlay", "fit_clear",
                      "fit_load", "fit_save_scenario", "fit_run_nls", "fit_apply_map",
                      "fit_model_sel", "fit_job", "fit_stop", "fit_tbl_reset",
                      "fit_spec_from_tables", "fit_spec_to_tables", "fit_spec_rev",
                      "fit_share_go"}
    _fcfg = st.session_state.setdefault("fit_config", {})
    for _wk, _wv in list(_fcfg.items()):
        if _wk in _FIT_NOPERSIST:
            _fcfg.pop(_wk, None)  # scrub any stale non-persistable key
            continue
        if _wk not in st.session_state:
            try:
                st.session_state[_wk] = _wv
            except Exception:
                pass  # widget type refuses assignment (e.g. a button) — skip it

    # ── 1. Upload + column mapping ───────────────────────────────────────────
    st.markdown("### 1 · Upload data")
    _up = st.file_uploader("Experimental data (CSV)", type=["csv"], key="fit_csv")
    if _up is not None:
        try:
            _raw = read_uploaded_csv(_up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            _raw = None
        if _raw is not None:
            st.dataframe(_raw.head(8), width="stretch")
            _cols = list(_raw.columns)
            _low = [c.lower() for c in _cols]

            def _guess(cands, default=0):
                for c in cands:
                    if c in _low:
                        return _low.index(c)
                return default

            _canonical = all(k in _low for k in ("time", "arm", "observable", "value"))
            if _canonical:
                st.success("Detected pbisim-fit long format (time, arm, observable, value).")
            with st.expander("Column mapping", expanded=not _canonical):
                _tc = st.selectbox("Time column", _cols, index=_guess(["time"]))
                _vc = st.selectbox("Value (measurement) column", _cols, index=_guess(["value", "dv"]))
                _obs_from_col = st.checkbox("Observable is in a column", value=("observable" in _low))
                if _obs_from_col:
                    _obs = st.selectbox("Observable column", _cols, index=_guess(["observable"]))
                else:
                    _obs = st.selectbox("Observable type (fixed for all rows)", list(OBSERVABLES),
                                        format_func=lambda k: OBSERVABLES[k]["label"])
                _default_arms = [c for c in _cols if c.lower() in ("phage", "moi", "arm", "experi", "experi_num")]
                _ac = st.multiselect("Arm-defining column(s)", _cols, default=_default_arms or ([_cols[0]] if _cols else []))
                # Auto-detect a dose column by substring (dose/pfu/moi/titre) so e.g.
                # `dose_phage_pfu` is picked up without manual selection.
                _dose_idx = next((i for i, c in enumerate(_low)
                                  if any(k in c for k in ("dose", "pfu", "moi", "titre", "titer"))), None)
                _mc = st.selectbox("Phage-dose column (optional — drives the simulated dose per arm)",
                                   ["(none)"] + _cols,
                                   index=(1 + _dose_idx) if _dose_idx is not None else 0)
                _mc = None if _mc == "(none)" else _mc
                _dunit_lbl = st.radio(
                    "Dose unit", ["MOI (× B₀)", "PFU/mL (absolute)"],
                    index=(1 if (_mc and "pfu" in _mc.lower()) else 0), horizontal=True,
                    help="How to read the dose column: MOI multiplies each arm's B₀; "
                         "PFU/mL is an absolute phage titre (e.g. a column like dose_phage_pfu).")
            if st.button("Load dataset", key="fit_load", width="stretch"):
                st.session_state.fit_dataset = {
                    "raw": _raw, "time": _tc, "value": _vc, "observable": _obs,
                    "arm_cols": _ac, "moi": _mc,
                    "dose_unit": ("pfu" if str(_dunit_lbl).startswith("PFU") else "moi"),
                }
                st.success(f"Loaded {len(_raw)} rows. Configure grouping / filters / statistics below.")
                st.rerun()

    # ── 2. Filter · group · statistics · overlay ─────────────────────────────
    _ds = st.session_state.get("fit_dataset")
    if not _ds:
        st.info("Upload a dataset above to begin.")
    else:
        _raw = _ds["raw"]
        _cols = list(_raw.columns)
        _tc, _vc, _obs, _mc = _ds["time"], _ds["value"], _ds["observable"], _ds["moi"]
        _dose_unit = _ds.get("dose_unit", "moi")
        _dose_lbl = "PFU/mL" if _dose_unit == "pfu" else "MOI"

        # -- Filters --------------------------------------------------------
        st.markdown("### 2 · Filter rows")
        _filter_cols = st.multiselect(
            "Filter on column(s) (leave a value list empty = include all)", _cols,
            default=[], key="fit_filter_cols",
        )
        _filters = {}
        for _fc in _filter_cols:
            _uniques = sorted(_raw[_fc].dropna().astype(str).unique().tolist())
            _filters[_fc] = st.multiselect(f"Include {_fc} =", _uniques, default=[], key=f"fit_filter_{_fc}")

        # -- Grouping + statistic -------------------------------------------
        st.markdown("### 3 · Grouping & statistics")
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            _group_cols = st.multiselect("Grouping variables (define the curves/arms)", _cols,
                                         default=_ds["arm_cols"], key="fit_group_cols")
        with gc2:
            _stat = st.selectbox("Statistic over replicates", ["Raw points", "Mean", "Median"], key="fit_stat")
        with gc3:
            _band_choice = st.selectbox("Percentile band", ["None", "10–90", "25–75", "5–95"],
                                        index=0, disabled=(_stat == "Raw points"), key="fit_band")
        _stat_key = {"Raw points": "raw", "Mean": "mean", "Median": "median"}[_stat]
        _band = None if (_band_choice == "None" or _stat_key == "raw") else tuple(int(x) for x in _band_choice.split("–"))

        # Filter → normalise → aggregate (cached; recomputes only when inputs change)
        try:
            _filters_key = tuple((c, tuple(v)) for c, v in _filters.items())
            _filtered, _long, _conds, _agg = calibration_processed(
                _raw, _filters_key, _tc, _vc, _obs, tuple(_group_cols), _mc, _stat_key, _band)
        except Exception as e:
            st.error(f"Could not build the grouped dataset: {e}")
            _filtered, _long, _agg = _raw, None, None
        st.caption(f"{len(_filtered)} / {len(_raw)} rows after filtering.")

        if _long is not None and len(_long):
            _arms = sorted(_long["arm"].unique())
            _obs_keys = sorted(_long["observable"].unique())

            st.markdown("### 4 · Overlay")
            st.caption(f"{len(_arms)} group(s) · {len(_obs_keys)} observable(s) in the data · "
                       "pick what to overlay against the current model. Each arm is simulated once; "
                       "every observable is projected from that one trajectory.")
            oc1, oc2 = st.columns(2)
            with oc1:
                _sel_arms = st.multiselect("Groups to overlay", _arms, default=_arms[:min(4, len(_arms))], key="fit_arms")
            with oc2:
                _sel_obs = st.multiselect("Observables", _obs_keys, default=_obs_keys,
                                          format_func=lambda k: OBSERVABLES.get(k, {}).get("label", k), key="fit_obs_sel")

            # Per-observable link parameters (OD = biomass / od_to_cfu; luminescence =
            # active biomass × rlu_per_cell). One input per selected observable that needs
            # one; OD instead uses the debris module's get_od() when it is enabled.
            _debris_on = st.session_state.get("int_debris_enabled", False)
            _link_vals = {}
            _link_obs = [k for k in _sel_obs
                         if OBSERVABLES.get(k, {}).get("link") and not (k == "od" and _debris_on)]
            if _link_obs:
                _lcols = st.columns(min(3, len(_link_obs)))
                for _li, _ok in enumerate(_link_obs):
                    _sp = OBSERVABLES[_ok]
                    _pn, _op, _dflt = _sp["link"]
                    _link_vals[_ok] = _lcols[_li % len(_lcols)].number_input(
                        f"Link · {_sp['label']} ({_pn})", value=float(_dflt), format="%.3e",
                        key=f"fit_link_{_ok}",
                        help="Scales model state → signal. Tunable below / future fit parameter.")
            if "od" in _sel_obs and _debris_on:
                st.caption("OD uses the **debris module** (`get_od`, includes lysed-cell debris). "
                           "Tune `od_to_cfu` and the debris rates in *Global & structural* below.")
            _t_end_fit = st.number_input("Overlay duration (h)", value=float(np.ceil(_long["time"].max())), step=1.0, key="fit_tend")

            # Per-arm conditions — each group can carry its own growth phase (pre-run),
            # initial density B₀, and MOI, so e.g. log-phase and stationary-phase CFU
            # (same observable, different condition) can be fit together. MOI auto-fills
            # from the data; B₀ defaults to the nominal inoculum.
            _nom_b0 = float(sum(float(_s.get("initial_B", 0.0))
                                for _s in st.session_state.get("int_strains", []))) or 1e7
            _arm_cond = {}
            _dose_help = ("Dose seeds the phage inoculum as an absolute PFU/mL titre for that arm."
                          if _dose_unit == "pfu" else
                          "MOI seeds the phage inoculum as MOI × B₀ for that arm.")
            with st.expander(f"Per-arm conditions (growth phase · B₀ · {_dose_lbl})", expanded=False):
                st.caption("Log phase → pre-run 0 (fresh inoculum). Stationary phase → set a "
                           "pre-run duration to equilibrate toward carrying capacity before t=0. "
                           + _dose_help)
                for _arm in _sel_arms:
                    _cc = st.columns([2, 1, 1, 1])
                    _cc[0].markdown(f"**{_arm}**")
                    _arm_cond[_arm] = {
                        "b0": _cc[1].number_input("B₀ (CFU/mL)", value=_nom_b0, format="%.2e",
                                                  key=f"fit_cond_b0_{_arm}"),
                        "prerun": _cc[2].number_input("Pre-run (h)", value=0.0, step=4.0,
                                                      key=f"fit_cond_prerun_{_arm}"),
                        "moi": _cc[3].number_input(_dose_lbl, format="%g",
                                                   value=float(_conds.get(_arm, {}).get("moi", 0.0)),
                                                   key=f"fit_cond_moi_{_arm}"),
                    }

            # ── 5. Manual parameter tuning (Phase B) ─────────────────────────
            # Edit the model's ACTUAL parameter values (absolute, per entity — like
            # the Interactive Simulator), not multipliers. These widgets read from
            # and write to the shared int_strains / int_phages dicts, so edits ARE
            # the live model: no separate "apply" step, and they're savable as Parts.
            # The widgets are seeded from the dict each render (value=), so they stay
            # in sync with edits made on the Simulator page.
            _tstrains = st.session_state.get("int_strains", [])
            _tphages = st.session_state.get("int_phages", [])
            with st.expander("Manual parameter tuning", expanded=False):
                st.caption("Edit the model's real parameter values, then re-overlay. Changes update the live "
                           "Interactive-Simulator model directly (no separate apply step) and can be saved as "
                           "a Scenario or as Parts in the Library.")

                # ── Global & structural parameters ───────────────────────────
                st.markdown("**Global & structural**")
                _track_nut = st.session_state.get("int_track_nutrients", True)
                gk1, gk2, gk3 = st.columns(3)
                with gk1:
                    st.session_state["int_n_latent"] = int(st.number_input(
                        "Latent compartments (L)", min_value=1, max_value=50,
                        value=int(st.session_state.get("int_n_latent", 5)), step=1, key="fit_edit_n_latent",
                        help="Number of phage latent (eclipse) stages — Erlang shape of the latent period."))
                with gk2:
                    st.session_state["int_carrying_capacity"] = st.number_input(
                        "Carrying capacity K (CFU·mL⁻¹)", value=float(st.session_state.get("int_carrying_capacity", 1e9)),
                        format="%.3e", key="fit_edit_K")
                with gk3:
                    st.session_state["int_monod_constant"] = st.number_input(
                        "Monod constant (Ks)", value=float(st.session_state.get("int_monod_constant", 0.3)),
                        format="%g", key="fit_edit_Ks")
                if _track_nut:
                    nk1, nk2, nk3, nk4 = st.columns(4)
                    with nk1:
                        st.session_state["int_initial_S"] = st.number_input(
                            "Initial nutrient (S₀)", value=float(st.session_state.get("int_initial_S", 1.0)),
                            format="%g", key="fit_edit_S0")
                    with nk2:
                        st.session_state["int_recycle_fraction"] = st.number_input(
                            "Recycle fraction", value=float(st.session_state.get("int_recycle_fraction", 0.0)),
                            format="%g", key="fit_edit_recycle")
                    with nk3:
                        st.session_state["int_s_in"] = st.number_input(
                            "Nutrient inflow (s_in)", value=float(st.session_state.get("int_s_in", 0.0)),
                            format="%g", key="fit_edit_s_in")
                    with nk4:
                        st.session_state["int_s_out"] = st.number_input(
                            "Nutrient washout (s_out)", value=float(st.session_state.get("int_s_out", 0.0)),
                            format="%g", key="fit_edit_s_out")
                else:
                    st.caption("Nutrient tracking is off (constant/logistic growth) — S₀/recycle/inflow/washout "
                               "are inactive. Enable it in the Interactive Simulator to fit them.")
                if st.session_state.get("int_debris_enabled", False):
                    st.markdown("*OD / debris module*")
                    dk1, dk2, dk3, dk4 = st.columns(4)
                    with dk1:
                        st.session_state["int_od_to_cfu_conversion_factor"] = st.number_input(
                            "od_to_cfu", value=float(st.session_state.get("int_od_to_cfu_conversion_factor", 2e8)),
                            format="%.3e", key="fit_edit_od2cfu",
                            help="CFU per OD unit: OD = (biomass + debris) / od_to_cfu.")
                    with dk2:
                        st.session_state["int_debris_u"] = st.number_input(
                            "Debris yield · deaths (u)", value=float(st.session_state.get("int_debris_u", 0.4)),
                            format="%g", key="fit_edit_debris_u")
                    with dk3:
                        st.session_state["int_debris_v"] = st.number_input(
                            "Debris yield · lysis (v)", value=float(st.session_state.get("int_debris_v", 0.2)),
                            format="%g", key="fit_edit_debris_v")
                    with dk4:
                        st.session_state["int_debris_kdis"] = st.number_input(
                            "Debris dissolution (k_dis)", value=float(st.session_state.get("int_debris_kdis", 0.01)),
                            format="%g", key="fit_edit_debris_kdis")

                # Bacterial parameters. IMPORTANT: in Binary-Genotypes (BRG) mode the
                # strain kinetics live on `int_brg_base_*` session keys (a single WT base
                # from which the genotypes are derived), NOT the per-strain dicts — and
                # initial_B comes from the equilibrium IC / per-genotype table. So the
                # per-strain-dict editors below (correct for Direct / Custom-Strains)
                # would be silently ignored in BRG. Edit the right storage per mode.
                _is_brg = st.session_state.get("int_builder_mode", "").startswith("Binary")
                if _is_brg:
                    st.markdown("**Base strain (WT) — genotypes derived**")
                    _bc = st.columns(3)
                    with _bc[0]:
                        st.session_state["int_brg_base_growth"] = st.number_input(
                            "Growth rate (h⁻¹)", value=float(st.session_state.get("int_brg_base_growth", 1.2)),
                            format="%g", key="fit_edit_brg_growth")
                    with _bc[1]:
                        st.session_state["int_brg_base_ratio"] = st.number_input(
                            "Bacteria/resource", value=float(st.session_state.get("int_brg_base_ratio", 1e9)),
                            format="%.2e", key="fit_edit_brg_ratio")
                    with _bc[2]:
                        st.session_state["int_brg_death_rate_B"] = st.number_input(
                            "Natural death rate (h⁻¹)", value=float(st.session_state.get("int_brg_death_rate_B", 0.0)),
                            format="%g", key="fit_edit_brg_death")
                    if st.session_state.get("int_brg_use_eq_ic", False):
                        st.session_state["int_brg_eq_total_B"] = st.number_input(
                            "Total bacteria (equilibrium IC)",
                            value=float(st.session_state.get("int_brg_eq_total_B", 1e7)),
                            format="%.3e", key="fit_edit_brg_eqtotal",
                            help="With the equilibrium initial condition on, per-genotype B₀ is derived "
                                 "from this total (and fitness cost) — individual strain B₀ is not used.")
                    else:
                        st.caption("Per-genotype initial counts are set in the BRG phage-loci table on the "
                                   "Interactive Simulator.")
                _tune_strains = [] if _is_brg else _tstrains
                if _tune_strains:
                    st.markdown("**Bacterial strains**")
                for _si, _s in enumerate(_tune_strains):
                    st.markdown(f"*{_s.get('name', f'Strain {_si}')}*")
                    _scols = st.columns(len(STRAIN_TUNABLES))
                    for _sc, _knob in zip(_scols, STRAIN_TUNABLES):
                        with _sc:
                            _s[_knob["key"]] = st.number_input(
                                _knob["label"], value=float(_s.get(_knob["key"], _knob["default"]) or 0.0),
                                format=_knob["fmt"], key=f"fit_edit_s_{_knob['key']}_{_si}")
                    # dormancy kinetics + depth compartments — only when enabled
                    if _s.get("dormancy_enabled"):
                        _dcols = st.columns(len(STRAIN_DORMANCY_TUNABLES) + 1)
                        with _dcols[0]:
                            _s["dormancy_depth"] = int(st.number_input(
                                "Depth layers (Q)", min_value=1, max_value=10,
                                value=int(_s.get("dormancy_depth", 1)), step=1,
                                key=f"fit_edit_s_dormancy_depth_{_si}",
                                help="Number of dormancy-depth compartments (max across strains sets n_depth)."))
                        for _dc, _knob in zip(_dcols[1:], STRAIN_DORMANCY_TUNABLES):
                            with _dc:
                                _s[_knob["key"]] = st.number_input(
                                    _knob["label"], value=float(_s.get(_knob["key"], _knob["default"]) or 0.0),
                                    format=_knob["fmt"], key=f"fit_edit_s_{_knob['key']}_{_si}")

                # Adsorption is a strain×phage property; its storage is builder-mode
                # specific. Direct / Custom-Strains keep it in the pairwise
                # ads_{strain}_{phage} session keys (edited per pair here); Binary-
                # Genotypes keeps it on the phage dict as adsorption_s.
                _ads_pairwise = not st.session_state.get("int_builder_mode", "").startswith("Binary")

                if _tphages:
                    st.markdown("**Phages**")
                for _pj, _p in enumerate(_tphages):
                    st.markdown(f"*{_p.get('name', f'Phage {_pj}')}*")
                    _pcols = st.columns(len(PHAGE_TUNABLES))
                    for _pc, _knob in zip(_pcols, PHAGE_TUNABLES):
                        with _pc:
                            _p[_knob["key"]] = st.number_input(
                                _knob["label"], value=float(_p.get(_knob["key"], _knob["default"]) or 0.0),
                                format=_knob["fmt"], key=f"fit_edit_p_{_knob['key']}_{_pj}")
                    # Mutation rate / fitness cost — only in Binary-Genotypes, the only
                    # mode that reads them from the phage dict. (Direct-mode phage dicts
                    # may carry a stale `mu`, but Direct/Custom-Strains take mutation from
                    # the strain→strain graph edited on the Simulator, so editing it here
                    # would be a silent no-op.)
                    _opt = PHAGE_OPTIONAL_TUNABLES if _is_brg else []
                    if _opt:
                        _ocols = st.columns(len(_opt))
                        for _oc, _knob in zip(_ocols, _opt):
                            with _oc:
                                _p[_knob["key"]] = st.number_input(
                                    _knob["label"], value=float(_p.get(_knob["key"], _knob["default"]) or 0.0),
                                    format=_knob["fmt"], key=f"fit_edit_p_{_knob['key']}_{_pj}")
                    # adsorption inputs (per strain in pairwise modes: active + dormant)
                    if _ads_pairwise and _tstrains:
                        _acols = st.columns(len(_tstrains))
                        for _si, _s in enumerate(_tstrains):
                            _adk = f"ads_{_si}_{_pj}"
                            with _acols[_si]:
                                st.session_state[_adk] = st.number_input(
                                    f"Adsorption → {_s.get('name', f'Strain {_si}')}",
                                    value=float(st.session_state.get(_adk, 1e-8 if _si == 0 else 0.0)),
                                    format="%.3e", key=f"fit_edit_ads_{_si}_{_pj}")
                            # dormant-cell adsorption for strains with dormancy on
                            if _s.get("dormancy_enabled"):
                                _addk = f"ads_dorm_{_si}_{_pj}"
                                with _acols[_si]:
                                    st.session_state[_addk] = st.number_input(
                                        f"Dormant ads → {_s.get('name', f'Strain {_si}')}",
                                        value=float(st.session_state.get(_addk, 0.0)),
                                        format="%.3e", key=f"fit_edit_adsdorm_{_si}_{_pj}")
                    elif not _ads_pairwise:
                        _adk = entity_param_key(_p, ADSORPTION_PHAGE_KEYS)
                        _p[_adk] = st.number_input(
                            "Adsorption (adsorption_s)", value=float(_p.get(_adk, 5e-8) or 0.0),
                            format="%.3e", key=f"fit_edit_adss_{_pj}")
                        if "adsorption_r" in _p:
                            _p["adsorption_r"] = st.number_input(
                                "Adsorption resistant (adsorption_r)", value=float(_p.get("adsorption_r", 0.0) or 0.0),
                                format="%.3e", key=f"fit_edit_adsr_{_pj}")
                st.caption("Tip: B₀ may be overridden by an equilibrium/pre-run initial condition in some builder "
                           "modes; the phage inoculum in the overlay comes from each group's MOI × B₀.")

            # Compute the overlay only when the button is clicked; store the plot
            # data in session_state so the visualization stays alive across page
            # navigation (and reruns) until it is explicitly re-run or the dataset
            # is cleared.
            # Context shared by the manual overlay and the post-fit fitted overlay.
            _ovl_ctx = {
                "sel_arms": _sel_arms, "sel_obs": _sel_obs, "arm_cond": _arm_cond,
                "conds": _conds, "agg": _agg, "link_vals": _link_vals,
                "debris_on": _debris_on, "band": _band, "band_choice": _band_choice,
                "stat_key": _stat_key, "stat": _stat, "t_end": _t_end_fit,
                "dose_unit": _dose_unit, "dose_label": _dose_lbl,
                "method": st.session_state.get("int_solver_method", "BDF"),
                "thr": st.session_state.get("int_extinction_threshold", 1.0) or None,
            }
            if st.button("Overlay model on data", key="fit_overlay", width="stretch", type="primary"):
                try:
                    _config, _iB, _iP, _iS, _mk = build_nominal_config_from_gui()
                    st.session_state["calib_overlay_result"] = _compute_overlay(
                        _config, _iB, _iP, _iS, _mk, _ovl_ctx)
                except Exception as e:
                    st.session_state["calib_overlay_result"] = None
                    st.error(f"Overlay failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # Render the (persisted) overlay result if one exists.
            _ovr = st.session_state.get("calib_overlay_result")
            if _ovr and _ovr.get("panels"):
                import plotly.graph_objects as go
                _palette = ["#0d7a68", "#c1873a", "#5457a6", "#b5487f", "#3b6fb5",
                            "#2e8b57", "#a0522d", "#6a5acd"]

                def _rgba(hexc, a):
                    h = hexc.lstrip("#")
                    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

                st.markdown(f"#### {_ovr['title']}")
                _npan = len(_ovr["panels"])
                # Combined objective (the single number to minimise; matches what
                # pbisim-fit will jointly minimise) + one R²/RMSE chip per observable.
                _cols = st.columns(1 + _npan)
                _cval = _ovr.get("combined", float("nan"))
                _cols[0].markdown(
                    f"""<div class="metric-container">
                        <div class="metric-label">Combined objective J</div>
                        <div class="metric-value">{_cval:.3f}</div>
                        <div class="metric-sub">pooled log₁₀ least-squares — the pbisim-fit NLS objective</div>
                    </div>""", unsafe_allow_html=True)
                for _ci, _pan in enumerate(_ovr["panels"]):
                    _cols[_ci + 1].markdown(
                        f"""<div class="metric-container">
                            <div class="metric-label">{_pan['label']} · R²</div>
                            <div class="metric-value">{_pan['r2']:.3f}</div>
                            <div class="metric-sub">RMSE (log₁₀) {_pan['rmse']:.3f} · n={_pan['n']}</div>
                        </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # One panel per observable (small multiples — units differ).
                for _pan in _ovr["panels"]:
                    _pfig = go.Figure()
                    for _i, _s in enumerate(_pan["series"]):
                        _color = _palette[_i % len(_palette)]
                        _yp = np.maximum(_s["pred"], 1e-30) if _pan["log"] else _s["pred"]
                        if (not _s["is_raw"]) and _s["obs_lo"] is not None:
                            _pfig.add_trace(go.Scatter(x=_s["obs_time"], y=_s["obs_hi"], mode="lines",
                                                       line=dict(width=0), showlegend=False, hoverinfo="skip"))
                            _pfig.add_trace(go.Scatter(x=_s["obs_time"], y=_s["obs_lo"], mode="lines",
                                                       line=dict(width=0), fill="tonexty",
                                                       fillcolor=_rgba(_color, 0.15), showlegend=False, hoverinfo="skip"))
                        _pfig.add_trace(go.Scatter(x=_s["time"], y=_yp, mode="lines",
                                                   name=f"{_s['label']} (model)", line=dict(color=_color, width=2)))
                        _mk = dict(color=_color, size=5 if _s["is_raw"] else 7,
                                   opacity=0.45 if _s["is_raw"] else 1.0)
                        if not _s["is_raw"]:
                            _mk["line"] = dict(color="#16211f", width=0.4)
                        _pfig.add_trace(go.Scatter(x=_s["obs_time"], y=_s["obs_value"], mode="markers",
                                                   name=f"{_s['label']} (obs)", marker=_mk))
                    _pfig.update_layout(title=_pan["label"], xaxis_title="Time (h)", yaxis_title=_pan["label"],
                                        template="plotly_white", height=380, margin=dict(t=44, b=40),
                                        legend=dict(orientation="h", yanchor="bottom", y=-0.32, x=0))
                    apply_axis_plotly(_pfig, plot_axis_controls(
                        f"calib_ovl_{_pan['obs']}", default_y="Log" if _pan["log"] else "Linear"))
                    st.plotly_chart(_pfig, width="stretch")

                st.markdown(f"#### Fit quality per group × observable (vs {_ovr['stat_label']})")
                st.dataframe(pd.DataFrame(_ovr["metrics"]), width="stretch", hide_index=True)
                st.caption("Edit the parameter values above and re-overlay to improve the fit "
                           "(minimise the combined objective). Edits update the live model directly "
                           "and can be saved as Parts in the Library.")

            # ── 5b · Export fit specification (pbisim-fit hand-off) ───────────
            st.markdown("### 5b · Export fit specification")
            st.caption("Bundle the data (long format), per-arm conditions (growth phase / B₀ / MOI), "
                       "the selected observables + detection floors, and the current parameter values "
                       "into a **pbisim-fit specification** — exactly the inputs its NLS `refine_nls` / "
                       "`fit()` consume (dataset → ExperimentalDataset, nls_cfg → NLSConfig).")
            try:
                _spec_cfg, _sB, _sP, _sS, _smk = build_nominal_config_from_gui()
                _od_link = _link_vals.get("od")
                if _od_link is None and _debris_on:
                    _od_link = st.session_state.get("int_od_to_cfu_conversion_factor")
                _spec = build_fit_spec(_agg, _sel_arms, _sel_obs, _arm_cond,
                                       od_to_cfu=_od_link, dose_unit=_dose_unit,
                                       model_params=config_param_snapshot(_spec_cfg))
                st.download_button(
                    "Download fit spec (JSON)", data=json.dumps(_spec, indent=2),
                    file_name="pbisim_fit_spec.json", mime="application/json",
                    key="fit_export_spec", width="stretch")
            except Exception as _e:
                st.caption(f"(Select at least one group and observable first — {_e})")

            # ── 5c · Run NLS fit (pbisim-fit, in-app) ─────────────────────────
            st.markdown("### 5c · Run NLS fit (pbisim-fit)")
            st.caption("Fit the selected parameters to the selected arms/observables with "
                       "pbisim-fit's non-linear least squares (`refine_nls`) — the same engine "
                       "the export spec feeds. Runs locally; needs `pbisim-fit` installed.")
            try:
                from pbisim_app import nls_fit as _nls
                # Fit runs against a CHOSEN Model, not the live builder — a frozen
                # saved/demo model can't be contaminated by unrelated builder edits.
                _mopts = model_options()
                _mdef = st.session_state.active_model if st.session_state.active_model in _mopts else WORKING_DRAFT_LABEL
                _fit_model = st.selectbox(
                    "Model to fit", _mopts, index=_mopts.index(_mdef), key="fit_model_sel",
                    help="The base model whose free parameters are estimated. Saved/demo "
                         "models are frozen snapshots; 'Working draft (live)' uses the "
                         "current builder state.")
                _fit_snap = resolve_model_snapshot(_fit_model)
                _fit_cfg, _fB, _fP, _fS, _fmk = build_config_from_model(_fit_snap)
                # ── Comprehensive parameter table: fix / free each model parameter ─
                _targets_cat = _nls.available_targets(
                    _fit_cfg, initial_cfu=float(np.sum(_fB)) if _fB is not None else None,
                    initial_pfu=(float(_fP[0]) if _fP is not None and len(_fP) else None))
                _path_label = {p: lab for (lab, p, *_r) in _targets_cat}
                _label_to_path = {lab: p for (lab, p, *_r) in _targets_cat}
                def _bound(s, log, is_lower):
                    """Blank bound → unconstrained on that side: +inf above; -inf (or 0
                    for a log-space parameter) below. A set value (scientific notation
                    accepted) is used verbatim."""
                    v = _pf(s)
                    if v is not None:
                        return float(v)
                    if not is_lower:
                        return float("inf")
                    return 0.0 if log else float("-inf")

                import hashlib as _hl
                # A revision counter (bumped when a text spec is applied) is appended to
                # the editor KEYS only — it forces the keyed data_editors to reload from
                # the freshly-parsed dataframes WITHOUT triggering the defaults rebuild
                # (which keys off _sig).
                _rev = int(st.session_state.get("fit_spec_rev", 0))
                _sig = _hl.md5(("|".join(p for (_l, p, *_r) in _targets_cat) + "|" + _fit_model)
                               .encode()).hexdigest()[:10]
                _reset_tbl = st.button("Reset table to model defaults", key="fit_tbl_reset")
                # Default table on model change / reset: one row per model
                # parameter, role=Fixed, bounds blank. Cells are TEXT (sci notation).
                if st.session_state.get("fit_targets_sig") != _sig or _reset_tbl:
                    _rows = [{"parameter": lab, "path": p, "role": "Fixed", "value": f"{v:g}",
                              "lower": "", "upper": "", "log": bool(log),
                              "prior \u03bc": "", "prior \u03c3": "", "expression": ""}
                             for (lab, p, v, lo_f, hi_f, log) in _targets_cat]
                    st.session_state["fit_targets_df"] = pd.DataFrame(_rows)
                    st.session_state["fit_targets_sig"] = _sig

                st.markdown("**Fit parameters** \u2014 set each parameter's **role**: "
                            "**Fixed** (held at value) \u00b7 **Free** (estimated with bounds/prior) "
                            "\u00b7 **Derived** (`= expression` of a \u03b8, defined below).")
                st.caption("Bounds/priors optional (blank = unconstrained / no prior); values accept "
                           "scientific notation (`1e7`); **log** = log-space search. A **Derived** row "
                           "draws its value/bounds/prior from its \u03b8 (those cells blank out \u2014 set them "
                           "on the \u03b8 below). **Sharing** = set several rows to *Derived* with the same "
                           "\u03b8 (use the **Share** helper). It's all in this one table \u2014 no separate sections.")
                _tg_ed = st.data_editor(
                    st.session_state["fit_targets_df"], key=f"fit_targets_editor_{_sig}_{_rev}",
                    hide_index=True, width="stretch",
                    column_config={
                        "parameter": st.column_config.TextColumn("parameter", disabled=True),
                        "path": None,
                        "role": st.column_config.SelectboxColumn(
                            "role", options=["Fixed", "Free", "Derived"], required=True,
                            help="Fixed: held at value \u00b7 Free: estimated 1:1 \u00b7 Derived: = expression of \u03b8"),
                        "value": st.column_config.TextColumn("value / start",
                                                             help="Fixed value, or start for a Free fit. e.g. 1e8"),
                        "lower": st.column_config.TextColumn("lower", help="Free only. Blank = unbounded. e.g. 1e7"),
                        "upper": st.column_config.TextColumn("upper", help="Free only. Blank = unbounded. e.g. 1e10"),
                        "log": st.column_config.CheckboxColumn("log", help="Free/\u03b8: log-space search."),
                        "prior \u03bc": st.column_config.TextColumn("prior \u03bc", help="Free only. Gaussian prior mean -> MAP."),
                        "prior \u03c3": st.column_config.TextColumn("prior \u03c3", help="Free only. Prior stdev."),
                        "expression": st.column_config.TextColumn("= expression (Derived)",
                                                                  help="For Derived rows: an expression of \u03b8, e.g. g*(1-cost)"),
                    })

                # A Derived row inherits value/bounds/priors from its \u03b8, so blank those
                # cells (they read as inactive) and keep only role + expression live.
                _recs = _tg_ed.to_dict("records")
                _norm = False
                for _r in _recs:
                    if str(_r.get("role")) == "Derived":
                        for _c in ("value", "lower", "upper", "prior \u03bc", "prior \u03c3"):
                            if str(_r.get(_c, "")).strip():
                                _r[_c] = ""; _norm = True
                        if bool(_r.get("log")):
                            _r["log"] = False; _norm = True
                if _norm:
                    st.session_state["fit_targets_df"] = pd.DataFrame(_recs)
                    st.session_state["fit_spec_rev"] = _rev + 1
                    st.rerun()

                # -- Share helper: tie selected rows to one new theta --
                if st.session_state.pop("_share_clear", None):
                    for _k in ("fit_share_pick", "fit_share_name"):
                        st.session_state.pop(_k, None)
                with st.expander("Share \u2014 tie several parameters to one \u03b8", expanded=False):
                    st.caption("Sets the selected parameters to **Derived = one new \u03b8**; define that "
                               "\u03b8's bounds/prior in the \u03b8 table below.")
                    _sh_pick = st.multiselect("Parameters to share",
                                              [l for (l, *_r) in _targets_cat], key="fit_share_pick")
                    _shc = st.columns([3, 1])
                    _sh_name = _shc[0].text_input("\u03b8 name (blank = auto)", key="fit_share_name")
                    if _shc[1].button("Share \u2192", key="fit_share_go", width="stretch"):
                        if len(_sh_pick) < 2:
                            st.warning("Pick at least two parameters to share.")
                        else:
                            _exist = (set(st.session_state["fit_thetas_df"]["name"])
                                      if "fit_thetas_df" in st.session_state else set())
                            _nm = (_sh_name.strip() or next(
                                f"shared{i}" for i in range(1, 999) if f"shared{i}" not in _exist))
                            _picked = {_label_to_path[l] for l in _sh_pick}
                            _df = st.session_state["fit_targets_df"].copy()
                            for _i, _r in _df.iterrows():
                                if _r["path"] in _picked:
                                    _df.at[_i, "role"] = "Derived"
                                    _df.at[_i, "expression"] = _nm
                            st.session_state["fit_targets_df"] = _df
                            _thd = (st.session_state["fit_thetas_df"].copy()
                                    if "fit_thetas_df" in st.session_state else
                                    pd.DataFrame(columns=["name", "lower", "upper", "log",
                                                          "initial", "prior \u03bc", "prior \u03c3"]))
                            _thd = pd.concat([_thd, pd.DataFrame([{
                                "name": _nm, "lower": "", "upper": "", "log": False,
                                "initial": "", "prior \u03bc": "", "prior \u03c3": ""}])], ignore_index=True)
                            st.session_state["fit_thetas_df"] = _thd
                            st.session_state["fit_spec_rev"] = _rev + 1
                            st.session_state["_share_clear"] = True
                            st.rerun()

                # -- Custom parameters (theta) --
                with st.expander("Custom parameters (\u03b8) \u2014 used by Derived rows", expanded=False):
                    st.caption("Estimated quantities referenced by Derived expressions. Name + bounds "
                               "(blank = unconstrained); give large-scale \u03b8 an initial. Optional "
                               "prior \u03bc/\u03c3. Unreferenced \u03b8 are ignored.")
                    if "fit_thetas_df" not in st.session_state:
                        st.session_state["fit_thetas_df"] = pd.DataFrame(
                            [{"name": "", "lower": "", "upper": "", "log": False, "initial": "",
                              "prior \u03bc": "", "prior \u03c3": ""}])
                    _th_ed = st.data_editor(
                        st.session_state["fit_thetas_df"], key=f"fit_thetas_editor_{_rev}",
                        num_rows="dynamic", hide_index=True, width="stretch",
                        column_config={
                            "name": st.column_config.TextColumn("\u03b8 name"),
                            "lower": st.column_config.TextColumn("lower", help="Blank = unbounded below. e.g. 1e7"),
                            "upper": st.column_config.TextColumn("upper", help="Blank = unbounded above. e.g. 1e10"),
                            "log": st.column_config.CheckboxColumn("log", help="Log-space search."),
                            "initial": st.column_config.TextColumn("initial (start)",
                                                                   help="Start value \u2014 important when unbounded. e.g. 1e9"),
                            "prior \u03bc": st.column_config.TextColumn("prior \u03bc", help="Optional prior mean -> MAP."),
                            "prior \u03c3": st.column_config.TextColumn("prior \u03c3", help="Prior stdev."),
                        })
                    _th_names = [str(r["name"]).strip() for r in _th_ed.to_dict("records")
                                 if str(r.get("name", "")).strip()]

                # -- Assembly: role -> free targets + Derived mappings; thetas --
                _targets, _mappings, _bad = [], [], []
                for r in _tg_ed.to_dict("records"):
                    _role = str(r.get("role", "Fixed"))
                    _log = bool(r["log"])
                    _lo = _bound(r.get("lower"), _log, True)
                    _hi = _bound(r.get("upper"), _log, False)
                    if np.isfinite(_lo) and np.isfinite(_hi) and _lo >= _hi:
                        if _role == "Free":
                            st.warning(f"'{_path_label.get(r['path'], r['path'])}': lower \u2265 upper \u2014 unbounded.")
                        _lo, _hi = (0.0 if _log else float("-inf")), float("inf")
                    _val = _pf(r.get("value"))
                    _targets.append({"path": r["path"], "free": (_role == "Free"),
                                     "value": (_val if _val is not None else 0.0),
                                     "lo": _lo, "hi": _hi, "log": _log,
                                     "prior_mu": _pf(r.get("prior \u03bc")), "prior_sd": _pf(r.get("prior \u03c3"))})
                    if _role == "Derived":
                        _ex = str(r.get("expression", "")).strip()
                        if not _ex:
                            _bad.append(f"{r['path']}: Derived but no expression")
                        else:
                            _ok, _msg = _nls.validate_expr(_ex, _th_names)
                            if _ok:
                                _mappings.append({"path": r["path"], "expr": _ex})
                            else:
                                _bad.append(f"{r['path']} = {_ex}: {_msg}")
                for _b in _bad:
                    st.warning(_b)
                _used = {n for n in _th_names
                         if any(re.search(rf"\b{re.escape(n)}\b", m["expr"]) for m in _mappings)}
                _thetas, _theta_bad = [], []
                for r in _th_ed.to_dict("records"):
                    _nm = str(r.get("name", "")).strip()
                    if _nm not in _used:
                        continue
                    _log = bool(r.get("log"))
                    _lo = _bound(r.get("lower"), _log, True)
                    _hi = _bound(r.get("upper"), _log, False)
                    if np.isfinite(_lo) and np.isfinite(_hi) and _lo >= _hi:
                        _theta_bad.append(_nm)
                        continue
                    _thetas.append({"name": _nm, "lo": _lo, "hi": _hi, "log": _log,
                                    "initial": _pf(r.get("initial")),
                                    "prior_mu": _pf(r.get("prior \u03bc")), "prior_sd": _pf(r.get("prior \u03c3"))})
                if _theta_bad:
                    st.warning("\u03b8 with lower \u2265 upper: " + ", ".join(sorted(set(_theta_bad))))

                # -- Fit spec (text) -- two-way with the table above --
                _sp_pending = st.session_state.pop("_spec_pending", None)
                if _sp_pending is not None:
                    st.session_state["fit_spec_text"] = _sp_pending
                with st.expander("Fit spec (text) \u2014 two-way with the table above", expanded=False):
                    st.caption("Compact text form; round-trips with the table. Grammar (one per line): "
                               "`free <path> [init= bounds=LO..HI prior=MU,SD log]`, `fix <path> = <v>`, "
                               "`theta <name> [...]`, `map <path> = <expr>`. Generate-from-table lists the "
                               "model's parameters as a comment header.")
                    _spec_txt = st.text_area("Spec", key="fit_spec_text", height=220,
                                             placeholder="theta g bounds=0.1..3.0 prior=1.2,0.3\n"
                                                         "map growth_rates[0] = g\nmap growth_rates[1] = g*(1-cost)")
                    _spc = st.columns(2)
                    if _spc[0].button("\u2191 Generate from table", key="fit_spec_from_tables", width="stretch"):
                        st.session_state["_spec_pending"] = _nls.serialize_fit_spec(
                            _targets, _thetas, _mappings, _targets_cat)
                        st.rerun()
                    if _spc[1].button("\u2193 Apply to table", key="fit_spec_to_tables",
                                      type="primary", width="stretch"):
                        _tdf, _thdf, _errs = _nls.parse_fit_spec(_spec_txt or "", _targets_cat)
                        if _errs:
                            for _e in _errs:
                                st.error(_e)
                        else:
                            st.session_state["fit_targets_df"] = _tdf
                            st.session_state["fit_thetas_df"] = _thdf
                            st.session_state["fit_spec_rev"] = _rev + 1
                            st.success("Applied to the table above.")
                            st.rerun()

                # OD link (CFU per OD unit): pbisim-fit reads it off the config, so it
                # is REQUIRED to fit 'od'. Prefer the data's own median CFU/OD ratio
                # (most principled when both are measured) → debris factor → overlay
                # link → 2e8; the user can override.
                _od_link = None
                if "od" in _sel_obs:
                    _od_link = _nls.estimate_od_to_cfu(_agg, _sel_arms)
                    if _od_link is None and _debris_on:
                        _od_link = st.session_state.get("int_od_to_cfu_conversion_factor")
                    if _od_link is None:
                        _od_link = _link_vals.get("od")
                    _od_link = st.number_input(
                        "OD link — CFU per OD unit (od_to_cfu)",
                        min_value=0.0, value=float(_od_link or 2e8), key="fit_nls_od2cfu",
                        help="Model OD = biomass ÷ this factor. Defaults to the data's "
                             "median CFU/OD ratio; required to fit the OD observable.")
                _c1, _c2 = st.columns(2)
                with _c1:
                    _restarts = int(st.number_input("Restarts", 1, 10, 3, key="fit_nls_restarts"))
                with _c2:
                    _maxnfev = int(st.number_input("Max evaluations / restart", 50, 2000, 300,
                                                   step=50, key="fit_nls_maxnfev"))
                if _nls.has_unbounded(_targets, _thetas):
                    st.caption("Unbounded parameter(s) present → the fit uses a **single start** "
                               "(multi-start needs finite bounds).")
                _any_free = any(t["free"] for t in _targets) or bool(_thetas)
                _job = st.session_state.get("fit_job")
                _running = bool(_job and _job.get("status") == "running")
                # The fit runs in a background thread so the UI never freezes and a bad
                # fit can be stopped (pbisim-fit's refine_nls has no cancel hook, so
                # Stop discards the result; the running restart finishes in the bg).
                if st.button("Run NLS fit", key="fit_run_nls", type="primary",
                             width="stretch", disabled=_running):
                    if _theta_bad:
                        st.error("theta(s) have lower ≥ upper: "
                                 + ", ".join(sorted(set(_theta_bad))) + " — fix or blank a bound.")
                    elif not _any_free:
                        st.error("Set at least one parameter's role to **Free** "
                                 "(or add a referenced θ) — nothing is being estimated.")
                    elif not _sel_arms or not _sel_obs:
                        st.error("Select at least one arm and observable above.")
                    else:
                        try:
                            _ds_fit = _nls.build_dataset(_agg, _sel_arms, _sel_obs, _arm_cond,
                                                         od_to_cfu=_od_link, dose_unit=_dose_unit)
                            _holder = {
                                "status": "running", "t0": _time.time(), "fp": None, "error": None,
                                "cfg": _fit_cfg, "targets": _targets, "thetas": _thetas,
                                "mappings": _mappings, "ds": _ds_fit, "obs": list(_sel_obs),
                                "od_link": _od_link, "restarts": _restarts, "maxnfev": _maxnfev,
                                # post-processing context captured now, so edits during the
                                # fit don't change how the result is interpreted/overlaid.
                                "fit_model": _fit_model, "path_label": dict(_path_label),
                                "fB": _fB, "fP": _fP, "fS": _fS, "fmk": _fmk,
                                "ovl_ctx": dict(_ovl_ctx),
                                # short labels of what's being estimated (for the status line)
                                "param_preview": (
                                    [_path_label.get(t["path"], t["path"]) for t in _targets
                                     if t["free"] and t["path"] not in {m["path"] for m in _mappings}]
                                    + [f"θ {th['name']}" for th in _thetas]),
                            }
                            st.session_state["fit_job"] = _holder
                            threading.Thread(target=_fit_worker, args=(_holder,), daemon=True).start()
                            st.rerun()
                        except Exception as _fe:
                            st.error(f"Could not start fit: {_fe}")

                # Poll / render the background fit.
                _job = st.session_state.get("fit_job")
                if _job is not None and _job.get("status") == "running":
                    _el = _time.time() - _job.get("t0", _time.time())
                    _pnames = ", ".join(_job.get("param_preview", []))
                    st.info(f"⏳ Fitting **{len(_job.get('param_preview', []))} parameter(s)** — "
                            f"~{_el:0.0f}s elapsed. The app stays responsive; press **Stop** to abandon."
                            + (f"  \nEstimating: {_pnames}" if _pnames else ""))
                    if st.button("Stop fit", key="fit_stop"):
                        _job["status"] = "cancelled"
                        st.session_state["fit_job"] = None
                        st.warning("Fit stopped — result discarded. (Fix the bounds and re-run.)")
                        st.rerun()
                    # Poll every ~1 s (the solver has no per-iteration callback, so we
                    # can't stream MSE) — a slower cadence keeps the page from flickering.
                    _time.sleep(1.0)
                    st.rerun()
                elif _job is not None and _job.get("status") == "done":
                    _fp = _job["fp"]
                    try:
                        _map = _fp.map()
                        try:
                            _ci = _fp.credible_interval(0.95)
                        except Exception:
                            _ci = {}
                        _mapped = {m["path"] for m in _job["mappings"]}
                        _pl = _job["path_label"]
                        _pmeta = [{"label": _pl.get(t["path"], t["path"]), "key": f"free{k}"}
                                  for k, t in enumerate(_job["targets"])
                                  if t["free"] and t["path"] not in _mapped]
                        _pmeta += [{"label": f"θ {th['name']}", "key": th["name"]}
                                   for th in _job["thetas"]]
                        st.session_state["calib_fit_result"] = {
                            "map": {k: float(v) for k, v in _map.items()},
                            "ci": {k: [float(a), float(b)] for k, (a, b) in _ci.items()},
                            "params": _pmeta, "model": _job["fit_model"],
                        }
                        _fcfg = _fp.to_config()
                        st.session_state["calib_fitted_config"] = _fcfg
                        try:
                            # If B₀ / initial phage were estimated (fit_initial_cfu/pfu),
                            # reflect them in the overlay's inoculum so the fitted curve
                            # matches the data (the fit used them as the ICs). NOTE: the
                            # per-arm condition editor sets arm_cond["b0"], which overrides
                            # the nominal B₀ inside _compute_overlay — so the estimate must
                            # be written into arm_cond, not just the iB vector.
                            _fB2, _fP2 = _job["fB"], _job["fP"]
                            _fovl = dict(_job["ovl_ctx"])
                            _ic = getattr(_fcfg, "fit_initial_cfu", None)
                            if _ic is not None and np.isfinite(_ic) and _ic > 0:
                                if float(np.sum(_fB2)) > 0:
                                    _fB2 = _fB2 * (float(_ic) / float(np.sum(_fB2)))
                                _fovl["arm_cond"] = {a: {**c, "b0": float(_ic)}
                                                     for a, c in _job["ovl_ctx"]["arm_cond"].items()}
                            _ip = getattr(_fcfg, "fit_initial_pfu", None)
                            if _ip is not None and np.isfinite(_ip) and _ip > 0 and len(_fP2):
                                _fP2 = np.asarray(_fP2, dtype=float).copy(); _fP2[0] = float(_ip)
                            _fovl["title"] = "Fitted model vs observations (NLS MAP)"
                            if _job["od_link"]:
                                _fovl["link_vals"] = dict(_job["ovl_ctx"]["link_vals"], od=_job["od_link"])
                            st.session_state["calib_overlay_result"] = _compute_overlay(
                                _fcfg, _fB2, _fP2, _job["fS"], _job["fmk"], _fovl)
                        except Exception:
                            pass
                    finally:
                        st.session_state["fit_job"] = None
                    st.rerun()
                elif _job is not None and _job.get("status") == "error":
                    st.error(f"Fit failed: {_job.get('error')}")
                    st.session_state["fit_job"] = None

                _fr = st.session_state.get("calib_fit_result")
                if _fr:
                    _rows = []
                    for _pm in _fr.get("params", []):
                        _mv = _fr["map"].get(_pm["key"])
                        _lo, _hi = (_fr["ci"].get(_pm["key"]) or [None, None])
                        _rows.append({
                            "parameter": _pm["label"],
                            "MAP": (f"{_mv:.4g}" if _mv is not None else "—"),
                            "95% CI low": (f"{_lo:.4g}" if _lo is not None else "—"),
                            "95% CI high": (f"{_hi:.4g}" if _hi is not None else "—"),
                        })
                    st.dataframe(pd.DataFrame(_rows), hide_index=True, width="stretch")
                    st.caption("The overlay above already shows the fitted curves. **Apply** writes "
                               "these values into the model the fit ran against, so every task can reuse it.")
                    if st.button("Apply fitted values to model", key="fit_apply_map",
                                 width="stretch"):
                        _tgt = _fr.get("model", WORKING_DRAFT_LABEL)
                        # Make the builder the fit's base model, then write the FITTED
                        # CONFIG's resolved parameters onto it (robust for reparameterized
                        # fits, where the MAP keys are theta names rather than paths).
                        _tsnap = resolve_model_snapshot(_tgt)
                        if _tsnap is not None:
                            apply_model_to_state(_tsnap)
                        _fcfg = st.session_state.get("calib_fitted_config")
                        if _fcfg is not None:
                            _apply_config_to_state(_fcfg)
                        # Drop the manual-tuning widgets so they re-seed from the
                        # updated dicts — otherwise their stale stored values would
                        # overwrite the just-applied fit on the next rerun.
                        for _k in [k for k in list(st.session_state.keys())
                                   if k.startswith("fit_edit_")]:
                            st.session_state.pop(_k, None)
                        if _tgt in st.session_state.user_models:
                            st.session_state.user_models[_tgt]["state"] = dump_model()
                            st.session_state["_pending_active_model"] = _tgt
                            st.success(f"Applied fitted values to saved model '{_tgt}' and updated it.")
                        else:
                            _reason = ("demo model is read-only" if _tgt != WORKING_DRAFT_LABEL
                                       else "working draft")
                            st.success(f"Applied fitted values to the builder ({_reason}) — "
                                       "save it as a Model (sidebar) to keep.")
                        st.rerun()
            except Exception as _e:
                st.caption(f"(Load data and select parameters to run a fit — {_e})")

        # ── 6. Save the calibrated model ─────────────────────────────────────
        if _ds:
            st.markdown("### 6 · Save the calibrated model")
            st.caption("Parameter edits in the tuning panel are **already applied** to the live "
                       "Interactive-Simulator model. Save the whole calibrated configuration as a "
                       "Scenario to reload it later (from the Library page), or save individual "
                       "strains/phages as Parts in the Library.")
            _cs1, _cs2 = st.columns([3, 2])
            with _cs1:
                _cal_name = st.text_input("Scenario name", value="calibrated", key="fit_save_name")
            with _cs2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Save calibrated config as Scenario", key="fit_save_scenario", width="stretch"):
                    _nm = (_cal_name or "").strip()
                    if not _nm:
                        st.error("Enter a scenario name.")
                    else:
                        _scen = st.session_state.user_scenarios
                        _scen[_nm] = {
                            "annotation": "Saved from Calibration",
                            "schema_version": SCENARIO_SCHEMA_VERSION,
                            "state": dump_state_to_scenario(),
                        }
                        st.session_state.user_scenarios = _scen
                        st.success(f"Saved scenario '{_nm}'. Reload it from the Library page.")

        if st.button("Clear dataset", key="fit_clear"):
            st.session_state.fit_dataset = None
            st.session_state.fit_config = {}
            st.session_state.calib_overlay_result = None
            st.rerun()

    # Save the current Calibration widget selections to the persistent config so they
    # survive navigation (see the re-seed block at the top of this page). The
    # parameter-tuning widgets (fit_edit_*) are excluded: they mirror the live
    # int_strains/int_phages dicts (authoritative + already persistent), so caching
    # and re-seeding them would let a stale copy override edits made elsewhere.
    for _wk in list(st.session_state.keys()):
        if (_wk.startswith("fit_") and _wk not in _FIT_NOPERSIST
                and not _wk.startswith("fit_edit_")
                and not _wk.startswith("fit_targets")   # parameter table (df + editor + sig)
                and not _wk.startswith("fit_thetas")     # theta table
                and not _wk.startswith("fit_share")      # Share helper (transient; self-clears)
                and not _wk.startswith("fit_map")        # (removed) mapping table
                and not _wk.startswith("fit_shrq_")      # (removed) shared-param quick builder
                and not _wk.startswith("fit_shr_")       # (removed) sharing-builder widgets
                and not _wk.startswith("fit_der_")):     # (removed) derived-link builder widgets
            st.session_state.fit_config[_wk] = st.session_state[_wk]


# ── AI Simulation Assistant Page ──────────────────────────────────────────────
