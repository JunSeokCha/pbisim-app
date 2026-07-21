"""Rendered by app.py when this page is selected."""
from pbisim_app.common import *  # noqa: F401,F403


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
                      "fit_load", "fit_save_scenario"}
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
                _mc = st.selectbox("Phage-dose / MOI column (optional — drives the simulated dose per arm)",
                                   ["(none)"] + _cols, index=(1 + _guess(["moi", "dose_phage"])) if ("moi" in _low or "dose_phage" in _low) else 0)
                _mc = None if _mc == "(none)" else _mc
            if st.button("Load dataset", key="fit_load", width="stretch"):
                st.session_state.fit_dataset = {
                    "raw": _raw, "time": _tc, "value": _vc, "observable": _obs,
                    "arm_cols": _ac, "moi": _mc,
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
            if st.button("Overlay model on data", key="fit_overlay", width="stretch", type="primary"):
                try:
                    _config, _iB, _iP, _iS, _mk = build_nominal_config_from_gui()
                    _B0 = float(np.sum(_iB))
                    _method = st.session_state.get("int_solver_method", "BDF")
                    _thr = st.session_state.get("int_extinction_threshold", 1.0) or None
                    # One simulation per arm; every selected observable is projected from
                    # that single trajectory into its own panel (small multiples).
                    _panels, _metrics = {}, []
                    for _arm in _sel_arms:
                        _moi = float(_conds.get(_arm, {}).get("moi", 0.0))
                        _armP = np.zeros(len(_iP))
                        if len(_iP):
                            _armP[0] = _moi * _B0
                        _m = PBIModel(_config, initial_B=_iB, initial_P=_armP, initial_S=_iS, **_mk)
                        _r = solve_ode(_m, t_end=_t_end_fit, dt=0.25, method=_method, extinction_threshold=_thr)
                        for _ok in _sel_obs:
                            _sp = OBSERVABLES.get(_ok, {"log": True, "link": None, "label": _ok})
                            _umo = (_ok == "od" and _debris_on)
                            _pred = predicted_observable(_r, _ok, _link_vals.get(_ok), use_model_od=_umo)
                            _d = _agg[(_agg["arm"] == _arm) & (_agg["observable"] == _ok)].sort_values("time")
                            if not len(_d):
                                continue  # this arm carries no data for this observable
                            _has_band = _band is not None and _d["lo"].notna().any()
                            _pan = _panels.setdefault(_ok, {"obs": _ok, "label": _sp.get("label", _ok),
                                                            "log": bool(_sp.get("log")), "series": []})
                            _pan["series"].append({
                                "label": _arm,
                                "time": np.asarray(_r.time),
                                "pred": np.asarray(_pred),
                                "obs_time": _d["time"].to_numpy(),
                                "obs_value": _d["value"].to_numpy(),
                                "obs_lo": _d["lo"].to_numpy() if _has_band else None,
                                "obs_hi": _d["hi"].to_numpy() if _has_band else None,
                                "is_raw": _stat_key == "raw",
                            })
                            _metrics.append({"observable": _sp.get("label", _ok), "group": _arm, "MOI": _moi,
                                             "n_points": len(_d),
                                             "RMSE": fit_residual(_r.time, _pred, _d["time"].values,
                                                                  _d["value"].values, _sp.get("log", False))})

                    # Per-observable pooled RMSE + R² (model interpolated onto obs times;
                    # log10 space for logged observables), plus a combined objective =
                    # equal-weight mean of the per-observable RMSEs.
                    _panel_list, _rmses = [], []
                    for _ok, _pan in _panels.items():
                        _oa, _pa = [], []
                        for _s in _pan["series"]:
                            _pt = np.interp(_s["obs_time"], _s["time"], _s["pred"])
                            _ov = np.asarray(_s["obs_value"], dtype=float)
                            if _pan["log"]:
                                _pt = np.log10(np.maximum(_pt, 1e-30))
                                _ov = np.log10(np.maximum(_ov, 1e-30))
                            _oa.append(_ov)
                            _pa.append(np.asarray(_pt, dtype=float))
                        _oa = np.concatenate(_oa) if _oa else np.array([])
                        _pa = np.concatenate(_pa) if _pa else np.array([])
                        _msk = np.isfinite(_oa) & np.isfinite(_pa)
                        _oa, _pa = _oa[_msk], _pa[_msk]
                        _rmse = float(np.sqrt(np.mean((_oa - _pa) ** 2))) if _oa.size else float("nan")
                        _sst = float(np.sum((_oa - _oa.mean()) ** 2)) if _oa.size else 0.0
                        _pan["rmse"] = _rmse
                        _pan["r2"] = (1.0 - float(np.sum((_oa - _pa) ** 2)) / _sst) if _sst > 0 else float("nan")
                        _pan["n"] = int(_oa.size)
                        _panel_list.append(_pan)
                        if np.isfinite(_rmse):
                            _rmses.append(_rmse)

                    _stat_label = _stat if _stat_key != "raw" else "raw points"
                    st.session_state["calib_overlay_result"] = {
                        "panels": _panel_list,
                        "metrics": _metrics,
                        "combined": float(np.mean(_rmses)) if _rmses else float("nan"),
                        "stat_label": _stat_label,
                        "title": (f"Model vs observations ({_stat_label}"
                                  + (f" + {_band_choice} band)" if _band else ")")),
                    }
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
                        <div class="metric-sub">equal-weight mean of per-observable RMSE</div>
                    </div>""", unsafe_allow_html=True)
                for _ci, _pan in enumerate(_ovr["panels"]):
                    _cols[_ci + 1].markdown(
                        f"""<div class="metric-container">
                            <div class="metric-label">{_pan['label']} · R²</div>
                            <div class="metric-value">{_pan['r2']:.3f}</div>
                            <div class="metric-sub">RMSE{' (log₁₀)' if _pan['log'] else ''} {_pan['rmse']:.3f} · n={_pan['n']}</div>
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
                and not _wk.startswith("fit_edit_")):
            st.session_state.fit_config[_wk] = st.session_state[_wk]


# ── AI Simulation Assistant Page ──────────────────────────────────────────────
