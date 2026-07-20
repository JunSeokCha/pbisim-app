"""Rendered by app.py when this page is selected."""
from pbisim_app.common import *  # noqa: F401,F403


def render():
    theme_mode = st.session_state.get("theme_mode", "Light")
    st.title("Library")
    st.caption("Reusable building blocks. **Scenarios** = whole configurations; "
               "**Parts** = individual bacteria / phages / antibiotics you compose.")

    st.markdown("## Scenarios")
    st.markdown(
        "<div class='info-banner'>A scenario captures your <b>entire</b> configuration. "
        "Loading one configures the <b>Interactive Simulator</b> and applies across all pages "
        "(sweeps, clinical trials). Export to JSON to keep a portable personal library.</div>",
        unsafe_allow_html=True,
    )

    # ── My Scenarios (save / load / export / import full configurations) ──────
    st.caption(
        "Save the **entire current configuration** (builder mode, strains/phages/"
        "antibiotics, dosing, nutrient, immune, solver, prerun, and trial design) as a "
        "reusable scenario. Scenarios live in this browser session — **export to JSON to "
        "keep them** (your portable personal library) and re-import any time."
    )
    _scenarios = st.session_state.user_scenarios

    sc_save, sc_io = st.columns(2)
    with sc_save:
        with st.expander("+ Save current configuration", expanded=not _scenarios):
            _sc_name = st.text_input("Scenario name", value=f"Scenario {len(_scenarios) + 1}", key="sc_save_name")
            _sc_note = st.text_area(
                "Annotation (optional)", key="sc_save_note",
                placeholder="e.g. PA high-persister + fast-adsorbing phage, immunocompromised host",
            )
            if st.button("Save scenario", key="sc_save_btn", width="stretch"):
                _name = (_sc_name or "").strip()
                if not _name:
                    st.error("Please enter a scenario name.")
                else:
                    _scenarios[_name] = {
                        "annotation": _sc_note or "",
                        "schema_version": SCENARIO_SCHEMA_VERSION,
                        "state": dump_state_to_scenario(),
                    }
                    st.session_state.user_scenarios = _scenarios
                    st.success(f"Saved '{_name}'.")
                    st.rerun()
    with sc_io:
        with st.expander("Export / Import library", expanded=False):
            st.download_button(
                "Export all scenarios (JSON)",
                data=export_scenarios_json(_scenarios),
                file_name="pbisim_scenarios.json",
                mime="application/json",
                width="stretch",
                disabled=not _scenarios,
            )
            _up = st.file_uploader("Import scenarios (JSON)", type=["json"], key="sc_import")
            if _up is not None and st.button("Merge imported scenarios", key="sc_import_btn"):
                try:
                    imported = import_scenarios_json(_up.getvalue().decode("utf-8"))
                    _scenarios.update(imported)
                    st.session_state.user_scenarios = _scenarios
                    st.success(f"Imported {len(imported)} scenario(s).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    if _scenarios:
        st.markdown("#### Saved scenarios")
        for _name in list(_scenarios.keys()):
            _sc = _scenarios[_name]
            c_info, c_load, c_del = st.columns([6, 1, 1])
            with c_info:
                _note = _sc.get("annotation", "")
                st.markdown(f"**{_name}**" + (f" — {_note}" if _note else ""))
            with c_load:
                if st.button("Load", key=f"sc_load_{_name}"):
                    load_scenario_to_state(_sc["state"])
                    st.session_state._nav_to = "Interactive Simulator"
                    st.success(f"Loaded '{_name}'.")
                    st.rerun()
            with c_del:
                if st.button(":material/delete:", key=f"sc_del_{_name}"):
                    _scenarios.pop(_name, None)
                    st.session_state.user_scenarios = _scenarios
                    st.rerun()
    else:
        st.info("No saved scenarios yet — configure a simulation, then save it above.")

    # ── Parts (composable building blocks) ────────────────────────────────
    st.markdown("---")
    st.markdown("## Parts")
    st.caption(
        "Save individual **bacteria / phages / antibiotics** as reusable parts and compose "
        "them into any configuration. Loading a part adds it to the current strains / phages "
        "/ antibiotics (shared across all pages). Phage kinetics (burst / latent / adsorption) "
        "depend on the host, so phage parts record the **reference host** they were "
        "characterised against and warn if you reuse them elsewhere."
    )
    _lib = st.session_state.parts_library

    with st.expander("Export / Import parts library (JSON)"):
        _has_parts = any(_lib[c] for c in PART_CATEGORIES)
        st.download_button(
            "Export parts (JSON)", data=export_parts_json(_lib),
            file_name="pbisim_parts.json", mime="application/json",
            width="stretch", disabled=not _has_parts,
        )
        _pup = st.file_uploader("Import parts (JSON)", type=["json"], key="parts_import")
        if _pup is not None and st.button("Merge imported parts", key="parts_import_btn"):
            try:
                _imported = import_parts_json(_pup.getvalue().decode("utf-8"))
                for _c in PART_CATEGORIES:
                    _lib[_c].update(_imported.get(_c, {}))
                st.session_state.parts_library = _lib
                st.success("Parts imported.")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")

    _part_tabs = st.tabs([PART_CATEGORIES[c]["label"] for c in PART_CATEGORIES])
    for _tab, _cat in zip(_part_tabs, PART_CATEGORIES):
        with _tab:
            _meta = PART_CATEGORIES[_cat]
            _singular = _meta["label"][:-1].lower() if _meta["label"].endswith("s") else _meta["label"].lower()
            _entities = st.session_state.get(_meta["key"], [])
            _store = _lib[_cat]

            with st.expander(f"+ Save a current {_singular} as a part", expanded=not _store):
                if not _entities:
                    st.info(f"No {_meta['label'].lower()} configured yet — set one up in the Interactive Simulator first.")
                else:
                    _names = [f"{i}: {e.get('name', _singular)}" for i, e in enumerate(_entities)]
                    _pick_label = st.selectbox(f"Which {_singular}?", _names, key=f"part_pick_{_cat}")
                    _pick = _names.index(_pick_label) if _pick_label in _names else 0
                    _pname = st.text_input("Part name", value=_entities[_pick].get("name", _singular), key=f"part_name_{_cat}")
                    _psrc = st.selectbox("Source (provenance)", PART_SOURCES, key=f"part_src_{_cat}")
                    _pnote = st.text_area(
                        "Annotation", key=f"part_note_{_cat}",
                        placeholder="e.g. PA clinical isolate; high persister fraction",
                    )
                    _pref = ""
                    if _cat == "phages":
                        _sn = [s.get("name", "") for s in st.session_state.get("int_strains", [])]
                        _pref = st.selectbox(
                            "Reference host (bacterium it was characterised against)",
                            _sn + ["(unspecified)"], key=f"part_refhost_{_cat}",
                            help="Burst/latent/adsorption are phage×host properties — record the host so reuse elsewhere is flagged.",
                        )
                        _pref = "" if _pref == "(unspecified)" else _pref
                    if st.button("Save part", key=f"part_save_{_cat}", width="stretch"):
                        _nm = (_pname or "").strip()
                        if not _nm:
                            st.error("Please enter a part name.")
                        else:
                            _entry = {
                                "source": _psrc,
                                "annotation": _pnote or "",
                                "params": _json_safe(copy.deepcopy(_entities[_pick])),
                            }
                            if _cat == "phages":
                                _entry["reference_host"] = _pref
                            _store[_nm] = _entry
                            st.session_state.parts_library = _lib
                            st.success(f"Saved {_singular} part '{_nm}'.")
                            st.rerun()

            if not _store:
                st.caption("No saved parts yet.")
            for _pn in list(_store.keys()):
                _p = _store[_pn]
                _ci, _cl, _cd = st.columns([6, 1, 1])
                with _ci:
                    _bits = [f"**{_pn}**", f"`{_p.get('source', '?')}`"]
                    if _cat == "phages" and _p.get("reference_host"):
                        _bits.append(f"· host *{_p['reference_host']}*")
                    if _p.get("annotation"):
                        _bits.append("— " + _p["annotation"])
                    st.markdown(" ".join(_bits))
                with _cl:
                    if st.button("Load", key=f"part_load_{_cat}_{_pn}"):
                        _cur = list(st.session_state.get(_meta["key"], []))
                        if len(_cur) >= _meta["max"]:
                            st.warning(f"At most {_meta['max']} {_meta['label'].lower()} are supported — remove one first.")
                        else:
                            _cur.append(copy.deepcopy(_p["params"]))
                            st.session_state[_meta["key"]] = _cur
                            clear_entity_widgets()
                            _kind, _msg = "success", f"Added '{_pn}' to the configuration."
                            if _cat == "phages" and _p.get("reference_host"):
                                _sn = [s.get("name", "") for s in st.session_state.get("int_strains", [])]
                                if _p["reference_host"] not in _sn:
                                    _kind = "warning"
                                    _msg = (f"Added '{_pn}', but it was characterised against "
                                            f"'{_p['reference_host']}', which isn't among your current "
                                            "strains — verify burst/latent/adsorption for your host.")
                            st.session_state._flash = {"kind": _kind, "msg": _msg}
                            st.session_state._nav_to = "Interactive Simulator"
                            st.rerun()
                with _cd:
                    if st.button(":material/delete:", key=f"part_del_{_cat}_{_pn}"):
                        _store.pop(_pn, None)
                        st.session_state.parts_library = _lib
                        st.rerun()


# ── Calibration Page (Phase A: data upload + overlay + fit metric) ────────────
