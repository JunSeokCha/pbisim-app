# CLAUDE.md — pbisim-app session context

This file is read at the start of every Claude Code session in this repository.

> **Workspace context:** `pbisim-app` is one of three packages in the
> phage-therapy-sim workspace. **Read the root
> [`../CLAUDE.md`](../CLAUDE.md) first** for workspace-wide conventions, and
> [`../ECOSYSTEM.md`](../ECOSYSTEM.md) for architecture and the cross-package API
> contracts. This file covers only pbisim-app specifics.

---

## Project identity

**pbisim-app v0.1.0** — AI-powered (Anthropic Claude) Streamlit interface for the
`pbisim` phage-bacteria simulation engine. Users can explore the engine interactively,
run parameter sweeps, design clinical trials, and ask an AI assistant to build and
explain simulations in natural language.

**Status:** active development (**orchestrator owns this repo** — antigravity built
the initial scaffold; API wiring requires engine-author oversight). **56 tests passing.** Depends on `pbisim>=1.0` (and,
optionally and not-yet-wired-up, `pbisim-fit>=0.1` — see §5.3 in ECOSYSTEM.md).

---

## Repository layout

```
pbisim-app/
├── pbisim_app/
│   ├── __init__.py          package docstring / launch instructions
│   ├── app.py     (~3640)   Streamlit UI — 6 pages (see below), custom CSS,
│   │                        self-healing AI loop, all simulation and trial logic.
│   ├── agent.py   (~126)    SimulationAgent (wraps anthropic.Anthropic),
│   │                        AgentResponse, _parse_response() (regex extraction).
│   ├── executor.py(~155)    execute_code(): sandboxed namespace, exec()s
│   │                        generated code, captures stdout + matplotlib figures.
│   ├── sweep_helper.py      ModelConfig mutation helpers for 1D/2D parameter sweeps
│   │                        and dose-response sweeps; vector padding for MOI sweeps.
│   ├── trial_helper.py      IIV distribution factory, ClinicalTrial orchestration,
│   │                        Plotly KM + box plots, metric dataframe helpers.
│   └── fit_helper.py        Calibration/Phase-A: observable registry (CFU/PFU/OD/lum),
│                            data ingestion (normalize→pbisim-fit long format), overlay/RMSE.
├── prompts/
│   └── system_prompt.md     ~400-line hand-maintained pbisim API reference for the
│                            AI assistant. Sections 1–11 original; §12–17 added
│                            2026-06-23 covering new pbisim features.
├── tests/
│   ├── test_agent.py        5 tests — response parsing
│   ├── test_executor.py     10 tests — sandbox, capture, security, errors
│   ├── test_builder_modes.py 8 tests — BRG, StrainSet, cohort, phage-leak + immune + prerun guards
│   ├── test_sweeps.py       4 tests — sweep_helper parameter application
│   ├── test_self_healing.py 2 tests — self-healing loop and history rollback
│   ├── test_trial_features.py 5 tests — dose regimens, multi-arm, metrics, PK/PD trajectories
│   ├── test_scenarios.py     2 tests — scenario save/load round-trip (AppTest)
│   ├── test_parts.py         2 tests — parts save/load/export, host-tag (AppTest)
│   └── test_fit_helper.py    6 tests — observable registry, ingestion, overlay/RMSE
│   (test_system_prompt_sync.py  — sync guard, run after pbisim API changes)
└── pyproject.toml           entry point: pbisim-app = "pbisim_app.app:main"
```

---

## App pages

| Page | Description |
|---|---|
| Interactive Simulator | Three builder modes (Direct/ModelBuilder, Binary Genotypes/BRG, Custom Strains/StrainSet). Repeat-dosing regimen builder. Run button → plots + metrics. |
| Dose-Response Sweeps | Log/Lin dose range per agent, MOI scaling, vector padding warnings, color-coded trajectories. |
| Parameter Sweeps | 1D/2D sweeps over any ModelConfig field. Contour maps for 2D. n_depth resizing guard. |
| Clinical Trials & Cohorts | Full ClinicalTrial API integration: IIV, PretreatmentPhase, parallel arms, KM plots, metric distributions, CSV/NLME export. |
| Calibration | Upload experimental CSV → normalize to pbisim-fit long format (auto-detect + Monolix column-map) → **filter rows**, **regroup** by chosen variables, aggregate replicates (**raw / mean / median + percentile band**) → overlay the current model vs observations (group multiselect) with live RMSE. Extensible observable registry (CFU/PFU/OD/luminescence). Phase A of the pbisim-fit integration. |
| AI Assistant | Natural-language → pbisim code. Self-healing loop (up to 3 retries with history rollback). Dynamic model listing from `/v1/models`. |
| Library | Two sections: **💾 Scenarios** (save/load full-config snapshots) and **🧬 Parts** (composable bacteria/phages/antibiotics — save a current entity, load into config, host-tagged phages); each export/import as versioned JSON. (Tutorial presets + `presets.py`/`test_presets.py` removed 2026-07-10 — they tracked the pbisim tutorials, which change independently.) |

---

## How the AI assistant works (data flow)

```
Streamlit UI (app.py)
   → SimulationAgent.ask(user_message)                     agent.py
   → Claude (model dynamically selected), system = prompts/system_prompt.md
   → _parse_response() → AgentResponse(code, narrative, assumptions, raw_text)
   → executor.execute_code(code) → ExecutionResult(success, figures, stdout, error)
     ↓ on failure (up to 3 times):
     → agent.ask(traceback) → corrected code → execute_code()
     → on final failure: history rolled back to pre-request state
   → UI renders narrative + figures (+ optional code/assumptions/stdout)
   → conversation history kept in agent.history (multi-turn)
```

- **No tool-use, no streaming.** Claude returns markdown; code is pulled from a
  ```` ```python ```` block by regex (`agent.py:_parse_response`).
- **API key:** `ANTHROPIC_API_KEY` env var, else Streamlit sidebar password input.
- **Model:** dynamically selected in sidebar; default pin is `claude-sonnet-4-6`.

---

## Package-specific conventions

1. **The system prompt is the contract with the engine.** `prompts/system_prompt.md`
   hard-codes `pbisim` signatures — it is NOT generated. If the `pbisim` public
   API changes, update this prompt in lockstep (coordinate via the integration
   role). Never let the prompt invent method names.
2. **Generated code may use either pre-loaded sandbox names OR `import` statements.**
   The executor sandbox (`executor.py`) pre-loads: `np/numpy`, `plt/matplotlib`,
   `ModelBuilder`, `PBIModel`, `solve_ode`, `DoseSchedule`, `DoseEvent`, `StrainSet`,
   `StrainDefinition`, `BinaryResistanceGenotypes`, `BacterialStrain`, `PhageStrain`,
   `Antibiotic`, `PhagePKConfig`, `AntibioticDefinition`, `AntibioticSensitivity`,
   `stationary_phase_ic`, `time_to_clearance`, `time_to_log_reduction`, and (if
   pbisim.trial is installed) `TrialRunner`, `IIVSpec`, `VirtualPopulation`,
   `default_metrics`, `LogNormal`, `Normal`, `Uniform`, `Fixed`, `ClinicalTrial`,
   `TreatmentArm`, `PretreatmentPhase`. Add to the namespace AND the prompt
   together when new surface is needed.
3. **Honor the shared engine contracts** (see ECOSYSTEM.md §3.2): the prompt must
   keep instructing the model to use `result.sum_prefixes("B","D","I","H")` for
   CFU, `result.get('B0')` for individual series, `np.maximum(x, 1.0)` before
   `log10`, and to set `initial_S` on `PBIModel` (not `solve_ode`).
4. **Resistance seeding rule** (in the prompt): never start the resistant strain
   at exactly 0 — use `initial_B[resistant] = max(mutation_rate * initial_B[0], 10.0)`,
   because mutation flux → 0 when growth → 0 under nutrient limitation.
5. **Sandbox is research-grade only.** It strips `open/exec/eval/compile` and
   captures stdout/figures, but is not safe for untrusted/public input. Do not
   weaken it; do not deploy publicly without real isolation (Docker /
   RestrictedPython).
6. **Never commit API keys.** Keys come from env or UI at runtime only.
7. **`sweep_helper.py` pk_array1d** always targets the antibiotic `PKConfig` — never
   `phage_pk_config`. Both are on `ModelConfig` and must not be confused.

---

## Test commands

Activate the project env first (`conda activate pbisim`), then:

```bash
# Full suite — expected: 56 passed
python -m pytest tests/ -q

# By file:
python -m pytest tests/test_executor.py -q        # 10 tests
python -m pytest tests/test_agent.py -q           # 5 tests
python -m pytest tests/test_builder_modes.py -q   # 8 tests (BRG, StrainSet, cohort, phage-leak + immune + prerun)
python -m pytest tests/test_sweeps.py -q          # 4 tests
python -m pytest tests/test_self_healing.py -q    # 2 tests
python -m pytest tests/test_trial_features.py -q  # 5 tests (dose regimens, multi-arm, metrics, PK/PD trajectories)
```

After any pbisim API change, also run:
```bash
python -m pytest tests/test_system_prompt_sync.py -v
```

## Running the app

```bash
python -m streamlit run pbisim_app/app.py
# or, once installed:  pbisim-app
```

---

## Known gaps / next steps

- **pbisim-fit integration is deferred by design** until pbisim-fit v1.0 ships;
  the `[fit]` extra is a documented placeholder. Intended hook:
  `pbisim_fit.output.to_model_config` (ECOSYSTEM.md §3.5, §5.3).
- No streaming, no tool-use, brittle regex parsing, no rate limiting, no UI tests.
- Model pin lives in `agent.py` `_MODEL` (currently `claude-sonnet-4-6`); keep it
  current and re-run `tests/test_system_prompt_sync.py` after any pbisim upgrade.
- Preset `script_code` strings for type="single" presets (01–10, 13) are reference
  only — they are not executed. Any API mismatch there is cosmetic but should be fixed.

## Done this session (2026-06-23 continued)

- **Adsorption default 2e-9 → 1e-8** for WT phage in all preset parameter dicts and
  script_code strings (`presets.py`); list first-elements updated per-strain; collateral-
  sensitivity preset ([1e-9, 5e-9]) left intact; IIV distribution lines left intact.
- **`system_prompt.md`** example updated: `adsorption_s=2e-9` → `adsorption_s=1e-8`.
- **`tests/test_presets.py`** fallback default updated: `2e-9` → `1e-8`. 48 tests pass.
- **`USER_GUIDE.md`** written: task-oriented GUI guide (12 sections, 5 worked workflows,
  parameter reference tables, troubleshooting). See `USER_GUIDE.md` at repo root.
- **`README.md`** written: repo front door with features, install, run/stop, pages table,
  doc links, test command. See `README.md` at repo root.
- **Immunity module full audit and fix** (commit `a397edd`):
  - Preset Tutorial 04: removed `adaptive_stimulation_rate/decay_rate/max/delay`
    (invented by scaffold, not in pbisim); added `immune_module`, `imm_stim_rate`,
    `imm_stim50`, `innate_decay_rate`, `imm_initial` with correct pbisim names.
  - `load_preset_to_state()`: backward-compat translation of `adaptive_decay_rate` →
    `innate_decay_rate`, `"adaptive"` module → `"innate"`; new keys `int_imm_stim_rate`,
    `int_imm_stim50`, `int_imm_initial`.
  - Direct / BRG / StrainSet simulation paths: all now pass `imm_stim_rate`,
    `imm_stim50`, correct `immune_module`, and `initial_Imm` (was inadvertently set
    to `imm_max` value).
  - Immunity UI (Tab 2): module selector fixed to `["innate", "hill"]` (removed invalid
    `"adaptive"`); `imm_stim_rate`, `imm_stim50`, `initial_Imm` widgets added; `imm_max`
    made conditional on hill module.
  - Repro code (Direct): updated with all new immunity parameters.
  - Repro code (BRG): was entirely missing immunity; now emits `brg.to_config(...)` with
    all immunity kwargs when enabled.
  - Repro code (StrainSet): was hardcoded `imm_stim_rate=0.0, imm_kill_rate=0.0`; now
    reads session state; `ss.to_config()` now includes immunity params, actual phage
    decay rates, and correct `n_depth`.
- **Clinical trial Combo arm guard** (same commit): `st.warning` emitted when Combo arm
  contains only phage or only antibiotic doses (Combo = monotherapy → identical KM
  curves). Explains the "strange outputs" the user observed.

## Done this session (2026-07-16) — OD trajectories in parameter sweep

- **Parameter sweep now plots OD trajectories** when the OD/debris module is enabled
  (previously only the dose-response sweep did). Added `od_trajectories` collection to
  the 1D and Coupled compute loops (`result.get_od()`), stored in `param_sweep_result`,
  and an "Optical Density" chart rendered after the CFU trajectories in both. Omitted
  when debris is off; 2D (heatmaps only) is unaffected. Test in `test_sweeps.py`.
  **73 tests pass.**

## Done this session (2026-07-16) — mutation-rate persistence + coupled/broadcast sweeps

- **Direct-mode mutation rate reverted to 1e-7 on navigation.** The 2^m-shortcut
  mutation widget (`direct_mu_{j}`) read its `value=` from `direct_phg_res_mu_{j}`
  (sic — `direct_phg_mu_{j}`), a key that was NEVER written; the actual value lives in
  `direct_phg_res_rates`. So when the widget key was dropped on navigation the input
  re-defaulted to 1e-7 and overwrote a user's 0. Fixed to seed `value=` from the
  persisted `direct_phg_res_rates` list. Regression test in `test_builder_modes.py`.
- **Sweeping shared / linked parameters** (parameter-sweep feature):
  - **Broadcast params** (`sweep_helper`): `get_sweep_parameters` now emits
    `… (ALL strains)` / `(ALL phages)` entries (types `array1d_broadcast` /
    `array1d_broadcast_or_none` / `initial_B_broadcast`) that apply one value to every
    strain/phage — e.g. sweep a growth rate shared by WT + mutant with one control.
  - **Coupled (linked) sweep**: a third `ps_sweep_type` mode. Pick several parameters
    (multiselect `pc_labels`) and give each a value series of equal length
    (`pc_series_{i}`); at step k, value[k] of every parameter is applied together —
    e.g. (dormancy, resuscitation) = (0,1),(0.5,0.5),(1,0). Summary has a column per
    parameter; trajectories labelled by the tuple; metrics vs step index. Controls +
    result persist across navigation (added `pc_` to the persist prefixes).
  - Tests: `test_sweeps.py` (broadcast unit + coupled AppTest + mismatch guard).
- **72 tests pass.**

## Done this session (2026-07-15) — sweep CONFIG persistence + zero-dose default

- **Sweep controls now survive navigation** (previously only the *results* did). The
  dose-response `dr_sweep_*` widgets and the parameter-sweep `p1_*`/`p2_*`/`ps_*`
  widgets got dropped on navigation (Streamlit discards un-rendered widget keys).
  Added reusable `reseed_widget_config()` / `save_widget_config()` (shadow the
  selections into `dr_sweep_config` / `param_sweep_config` plain dicts, re-seed before
  render). For the parameter sweep the previously-unkeyed `sweep_type` (radio) and the
  1D `ps_1d_min/max/steps/spacing` inputs were keyed so they can persist; the 1D range
  widgets re-autoscale when the swept parameter changes (`_ps_1d_last_param` guard pops
  them so a persisted key doesn't pin a stale range).
- **Dose-response default phage series now `0, 1e3, 1e5, 1e7, 1e9`** (adds the zero-dose
  control). Test `test_dose_response_shows_od_trajectories_when_enabled` updated.
- Test: `test_sweeps.py::test_sweep_configs_survive_navigation`. **69 pass.**

## Done this session (2026-07-15) — pre-run collapse, sweep persistence, BRG calibration clash

- **Pre-run made CFU/OD scale very low (reported as a dose-response OD bug).** Root
  cause: `stationary_phase_ic` applies `death_rate_B` throughout the pre-run, but once
  nutrients exhaust growth stops — so with a death rate and no dormancy the culture
  declines for the whole pre-run and the treatment starts from a decimated inoculum
  (reproduced: death 0.5 → OD 1.16 no-prerun vs 0.03 after 24 h pre-run). It is the
  engine's documented behavior, not an app bug, but was silent. Added
  `prerun_collapse_fraction()` / `warn_if_prerun_collapses()` and surfaced the warning
  in the main sim run, the dose-response sweep and the parameter sweep when the pre-run
  leaves <10% of the inoculum (B+D). (Dormancy avoids it — persisters survive.)
- **Dose-response + parameter sweep results now survive navigation.** Both drew results
  inside the Run button's `if`, so navigating away wiped them. Split compute-from-render:
  the button stores results in `dr_sweep_result` / `param_sweep_result`; a separate block
  renders from the stored data every run (alive until re-run).
- **Calibration BRG clash + missed params** (see the 2026-07-15 tuning entry below for
  the panel details): BRG strain kinetics live on `int_brg_base_*` (not the per-strain
  dicts), so BRG strain edits were silent no-ops — the panel is now mode-aware. Mutation
  rate (μ) + fitness cost added per phage in BRG; dormant adsorption + adsorption_r added.
- Tests: `test_sweeps.py` (+2 persistence, +1 collapse warning), plus the earlier
  calibration/fit_helper additions. **68 pass.**

## Done this session (2026-07-15) — OD-debris propagation, structural/global tuning, save-calibrated

- **OD/debris now propagates into the Calibration overlay.** The overlay always
  simulated *with* debris (build_nominal includes it), but the OD *observable* was
  computed as biomass/link and ignored the debris state. `predicted_observable()`
  gained `use_model_od`: when the OD/debris module is on, OD comes from the model's
  debris-inclusive `result.get_od()` (uses `od_to_cfu_conversion_factor`) instead of
  the simple biomass/link scaling. The overlay's link input is then replaced by a note
  pointing to the OD/debris params in the tuning panel.
- **Manual calibration now exposes global & structural parameters** (a "Global &
  structural" block in the tuning panel): n_latent (latent compartments), K, Ks;
  nutrient S₀ / recycle / s_in / s_out (when nutrient tracking is on); the OD/debris
  params (od_to_cfu, u, v, k_dis) when the module is on; and per-strain dormancy depth
  (Q) added to the dormancy row. All edit the live `int_*` session keys directly.
- **Save the calibrated model** (section 6): edits are already live in the Interactive
  Simulator (same dicts), so "apply to builder" is automatic; added a **💾 Save
  calibrated config as Scenario** button (reuses `dump_state_to_scenario`) to persist
  the whole config to the Library for reload. `fit_save_scenario` (a button) is in
  `_FIT_NOPERSIST` — persisting a button key pre-sets it and makes `st.button` raise;
  the fit_config re-seed loop now also scrubs stale non-persistable keys.
- Tests: `test_fit_helper.py` (+1 debris-OD), `test_calibration.py` (+1 globals/debris/
  save-scenario). **66 pass.**

## Done this session (2026-07-15) — overlay persistence, builder-mode param parity, fuller tuning

- **Calibration overlay now survives navigation** (item 1): the overlay was drawn
  inside `if st.button(...)`, so leaving the page and returning (without re-clicking)
  wiped the plot. Split compute-from-render: the button computes and stores the plot
  data in `st.session_state.calib_overlay_result`; a separate block renders it on
  every run while present. Stays alive until re-run or the dataset is cleared.
- **Builder-mode parameter parity** (item 2): closed the "read-but-not-editable" gaps
  found by auditing all three builder UIs vs `build_nominal_config_from_gui`:
  - `bacteria_to_resource_ratio` — was read (default 1e9) but had no widget in
    **Direct** (`str_ratio_{i}`) or **StrainSet** (`ss_str_ratio_{i}`); added to both
    (BRG already had `int_brg_base_ratio`).
  - Pseudolysogeny (`hibernation_rate_s/r`, `lytic_resumption_rate_s/r`) — read into
    `PhageStrain` but no widgets in **BRG** (`brg_phg_{hib,res}_{s,r}_{idx}`) or
    **StrainSet** (`ss_phg_...`); added to both (Direct already had them).
- **Manual calibration exposes the fuller parameter set** (item 3): `STRAIN_TUNABLES`
  now growth / bacteria_to_resource_ratio / death_rate_B / initial_B; a conditional
  `STRAIN_DORMANCY_TUNABLES` row (dormancy/resuscitation/diffusion/dormant-death) shown
  per strain when dormancy is on; `PHAGE_TUNABLES` adds phage_decay_Km + attenuation_rate
  (on top of burst/latent/decay + the mode-aware adsorption editor).
- Tests: `test_calibration.py` (+1 overlay persistence), `test_builder_modes.py`
  (+1 all-mode ratio/pseudolysogeny), `test_fit_helper.py` registry update. **64 pass.**

## Done this session (2026-07-15) — dose-response zero-dose phage leak

- **Dose-Response Sweeps: a swept phage dose of 0 still suppressed the bacteria.**
  Same class of bug as the clinical-trial phage-leak fix: the model always seeds
  phage from the phage config's `initial_P` (default 1e6) at t=0, independent of the
  dose. The sweep overrides only the dose *events*, so `dose=0` still started with
  1e6 PFU and crushed the culture. Fix (`app.py`, Dose-Response Sweeps page): for the
  duration of the sweep, zero the swept phages' `initial_P` (saved/restored in the
  `finally` alongside `int_doses`) so the swept dose fully controls phage exposure;
  `dose=0` now means no phage. Antibiotics need no analog (no baseline inoculum — all
  via dose events). Verified: `dose=0` nadir stays ~1e7, `dose=1e8` eradicates.
  Regression test `tests/test_sweeps.py::test_dose_response_zero_phage_dose_does_not_suppress`.
  **61 tests pass.**

## Done this session (2026-07-14) — Calibration Phase B + config persistence

- **Calibration config now survives navigation** (commit `07315d1`): Streamlit
  drops a widget's key from `session_state` when the widget isn't rendered on a
  rerun, so leaving Calibration to change the model reset all the filter/grouping/
  statistic selections. Fix: shadow the `fit_*` widget selections into a plain
  `fit_config` session key (survives) and re-seed the widgets from it at the top of
  the page, before they render. Buttons + the file-uploader are excluded from the
  shadow (they can't be re-seeded). Regression test `tests/test_calibration.py`.
- **Phase B — manual parameter tuning** on the Calibration page (section 5, a
  "🎛 Manual parameter tuning" expander). Edits the model's **actual absolute
  parameter values** (like the ModelBuilder), *not* multipliers — the widgets read
  from / write to the live `int_strains`/`int_phages` dicts and session keys, so an
  edit **is** the model (no separate apply step) and is savable as a Part.
  - Exposed: global K / Ks; per strain growth_rate + initial_B; per phage burst /
    latent / decay; and **adsorption**, which is builder-mode specific — Direct &
    Custom-Strains store it in the pairwise `ads_{i}_{j}` session keys (edited per
    strain×phage pair), Binary-Genotypes on the phage dict `adsorption_s`. The link
    factor (od_to_cfu / rlu_per_cell) is keyed per-observable (`fit_link_{obs}`).
  - `fit_helper.STRAIN_TUNABLES`/`PHAGE_TUNABLES`/`ADSORPTION_PHAGE_KEYS` +
    `entity_param_key()` (mode-aware key resolution). Pure/no-Streamlit → unit-tested.
  - `fit_edit_*` widgets are excluded from the `fit_config` shadow (the dicts are
    authoritative + already persistent; re-seeding a stale copy would clobber edits).
  - Tests: `test_fit_helper.py`, `test_calibration.py`. **60 tests pass.** (Earlier
    in this session an initial ×multiplier version shipped, then was replaced per
    owner feedback: multipliers were inconvenient and silently missed Direct-mode
    adsorption, which lives in `adsorption_rates`/`ads_{i}_{j}`, not `adsorption_s`.)

**Known gaps / still open after this session:**
- CS/CR (26 `cr_*` fields on `PhageStrain`/`BacterialStrain`) not exposed in BRG UI —
  complex feature, deferred. Document in USER_GUIDE as "advanced scripting only".
- `immune_module="custom"` has no UI support — deferred.
- StrainSet repro code: `n_depth` calculation uses `dormancy_depth` key which may
  differ from actual session state key; edge case, not a blocker.
- Test count: 48.

## Done this session (2026-07-07) — clinical trial phage-leak fix

- **CRITICAL BUG FIXED — Control arm was secretly receiving the phage inoculum**
  (`app.py`, Clinical Trials & Cohorts page). Root cause: the crossover
  `ClinicalTrial` shares `initial_conditions` across all arms and differentiates
  them only by `dose_schedule` (`ClinicalTrial._apply_arm` never touches ICs). The
  app seeded phage via `initial_P` on the shared `base_cfg.initial_conditions.P`, so
  **every** arm — Control and Antibiotic-Only included — started with the full phage
  inoculum (default `1e6`/mL), which eradicated the bacteria in ~6 h. Every arm
  looked identical and the untreated control "cured" all patients (the strange
  outputs the user reported).
  - **Fix:** phage is the intervention, so it is now delivered per-arm. The trial
    starts every arm at **zero free phage** (`base_P = np.zeros_like(init_P)`) and
    injects the configured inoculum as a **t=0 phage bolus** only into the Phage-Only
    and Combo arms (verified numerically identical to seeding `initial_P`). Control
    and Antibiotic-Only are now genuine no-phage arms.
  - Arm-existence guards + Combo "identical to monotherapy" warning updated to treat
    the inoculum as phage presence (`_has_phage = inoculum or nominal phage dose`).
  - `base_initial_P` passed to `run_trial_simulation` also zeroed (model-factory
    fallback consistency).
  - Verified post-fix (30 patients, IIV on growth): Control 0/30 eradicated
    (bacteria → ~1e9), Antibiotic-Only 1/30 (weak monotx), Phage/Combo 30/30.
- **Regression test added**: `test_builder_modes.py::test_trial_control_arm_has_no_phage`
  — asserts Control keeps `P≈0` throughout and bacteria grow, while the phage arm
  delivers phage and eradicates. **49 tests passing** (was 48).

**Known limitation introduced:** the IIV option "Initial Phage Density" (`ic.P`) now
acts on a zeroed baseline in trials, so it no longer varies the inoculum — phage comes
from the dose. If per-patient phage-dose variability is wanted, wire IIV to the dose
amount (small follow-up).

## Done this session (2026-07-07) — immune module investigation

- **Investigated "immune module doesn't work" reports. The immune module is NOT
  broken** — innate immunity kills active bacteria (both strains) correctly in all
  three builder modes and the exact app solve path (BDF + extinction_threshold). Two
  real issues found:
  1. **Dormancy is an immune refuge (root cause of the user's 2-strain report).**
     Reproduced exactly (2 strains, dormancy on, defaults, immunity on → nadir 1.57e6,
     never clears, resistant fraction ~99.7% — matches user's numbers). Mechanism:
     immunity crushes the *active* resistant strain (B1: 4.7e8→2.9e6) but the resistant
     population mass-converts to dormancy when nutrients crash, and dormant/hibernating
     cells are **immune-privileged** — the engine neither kills them (`imm_kill_rate_D`
     defaults to 0/None) nor lets them stimulate immunity (excluded from
     `bac_total_active`). So a ~3.6e8 dormant reservoir persists and the infection never
     clears. Confirmed fix path: `imm_kill_rate_D > 0` drops the burden to 1.8e5. This
     is correct model biology (persister immune evasion), but the app gave no hint.
     **→ Added a UI warning** in the immunity tab (Tab 2) when dormancy + immunity are
     both on and `imm_kill_rate_D <= 0`, explaining the refuge and pointing to the
     control. No biology/default changes. Regression test
     `test_builder_modes.py::test_dormancy_creates_immune_refuge`. **51 tests passing.**
  2. **The `hill` immune module — FIXED this session.** Hill killing is
     `imm_max·B/(imm_kill50+B)` with `Imm` frozen, so it **ignores** `imm_kill_rate`,
     `imm_stim_rate`, `imm_decay_rate`, `initial_Imm` (verified: 1e7→1e13 on
     `imm_kill_rate` changes nothing). Previously `imm_max=1e7` did nothing at
     `imm_kill50=1e8`; the owner's `imm_kill50` 1e8→1e5 change (see below) now makes
     hill effective at the default `imm_max=1e7` (OFF→1e9, ON→0). Remaining defects
     were UX/docs: (a) the module-selector help described a stimulation formula the
     engine doesn't implement — corrected; (b) the immunity tab showed the inert
     innate-only fields in hill mode — now hill renders only `imm_max` + `imm_kill50`
     (+ `imm_kill_rate_D`) with a caption explaining `Imm` is frozen. Regression test
     `test_hill_immune_module_active_at_app_defaults` (hill controls the bloom at
     defaults; output is invariant to the inert fields).

- **Fixed blank "Antibiotics & Host Immunity" results graph.** When immunity was on
  but no antibiotic was configured, the plot did `ax2 = ... else plt.subplots(...)[1]`,
  drawing `Imm` onto a throwaway figure while `st.pyplot(fig)` showed the empty
  original — so the graph was blank. Now plots `Imm` on `ax1` (the displayed figure)
  in the no-antibiotic case. (User's reported symptom.)
- **Updated default innate immune parameters** (per owner): `imm_stim_rate` 1.0→**0.1**,
  `imm_decay_rate` 0.05→**0.1**, `imm_kill50` 1e8→**1e5**; `imm_kill_rate` (1e7) and
  `imm_stim50` (1e6) unchanged. Applied across session defaults, all three builder
  paths, UI widgets, preset-load, and repro-code.
- **Exposed `extinction_check_interval`** (pbisim solver arg) as a new "Extinction
  check interval (hours)" UI control (0 = check only at dose boundaries). Wired into
  the main sim, 1D/2D sweep solves, repro-code, and preset-load. Zeroes a sub-threshold
  strain at the chosen cadence so it can't regrow from a below-threshold pool (verified:
  a 3.4-cell residual regrows to 9.35e8 with interval off, stays extinct at interval=6h).

- **Fixed the `hill` immune-module UX/docs** (see #2 above): corrected the
  module-selector help text and hid the inert innate-only fields in hill mode.

**Still open (immune):** none outstanding.

## Done this session (2026-07-07) — stationary-phase pre-run fix

- **Fixed: long pre-run collapsed the treatment to a flat 0 CFU curve.**
  `run_sim_from_gui_params` (and the 1D/2D sweeps + repro-code) equilibrated with
  `stationary_phase_ic` but kept only `ic.B` and `ic.S`, **discarding the dormant
  reservoir `ic.D` and immune priming `ic.Imm`**. At stationary phase most of the
  culture is dormant, so the longer the pre-run the more cells were silently thrown
  away — the surviving active `B` shrinks with pre-run length until it drops below the
  extinction floor and the treatment plots as flat 0 (reproduced: with dormancy, active
  `ic.B` falls 2.6e8→2.0e5 as t_prerun 12→48 while `ic.D` grows to ~1e9; treatment peak
  collapses 2.6e8→2.0e5). Fix: carry the full stationary state — `initial_D = ic.D`,
  `initial_Imm = ic.Imm` — into the treatment `PBIModel`; treatment now starts at the
  full ~1e9 population for any pre-run length.
- Also: removed the bogus `S0=` kwarg passed to `stationary_phase_ic` (no such
  parameter — it was silently forwarded to scipy and ignored with a warning), and
  clamped the carried nutrient `initial_S = max(ic.S, 0)` (the pre-run can leave S
  slightly negative numerically). Applied in the main sim, 1D + 2D sweeps, and the
  auto-generated reproduction code. Regression test
  `test_prerun_carries_dormant_reservoir`. **52 tests passing.**

- **Fixed the same drop in the clinical-trial `PretreatmentPhase` path**
  (`trial_helper.create_model_factory`). The factory read `ic.B/P/S` from the
  per-patient config but took `initial_D`/`initial_Imm` from the GUI base kwargs, so a
  trial pretreatment discarded its dormant reservoir + immune priming. Now the factory
  prefers `config.initial_conditions.D`/`.Imm` (set by `PretreatmentPhase`) and clamps
  `initial_S = max(ic.S, 0)`. Verified: a 48 h pretreatment on a dormancy model now
  starts every patient's treatment at ~1e9 (was collapsing). Regression test
  `test_trial_pretreatment_carries_dormant_reservoir`. **53 tests passing.**

**Still open (pre-run):** none outstanding.

## Done this session (2026-07-08) — trial dosing, PK/PD outputs, builder consistency

Six owner-requested feature updates (all UI-verified via streamlit AppTest: every page
+ builder mode renders and a full trial runs with 0 exceptions):

1. **Antibiotic-aware default dose** (Environment & Dosing): dose-amount default is now
   target-specific — phage 1e8 PFU, antibiotic 10 mg, nutrient 1.0 — via
   `DOSE_AMOUNT_DEFAULTS`/`DOSE_AMOUNT_LABELS`; target selector moved before the amount
   (keyed per target). Applies to the single and repeat-regimen forms.
2. **`attenuation_rate` exposed** (phage config, all 3 builders): engine param
   "phage penetration decay with dormancy depth" — effective dormant adsorption =
   `adsorption_dormant × exp(−attenuation × depth)`. Per-phage input on the shared
   `phages` dict; wired into Direct (`with_phage_params`), BRG (`to_config`, broadcast),
   StrainSet (`StrainDefinition`), and all repro-code.
3. **Trial output dropdowns** (item 1c/1d): survival endpoint keeps tte + tt2lr;
   distribution-metric options are now Maximum Log Reduction, Log Reduction (baseline→last
   obs), Bacterial AUC, Nadir Count (time_to_clearance removed — it's a survival endpoint).
   New per-patient metrics `max_log_reduction` / `log_reduction_final` added to
   `trial_helper.trial_metric_fns()` and passed to `ClinicalTrial(metric_fns=...)`.
4. **Configurable n_latent + n_depth** (item 4): global **n_latent** control (Solver
   Settings → Model structure, `int_n_latent`) used by all three builders (replaced the
   hardcoded 5). **n_depth**: BRG gained a "Dormancy depth layers (Q)" control
   (`int_brg_n_depth`); StrainSet gained a per-strain "Depth layers (Q)" input (Direct
   already had one). All wired into build + repro.
5. **Trial dedicated dose editor + regimen** (item 1a): the Clinical Trials page now has
   its own "Trial Dosing Regimen" section — per-agent (phage/antibiotic) amount + start
   time + regimen (single, or repeat qX h × N) via `trial_helper.build_regimen_doses`.
   Arms (Control/Phage/Antibiotic/Combo) derive from it instead of the simulator's
   `int_doses`; the old init_P-inoculum injection is gone (the editor's t=0 dose is the
   inoculum). Base config still starts every arm at zero free phage.
6. **Trial raw PK/PD trajectories** (item 1b): per-arm median + IQR-band Plotly plots of
   total bacteria (CFU) and free phage (PFU) via `trial_helper.plot_pkpd_trajectories_plotly`,
   shown at the top of the trial outputs.

Regression tests in new `tests/test_trial_features.py` (regimen builder, distribution
metrics, PK/PD trajectory plots). **56 tests passing.**

**Note:** the Clinical Trials page no longer reads the simulator's Environment & Dosing
`int_doses` — trial doses come solely from the new Trial Dosing editor.

## Done this session (2026-07-08 cont.) — multi-arm trials + BRG fitness cost

- **Clinical trial now supports arbitrary named dose arms** (item follow-up: the fixed
  Control/Phage/Antibiotic/Combo checkboxes only gave one dose level). The Clinical
  Trials page has a **Treatment Arms builder**: add any number of named arms (e.g.
  "Low dose" / "High dose"), each with its own phage + antibiotic regimen (single or
  repeat qX h × N) via `render_regimen_config` / `arm_dose_events` (app.py) →
  `trial_helper.build_regimen_doses`. Arms stored in `st.session_state.trial_arms`; a
  separate "Include Control arm" toggle. Arm assembly builds one `TreatmentArm` per
  config with de-duplicated names; every arm still starts at zero free phage. Verified
  via AppTest (arms = Control / Low dose / High dose, 0 exceptions) and regression test
  `test_multiple_dose_arms_produce_distinct_outcomes` (higher dose → lower bacterial AUC,
  larger phage peak).
- **BRG fitness-cost default 0.0 → 0.05** (phage loci + antibiotics, append + preset-load
  defaults) with help text noting it drives the equilibrium IC. The engine already wired
  `fitness_cost` correctly (fc=0 → resistant neutral → resistant-dominated equilibrium;
  fc=0.05 → WT-dominated `[1e7, 20]`); the input existed but defaulted to 0, which
  produced the fully-resistant equilibrium the owner observed. **56 tests passing.**

**Note:** the old fixed-arm checkboxes (`run_control`/`run_phage`/`run_abx`/`run_combo`)
and the single Trial Dosing editor are gone — replaced by the Treatment Arms builder.

## Done this session (2026-07-10) — bugfixes + Tier-1 scenario library

Bugfixes (each committed separately, all verified via streamlit AppTest):
- **OD/debris crash (Direct mode):** the app passed debris params to
  `builder.build(**extra_kwargs)`, but `ModelBuilder.build()` takes no kwargs →
  `TypeError` whenever "Track Bacteriolytic Cell Debris" was enabled. Fixed to use
  `builder.with_od_debris(u, v, kdis, od_to_cfu_conversion_factor)`. BRG/StrainSet
  were fine (their `to_config` accepts `**extra_config_kwargs`).
- **Number-input precision:** Streamlit infers `"%.2f"` from the step, so 0.001 was
  rounded to 0.00 on entry. Global wrapper injects `format="%g"` for any float
  `st.number_input` without an explicit format (auto-skips ints / scientific).
- **BRG + StrainSet phage PK/advanced:** those two modes lacked the Advanced Phage
  Kinetics (`phage_decay_Km`) + Phage PK (`pk_mode`/Vc/k_elim/k_in/k_out/Vi/Km_elim)
  widgets Direct had (build paths already read them; UI was missing). Added widgets +
  `phage_decay_Km` forwarding for both.

**Tier-1 Scenario library (presets rework):** a "scenario" = the *entire* input
configuration (builder mode, strains/phages/antibiotics, pairwise adsorption, dosing,
nutrient, immune, debris, solver, prerun, trial arms/IIV). Implemented as a **session
snapshot** (`dump_state_to_scenario` captures all `int_*` + `ads_<s>_<p>` +
`direct_phg_res_rates` + `trial_*` keys; `load_scenario_to_state` clears widget keys
then restores) rather than an inverse of `load_preset_to_state` — so new params are
captured automatically and every builder mode is covered. UI in the **Presets &
Tutorials** page: Save current config, list/Load/Delete, and **Export/Import the whole
library as versioned JSON** (`schema_version=1`) — the portable "personal DB" that works
on the stateless deploy. Also fixed a **pre-existing navigation bug**: the keyed nav
radio overrode `current_page`, so Load buttons (scenario *and* tutorial) didn't switch
pages — now routed through a pending-`_nav_to` hop applied before the radio instantiates.
Round-trip verified (config + adsorption + builder mode restored, navigates, re-runs).
Tests: `tests/test_scenarios.py` (AppTest round-trip + JSON export).
- **Tutorial presets REMOVED** (owner's call): deleted `pbisim_app/presets.py` +
  `tests/test_presets.py`, dropped the tutorial-card browser, and renamed the page
  **"Presets & Tutorials" → "Scenarios"** (nav string + routing). A new inline
  `DEFAULT_SCENARIO` constant in `app.py` replaces `TUTORIALS[0]` as the fresh-session /
  Reset startup config, so the app no longer depends on the pbisim tutorials.
  **48 tests passing** (was 60; −12 removed preset tests).

**Design direction (agreed with owner):** presets are a two-tier model — **Tier 1 =
Scenarios (full-config snapshots, DONE)**; **Tier 2 = a composable Parts library**
(bacteria / phages / antibiotics). Phage kinetics (burst/latent/adsorption) are NOT
phage-intrinsic — they're phage×host pair properties — so Tier-2 phage parts will carry
a **reference-host tag** + soft "verify for this strain" flag, and map directly onto a
future `pbisim-fit` "fit → save part" pipeline. Persistence stays JSON export/import
(a real per-user DB needs auth+storage, deferred).

## Done this session (2026-07-13) — Tier-2 Parts library

- **Parts library** (composable building blocks) added to the **Library** page
  (renamed from "Scenarios"; now has two sections: 💾 Scenarios + 🧬 Parts).
  A part = one reusable entity (bacterium / phage / antibiotic) = its param dict +
  provenance (`source`: educated guess | literature | pbisim-fit | experimental) +
  annotation. Three tabs; each: **save a current entity as a part**, list with
  **Load** (appends to the shared `int_strains`/`int_phages`/`int_antibiotics`, so it
  composes across all pages, capped at the builder max) and **Delete**, plus
  **Export/Import the whole library as versioned JSON** (`PARTS_SCHEMA_VERSION=1`).
- **Host-tagged phages** (agreed design): phage kinetics (burst/latent/adsorption) are
  phage×host properties, not phage-intrinsic — so phage parts carry a `reference_host`
  (selected from current strains) and loading one whose host isn't among the current
  strains raises a soft "verify kinetics for this host" flash. This maps directly onto
  a future `pbisim-fit` "fit → save part" pipeline (fit output = host-tagged phage part).
- Helpers in `app.py`: `PART_CATEGORIES` / `PART_SOURCES`, `export_parts_json` /
  `import_parts_json`, `clear_entity_widgets` (pops entity widget keys so appended
  entities re-read from data), and a cross-rerun `_flash` mechanism displayed after the
  sidebar (used for part-load feedback + host warning + navigation).
- Tests: `tests/test_parts.py` (save/load/append + host-tag + JSON export). **50 passing.**
- Note: AppTest raises a spurious `KeyError('part_pick_bacteria')` if you do an *extra*
  `.run()` after a part-load navigation — a harness artifact (leaving a page with a keyed
  widget). The real app is fine (the load render itself is 0-exception); tests avoid the
  post-nav extra run.

**Design status:** both preset tiers now DONE — **Tier 1 Scenarios** + **Tier 2 Parts**.
Future: `pbisim-fit` → save-part pipeline; a real per-user DB (needs auth+storage) if the
JSON export/import backbone is outgrown.

## Done this session (2026-07-14) — mutation-graph fix, segfault hardening, Calibration (Phase A)

- **Direct-mode custom mutation network**: the 2^m-strains rule was a pbisim-app
  limitation (only used `with_mutations(phage_resistance_rates=...)`); added a
  strain→strain→rate graph editor (shared `int_transitions`) → builds the (n,n)
  mass-conserving matrix (`mutation_matrix_from_transitions`) → `with_mutations(
  mutation_rates=...)`. Any strain count now supports mutation.
- **SIGSEGV (exit 139) hardening** for Render: forced `matplotlib.use("Agg")` before
  pyplot (GUI backend from Streamlit's thread segfaults headless); Dockerfile
  `MPLBACKEND=Agg` + `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` (OpenBLAS reads the host
  core count and over-spawns threads in the container → classic segfault); clinical-
  trial `n_jobs` default 4→1 (joblib loky forking); enabled `faulthandler`. Root cause
  was environmental (host placement / thread over-subscription), exposed by frequent
  auto-deploys — the thread cap fixes it deterministically. Crashes stopped.
- **Calibration page (Phase A of pbisim-fit integration)** — `pbisim_app/fit_helper.py`:
  - **Observable registry** (`OBSERVABLES`): CFU / PFU / OD / **luminescence**, each
    declaring the model compartments it reflects + a **link** (None | ÷param | ×param).
    OD = biomass ÷ `od_to_cfu`; luminescence = *active* biomass (`B` only) × `rlu_per_cell`.
    Adding a signal = one entry (extensible, per owner's bioluminescence use case).
  - **Ingestion** (`normalize_fit_dataframe`): any CSV → canonical **pbisim-fit long
    format** (`time, arm, observable, value` + per-arm MOI conditions). Auto-detects the
    pbisim-fit format; a column-mapping UI handles Monolix (`ID,TIME,DV,MOI,PHAGE,EXPERI`,
    DV=OD, PHAGE×MOI=arm). Validated against the real `monophage_data`/`ck_data` (33 arms).
  - **Overlay**: arm **multiselect** → simulate the current model per arm (`initial_P =
    MOI × B0`) → overlay predicted vs observed + per-arm **RMSE** (log for CFU/PFU).
  - The app does NOT import pbisim_fit yet (kept out of the deploy); it mirrors the schema
    so the ingested dataset feeds pbisim-fit directly when wired.
  - Tests: `tests/test_fit_helper.py` (registry, links, Monolix ingestion, RMSE). **54 passing.**

**Design plan (agreed):** Phase A DONE. **Phase B** = manual parameter tuning (focused
sliders + the link factors `od_to_cfu`/`rlu_per_cell`, re-overlay, save tuned params as a
Part). **Phase C** = pbisim-fit hand-off (manual tune → `NLSRefiner` warm-start → NPE
posterior → host-tagged Part → posterior→IIV for trials). Manual tuning is kept as the
human-in-the-loop / warm-start front-end, not deleted. Luminescence fitting later needs
pbisim-fit to add the observable (cross-package coordination item).
