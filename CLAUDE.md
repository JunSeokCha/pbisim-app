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
the initial scaffold; API wiring requires engine-author oversight). **49 tests passing.** Depends on `pbisim>=1.0` (and,
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
│   ├── presets.py (~1185)   All 13 pbisim tutorials as structured parameter dicts
│   │                        + reference script_code strings.
│   ├── sweep_helper.py      ModelConfig mutation helpers for 1D/2D parameter sweeps
│   │                        and dose-response sweeps; vector padding for MOI sweeps.
│   └── trial_helper.py      IIV distribution factory, ClinicalTrial orchestration,
│                            Plotly KM + box plots, metric dataframe helpers.
├── prompts/
│   └── system_prompt.md     ~400-line hand-maintained pbisim API reference for the
│                            AI assistant. Sections 1–11 original; §12–17 added
│                            2026-06-23 covering new pbisim features.
├── tests/
│   ├── test_agent.py        5 tests — response parsing
│   ├── test_executor.py     10 tests — sandbox, capture, security, errors
│   ├── test_presets.py      18 tests — preset structure and parameter validity
│   ├── test_builder_modes.py 15 tests — BRG, StrainSet, cohort trial, phage-leak guard
│   ├── test_sweeps.py       9 tests — sweep_helper parameter application
│   └── test_self_healing.py 6 tests — self-healing loop and history rollback
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
| AI Assistant | Natural-language → pbisim code. Self-healing loop (up to 3 retries with history rollback). Dynamic model listing from `/v1/models`. |
| Presets & Tutorials | Browser for all 13 tutorials. type="single" → load into simulator; type="script" → execute via executor. |

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
# Full suite — expected: 48 passed
python -m pytest tests/ -q

# By file:
python -m pytest tests/test_executor.py -q        # 10 tests
python -m pytest tests/test_agent.py -q           # 5 tests
python -m pytest tests/test_presets.py -q         # 18 tests
python -m pytest tests/test_builder_modes.py -q   # 15 tests (BRG, StrainSet, cohort, phage-leak guard)
python -m pytest tests/test_sweeps.py -q          # 9 tests
python -m pytest tests/test_self_healing.py -q    # 6 tests
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
