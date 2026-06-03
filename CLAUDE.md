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
`pbisim` phage-bacteria simulation engine. A user describes a simulation in
natural language; Claude generates `pbisim` Python code; a sandboxed executor
runs it and returns figures + narrative.

**Status:** validated feasibility prototype, **deliberately parked**. The
concept (Claude turning NL requests into `pbisim`/`pbisim-fit` work) is proven;
end-to-end flow works. **Full development resumes once `pbisim-fit` v1.0 ships.**
27 tests. Position in the chain: top layer. Depends on `pbisim>=1.0` (and,
optionally and not-yet-wired-up, `pbisim-fit>=0.1` — see §5.3 in ECOSYSTEM.md).

---

## Repository layout

```
pbisim-app/
├── pbisim_app/
│   ├── __init__.py        package docstring / launch instructions
│   ├── app.py     (~178)  Streamlit UI: chat, history, sidebar API-key input,
│   │                      renders narrative/code/figures/stdout. `main()` is a
│   │                      no-op placeholder for the console-script entry point.
│   ├── agent.py   (~121)  SimulationAgent (wraps anthropic.Anthropic),
│   │                      AgentResponse, _parse_response() (regex extraction).
│   └── executor.py(~141)  execute_code(): builds a sandboxed namespace, exec()s
│                          generated code, captures stdout + matplotlib figures.
├── prompts/
│   └── system_prompt.md   312-line hand-maintained mirror of the pbisim API:
│                          signatures, worked example, antibiotic PK table,
│                          resistance-seeding rule, output requirements.
├── tests/
│   ├── test_agent.py      5 tests — response parsing
│   └── test_executor.py   10 tests — sandbox, capture, security, errors
└── pyproject.toml         entry point: pbisim-app = "pbisim_app.app:main"
```

---

## How it works (data flow)

```
Streamlit UI (app.py)
   → SimulationAgent.ask(user_message)            agent.py
   → Claude (model in agent.py:_MODEL), system = prompts/system_prompt.md
   → _parse_response() → AgentResponse(code, narrative, assumptions, raw_text)
   → executor.execute_code(code) → ExecutionResult(success, figures, stdout, error)
   → UI renders narrative + figures (+ optional code/assumptions/stdout)
   → conversation history kept in agent.history (multi-turn)
```

- **No tool-use, no streaming.** Claude returns markdown; code is pulled from a
  ```` ```python ```` block by regex (`agent.py:_parse_response`).
- **API key:** `ANTHROPIC_API_KEY` env var, else Streamlit sidebar password input.
- **Model:** set in `agent.py` `_MODEL` (currently `claude-sonnet-4-5` — **stale,
  bump to a current model**; see ECOSYSTEM.md §5.1).

---

## Package-specific conventions

1. **The system prompt is the contract with the engine.** `prompts/system_prompt.md`
   hard-codes `pbisim` signatures — it is NOT generated. If the `pbisim` public
   API changes, update this prompt in lockstep (coordinate via the integration
   role). Never let the prompt invent method names.
2. **Generated code may use only names the executor pre-loads.** The sandbox
   namespace (`executor.py`) exposes `np/numpy`, `plt/matplotlib`, `ModelBuilder`,
   `PBIModel`, `solve_ode`, `DoseSchedule`, `DoseEvent`, `StrainSet`,
   `StrainDefinition`, `BinaryResistanceGenotypes`, `PhagePKConfig`,
   `AntibioticDefinition`, `AntibioticSensitivity`, and (optionally)
   `TrialRunner`, `IIVSpec`, `VirtualPopulation`, `default_metrics`. Add to the
   namespace AND the prompt together if new surface is needed.
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

---

## Test commands

Activate the project env first (`conda activate pbisim202606` on Windows /
`pbisim202602` on Linux), then:

```powershell
# Full suite (fast — no network, Anthropic client is constructed but not called):
python -m pytest tests/ -q

# By file:
python -m pytest tests/test_executor.py -q   # 10 tests
python -m pytest tests/test_agent.py -q      # 5 tests
```

Tests cover response parsing and the executor (capture, ModelBuilder
availability, a full pbisim run, figure capture, blocked `open()`, error
handling). **Not covered:** Streamlit UI, real API calls / failures, multi-turn
history, `agent.reset()`, and any pbisim-fit integration.

## Running the app

```powershell
python -m streamlit run pbisim_app/app.py
# or, once installed:  pbisim-app
```

---

## Known gaps / next steps (see ECOSYSTEM.md §4, §5)

- **pbisim-fit integration is deferred by design** until pbisim-fit v1.0 ships;
  the `[fit]` extra is a documented placeholder. Intended hook:
  `pbisim_fit.output.to_model_config` (ECOSYSTEM.md §3.5, §5.3).
- No streaming, no tool-use, brittle regex parsing, no rate limiting, no UI tests.
- Model pin lives in `agent.py` `_MODEL` (currently `claude-sonnet-4-6`); keep it
  current and re-run `tests/test_system_prompt_sync.py` after any pbisim upgrade.
