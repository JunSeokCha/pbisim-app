# pbisim-app

**AI-powered interactive interface for the [`pbisim`](https://github.com/phage-therapy-sim/pbisim) phage–bacteria simulation engine, with parameter estimation via [`pbisim-fit`](https://github.com/phage-therapy-sim/pbisim-fit).**

Simulate phage therapy, antibiotic treatment, and combination regimens through a web UI — no Python required. Explore dose-response relationships, sweep biological parameters, design virtual clinical trials, **calibrate the model against your own experimental data**, and ask an AI assistant to build simulations from plain-language descriptions.

---

## Features

- **Interactive Simulator** — configure bacteria, phages, and antibiotics with three builder modes (Direct, Binary Genotypes, Custom StrainSet); pick growth/death signal functions, dormancy, immunity, and OD/debris; run ODE simulations and export reproduction code. A click-to-view **model-config snapshot** shows the whole configuration in one place.
- **Calibration** — upload experimental CSVs (CFU / PFU / OD / luminescence), overlay the model against the data, and **fit parameters with pbisim-fit's NLS** (`refine_nls`). Unified role-based parameter table (Fixed / Free / Derived + custom θ), MAP priors, a compact fit-spec DSL, additive-B₀ inoculum (dose / estimate / first-observation), and NONMEM/Monolix dose-row import.
- **Dose-Response Sweeps** — sweep the dose range for one or two agents across log or linear spacing; view color-coded trajectories and a metrics table.
- **Parameter Sweeps** — 1D and 2D sweeps over any model parameter (plus "all-strains" broadcasts and coupled multi-parameter sweeps); contour maps for 2D.
- **Clinical Trials & Cohorts** — virtual patient cohorts with inter-individual variability, arbitrary named dose arms, stationary-phase pre-treatment, Kaplan-Meier curves, per-arm PK/PD trajectories, and CSV / NLME (NONMEM/Monolix) export.
- **AI Assistant** — natural-language → pbisim code via Claude, with a self-healing retry loop (up to 3 attempts) and history rollback on persistent failure.
- **Library** — save/load the full configuration as **Scenarios**, or individual bacteria/phages/antibiotics as composable **Parts**; export/import as versioned JSON.
- **Models** — freeze the current organism/kinetics as a named **Model**; every task (sweeps, trials, calibration) can run against a chosen frozen Model instead of the live builder.

---

## Requirements

- Python ≥ 3.10
- [`pbisim`](https://github.com/phage-therapy-sim/pbisim) ≥ 1.0 (the ODE engine)
- [`pbisim-fit`](https://github.com/phage-therapy-sim/pbisim-fit) ≥ 0.1 (parameter estimation; used by the Calibration page — imported lazily, so the app stays torch-free)
- An Anthropic API key (only for the AI Assistant; every other feature runs offline)

---

## Installation

### Option A — existing environment (quickest)

Install in dependency order from the workspace root:

```bash
pip install -e pbisim/      # engine
pip install -e pbisim-fit/  # estimation (Calibration page)
pip install -e pbisim-app/  # this package
```

### Option B — dedicated conda environment (recommended)

```bash
# From the workspace root
conda env create -f environment.yml   # create once
conda activate pbisim
pip install -e pbisim/
pip install -e pbisim-fit/
pip install -e pbisim-app/
```

### Full ecosystem (all three packages)

```bash
conda env create -f environment.yml
conda activate pbisim
./install.sh          # Linux/macOS — installs pbisim, pbisim-fit, pbisim-app
# .\install.ps1       # Windows equivalent
```

---

## Running

### Browser (default)

```bash
conda activate pbisim
python -m streamlit run pbisim_app/app.py
# or, after pip install -e .:
pbisim-app
```

Open **http://localhost:8501**. The Anthropic API key (AI Assistant only) is entered in the sidebar and is never stored between sessions.

**Stopping:** `Ctrl+C` in the terminal (or `pkill -f "streamlit run"` if it persists).

### Desktop window (optional)

Wrap the local app in a native OS window instead of a browser tab:

```bash
pip install -e '.[desktop]'   # adds pywebview
pbisim-app-desktop
```

It launches the same local server in a native window (falls back to the browser if no native webview engine is available; set `PBISIM_APP_BROWSER=1` to force the browser). On Linux the native window needs a system webview library (e.g. `apt install gir1.2-webkit2-4.1 python3-gi`) and a reasonably current one — Windows (WebView2) and macOS (WKWebView) work out of the box.

### Hosted

A deployed instance runs on Render (auto-deploys from `main`). It may be gated by a shared password.

---

## App pages

| Page | Purpose |
|---|---|
| Interactive Simulator | Build and run a single simulation; view the model-config snapshot |
| Calibration | Fit the model to experimental data (pbisim-fit NLS) |
| Dose-Response Sweeps | Sweep the dose range for one or two agents |
| Parameter Sweeps | Sweep any model parameter (1D or 2D grid) |
| Clinical Trials & Cohorts | Virtual patient cohorts, KM curves, NLME export |
| AI Assistant | Natural-language simulation builder powered by Claude |
| Library | Save/load full-config Scenarios and composable Parts |

A **Model selector** in the sidebar (and on each task page) chooses whether the page runs against the live builder draft or a frozen saved/demo Model.

---

## Documentation

- **[User Guide](USER_GUIDE.md)** — full task-oriented guide with worked workflows, parameter reference, and troubleshooting
- **[pbisim API Reference](https://phage-therapy-sim.github.io/pbisim-docs/API_REFERENCE.html)** — underlying ODE engine
- **[pbisim Tutorials](https://phage-therapy-sim.github.io/pbisim-docs)** — executed notebooks covering the full pbisim feature set

---

## Testing

```bash
python -m pytest tests/ -q    # 288 passed
```

After any pbisim API change, also run the prompt↔engine sync guard:

```bash
python -m pytest tests/test_system_prompt_sync.py -v
```

---

## License

MIT
