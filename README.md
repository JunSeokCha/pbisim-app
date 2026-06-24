# pbisim-app

**AI-powered interactive interface for the [`pbisim`](https://github.com/phage-therapy-sim/pbisim) phage–bacteria simulation engine.**

Simulate phage therapy, antibiotic treatment, and combination regimens through a web UI — no Python required. Explore dose-response relationships, sweep biological parameters, design virtual clinical trials, and ask an AI assistant to build simulations from plain-language descriptions.

---

## Features

- **Interactive Simulator** — configure bacteria, phages, and antibiotics with three builder modes (Direct, Binary Genotypes, Custom StrainSet); run ODE simulations and export reproduction code
- **Dose-Response Sweeps** — sweep dose range for one or two agents across log or linear spacing; view color-coded trajectories and a metrics table
- **Parameter Sweeps** — 1D and 2D sweeps over any model parameter; contour maps for 2D
- **Clinical Trials & Cohorts** — virtual patient cohorts with inter-individual variability; Kaplan-Meier curves; CSV and NLME (NONMEM/Monolix) export
- **AI Assistant** — natural-language → pbisim code via Claude, with a self-healing retry loop (up to 3 attempts) and history rollback on persistent failure
- **Presets & Tutorials** — all 13 pbisim tutorials available as one-click configurations or executable scripts

---

## Requirements

- Python ≥ 3.10
- [`pbisim`](https://github.com/phage-therapy-sim/pbisim) ≥ 1.0 installed
- An Anthropic API key (only for the AI Assistant; all simulation features run offline)

---

## Installation

### Option A — existing environment (quickest)

Install in dependency order from the workspace root:

```bash
pip install -e pbisim/      # engine (dependency)
pip install -e pbisim-app/  # this package
```

### Option B — dedicated conda environment (recommended)

```bash
# From the workspace root
conda env create -f environment.yml   # create once
conda activate pbisim
pip install -e pbisim/
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

```bash
# Activate the environment first
conda activate pbisim

# Start the app
python -m streamlit run pbisim_app/app.py
# or, after pip install -e .:
pbisim-app
```

Open **http://localhost:8501** in your browser. The Anthropic API key (for the AI
Assistant only) can be entered in the sidebar — it is never stored between sessions.

**Stopping:** press `Ctrl+C` in the terminal. If the process persists:

```bash
pkill -f "streamlit run"
```

---

## App pages

| Page | Purpose |
|---|---|
| Interactive Simulator | Build and run a single simulation manually |
| Dose-Response Sweeps | Sweep dose range for one or two agents |
| Parameter Sweeps | Sweep any model parameter (1D or 2D grid) |
| Clinical Trials & Cohorts | Virtual patient cohorts, KM curves, NLME export |
| AI Assistant | Natural-language simulation builder powered by Claude |
| Presets & Tutorials | Load any of the 13 pbisim tutorials as a ready-to-run configuration |

---

## Documentation

- **[User Guide](USER_GUIDE.md)** — full task-oriented guide with worked workflows, parameter reference, and troubleshooting
- **[pbisim API Reference](https://phage-therapy-sim.github.io/pbisim-docs/API_REFERENCE.html)** — underlying ODE engine documentation
- **[pbisim Tutorials](https://phage-therapy-sim.github.io/pbisim-docs)** — 13 executed notebooks covering the full pbisim feature set

---

## Testing

```bash
python -m pytest tests/ -q    # expected: 48 passed
```

After any pbisim API change, also run:

```bash
python -m pytest tests/test_system_prompt_sync.py -v
```

---

## License

MIT
