---
triggers: model structure, modeling, modelling, parameter default, default parameters, calibrate, calibration, immunophage, critical bacterial concentration, cbc, saturating adsorption, centroid index, phage auc, efficiency metric, validate model, literature parameters
---
**Phage PK/PD modelling — structure, defaults, and the headline lessons**
`[src: de Boer 2025, Rodriguez-Gonzalez 2024, Attwood 2025, Hosseini 2024]`

- **pbisim's structure is well-founded.** Two independent ODE phage-therapy models use exactly the
  same state set (susceptible + resistant bacteria, infected/latent compartment, phage, immune,
  antibiotic), logistic density-dependent growth, mass-action-at-low-density adsorption, explicit
  burst from the infected compartment, per-capita phage decay, and a **saturating** immune-kill term
  `[src: de Boer 2025, Rodriguez-Gonzalez 2024]`.
- **Citable default parameter set** (human unless noted) `[src: de Boer 2025]`: growth `r` 0.31 h⁻¹
  (mouse 0.75), death 0.01 h⁻¹, **burst 100**, latent ≈ 0.5 h (`d_I`=2 h⁻¹), **phage decay 3.5 h⁻¹ in
  human serum vs 0.07 h⁻¹ in mouse** (~50× faster in humans), carrying capacity K 1e9–1e10 CFU/mL,
  mutation 2.85e-8/division, minimal effective adsorption ≈ 2e-8 g/(PFU·h). *P. aeruginosa* mouse lung:
  resistant growth = 0.9× susceptible (10% cost), immune half-kill ≈ 6e6 CFU/mL `[src: Rodriguez-Gonzalez 2024]`.
- **Immunophage synergy is the core mechanism** — phage rarely eradicates alone; its real job is to
  push bacteria **below a critical bacterial concentration (CBC)** so the immune system finishes.
  Both models fail to clear without competent immunity (matches the ≥20/50% immune thresholds in the
  base layer) `[src: de Boer 2025, Rodriguez-Gonzalez 2024]`. pbisim's `extinction_threshold` /
  immune `kill50` approximate the CBC.
- **Dose/MOI is secondary; resistance-regrowth is the race.** Phage self-amplify fast (max rate
  ~9 h⁻¹), so initial dose barely changes outcome once susceptibles are present — but **resistant
  strains bloom by competitive release the moment susceptibles fall.** Success = breaching the CBC
  before resistant mutants do → **"hit hard and early" with a receptor-diverse cocktail; the *number*
  of independent phage groups matters more than tuning adsorption** (need ≥1/2/3 groups for 5/10/20%
  pre-existing resistance) `[src: de Boer 2025]`.
- **Optional structural refinements** (vs pbisim's plain mass-action): saturating adsorption
  `βSP/(1+P/h_P+S/h_B)` `[src: de Boer 2025]` or sub-linear `φ·P^σ` (σ≈0.6) `[src: Rodriguez-Gonzalez 2024]`.
- **Output metrics worth reporting** beyond nadir/AUC: **phage-exposure AUC** (bacterial kill tracks
  it linearly, R²≈0.93–0.99 — the phage analog of AUC/MIC) `[src: Attwood 2025]`; and the **Centroid
  Index** `CI = 1 − (x̄·ȳ)/(x̄·ȳ)_control` on the CFU/OD trajectory (0=useless, 1=full suppression,
  <0=worse than control), which **penalises late regrowth** that endpoint/AUC miss `[src: Hosseini 2024]`.
