---
triggers: escherichia, coli, upec, uropathogenic, uti, ompc, ompf, lamb, fhua, btub, stationary phase, lbp-ec01
---
**Escherichia coli** `[src: Bryan 2016, LBP-EC01 2023; receptors unverified]`
- **Receptors:** classically diverse — **LPS** (O-antigen/core), **OmpC/OmpF/OmpA/BtuB/FhuA/LamB**, and
  type-1/F pili (the model organism for receptor biology) `[src: unverified]` (the clinical/dormancy
  sources below don't name molecular receptors). OMP/LPS mutants → receptor loss (map: `adsorption_r`↓),
  often with a fitness cost. UPEC/biofilm → persisters (see [[persisters_dormancy]]).
- **Dormancy nuance (E. coli-specific):** for the T4×E. coli pair, dormant/stationary cells are **NOT an
  inert refuge** — T4 still **adsorbs fully (≥99.9%) and kills** them, establishing a **"hibernation"
  infection that pauses mid-cycle and completes when nutrients return** `[src: Bryan 2016]`. So model
  dormant E. coli as **adsorption preserved, burst paused (long `latent_period_dormant`) and reduced
  (~40, up to ~200/cell on resumption)** rather than uninfectable. At **high MOI**, a "scavenger" /
  **lysis-from-without** mode kills with ~no amplification (burst ≈1) — killing decoupled from `burst_size`.
- **Clinical anchors (LBP-EC01, Phase 2 UTI):** a **6-phage cocktail (3 CRISPR-Cas3-engineered + 3
  wild-type, multi-receptor)** gave ~4-log CFU reduction, 88% microbiological cure, and **no resistance
  emergence** — a strong `cr_*`/escape anchor for multi-receptor + CRISPR cocktails `[src: LBP-EC01 2023]`.
  Bladder PK: urine peak **~6.3e8 PFU/mL, Tmax 1 h, cleared by day 4–5**; plasma (IV) rapid, dose-dependent
  (peak ~2.5e5); **targeted MOI ≈ 1**. Mild, transient innate response at high IV only.
- **Antibiotic partners:** **TMP–SMX (co-trimoxazole)** — additive/synergistic, effective even vs
  TMP–SMX-resistant E. coli `[src: LBP-EC01 2023]`; also ciprofloxacin, ceftriaxone, fosfomycin, nitrofurantoin.
