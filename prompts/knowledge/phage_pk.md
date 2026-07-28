---
triggers: pharmacokinetic, phage pk, half-life, half life, clearance, biodistribution, plasma, compartment, antibody, phage decay, in vivo, reticuloendothelial, spleen, liver, neutralising, neutralizing
---
**Phage PK by compartment (free-phage decay).** Companion to §9's antibiotic PK table. Free-phage
decay is **compartment-dependent**; pick the decay for the setting you're modelling (don't apply
plasma clearance to an in-vitro run). Values below are for mapping `phage_decay` / phage PK
(`k_elim`, `Vc`) — real, cited where noted.

| Compartment | Free-phage decay (first-order k) | Note |
|---|---|---|
| In vitro (broth/batch) | ≈ negligible (~0.01 h⁻¹, modelling assumption) | phage persists; dynamics set by adsorption/burst |
| Plasma / systemic (in vivo) | **~0.15–4.5 h⁻¹** (t½ ~9 min–4.5 h); modelling defaults **mouse 0.07 h⁻¹ vs human serum 3.5 h⁻¹** (~50× faster in humans) [src: de Boer 2025] | mouse IV k≈3–4.5 h⁻¹ [src: Wang 2023]; RES (liver-fast/spleen-slow) uptake dominates; ~100× immediate drop from phagocytosis [src: Holger 2021]; cleared over ~1–3 h but replicates at the infection site ("auto-dosing") [src: Strathdee 2023] |
| Tissue / abscess | slower than plasma | penetration-limited (IV lung t½ ~3 h vs intratracheal ~12 h [src: Rao 2024]); local replication if bacteria present |
| Biofilm / mucus | retained; adsorption-limited *for phages lacking depolymerase* | matrix binding; depolymerase/EPS-hydrolase phages penetrate (EPS can even be the receptor) [src: Dabrowska] |

- **Antibody / neutralisation** [src: Dabrowska, Luong 2025, Bosco 2023]: accelerates clearance on
  repeat/longer courses, not first-dose. Onset ~**day 15 for de-novo** antibody, ~**day 10 if
  pre-existing** (prior exposure to a near-identical phage) — pre-existing immunity can drop a phage's
  availability ~1000× and make it fail to establish. Present in ~40% of monitored patients, outcome
  link variable. Renal excretion negligible (phage too large to filter).
- **Active vs total phage** [src: Bosco 2023]: a plaque assay measures *infectious* phage; qPCR
  measures *total* DNA (can't tell neutralised/fragmented from active). Model/report the active titre.
- **Dose-response ceiling** [src: Chang 2022]: bacterial kill **saturates** — across MOI ~4→25,000 the
  CFU drop was the same; the lowest dose *replicated most* (self-dosing is inverse to dose). Higher
  dose mainly **speeds resistance**, not depth of kill. Nebulized delivery: only ~58% of dose reaches
  the lower airway (~15–20% of nominal) [src: BX004-A 2025].
- **Phage self-amplification** — unlike an antibiotic, phage titre can *rise* where bacteria are
  dense (source = latent×burst), partly offsetting decay and even overcoming a shorter half-life at
  high burden [src: Rao 2024]; report the PFU trajectory, not just the dose.
