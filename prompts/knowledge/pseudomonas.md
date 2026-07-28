---
triggers: pseudomonas, aeruginosa, pao1, pa14, pa01, omko1, pyocyanin, mucoid, cystic fibrosis
---
**Pseudomonas aeruginosa** `[src: Holger 2021, Kim 2024, Chang 2022, Luong 2025, BX004-A 2025, 100 cases 2024]`
— the most common clinical phage-therapy target (~half of cases).
- **Receptors:** type-IV **pili/PilA** (twitching motility) and **LPS O-antigen**; also documented
  therapeutic-phage receptors **BtuB, TolC (efflux), PfeB** (BX004-A cocktail) and the **OprM efflux
  component** (OMKO1, needs flagella too). Distinct receptors → basis for a **receptor-diverse
  cocktail** `[src: Kim 2024, BX004-A 2025]`.
- **Resistance / refuge:** pilus/LPS mutants → **receptor loss** (map: `adsorption_r`↓ or `cr_adsorption`),
  typically with a **fitness cost / reduced virulence** — resistant isolates show reduced
  twitching/pyocyanin and hyperpigmentation (`hmgA`), and stay **low-frequency (<15%)** in vivo
  `[src: Chang 2022, Luong 2025]`. **Collateral sensitivity confirmed:** PEV31-resistant cells became
  **cipro-susceptible** `[src: Chang 2022]`; OMKO1-resistant lose OprM efflux → re-sensitise to
  cipro/ceftazidime `[src: Holger 2021]` (map: cross `cr_ec50`). **Mucoid (alginate)** conversion +
  **biofilm** = a **physical** refuge unaffected by both phage and antibiotics (not genetic resistance)
  `[src: Luong 2025]` (map: dormancy, `imm_kill_rate_D`≈0; depolymerase phages degrade the matrix; see
  [[persisters_dormancy]] — Paride can lyse dormant Pa).
- **Dosing behaviour:** kill **saturates** — across MOI ~4→25,000 the lung CFU drop was the same
  (~1.3–1.9 log); higher dose does *not* deepen the nadir but **speeds resistance** (30/74/91% at
  low/mid/high dose) `[src: Chang 2022]`. Phage **self-amplifies in situ** (MOI<1 can suffice; 141
  virions/cell observed) `[src: Luong 2025]`. Real doses: nebulized ~1e10 PFU BID (CF) `[src: BX004-A 2025]`,
  IV ~2e8–2e9 PFU q8h `[src: Luong 2025]`.
- **Antibiotic partners (PAS):** ciprofloxacin, ceftazidime, ceftolozane/tazobactam, colistin
  (well-evidenced); no antagonism seen with aztreonam/colistin/tobramycin `[src: BX004-A 2025]`. Note
  colistin can *also* limit phage propagation by destabilising the membrane — model both effects.
