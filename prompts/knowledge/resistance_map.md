---
triggers: resistance, cross-resistance, cross resistance, collateral, cr_adsorption, cr_ec50, receptor loss, efflux, crispr, restriction-modification, restriction modification, re-sensit, resensit, fitness cost, escape mutant, superinfection exclusion, abortive infection, phage defense
---
**Resistance-mechanism → pbisim parameter map** `[src: Kim 2024, Holger 2021, Bleriot 2024, Mora-Quilis 2025, 100 cases 2024]`
— how a real escape mechanism maps onto a model knob. See §16 (`BinaryResistanceGenotypes.from_strains`
+ `cr_*` matrices) and the §10 pre-existing-mutant seeding rule. **Key theme:** much resistance is
**reversible/physiological** (capsule downregulation, dormancy, TA-latency, QS), not a fixed mutation —
complement `cr_*`/`mutation_rate` (fixed-strain) machinery with **reversible sub-state compartments**.

| Mechanism | Biology | pbisim knob | Typical effect |
|---|---|---|---|
| Receptor loss / modification | phage can't adsorb (pilus/LPS/capsule/OMP/WTA change) | `adsorption_r`↓ (→~0) or `cr_adsorption` | resistant strain unattacked → nadir-then-regrowth |
| Reversible phenotypic (capsule/receptor down-regulation) | non-genetic, bistable low-adsorption sub-state; reverts when phage stops `[src: Mora-Quilis 2025]` | reversible sub-state (dormancy-like switch), phage-pressure-dependent rates | fast escape + relapse to susceptible/virulent |
| Capsule ↔ depolymerase | capsule blocks unless phage digests it | adsorption gated on capsule; depolymerase ↑ effective adsorption | depolymerase phage vs capsule-switch escape |
| Fitness cost of resistance | receptor/capsule loss slows growth, cuts virulence | `fitness_cost` (BRG) / lower `growth_rate` on R strain | resistant strain loses when phage withdrawn |
| Collateral sensitivity | phage-resistance re-sensitises to an antibiotic (e.g. efflux loss) | cross `cr_ec50`/`cr_emax` (phage locus → antibiotic axis) | the core of phage→antibiotic steering |
| Efflux / target change (abx) | antibiotic MIC rises | `ec50_r`↑ and/or `emax_r`↓ | antibiotic-resistant genotype survives drug |
| CRISPR-Cas / restriction-mod. | adsorbs + injects but DNA degraded intracellularly `[src: Bleriot 2024]` | `burst_size_r`→0 (not adsorption↓); per-phage `cr_*` | fewer productive infections, entry not blocked |
| Superinfection exclusion (prophage) | resident prophage blocks a related phage's DNA injection `[src: Bleriot 2024]` | per-phage infection block / `cr_*` | that phage adsorbs but yields no progeny |
| Abortive infection / CBASS | infected cell dies but releases **no** progeny (altruistic suicide) `[src: Bleriot 2024]` | `burst_size`→0 **with** infected-cell death retained | kills the cell, stops epidemic spread |
| TA-module metabolic latency | infection triggers host dormancy halting replication `[src: Bleriot 2024, Niu 2024]` | dormancy sub-state; `burst`→0 in dormant compartment | ties resistance to the dormancy machinery |
| QS / density-dependent adsorption | autoinducers up- OR down-regulate the receptor `[src: Bleriot 2024]` | `adsorption` as a function of bacterial density | context-dependent (both signs seen) |
| OMV decoys | secreted vesicles carry receptor, adsorb phage as bait `[src: Bleriot 2024]` | extra phage `decay`/loss sink | fewer productive infections |
| Persister / dormancy | slow/non-growing cells tolerate phage + abx | dormancy on; `imm_kill_rate_D`≈0 | immune/phage refuge → incomplete clearance |
