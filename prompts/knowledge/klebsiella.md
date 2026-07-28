---
triggers: klebsiella, pneumoniae, kpc, hypervirulent, hypermucoviscous, k-type, capsule depolymerase, capsule downregulation
---
**Klebsiella pneumoniae** `[src: Mora-Quilis 2025, Bleriot 2024]`
- **Receptors:** the **capsule (CPS, K-type)** is usually the *primary* receptor **and** a physical
  barrier — most therapeutic Klebsiella phages carry a **capsule depolymerase** to digest it
  (confirmed for phage Cap62 on K1) `[src: Mora-Quilis 2025]`. Hypervirulent hypermucoviscous strains
  = thick capsule → **depolymerase essential**.
- **Resistance is dominantly REVERSIBLE and phenotypic — not a fixed mutation.** Under phage pressure
  the population **downregulates capsule biosynthesis** (a non-genetic, bistable "phase-variation"
  sub-state), shifting from ~0.3% to **~92% capsule-off** cells; when phage is removed it **relaxes
  back** toward capsulated/susceptible over passages `[src: Mora-Quilis 2025]`. Model this as a
  **reversible low-adsorption sub-state** (capsule-ON↔OFF) with phage-pressure-dependent switch rates
  — structurally like dormancy/resuscitation, **not** a fixed `cr_*` strain. A *minority* stable
  acapsular fraction is genuinely mutational (e.g. `rfaH`) → the usual `mutation_rate` strain.
- **Cost / coupling:** capsule-off cells have **little growth-rate cost in vitro** but lost virulence
  / immune-evasion, and the switch is **coupled to a low-energy persister-like state** (map:
  `fitness_cost` small on growth; big on virulence; link to [[persisters_dormancy]]) `[src: Mora-Quilis 2025]`.
  Because transiently-resistant cells revert to virulent susceptibles when phage stops, **combine
  phage with antibiotics or a second phage** to hit revertants.
- **Antibiotic partners:** meropenem, ceftazidime-avibactam, colistin, aminoglycosides;
  depolymerase-exposed cells are more antibiotic-/immune-accessible.
