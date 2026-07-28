---
triggers: persister, persisters, dormant, dormancy, tolerance, tolerant, hibernation, ppgpp, stringent response, toxin-antitoxin, vbnc, small colony variant, scv, refuge, paride, non-growing, quiescent
---
**Persisters & dormancy — the refuge, and when phage can breach it** `[src: Maffei 2023, Niu 2024, Bleriot 2024]`

- **Default: dormant cells ARE a phage refuge.** Most phages adsorb to dormant/persister cells but
  then **hibernate** — replication is suspended until the host resuscitates. Model as **dormant
  adsorption > 0 but dormant burst ≈ 0** (route to the `H` compartment; wakes at `resuscitation_rate`)
  `[src: Maffei 2023]`. Persisters are genetically drug-susceptible, non-growing, and **regrow after
  treatment stops** (relapse) — a phenotypic state, not a resistant strain `[src: Niu 2024]`.
- **Exception: some phages productively lyse dormant cells.** Phage Paride (and engineered
  phages/endolysins) replicate in deep-dormant *P. aeruginosa* — model with **nonzero dormant burst
  (~15% of growing, ~1.7× latent period)** `[src: Maffei 2023]`. It needs the host's own stringent
  response ((p)ppGpp/RpoS) intact.
- **Formation → `dormancy_rate`:** persister fraction is growth-phase dependent — ~0 in exponential,
  rising to **~1% at stationary phase**, driven by nutrient starvation via the (p)ppGpp stringent
  response and toxin–antitoxin modules `[src: Niu 2024]`. So scale `dormancy_rate` with nutrient
  depletion (a triggered term) plus a small constant stochastic term. Host/macrophage internalisation
  also raises it in vivo.
- **Antibiotic tolerance of dormant cells:** low ATP/PMF → β-lactams and aminoglycosides have **~no
  effect on `D`** (set dormant antibiotic kill ≈ 0); a few drug classes (e.g. ADEP4, bedaquiline)
  kill dormant cells directly `[src: Niu 2024]`.
- **Clearing the refuge (combo logic):** phage lysis products / spent-medium factors **resuscitate**
  bystander persisters → they resume wall synthesis → an antibiotic (e.g. meropenem) then kills them
  (**Paride + meropenem sterilises** a dormant culture; neither alone works) `[src: Maffei 2023,
  Niu 2024]`. In the model this is a **lysis-coupled `resuscitation_rate`** + antibiotic on the woken
  cells, and/or **`imm_kill_rate_D` > 0** (let immunity clear the dormant pool — the knob whose
  default 0 creates the immune refuge).
