---
triggers: acinetobacter, baumannii, crab, resensitization, resensitisation
---
**Acinetobacter baumannii** `[src: Gordillo Altamirano 2020, Gordillo Altamirano 2022, Jeon 2019, Huo 2022]`
- **Receptor = capsule (K-locus polysaccharide)**, confirmed by knockout/complementation; therapeutic
  phages are capsule-specific and **depolymerase**-bearing (hazy plaque halos) `[src: Gordillo Altamirano 2020,
  Jeon 2019]`. Phage kinetics span **burst ~15–142 PFU/cell, latent ~40–50 min**.
- **Escape = K-locus loss-of-function** (e.g. `gtr29`, `gpi`, `wzx`, `pgm`) → **adsorption_r → ~0**
  (map: `adsorption_r`/`cr_adsorption`), with reduced capsule/biofilm and an **in-vivo fitness/virulence
  cost**. Escape is common (~96% in vivo) but it **steers**: capsule-minus mutants are killed **>4-log by
  human complement/serum in ~1 h** (map: much higher `imm_kill_rate` on the resistant strain) and often
  become sensitive to a **second phage** `[src: Gordillo Altamirano 2020, 2022]`.
- **Collateral sensitivity (β-lactam-biased):** capsule-loss re-sensitises to **ceftazidime (MIC ↓16×,
  R→S)**, ampicillin-sulbactam / imipenem / ciprofloxacin (↓2×) — but it's **not universal** (amikacin MIC
  ↑2×), so the `cr_ec50` matrix should encode both signs `[src: Gordillo Altamirano 2020]`.
- **Phage + antibiotic combination is superior (true synergy, "one-two punch"):** phage kills WT and
  drives capsule-loss escape → the β-lactam (ceftazidime) then kills the re-sensitised escaper; combo beat
  either monotherapy in vivo and suppressed the phage-only regrowth `[src: Gordillo Altamirano 2022]`.
- **Immune status gates resistance (key modelling coupling):** immunocompetence (neutrophils) suppresses
  outgrowth/fixation of resistant variants and restricts which resistance alleles are accessible;
  **immunosuppression turns the host into a resistance reservoir** and enables **persister-mediated failure
  even without an MIC change** (AdeFGH efflux persistence) `[src: Huo 2022]`. → couple the immune module to
  `fitness_cost` and the dormancy/persister refuge (see [[persisters_dormancy]]).
- **Antibiotic partners:** ceftazidime (demonstrated combo), colistin, ampicillin-sulbactam, meropenem/
  imipenem, minocycline, tigecycline; also **desiccation/persistence** and device biofilm.
