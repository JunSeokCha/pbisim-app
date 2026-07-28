---
triggers: cocktail, depolymerase, complementarity, receptor-diverse, receptor diverse, multi-phage, multiphage, multiple phages, phage combination
---
**Cocktail & depolymerase design** `[src: Kim 2024, de Boer 2025, Bleriot 2024, Bull 2014]`.
- **Complementarity Groups (CGs).** Cross-resistance is confined *within* a shared-receptor group:
  losing one phage's receptor loses susceptibility to **all** phages using that receptor, and to
  **none** outside it. So group phages by receptor (e.g. Pa: T4P-pili / LPS-OSA / flagella-OprM CGs)
  and **combine 2–3 phages from *different* CGs**. Escaping a multi-CG cocktail by receptor mutation
  is effectively foreclosed — survivors need rare, costly **multi-gene metabolic** mutations that
  also impair growth/phage replication. (Analogy the authors draw: HIV triple therapy.)
- **Model a cocktail** as multiple phages with **distinct adsorption targets**; set `cr_adsorption`
  so escape of phage A does *not* confer escape of phage B (between-CG independence) while phages
  *within* one CG share resistance. For a small, defined resistance space, BRG genotype **loci**
  (one per receptor/CG) capture the combinatorics directly. **Deliver simultaneously**, not
  sequentially — concurrent multi-CG pressure suppresses resistance better than phage-in-series.
- **Depolymerase phages** for **capsuled hosts** (Klebsiella, Acinetobacter): the enzyme strips the
  capsule, exposing the cell to the phage, other phages, antibiotics, and immune killing — a strong
  PAS/steering lever. Capsule-switch is the main escape.
- **Diversity beats potency — "hit hard and early."** In simulation, raising adsorption above the
  minimum effective value "hardly improves" therapy, whereas adding **independent phage groups**
  matters far more (need ≥1 / 2 / 3 groups for ~5 / 10 / 20% pre-existing resistance). Give all phages
  **simultaneously**, not sequentially — sequential dosing lets resistance to the later phage pre-expand
  `[src: de Boer 2025]`. A well-designed multi-receptor cocktail can show **no resistance emergence** over
  a course `[src: BX004-A 2025]`.
- **Depolymerases & training** counter capsule/biofilm resistance by raising effective adsorption and
  broadening host range (trained/Appelmans phages) `[src: Bleriot 2024]`.
- **Antibiotic pairing is predictable but organism-specific** `[src: Kim 2024]`: T4P-pili Pa phages
  synergise with β-lactams/aminoglycosides (via filamentation); interaction *sign can flip* between
  species (rifampin: Pa-synergistic, Staph-antagonistic). Frame as scenarios to simulate (compare
  arms), not fixed clinical protocol.
