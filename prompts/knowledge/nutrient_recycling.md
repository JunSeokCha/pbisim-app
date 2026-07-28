---
triggers: nutrient recycling, recycling, lysate, cross-feeding, cross feeding, dead cell, post-mortem, overcompensation, nutrient release, lysis products, debris nutrient
---
**Nutrient recycling from lysed cells → survivor growth feedback** `[src: Fara 2026, Gibson 2025]`

Lysis (phage burst, and antibiotic/natural death) releases dead-cell material that survivors can
consume — a feedback that couples pbisim's **death/lysis → nutrient (S) recycling → Monod growth**
of the remaining bacteria, visible mainly in the **debris/OD** module.

- **Effect:** a transient **+11–15% growth-rate boost** in survivors, peaking ~3 h after lysis begins,
  window ~1–7 h; makes measured biomass loss **smaller than the fraction killed** `[src: Fara 2026]`.
- **Matters most under nutrient limitation / near-stationary S** — negligible in nutrient-rich
  conditions `[src: Fara 2026]`.
- **Recycling efficiency ε is partial and tunable, NOT 100%.** Only a fraction of dead-cell mass
  becomes usable nutrient: cells take up small peptides/amino acids, not intact protein, and the
  conversion is gated by **post-mortem Lon-protease** digestion — Lon-null lysate gives *no* benefit
  `[src: Gibson 2025]`. So a "dead cells release nutrients" term should carry an efficiency knob
  (0 → ~full), not assume full bioavailability. Uptake is Monod/saturating in recycled-nutrient
  concentration `[src: Fara 2026, Gibson 2025]`.
- **Sign caveat:** recycling can also be **inhibitory** if released material acts as a stress/damage
  signal — support a net-negative regime, not only a boost `[src: Fara 2026]`.
- **Map:** couple cell-death/lysis to an `s_in`-like source on nutrient S scaled by ε; survivors'
  Monod growth then responds. Observable in `get_od()` / debris, not a naïve CFU curve.
