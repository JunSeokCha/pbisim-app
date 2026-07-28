"""The eval test set: representative natural-language requests + automatic checks.

Each case is what a user might type in the AI Assistant, paired with objective checks
that its generated code actually did the job. Keep prompts realistic and checks lenient
enough to allow reasonable modelling freedom — the point is "did it produce working,
contract-respecting code," not "did it match one exact implementation."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evals.checks import (
    runs_ok, has_figure, stdout_contains, stdout_contains_any, code_contains,
    uses_cfu_sum_prefixes, no_code_run, answer_contains_any,
)


@dataclass
class EvalCase:
    id: str
    prompt: str
    checks: list = field(default_factory=list)
    tags: tuple = ()
    intent: str = "code"   # "code" (writes+runs a sim) | "chat" (answers, no code)


CASES = [
    EvalCase(
        "basic_phage",
        "Simulate 1 wild-type bacterial strain and 1 phage with burst size 50 and "
        "adsorption 1e-8. Plot total bacteria over 48 hours.",
        [runs_ok(), has_figure(), uses_cfu_sum_prefixes()],
        tags=("core",),
    ),
    EvalCase(
        "phage_moi_compare",
        "Compare three phage doses (MOI 0.01, 0.1, 1) against one bacterial strain and "
        "plot the bacterial trajectories on the same log axis.",
        [runs_ok(), has_figure()],
        tags=("core", "dosing"),
    ),
    EvalCase(
        "antibiotic_only",
        "Model a single bacterial strain treated with ciprofloxacin dosed 400 mg every "
        "12 hours for 3 days. Plot bacteria vs time.",
        [runs_ok(), has_figure(), code_contains("DoseEvent")],
        tags=("antibiotic", "dosing"),
    ),
    EvalCase(
        "phage_plus_abx",
        "Combine phage therapy with an antibiotic against one strain and plot total "
        "bacteria over time.",
        [runs_ok(), has_figure()],
        tags=("combo",),
    ),
    EvalCase(
        "resistance_evolution",
        "Two bacterial strains: a wild type sensitive to phage and a phage-resistant "
        "mutant seeded at the mutation-selection level. Show resistance taking over "
        "under phage pressure; plot both strains.",
        [runs_ok(), has_figure()],
        tags=("resistance", "multistrain"),
    ),
    EvalCase(
        "brg_genotypes",
        "Use BinaryResistanceGenotypes to build a system with 1 phage and 1 antibiotic "
        "(4 genotypes) and simulate combination therapy. Plot total bacteria.",
        [runs_ok(), has_figure(), code_contains("BinaryResistanceGenotypes")],
        tags=("brg",),
    ),
    EvalCase(
        "dormancy_persisters",
        "Simulate a strain with dormancy/persister dynamics under phage attack and plot "
        "active vs dormant bacteria over time.",
        [runs_ok(), has_figure()],
        tags=("dormancy",),
    ),
    EvalCase(
        "stationary_prerun",
        "Grow bacteria to stationary phase first using stationary_phase_ic, then start "
        "phage treatment, and plot the treatment phase.",
        [runs_ok(), has_figure(), code_contains("stationary_phase_ic")],
        tags=("prerun",),
    ),
    EvalCase(
        "immunity",
        "Add innate host immunity to a single-strain phage simulation and plot bacteria "
        "and the immune response over time.",
        [runs_ok(), has_figure()],
        tags=("immunity",),
    ),
    EvalCase(
        "chemostat",
        "Model a chemostat with constant nutrient inflow and washout (s_in and s_out) "
        "for one bacterial strain and plot bacteria and nutrient.",
        [runs_ok(), has_figure()],
        tags=("nutrient",),
    ),
    EvalCase(
        "od_tracking",
        "Track optical density via the debris ODE for a phage-treated culture and plot "
        "OD over time.",
        [runs_ok(), has_figure(), code_contains("with_od_debris")],
        tags=("od",),
    ),
    EvalCase(
        "clearance_metric",
        "Simulate phage therapy and print the time to clearance and the time to a 2-log "
        "reduction in bacteria.",
        # accept any reasonable phrasing of the reported metrics
        [runs_ok(), stdout_contains_any(["clear", "reduction", "log", "eradicat"])],
        tags=("metrics",),
    ),
    EvalCase(
        "strainset_graph",
        "Use StrainSet with a mutation graph to model a wild type mutating to a "
        "resistant strain, treated with a phage. Plot both strains.",
        [runs_ok(), has_figure(), code_contains("StrainSet")],
        tags=("strainset", "resistance"),
    ),
    EvalCase(
        "burst_latent_sweep",
        "Show how phage burst size (20, 50, 100) affects bacterial suppression; overlay "
        "the three trajectories.",
        [runs_ok(), has_figure()],
        tags=("core",),
    ),
    EvalCase(
        "two_phage_cocktail",
        "Simulate a two-phage cocktail against a single bacterial strain and plot total "
        "bacteria and both free-phage populations.",
        [runs_ok(), has_figure()],
        tags=("core", "multiphage"),
    ),
    EvalCase(
        "inoculum_response",
        "Compare low vs high starting bacterial inoculum (1e5 vs 1e8) under identical "
        "phage treatment and plot both.",
        [runs_ok(), has_figure()],
        tags=("core",),
    ),

    # ── chat / Q&A: the copilot should ANSWER, not run a simulation (Phase 1 routing).
    #    These also score the ANSWER for the expected domain reasoning (answer_contains_any)
    #    — the measurable target of the domain-expert prompt layer. Option lists are broad:
    #    the point is "did it name the right mechanism," not one exact phrasing. ─────────────
    EvalCase(
        "q_adsorption",
        "What is a realistic adsorption rate for a lytic phage, and what are its units?",
        [no_code_run(), answer_contains_any(["1e-8", "1e-9", "10^-8", "10^-9", "ml/", "ml ",
                                             "per pfu"])],
        tags=("qa",), intent="chat",
    ),
    EvalCase(
        "q_burst_meaning",
        "Explain what phage burst size means and give a typical value.",
        [no_code_run(), answer_contains_any(["progeny", "released", "per cell", "per infected",
                                             "lysis", "20", "50", "100"])],
        tags=("qa",), intent="chat",
    ),
    EvalCase(
        "q_immune_resistance",
        "Why can phage-resistant bacteria sometimes still be cleared by the host immune system?",
        [no_code_run(), answer_contains_any(["immune", "phagocyt", "clear", "innate"])],
        tags=("qa",), intent="chat",
    ),
    EvalCase(
        "q_synergy_reason",
        "Why might combining a phage with an antibiotic outperform either one alone against "
        "the same bacterial strain?",
        [no_code_run(), answer_contains_any(["synerg", "collateral", "re-sensit", "resensit",
                                             "fitness cost", "trade-off", "tradeoff",
                                             "different subpop", "steer", "evolution"])],
        tags=("qa", "combo"), intent="chat",
    ),
    EvalCase(
        "q_implausible_adsorption",
        "I set the phage adsorption rate to 1e-5 mL/h. Is that realistic?",
        [no_code_run(), answer_contains_any(["too high", "too fast", "implausible", "unrealistic",
                                             "orders of magnitude", "1e-9", "1e-8", "high"])],
        tags=("qa", "plausibility"), intent="chat",
    ),
    EvalCase(
        "q_regrowth_meaning",
        "My bacterial curve drops to a low nadir and then climbs back up. What does that "
        "rebound most likely mean?",
        [no_code_run(), answer_contains_any(["resist", "mutant", "escape", "persist", "dorman",
                                             "refuge", "regrow", "selection"])],
        tags=("qa", "interpret"), intent="chat",
    ),
    EvalCase(
        "q_dormancy_refuge",
        "Why does adding host immunity sometimes still fail to clear an infection when the "
        "bacteria can go dormant?",
        [no_code_run(), answer_contains_any(["dorman", "persist", "refuge", "shield",
                                             "imm_kill_rate_d", "reservoir", "privileg",
                                             "tolerat"])],
        tags=("qa", "dormancy"), intent="chat",
    ),

    # ── pathogen / mechanism knowledge (curated domain layer) ──
    EvalCase(
        "q_klebsiella_depolymerase",
        "Why do many Klebsiella pneumoniae phages carry a depolymerase enzyme?",
        [no_code_run(), answer_contains_any(["capsule", "cps", "k-type", "k type",
                                             "polysaccharide", "barrier", "degrad", "digest"])],
        tags=("qa", "pathogen", "klebsiella"), intent="chat",
    ),
    EvalCase(
        "q_pseudomonas_mucoid",
        "A Pseudomonas aeruginosa culture turns mucoid under phage pressure. What happened, "
        "and how would I represent it in the model?",
        [no_code_run(), answer_contains_any(["alginate", "mucoid", "adsorption", "receptor",
                                             "refuge", "fitness cost", "biofilm"])],
        tags=("qa", "pathogen", "pseudomonas"), intent="chat",
    ),
    EvalCase(
        "q_receptor_loss_knob",
        "Which pbisim parameter represents phage resistance by receptor loss, and how should I "
        "set it?",
        [no_code_run(), answer_contains_any(["adsorption", "cr_adsorption", "adsorption_r"])],
        tags=("qa", "mechanism"), intent="chat",
    ),
    EvalCase(
        "q_cocktail_rationale",
        "Why would I use a receptor-diverse phage cocktail instead of a single very strong phage?",
        [no_code_run(), answer_contains_any(["cross-resist", "cross resist", "multiple mutation",
                                             "different receptor", "diverse receptor",
                                             "independent", "escape", "suppress resist"])],
        tags=("qa", "cocktail"), intent="chat",
    ),
]


def case_by_id(cid: str) -> EvalCase:
    for c in CASES:
        if c.id == cid:
            return c
    raise KeyError(cid)
