"""Tests for the keyword-gated domain-knowledge cards (knowledge.py)."""

from __future__ import annotations

from pbisim_app import knowledge
from pbisim_app.knowledge import CARDS, matching_cards, select_cards


def test_cards_load_with_triggers_and_body():
    assert len(CARDS) >= 11, "expected the pathogen + mechanism + modelling cards to load"
    names = {c.name for c in CARDS}
    for expect in ("pseudomonas", "klebsiella", "acinetobacter", "staph", "ecoli",
                   "resistance_map", "cocktails", "phage_pk",
                   "persisters_dormancy", "pkpd_modeling", "nutrient_recycling"):
        assert expect in names, f"missing card {expect}"
    for c in CARDS:
        assert c.triggers, f"{c.name} has no triggers"
        assert c.body.strip(), f"{c.name} has no body"
        assert not c.body.lstrip().startswith("---"), f"{c.name} frontmatter not stripped"


def test_select_organism_card():
    out = select_cards("How should I model a Pseudomonas aeruginosa infection?")
    assert "Pseudomonas aeruginosa" in out
    # unrelated organism cards must NOT be loaded
    assert "Klebsiella pneumoniae" not in out


def test_no_triggers_returns_empty():
    # a generic simulation request mentions no pathogen/topic → no cards
    assert select_cards("Plot total bacteria over 24 hours with one phage.") == ""


def test_whole_word_matching_avoids_false_positive():
    # 'coli' is an E. coli trigger but must not fire on 'colistin'
    out = select_cards("Dose colistin 2 mg every 8 hours.")
    assert "Escherichia coli" not in out
    assert all(c.name != "ecoli" for c in matching_cards("Dose colistin 2 mg."))
    # but a real E. coli mention does fire
    assert "Escherichia coli" in select_cards("model an E. coli culture")


def test_multiple_cards_selected():
    names = {c.name for c in matching_cards("compare a Klebsiella cocktail vs Staphylococcus")}
    assert {"klebsiella", "staph", "cocktails"} <= names


def test_new_topic_cards_trigger():
    assert "persisters_dormancy" in {c.name for c in matching_cards("how do persister cells survive?")}
    assert "nutrient_recycling" in {c.name for c in matching_cards("does lysate feed surviving cells?")}
    assert "pkpd_modeling" in {c.name for c in matching_cards("what default parameters and model structure?")}


def test_header_present_only_when_cards_match():
    assert knowledge._HEADER.split("\n", 1)[0] in select_cards("Pseudomonas question")
    assert select_cards("just plot something") == ""


def test_agent_system_blocks_inject_cards():
    from pbisim_app import agent as ag
    # base prompt is always the first, cached block
    base = ag._system_blocks("plot bacteria over 24h")
    assert base[0]["cache_control"] == {"type": "ephemeral"}
    assert len(base) == 1  # no card appended for a card-less query

    withcard = ag._system_blocks("model a Pseudomonas aeruginosa cocktail")
    assert withcard[0]["text"] == ag._SYSTEM_PROMPT           # cached base unchanged
    assert "cache_control" not in withcard[-1]                # card tail is uncached
    assert "Pseudomonas aeruginosa" in withcard[-1]["text"]

    # extra (tool-instruction) block sits between base and cards
    blocks = ag._system_blocks("Pseudomonas", extra=[{"type": "text", "text": "TOOLS"}])
    assert blocks[0]["text"] == ag._SYSTEM_PROMPT
    assert blocks[1]["text"] == "TOOLS"
    assert "Pseudomonas aeruginosa" in blocks[-1]["text"]
