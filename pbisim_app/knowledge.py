"""Keyword-gated domain-knowledge cards for the AI assistant.

The system prompt (`prompts/system_prompt.md`) carries the **bounded** part: the pbisim
API contract plus the general phage-therapy reasoning/parameter/guardrail layer. The
**growing** part — detailed, cited organism playbooks and mechanism/PK reference tables —
lives here as small *cards*, one markdown file per topic under ``prompts/knowledge/``.

At request time :func:`select_cards` returns only the cards whose triggers appear in the
user's message, so the assistant sees the relevant knowledge without the base prompt
growing every time a new pathogen or paper is added. This keeps the always-cached prompt
roughly constant-size while the knowledge library scales indefinitely (the RAG-lite step).

Card format — a small frontmatter block, then the card body::

    ---
    triggers: pseudomonas, aeruginosa, pao1
    ---
    **Pseudomonas aeruginosa** ...

Triggers are matched as **whole words** (case-insensitive), so a short stem like ``coli``
matches "E. coli" but not "colistin". Add both a stem and its expansions when needed
(e.g. ``staph`` and ``staphylococcus``) — word matching won't treat one as the other.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

_CARD_DIR = Path(__file__).parent.parent / "prompts" / "knowledge"


class Card(NamedTuple):
    name: str            # file stem, e.g. "pseudomonas"
    triggers: tuple      # lowercased trigger phrases
    body: str            # the card text (without frontmatter)
    pattern: object      # compiled whole-word alternation over the triggers (or None)


def _parse_card(path: Path) -> Card:
    text = path.read_text(encoding="utf-8")
    triggers: tuple = ()
    body = text
    if text.lstrip().startswith("---"):
        # Strip a leading `--- ... ---` frontmatter block and read `triggers:`.
        stripped = text.lstrip()
        end = stripped.find("\n---", 3)
        if end != -1:
            front = stripped[3:end]
            body = stripped[end + 4:].lstrip("\n")
            for line in front.splitlines():
                line = line.strip()
                if line.lower().startswith("triggers:"):
                    raw = line.split(":", 1)[1]
                    triggers = tuple(t.strip().lower() for t in raw.split(",") if t.strip())
    pattern = None
    if triggers:
        alt = "|".join(re.escape(t) for t in triggers)
        pattern = re.compile(r"\b(?:" + alt + r")\b", re.IGNORECASE)
    return Card(path.stem, triggers, body.strip(), pattern)


def _load_cards() -> list:
    if not _CARD_DIR.is_dir():
        return []
    return [_parse_card(p) for p in sorted(_CARD_DIR.glob("*.md"))]


CARDS = _load_cards()

# One-line index of available topics — for the base prompt, so the model knows a library
# exists even when no card is loaded for the current turn.
KNOWLEDGE_INDEX = ", ".join(c.name for c in CARDS)

_HEADER = (
    "## Domain-knowledge cards (loaded because your query referenced them)\n"
    "Curated, cited reference cards. Use them; the `[src: ...]` tags are provenance for the "
    "modeller — never surface them to the user.\n"
)


def matching_cards(text: str, cards=None) -> list:
    """Return the cards whose triggers appear as whole words in ``text``."""
    if not text:
        return []
    cards = CARDS if cards is None else cards
    return [c for c in cards if c.pattern is not None and c.pattern.search(text)]


def select_cards(text: str, cards=None) -> str:
    """Concatenated bodies of the cards triggered by ``text`` (``""`` if none).

    Suitable for appending to the system prompt as a small, per-query (uncached) block.
    """
    hits = matching_cards(text, cards)
    if not hits:
        return ""
    return "\n\n".join([_HEADER] + [c.body for c in hits])
