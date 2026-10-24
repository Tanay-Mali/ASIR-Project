"""
graph_builder/entity_extractor.py
===================================
Optimized Entity Extraction for Digital Rights and Laws.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.util import filter_spans
from spacy.matcher import PhraseMatcher
from rapidfuzz import fuzz

import config as C

# Logger initialization MUST come before try/except
log = logging.getLogger(__name__)

try:

    from graph_builder.digital_rights_entity_dictionary import (

        DIGITAL_RIGHTS_ENTITIES,

        build_alias_lookup

    )

except ImportError:

    log.error("Digital Rights Dictionary not found! Entity linking will be limited.")

    DIGITAL_RIGHTS_ENTITIES, build_alias_lookup = {}, lambda: {}

# ─────────────────────────────────────────────────────────────────────
# Initialization & Global Model
# ─────────────────────────────────────────────────────────────────────

try:
    _NLP = spacy.load("en_core_web_lg")
except OSError:
    _NLP = spacy.load("en_core_web_sm")

# Safe build of Alias Lookup
_RAW_LOOKUP = build_alias_lookup() or {}
_ALIAS_LOOKUP = {k.lower(): v for k, v in _RAW_LOOKUP.items()}

# Batch-initialize PhraseMatcher
_MATCHER = PhraseMatcher(_NLP.vocab, attr="LOWER")
if _ALIAS_LOOKUP:
    patterns = [_NLP.make_doc(alias) for alias in _ALIAS_LOOKUP.keys()]
    _MATCHER.add("DICT_ENTITY", patterns)

LABEL_MAP = {
    "org": "organization",
    "person": "person",
    "gpe": "location",
    "date": "date"
}

@dataclass
class ExtractedEntity:
    text: str
    canonical: str
    label: str
    start: int
    end: int

# ─────────────────────────────────────────────────────────────────────
# Main Logic
# ─────────────────────────────────────────────────────────────────────

def extract_entities(text: str) -> list[ExtractedEntity]:
    if not text or len(text) < 3:
        return []

    doc = _NLP(text)
    spans = []

    # 1. Tier 1: Dictionary Matching
    dict_matches = _MATCHER(doc)
    for match_id, start, end in dict_matches:
        spans.append(doc[start:end])

    # 2. Tier 2: General NER (Defensive check for doc.ents)
    relevant_labels = {"ORG", "PERSON", "GPE", "DATE"}
    for ent in getattr(doc, "ents", []):
        if ent.label_ in relevant_labels:
            spans.append(ent)

    # 3. Resolve Overlaps
    filtered = filter_spans(spans)
    
    results = []
    stats = {"dict": 0, "ner": 0}

    for span in filtered:
        surface_form = span.text
        canonical = _normalise(surface_form)
        
        # Optimization: Use canonical directly (assumed lowercase from _normalise)
        dict_entry = DIGITAL_RIGHTS_ENTITIES.get(canonical)
        if dict_entry:
            label = dict_entry.get("type", "concept")
            stats["dict"] += 1
        else:
            raw_label = span.label_.lower() if hasattr(span, 'label_') else "concept"
            label = LABEL_MAP.get(raw_label, raw_label)
            stats["ner"] += 1

        results.append(ExtractedEntity(
            text=surface_form,
            canonical=canonical,
            label=label,
            start=span.start_char,
            end=span.end_char
        ))

    log.debug("Extraction stats: %d dict-match, %d ner-match", stats["dict"], stats["ner"])
    return sorted(results, key=lambda e: (e.start, -(e.end - e.start)))

# ─────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def _normalise(surface_form: str) -> str:
    lowered = " ".join(surface_form.lower().strip().split())

    # 1. Exact Match (Fast Path)
    if lowered in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[lowered]

    # 2. Fast Exit
    if not _ALIAS_LOOKUP or len(lowered) < 4 or len(lowered) > 60:
        return lowered

    # 3. Fuzzy Match
    best_score = 0
    best_canonical = None

    for alias, canonical in _ALIAS_LOOKUP.items():
        if abs(len(alias) - len(lowered)) > 6 or alias[0] != lowered[0]:
            continue
            
        score = fuzz.token_sort_ratio(lowered, alias)
        
        # Early break for perfect matches
        if score == 100:
            return canonical
            
        if score >= 88 and score > best_score:
            best_score = score
            best_canonical = canonical

    return best_canonical if best_canonical else lowered
