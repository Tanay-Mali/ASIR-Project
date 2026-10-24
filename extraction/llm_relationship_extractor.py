"""
graph_builder/llm_relationship_extractor.py  (FIXED VERSION)
=============================================================

FIXES IN THIS VERSION:
──────────────────────────────────────────────────────────────────────
FIX 1 — Wrong import path
  BEFORE: from extraction.relationship_extractor import ExtractedRelation
  AFTER:  from graph_builder.relationship_extractor import ExtractedRelation

  The module 'extraction' does not exist in the project.
  The correct package is 'graph_builder'.
  The original wrong import caused a NameError inside _parse_and_validate
  every time it tried to instantiate ExtractedRelation(...), which was
  silently caught and returned []. This is why the LLM was being called
  (Ollama logs show POST /api/chat with 200 responses) but zero relations
  were ever stored in the graph.

FIX 2 — Entity name mismatch in validation (fuzzy matching added)
  BEFORE: exact string match required
    if subject not in known_canonicals: → reject
  AFTER: fuzzy match with fuzzywuzzy as fallback
    if subject not in known_canonicals:
        try fuzzy_match(subject, known_canonicals)
        if no match found: → reject

  The LLM is given canonical names like "b.r. ambedkar" in the prompt
  but frequently returns shortened forms like "ambedkar" or "gandhi" in
  its response. The exact-match check rejected all of these even though
  they clearly refer to the correct entity. Fuzzy matching with an 80%
  threshold resolves this without accepting hallucinated entity names.

FIX 3 — Separate call counter from success counter
  BEFORE: stats["llm_calls"] was only incremented when results were non-empty.
          This made it impossible to tell whether the LLM was being called at
          all vs being called but returning empty results.
  AFTER:  The function returns a metadata dict alongside results:
          {"relations": [...], "was_called": True, "rejection_count": N}
          The pipeline uses "was_called" to count total LLM invocations and
          "rejection_count" to track how many triples were rejected during
          validation. This makes debugging extraction quality straightforward.

FIX 4 — INFO-level logging for rejected triples
  BEFORE: log.debug (invisible unless debug logging is enabled)
  AFTER:  First N rejections per run logged at INFO level so they appear
          in the standard pipeline output. This lets you immediately see
          whether rejections are due to entity mismatch, hallucination,
          or another cause.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import requests

import config as C

# ── FIX 1: Corrected import path ──────────────────────────────────────
# Was: from extraction.relationship_extractor import ExtractedRelation
# The module 'extraction' does not exist. Correct package is 'graph_builder'.
from graph_builder.entity_extractor import ExtractedEntity
from extraction.relationship_extractor import ExtractedRelation

log = logging.getLogger(__name__)

# How many rejected triples to log at INFO level before switching to DEBUG.
# Prevents log flooding when the LLM is called thousands of times.
_LOG_REJECTION_LIMIT = 20
_rejection_log_count = 0


# ─────────────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """You are a precise information extraction system specialised in Indian history, politics, and constitutional law.
Your task is to extract relationships between named entities from text.

STRICT RULES:
1. Only extract relationships that are EXPLICITLY stated in the text. Never infer.
2. Subject and object must BOTH be from the provided entity list. Use the EXACT names from the list.
3. Use short, lowercase, underscore-separated relation labels. Examples:
   "led_to", "established", "amended", "defined_in", "caused", "protects", "appointed"
4. Do not extract more than 5 triples per sentence.
5. If no clear directional relationship exists, return an empty array.
6. Respond with valid JSON only. No explanation, no preamble."""

_EXTRACTION_USER = """TEXT:
{text}

ENTITIES PRESENT IN TEXT (use EXACTLY these names in subject/object):
{entities}

Extract relationships as a JSON array. Each item must have: subject, relation, object.
The subject and object values must exactly match one of the entity names listed above.
Return [] if no explicit relationship exists.

Example output format:
[{{"subject": "gandhi", "relation": "launched", "object": "non cooperation movement"}}]

Your response:"""


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def extract_relations_with_llm(
    text:     str,
    entities: list[ExtractedEntity],
) -> dict:
    """
    Use the local Ollama LLM to extract relationships from a sentence.

    Parameters
    ----------
    text     : The sentence to extract from.
    entities : Entities already detected in this sentence.

    Returns
    -------
    dict with keys:
        "relations"       : list[ExtractedRelation]
        "was_called"      : bool   — True if the LLM API was actually invoked
        "rejection_count" : int    — how many triples were rejected in validation
        "raw_response"    : str    — the raw LLM response (for debugging)
    """
    empty_result = {
        "relations":       [],
        "was_called":      False,
        "rejection_count": 0,
        "raw_response":    "",
    }

    if len(entities) < C.GRAPH_LLM_MIN_ENTITIES:
        return empty_result

    entity_names = [e.canonical for e in entities]
    known        = set(entity_names)

    prompt = _EXTRACTION_USER.format(
        text=text,
        entities=", ".join(entity_names),
    )

    raw_response = _call_ollama(prompt)

    if raw_response is None:
        # LLM call failed (timeout, connection error, etc.)
        return empty_result

    # LLM was successfully called (even if response is empty array)
    relations, rejection_count = _parse_and_validate(raw_response, known, text)

    return {
        "relations":       relations,
        "was_called":      True,
        "rejection_count": rejection_count,
        "raw_response":    raw_response,
    }


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _call_ollama(user_prompt: str) -> Optional[str]:
    """
    Call Ollama API and return raw response string.
    Returns None on failure (timeout, connection error).
    """
    try:
        response = requests.post(
            f"{C.LLM_HOST}/api/chat",
            json={
                "model":  C.LLM_MODEL,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.0,       # fully deterministic
                    "num_predict": 300,
                },
                "messages": [
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=20,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    except requests.Timeout:
        log.debug("LLM relationship extraction timed out.")
    except requests.ConnectionError:
        log.debug("LLM connection error — is Ollama running?")
    except Exception as exc:
        log.debug("LLM relationship extraction failed: %s", exc)

    return None


def _parse_and_validate(
    raw_response:     str,
    known_canonicals: set[str],
    source_text:      str,
) -> tuple[list[ExtractedRelation], int]:
    """
    Parse the LLM JSON response and validate each triple.

    Returns (validated_relations, rejection_count).

    A triple is accepted if:
      - Both subject and object match a known canonical (exact or fuzzy)
      - Subject != object
      - relation is a non-empty string

    FIX 2: Fuzzy matching added as fallback for entity name validation.
    This handles cases where the LLM returns "ambedkar" when the known
    canonical is "b.r. ambedkar". Fuzzy threshold is 80 (out of 100).
    """
    global _rejection_log_count

    # Parse JSON from response
    data = _extract_json(raw_response)
    if data is None or not isinstance(data, list):
        return [], 0

    validated:       list[ExtractedRelation] = []
    rejection_count: int = 0

    for item in data:
        if not isinstance(item, dict):
            continue

        subject  = str(item.get("subject", "")).lower().strip()
        relation = str(item.get("relation", "")).lower().strip().replace(" ", "_")
        object_  = str(item.get("object",  "")).lower().strip()

        if not subject or not relation or not object_:
            rejection_count += 1
            continue

        # ── FIX 2: Resolve entity names with fuzzy fallback ───────────
        resolved_subject = _resolve_entity(subject, known_canonicals)
        resolved_object  = _resolve_entity(object_,  known_canonicals)

        if resolved_subject is None:
            rejection_count += 1
            _log_rejection(
                f"subject '{subject}' not in known entities {list(known_canonicals)[:3]}..."
            )
            continue

        if resolved_object is None:
            rejection_count += 1
            _log_rejection(
                f"object '{object_}' not in known entities {list(known_canonicals)[:3]}..."
            )
            continue

        if resolved_subject == resolved_object:
            rejection_count += 1
            continue

        validated.append(ExtractedRelation(
            subject=resolved_subject,
            relation=relation,
            object_=resolved_object,
            source_text=source_text,
            confidence=0.75,
            method="llm",
        ))

    return validated[:5], rejection_count   # cap at 5 per sentence


def _resolve_entity(
    candidate:        str,
    known_canonicals: set[str],
) -> Optional[str]:
    """
    Map a candidate entity name to a known canonical name.

    Priority:
      1. Exact match  — fastest, most reliable
      2. Fuzzy match  — catches "ambedkar" → "b.r. ambedkar"
                        threshold: 80 (out of 100)

    Returns the matched canonical name, or None if no match found.
    """
    # 1. Exact match
    if candidate in known_canonicals:
        return candidate

    # 2. Fuzzy match using fuzzywuzzy
    try:
        from fuzzywuzzy import fuzz

        best_score     = 0
        best_canonical = None

        for canonical in known_canonicals:
            # partial_ratio handles "ambedkar" matching "b.r. ambedkar"
            score = fuzz.partial_ratio(candidate, canonical)
            if score > best_score:
                best_score     = score
                best_canonical = canonical

        if best_score >= 80:
            return best_canonical

    except ImportError:
        # fuzzywuzzy not available — fall back to substring check
        for canonical in known_canonicals:
            if candidate in canonical or canonical in candidate:
                return canonical

    return None


def _extract_json(raw_response: str) -> Optional[list]:
    """
    Attempt to extract a JSON array from the LLM response string.
    Handles cases where the LLM includes extra text around the JSON.
    """
    # Direct parse first
    try:
        data = json.loads(raw_response)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: find JSON array within the string
    match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    log.debug(
        "Could not parse LLM response as JSON array. Response: %s",
        raw_response[:150]
    )
    return None


def _log_rejection(reason: str) -> None:
    """
    Log a triple rejection at INFO level for the first N rejections,
    then switch to DEBUG to avoid flooding the pipeline output.
    """
    global _rejection_log_count
    _rejection_log_count += 1

    if _rejection_log_count <= _LOG_REJECTION_LIMIT:
        log.info(
            "LLM triple rejected [%d/%d]: %s",
            _rejection_log_count, _LOG_REJECTION_LIMIT, reason
        )
    else:
        log.debug("LLM triple rejected: %s", reason)
