"""
graph_builder/relationship_extractor.py
=========================================
Extracts relationships between entities using two methods.
Domain Focus: Digital Rights and Laws

METHOD A — Dependency Parse (confidence 0.90):
  Uses spaCy's dependency tree to find subject-verb-object triples.
  Filters out weak verbs (be, have, do).

METHOD B — Co-occurrence "Dependency Glue" (confidence 0.10):
  When two entities share a sentence, stores a weak edge
  ONLY IF they meet strict structural criteria:
    1. The sentence contains an active verb.
    2. No strong Method A relation already exists between them.
    3. Token distance <= 5 OR they share the same syntactic head.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from graph_builder.entity_extractor import ExtractedEntity, _NLP

log = logging.getLogger(__name__)

@dataclass
class ExtractedRelation:
    subject: str
    relation: str
    object_: str
    source_text: str
    confidence: float
    method: str

def extract_all_relations(text: str, entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
    if len(entities) < 2:
        return []

    doc = _NLP(text)
    dep_relations = extract_dependency_relations(doc, entities)
    cooc_relations = extract_cooccurrence_relations(doc, entities, dep_relations)

    log.debug("Extracted %d dependency and %d co-occurrence relations", 
              len(dep_relations), len(cooc_relations))

    return dep_relations + cooc_relations

def extract_dependency_relations(doc, entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
    triples = []
    seen_triples = set()
    token_to_entity = {}

    for ent in entities:
        span = doc.char_span(ent.start, ent.end, alignment_mode="expand")
        if not span:
            span = doc.char_span(ent.start, ent.end, alignment_mode="contract")
        if span:
            token_to_entity[span.root.i] = ent.canonical

    for token in doc:
        if token.pos_ == "VERB":
            relation_label = token.lemma_.lower()
            if relation_label in {"be", "have", "do"}:
                continue
                
            subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            objects = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr", "dative", "oprd")]

            for subj in subjects:
                for obj in objects:
                    e1 = token_to_entity.get(subj.i)
                    e2 = token_to_entity.get(obj.i)

                    if e1 and e2 and e1 != e2:
                        triple_key = (e1, relation_label, e2)
                        if triple_key not in seen_triples:
                            seen_triples.add(triple_key)
                            triples.append(ExtractedRelation(
                                subject=e1,
                                relation=relation_label,
                                object_=e2,
                                source_text=doc.text,
                                confidence=0.90,
                                method="dependency_parse"
                            ))
    return triples

def extract_cooccurrence_relations(
    doc,
    entities: list[ExtractedEntity],
    existing_relations: list[ExtractedRelation]
) -> list[ExtractedRelation]:
    if len(entities) < 2:
        return []

    if not any(token.pos_ == "VERB" for token in doc):
        return []

    existing_pairs = set()
    for rel in existing_relations:
        existing_pairs.add((rel.subject, rel.object_))
        existing_pairs.add((rel.object_, rel.subject))

    ent_spans = {}
    for e in entities:
        span = doc.char_span(e.start, e.end, alignment_mode="expand")
        if not span:
            span = doc.char_span(e.start, e.end, alignment_mode="contract")
        if span:
            ent_spans[e.canonical] = span

    triples = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            e1 = entities[i].canonical
            e2 = entities[j].canonical

            if e1 == e2 or (e1, e2) in existing_pairs:
                continue

            span1, span2 = ent_spans.get(e1), ent_spans.get(e2)
            if not span1 or not span2:
                continue

            tok_dist = abs(span2.start - span1.end) if span1.start < span2.start else abs(span1.start - span2.end)
            shared_head = (span1.root.head == span2.root.head)

            if tok_dist <= 5 or shared_head:
                for subj, obj in [(e1, e2), (e2, e1)]:
                    triples.append(ExtractedRelation(
                        subject=subj,
                        relation="co_occurs_near_bidirectional",
                        object_=obj,
                        source_text=doc.text,
                        confidence=0.10,
                        method="cooccurrence"
                    ))
    return triples
