"""
graph_builder/document_loader.py
=================================
Loads PDF documents into sentence-level TextUnit objects
with contextual windows and legal citation extraction 
for graph-based retrieval.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import config.core_config as C 

log = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")


# ─────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TextUnit:
    sentence: str
    context: str
    source: str
    page: int
    chunk_id: str
    sentence_idx: int
    # Domain Specific: Stores extracted citations (e.g., ["Article 14"])
    legal_citations: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> List[TextUnit]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    log.info("Loading PDF: %s", pdf_path.name)

    pages = _extract_pages(pdf_path)

    all_sentences: List[Tuple[str, int]] = []

    # Extract, split, and normalize
    for page_num, page_text in pages:
        sentences = _split_sentences(page_text)
        for s in sentences:
            s = _normalize_sentence(s)
            if _is_valid_sentence(s):
                all_sentences.append((s, page_num))

    log.info(
        "%s: %d valid sentences extracted from %d pages.",
        pdf_path.name, len(all_sentences), len(pages)
    )

    units = []
    stem = pdf_path.stem

    # Build TextUnits with sliding context window and citation extraction
    for idx, (sentence, page) in enumerate(all_sentences):
        context = _build_context(all_sentences, idx)
        chunk_id = _make_chunk_id(stem, page, idx, sentence)
        
        # Extract citations from the broader context block
        citations = _extract_legal_citations(context)

        units.append(TextUnit(
            sentence=sentence,
            context=context,
            source=pdf_path.name,
            page=page,
            chunk_id=chunk_id,
            sentence_idx=idx,
            legal_citations=citations,
        ))

    return units


def load_pdfs(pdf_paths: List[Path]) -> List[TextUnit]:
    all_units = []
    for path in pdf_paths:
        try:
            all_units.extend(load_pdf(path))
        except Exception as exc:
            log.error("Failed to load %s: %s", path, exc)
    return all_units


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────

def _extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    doc = fitz.open(str(pdf_path))
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")
        cleaned = _clean_page_text(raw_text)

        if cleaned.strip():
            pages.append((page_num + 1, cleaned))
        else:
            log.warning("Empty page detected: %s page %d", pdf_path.name, page_num + 1)

    doc.close()
    return pages


def _clean_page_text(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'-\n(?=[a-z])', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, hardened for Legal terminology.
    """
    abbreviations = [
        "Dr.", "Mr.", "Mrs.", "Prof.", "i.e.", "e.g.", "etc.", "cf.", 
        "U.S.", "U.K.", "E.U.", "Smt.", "Shri.", "Hon'ble", 
        "vs.", "v.", "Art.", "Sec.", "No.", "Vol.", "Ch.", 
        "Reg.", "Para.", "Cl.", "Sub-sec.", "Dept.", "Govt."
    ]

    protected = text
    for abbr in abbreviations:
        # Protect abbreviations by swapping the period for a placeholder
        protected = protected.replace(abbr, abbr.replace(".", "§"))

    # Protect numbered lists like "1.1.2" from splitting
    protected = re.sub(r'(\d)\.(\d)', r'\1§\2', protected)

    # Split on sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+', protected)

    return [s.replace("§", ".").strip() for s in raw_sentences if s.strip()]


def _normalize_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def _is_valid_sentence(sentence: str) -> bool:
    """
    Return True if a sentence is worth processing for entity extraction.
    Filters out headers, fragments, and massive run-on legal clauses.
    """
    tokens = sentence.split()

    # Guard 1: Too short (Headers, page numbers)
    if len(tokens) < getattr(C, "GRAPH_MIN_SENTENCE_TOKENS", 5):
        return False

    # Guard 2: Too long (Run-on legal clauses or OCR failures)
    if len(tokens) > getattr(C, "GRAPH_MAX_SENTENCE_TOKENS", 100):
        return False

    # Guard 3: Skip sentences that are mostly numbers (tables, page ranges)
    digit_ratio = sum(c.isdigit() for c in sentence) / max(len(sentence), 1)
    if digit_ratio > 0.5:
        return False

    # Guard 4: Skip uppercase-only lines (section headers in some PDFs)
    alpha_chars = [c for c in sentence if c.isalpha()]
    if alpha_chars and all(c.isupper() for c in alpha_chars) and len(tokens) < 8:
        return False

    return True


def _build_context(all_sentences: List[Tuple[str, int]], idx: int) -> str:
    window = getattr(C, "GRAPH_CONTEXT_WINDOW", 2)

    start = max(0, idx - window)
    end = min(len(all_sentences), idx + window + 1)

    return " ".join(all_sentences[i][0] for i in range(start, end))


def _extract_legal_citations(text: str) -> List[str]:
    """
    Extracts standard legal citations from the text using regex.
    """
    pattern = r'\b(?:Article|Art\.|Section|Sec\.|Rule|Regulation)\s+\d+[a-zA-Z]*(?:\(\d+\))?\b'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return list(set(m.title() for m in matches))


def _make_chunk_id(stem: str, page: int, idx: int, sentence: str) -> str:
    hash_input = f"{sentence}_{page}_{idx}"
    content_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    return f"{stem}_p{page:04d}_s{idx:06d}_{content_hash}"
