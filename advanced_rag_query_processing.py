"""
Query preprocessing utilities for a more resilient RAG pipeline.

Responsibilities:
  - normalize noisy user input
  - extract lightweight routing filters from the query
  - generate retrieval-friendly query variants
  - classify the query into a coarse search mode
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


ABBREVIATION_MAP = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "nlp": "natural language processing",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "db": "database",
    "pg": "postgresql",
}


@dataclass
class ProcessedQuery:
    raw_query: str
    normalized_query: str
    retrieval_query: str
    variants: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    search_mode: str = "general"
    issues: list[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Trim whitespace and collapse repeated punctuation/noise."""
    text = text.strip()
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"([?.!,]){2,}", r"\1", text)
    return text


def expand_abbreviations(text: str) -> str:
    """Expand common technical abbreviations that often hurt retrieval quality."""
    expanded = text
    for short, full in ABBREVIATION_MAP.items():
        expanded = re.sub(rf"\b{re.escape(short)}\b", full, expanded, flags=re.IGNORECASE)
    return expanded


def extract_filters(query: str) -> tuple[str, dict]:
    """
    Parse simple routing hints such as:
      file:attention
      source:lecture
      type:pdf
    """
    filters: dict[str, str] = {}
    cleaned_query = query

    patterns = {
        "filename": r"\b(?:file|doc|document):([^\s]+)",
        "source": r"\bsource:([^\s]+)",
        "source_type": r"\btype:([^\s]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, cleaned_query, flags=re.IGNORECASE)
        if match:
            filters[key] = match.group(1).strip().lower()
            cleaned_query = re.sub(pattern, "", cleaned_query, flags=re.IGNORECASE).strip()

    return normalize_text(cleaned_query), filters


def infer_search_mode(query: str) -> str:
    """Classify the query into a coarse mode for downstream routing/debugging."""
    lowered = query.lower()
    if any(word in lowered for word in ("compare", "difference", "versus", "vs")):
        return "comparison"
    if any(word in lowered for word in ("summarize", "summary", "overview")):
        return "summary"
    if any(word in lowered for word in ("cite", "reference", "page", "where")):
        return "grounded_lookup"
    return "general"


def build_query_variants(normalized_query: str) -> list[str]:
    """Generate a small set of variants to improve recall without overcomplicating retrieval."""
    variants = [normalized_query]

    expanded = expand_abbreviations(normalized_query)
    if expanded != normalized_query:
        variants.append(expanded)

    lowered = normalized_query.lower()
    if lowered not in variants:
        variants.append(lowered)

    compact = re.sub(r"[^a-zA-Z0-9\s]", " ", normalized_query)
    compact = normalize_text(compact)
    if compact and compact not in variants:
        variants.append(compact)

    return variants


def detect_query_issues(raw_query: str, normalized_query: str) -> list[str]:
    """Flag lightweight quality issues so the app can surface them if needed."""
    issues: list[str] = []
    if len(normalized_query) < 8:
        issues.append("query_too_short")
    if raw_query != raw_query.strip():
        issues.append("extra_whitespace")
    if re.search(r"(.)\1{4,}", raw_query):
        issues.append("repeated_characters")
    return issues


def process_user_query(raw_query: str) -> ProcessedQuery:
    """Turn raw user input into a structured retrieval request."""
    normalized = normalize_text(raw_query)
    filtered_query, filters = extract_filters(normalized)
    retrieval_query = expand_abbreviations(filtered_query)
    variants = build_query_variants(filtered_query)
    search_mode = infer_search_mode(filtered_query)
    issues = detect_query_issues(raw_query, filtered_query)

    return ProcessedQuery(
        raw_query=raw_query,
        normalized_query=filtered_query,
        retrieval_query=retrieval_query,
        variants=variants,
        filters=filters,
        search_mode=search_mode,
        issues=issues,
    )
