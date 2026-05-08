"""
Routing layer for an advanced RAG pipeline.

Responsibilities:
  - decide which source retrievers should be used for a query
  - retrieve from one or more source-specific indexes
  - merge and deduplicate routed results
The main class in this file is MultiSourceRouter, which takes a user query, 
processes it using the utilities from advanced_rag_query_processing.py, and 
then applies routing logic to determine which indexed sources to query. 
It uses both explicit filters extracted from the query and soft keyword matching 
against source metadata to make informed routing decisions. The router then 
retrieves documents from the selected sources, merges and deduplicates them, 
and returns a structured RoutedRetrieval object that includes the processed 
query information, the route taken, the sources used, and the retrieved documents. 
This design allows for a flexible and resilient RAG pipeline that can adapt to 
noisy user input and varying source characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass

from advanced_rag_query_processing import ProcessedQuery, process_user_query


@dataclass
class RoutedRetrieval:
    processed_query: ProcessedQuery
    route_name: str
    selected_sources: list[str]
    documents: list


class MultiSourceRouter:
    def __init__(self, indexed_sources: dict[str, object], global_retriever, default_k: int = 4):
        self.indexed_sources = indexed_sources
        self.global_retriever = global_retriever
        self.default_k = default_k

    def route(self, raw_query: str, k: int | None = None) -> RoutedRetrieval:
        processed = process_user_query(raw_query)
        top_k = k or self.default_k
        selected_sources = self._select_sources(processed)

        if not selected_sources:
            docs = self._retrieve_from_variants(self.global_retriever, processed.variants, top_k)
            return RoutedRetrieval(
                processed_query=processed,
                route_name="global_fallback",
                selected_sources=[],
                documents=self._deduplicate_docs(docs)[:top_k],
            )

        docs = []
        per_source_k = max(1, top_k)
        for source_id in selected_sources:
            indexed_source = self.indexed_sources[source_id]
            docs.extend(self._retrieve_from_variants(indexed_source.retriever, processed.variants, per_source_k))

        return RoutedRetrieval(
            processed_query=processed,
            route_name="targeted_multi_source" if len(selected_sources) > 1 else "targeted_single_source",
            selected_sources=selected_sources,
            documents=self._deduplicate_docs(docs)[: max(top_k, len(selected_sources))],
        )

    def _select_sources(self, processed: ProcessedQuery) -> list[str]:
        """Choose source ids using explicit filters first, then soft keyword matching."""
        filters = processed.filters
        selected: list[str] = []

        for source_id, indexed_source in self.indexed_sources.items():
            config = indexed_source.config
            haystack = " ".join(
                [
                    config.source_id.lower(),
                    config.name.lower(),
                    config.metadata.get("filename", "").lower(),
                    config.source_type.lower(),
                ]
            )

            if filters.get("filename") and filters["filename"] in haystack:
                selected.append(source_id)
                continue

            if filters.get("source") and filters["source"] in haystack:
                selected.append(source_id)
                continue

            if filters.get("source_type") and filters["source_type"] == config.source_type.lower():
                selected.append(source_id)
                continue

        if selected:
            return selected

        query_terms = set(processed.retrieval_query.lower().split())
        scored_sources: list[tuple[int, str]] = []

        for source_id, indexed_source in self.indexed_sources.items():
            config = indexed_source.config
            source_terms = set((config.name + " " + config.metadata.get("filename", "")).lower().replace(".", " ").split())
            overlap = len(query_terms & source_terms)
            if overlap:
                scored_sources.append((overlap, source_id))

        scored_sources.sort(reverse=True)
        return [source_id for _, source_id in scored_sources[:2]]

    @staticmethod
    def _retrieve_from_variants(retriever, variants: list[str], k: int) -> list:
        docs = []
        for variant in variants:
            docs.extend(retriever.invoke(variant)[:k])
        return docs

    @staticmethod
    def _deduplicate_docs(documents: list) -> list:
        deduped = []
        seen: set[str] = set()

        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = f"{doc.metadata.get('filename', '?')}:{doc.metadata.get('page', '?')}:{hash(doc.page_content)}"
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped.append(doc)

        return deduped
