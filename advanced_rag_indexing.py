"""
Source-aware indexing utilities for an advanced RAG pipeline.

Responsibilities:
  - define multiple local data sources
  - load documents by source type
  - split documents with per-source chunking settings
  - build both a global retriever and source-specific retrievers
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from IndexingDocs_for_rag import (
    PDF_FILES,
    build_embeddings,
    build_vectorstore,
    clean_text,
    get_or_create_doc_id,
    load_tracker,
    update_tracker,
)


@dataclass 
class SourceConfig:
    source_id: str
    name: str
    source_type: str
    paths: list[str]
    chunk_size: int = 500
    chunk_overlap: int = 50
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexedSource:
    config: SourceConfig
    documents: list[Document]
    splits: list[Document]
    retriever: object


def default_source_configs() -> list[SourceConfig]:
    """Build source definitions from the current project documents."""
    configs: list[SourceConfig] = []
    for index, path in enumerate(PDF_FILES, start=1):
        path_obj = Path(path)
        configs.append(
            SourceConfig(
                source_id=f"pdf_source_{index}",
                name=path_obj.stem,
                source_type="pdf",
                paths=[str(path_obj)],
                chunk_size=700 if "attention" in path_obj.stem.lower() else 500,
                chunk_overlap=100 if "attention" in path_obj.stem.lower() else 50,
                metadata={"filename": path_obj.name},
            )
        )
    return configs


def load_text_file(path: Path) -> str:
    """Load plain text-like files into a single string."""
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return json.dumps(payload, indent=2)

    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_source_documents(source: SourceConfig, tracker: dict) -> list[Document]:
    """Load Documents for one source definition."""
    documents: list[Document] = []

    for path_str in source.paths:
        path = Path(path_str)
        doc_id = get_or_create_doc_id(str(path), tracker)

        if source.source_type == "pdf":
            with pdfplumber.open(path) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    cleaned = clean_text(text)
                    if len(cleaned) < 30:
                        continue
                    documents.append(
                        Document(
                            page_content=cleaned,
                            metadata={
                                "source": str(path),
                                "filename": path.name,
                                "doc_id": doc_id,
                                "page": page_index,
                                "source_id": source.source_id,
                                "source_name": source.name,
                                "source_type": source.source_type,
                            },
                        )
                    )
        elif source.source_type in {"txt", "md", "json"}:
            content = clean_text(load_text_file(path))
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "doc_id": doc_id,
                            "page": 1,
                            "source_id": source.source_id,
                            "source_name": source.name,
                            "source_type": source.source_type,
                        },
                    )
                )
        else:
            raise ValueError(f"Unsupported source_type: {source.source_type}")

    return documents


def split_source_documents(source: SourceConfig, documents: list[Document]) -> list[Document]:
    """Split documents using source-specific chunking parameters."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=source.chunk_size,
        chunk_overlap=source.chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    splits = splitter.split_documents(documents)
    for index, chunk in enumerate(splits):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['doc_id']}_{source.source_id}_chunk_{index}"
    return splits


def build_advanced_retrieval_index(k: int = 4) -> dict:
    """
    Build a global retriever plus one retriever per source.
    This keeps the current local setup intact while enabling routing by source.
    """
    tracker = load_tracker()
    embeddings = build_embeddings()
    source_configs = default_source_configs()

    global_splits: list[Document] = []
    indexed_sources: dict[str, IndexedSource] = {}

    for source in source_configs:
        documents = load_source_documents(source, tracker)
        splits = split_source_documents(source, documents)

        source_store = build_vectorstore(embeddings)
        if splits:
            source_store.add_documents(splits)

            for path_str in source.paths:
                path = Path(path_str)
                doc_id = get_or_create_doc_id(str(path), tracker)
                num_chunks = sum(1 for chunk in splits if chunk.metadata.get("doc_id") == doc_id)
                update_tracker(str(path), tracker, num_chunks, doc_id)

        indexed_sources[source.source_id] = IndexedSource(
            config=source,
            documents=documents,
            splits=splits,
            retriever=source_store.as_retriever(search_kwargs={"k": k}),
        )
        global_splits.extend(splits)

    global_store = build_vectorstore(embeddings)
    if global_splits:
        global_store.add_documents(global_splits)

    return {
        "embeddings": embeddings,
        "global_store": global_store,
        "global_retriever": global_store.as_retriever(search_kwargs={"k": k}),
        "indexed_sources": indexed_sources,
        "all_splits": global_splits,
    }
