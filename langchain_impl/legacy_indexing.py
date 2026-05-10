"""
The current file is legacy_indexing.py located in the langchain_impl directory of your workspace. 
It's a Python module focused on source-aware indexing utilities for an advanced Retrieval-Augmented Generation (RAG) pipeline using LangChain.

Key Purpose and Responsibilities
This file handles the indexing of documents from multiple local data sources, enabling both global and source-specific retrieval. 
Its main goals are:
--Defining and configuring multiple data sources (e.g., PDFs, text files).
--Loading documents based on their source type (PDF, TXT, MD, JSON).
--Splitting documents into chunks with customizable settings per source.
--Building vector stores and retrievers for both global search and source-specific queries.

Imported Modules from indexing_core.py and Their Uses in legacy_indexing.py
    PDF_FILES (list of PDF paths):
        Purpose: Auto-discovered PDF files from RAG_DOCS_FOLDER.
        Use: Iterated in default_source_configs() to create SourceConfig for each PDF with dynamic chunk sizes. 
    build_embeddings (→ HuggingFaceEmbeddings):
        Purpose: Initializes CPU-based HuggingFace embedding model for normalized vectors.
        Use: Called in build_advanced_retrieval_index() to get embeddings for vector stores.
    build_vectorstore (→ InMemoryVectorStore):
        Purpose: Creates empty in-memory vector store.
        Use: Called for global and per-source vector stores in build_advanced_retrieval_index().
    clean_text (str → str):
        Purpose: Removes garbled chars, normalizes whitespace.
        Use: Preprocesses text in load_source_documents() for PDFs and text files.
    get_or_create_doc_id (str, dict → str):
        Purpose: Returns existing or new UUID doc_id from tracker.
        Use: Assigns doc_id in load_source_documents() for metadata consistency.
    load_tracker (→ dict):
        Purpose: Loads JSON tracker file or returns empty dict.
        Use: Loads tracker at start of build_advanced_retrieval_index() for change tracking.
    update_tracker (str, dict, int, str → None):
        Purpose: Updates tracker with hash, timestamp, chunk count, and saves to disk.
        Use: Updates tracker per source in build_advanced_retrieval_index() after indexing.


Main Components
Imports: Relies on LangChain components (e.g., Document, RecursiveCharacterTextSplitter), pdfplumber for PDF processing, 
and local modules from indexing_core (e.g., for embeddings, vector stores, and tracking).

Data Classes:
SourceConfig: Defines a data source with ID, name, type, paths, chunking parameters, and metadata.
IndexedSource: Represents an indexed source with its config, loaded documents, splits, and retriever.

Functions:
--default_source_configs(): Generates source configs from a list of PDF files (PDF_FILES), with dynamic chunk sizes based on filename (e.g., larger chunks for "attention"-related files).
--load_text_file(path): Loads plain text or JSON files into strings.
--load_source_documents(source, tracker): Loads documents for a source, extracting text from PDFs (page-by-page) or text files, and assigns metadata like doc ID, page number, and source info.
--split_source_documents(source, documents): Splits documents into chunks using source-specific settings and adds chunk IDs.
--build_advanced_retrieval_index(k=4): The core function that builds a global vector store and retriever, plus per-source retrievers. It loads, splits, and indexes documents, updating a tracker for change detection.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_impl.indexing_core import (
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
