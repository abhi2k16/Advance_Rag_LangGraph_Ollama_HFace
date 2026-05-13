"""
Source-aware indexing utilities for an advanced RAG pipeline.

Responsibilities:
  - auto-discover all PDF files in the project folder (no hardcoded list)
  - load documents by source type
  - split documents with per-source chunking settings
  - build both a global retriever and source-specific retrievers
  - track document hashes and last-indexed timestamps in a JSON file to avoid unnecessary re-indexing
The imported functions are used in the main retrieval.py and generation.py pipelines, 
but the core indexing logic is all contained here in advanced_rag_indexing_2.py for better modularity and testability.
The functions in this file are:
  - discover_pdf_files()           # Find all PDFs in the project folder
  - default_source_configs()       # Build one SourceConfig per PDF
  - load_source_documents()        # Load raw documents from source files
  - split_source_documents()       # Split documents into chunks with source-specific settings
  - build_advanced_retrieval_index() # Main function that ties it all together
"""

from __future__ import annotations # for Python 3.10 compatibility

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure the parent folder is in the path for local imports, regardless of how this module is run
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Local imports (make sure to update these if you rename the file or move it to a subfolder)
from langchain_impl.indexing_core import (
    RAG_DOCS_FOLDER,
    build_embeddings,
    build_vectorstore,
    clean_text,
    get_or_create_doc_id,
    load_tracker,
    update_tracker,
)


# ── Auto-discover all PDFs in the same folder as this script ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _candidate_pdf_roots() -> list[Path]:
    """Return candidate PDF roots in priority order, without duplicates."""
    candidates: list[Path] = []

    env_pdf_folder = os.environ.get("PDF_FOLDER")
    if env_pdf_folder:
        candidates.append(Path(env_pdf_folder).expanduser())

    if RAG_DOCS_FOLDER:
        candidates.append(Path(RAG_DOCS_FOLDER).expanduser())

    candidates.append(PROJECT_ROOT / "rag_docs")
    if PROJECT_ROOT.parent != PROJECT_ROOT:
        candidates.append(PROJECT_ROOT.parent / "rag_docs")

    candidates.append(PROJECT_ROOT)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return unique_candidates


_BASE_DIR = _candidate_pdf_roots()[0]  # Preferred root for PDF discovery

# Folders to skip so we don't accidentally index venv / cache PDFs
_SKIP_DIRS = {"__pycache__", ".venv", "venv", "env", ".git", "node_modules"} 

def discover_pdf_files(root: Path = _BASE_DIR) -> list[Path]:
    """
    Recursively find every .pdf file under `root`, skipping common
    non-project directories.  Sorted so the order is deterministic.
    """
    found: list[Path] = []
    for pdf_path in sorted(root.rglob("*.pdf")):
        # skip any path that passes through a blacklisted folder
        if any(skip in pdf_path.parts for skip in _SKIP_DIRS):
            continue
        found.append(pdf_path)
    return found

# Auto-discover PDFs at the module level so it's only done once and can be overridden by passing a 
# custom list to build_advanced_retrieval_index()
PDF_FILES: list[Path] = discover_pdf_files() 


# ── Chunk-size heuristics ─────────────────────────────────────────────────────
# Map lowercase keywords found in a filename → (chunk_size, chunk_overlap).
# The first matching rule wins; everything else falls back to the default.
CHUNK_RULES: list[tuple[str, int, int]] = [
    ("attention",   700, 100),
    ("transformer", 700, 100),
    ("lecture",     500,  50),
    ("survey",      600,  80),
    ("report",      600,  80),
]
DEFAULT_CHUNK = (500, 50) # default chunk size and overlap if no keywords match


def _chunk_params(stem: str) -> tuple[int, int]:
    lower = stem.lower()
    for keyword, size, overlap in CHUNK_RULES:
        if keyword in lower:
            return size, overlap
    return DEFAULT_CHUNK


# ── Dataclasses ───────────────────────────────────────────────────────────────
# These are the core data structures used to define the indexing pipeline.
@dataclass
class SourceConfig:
    source_id: str
    name: str
    source_type: str
    paths: list[str]
    chunk_size: int = 500
    chunk_overlap: int = 50
    metadata: dict = field(default_factory=dict)

# This class bundles everything related to one source, including its config, loaded documents,
# split chunks, and the retriever built on those chunks.
@dataclass
class IndexedSource:
    config: SourceConfig                               # source definition and metadata
    documents: list[Document]                          # raw loaded documents (e.g. one per PDF page)
    splits: list[Document]                             # split chunks derived from the raw documents
    retriever: object                                  # retriever built on the splits, ready for querying   


# ── Source config builder ─────────────────────────────────────────────────────#
def default_source_configs(pdf_files: list[Path] | None = None) -> list[SourceConfig]:
    """
    Build one SourceConfig per discovered PDF.
    Pass an explicit list to override auto-discovery (useful for testing).
    """
    if pdf_files is None:
        pdf_files = PDF_FILES

    if not pdf_files:
        searched_roots = "\n".join(
            f"  - {candidate}" for candidate in _candidate_pdf_roots()
        )
        raise FileNotFoundError(
            "No PDF files found.\n"
            f"Searched these locations:\n{searched_roots}\n"
            "Pass `--pdf-folder`, set `PDF_FOLDER`, or place PDFs in `rag_docs`."
        )

    configs: list[SourceConfig] = []
    for index, path_obj in enumerate(pdf_files, start=1):
        chunk_size, chunk_overlap = _chunk_params(path_obj.stem)
        configs.append(
            SourceConfig(
                source_id=f"pdf_source_{index}",
                name=path_obj.stem,
                source_type="pdf",
                paths=[str(path_obj)],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata={"filename": path_obj.name},
            )
        )

    print(f"\n  PDF files discovered ({len(configs)}):")
    for cfg in configs:
        print(f"    • {cfg.metadata['filename']}  "
              f"[chunk={cfg.chunk_size}, overlap={cfg.chunk_overlap}]")
    print()
    return configs


# ── Loaders ───────────────────────────────────────────────────────────────────#
def load_text_file(path: Path) -> str:
    """Load plain text-like files into a single string."""
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return json.dumps(payload, indent=2)

    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()

# -------load_source_documents is in advanced_rag_indexing_2.py since it's shared with the test suite -------#
def load_source_documents(source: SourceConfig, tracker: dict) -> list[Document]:
    """Load Documents for one source definition."""
    documents: list[Document] = [] # This will hold the raw loaded documents (e.g. one per PDF page)

    for path_str in source.paths:            # Loop over all paths defined for this source (usually just one PDF file)
        path = Path(path_str)                

        if not path.is_file():
            print(f"  [WARNING] Skipping missing file: {path}")
            continue

        doc_id = get_or_create_doc_id(str(path), tracker) # Get a unique doc_id for this file, used for tracking in the JSON tracker

        if source.source_type == "pdf":
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                extracted   = 0
                for page_index, page in enumerate(pdf.pages, start=1): # Loop over each page in the PDF
                    text = page.extract_text()
                    if not text or not text.strip(): # Skip empty pages
                        continue
                    cleaned = clean_text(text)
                    if len(cleaned) < 30:
                        continue
                    extracted += 1
                    documents.append(
                        Document(
                            page_content=cleaned,
                            metadata={
                                "source":      str(path),
                                "filename":    path.name,
                                "doc_id":      doc_id,
                                "page":        page_index,
                                "total_pages": total_pages,
                                "source_id":   source.source_id,
                                "source_name": source.name,
                                "source_type": source.source_type,
                            },
                        )
                    )
            print(f"    ✓ {path.name}: {extracted}/{total_pages} pages extracted")

        elif source.source_type in {"txt", "md", "json"}:
            content = clean_text(load_text_file(path))
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source":      str(path),
                            "filename":    path.name,
                            "doc_id":      doc_id,
                            "page":        1,
                            "source_id":   source.source_id,
                            "source_name": source.name,
                            "source_type": source.source_type,
                        },
                    )
                )
        else:
            raise ValueError(f"Unsupported source_type: {source.source_type}")

    return documents


# ── Splitter ──────────────────────────────────────────────────────────────────#
def split_source_documents(source: SourceConfig, documents: list[Document]) -> list[Document]:
    """Split documents using source-specific chunking parameters."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=source.chunk_size,
        chunk_overlap=source.chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    splits = splitter.split_documents(documents)
    for index, chunk in enumerate(splits):
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['doc_id']}_{source.source_id}_chunk_{index}"
        )
    return splits


# ── Main index builder ────────────────────────────────────────────────────────# 
# This is the main function that builds both the global retriever and the source-specific retrievers.
def build_advanced_retrieval_index(
    k: int = 4,
    pdf_folder: Path | None = None, # Optional override for the PDF discovery folder (defaults to the folder containing this script)
) -> dict:
    """
    Build a global retriever plus one retriever per PDF source.

    Args:
        k:          Number of chunks to retrieve per query.
        pdf_folder: Override the folder to scan for PDFs.
                    Defaults to the folder containing this script.
    """
    # Re-discover PDFs if a custom folder is given
    if pdf_folder is not None:
        pdf_files = discover_pdf_files(pdf_folder)
    else:
        pdf_files = PDF_FILES

    tracker        = load_tracker()                     # Load the JSON tracker that keeps track of document hashes and last-indexed timestamps
    embeddings     = build_embeddings()                 # Build the embedding model (e.g. SentenceTransformer)
    source_configs = default_source_configs(pdf_files)  # Build the list of SourceConfig objects based on the discovered PDFs (or the provided list)

    global_splits:   list[Document]          = []       # This will hold all the split chunks from all sources, used to build the global retriever
    indexed_sources: dict[str, IndexedSource] = {}      # This will hold the IndexedSource objects for each source, keyed by source_id

    for source in source_configs:
        documents = load_source_documents(source, tracker)
        splits    = split_source_documents(source, documents)

        source_store = build_vectorstore(embeddings)   # Build a new vector store for this source
        if splits:
            source_store.add_documents(splits)
            for path_str in source.paths:
                path   = Path(path_str)
                doc_id = get_or_create_doc_id(str(path), tracker)
                num_chunks = sum(
                    1 for chunk in splits
                    if chunk.metadata.get("doc_id") == doc_id
                )
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

    print(f"\n  Index summary:")
    print(f"    Sources  : {len(indexed_sources)}")
    print(f"    Chunks   : {len(global_splits)}")

    return {
        "embeddings":       embeddings,
        "global_store":     global_store,
        "global_retriever": global_store.as_retriever(search_kwargs={"k": k}),
        "indexed_sources":  indexed_sources,
        "all_splits":       global_splits,
    }
