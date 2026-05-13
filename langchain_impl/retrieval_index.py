"""
retrieval_index.py  —  PGVector-backed persistent indexing for the advanced RAG pipeline.

What changed vs the original InMemoryVectorStore version
─────────────────────────────────────────────────────────
BEFORE  Every startup re-embedded ALL chunks from scratch into RAM.
        Vectors were lost the moment the process exited.

AFTER   Vectors are stored permanently in PostgreSQL (PGVector extension).
        On startup the code compares each PDF's MD5 hash against the
        hf_doc_change_tracker.json entry:

          • UNCHANGED  →  connect to existing PGVector collection, skip embedding
          • NEW / MODIFIED  →  delete old chunks for that source, re-embed, upsert

        Result: first run is slow (embeds everything); every subsequent run
        is near-instant unless a PDF actually changed.

PGVector collections used
─────────────────────────
  hf_global          — all chunks across all sources (used by MultiSourceRouter global retriever)
  hf_pdf_source_1    — chunks from the first PDF only
  hf_pdf_source_2    — chunks from the second PDF only
  … one collection per PDF source …

The collection names are stable across runs because source_id is derived
from the sorted PDF filename list, not from a random UUID.

Pre-requisites
──────────────
  Docker (already used in original project):
      docker run -d --name pgvector \
          -e POSTGRES_USER=Langchain \
          -e POSTGRES_PASSWORD=Langchain \
          -e POSTGRES_DB=Langchain \
          -p 6024:5432 \
          pgvector/pgvector:pg16

  Python packages  (already in project + one new):
      pip install langchain-postgres psycopg[binary]

Force a full re-index (wipes all PGVector data and re-embeds):
      python main.py memory --reindex
  or pass force_reindex=True directly to build_advanced_retrieval_index().

Public API (unchanged — drop-in replacement)
────────────────────────────────────────────
  build_advanced_retrieval_index(k, pdf_folder, force_reindex) → dict with keys:
      embeddings, global_store, global_retriever, indexed_sources, all_splits
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector

# ── path bootstrap (unchanged from original) ──────────────────────────────────
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_impl.indexing_core import (
    RAG_DOCS_FOLDER,
    PG_CONNECTION,       # postgresql+psycopg://Langchain:Langchain@localhost:6024/Langchain
    build_embeddings,
    check_document_changes,
    clean_text,
    get_or_create_doc_id,
    load_tracker,
    update_tracker,
)

# ── project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── PGVector collection names ──────────────────────────────────────────────────
# Global collection holds ALL chunks; per-source collections hold one PDF each.
PG_GLOBAL_COLLECTION   = "hf_global"
PG_SOURCE_PREFIX       = "hf_"          # e.g. "hf_pdf_source_1"

# ── PDF discovery (unchanged from original) ────────────────────────────────────
_SKIP_DIRS = {"__pycache__", ".venv", "venv", "env", ".git", "node_modules"}


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

    unique: list[Path] = []
    seen:   set[Path]  = set()
    for c in candidates:
        r = c.resolve(strict=False)
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


_BASE_DIR = _candidate_pdf_roots()[0]


def discover_pdf_files(root: Path = _BASE_DIR) -> list[Path]:
    """Recursively find every .pdf under *root*, skipping common non-project dirs."""
    found: list[Path] = []
    for p in sorted(root.rglob("*.pdf")):
        if any(skip in p.parts for skip in _SKIP_DIRS):
            continue
        found.append(p)
    return found


PDF_FILES: list[Path] = discover_pdf_files()

# ── Chunk-size heuristics (unchanged) ─────────────────────────────────────────
CHUNK_RULES: list[tuple[str, int, int]] = [
    ("attention",   700, 100),
    ("transformer", 700, 100),
    ("lecture",     500,  50),
    ("survey",      600,  80),
    ("report",      600,  80),
]
DEFAULT_CHUNK = (500, 50)


def _chunk_params(stem: str) -> tuple[int, int]:
    lower = stem.lower()
    for keyword, size, overlap in CHUNK_RULES:
        if keyword in lower:
            return size, overlap
    return DEFAULT_CHUNK


# ── Dataclasses (unchanged public interface) ───────────────────────────────────
@dataclass
class SourceConfig:
    source_id:    str
    name:         str
    source_type:  str
    paths:        list[str]
    chunk_size:   int          = 500
    chunk_overlap: int         = 50
    metadata:     dict         = field(default_factory=dict)


@dataclass
class IndexedSource:
    config:    SourceConfig
    documents: list[Document]   # raw pages  (empty when loaded from PGVector)
    splits:    list[Document]   # chunks     (empty when loaded from PGVector)
    retriever: object           # ready-to-use retriever


# ── Source-config builder (unchanged) ─────────────────────────────────────────
def default_source_configs(pdf_files: list[Path] | None = None) -> list[SourceConfig]:
    if pdf_files is None:
        pdf_files = PDF_FILES
    if not pdf_files:
        searched = "\n".join(f"  - {c}" for c in _candidate_pdf_roots())
        raise FileNotFoundError(
            f"No PDF files found.\nSearched:\n{searched}\n"
            "Pass --pdf-folder, set PDF_FOLDER, or place PDFs in rag_docs/."
        )
    configs: list[SourceConfig] = []
    for idx, path_obj in enumerate(pdf_files, start=1):
        size, overlap = _chunk_params(path_obj.stem)
        configs.append(SourceConfig(
            source_id    = f"pdf_source_{idx}",
            name         = path_obj.stem,
            source_type  = "pdf",
            paths        = [str(path_obj)],
            chunk_size   = size,
            chunk_overlap= overlap,
            metadata     = {"filename": path_obj.name},
        ))
    print(f"\n  PDF files discovered ({len(configs)}):")
    for cfg in configs:
        print(f"    • {cfg.metadata['filename']} "
              f"[chunk={cfg.chunk_size}, overlap={cfg.chunk_overlap}]")
    print()
    return configs


# ── Text loaders (unchanged) ───────────────────────────────────────────────────
def load_text_file(path: Path) -> str:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            return json.dumps(json.load(fh), indent=2)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_source_documents(source: SourceConfig, tracker: dict) -> list[Document]:
    """Load raw Document objects for one source (one PDF → one Document per page)."""
    documents: list[Document] = []
    for path_str in source.paths:
        path = Path(path_str)
        if not path.is_file():
            print(f"  [WARNING] Skipping missing file: {path}")
            continue
        doc_id = get_or_create_doc_id(str(path), tracker)
        if source.source_type == "pdf":
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                extracted   = 0
                for page_idx, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text or not text.strip():
                        continue
                    cleaned = clean_text(text)
                    if len(cleaned) < 30:
                        continue
                    extracted += 1
                    documents.append(Document(
                        page_content=cleaned,
                        metadata={
                            "source":      str(path),
                            "filename":    path.name,
                            "doc_id":      doc_id,
                            "page":        page_idx,
                            "total_pages": total_pages,
                            "source_id":   source.source_id,
                            "source_name": source.name,
                            "source_type": source.source_type,
                        },
                    ))
                print(f"  ✓ {path.name}: {extracted}/{total_pages} pages extracted")
        elif source.source_type in {"txt", "md", "json"}:
            content = clean_text(load_text_file(path))
            if content:
                documents.append(Document(
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
                ))
        else:
            raise ValueError(f"Unsupported source_type: {source.source_type!r}")
    return documents


# ── Splitter (unchanged) ───────────────────────────────────────────────────────
def split_source_documents(source: SourceConfig, documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = source.chunk_size,
        chunk_overlap = source.chunk_overlap,
        separators    = ["\n\n", "\n", ".", " "],
    )
    splits = splitter.split_documents(documents)
    for idx, chunk in enumerate(splits):
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['doc_id']}_{source.source_id}_chunk_{idx}"
        )
    return splits


# ── PGVector helpers ───────────────────────────────────────────────────────────

def _pg_collection(collection_name: str, embeddings) -> PGVector:
    """
    Connect to (or create) a PGVector collection.
    pre_delete_collection=False  →  keeps existing data; we manage deletions manually.
    use_jsonb=True               →  metadata stored as JSONB for flexible filtering.
    """
    return PGVector(
        collection_name      = collection_name,
        embeddings           = embeddings,
        connection           = PG_CONNECTION,
        use_jsonb            = True,
        pre_delete_collection= False,   # IMPORTANT: don't wipe on connect
    )


def _pg_collection_exists(collection_name: str, embeddings) -> bool:
    """
    Return True if the PGVector collection already has at least one vector.
    We use this to decide whether to skip re-indexing for an UNCHANGED source.
    """
    try:
        store = _pg_collection(collection_name, embeddings)
        # similarity_search with an empty string returns results if data exists
        results = store.similarity_search("test", k=1)
        return len(results) > 0
    except Exception:
        return False


def _pg_delete_collection(collection_name: str, embeddings) -> None:
    """
    Drop an entire PGVector collection (used before re-indexing a changed source).
    PGVector.from_documents with pre_delete_collection=True is the cleanest way.
    We pass an empty list so nothing is inserted — just the collection is wiped.
    """
    try:
        PGVector(
            collection_name      = collection_name,
            embeddings           = embeddings,
            connection           = PG_CONNECTION,
            use_jsonb            = True,
            pre_delete_collection= True,   # wipe it
        )
        print(f"    ↳ Dropped old PGVector collection: {collection_name!r}")
    except Exception as exc:
        print(f"    [WARNING] Could not drop collection {collection_name!r}: {exc}")


# ── Main index builder — PGVector persistent version ──────────────────────────

def build_advanced_retrieval_index(
    k:             int          = 4,
    pdf_folder:    Path | None  = None,
    force_reindex: bool         = False,
) -> dict:
    """
    Build a persistent global retriever + one retriever per PDF source.

    Startup behaviour
    -----------------
    First run (empty DB):
        Embeds every chunk → writes to PGVector → saves tracker JSON.
        Slow, but only happens once.

    Subsequent runs (PDFs unchanged):
        MD5 hash matches tracker → connects to existing PGVector collection.
        Embedding step is SKIPPED entirely → near-instant startup.

    PDF added or modified:
        Hash mismatch (or new file) → old collection dropped → re-embedded
        and upserted.  Other PDFs are unaffected.

    Force full rebuild:
        Pass force_reindex=True (or --reindex on the CLI).
        Drops all collections and re-embeds from scratch.

    Args:
        k:             Top-k chunks returned per query.
        pdf_folder:    Override the folder scanned for PDFs.
        force_reindex: Wipe all PGVector data and re-embed from scratch.

    Returns dict with keys:
        embeddings, global_store, global_retriever, indexed_sources, all_splits
    """

    print("\n" + "=" * 60)
    print("  BUILD ADVANCED RETRIEVAL INDEX  (PGVector backend)")
    print("=" * 60)

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    pdf_files      = discover_pdf_files(pdf_folder) if pdf_folder else PDF_FILES
    tracker        = load_tracker()
    embeddings     = build_embeddings()
    source_configs = default_source_configs(pdf_files)

    # Tracks whether the global collection needs rebuilding
    any_source_changed = False

    global_splits:   list[Document]          = []
    indexed_sources: dict[str, IndexedSource] = {}

    # ── 1. Per-source processing ───────────────────────────────────────────────
    for source in source_configs:
        collection_name = f"{PG_SOURCE_PREFIX}{source.source_id}"
        print(f"\n  ── Source: {source.name!r}  [{collection_name}]")

        # --- Decide whether this source needs re-indexing ---
        needs_reindex = force_reindex
        if not needs_reindex:
            for path_str in source.paths:
                path = Path(path_str)
                if not path.is_file():
                    needs_reindex = True          # missing file → treat as new
                    break
                changed, status = check_document_changes(str(path), tracker)
                if changed:
                    needs_reindex = True
                    print(f"    [{status}] {path.name}  →  will re-index")
                else:
                    # Extra safety: verify the collection actually exists in PG
                    if not _pg_collection_exists(collection_name, embeddings):
                        print(f"    [MISSING IN DB] {path.name}  →  will re-index")
                        needs_reindex = True
                    else:
                        print(f"    [UNCHANGED] {path.name}  →  loading from PGVector")

        # --- Re-index branch ---
        if needs_reindex:
            any_source_changed = True

            # 1a. Drop old collection for this source
            _pg_delete_collection(collection_name, embeddings)

            # 1b. Load, split, embed, upsert
            documents = load_source_documents(source, tracker)
            splits    = split_source_documents(source, documents)

            if splits:
                print(f"    Embedding {len(splits)} chunks → {collection_name!r} ...")
                # from_documents creates the collection and inserts in one shot
                source_store = PGVector.from_documents(
                    documents            = splits,
                    embedding            = embeddings,
                    collection_name      = collection_name,
                    connection           = PG_CONNECTION,
                    use_jsonb            = True,
                    pre_delete_collection= False,  # collection was already dropped above
                )
                print(f"    ✓ {len(splits)} chunks stored in PGVector")
            else:
                # Empty source — create an empty collection so future loads work
                source_store = _pg_collection(collection_name, embeddings)
                print(f"    [WARNING] No chunks produced for {source.name!r}")

            # 1c. Update the hash tracker so next run skips this source
            for path_str in source.paths:
                path = Path(path_str)
                if not path.is_file():
                    continue
                doc_id     = get_or_create_doc_id(str(path), tracker)
                num_chunks = sum(
                    1 for c in splits if c.metadata.get("doc_id") == doc_id
                )
                update_tracker(str(path), tracker, num_chunks, doc_id)

            global_splits.extend(splits)

        # --- Load-from-DB branch ---
        else:
            documents = []   # raw pages not needed — they live in PGVector
            splits    = []   # same
            source_store = _pg_collection(collection_name, embeddings)
            print(f"    ✓ Connected to existing PGVector collection")

        # --- Register the IndexedSource (same shape as before) ---
        indexed_sources[source.source_id] = IndexedSource(
            config    = source,
            documents = documents,
            splits    = splits,
            retriever = source_store.as_retriever(search_kwargs={"k": k}),
        )

    # ── 2. Global store ────────────────────────────────────────────────────────
    print(f"\n  ── Global collection: {PG_GLOBAL_COLLECTION!r}")

    if force_reindex or any_source_changed:
        # We need to rebuild the global collection.
        # Collect ALL chunks: newly embedded ones + reload from PG for unchanged sources.
        all_chunks_for_global: list[Document] = list(global_splits)  # already-embedded

        for sid, isrc in indexed_sources.items():
            if isrc.splits:
                # splits already in global_splits — skip (avoid double-insert)
                continue
            # UNCHANGED source: retrieve its chunks back from its per-source collection
            cname = f"{PG_SOURCE_PREFIX}{sid}"
            try:
                store     = _pg_collection(cname, embeddings)
                retrieved = store.similarity_search("", k=10_000)  # fetch everything
                all_chunks_for_global.extend(retrieved)
                print(f"    ↳ Pulled {len(retrieved)} chunks from {cname!r} for global rebuild")
            except Exception as exc:
                print(f"    [WARNING] Could not pull chunks from {cname!r}: {exc}")

        print(f"  Rebuilding global collection with {len(all_chunks_for_global)} chunks ...")
        _pg_delete_collection(PG_GLOBAL_COLLECTION, embeddings)

        if all_chunks_for_global:
            global_store = PGVector.from_documents(
                documents            = all_chunks_for_global,
                embedding            = embeddings,
                collection_name      = PG_GLOBAL_COLLECTION,
                connection           = PG_CONNECTION,
                use_jsonb            = True,
                pre_delete_collection= False,
            )
        else:
            global_store = _pg_collection(PG_GLOBAL_COLLECTION, embeddings)

        print(f"  ✓ Global collection rebuilt")

    else:
        # All sources unchanged — global collection is already up-to-date
        global_store = _pg_collection(PG_GLOBAL_COLLECTION, embeddings)
        print(f"  ✓ No changes detected — connected to existing global collection")

    # ── 3. Summary ────────────────────────────────────────────────────────────
    print(f"\n  Index summary:")
    print(f"    Sources      : {len(indexed_sources)}")
    print(f"    New chunks   : {len(global_splits)}  (0 = all loaded from DB)")
    print(f"    PG connection: {PG_CONNECTION}")
    print(f"    Collections  : {PG_GLOBAL_COLLECTION} + "
          f"{', '.join(f'{PG_SOURCE_PREFIX}{s}' for s in indexed_sources)}")
    print("=" * 60 + "\n")

    return {
        "embeddings":       embeddings,
        "global_store":     global_store,
        "global_retriever": global_store.as_retriever(search_kwargs={"k": k}),
        "indexed_sources":  indexed_sources,
        "all_splits":       global_splits,   # only newly-embedded chunks; [] on warm load
    }
