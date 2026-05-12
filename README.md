# Advanced RAG With LangGraph

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/framework-LangGraph-orange)
![LLM](https://img.shields.io/badge/llm-Ollama-yellow)
![Embedding](https://img.shields.io/badge/Embedding-HuggingFace-red)

This project is a local Retrieval-Augmented Generation system built with LangChain, LangGraph, HuggingFace embeddings, Ollama, and optional Tavily web search.

## Overview

The project supports three runnable workflows through [main.py](./main.py):

- `memory`: LangGraph agentic + knowledge + memory RAG CLI
- `advanced`: standalone advanced generation CLI
- `indexing`: older indexing/simple RAG demo pipeline

The codebase is now organized into two packages:

- [langchain_impl](./langchain_impl): indexing, routing, retrieval, generation
- [langgraph_impl](./langgraph_impl): graph builder, graph nodes, memory CLI

## Architecture Diagram

The LangGraph RAG extension architecture is documented here:

- [langgraph_rag_extension_architecture.svg](./langgraph_rag_extension_architecture.svg)

## Project Structure

```text
.
├─ main.py
├─ test_project_structure.py
├─ langchain_impl/
│  ├─ __init__.py
│  ├─ indexing_core.py
│  ├─ retrieval_index.py
│  ├─ legacy_indexing.py
│  ├─ query_processing.py
│  ├─ retriever_router.py
│  └─ generation_pipeline.py
├─ langgraph_impl/
│  ├─ __init__.py
│  ├─ graph_builder.py
│  ├─ graph_nodes.py
│  └─ memory_cli.py
└─ rag_docs/
```

## Module Roles

- [main.py](./main.py): single launcher for the project
- [langchain_impl/indexing_core.py](./langchain_impl/indexing_core.py): shared indexing utilities, prompt helpers, trackers, embeddings, vector store setup
- [langchain_impl/retrieval_index.py](./langchain_impl/retrieval_index.py): current source-aware PDF indexing and retriever builder
- [langchain_impl/legacy_indexing.py](./langchain_impl/legacy_indexing.py): older indexing implementation kept as reference
- [langchain_impl/query_processing.py](./langchain_impl/query_processing.py): query normalization, filter extraction, query variants
- [langchain_impl/retriever_router.py](./langchain_impl/retriever_router.py): source selection and routed retrieval
- [langchain_impl/generation_pipeline.py](./langchain_impl/generation_pipeline.py): advanced generation CLI and retrieval/generation helpers
- [langgraph_impl/graph_builder.py](./langgraph_impl/graph_builder.py): `AgentState`, graph topology, graph compilation, router builder
- [langgraph_impl/graph_nodes.py](./langgraph_impl/graph_nodes.py): query, routing, retrieval, grading, generation, and memory-related nodes
- [langgraph_impl/memory_cli.py](./langgraph_impl/memory_cli.py): session-oriented memory CLI built on LangGraph
- [test_project_structure.py](./test_project_structure.py): basic package/workflow structure tests

## Usage

Use `main.py` as the primary entry point.

List workflows:

```powershell
.\.venv\Scripts\python.exe main.py --list
```

Run the default workflow:

```powershell
.\.venv\Scripts\python.exe main.py
```

Run a specific workflow:

```powershell
.\.venv\Scripts\python.exe main.py memory
.\.venv\Scripts\python.exe main.py advanced
.\.venv\Scripts\python.exe main.py indexing
```

Run the memory workflow with a custom PDF folder:

```powershell
.\.venv\Scripts\python.exe main.py memory --pdf-folder .\rag_docs
```

You can also run package modules directly:

```powershell
.\.venv\Scripts\python.exe -m langgraph_impl.memory_cli
.\.venv\Scripts\python.exe -m langchain_impl.generation_pipeline
.\.venv\Scripts\python.exe -m langchain_impl.indexing_core
```

## Workflow Mapping

`main.py` launches these modules:

- `memory` -> `langgraph_impl.memory_cli`
- `advanced` -> `langchain_impl.generation_pipeline`
- `indexing` -> `langchain_impl.indexing_core`

Helper modules listed by `main.py --list`:

- `langgraph_impl.graph_builder`
- `langgraph_impl.graph_nodes`
- `langchain_impl.retrieval_index`
- `langchain_impl.retriever_router`
- `langchain_impl.query_processing`
- `langchain_impl.legacy_indexing`

## LangGraph Flow

The memory-enabled graph follows this path:

```text
START
  -> query_node
  -> router_node
  -> route_to_retriever
      -> local_rag_node
      -> multiquery_rag_node
      -> rag_fusion_node
      -> web_search_node
      -> hybrid_node
  -> grader_node
  -> route_after_grading
      -> retry retrieval mode, or
      -> generation_node
  -> END
```

## Retrieval Modes

- `rewrite_rag`: single rewrite, then local retrieval
- `multiquery_rag`: multiple retrieval queries, then deduplication
- `rag_fusion`: multiple retrieval queries with reciprocal rank fusion
- `web_search`: Tavily-based web retrieval
- `hybrid`: local retrieval plus web retrieval

## Prompt Types

- `default`
- `concise`
- `detailed`
- `bullet`

## Memory CLI Commands

Available in `langgraph_impl.memory_cli`:

```text
mode
prompt
session <name>
sessions
state
history
new
stream:<query>
batch:<q1>|<q2>
exit
<query>
```

## Advanced CLI Commands

Available in `langchain_impl.generation_pipeline`:

```text
prompt
mode
preview:<query>
rewrite:<query>
route:<query>
stream:<query>
batch:<q1>|<q2>
exit
<query>
```

## Data And Paths

- Default PDF folder: [rag_docs](./rag_docs)
- Tracker file: `hf_doc_change_tracker.json`
- Record manager DB: `hf_record_manager.db`
- Older tracker file: `doc_change_tracker.json`

`indexing_core.py` now resolves project-relative paths from the repository root, so the package split does not change where trackers and `rag_docs` live.

## Requirements

Main runtime dependencies:

- `langchain`
- `langgraph`
- `langchain-core`
- `langchain-ollama`
- `langchain-huggingface`
- `langchain-text-splitters`
- `pdfplumber`
- `sentence-transformers`
- optional `langchain-community`
- optional `TAVILY_API_KEY`
- optional PostgreSQL/PGVector support for the older indexing demo

Ollama setup:

```powershell
ollama pull llama3.2:1b
ollama serve
```

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Testing

Basic structure validation:

```powershell
.\.venv\Scripts\python.exe -m unittest test_project_structure.py
```

Compile check:

```powershell
.\.venv\Scripts\python.exe -m compileall main.py langchain_impl langgraph_impl test_project_structure.py
```

## Recommended Workflow

1. Put PDFs into [rag_docs](./rag_docs) or pass a custom folder to `main.py memory --pdf-folder`.
2. Start Ollama and make sure `llama3.2:1b` is available.
3. Run `.\.venv\Scripts\python.exe main.py`.
4. Choose the retrieval mode based on the question type.
5. Use the same session for follow-up questions when you want memory.

## Notes

- On this machine, plain `python` may resolve to the WindowsApps launcher stub instead of a working interpreter.
- Prefer `.\.venv\Scripts\python.exe` for local runs in this project.
- `MemorySaver` in the current LangGraph CLI is process-memory only; it does not persist across full restarts.

## License

No license file is currently included. Until a license is added, all rights are reserved by the repository owner by default.
