# Advanced RAG With LangGraph

This folder contains a local Retrieval Augmented Generation (RAG) project built with LangChain, LangGraph, HuggingFace embeddings, Ollama, and optional Tavily web search.

## Project Summary

This project is a complete local RAG system for answering questions over document collections such as PDFs. It indexes documents, builds vector search indexes, routes queries to the right retrieval strategy, grades candidate results, and generates final answers with an LLM. When using the LangGraph memory-enabled flow, the system can also keep conversation history and support follow-up questions.

## Libraries and Their Roles

- **LangChain**: Orchestrates the RAG workflow, embedding generation, vector store creation, and prompt-based LLM calls.
- **LangGraph**: Defines and executes the graph-based retrieval and generation pipeline, stores state, and manages memory-aware conversation flows.
- **HuggingFace**: Provides embeddings for text chunks and model interfaces for local or hosted embedding generation.
- **Ollama**: Executes local LLM calls for generation and chat completion inside the project.
- **Tavily (optional)**: Adds external web search capability for hybrid retrieval when the answer may require current or online information.

The main workflow indexes PDFs, routes questions to the right retrieval strategy, grades retrieved documents, generates answers, and can preserve conversation history across turns through LangGraph memory.

## Main Entry Points

Use `langgraph_rag_memory.py` for the complete LangGraph agentic + knowledge + memory RAG CLI:

```powershell
python langgraph_rag_memory.py
```

Expected terminal output at startup:

```text
============================================================
LANGGRAPH AGENTIC + KNOWLEDGE + MEMORY RAG
Embeddings : HuggingFace all-MiniLM-L6-v2
LLM        : Ollama llama3.2:1b
Memory     : LangGraph MemorySaver
============================================================


============================================================
Building retrieval index...
============================================================

  PDF files discovered (<number>):
    - <pdf-name>.pdf  [chunk=<size>, overlap=<overlap>]

    - <pdf-name>.pdf: <extracted_pages>/<total_pages> pages extracted

  Index summary:
    Sources  : <number>
    Chunks   : <number>

  Indexed sources : <number>
  Total chunks    : <number>
============================================================


[MemoryRAGSession] thread_id = <random-uuid>
  mode=rewrite_rag  prompt=default  k=6  max_retries=2
[SessionManager] Created session 'default' (thread=<same-random-uuid>)
============================================================
Commands: mode | prompt | session <name> | sessions | state
          history | new | stream:<q> | batch:<q1>|<q2> | exit
============================================================

[default][rewrite_rag][default] Query:
```

Expected output after entering a normal question:

```text
  Generating answer (mode=rewrite_rag, prompt=default)...

[query_node]
  Raw query       : <your question>
  Retrieval query : <processed query>
  Search mode     : <mode>
  Filters         : <filters or none>
  Issues          : <issues or none>

[router_node]
  Route           : <route name>
  Sources         : <sources>
  Docs retrieved  : <number>

[route_to_retriever] mode='rewrite_rag' -> local_rag_node

[local_rag_node]
  Rewritten query : <rewritten query>
  Docs retrieved  : <number>

[grader_node] Grading <number> docs (retry #0)...
  [1] <KEEP/DROP> <filename> p.<page>
  Relevant: <x>/<y>  (<percent>)
  Grade: PASS

[route_after_grading] -> generation_node

[generation_node]
  Prompt type : default
  Context len : <number> chars
  History turns: <number>
  Answer len  : <number> chars

Answer:
<final generated answer>

------------------------------------------------------------
```

Optional custom PDF folder:

```powershell
python. langgraph_rag_memory.py C:\path\to\pdf_folder
```

Use `advanced_generation_rag.py` for the standalone advanced generation CLI without LangGraph state memory:

```powershell
python advanced_generation_rag.py
```

Use `IndexingDocs_for_rag.py` directly only when you want to run the older indexing and simple RAG demo pipeline:

```powershell
python IndexingDocs_for_rag.py
```

## File Structure And Purpose

| File | Purpose |
| --- | --- |
| `langgraph_rag_memory.py` | Main memory-enabled CLI. Creates sessions, compiles the graph with `MemorySaver`, registers the runtime router, and exposes commands such as `mode`, `prompt`, `state`, `history`, `stream:<query>`, and `batch:<q1>|<q2>`. |
| `langgraph_rag_graph.py` | Defines `AgentState`, builds the LangGraph topology, wires nodes and conditional edges, compiles the graph, and builds the shared `MultiSourceRouter`. |
| `langgraph_rag_nodes.py` | Contains all LangGraph node functions: query preprocessing, routing, retrieval modes, grading, generation, memory history formatting, and conditional edge routing. |
| `advanced_generation_rag.py` | Standalone advanced RAG generation CLI and reusable retrieval helpers. Provides query rewrite, multi-query RAG, RAG fusion, web search, hybrid retrieval, prompt preview, streaming, and batch generation. |
| `advanced_rag_indexing_2.py` | Current source-aware indexing module. Auto-discovers PDFs, loads pages, applies source-specific chunking, builds per-source retrievers and a global retriever. This is used by the LangGraph flow. |
| `advanced_rag_router.py` | Routes processed queries to specific sources or the global retriever. Returns selected sources, route name, and retrieved documents. |
| `advanced_rag_query_processing.py` | Normalizes user queries, expands abbreviations, extracts filters like `file:` and `source:`, detects query type, and creates query variants. |
| `IndexingDocs_for_rag.py` | Shared base utilities: text cleaning, document change tracking, HuggingFace embeddings, vector store creation, document formatting, and prompt templates. |
| `advanced_rag_indexing.py` | Earlier source-aware indexing version. Kept as a reference/alternate implementation; the newer LangGraph path imports `advanced_rag_indexing_2.py`. |

## Import Relationships

`langgraph_rag_memory.py` imports:

- `MemorySaver` from LangGraph for checkpoint memory.
- `build_router`, `compile_graph`, and `AgentState` from `langgraph_rag_graph.py`.
- `register_runtime_router` from `langgraph_rag_nodes.py` so live router objects are available at runtime without being checkpointed.

`langgraph_rag_graph.py` imports:

- `build_advanced_retrieval_index` from `advanced_rag_indexing_2.py`.
- `MultiSourceRouter` from `advanced_rag_router.py`.
- `ProcessedQuery` from `advanced_rag_query_processing.py`.
- Node functions and route functions from `langgraph_rag_nodes.py`.

`langgraph_rag_nodes.py` imports:

- `process_user_query` from `advanced_rag_query_processing.py`.
- retrieval/generation helpers from `advanced_generation_rag.py`, including rewrite templates, web search builder, multi-query parser, and reciprocal rank fusion.
- `build_prompt` and `format_docs` from `IndexingDocs_for_rag.py`.
- `ChatOllama`, `ChatPromptTemplate`, and `StrOutputParser` for LLM calls.

`advanced_generation_rag.py` imports:

- `build_prompt` and `format_docs` from `IndexingDocs_for_rag.py`.
- `build_advanced_retrieval_index` from `advanced_rag_indexing_2.py`.
- `MultiSourceRouter` from `advanced_rag_router.py`.

`advanced_rag_indexing_2.py` imports:

- `build_embeddings`, `build_vectorstore`, `clean_text`, `load_tracker`, `get_or_create_doc_id`, and `update_tracker` from `IndexingDocs_for_rag.py`.
- `Document`, `pdfplumber`, and `RecursiveCharacterTextSplitter` for loading and chunking PDFs.

`advanced_rag_router.py` imports:

- `ProcessedQuery` and `process_user_query` from `advanced_rag_query_processing.py`.

## LangGraph Flow

The memory-enabled graph runs this sequence:

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

Important details:

- `query_node` normalizes the user query and builds a `ProcessedQuery`.
- `router_node` selects sources and retrieves initial documents.
- retrieval nodes fetch context using the active retrieval mode.
- `grader_node` checks whether retrieved documents are relevant and can retry retrieval.
- `generation_node` builds the final prompt, includes recent conversation history, and stores the new user/assistant turn back into `conversation_history`.
- `MemorySaver` checkpoints serializable graph state by `thread_id`.

## Retrieval Modes

Inside `langgraph_rag_memory.py`, type `mode` and select one of these:

| Mode | What It Does | Best For |
| --- | --- | --- |
| `rewrite_rag` | Rewrites the question once, then retrieves local PDF chunks. | Default local document Q&A. |
| `multiquery_rag` | Generates multiple query variants, retrieves for each, then deduplicates. | Broad questions where one wording may miss relevant chunks. |
| `rag_fusion` | Generates multiple queries, retrieves ranked lists, then combines them with Reciprocal Rank Fusion. | More robust retrieval for complex or multi-aspect questions. |
| `web_search` | Rewrites the query for web search and uses Tavily if configured. | Current or external information not present in PDFs. |
| `hybrid` | Runs local RAG and web search, then merges both contexts. | Questions that need both your PDFs and current/external context. |

Example:

```text
mode
3
```

This switches the current session to `rag_fusion`.

## Prompt Types

Inside the CLI, type `prompt` and select one:

| Prompt Type | Behavior |
| --- | --- |
| `default` | Standard answer using retrieved context. |
| `concise` | Short 1-2 sentence answer. |
| `detailed` | Thorough answer with document/page citations where possible. |
| `bullet` | Bullet-point answer. |

Example:

```text
prompt
3
```

This switches the current session to the detailed prompt.

All prompt types include:

- retrieved document context,
- current question,
- recent conversation history,
- instruction to avoid unsupported answers when the answer is not in context.

## Conversation Memory

Memory is active when running:

```powershell
C:\Users\abhij\anaconda3\python.exe langgraph_rag_memory.py
```

The graph stores conversation turns in `conversation_history`:

```python
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
```

`generation_node` injects recent history into the prompt so follow-up questions can refer to previous turns.

Useful memory commands:

```text
state
```

Shows the current checkpoint summary, including `memory turns`.

```text
history
```

Shows checkpoint count for the active session.

```text
new
```

Starts a fresh session with a new `thread_id`.

```text
session research_notes
```

Switches to or creates a named session.

Note: `MemorySaver` is in-memory. Conversation memory persists while the Python process is running. It does not survive a full script restart unless you replace it with a persistent checkpointer.

## How To Ask Better Questions

For better RAG answers, include the target topic, expected format, and any source hints.

Good:

```text
Explain denoising diffusion models. Include architecture, training process, and image/video applications. Use bullet points and cite document pages if available.
```

Better with source hints:

```text
file:Attention_is_All_You_Need.pdf Explain how attention differs from recurrent models. Give a concise answer with page references.
```

Good follow-up with memory:

```text
How does that compare with diffusion models?
```

This works best after the previous turn established what "that" refers to.

Use `detailed` prompt when you need citations:

```text
prompt
3
What are the main components of the Transformer architecture? Cite pages.
```

Use `multiquery_rag` or `rag_fusion` for broad questions:

```text
mode
3
Summarize the training process, architecture, and limitations discussed in the documents.
```

Use `hybrid` when the question may require information outside the local PDFs:

```text
mode
5
Compare the paper's approach with recent diffusion model applications in video generation.
```

## CLI Command Reference

Available in `langgraph_rag_memory.py`:

```text
mode                  Switch retrieval mode
prompt                Switch prompt style
session <name>        Switch to or create a named session
sessions              List active sessions
state                 Show latest checkpoint summary
history               Show checkpoint count
new                   Start a fresh session
stream:<query>        Stream answer from generation node
batch:<q1>|<q2>       Run multiple questions sequentially
exit                  Quit
<query>               Ask a normal question
```

Available in `advanced_generation_rag.py`:

```text
prompt                Switch prompt style
mode                  Switch retrieval mode
preview:<query>       Show rendered prompt
rewrite:<query>       Show rewritten RAG/web/multiquery forms
route:<query>         Show routing and retrieved chunks
stream:<query>        Stream answer
batch:<q1>|<q2>       Batch multiple questions
exit                  Quit
<query>               Generate a normal answer
```

## Requirements And Runtime Notes

Main runtime dependencies include:

- `langchain`
- `langgraph`
- `langchain-core`
- `langchain-ollama`
- `langchain-huggingface`
- `langchain-text-splitters`
- `pdfplumber`
- `sentence-transformers`
- optional `langchain-community` and `TAVILY_API_KEY` for Tavily web search
- optional PostgreSQL/PGVector support for the older `IndexingDocs_for_rag.py` demo

Ollama must be running locally with the configured model:

```powershell
ollama pull llama3.2:1b
ollama serve
```

The embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The first run may download this model from HuggingFace if it is not already cached.

## Data Files

PDFs currently in this folder:

The current index builder scans PDFs under the project folder, including `rag_docs`.

Generated/support files:

- `hf_doc_change_tracker.json`: tracks document hashes and indexing metadata.
- `hf_record_manager.db`: SQLite record manager database.
- `doc_change_tracker.json`: older document tracking file.
- `__pycache__`: Python cache directory.

## Recommended Workflow

1. Put your PDFs in this folder or in `rag_docs`.
2. Start Ollama and make sure `llama3.2:1b` is available.
3. Run:

```powershell
python.exe langgraph_rag_memory.py
```

4. Choose `mode` based on the question complexity.
5. Choose `prompt` based on answer style.
6. Ask questions and follow-ups in the same session to use memory.
7. Use `state` to confirm memory turns are being stored.
