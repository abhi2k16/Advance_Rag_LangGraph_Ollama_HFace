# How to Built a Local, Memory-Aware RAG System with LangGraph, Ollama, and HuggingFace — From Scratch

*A deep-dive into building an agentic retrieval-augmented generation pipeline that actually remembers what you said five questions ago.*

---

## The Problem With Most RAG Tutorials

You've probably seen the tutorials. Load a PDF. Split it into chunks. Embed the chunks. Ask a question. Get an answer. Done.

But that's not how real use looks. Real use means follow-up questions. It means "wait, go back to what you said about attention heads." It means querying two papers at once and wanting the system to know which one to pull from. It means an answer that was confidently wrong because no one checked whether the retrieved chunks were actually relevant.

This project is the attempt to fix all of that — without spinning up a cloud service, without paying for an API per query, and without modifying a single line of the original codebase when adding new features.

By the end of this article, you'll understand how the whole system fits together: from raw PDF bytes to a multi-turn, agentic, memory-aware conversation powered entirely by local models.

---

## What We're Building

The system has three distinct RAG modes, all running through the same compiled LangGraph graph:

- **Agentic RAG** — a grader node that scores retrieved documents for relevance and loops back to retry retrieval if the results aren't good enough, before ever handing anything to the LLM for generation.
- **Knowledge RAG** — multiple retrieval strategies (single rewrite, multi-query expansion, RAG Fusion with Reciprocal Rank Fusion) backed by a persistent PGVector store so embeddings survive restarts.
- **Memory RAG** — LangGraph's `MemorySaver` checkpointer gives every conversation thread its own persistent history, so follow-up questions actually work.

The stack:

| Component | Choice | Why |
|---|---|---|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | 384-dim, CPU-friendly, no API key |
| LLM | Ollama `llama3.2:1b` | Runs locally, zero cost per query |
| Vector store | PGVector (PostgreSQL) | Persistent across restarts |
| Graph runtime | LangGraph `StateGraph` | Native conditional routing + memory |
| Document loading | pdfplumber | Reliable page-level text extraction |

---

## Project Structure at a Glance

The codebase is split into two packages, and the split is intentional: the `langchain_impl` layer owns all the indexing, routing, and generation logic. The `langgraph_impl` layer wraps that logic in a graph without touching a single existing file.

```
.
├── main.py                          # Single entry point for all workflows
├── langchain_impl/
│   ├── indexing_core.py             # Embeddings, vector stores, prompts, trackers
│   ├── retrieval_index.py           # PGVector-backed persistent indexing
│   ├── query_processing.py          # Query normalization, filter extraction, variants
│   ├── retriever_router.py          # MultiSourceRouter — per-source + global retrieval
│   └── generation_pipeline.py       # All retrieval modes + generation CLI
└── langgraph_impl/
    ├── graph_builder.py             # AgentState, graph topology, compile_graph
    ├── graph_nodes.py               # Every node function
    └── memory_cli.py                # Session management + interactive CLI
```

Every new feature in `langgraph_impl` is purely additive. The existing pipeline never knew it existed.

---

## Part 1 — The Foundation: Indexing That Doesn't Repeat Itself

The most invisible but impactful part of the system is how it handles indexing across restarts.

### The Problem with Naive Indexing

Most RAG demos re-embed everything on every startup. For two small PDFs that's fine. For ten papers, it's thirty seconds of GPU (or CPU) time before you can ask your first question. And if one paper changes, you re-embed all of them.

### Change Tracking with MD5

`indexing_core.py` maintains a JSON tracker file (`hf_doc_change_tracker.json`). Every time a document is indexed, its MD5 hash is saved alongside the timestamp and chunk count.

```python
def check_document_changes(filepath: str, tracker: dict) -> tuple[bool, str]:
    current_hash = compute_file_hash(filepath)
    filename = Path(filepath).name
    if filename not in tracker:
        return True, "NEW"
    elif tracker[filename]["hash"] != current_hash:
        return True, "MODIFIED"
    return False, "UNCHANGED"
```

On startup, `retrieval_index.py` checks every PDF:

- **UNCHANGED** → connect to the existing PGVector collection, skip embedding entirely.
- **NEW or MODIFIED** → drop the old collection for that source, re-embed, upsert to PGVector.
- **Other sources** → completely unaffected.

The result: the first run is slow. Every subsequent run is near-instant unless a PDF actually changed.

### Per-Source Collections in PGVector

The indexing layer maintains one PGVector collection per PDF source plus one global collection that spans all sources:

```
hf_global         → all chunks across all PDFs
hf_pdf_source_1   → only chunks from the first PDF
hf_pdf_source_2   → only chunks from the second PDF
```

This is what makes per-source routing possible later. The global retriever is a safety net; the per-source retrievers are the precision tool.

### Stable Source IDs

Source IDs (`pdf_source_1`, `pdf_source_2`, …) are derived from the sorted order of discovered PDF filenames — not random UUIDs. This means the collection names are stable across restarts, so PGVector can always find the right data.

---

## Part 2 — Query Processing: Treating User Input as Noisy Data

User queries are messy. They have typos, abbreviations, missing context from previous turns, and implicit filters ("that lecture PDF"). The system handles this in `query_processing.py` before anything touches the vector store.

### The ProcessedQuery Dataclass

Every raw query becomes a structured `ProcessedQuery`:

```python
@dataclass
class ProcessedQuery:
    raw_query: str
    normalized_query: str
    retrieval_query: str          # abbreviations expanded
    variants: list[str]           # multiple forms for better recall
    filters: dict                 # e.g. {"filename": "attention"}
    search_mode: str              # "general", "comparison", "summary", etc.
    issues: list[str]             # "query_too_short", "repeated_characters", etc.
```

### Explicit Routing Filters

A user can write `file:attention` or `source:lecture` in their query and the system will parse that into a routing filter — then strip it from the retrieval query so the LLM never sees it.

```python
def extract_filters(query: str) -> tuple[str, dict]:
    filters: dict[str, str] = {}
    patterns = {
        "filename": r"\b(?:file|doc|document):([^\s]+)",
        "source":   r"\bsource:([^\s]+)",
        "source_type": r"\btype:([^\s]+)",
    }
    # match, extract, remove from query
    ...
```

### Abbreviation Expansion

`rag` becomes `retrieval augmented generation`. `llm` becomes `large language model`. This matters because the indexed text almost certainly uses the full form, and cosine similarity between short abbreviations and their expansions is surprisingly poor.

### Query Variants

The system generates multiple forms of every query — the original, the lowercased version, an abbreviation-expanded form, and a punctuation-stripped compact form. All variants are tried during retrieval. This is cheap insurance against the brittleness of single-query vector search.

---

## Part 3 — The Router: Deciding Where to Look

`MultiSourceRouter` takes a processed query and decides which source-specific retrievers to use. The logic runs in two passes.

### Pass 1: Explicit Filters

If the `ProcessedQuery` has a `filename` or `source` filter, the router checks every indexed source's metadata against it. An exact match immediately selects that source.

### Pass 2: Soft Keyword Matching

If no explicit filter exists, the router computes a token overlap score between the query terms and each source's name and filename. The top two scoring sources win.

```python
query_terms = set(processed.retrieval_query.lower().split())
for source_id, indexed_source in self.indexed_sources.items():
    config = indexed_source.config
    source_terms = set(
        (config.name + " " + config.metadata.get("filename", ""))
        .lower().replace(".", " ").split()
    )
    overlap = len(query_terms & source_terms)
    if overlap:
        scored_sources.append((overlap, source_id))
```

### Fallback to Global

If nothing matches, the global retriever takes over. This is the `global_fallback` route — it has no source preference, just top-k chunks across everything.

### Deduplication

Because the same chunk might be returned by multiple source retrievers (or multiple query variants), every set of retrieved documents passes through a deduplication step keyed on `chunk_id`. Without this, the LLM context fills up with repeated content.

---

## Part 4 — Five Retrieval Strategies, One Router

`generation_pipeline.py` implements five distinct retrieval strategies, all routing through `MultiSourceRouter`.

### 1. Rewrite RAG

The simplest strategy. The user's query is rewritten once to be more retrieval-friendly (more technical, more standalone), then used for a single retrieval pass.

```
"what's the thing with multiple heads in transformers?"
→ "multi-head attention mechanism Transformer architecture"
→ retrieve → generate
```

### 2. Multi-Query RAG

The LLM generates three distinct retrieval queries from the original question. Each query retrieves independently. The results are merged and deduplicated, capping at eight chunks. The intuition: different phrasings surface different relevant passages.

### 3. RAG Fusion

A step further than multi-query. Four queries are generated. Each retrieves a ranked list of documents. The lists are merged using Reciprocal Rank Fusion (RRF):

```python
score(doc) += 1 / (k + rank)
```

where `k = 60` is a smoothing constant and `rank` is the document's position in a particular query's result list. Documents that appear consistently across multiple query lists accumulate high scores regardless of which individual query put them first. The top five fused documents go to the LLM.

### 4. Web Search

The query is rewritten for a search engine (shorter, keyword-dense, no filler). Tavily is the first choice; DuckDuckGo is the fallback. Results are formatted into a consistent context block regardless of source.

### 5. Hybrid

Both local RAG and web search run sequentially. Their contexts are concatenated with clear section headers so the LLM knows which information came from which source.

---

## Part 5 — The LangGraph Graph: Wiring It All Together

This is where the three RAG modes (Agentic, Knowledge, Memory) converge into a single compiled graph.

### AgentState: The Spine of the Graph

Every node reads from and writes to a shared `AgentState` TypedDict. LangGraph merges partial dicts returned by each node back into the full state.

```python
class AgentState(TypedDict, total=False):
    raw_query:            str
    retrieval_mode:       str
    prompt_type:          str
    top_k:                int
    max_retries:          int
    processed_query:      ProcessedQuery
    routed_docs:          list
    selected_sources:     list[str]
    route_name:           str
    retrieved_docs:       list
    retrieval_context:    str
    grade:                str
    retry_count:          int
    conversation_history: list[dict]
    answer:               str
```

`total=False` means every key is optional — nodes return only the keys they update, and LangGraph handles the merge.

### The Graph Topology

```
START
  └─► query_node           # ProcessedQuery from raw input
        └─► router_node    # source selection + initial retrieval
              └─► [conditional: route_to_retriever]
                    ├─► local_rag_node       # single rewrite + retrieval
                    ├─► multiquery_rag_node  # 3-query expansion + dedup
                    ├─► rag_fusion_node      # 4-query + RRF
                    ├─► web_search_node      # Tavily / DuckDuckGo
                    └─► hybrid_node          # local + web merged
                          │
                        grader_node          # relevance scoring (Agentic RAG)
                          │
                    [conditional: route_after_grading]
                          ├─► (retry) → back to retrieval node
                          └─► generation_node
                                  └─► END
```

### The Grader Node: The Agentic Heart

`grader_node` is what makes this "agentic" rather than just "a pipeline." It asks the LLM to score each retrieved document independently:

```
You are a strict document relevance grader.
Given a user question and a retrieved document, decide if the document is
relevant to answering the question.
Respond with EXACTLY one word: "relevant" or "irrelevant". Nothing else.
```

If fewer than 50% of documents are relevant and the retry budget hasn't been exhausted, the grade is `"retry"` and the graph routes back to the same retrieval node for another attempt. Only when the relevance threshold is met (or retries run out) does execution proceed to generation.

```python
if relevance_ratio >= 0.5 or retry_count >= max_retries:
    grade = "pass"
else:
    grade = "retry"
```

This is the difference between a system that confidently answers from irrelevant chunks and one that at least tries to get better data first.

### Conditional Edges

LangGraph's conditional edges are functions that return a string key. The graph uses the key to look up which node to go to next.

```python
def route_to_retriever(state: State) -> str:
    mode = state.get("retrieval_mode", "rewrite_rag")
    routing_map = {
        "rewrite_rag":    "local_rag_node",
        "multiquery_rag": "multiquery_rag_node",
        "rag_fusion":     "rag_fusion_node",
        "web_search":     "web_search_node",
        "hybrid":         "hybrid_node",
    }
    return routing_map.get(mode, "local_rag_node")
```

The retry loop reuses the same function — `route_after_grading` calls `route_to_retriever` internally when the grade is `"retry"`, so the retry always goes back to the same strategy that was used in the first pass.

---

## Part 6 — Memory: Making the Graph Remember

Without memory, every question is a fresh start. The system uses LangGraph's `MemorySaver` checkpointer to give each conversation thread its own persistent state.

### How MemorySaver Works

`MemorySaver` stores the full `AgentState` after every node execution, keyed by `(thread_id, checkpoint_id)`. When the same `thread_id` is used for a follow-up question, LangGraph loads the last checkpoint and merges the new input on top.

```python
self.memory = MemorySaver()
self.app = compile_graph(checkpointer=self.memory)

# Every invoke carries the thread_id in config
config = {"configurable": {"thread_id": self.thread_id}}
result = self.app.invoke(input_state, config=config)
```

### Conversation History in the Prompt

`generation_node` reads `conversation_history` from state and injects the last six turns into the prompt before calling the LLM:

```python
def _format_conversation_history(history: list[dict], max_turns: int = 6) -> str:
    recent = history[-max_turns * 2:]
    lines = []
    for item in recent:
        role = str(item.get("role", "unknown")).capitalize()
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None."
```

After generation, the new turn (user question + assistant answer) is appended to the history, capped at twelve turns to avoid context overflow.

### MemoryRAGSession: One Object, One Conversation

`memory_cli.py` wraps all of this in a clean `MemoryRAGSession` class:

```python
session = MemoryRAGSession(router, retrieval_mode="rag_fusion")

# Turn 1
answer1 = session.invoke("What is multi-head attention?")

# Turn 2 — the session remembers turn 1
answer2 = session.invoke("How does it compare to RNNs?")
```

Multiple sessions can run concurrently via `SessionManager`, each with its own `thread_id`. This makes the architecture suitable for multi-user scenarios — every user gets their own isolated conversation state.

---

## Part 7 — The Runtime Router Registry

There's an architectural challenge: graph nodes need access to the live `MultiSourceRouter` instance, but nodes are plain functions registered at graph-compile time. You can't pass the router as a constructor argument.

The solution is a simple module-level registry in `graph_nodes.py`:

```python
_RUNTIME_ROUTERS: dict[str, Any] = {}

def register_runtime_router(thread_id: str, router: Any) -> None:
    _RUNTIME_ROUTERS[thread_id] = router

def _get_router(config: RunnableConfig) -> Any:
    thread_id = config.get("configurable", {}).get("thread_id")
    return _RUNTIME_ROUTERS[thread_id]
```

When a `MemoryRAGSession` is created, it registers its router against its `thread_id`. Every node that needs the router calls `_get_router(config)` — LangGraph passes the `config` dict (which contains the `thread_id`) into every node automatically via `RunnableConfig`. Different sessions, different thread IDs, different routers — no globals bleeding between conversations.

---

## Part 8 — The LLM Is a Lazy Singleton

Every node that needs the LLM calls `get_llm()`. The first call initializes `ChatOllama`; subsequent calls return the cached instance.

```python
_llm: ChatOllama | None = None

def get_llm(model: str = "llama3.2:1b", temperature: float = 0) -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=model, temperature=temperature)
    return _llm
```

This pattern avoids creating a new Ollama connection on every node invocation while keeping the initialization lazy — it only happens when a query actually arrives, not at import time.

---

## Part 9 — Prompt Types

`indexing_core.py` defines four prompt templates, selectable at runtime:

| Type | Behavior |
|---|---|
| `default` | Helpful assistant, answer from context only |
| `concise` | 1–2 sentence answer |
| `detailed` | Thorough answer with document citations |
| `bullet` | Bullet-point formatted answer |

All four templates include `{conversation_history}` as a placeholder, but they use it only for resolving follow-up references — the instruction explicitly says "use the conversation history only to understand follow-up references," not to answer from it. This keeps the grounding tight.

---

## Part 10 — Running the System

Everything runs through `main.py`:

```powershell
# List available workflows
python main.py --list

# Run the memory-enabled LangGraph pipeline (default)
python main.py memory

# Run with a custom PDF folder
python main.py memory --pdf-folder .\rag_docs

# Run the standalone advanced generation CLI
python main.py advanced
```

Once inside the memory CLI, the available commands are:

```
mode              # switch retrieval strategy mid-conversation
prompt            # switch prompt style
session <name>    # create or switch to a named conversation session
sessions          # list all active sessions
state             # inspect the current checkpointed AgentState
history           # count checkpoints in the current session
new               # start a fresh session with a new thread_id
stream:<query>    # stream the answer token by token
batch:<q1>|<q2>   # run multiple questions in the current session
exit
<query>           # standard invoke
```

---

## What This Architecture Gets Right

**The additive constraint is real.** The `langgraph_impl` package was added after `langchain_impl` was fully working. Not a single line in the original package was modified. This matters: teams can extend existing pipelines with graph-level orchestration without risking regressions in production code.

**The grader loop is worth the latency.** For every query the grader adds one LLM call per retrieved document. On `llama3.2:1b` locally that's a few seconds. But the alternative — sending irrelevant chunks to generation — produces worse answers that users then have to rephrase and retry anyway. The grader pays for itself.

**Per-source PGVector collections make routing meaningful.** If the global retriever were the only option, the routing layer would be decorative. With per-source collections, a query that specifies `file:attention` actually hits a smaller, more relevant index.

**MemorySaver is process-memory only.** The current `MemorySaver` does not persist across full process restarts. Across turns in the same session it works perfectly. For true persistence between restarts, the checkpoint would need to be stored in a database — LangGraph supports this via `PostgresSaver`, which is a natural next step for this project.

---

## What Could Come Next

A few natural extensions:

- **PostgresSaver** for cross-restart conversation persistence.
- **A FastAPI wrapper** around `SessionManager` to serve multiple users over HTTP, each with their own `thread_id`.
- **Streaming to the frontend** — `session.stream()` already yields node-level updates; connecting this to a WebSocket would give token-by-token streaming in a browser.
- **Re-indexing CLI flag** — `main.py memory --reindex` already works to force a full PGVector rebuild; surfacing this more prominently in the README would help new contributors.
- **Evaluation harness** — the grader node produces a relevance score per document. Aggregating these across a test set would give a quantitative signal on which retrieval mode performs best for a given document corpus.

---

## Conclusion

The system described here isn't a tutorial toy. It handles the things that matter in production: incremental indexing, query normalization, multi-source routing, relevance-gated generation, and multi-turn memory — all running locally, all for free after the initial setup.

The architecture shows how LangGraph and LangChain complement each other. LangChain handles the document-level primitives well — loaders, splitters, embeddings, vector stores. LangGraph handles the control flow that turns those primitives into something that can reason, retry, and remember.

The most important design decision was the additive constraint: `langgraph_impl` never touches `langchain_impl`. That discipline keeps the codebase legible, keeps the existing pipeline safe, and keeps future extension straightforward.

---

*The full source is organized into `langchain_impl/` and `langgraph_impl/` packages, launched from a single `main.py`. Ollama must be running locally with `llama3.2:1b` pulled. PostgreSQL with the PGVector extension handles persistent vector storage. HuggingFace embeddings run on CPU with no API key required.*
