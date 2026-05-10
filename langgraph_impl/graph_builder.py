"""
graph_builder.py
──────────────────────
Defines the AgentState TypedDict, wires all nodes into a StateGraph,
attaches MemorySaver for per-thread conversation memory, and compiles
the graph.

Three RAG modes in one graph
─────────────────────────────
  Agentic RAG   →  grader_node decides to retry retrieval or proceed to generation
  Knowledge RAG →  rag_fusion_node / multiquery_rag_node / local_rag_node 
                   powered by the multi-source indexed knowledge base built at startup
  Memory RAG    →  MemorySaver checkpointer with thread_id for multi-turn conversation memory, 
                    accessible in retrieval nodes for context-aware retrieval

Existing files imported (zero modifications):
    - graph_nodes.py                        →  All node functions (query_node, router_node, etc.)
    - langchain_impl/retrieval_index.py     →  build_advanced_retrieval_index()
    - langchain_impl/retriever_router.py    →  MultiSourceRouter
    - langchain_impl.query_processing.py    →  ProcessedQuery
    - langchain_impl.indexing_core.py       →  format_docs, build_prompt
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# ── Existing project imports (no modifications) ────────────────────────────────
from langchain_impl.query_processing import ProcessedQuery
from langchain_impl.retrieval_index import build_advanced_retrieval_index
from langchain_impl.retriever_router import MultiSourceRouter

# ── New node imports ───────────────────────────────────────────────────────────
from langgraph_impl.graph_nodes import (
    generation_node,
    grader_node,
    hybrid_node,
    local_rag_node,
    multiquery_rag_node,
    query_node,
    rag_fusion_node,
    router_node,
    route_after_grading,
    route_to_retriever,
    web_search_node,
)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# Each key is optional so LangGraph can merge partial dicts from nodes cleanly.
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    raw_query:        str                  # Original user question
    retrieval_mode:   str                  # One of: rewrite_rag | multiquery_rag |
                                           #         rag_fusion | web_search | hybrid
    prompt_type:      str                  # One of: default | concise | detailed | bullet
    top_k:            int                  # Chunks to retrieve per query (default 6)
    max_retries:      int                  # Max grader retry loops (default 2)

    # ── Query preprocessing ────────────────────────────────────────────────
    processed_query:  ProcessedQuery       # Normalized query + variants + filters

    # ── Routing metadata ───────────────────────────────────────────────────
    routed_docs:      list                 # Docs from initial router pass
    selected_sources: list[str]            # Source IDs chosen by the router
    route_name:       str                  # e.g. "targeted_single_source"

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieved_docs:   list                 # Docs after retrieval node
    retrieval_context: str                 # Formatted context string for the LLM

    # ── Agentic grading ────────────────────────────────────────────────────
    grade:            str                  # "pass" or "retry"
    retry_count:      int                  # How many grader loops have run

    # ── Memory (conversation history) ─────────────────────────────────────
    conversation_history: list[dict]       # [{role, content}, ...] for multi-turn

    # ── Output ─────────────────────────────────────────────────────────────
    answer:           str                  # Final generated answer


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Construct and return the compiled LangGraph StateGraph.

    Graph topology
    ──────────────
    START
      └─► query_node
            └─► router_node
                  └─► [conditional: route_to_retriever]
                        ├─► local_rag_node ──────────┐
                        ├─► multiquery_rag_node ─────┤
                        ├─► rag_fusion_node ──────────┤
                        ├─► web_search_node ──────────┤
                        └─► hybrid_node ──────────────┘
                                │
                              grader_node
                                │
                          [conditional: route_after_grading]
                                ├─► (retry) → back to retriever
                                └─► generation_node
                                        └─► END
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────
    graph.add_node("query_node",          query_node)
    graph.add_node("router_node",         router_node)
    graph.add_node("local_rag_node",      local_rag_node)
    graph.add_node("multiquery_rag_node", multiquery_rag_node)
    graph.add_node("rag_fusion_node",     rag_fusion_node)
    graph.add_node("web_search_node",     web_search_node)
    graph.add_node("hybrid_node",         hybrid_node)
    graph.add_node("grader_node",         grader_node)
    graph.add_node("generation_node",     generation_node)

    # ── Fixed edges ────────────────────────────────────────────────────────
    graph.add_edge(START,             "query_node")
    graph.add_edge("query_node",      "router_node")

    # After router → choose retrieval node based on retrieval_mode
    graph.add_conditional_edges(
        "router_node",
        route_to_retriever,
        {
            "local_rag_node":      "local_rag_node",
            "multiquery_rag_node": "multiquery_rag_node",
            "rag_fusion_node":     "rag_fusion_node",
            "web_search_node":     "web_search_node",
            "hybrid_node":         "hybrid_node",
        },
    )

    # All retrieval nodes converge at grader
    for retrieval_node in (
        "local_rag_node",
        "multiquery_rag_node",
        "rag_fusion_node",
        "web_search_node",
        "hybrid_node",
    ):
        graph.add_edge(retrieval_node, "grader_node")

    # After grader → retry (back to retriever) or pass (generation)
    graph.add_conditional_edges(
        "grader_node",
        route_after_grading,
        {
            # Retry routes (mirror of route_to_retriever targets)
            "local_rag_node":      "local_rag_node",
            "multiquery_rag_node": "multiquery_rag_node",
            "rag_fusion_node":     "rag_fusion_node",
            "web_search_node":     "web_search_node",
            "hybrid_node":         "hybrid_node",
            # Pass route
            "generation_node":     "generation_node",
        },
    )

    graph.add_edge("generation_node", END)

    return graph


# ══════════════════════════════════════════════════════════════════════════════
# COMPILED GRAPH  (with MemorySaver for memory RAG)
# ══════════════════════════════════════════════════════════════════════════════

def compile_graph(checkpointer: Optional[MemorySaver] = None):
    """
    Compile the StateGraph with an optional MemorySaver checkpointer.

    Args:
        checkpointer: A MemorySaver (or any LangGraph checkpointer).
                      Pass None to compile without memory (stateless).

    Returns:
        A compiled LangGraph app ready for .invoke() / .stream()
    """
    graph = build_graph()
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# INDEX BUILDER  (called once at startup — reuses existing pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def build_router(pdf_folder: Optional[Path] = None, k: int = 6) -> MultiSourceRouter:
    """
    Build the retrieval index and return a MultiSourceRouter.

    Args:
        pdf_folder: Folder to scan for PDFs (defaults to the auto-discovery
                    logic inside advanced_rag_indexing_2.py).
        k:          Number of chunks to retrieve per query.

    Returns:
        MultiSourceRouter backed by per-source + global retrievers.
    """
    print("\n" + "=" * 60)
    print("Building retrieval index...")
    print("=" * 60)

    bundle = build_advanced_retrieval_index(k=k, pdf_folder=pdf_folder)

    router = MultiSourceRouter(
        indexed_sources=bundle["indexed_sources"],
        global_retriever=bundle["global_retriever"],
        default_k=k,
    )

    print(f"\n  Indexed sources : {len(bundle['indexed_sources'])}")
    print(f"  Total chunks    : {len(bundle['all_splits'])}")
    print("=" * 60 + "\n")

    return router
