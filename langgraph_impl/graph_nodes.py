"""
graph_nodes.py
──────────────────────
Pure node functions for the LangGraph agentic RAG graph. 

Imported Modules in graph_nodes.py from other modules and their Uses:
  - langchain_impl.query_processing.py              →  process_user_query (returns ProcessedQuery)
  - langchain_impl.indexing_core.py                 →  format_docs, build_prompt
  - langchain_impl.generation_pipeline.py           →  REWRITE_TEMPLATE, MULTIQUERY_TEMPLATE,
                                                    RAG_FUSION_TEMPLATE, WEB_REWRITE_TEMPLATE,
                                                    parse_multi_queries, reciprocal_rank_fusion,
                                                    format_web_results, build_web_search_tool
  - langchain_impl.indexing_core.py                 →  format_docs, build_prompt

The main functionality of current nodes includes:
  - query_node: preprocesses raw user query into structured ProcessedQuery
  - router_node: routes the query to appropriate sources and retrieves initial docs
  - local_rag_node: performs RAG with a single rewritten query
  - multiquery_rag_node: performs RAG with multiple generated queries and deduplication
  - rag_fusion_node: performs RAG with multiple queries and Reciprocal Rank Fusion
  - web_search_node: rewrites query for web search and retrieves results from Tavily
  - hybrid_node: combines local RAG and web search contexts
  - grader_node: grades retrieved docs for relevance and decides whether to retry retrieval
  - generation_node: generates the final answer from the retrieval context and question

Every function here:
  - accepts the full AgentState dict
  - returns a *partial* dict that LangGraph merges back into state
  - imports only from existing project modules (no edits to those files)
This file also includes the shared get_llm() function and the runtime router registry functions that 
allow nodes to access the live MultiSourceRouter instance without a direct import (to avoid circular dependencies). 
The router instance is registered at runtime by the graph execution code and keyed by thread_id to support 
multiple concurrent graphs (e.g. in a web server).
"""

from __future__ import annotations                        #for Python 3.10+ type hinting features (e.g. dict[str, Any])

from datetime import datetime
from typing import Any                                    #for type hinting the State dict values

from langchain_core.output_parsers import StrOutputParser #for parsing LLM output as plain strings
from langchain_core.prompts import ChatPromptTemplate     #for building LLM prompts from templates
from langchain_core.runnables import RunnableConfig       #for accessing LangGraph config within nodes
from langchain_ollama import ChatOllama

# ── Existing project imports (no modifications) ────────────────────────────────
from langchain_impl.generation_pipeline import (
    MULTIQUERY_TEMPLATE,
    RAG_FUSION_TEMPLATE,
    REWRITE_TEMPLATE,
    WEB_REWRITE_TEMPLATE,
    build_web_search_tool,
    format_web_results,
    parse_multi_queries,
    reciprocal_rank_fusion,
)
from langchain_impl.indexing_core import build_prompt, format_docs
from langchain_impl.query_processing import process_user_query

State = dict[str, Any]                  # for type hinting the AgentState dict passed to each node
_RUNTIME_ROUTERS: dict[str, Any] = {}   # Maps LangGraph thread_id to the live MultiSourceRouter instance for that thread


def register_runtime_router(thread_id: str, router: Any) -> None: #type hinting for the router parameter is 'Any' since we want to avoid importing MultiSourceRouter here
    """Register a live router for a LangGraph thread."""
    _RUNTIME_ROUTERS[thread_id] = router  # keyed by thread_id to support multiple concurrent graphs (e.g. in a web server)
    print(f"[register_runtime_router] Registered router for thread_id={thread_id!r}")


def _get_router(config: RunnableConfig | None) -> Any: 
    """Return the runtime router registered for this LangGraph thread."""
    if not config:
        raise ValueError("Missing LangGraph config; cannot access runtime router.")
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        raise ValueError("Missing thread_id in config['configurable'].")
    router = _RUNTIME_ROUTERS.get(thread_id)
    if router is None:
        raise ValueError(f"No runtime router registered for thread_id={thread_id!r}.")
    return router


def _format_conversation_history(history: list[dict], max_turns: int = 6) -> str:  
    # type hinting for conversation history: list of dicts with 'role' and 'content' keys 
    # (e.g. {"role": "user", "content": "What is RAG?"})
    """Format recent chat history for the generation prompt."""
    if not history:
        return "None."
    recent = history[-max_turns * 2 :]  # get the last N turns (user + assistant = 2 entries per turn)
    lines = []
    for item in recent:
        role = str(item.get("role", "unknown")).capitalize()
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "None."


def _append_conversation_turn(history: list[dict], question: str, answer: str, max_turns: int = 12) -> list[dict]: 
    # type hinting for conversation history (same as above) and question/answer as strings, 
    # list[dict] used for the returned updated conversation history with new turns appended to it
    """Return history with the latest user/assistant turn appended."""
    updated = [dict(item) for item in history]
    updated.append({"role": "user", "content": question})
    updated.append({"role": "assistant", "content": answer})
    return updated[-max_turns * 2 :]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED LLM  (lazy singleton — created once, shared across nodes)
# ══════════════════════════════════════════════════════════════════════════════

_llm: ChatOllama | None = None  # module-level variable to hold the shared LLM instance


def get_llm(model: str = "llama3.2:1b", temperature: float = 0) -> ChatOllama: 
    # type hinting for the parameters of this function is str for model and float for temperature
    # type hinting for the return value of this function is ChatOllama, which is the LLM class we're using
    """Return a cached ChatOllama instance (created on first call)."""
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=model, temperature=temperature)
        print(f"  [LLM] Initialized: {model}")
    return _llm


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — QUERY PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def query_node(state: State) -> dict: #dict function return dictionary type hint
    """
    Normalize raw user input, expand abbreviations, extract routing filters,
    infer search mode, and generate query variants.

    Wraps: process_user_query() from advanced_rag_query_processing.py
    Updates state keys: processed_query, retrieval_mode (if not already set)
    """
    raw = state["raw_query"]
    processed = process_user_query(raw)

    print(f"\n[query_node]")
    print(f"  Raw query       : {processed.raw_query}")
    print(f"  Retrieval query : {processed.retrieval_query}")
    print(f"  Search mode     : {processed.search_mode}")
    print(f"  Filters         : {processed.filters or 'none'}")
    print(f"  Issues          : {processed.issues or 'none'}")

    # Honour any retrieval_mode already set in state (e.g. by the caller);
    # otherwise fall back to the mode inferred from the query.
    mode = state.get("retrieval_mode") or "rewrite_rag"

    return {
        "processed_query": processed,
        "retrieval_mode": mode,
        "retry_count": state.get("retry_count", 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — MULTI-SOURCE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def router_node(state: State, config: RunnableConfig) -> dict:
    """
    Decide which indexed sources to query and perform an initial retrieval
    using the ProcessedQuery variants built by query_node.

    Wraps: MultiSourceRouter.route() from advanced_rag_router.py
    Updates state keys: routed_docs, selected_sources, route_name
    """
    router = _get_router(config)
    processed = state["processed_query"]

    routed = router.route(processed.retrieval_query, k=state.get("top_k", 6))

    print(f"\n[router_node]")
    print(f"  Route           : {routed.route_name}")
    print(f"  Sources         : {routed.selected_sources or ['global']}")
    print(f"  Docs retrieved  : {len(routed.documents)}")

    return {
        "routed_docs": routed.documents,
        "selected_sources": routed.selected_sources,
        "route_name": routed.route_name,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3a — LOCAL RAG WITH QUERY REWRITE
# ══════════════════════════════════════════════════════════════════════════════

def local_rag_node(state: State, config: RunnableConfig) -> dict:
    """
    Rewrite the query once for better vector retrieval, then retrieve from the
    routed (or global) retriever.

    Wraps: rewriter chain + MultiSourceRouter from advanced_generation_rag.py
    Updates state keys: retrieved_docs, retrieval_context
    """
    llm = get_llm()
    router = _get_router(config)
    processed = state["processed_query"]

    rewriter = ChatPromptTemplate.from_template(REWRITE_TEMPLATE) | llm | StrOutputParser()
    optimized = rewriter.invoke({"question": processed.retrieval_query}).strip()
    if not optimized:
        optimized = processed.retrieval_query

    routed = router.route(optimized, k=state.get("top_k", 6))
    docs = routed.documents

    print(f"\n[local_rag_node]")
    print(f"  Rewritten query : {optimized}")
    print(f"  Docs retrieved  : {len(docs)}")

    return {
        "retrieved_docs": docs,
        "retrieval_context": format_docs(docs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3b — MULTI-QUERY RAG
# ══════════════════════════════════════════════════════════════════════════════

def multiquery_rag_node(state: State, config: RunnableConfig) -> dict:
    """
    Generate 3 distinct retrieval queries, retrieve for each, then deduplicate.

    Wraps: multiquery_rewriter + MultiSourceRouter from advanced_generation_rag.py
    Updates state keys: retrieved_docs, retrieval_context
    """
    llm = get_llm()
    router = _get_router(config)
    processed = state["processed_query"]

    mq_rewriter = (
        ChatPromptTemplate.from_template(MULTIQUERY_TEMPLATE) | llm | StrOutputParser()
    )
    raw = mq_rewriter.invoke({"question": processed.retrieval_query}).strip()
    queries = parse_multi_queries(raw) or [processed.retrieval_query]

    print(f"\n[multiquery_rag_node]")
    print(f"  Generated queries ({len(queries)}):")
    for i, q in enumerate(queries, 1):
        print(f"    [{i}] {q}")

    all_docs: list = []
    for q in queries:
        routed = router.route(q, k=4)
        all_docs.extend(routed.documents)

    deduped = router._deduplicate_docs(all_docs)[:8]

    print(f"  Docs after dedup: {len(deduped)}")

    return {
        "retrieved_docs": deduped,
        "retrieval_context": format_docs(deduped),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3c — RAG FUSION (multi-query + RRF reranking)
# ══════════════════════════════════════════════════════════════════════════════

def rag_fusion_node(state: State, config: RunnableConfig) -> dict:
    """
    Generate 4 queries, retrieve for each independently, then fuse with
    Reciprocal Rank Fusion (RRF) to surface the most consistently-ranked chunks.

    Wraps: rag_fusion_query_generator + reciprocal_rank_fusion()
           from advanced_generation_rag.py
    Updates state keys: retrieved_docs, retrieval_context
    """
    llm = get_llm()
    router = _get_router(config)
    processed = state["processed_query"]

    fusion_gen = (
        ChatPromptTemplate.from_template(RAG_FUSION_TEMPLATE) | llm | StrOutputParser()
    )
    raw = fusion_gen.invoke({"question": processed.retrieval_query}).strip()
    queries = parse_multi_queries(raw) or [processed.retrieval_query]

    print(f"\n[rag_fusion_node]")
    print(f"  Fusion queries ({len(queries)}):")
    for i, q in enumerate(queries, 1):
        print(f"    [{i}] {q}")

    ranked_lists: list[list] = []
    for i, q in enumerate(queries, 1):
        routed = router.route(q, k=4)
        ranked_lists.append(routed.documents)
        print(f"  Query [{i}] → {len(routed.documents)} docs")

    fused = reciprocal_rank_fusion(ranked_lists, verbose=True)[:5]

    return {
        "retrieved_docs": fused,
        "retrieval_context": format_docs(fused),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3d — WEB SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def web_search_node(state: State) -> dict:
    """
    Rewrite the query for a web search engine and call Tavily (if available).

    Wraps: web_rewriter + build_web_search_tool() from advanced_generation_rag.py
    Updates state keys: retrieved_docs, retrieval_context
    """
    llm = get_llm()
    processed = state["processed_query"]

    web_rewriter = (
        ChatPromptTemplate.from_template(WEB_REWRITE_TEMPLATE) | llm | StrOutputParser()
    )
    raw_web_query = web_rewriter.invoke(
        {"question": processed.retrieval_query, "current_year": datetime.now().year}
    ).strip()
    # Extract only the first non-empty line to avoid LLM explanations leaking into the query
    web_query = next(
        (line.strip() for line in raw_web_query.splitlines() if line.strip()),
        processed.retrieval_query
    )

    print(f"\n[web_search_node]")
    print(f"  Web query : {web_query}")

    tool, source = build_web_search_tool()
    if tool is None:
        context = (
            "Web search unavailable. "
            "Install `langchain-community` and set TAVILY_API_KEY to enable it."
        )
        print("  [WARNING] Tavily not configured — returning fallback message.")
        return {"retrieved_docs": [], "retrieval_context": context}
    if source == "duckduckgo":
        print("  [INFO] Using DuckDuckGoSearchResults for web search.")
        results = tool.run(web_query)
        # Print URLs hit by DuckDuckGo
        if isinstance(results, list):
            print(f"  Web sources ({len(results)}):")
            for i, item in enumerate(results, 1):
                title = item.get("title", "Untitled")
                url = item.get("link", item.get("url", "N/A"))
                print(f"    [{i}] {title}\n        {url}")
        context = format_web_results(results, source="duckduckgo")
        return {"retrieved_docs": [], "retrieval_context": context}
    else:
        results = tool.invoke({"query": web_query})  # Tavily returns a list of dict results

    context = format_web_results(results, source=source)

    print(f"  Results   : {len(results)} web hits")

    return {"retrieved_docs": [], "retrieval_context": context}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3e — HYBRID (local RAG + web search merged)
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_node(state: State, config: RunnableConfig) -> dict:
    """
    Run local RAG and web search in sequence, then merge both contexts.

    Wraps: local_rag_node logic + web_search_node logic
    Updates state keys: retrieved_docs, retrieval_context
    """
    # Local leg
    local_result = local_rag_node(state, config)
    local_context = local_result["retrieval_context"]
    local_docs = local_result["retrieved_docs"]

    # Web leg — build a minimal state slice
    web_result = web_search_node(state)
    web_context = web_result["retrieval_context"]

    merged_context = "\n\n".join(
        ["=== Local document context ===", local_context,
         "=== Web search context ===",    web_context]
    )

    print(f"\n[hybrid_node] Merged {len(local_docs)} local docs + web results.")

    return {
        "retrieved_docs": local_docs,
        "retrieval_context": merged_context,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — DOCUMENT GRADER  (agentic retry gate)
# ══════════════════════════════════════════════════════════════════════════════

GRADER_TEMPLATE = """You are a strict document relevance grader.
Given a user question and a retrieved document, decide if the document is
relevant to answering the question.

Respond with EXACTLY one word: "relevant" or "irrelevant". Nothing else.

Question : {question}
Document : {document}

Verdict:"""


def grader_node(state: State) -> dict:
    """
    Score each retrieved document for relevance using the LLM.
    Sets grade to 'pass' if enough docs pass, else 'retry'.

    This is the agentic gate that decides whether to loop back and re-retrieve
    or to proceed to generation.

    Updates state keys: grade, retry_count, retrieved_docs
    """
    llm = get_llm()
    docs = state.get("retrieved_docs", [])
    question = state["processed_query"].retrieval_query
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not docs:
        print(f"\n[grader_node] No docs to grade → retry")
        return {"grade": "retry", "retry_count": retry_count + 1}

    grader_prompt = ChatPromptTemplate.from_template(GRADER_TEMPLATE)
    grader_chain = grader_prompt | llm | StrOutputParser()

    relevant_docs = []
    print(f"\n[grader_node] Grading {len(docs)} docs (retry #{retry_count})...")

    for i, doc in enumerate(docs):
        snippet = doc.page_content[:600]
        verdict = grader_chain.invoke({"question": question, "document": snippet}).strip().lower()
        is_relevant = verdict.startswith("relevant")
        status = "✓" if is_relevant else "✗"
        print(f"  [{i+1}] {status} {doc.metadata.get('filename','?')} p.{doc.metadata.get('page','?')}")
        if is_relevant:
            relevant_docs.append(doc)

    relevance_ratio = len(relevant_docs) / len(docs)
    print(f"  Relevant: {len(relevant_docs)}/{len(docs)}  ({relevance_ratio:.0%})")

    # Pass if ≥50% of docs are relevant OR we've exhausted retries
    if relevance_ratio >= 0.5 or retry_count >= max_retries:
        grade = "pass"
        final_docs = relevant_docs if relevant_docs else docs  # fallback: keep all
    else:
        grade = "retry"
        final_docs = docs

    print(f"  Grade: {grade.upper()}")

    return {
        "grade": grade,
        "retry_count": retry_count + 1,
        "retrieved_docs": final_docs,
        "retrieval_context": format_docs(final_docs),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generation_node(state: State) -> dict:
    """
    Feed the retrieval context + question into the LLM using the active
    prompt template and return the final answer.

    Wraps: build_prompt() from IndexingDocs_for_rag.py
    Updates state keys: answer
    """
    llm = get_llm()
    prompt_type = state.get("prompt_type", "default")
    prompt = build_prompt(prompt_type)
    reader_chain = prompt | llm | StrOutputParser()

    context = state.get("retrieval_context", "")
    question = state["processed_query"].retrieval_query
    raw_question = state.get("raw_query", question)
    history = state.get("conversation_history", [])
    formatted_history = _format_conversation_history(history)

    print(f"\n[generation_node]")
    print(f"  Prompt type : {prompt_type}")
    print(f"  Context len : {len(context)} chars")
    print(f"  History turns: {len(history) // 2}")

    answer = reader_chain.invoke(
        {
            "context": context,
            "question": question,
            "conversation_history": formatted_history,
        }
    )

    print(f"  Answer len  : {len(answer)} chars")

    return {
        "answer": answer,
        "conversation_history": _append_conversation_turn(history, raw_question, answer),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def route_to_retriever(state: State) -> str:
    """
    Conditional edge: decide which retrieval node to invoke based on
    the retrieval_mode stored in state.

    Returns the node name string that LangGraph will route to.
    """
    mode = state.get("retrieval_mode", "rewrite_rag")
    routing_map = {
        "rewrite_rag":    "local_rag_node",
        "multiquery_rag": "multiquery_rag_node",
        "rag_fusion":     "rag_fusion_node",
        "web_search":     "web_search_node",
        "hybrid":         "hybrid_node",
    }
    target = routing_map.get(mode, "local_rag_node")
    print(f"\n[route_to_retriever] mode={mode!r} → {target}")
    return target


def route_after_grading(state: State) -> str:
    """
    Conditional edge: after grading, either loop back to retrieval
    (if grade == 'retry') or proceed to generation.

    Returns the node name string.
    """
    grade = state.get("grade", "pass")
    if grade == "retry":
        print("[route_after_grading] → re-routing to retriever")
        return route_to_retriever(state)
    print("[route_after_grading] → generation_node")
    return "generation_node"
