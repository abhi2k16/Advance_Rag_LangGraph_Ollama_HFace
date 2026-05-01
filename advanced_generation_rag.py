"""
Advanced RAG generation pipeline built on top of the current local project.

Adds:
  - user-query preprocessing
  - multi-source routing
  - source-aware retrieval before generation
  - query rewriting for RAG and web search
  - multi-query retrieval expansion
  - optional hybrid web + local retrieval
  - prompt preview and route debugging
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not Path(ssl_cert_file).is_file():
    os.environ.pop("SSL_CERT_FILE", None)

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_ollama import ChatOllama

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    TavilySearchResults = None

from IndexingDocs_for_rag import PDF_FOLDER, build_prompt, format_docs
#from advanced_rag_indexing import build_advanced_retrieval_index
from advanced_rag_indexing_2 import build_advanced_retrieval_index
from advanced_rag_router import MultiSourceRouter


PROMPT_TYPES = {
    "1": ("default", "Standard helpful assistant"),
    "2": ("concise", "1-2 sentence answer"),
    "3": ("detailed", "Detailed answer with citations"),
    "4": ("bullet", "Bullet point answer"),
}

RETRIEVAL_MODES = {
    "1": ("rewrite_rag", "Rewrite once, then retrieve from local docs"),
    "2": ("multiquery_rag", "Generate multiple local retrieval queries"),
    "3": ("rag_fusion", "Generate multiple queries and fuse results with RRF"),
    "4": ("web_search", "Use optimized web search only"),
    "5": ("hybrid", "Combine local RAG with web search"),
}

REWRITE_TEMPLATE = """You are an AI assistant that rephrases user questions to be
more effective for a vector database search. Your goal is to make the query
standalone and include technical keywords.

Original question: {question}

Rephrased question:"""

MULTIQUERY_TEMPLATE = """You are an AI assistant helping a retrieval system.
Generate 3 distinct search queries for vector retrieval.

Rules:
1. Keep each query concise and technical.
2. Preserve the original user intent.
3. Vary wording to improve recall across different chunks.
4. Return one query per line and nothing else.

Original question: {question}

Search queries:"""

RAG_FUSION_TEMPLATE = """You are an AI assistant helping a retrieval system.
Generate 4 different search queries for the same user question.

Rules:
1. Keep each query concise and retrieval-focused.
2. Preserve the original user intent.
3. Vary wording and technical keywords to improve recall.
4. Return one query per line and nothing else.

Original question: {question}

Search queries:"""

WEB_REWRITE_TEMPLATE = """
You are an expert search engine optimizer. Your task is to take a user's
question and transform it into a high-performance web search query.

Rules:
1. Strip out conversational filler (e.g., "please tell me", "I was wondering").
2. Identify the core intent and add technical keywords or synonyms.
3. If the query implies recent information, add the current year or "latest".
4. Format the output as a single, concise search string.

Current year: {current_year}
User Question: {question}

Optimized Web Search Query:"""


def select_retrieval_mode() -> str:
    print("\n" + "=" * 60)
    print("SELECT RETRIEVAL MODE")
    print("=" * 60)
    for key, (mode, desc) in RETRIEVAL_MODES.items():
        print(f"  {key}. {mode:<14} - {desc}")
    print()

    choice = input("  Enter choice [1-5] (default=1): ").strip() or "1"
    mode, _ = RETRIEVAL_MODES.get(choice, RETRIEVAL_MODES["1"])
    print(f"\n  Selected mode   : {mode}\n")
    return mode


def parse_multi_queries(raw_text: str) -> list[str]:
    queries: list[str] = []
    for line in raw_text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if cleaned and cleaned not in queries:
            queries.append(cleaned)
    return queries


def reciprocal_rank_fusion(results: list[list], k: int = 60, verbose: bool = False) -> list:
    scores: dict[tuple, float] = {}
    doc_map = {}
    per_query_contributions: dict[tuple, list[tuple[int, int, float]]] = {} # doc_key -> list of (query_index, rank, score)

    for query_idx, ranked_docs in enumerate(results):
        for rank, doc in enumerate(ranked_docs, start=1):
            doc_key = (
                doc.page_content,
                tuple(sorted(doc.metadata.items())) if getattr(doc, "metadata", None) else (),
            )
            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank)
            doc_map[doc_key] = doc

            if doc_key not in per_query_contributions:
                per_query_contributions[doc_key] = []
            per_query_contributions[doc_key].append((query_idx + 1, rank, 1.0 / (k + rank)))

    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if verbose:
        print("\n" + "=" * 60)
        print("RAG FUSION — RRF SCORES")
        print(f"  k (smoothing constant) : {k}")
        print(f"  Formula                : score += 1 / (k + rank)")
        print(f"  Total unique docs      : {len(scores)}")
        print(f"  Total query lists      : {len(results)}")
        print("=" * 60)

        for position, (doc_key, total_score) in enumerate(ranked_docs, start=1):
            doc = doc_map[doc_key]
            filename  = doc.metadata.get("filename", "?")
            page      = doc.metadata.get("page", "?")
            source_id = doc.metadata.get("source_id", "?")
            snippet   = doc.page_content[:120].replace("\n", " ")
            contribs  = per_query_contributions[doc_key]

            print(f"\n  [{position:>2}] Total RRF Score : {total_score:.6f}")
            print(f"       File           : {filename}  |  Page {page}  |  {source_id}")
            print(f"       Snippet        : {snippet!r}...")
            print(f"       Appearances    : {len(contribs)} / {len(results)} query lists")
            for query_idx, rank, contrib in sorted(contribs):
                bar = "█" * max(1, int(contrib * 3000))
                print(f"         Query {query_idx}: rank #{rank:>2}  →  1/({k}+{rank}) = {contrib:.6f}  {bar}")

        print("\n" + "-" * 60)
        print(f"  Top 5 docs selected for context generation")
        print("=" * 60 + "\n")
    return [doc_map[doc_key] for doc_key, _ in ranked_docs]


def format_web_results(results) -> str:
    if not results:
        return ""

    lines = []
    for index, item in enumerate(results, start=1):
        if isinstance(item, dict):
            title = item.get("title", "Untitled")
            url = item.get("url", "N/A")
            content = item.get("content") or item.get("snippet") or ""
        else:
            title = f"Result {index}"
            url = "N/A"
            content = str(item)
        lines.append(f"[Web {index}] {title}\nURL: {url}\n{content}")
    return "\n\n".join(lines)


def build_web_search_tool():
    if TavilySearchResults is None:
        return None
    if not os.environ.get("TAVILY_API_KEY"):
        return None
    return TavilySearchResults(max_results=3)


def select_prompt() -> tuple[ChatPromptTemplate, str]:
    print("\n" + "=" * 60)
    print("SELECT PROMPT TYPE")
    print("=" * 60)
    for key, (ptype, desc) in PROMPT_TYPES.items():
        print(f"  {key}. {ptype:<10} - {desc}")
    print()

    choice = input("  Enter choice [1-4] (default=1): ").strip() or "1"
    prompt_type, _ = PROMPT_TYPES.get(choice, PROMPT_TYPES["1"])
    prompt = build_prompt(prompt_type)
    print(f"\n  Selected prompt : {prompt_type}\n")
    return prompt, prompt_type


def build_generation_chain(router: MultiSourceRouter, prompt: ChatPromptTemplate, llm, retrieval_mode: str = "rewrite_rag"):
    rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_TEMPLATE)
    multiquery_prompt = ChatPromptTemplate.from_template(MULTIQUERY_TEMPLATE)
    rag_fusion_prompt = ChatPromptTemplate.from_template(RAG_FUSION_TEMPLATE)
    web_rewrite_prompt = ChatPromptTemplate.from_template(WEB_REWRITE_TEMPLATE)
    rewriter = rewrite_prompt | llm | StrOutputParser()
    multiquery_rewriter = multiquery_prompt | llm | StrOutputParser()
    rag_fusion_query_generator = rag_fusion_prompt | llm | StrOutputParser()
    web_rewriter = web_rewrite_prompt | llm | StrOutputParser()
    reader_chain = prompt | llm | StrOutputParser()
    web_search = build_web_search_tool()

    def retrieve_local_context(search_query: str) -> tuple[str, list]:
        routed = router.route(search_query, k=6)
        return format_docs(routed.documents), routed.documents

    def retrieve_multiquery_context(question: str) -> tuple[str, list, list[str]]:
        generated = multiquery_rewriter.invoke({"question": question}).strip()
        queries = parse_multi_queries(generated)
        if not queries:
            queries = [question]

        all_docs = []
        for query in queries:
            routed = router.route(query, k=4)
            all_docs.extend(routed.documents)

        deduped_docs = router._deduplicate_docs(all_docs)[:8]
        return format_docs(deduped_docs), deduped_docs, queries

    def retrieve_rag_fusion_context(question: str) -> tuple[str, list, list[str]]:
        generated = rag_fusion_query_generator.invoke({"question": question}).strip()
        queries = parse_multi_queries(generated)
        if not queries:
            queries = [question]

        print("\n" + "=" * 60)
        print("RAG FUSION — GENERATED QUERIES")
        print("=" * 60)
        for i, q in enumerate(queries, start=1):
            print(f"  [{i}] {q}")
        print("=" * 60)

        ranked_results = []
        for i, query in enumerate(queries, start=1):
            routed = router.route(query, k=4)
            ranked_results.append(routed.documents)

            print(f"\n  Query [{i}] retrieved chunks:")
            for rank, doc in enumerate(routed.documents, start=1):
                filename  = doc.metadata.get("filename", "?")
                page      = doc.metadata.get("page", "?")
                snippet   = doc.page_content[:100].replace("\n", " ")
                print(f"    Rank #{rank} | {filename} p.{page} | {snippet!r}...")

        fused_docs = reciprocal_rank_fusion(ranked_results, verbose=True)[:5]
        return format_docs(fused_docs), fused_docs, queries

    def retrieve_web_context(question: str) -> tuple[str, str]:
        web_query = web_rewriter.invoke(
            {"question": question, "current_year": datetime.now().year}
        ).strip()
        if not web_query:
            web_query = question

        if web_search is None:
            return (
                "Web search unavailable. Install `langchain-community` and set `TAVILY_API_KEY` to enable it.",
                web_query,
            )

        results = web_search.invoke({"query": web_query})
        return format_web_results(results), web_query

    @chain
    def rewrite_retrieve_read_chain(user_input: str):
        print(f"Original: {user_input}")
        context_text = ""

        if retrieval_mode == "rewrite_rag":
            optimized_query = rewriter.invoke({"question": user_input}).strip()
            if not optimized_query:
                optimized_query = user_input
            print(f"Rewritten: {optimized_query}")
            context_text, _ = retrieve_local_context(optimized_query)

        elif retrieval_mode == "multiquery_rag":
            context_text, _, queries = retrieve_multiquery_context(user_input)
            print(f"Multi-query: {queries}")

        elif retrieval_mode == "rag_fusion":
            context_text, _, queries = retrieve_rag_fusion_context(user_input)
            print(f"RAG Fusion queries: {queries}")

        elif retrieval_mode == "web_search":
            context_text, web_query = retrieve_web_context(user_input)
            print(f"Web query: {web_query}")

        elif retrieval_mode == "hybrid":
            optimized_query = rewriter.invoke({"question": user_input}).strip()
            if not optimized_query:
                optimized_query = user_input
            print(f"Rewritten: {optimized_query}")
            local_context, _ = retrieve_local_context(optimized_query)
            web_context, web_query = retrieve_web_context(user_input)
            print(f"Web query: {web_query}")
            context_text = "\n\n".join(
                [
                    "Local document context:",
                    local_context,
                    "Web search context:",
                    web_context,
                ]
            )
        else:
            context_text, _ = retrieve_local_context(user_input)

        return reader_chain.invoke(
            {
                "context": context_text,
                "question": user_input,
            }
        )

    return rewrite_retrieve_read_chain


def show_route_debug(router: MultiSourceRouter, query: str):
    routed = router.route(query, k=6)
    print("\n" + "=" * 60)
    print("ROUTE DEBUG")
    print("=" * 60)
    print(f"  Raw query       : {routed.processed_query.raw_query}")
    print(f"  Retrieval query : {routed.processed_query.retrieval_query}")
    print(f"  Search mode     : {routed.processed_query.search_mode}")
    print(f"  Filters         : {routed.processed_query.filters or '{}'}")
    print(f"  Route           : {routed.route_name}")
    print(f"  Sources         : {routed.selected_sources or ['global']}")
    print(f"  Issues          : {routed.processed_query.issues or ['none']}")
    print(f"  Variants        : {routed.processed_query.variants}")
    print("\n  Retrieved chunks:")
    for index, doc in enumerate(routed.documents, start=1):
        print(f"  [{index}] {doc.metadata.get('filename', '?')} | Page {doc.metadata.get('page', '?')} | Source {doc.metadata.get('source_id', '?')}")
        print(f"      {doc.page_content[:180]!r}")
    print("=" * 60 + "\n")


def show_prompt_preview(router: MultiSourceRouter, prompt: ChatPromptTemplate, query: str):
    routed = router.route(query, k=6)
    rendered = prompt.format(
        context=format_docs(routed.documents),
        question=routed.processed_query.retrieval_query,
    )
    print("\n" + "=" * 60)
    print("RENDERED PROMPT PREVIEW")
    print("=" * 60)
    print(rendered[:2000] + ("..." if len(rendered) > 2000 else ""))
    print("=" * 60 + "\n")


def show_query_rewrites(llm, query: str):
    rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_TEMPLATE)
    multiquery_prompt = ChatPromptTemplate.from_template(MULTIQUERY_TEMPLATE)
    web_rewrite_prompt = ChatPromptTemplate.from_template(WEB_REWRITE_TEMPLATE)

    rewriter = rewrite_prompt | llm | StrOutputParser()
    multiquery_rewriter = multiquery_prompt | llm | StrOutputParser()
    web_rewriter = web_rewrite_prompt | llm | StrOutputParser()

    rag_query = rewriter.invoke({"question": query}).strip()
    multi_queries = parse_multi_queries(multiquery_rewriter.invoke({"question": query}))
    web_query = web_rewriter.invoke({"question": query, "current_year": datetime.now().year}).strip()

    print("\n" + "=" * 60)
    print("QUERY REWRITE PREVIEW")
    print("=" * 60)
    print(f"  Original      : {query}")
    print(f"  RAG rewrite   : {rag_query or query}")
    print(f"  Web rewrite   : {web_query or query}")
    print(f"  Multi-query   : {multi_queries or [query]}")
    print("=" * 60 + "\n")


def generate_stream(chain_obj, query: str):
    print("\nAnswer (streaming):\n")
    for token in chain_obj.stream(query):
        print(token, end="", flush=True)
    print("\n")


def generate_batch(chain_obj, queries: list[str]):
    print(f"\n  Running batch generation for {len(queries)} queries...")
    answers = chain_obj.batch(queries)
    for index, (question, answer) in enumerate(zip(queries, answers), start=1):
        print(f"\n  [{index}] Query  : {question}")
        print(f"       Answer : {answer}")
    print()
    return answers



def main():
    print("=" * 60)
    print("ADVANCED RAG GENERATION PIPELINE")
    print("Embeddings : HuggingFace")
    print("LLM        : Ollama llama3.2:1b")
    print("=" * 60 + "\n")
    
    print("[1/3] Building source-aware indexes...") # ✅ New — points to a specific uploads folder
    import shutil
    from pathlib import Path
    
    PDF_FOLDER = Path(__file__).parent / "rag_docs"   # ← change folder name here if needed
    
    # Auto-create the folder if it doesn't exist yet
    PDF_FOLDER.mkdir(exist_ok=True)
    
    print("[1/3] Building source-aware indexes...")
    print(f"  PDF folder : {PDF_FOLDER.resolve()}")
    
    # Check folder isn't empty before proceeding
    pdf_files_found = list(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files_found:
        print(f"\n  [ERROR] No PDF files found in '{PDF_FOLDER}'")
        print(f"  Drop your PDF files into that folder and restart.\n")
        sys.exit(1)
    
    print(f"  PDFs found : {[f.name for f in pdf_files_found]}\n")
    
    index_bundle = build_advanced_retrieval_index(k=6, pdf_folder=PDF_FOLDER)
    #index_bundle = build_advanced_retrieval_index(k=6)
    router = MultiSourceRouter(
        indexed_sources=index_bundle["indexed_sources"],
        global_retriever=index_bundle["global_retriever"],
        default_k=6,
    )
    print(f"  Indexed sources : {len(index_bundle['indexed_sources'])}")
    print(f"  Total chunks    : {len(index_bundle['all_splits'])}\n")
    
    print("[2/3] Initializing prompt...")
    current_prompt, current_label = build_prompt("default"), "default"
    print(f"  Active prompt   : {current_label}\n")
    
    print("[3/3] Initializing Ollama LLM...")
    llm = ChatOllama(model="llama3.2:1b", temperature=0)
    current_mode = "rewrite_rag"
    chain_obj = build_generation_chain(router, current_prompt, llm, current_mode)
    print("  LLM ready       : llama3.2:1b\n")
    print(f"  Web search      : {'enabled' if build_web_search_tool() else 'disabled'}")
    if not build_web_search_tool():
        print("  Note            : install `langchain-community` and set `TAVILY_API_KEY` for web mode")
    print()
    
    print("=" * 60)
    print("Advanced Generation Ready! Commands:")
    print("  prompt            - switch prompt type")
    print("  mode              - switch retrieval mode")
    print("  preview:<query>   - show rendered prompt")
    print("  rewrite:<query>   - show RAG/web/multiquery rewrites")
    print("  route:<query>     - show preprocessing + routed chunks")
    print("  stream:<query>    - stream answer token by token")
    print("  batch:<q1>|<q2>   - batch multiple queries")
    print("  <query>           - standard generate answer")
    print("  exit              - quit")
    print("=" * 60 + "\n")
    
    while True:
        user_input = input("Query: ").strip()
    
        if user_input.lower() in ("exit", "quit"):
            print("Exiting advanced generation pipeline.")
            break
    
        if not user_input:
            continue
    
        if user_input.lower() == "prompt":
            current_prompt, current_label = select_prompt()
            chain_obj = build_generation_chain(router, current_prompt, llm, current_mode)
            print(f"  Prompt switched to : {current_label}\n")
            continue
    
        if user_input.lower() == "mode":
            current_mode = select_retrieval_mode()
            chain_obj = build_generation_chain(router, current_prompt, llm, current_mode)
            print(f"  Retrieval mode    : {current_mode}\n")
            continue
    
        if user_input.startswith("preview:"):
            show_prompt_preview(router, current_prompt, user_input[8:].strip())
            continue
    
        if user_input.startswith("rewrite:"):
            show_query_rewrites(llm, user_input[8:].strip())
            continue
    
        if user_input.startswith("route:"):
            show_route_debug(router, user_input[6:].strip())
            continue
    
        if user_input.startswith("stream:"):
            print(f"\n  [Prompt: {current_label} | Mode: {current_mode}] Streaming answer...")
            generate_stream(chain_obj, user_input[7:].strip())
            print("-" * 60 + "\n")
            continue
    
        if user_input.startswith("batch:"):
            raw = user_input[6:].strip()
            queries = [query.strip() for query in raw.split("|") if query.strip()]
            if not queries:
                print("  [ERROR] No queries found. Use: batch:query1|query2|query3\n")
                continue
            print(f"\n  [Prompt: {current_label} | Mode: {current_mode}] Batch generation...")
            generate_batch(chain_obj, queries)
            print("-" * 60 + "\n")
            continue
    
        print(f"\n  [Prompt: {current_label} | Mode: {current_mode}] Generating answer...")
        answer = chain_obj.invoke(user_input)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
