"""
langgraph_rag_memory.py
───────────────────────
Memory layer + interactive CLI for the LangGraph agentic RAG system.

Responsibilities
─────────────────
  1. Attach MemorySaver to the compiled graph so every thread_id gets its
     own persistent conversation history across turns (Memory RAG).
  2. Expose helper functions for single-turn invoke, streaming, and batch.
  3. Provide a full interactive CLI with mode/prompt switching.

Three RAG types are all accessible via the same compiled graph:
  --mode rewrite_rag    →  Knowledge RAG  (local docs, query rewrite)
  --mode rag_fusion     →  Knowledge RAG  (multi-query + RRF reranking)
  --mode multiquery_rag →  Knowledge RAG  (multi-query retrieval)
  --mode web_search     →  Web RAG        (Tavily)
  --mode hybrid         →  Hybrid RAG     (local + web merged)
  (all modes pass through the grader node → Agentic RAG)
  (all modes write to MemorySaver      → Memory RAG)

No existing .py files are modified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Optional
from uuid import uuid4

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import MemorySaver

# ── Graph + index builder (new files only) ────────────────────────────────────
from langgraph_impl.graph_builder import AgentState, build_router, compile_graph
from langgraph_impl.graph_nodes import register_runtime_router

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RETRIEVAL_MODES = {
    "1": ("rewrite_rag",    "Local docs — single rewrite"),
    "2": ("multiquery_rag", "Local docs — multi-query expansion"),
    "3": ("rag_fusion",     "Local docs — multi-query + RRF reranking"),
    "4": ("web_search",     "Web only   — Tavily search"),
    "5": ("hybrid",         "Hybrid     — local docs + web merged"),
}

PROMPT_TYPES = {
    "1": ("default",  "Standard helpful assistant"),
    "2": ("concise",  "1-2 sentence answer"),
    "3": ("detailed", "Detailed answer with citations"),
    "4": ("bullet",   "Bullet point answer"),
}


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY SESSION
# Wraps a compiled graph + MemorySaver + a thread_id for one conversation.
# ══════════════════════════════════════════════════════════════════════════════

class MemoryRAGSession:
    """
    One conversation session backed by LangGraph MemorySaver.

    Each session has a unique thread_id so LangGraph persists the full
    graph state between turns — this is the Memory RAG layer.

    Usage
    ──────
    session = MemoryRAGSession(router)
    answer  = session.invoke("What is attention?")
    answer2 = session.invoke("How does it compare to RNNs?")  # remembers context
    """

    def __init__(
        self,
        router,
        thread_id: Optional[str] = None,
        retrieval_mode: str = "rewrite_rag",
        prompt_type: str = "default",
        top_k: int = 6,
        max_retries: int = 2,
    ):
        self.router = router
        self.thread_id = thread_id or str(uuid4())
        register_runtime_router(self.thread_id, router)
        self.retrieval_mode = retrieval_mode
        self.prompt_type = prompt_type
        self.top_k = top_k
        self.max_retries = max_retries

        # MemorySaver stores the full AgentState keyed by (thread_id, checkpoint_id)
        self.memory = MemorySaver()
        self.app = compile_graph(checkpointer=self.memory)

        print(f"\n[MemoryRAGSession] thread_id = {self.thread_id}")
        print(f"  mode={retrieval_mode}  prompt={prompt_type}  k={top_k}  max_retries={max_retries}")

    # ── Config helper ──────────────────────────────────────────────────────
    @property
    def _config(self) -> dict:
        """LangGraph config dict carrying the thread_id for MemorySaver."""
        return {"configurable": {"thread_id": self.thread_id}}

    # ── Build the initial state for this turn ──────────────────────────────
    def _build_input(self, question: str) -> AgentState:
        return AgentState(
            raw_query=question,
            retrieval_mode=self.retrieval_mode,
            prompt_type=self.prompt_type,
            top_k=self.top_k,
            max_retries=self.max_retries,
            retry_count=0,
        )

    # ── Single-turn invoke ─────────────────────────────────────────────────
    def invoke(self, question: str) -> str:
        """Run one turn and return the final answer string."""
        result = self.app.invoke(self._build_input(question), config=self._config)
        return result.get("answer", "[No answer generated]")

    # ── Streaming token-by-token ───────────────────────────────────────────
    def stream(self, question: str) -> Iterator[str]:
        """
        Stream node-level state updates.
        Yields the answer string once the generation_node completes.
        """
        for chunk in self.app.stream(
            self._build_input(question), config=self._config, stream_mode="updates"
        ):
            # Each chunk is {node_name: partial_state}
            if "generation_node" in chunk:
                answer = chunk["generation_node"].get("answer", "")
                if answer:
                    yield answer

    # ── Batch (multiple questions in one session) ──────────────────────────
    def batch(self, questions: list[str]) -> list[str]:
        """Run multiple questions sequentially in this session."""
        return [self.invoke(q) for q in questions]

    # ── Inspect memory checkpoint ──────────────────────────────────────────
    def get_state(self) -> dict:
        """Return the last checkpointed state for this thread."""
        snapshot = self.app.get_state(self._config)
        return dict(snapshot.values) if snapshot else {}

    def get_history(self) -> list:
        """Return all checkpointed state snapshots for this thread."""
        return list(self.app.get_state_history(self._config))


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-SESSION MANAGER
# Manages named sessions (useful for multi-user or topic-separated contexts).
# ══════════════════════════════════════════════════════════════════════════════

class SessionManager:
    """
    Maintain a pool of MemoryRAGSession objects keyed by session name.

    Usage
    ──────
    mgr = SessionManager(router)
    mgr.new_session("alice")
    print(mgr.ask("alice", "Explain transformers"))
    print(mgr.ask("alice", "What about positional encoding?"))
    """

    def __init__(self, router, **session_defaults):
        self.router = router
        self.defaults = session_defaults
        self._sessions: dict[str, MemoryRAGSession] = {}

    def new_session(
        self,
        name: str,
        retrieval_mode: str = "rewrite_rag",
        prompt_type: str = "default",
    ) -> MemoryRAGSession:
        """Create (or replace) a named session."""
        session = MemoryRAGSession(
            router=self.router,
            retrieval_mode=retrieval_mode,
            prompt_type=prompt_type,
            **self.defaults,
        )
        self._sessions[name] = session
        print(f"[SessionManager] Created session '{name}' (thread={session.thread_id})")
        return session

    def get_session(self, name: str) -> MemoryRAGSession:
        if name not in self._sessions:
            raise KeyError(f"Session '{name}' not found. Call new_session('{name}') first.")
        return self._sessions[name]

    def ask(self, session_name: str, question: str) -> str:
        return self.get_session(session_name).invoke(question)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# ══════════════════════════════════════════════════════════════════════════════
# CLI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _select_mode() -> str:
    print("\n" + "=" * 60)
    print("SELECT RETRIEVAL MODE")
    print("=" * 60)
    for k, (mode, desc) in RETRIEVAL_MODES.items():
        print(f"  {k}. {mode:<16} — {desc}")
    choice = input("\n  Choice [1-5] (default=1): ").strip() or "1"
    mode, _ = RETRIEVAL_MODES.get(choice, RETRIEVAL_MODES["1"])
    print(f"  → {mode}\n")
    return mode


def _select_prompt() -> str:
    print("\n" + "=" * 60)
    print("SELECT PROMPT TYPE")
    print("=" * 60)
    for k, (ptype, desc) in PROMPT_TYPES.items():
        print(f"  {k}. {ptype:<10} — {desc}")
    choice = input("\n  Choice [1-4] (default=1): ").strip() or "1"
    ptype, _ = PROMPT_TYPES.get(choice, PROMPT_TYPES["1"])
    print(f"  → {ptype}\n")
    return ptype


def _print_state_summary(session: MemoryRAGSession):
    state = session.get_state()
    if not state:
        print("  [No checkpointed state found]")
        return
    print(f"  thread_id      : {session.thread_id}")
    print(f"  retrieval_mode : {state.get('retrieval_mode', '?')}")
    print(f"  grade          : {state.get('grade', '?')}")
    print(f"  retry_count    : {state.get('retry_count', '?')}")
    print(f"  docs retrieved : {len(state.get('retrieved_docs', []))}")
    print(f"  memory turns   : {len(state.get('conversation_history', [])) // 2}")
    answer = state.get("answer", "")
    if answer:
        print(f"  last answer    : {answer[:120]}{'...' if len(answer) > 120 else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CLI
# ══════════════════════════════════════════════════════════════════════════════

def run_cli(pdf_folder: Optional[Path] = None):
    """
    Full interactive CLI for the LangGraph agentic + knowledge + memory RAG.

    Commands
    ─────────
      mode            — switch retrieval mode
      prompt          — switch prompt type
      session <name>  — switch to (or create) a named session
      sessions        — list active sessions
      state           — show current session's checkpointed state
      history         — show number of checkpoints in current session
      stream:<query>  — stream answer token by token
      batch:<q1>|<q2> — run multiple questions in current session
      new             — start a fresh session (new thread_id)
      exit            — quit
      <query>         — standard invoke
    """
    print("\n" + "=" * 60)
    print("LANGGRAPH AGENTIC + KNOWLEDGE + MEMORY RAG")
    print("Embeddings : HuggingFace all-MiniLM-L6-v2")
    print("LLM        : Ollama llama3.2:1b")
    print("Memory     : LangGraph MemorySaver")
    print("=" * 60 + "\n")

    # Build the retrieval index once
    router = build_router(pdf_folder=pdf_folder, k=6)

    # Start default session
    mode = "rewrite_rag"
    prompt = "default"
    mgr = SessionManager(router)
    mgr.new_session("default", retrieval_mode=mode, prompt_type=prompt)
    active = "default"

    print("=" * 60)
    print("Commands: mode | prompt | session <name> | sessions | state")
    print("          history | new | stream:<q> | batch:<q1>|<q2> | exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input(f"[{active}][{mode}][{prompt}] Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        low = user_input.lower()

        # ── Meta commands ──────────────────────────────────────────────────
        if low in ("exit", "quit"):
            print("Exiting LangGraph RAG pipeline.")
            break

        if low == "mode":
            mode = _select_mode()
            mgr.get_session(active).retrieval_mode = mode
            print(f"  Retrieval mode updated to: {mode}\n")
            continue

        if low == "prompt":
            prompt = _select_prompt()
            mgr.get_session(active).prompt_type = prompt
            print(f"  Prompt type updated to: {prompt}\n")
            continue

        if low.startswith("session "):
            name = user_input[8:].strip()
            if name not in mgr.list_sessions():
                mgr.new_session(name, retrieval_mode=mode, prompt_type=prompt)
            active = name
            print(f"  Switched to session '{active}'\n")
            continue

        if low == "sessions":
            print(f"  Active sessions: {mgr.list_sessions()}")
            print(f"  Current        : {active}\n")
            continue

        if low == "state":
            print(f"\n  State for session '{active}':")
            _print_state_summary(mgr.get_session(active))
            print()
            continue

        if low == "history":
            history = mgr.get_session(active).get_history()
            print(f"  Checkpoints in '{active}': {len(history)}\n")
            continue

        if low == "new":
            new_name = f"session_{len(mgr.list_sessions()) + 1}"
            mgr.new_session(new_name, retrieval_mode=mode, prompt_type=prompt)
            active = new_name
            print(f"  Started fresh session '{active}'\n")
            continue

        # ── Stream ─────────────────────────────────────────────────────────
        if low.startswith("stream:"):
            query = user_input[7:].strip()
            print(f"\n  Streaming answer...\n")
            got_output = False
            for chunk in mgr.get_session(active).stream(query):
                print(chunk, end="", flush=True)
                got_output = True
            if not got_output:
                # stream() yields only on generation_node — fallback to invoke
                answer = mgr.ask(active, query)
                print(answer, end="")
            print("\n" + "-" * 60 + "\n")
            continue

        # ── Batch ──────────────────────────────────────────────────────────
        if low.startswith("batch:"):
            parts = [q.strip() for q in user_input[6:].split("|") if q.strip()]
            if not parts:
                print("  Usage: batch:question 1|question 2|question 3\n")
                continue
            print(f"\n  Running batch ({len(parts)} questions)...\n")
            answers = mgr.get_session(active).batch(parts)
            for i, (q, a) in enumerate(zip(parts, answers), 1):
                print(f"  [{i}] Q: {q}")
                print(f"       A: {a}\n")
            print("-" * 60 + "\n")
            continue

        # ── Standard invoke ────────────────────────────────────────────────
        print(f"\n  Generating answer (mode={mode}, prompt={prompt})...")
        answer = mgr.ask(active, user_input)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Optional: pass a custom PDF folder as the first CLI argument
    # e.g.  python langgraph_rag_memory.py /path/to/pdfs
    pdf_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_cli(pdf_folder=pdf_folder)
