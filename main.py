"""
Main executable launcher for the current Advanced RAG + LangGraph project.

Examples:
    python main.py
    python main.py memory
    python main.py advanced
    python main.py indexing
    python main.py --list
    python main.py memory --pdf-folder .\\rag_docs

The selected workflow is executed only when requested. Import/helper modules
remain available to the runnable scripts, but they are not launched directly.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workflow:
    key: str
    script: str
    description: str
    aliases: tuple[str, ...] = ()
    accepts_pdf_folder: bool = False


WORKFLOWS = (
    Workflow(
        key="memory",
        script="langgraph_rag_memory.py",
        description="Complete LangGraph agentic + knowledge + memory RAG CLI.",
        aliases=("default", "langgraph", "langgraph-memory", "chat"),
        accepts_pdf_folder=True,
    ),
    Workflow(
        key="advanced",
        script="advanced_generation_rag.py",
        description="Advanced RAG generation CLI with prompt, mode, route, rewrite, stream, and batch commands.",
        aliases=("generation", "advanced-generation", "rag"),
    ),
    Workflow(
        key="indexing",
        script="IndexingDocs_for_rag.py",
        description="Older standalone indexing and simple RAG demo pipeline.",
        aliases=("index", "base", "demo"),
    ),
)

UTILITY_MODULES = (
    ("langgraph_rag_graph.py", "Defines AgentState, graph topology, conditional routing, and router builder."),
    ("langgraph_rag_nodes.py", "Contains LangGraph query, retrieval, grading, generation, and memory nodes."),
    ("advanced_rag_indexing_2.py", "Current source-aware PDF indexing utilities used by LangGraph and advanced generation."),
    ("advanced_rag_router.py", "Routes processed queries to source-specific or global retrievers."),
    ("advanced_rag_query_processing.py", "Normalizes questions and extracts source/file filters."),
    ("advanced_rag_indexing.py", "Earlier indexing implementation kept as reference."),
)


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_workflow(name: str) -> Workflow | None:
    normalized = name.strip().lower()
    for workflow in WORKFLOWS:
        if normalized in (workflow.key, *workflow.aliases):
            return workflow
    return None


def available_workflows() -> list[tuple[Workflow, bool]]:
    root = project_dir()
    return [(workflow, (root / workflow.script).is_file()) for workflow in WORKFLOWS]


def print_workflows() -> None:
    print("\nRunnable workflows:\n")
    for index, (workflow, exists) in enumerate(available_workflows(), start=1):
        status = "ready" if exists else "missing"
        aliases = f" | aliases: {', '.join(workflow.aliases)}" if workflow.aliases else ""
        print(f"  {index}. {workflow.key:<10} [{status}] {workflow.description}")
        print(f"     script: {workflow.script}{aliases}")

    print("\nIncluded helper modules:\n")
    root = project_dir()
    for module, description in UTILITY_MODULES:
        status = "found" if (root / module).is_file() else "missing"
        print(f"  - {module:<32} [{status}] {description}")
    print()


def ready_workflows() -> list[Workflow]:
    return [workflow for workflow, exists in available_workflows() if exists]


def choose_workflow() -> Workflow:
    ready = ready_workflows()
    if not ready:
        raise FileNotFoundError("No runnable workflow scripts were found in this project folder.")

    print_workflows()
    choice = input("Select workflow number or name [memory]: ").strip() or "memory"

    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(WORKFLOWS):
            return WORKFLOWS[index - 1]
        raise ValueError(f"Invalid workflow number: {choice}")

    workflow = resolve_workflow(choice)
    if workflow is None:
        raise ValueError(f"Unknown workflow: {choice}")
    return workflow


def run_workflow(workflow: Workflow, pdf_folder: str | None = None) -> None:
    script_path = project_dir() / workflow.script
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Workflow '{workflow.key}' requires missing script: {script_path}"
        )

    forwarded_args: list[str] = []
    if pdf_folder:
        if not workflow.accepts_pdf_folder:
            raise ValueError(f"Workflow '{workflow.key}' does not accept --pdf-folder.")
        forwarded_args.append(str(Path(pdf_folder).expanduser()))

    print(f"\nRunning workflow: {workflow.key}")
    print(f"Script: {script_path}")
    if forwarded_args:
        print(f"Arguments: {' '.join(forwarded_args)}")
    print()

    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *forwarded_args]
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one workflow from the current Advanced RAG + LangGraph project."
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        help="Workflow name or alias. Default: memory. Use --list to see options.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List runnable workflows and included helper modules, then exit.",
    )
    parser.add_argument(
        "--pdf-folder",
        help="Optional PDF folder for the memory workflow, for example .\\rag_docs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.list:
        print_workflows()
        return

    workflow = resolve_workflow(args.workflow) if args.workflow else choose_workflow()
    if workflow is None:
        print_workflows()
        raise ValueError(f"Unknown workflow: {args.workflow}")

    run_workflow(workflow, pdf_folder=args.pdf_folder)


if __name__ == "__main__":
    main()
