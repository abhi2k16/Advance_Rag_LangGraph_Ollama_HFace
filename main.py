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
    module: str
    description: str
    aliases: tuple[str, ...] = ()
    accepts_pdf_folder: bool = False


WORKFLOWS = (
    Workflow(
        key="memory",
        module="langgraph_impl.memory_cli",
        description="Complete LangGraph agentic + knowledge + memory RAG CLI.",
        aliases=("default", "langgraph", "langgraph-memory", "chat"),
        accepts_pdf_folder=True,
    ),
    Workflow(
        key="advanced",
        module="langchain_impl.generation_pipeline",
        description="Advanced RAG generation CLI with prompt, mode, route, rewrite, stream, and batch commands.",
        aliases=("generation", "advanced-generation", "rag"),
    ),
    Workflow(
        key="indexing",
        module="langchain_impl.indexing_core",
        description="Older standalone indexing and simple RAG demo pipeline.",
        aliases=("index", "base", "demo"),
    ),
)

UTILITY_MODULES = (
    ("langgraph_impl.graph_builder", "Defines AgentState, graph topology, conditional routing, and router builder."),
    ("langgraph_impl.graph_nodes", "Contains LangGraph query, retrieval, grading, generation, and memory nodes."),
    ("langchain_impl.retrieval_index", "Current source-aware PDF indexing utilities used by LangGraph and advanced generation."),
    ("langchain_impl.retriever_router", "Routes processed queries to source-specific or global retrievers."),
    ("langchain_impl.query_processing", "Normalizes questions and extracts source/file filters."),
    ("langchain_impl.legacy_indexing", "Earlier indexing implementation kept as reference."),
)


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def module_path(module: str) -> Path:
    return project_dir() / Path(*module.split(".")).with_suffix(".py")


def resolve_workflow(name: str) -> Workflow | None:
    normalized = name.strip().lower()
    for workflow in WORKFLOWS:
        if normalized in (workflow.key, *workflow.aliases):
            return workflow
    return None


def available_workflows() -> list[tuple[Workflow, bool]]:
    return [(workflow, module_path(workflow.module).is_file()) for workflow in WORKFLOWS]


def print_workflows() -> None:
    print("\nRunnable workflows:\n")
    for index, (workflow, exists) in enumerate(available_workflows(), start=1):
        status = "ready" if exists else "missing"
        aliases = f" | aliases: {', '.join(workflow.aliases)}" if workflow.aliases else ""
        print(f"  {index}. {workflow.key:<10} [{status}] {workflow.description}")
        print(f"     module: {workflow.module}{aliases}")

    print("\nIncluded helper modules:\n")
    for module, description in UTILITY_MODULES:
        status = "found" if module_path(module).is_file() else "missing"
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
    module_file = module_path(workflow.module)
    if not module_file.is_file():
        raise FileNotFoundError(
            f"Workflow '{workflow.key}' requires missing module file: {module_file}"
        )

    forwarded_args: list[str] = []
    if pdf_folder:
        if not workflow.accepts_pdf_folder:
            raise ValueError(f"Workflow '{workflow.key}' does not accept --pdf-folder.")
        forwarded_args.append(str(Path(pdf_folder).expanduser()))

    print(f"\nRunning workflow: {workflow.key}")
    print(f"Module: {workflow.module}")
    if forwarded_args:
        print(f"Arguments: {' '.join(forwarded_args)}")
    print()

    old_argv = sys.argv[:]
    try:
        sys.argv = [workflow.module, *forwarded_args]
        runpy.run_module(workflow.module, run_name="__main__")
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
