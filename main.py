"""
Main executable launcher for the RAG chat project.

Examples:
    python main.py
    python main.py generation
    python main.py retrieval
    python main.py --list

The selected workflow is executed with runpy so module code runs only when the
user chooses it.
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


WORKFLOWS = (
    Workflow(
        key="generation",
        script="generation_rag.py",
        description="Full interactive RAG chat model with prompt/debug/stream commands.",
        aliases=("chat", "rag", "default"),
    ),
    Workflow(
        key="retrieval",
        script="retrieval_rag.py",
        description="Run retrieval/indexing tests without the full chat loop.",
        aliases=("index", "search"),
    ),
    Workflow(
        key="basic",
        script="ollama_rag.py",
        description="Compact Ollama RAG demo over PDFs in rag_docs.",
        aliases=("simple",),
    ),
    Workflow(
        key="ollama-chat",
        script="ollama_chat.py",
        description="Basic Ollama chat connectivity test.",
        aliases=("test-chat",),
    ),
    Workflow(
        key="ollama-index",
        script="IndexingData_for_rag_langchain_ollama.py",
        description="Older multi-document Ollama embedding/indexing workflow.",
        aliases=("ollama-rag-index",),
    ),
    Workflow(
        key="advanced",
        script="advanced_generation_rag.py",
        description="Advanced generation workflow, if this optional script exists.",
        aliases=("advanced-generation",),
    ),
    Workflow(
        key="memory",
        script="langgraph_rag_memory.py",
        description="LangGraph RAG memory workflow, if this optional script exists.",
        aliases=("langgraph", "langgraph-memory"),
    ),
)


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_workflow(name: str) -> Workflow | None:
    normalized = name.strip().lower()
    for workflow in WORKFLOWS:
        names = (workflow.key, *workflow.aliases)
        if normalized in names:
            return workflow
    return None


def available_workflows() -> list[tuple[Workflow, bool]]:
    root = project_dir()
    return [(workflow, (root / workflow.script).exists()) for workflow in WORKFLOWS]


def print_workflows() -> None:
    print("\nAvailable workflows:\n")
    for index, (workflow, exists) in enumerate(available_workflows(), start=1):
        status = "ready" if exists else "missing"
        aliases = f" | aliases: {', '.join(workflow.aliases)}" if workflow.aliases else ""
        print(f"  {index}. {workflow.key:<13} [{status}] {workflow.description}")
        print(f"     script: {workflow.script}{aliases}")
    print()


def choose_workflow() -> Workflow:
    ready = [(workflow, exists) for workflow, exists in available_workflows() if exists]
    if not ready:
        raise FileNotFoundError("No workflow scripts were found in this project folder.")

    print_workflows()
    choice = input("Select workflow number or name [generation]: ").strip() or "generation"

    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(WORKFLOWS):
            return WORKFLOWS[index - 1]
        raise ValueError(f"Invalid workflow number: {choice}")

    workflow = resolve_workflow(choice)
    if workflow is None:
        raise ValueError(f"Unknown workflow: {choice}")
    return workflow


def run_workflow(workflow: Workflow) -> None:
    script_path = project_dir() / workflow.script
    if not script_path.exists():
        raise FileNotFoundError(
            f"Workflow '{workflow.key}' requires missing script: {script_path}"
        )

    print(f"\nRunning workflow: {workflow.key}")
    print(f"Script: {script_path}\n")
    runpy.run_path(str(script_path), run_name="__main__")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one RAG project workflow from a single executable file."
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        help="Workflow name or alias. Use --list to see options.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available workflows and exit.",
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

    run_workflow(workflow)


if __name__ == "__main__":
    main()
