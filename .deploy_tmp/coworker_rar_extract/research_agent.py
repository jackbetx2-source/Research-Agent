from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.research_agent.mock_llm import MockLLMClient
from src.research_agent.workflow import ResearchWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research agent workflow.")
    parser.add_argument("topic", nargs="*", help="Research topic from the user.")
    parser.add_argument(
        "--show-branches",
        action="store_true",
        help="Print each parallel branch output before the final synthesis.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the workflow with a local mock LLM instead of calling an API.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print workflow progress while running.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use shorter branch memos and a shorter final report for faster runs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the report body to the terminal.",
    )
    parser.add_argument(
        "--output",
        help="Write the full workflow result to a Markdown file.",
    )
    return parser.parse_args()


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    root = Path(__file__).resolve().parent
    env_path = root / ".env"
    example_env_path = root / ".env.example"
    load_dotenv(env_path if env_path.exists() else example_env_path)
    args = parse_args()
    topic = " ".join(args.topic).strip()

    if not topic:
        topic = input("Research topic: ").strip()

    if not topic:
        raise SystemExit("Topic cannot be empty.")

    workflow = ResearchWorkflow(
        llm=MockLLMClient() if args.mock else None,
        verbose=args.verbose,
        fast=args.fast,
    )
    result = await workflow.run(topic)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown_result(result), encoding="utf-8")
        print(f"\nSaved full result to: {output_path.resolve()}")

    if not args.quiet:
        print("\n=== Intent ===\n")
        print(result.intent.summary)

        if args.show_branches:
            print("\n=== Parallel Research Branches ===")
            for branch in result.branches:
                print(f"\n--- {branch.name} ---\n")
                print(branch.content)

        print("\n=== Final Report ===\n")
        print(result.final_report)


def render_markdown_result(result) -> str:
    branches = []
    for branch in result.branches:
        branches.append(
            f"## {branch.name}\n\n"
            f"**Focus:** {branch.focus}\n\n"
            f"{branch.content}"
        )

    references = "\n".join(format_reference(reference) for reference in result.references)
    if not references:
        references = "- 暂无明确文献"

    return (
        "# Research Agent Result\n\n"
        "## Intent\n\n"
        f"**Original topic:** {result.intent.original_topic}\n\n"
        f"**Summary:** {result.intent.summary}\n\n"
        "**Key questions:**\n"
        + "\n".join(f"- {question}" for question in result.intent.key_questions)
        + f"\n\n**Scope:** {result.intent.scope}\n\n"
        "# Parallel Research Branches\n\n"
        + "\n\n".join(branches)
        + "\n\n# Ranked References\n\n"
        + references
        + "\n\n# Final Report\n\n"
        + result.final_report
        + "\n"
    )


def format_reference(reference) -> str:
    title = reference.title
    if reference.source and reference.source != "未提供" and reference.source.startswith(("http://", "https://")):
        title = f"[{title}]({reference.source})"
    return f"- **{title}** ({reference.branch_name}) — {reference.relevance}"


if __name__ == "__main__":
    asyncio.run(main())
