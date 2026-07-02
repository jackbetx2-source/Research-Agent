from __future__ import annotations

import json

from .types import Intent, ResearchBranch


INTENT_SYSTEM_PROMPT = """
You are LLM1, the coordinator of a research workflow.
Your job is to understand the user's research topic and convert it into a clear research brief.
Return only valid JSON with this schema:
{
  "summary": "one concise sentence describing the user's true intent",
  "key_questions": ["3 to 6 concrete research questions"],
  "scope": "what should be included and excluded"
}
""".strip()


def intent_user_prompt(topic: str) -> str:
    return f"User research topic:\n{topic}"


RESEARCH_BRANCHES = [
    {
        "name": "Context Researcher",
        "focus": "background, definitions, history, current landscape, and important concepts",
    },
    {
        "name": "Evidence Researcher",
        "focus": "arguments, examples, patterns, tradeoffs, and practical evidence",
    },
    {
        "name": "Critical Researcher",
        "focus": "risks, limitations, counterarguments, blind spots, and open questions",
    },
]


def branch_system_prompt(name: str, focus: str) -> str:
    return f"""
You are {name}, one of three parallel LLM researchers.
Your focus: {focus}.

Produce a compact research memo in Markdown.
Keep it concise: 5 to 8 bullets total, no long tables, no code blocks, no preamble.
Focus on the highest-signal findings only.
Use the same language as the user's research topic when possible.
Be specific, structured, and avoid unsupported certainty.
Do not synthesize the final answer; only produce your branch's findings.
""".strip()


def branch_user_prompt(intent: Intent) -> str:
    payload = {
        "original_topic": intent.original_topic,
        "interpreted_intent": intent.summary,
        "key_questions": intent.key_questions,
        "scope": intent.scope,
    }
    return "Research brief:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


SYNTHESIS_SYSTEM_PROMPT = """
You are LLM1 again, now acting as the final synthesizer.
You will receive the original intent plus three parallel research memos.

Create the final research output in Markdown.
Use this structure:
1. Executive summary
2. Key findings
3. Detailed analysis
4. Risks and limitations
5. Suggested next steps

Resolve contradictions explicitly. Do not simply concatenate the branch outputs.
""".strip()


def synthesis_user_prompt(intent: Intent, branches: list[ResearchBranch]) -> str:
    branch_blocks = []
    for branch in branches:
        branch_blocks.append(
            f"## {branch.name}\nFocus: {branch.focus}\n\n{branch.content}"
        )

    payload = {
        "original_topic": intent.original_topic,
        "interpreted_intent": intent.summary,
        "key_questions": intent.key_questions,
        "scope": intent.scope,
    }

    return (
        "Intent:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nParallel research memos:\n\n"
        + "\n\n".join(branch_blocks)
    )
