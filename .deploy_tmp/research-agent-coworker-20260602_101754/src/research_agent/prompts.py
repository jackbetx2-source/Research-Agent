from __future__ import annotations

import json

from .types import Intent, ReportOutlineSection, ResearchBranch, ResearchPerspective


PROMPT_TOPIC_CHARS = 9000
PROMPT_BRANCH_CHARS = 4500
PROMPT_SYNTHESIS_BRANCH_CHARS = 2500


def _clip(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[Content clipped to keep the workflow within model limits.]"


INTENT_SYSTEM_PROMPT = """
You are LLM1, the coordinator of a research workflow.
Your job is to understand the user's research topic and convert it into a clear research brief.
If uploaded document excerpts are included, first infer their role. They may be
writing requirements, grading rubrics, assignment prompts, style constraints,
background notes, or source evidence. Do not treat requirements or rubrics as
factual evidence about the research subject; convert them into scope, audience,
format, and evaluation constraints.
Return only valid JSON with this schema:
{
  "summary": "one concise sentence describing the user's true intent",
  "key_questions": ["3 to 6 concrete research questions"],
  "scope": "what should be included and excluded, including any writing requirements or constraints from uploaded documents"
}
""".strip()


def intent_user_prompt(topic: str) -> str:
    return f"User research topic:\n{topic}"


FALLBACK_RESEARCH_BRANCHES = [
    ResearchPerspective(
        name="Context Researcher",
        focus="background, definitions, history, current landscape, and important concepts",
        questions=[
            "What context and definitions are needed to understand the topic?",
            "How did the current landscape develop?",
            "Which concepts or actors shape the problem space?",
        ],
    ),
    ResearchPerspective(
        name="Evidence Researcher",
        focus="arguments, examples, patterns, tradeoffs, and practical evidence",
        questions=[
            "What evidence or examples best answer the user's key questions?",
            "Which patterns and tradeoffs appear across cases?",
            "What practical implications follow from the available evidence?",
        ],
    ),
    ResearchPerspective(
        name="Critical Researcher",
        focus="risks, limitations, counterarguments, blind spots, and open questions",
        questions=[
            "What claims are uncertain, contested, or under-evidenced?",
            "Which risks or limitations could change the conclusion?",
            "What should be checked before acting on the findings?",
        ],
    ),
]

# Backward-compatible name for older imports.
RESEARCH_BRANCHES = FALLBACK_RESEARCH_BRANCHES


PERSPECTIVE_PLANNER_SYSTEM_PROMPT = """
You are a PerspectivePlanner for a research workbench inspired by STORM-style perspective discovery.
Given the user's topic and interpreted intent, generate 4 to 6 complementary research perspectives.

Return only valid JSON with this schema:
{
  "perspectives": [
    {
      "name": "short role name, e.g. Market Structure Researcher",
      "focus": "one sentence describing the angle this researcher should cover",
      "questions": ["2 to 4 concrete questions this perspective must answer"]
    }
  ]
}

The perspectives should be diverse, non-overlapping, and useful for a final synthesis.
Use the same language as the user's research topic when possible.
Do not include generic labels unless they are genuinely the best fallback.
""".strip()


def perspective_planner_user_prompt(intent: Intent) -> str:
    payload = {
        "original_topic": _clip(intent.original_topic, PROMPT_TOPIC_CHARS),
        "interpreted_intent": intent.summary,
        "key_questions": intent.key_questions,
        "scope": intent.scope,
    }
    return "Research brief:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def branch_system_prompt(
    perspective: ResearchPerspective,
    include_reference_section: bool = True,
) -> str:
    questions = "\n".join(f"- {question}" for question in perspective.questions)
    reference_instruction = """
At the end, add a separate "References" section. This section does not count toward
the 5 to 8 findings bullets. Include up to 3 sources ranked by relevance.
Only list named literature, reports, standards, datasets, institutions, or authors
that your branch actually used or mentioned and that directly match the user's
specific topic. Do not include tangential diagnosis, treatment, animal-model, or
wrong-modality sources in the References section. Use exactly this bullet format:
- Title/source name - Relevance: why it matters - Link/DOI: URL, DOI, or "not provided"
If there are no clear sources, write:
- No clear source - Relevance: this branch did not cite a listable source - Link/DOI: not provided
""".strip()
    if not include_reference_section:
        reference_instruction = (
            "Fast mode: do not add a separate reference section. "
            "When a finding depends on a named paper, report, standard, institution, "
            "author, or year, mention that source inline inside the relevant bullet."
        )

    return f"""
You are {perspective.name}, one parallel LLM researcher in a research workbench.
Your focus: {perspective.focus}.

Answer these perspective questions directly:
{questions}

Produce a compact research memo in Markdown.
Keep the findings concise: 4 to 6 bullets, no long tables, no code blocks, no preamble.
If the research brief includes uploaded documents, respect their inferred role:
requirements/rubrics/style guides constrain the memo, while source-evidence
documents may support claims. Do not cite a requirement document as evidence for
the research subject unless it contains substantive source material.
If the research brief includes related_literature_pool, use it as the primary
evidence context when it is relevant to your assigned perspective, and cite the
paper/source names inline.
Treat related_literature_pool as candidate evidence, not guaranteed evidence.
Only cite a candidate when its title, abstract/relevance, or metadata directly
matches the user's specific topic and modality/method constraints. If a source is
only broadly related, mention the evidence gap instead of citing it as support.
Each finding should, when possible, include:
- claim
- evidence
- implication
- uncertainty
- source
Use the same language as the user's research topic when possible.
Be specific, structured, and avoid unsupported certainty.
Do not synthesize the final answer; only produce your branch's findings.

{reference_instruction}
""".strip()


def branch_user_prompt(
    intent: Intent,
    perspective: ResearchPerspective,
    planned_perspectives: list[ResearchPerspective] | None = None,
    literature_pool: list | None = None,
) -> str:
    payload = {
        "original_topic": _clip(intent.original_topic, PROMPT_TOPIC_CHARS),
        "interpreted_intent": intent.summary,
        "key_questions": intent.key_questions,
        "scope": intent.scope,
        "perspective": {
            "name": perspective.name,
            "focus": perspective.focus,
            "questions": perspective.questions,
        },
        "planned_perspectives": [
            {
                "name": item.name,
                "focus": item.focus,
                "questions": item.questions,
                "is_assigned_perspective": item.name == perspective.name,
            }
            for item in (planned_perspectives or [perspective])
        ],
        "related_literature_pool": [
            {
                "title": item.title,
                "relevance": item.relevance,
                "source": item.source,
                "origin": item.branch_name,
            }
            for item in (literature_pool or [])[:16]
        ],
    }
    return "Research brief and assigned perspective:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )


OUTLINE_PLANNER_SYSTEM_PROMPT = """
You are an outline planner for the final research report.
Given the interpreted intent and branch research memos, design the final report outline.

Return only valid JSON with this schema:
{
  "sections": [
    {
      "title": "section title",
      "purpose": "why this section exists",
      "key_points": ["specific points or claims to cover"],
      "source_hints": ["branch names, source names, or evidence hints to bind to claims"]
    }
  ]
}

Create 4 to 7 sections. The outline must help resolve contradictions, connect sources to
specific claims, and avoid simply concatenating branch memos.
Use the same language as the user's research topic when possible.
""".strip()


def outline_user_prompt(intent: Intent, branches: list[ResearchBranch]) -> str:
    branch_blocks = []
    for branch in branches:
        questions = "\n".join(f"- {question}" for question in branch.questions)
        branch_blocks.append(
            f"## {branch.name}\n"
            f"Focus: {branch.focus}\n"
            f"Questions:\n{questions or '- not provided'}\n\n"
            f"{_clip(branch.content, PROMPT_BRANCH_CHARS)}"
        )

    payload = {
        "original_topic": _clip(intent.original_topic, PROMPT_TOPIC_CHARS),
        "interpreted_intent": intent.summary,
        "key_questions": intent.key_questions,
        "scope": intent.scope,
    }

    return (
        "Intent:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nBranch research memos:\n\n"
        + "\n\n".join(branch_blocks)
    )


SYNTHESIS_SYSTEM_PROMPT = """
You are LLM1 again, now acting as the final synthesizer.
You will receive the original intent, parallel research memos, and possibly a planned outline.

Create the final research output in Markdown.
If an outline is provided, use it to organize the report. If no outline is provided, use this structure:
1. Executive summary
2. Key findings
3. Detailed analysis
4. Risks and limitations
5. Suggested next steps

Resolve contradictions explicitly. Bind sources to specific claims whenever possible.
Do not invent citations, datasets, model scores, authors, years, or paper titles.
Use only sources present in the branch memos or related literature context, and
only when they directly support the claim being made. If evidence is weak or
tangential, say so explicitly rather than padding the report with unrelated citations.
Separate writing constraints from factual evidence when uploaded documents include
requirements, rubrics, or assignment instructions.
Do not simply concatenate the branch outputs.
Keep the report complete and bounded: prefer 900 to 1400 words, use compact paragraphs,
and do not expand every branch finding if it does not change the conclusion.
""".strip()


def synthesis_user_prompt(
    intent: Intent,
    branches: list[ResearchBranch],
    outline: list[ReportOutlineSection] | None = None,
) -> str:
    branch_blocks = []
    for branch in branches:
        questions = "\n".join(f"- {question}" for question in branch.questions)
        branch_blocks.append(
            f"## {branch.name}\n"
            f"Focus: {branch.focus}\n"
            f"Questions:\n{questions or '- not provided'}\n\n"
            f"{_clip(branch.content, PROMPT_SYNTHESIS_BRANCH_CHARS)}"
        )

    payload = {
        "original_topic": _clip(intent.original_topic, PROMPT_TOPIC_CHARS),
        "interpreted_intent": intent.summary,
        "key_questions": intent.key_questions,
        "scope": intent.scope,
    }

    outline_payload = None
    if outline is not None:
        outline_payload = [
            {
                "title": section.title,
                "purpose": section.purpose,
                "key_points": section.key_points,
                "source_hints": section.source_hints,
            }
            for section in outline
        ]

    return (
        "Intent:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nFinal report outline:\n"
        + (
            json.dumps(outline_payload, ensure_ascii=False, indent=2)
            if outline_payload
            else "No outline available; use the default synthesis structure."
        )
        + "\n\nParallel research memos:\n\n"
        + "\n\n".join(branch_blocks)
    )
