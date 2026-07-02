from __future__ import annotations

import asyncio
import json
import os

from .llm import LLMClient
from .prompts import (
    INTENT_SYSTEM_PROMPT,
    RESEARCH_BRANCHES,
    SYNTHESIS_SYSTEM_PROMPT,
    branch_system_prompt,
    branch_user_prompt,
    intent_user_prompt,
    synthesis_user_prompt,
)
from .types import Intent, ResearchBranch, ResearchResult


class ResearchWorkflow:
    def __init__(
        self,
        llm: LLMClient | None = None,
        verbose: bool = False,
        fast: bool = False,
    ) -> None:
        self.llm = llm or LLMClient()
        self.verbose = verbose
        self.fast = fast

    async def run(self, topic: str) -> ResearchResult:
        self._log("1/3 Understanding intent...")
        intent = await self._understand_intent(topic)
        self._log("2/3 Running three research branches in parallel...")
        branches = await self._run_parallel_research(intent)
        self._log("3/3 Synthesizing final report...")
        final_report = await self._synthesize(intent, branches)
        self._log("Done.")
        return ResearchResult(intent=intent, branches=branches, final_report=final_report)

    async def _understand_intent(self, topic: str) -> Intent:
        content = await self.llm.complete(
            system_prompt=INTENT_SYSTEM_PROMPT,
            user_prompt=intent_user_prompt(topic),
            model=os.getenv("INTENT_MODEL"),
            temperature=0.1,
            max_tokens=600 if self.fast else 900,
        )
        data = self._parse_json_object(content)
        return Intent(
            original_topic=topic,
            summary=str(data["summary"]),
            key_questions=[str(item) for item in data["key_questions"]],
            scope=str(data["scope"]),
        )

    async def _run_parallel_research(self, intent: Intent) -> list[ResearchBranch]:
        tasks = [self._run_branch(intent, branch) for branch in RESEARCH_BRANCHES]
        return list(await asyncio.gather(*tasks))

    async def _run_branch(self, intent: Intent, branch: dict[str, str]) -> ResearchBranch:
        content = await self.llm.complete(
            system_prompt=branch_system_prompt(branch["name"], branch["focus"]),
            user_prompt=branch_user_prompt(intent),
            model=os.getenv("RESEARCH_MODEL"),
            temperature=0.35,
            max_tokens=700 if self.fast else 1200,
        )
        return ResearchBranch(
            name=branch["name"],
            focus=branch["focus"],
            content=content,
        )

    async def _synthesize(
        self, intent: Intent, branches: list[ResearchBranch]
    ) -> str:
        return await self.llm.complete(
            system_prompt=self._synthesis_prompt(),
            user_prompt=synthesis_user_prompt(intent, branches),
            model=os.getenv("SYNTHESIS_MODEL"),
            temperature=0.2,
            max_tokens=2200 if self.fast else 4000,
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[workflow] {message}", flush=True)

    def _synthesis_prompt(self) -> str:
        if not self.fast:
            return SYNTHESIS_SYSTEM_PROMPT

        return (
            SYNTHESIS_SYSTEM_PROMPT
            + "\n\nFast mode: keep the final report concise. "
            "Use short sections, avoid tables, and prioritize the most actionable insights."
        )

    @staticmethod
    def _parse_json_object(content: str) -> dict:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object from intent LLM.")
        return data
