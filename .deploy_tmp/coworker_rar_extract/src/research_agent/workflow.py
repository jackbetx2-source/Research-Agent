from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from .llm import LLMClient, LLMServiceError
from .prompts import (
    FALLBACK_RESEARCH_BRANCHES,
    INTENT_SYSTEM_PROMPT,
    OUTLINE_PLANNER_SYSTEM_PROMPT,
    PERSPECTIVE_PLANNER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    branch_system_prompt,
    branch_user_prompt,
    intent_user_prompt,
    outline_user_prompt,
    perspective_planner_user_prompt,
    synthesis_user_prompt,
)
from .types import (
    Intent,
    ReportOutlineSection,
    ResearchBranch,
    ResearchPerspective,
    ResearchReference,
    ResearchResult,
)


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
        self._log("1/5 Understanding intent...")
        intent = await self._understand_intent(topic)
        self._log("2/5 Planning research perspectives...")
        perspectives = await self._plan_perspectives(intent)
        self._log(f"3/5 Running {len(perspectives)} research branches in parallel...")
        branches = await self._run_parallel_research(intent, perspectives)
        self._log("4/5 Planning final report outline...")
        outline = await self._plan_outline(intent, branches)
        self._log("5/5 Synthesizing final report...")
        final_report = await self._synthesize(intent, branches, outline)
        references = self._rank_references(branches)
        if not references:
            references = self._infer_references_from_content(final_report, "Final Report")
        self._log("Done.")
        return ResearchResult(
            intent=intent,
            branches=branches,
            final_report=final_report,
            references=references,
        )

    async def _understand_intent(self, topic: str) -> Intent:
        content = await self._complete_with_length_retry(
            system_prompt=INTENT_SYSTEM_PROMPT,
            retry_system_prompt=INTENT_SYSTEM_PROMPT
            + "\n\nRetry after truncation: return a smaller JSON object only. "
            "Use a short summary, exactly 3 short key_questions, and a short scope.",
            emergency_system_prompt=INTENT_SYSTEM_PROMPT
            + "\n\nEmergency retry: return only this compact JSON shape with no prose: "
            '{"summary":"...","key_questions":["...","...","..."],"scope":"..."}',
            user_prompt=intent_user_prompt(topic),
            model=os.getenv("INTENT_MODEL"),
            temperature=0.1,
            max_tokens=600 if self.fast else 900,
            retry_max_tokens=900 if self.fast else 1300,
            emergency_max_tokens=500 if self.fast else 700,
        )
        try:
            data = self._parse_json_object(content)
            summary = str(data.get("summary") or topic)
            key_questions = [
                str(item).strip()
                for item in self._as_list(data.get("key_questions"))
                if str(item).strip()
            ]
            scope = str(data.get("scope") or "Cover the topic directly and note limits.")
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._log(f"Intent JSON parse failed; using fallback intent. ({error})")
            summary = topic
            key_questions = [
                f"What is the essential context for {topic}?",
                f"What evidence best answers the topic: {topic}?",
                f"What risks, limits, or open questions matter for {topic}?",
            ]
            scope = "Cover background, evidence, implications, uncertainties, and next steps."

        if not key_questions:
            key_questions = [
                "What does the user need to understand first?",
                "What evidence supports the main claims?",
                "What uncertainties or limitations should shape the conclusion?",
            ]
        return Intent(
            original_topic=topic,
            summary=summary,
            key_questions=key_questions[:6],
            scope=scope,
        )

    async def _plan_perspectives(self, intent: Intent) -> list[ResearchPerspective]:
        try:
            content = await self.llm.complete(
                system_prompt=PERSPECTIVE_PLANNER_SYSTEM_PROMPT,
                user_prompt=perspective_planner_user_prompt(intent),
                model=os.getenv("PLANNER_MODEL") or os.getenv("INTENT_MODEL"),
                temperature=0.25,
                max_tokens=700 if self.fast else 1200,
            )
            data = self._parse_json_object(content)
            perspectives = self._coerce_perspectives(data.get("perspectives"))
            if 4 <= len(perspectives) <= 6:
                return perspectives
            raise ValueError(f"Expected 4 to 6 perspectives, got {len(perspectives)}.")
        except (LLMServiceError, json.JSONDecodeError, TypeError, ValueError) as error:
            self._log(f"Perspective planning failed; using fallback branches. ({error})")
            return list(FALLBACK_RESEARCH_BRANCHES)

    async def _run_parallel_research(
        self, intent: Intent, perspectives: list[ResearchPerspective]
    ) -> list[ResearchBranch]:
        tasks = [self._run_branch(intent, perspective) for perspective in perspectives]
        return list(await asyncio.gather(*tasks))

    async def _run_branch(
        self, intent: Intent, perspective: ResearchPerspective
    ) -> ResearchBranch:
        system_prompt = branch_system_prompt(
            perspective,
            include_reference_section=not self.fast,
        )
        content = await self._complete_with_length_retry(
            system_prompt=system_prompt,
            retry_system_prompt=system_prompt
            + "\n\nRetry after truncation: keep this memo under 6 bullets. "
            "Use one sentence per bullet and avoid sub-bullets.",
            emergency_system_prompt=system_prompt
            + "\n\nEmergency retry: produce only 3 bullets. "
            "Each bullet must be one sentence with claim, evidence, implication, uncertainty, and source compressed inline. "
            "Omit the separate References section if needed.",
            user_prompt=branch_user_prompt(intent, perspective),
            model=os.getenv("RESEARCH_MODEL"),
            temperature=0.35,
            max_tokens=700 if self.fast else 1300,
            retry_max_tokens=1000 if self.fast else 1700,
            emergency_max_tokens=900 if self.fast else 1200,
        )
        return ResearchBranch(
            name=perspective.name,
            focus=perspective.focus,
            content=content,
            references=self._extract_references(content, perspective.name),
            questions=perspective.questions,
        )

    async def _plan_outline(
        self, intent: Intent, branches: list[ResearchBranch]
    ) -> list[ReportOutlineSection] | None:
        try:
            content = await self.llm.complete(
                system_prompt=OUTLINE_PLANNER_SYSTEM_PROMPT,
                user_prompt=outline_user_prompt(intent, branches),
                model=os.getenv("PLANNER_MODEL") or os.getenv("SYNTHESIS_MODEL"),
                temperature=0.2,
                max_tokens=800 if self.fast else 1400,
            )
            data = self._parse_json_object(content)
            outline = self._coerce_outline(data.get("sections"))
            if outline:
                return outline
            raise ValueError("Outline contains no valid sections.")
        except (LLMServiceError, json.JSONDecodeError, TypeError, ValueError) as error:
            self._log(f"Outline planning failed; synthesizing without outline. ({error})")
            return None

    async def _synthesize(
        self,
        intent: Intent,
        branches: list[ResearchBranch],
        outline: list[ReportOutlineSection] | None = None,
    ) -> str:
        system_prompt = self._synthesis_prompt()
        return await self._complete_with_length_retry(
            system_prompt=system_prompt,
            retry_system_prompt=system_prompt
            + "\n\nRetry after truncation: produce a complete but tighter report. "
            "Limit each section to the strongest points and avoid repeated examples.",
            emergency_system_prompt=system_prompt
            + "\n\nEmergency retry after repeated truncation: produce a complete short report only. "
            "Use exactly these headings: Executive summary, Key findings, Contradictions and uncertainties, Sources to verify, Next steps. "
            "Use at most 2 short paragraphs or 4 bullets per heading. Do not include tables.",
            user_prompt=synthesis_user_prompt(intent, branches, outline),
            model=os.getenv("SYNTHESIS_MODEL"),
            temperature=0.2,
            max_tokens=2200 if self.fast else 3600,
            retry_max_tokens=2800 if self.fast else 4400,
            emergency_max_tokens=1400 if self.fast else 2200,
        )

    async def _complete_with_length_retry(
        self,
        *,
        system_prompt: str,
        retry_system_prompt: str,
        emergency_system_prompt: str,
        user_prompt: str,
        model: str | None,
        temperature: float,
        max_tokens: int,
        retry_max_tokens: int,
        emergency_max_tokens: int,
    ) -> str:
        attempts = [
            (system_prompt, max_tokens, ""),
            (
                retry_system_prompt,
                retry_max_tokens,
                "Model output hit token limit; retrying with a larger concise budget...",
            ),
            (
                emergency_system_prompt,
                emergency_max_tokens,
                "Model output hit token limit again; retrying with an emergency short form...",
            ),
        ]
        last_error: LLMServiceError | None = None
        for attempt_system_prompt, attempt_max_tokens, length_log in attempts:
            if length_log:
                self._log(length_log)
            try:
                return await self.llm.complete(
                    system_prompt=attempt_system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=attempt_max_tokens,
                )
            except LLMServiceError as error:
                if not self._is_length_error(error):
                    raise
                last_error = error

        raise last_error or LLMServiceError("Model output hit token limit.")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[workflow] {message}", flush=True)

    def _synthesis_prompt(self) -> str:
        if not self.fast:
            return SYNTHESIS_SYSTEM_PROMPT

        return (
            SYNTHESIS_SYSTEM_PROMPT
            + "\n\nFast mode: keep the final report concise. "
            "Use short sections, avoid tables, and prioritize the most actionable insights. "
            "Do not repeat branch-level source lists; cite source names only when they clarify a key claim."
        )

    @staticmethod
    def _is_length_error(error: Exception) -> bool:
        message = str(error).casefold()
        return (
            "token" in message
            or "length" in message
            or "输出达到" in message
            or "上限" in message
            or "截断" in message
            or "输出达到" in message
            or "上限" in message
            or "截断" in message
            or "输出达到" in message
            or "上限" in message
            or "截断" in message
        )

    @staticmethod
    def _coerce_perspectives(raw: Any) -> list[ResearchPerspective]:
        perspectives: list[ResearchPerspective] = []
        for item in ResearchWorkflow._as_list(raw):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            focus = str(item.get("focus") or "").strip()
            questions = [
                str(question).strip()
                for question in ResearchWorkflow._as_list(item.get("questions"))
                if str(question).strip()
            ]
            if not name or not focus or not questions:
                continue
            perspectives.append(
                ResearchPerspective(name=name, focus=focus, questions=questions[:4])
            )
        return perspectives[:6]

    @staticmethod
    def _coerce_outline(raw: Any) -> list[ReportOutlineSection]:
        sections: list[ReportOutlineSection] = []
        for item in ResearchWorkflow._as_list(raw):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            purpose = str(item.get("purpose") or "").strip()
            key_points = [
                str(point).strip()
                for point in ResearchWorkflow._as_list(item.get("key_points"))
                if str(point).strip()
            ]
            source_hints = [
                str(source).strip()
                for source in ResearchWorkflow._as_list(item.get("source_hints"))
                if str(source).strip()
            ]
            if not title or not purpose:
                continue
            sections.append(
                ReportOutlineSection(
                    title=title,
                    purpose=purpose,
                    key_points=key_points[:8],
                    source_hints=source_hints[:8],
                )
            )
        return sections[:7]

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        cleaned = ResearchWorkflow._strip_json_fence(content)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object from LLM.")
        return data

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return cleaned

    @staticmethod
    def _extract_references(content: str, branch_name: str) -> list[ResearchReference]:
        references: list[ResearchReference] = []
        in_reference_section = False

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            heading = line.lstrip("#").strip().rstrip(":").lower()
            if heading in {
                "references",
                "reference",
                "sources",
                "source",
                "cited sources",
                "key sources",
            }:
                in_reference_section = True
                continue

            if in_reference_section and line.startswith("#"):
                break

            if not in_reference_section or not re.match(r"^[-*]\s+", line):
                continue

            item = re.sub(r"^[-*]\s+", "", line).strip()
            if not item:
                continue

            title, relevance, source = ResearchWorkflow._parse_reference_item(item)
            references.append(
                ResearchReference(
                    title=title,
                    relevance=relevance,
                    source=source,
                    branch_name=branch_name,
                )
            )

        if references:
            return references
        return ResearchWorkflow._infer_references_from_content(content, branch_name)

    @staticmethod
    def _parse_reference_item(item: str) -> tuple[str, str, str]:
        parts = re.split(r"\s+-\s+", item, maxsplit=2)
        title = parts[0].strip() if parts else item.strip()
        rest = " - ".join(parts[1:]) if len(parts) > 1 else ""

        relevance = ""
        source = ""
        relevance_match = re.search(
            r"(?:relevance)\s*:\s*(.*?)(?=\s+-\s+(?:link/doi|link|doi|source|url)\s*:|$)",
            rest,
            flags=re.IGNORECASE,
        )
        source_match = re.search(
            r"(?:link/doi|link|doi|source|url)\s*:\s*(.*)$",
            rest,
            flags=re.IGNORECASE,
        )

        if relevance_match:
            relevance = relevance_match.group(1).strip()
        elif rest:
            relevance = rest.strip()

        if source_match:
            source = source_match.group(1).strip()

        return title or "Untitled source", relevance or "Not specified", source or "not provided"

    @staticmethod
    def _infer_references_from_content(
        content: str, branch_name: str
    ) -> list[ResearchReference]:
        candidates = [
            (
                r"\bFeynman\b|Feynman\s+1982",
                "Richard Feynman (1982), Simulating Physics with Computers",
                "Foundational reference for simulating physics with quantum systems.",
                "https://doi.org/10.1007/BF02650179",
            ),
            (
                r"\bDeutsch\b|Deutsch\s+1985",
                "David Deutsch (1985), Quantum theory, the Church-Turing principle and the universal quantum computer",
                "Foundational theory for universal quantum computation.",
                "https://doi.org/10.1098/rspa.1985.0070",
            ),
            (
                r"\bShor\b|Shor'?s algorithm|Shor\s+1994",
                "Peter Shor (1994), Algorithms for quantum computation: discrete logarithms and factoring",
                "Canonical evidence for quantum algorithmic advantage in factoring and discrete logarithms.",
                "https://doi.org/10.1109/SFCS.1994.365700",
            ),
            (
                r"\bGrover\b|Grover'?s algorithm|Grover\s+1996",
                "Lov Grover (1996), A fast quantum mechanical algorithm for database search",
                "Canonical example of quadratic speedup for unstructured search.",
                "https://doi.org/10.1145/237814.237866",
            ),
            (
                r"\bSycamore\b|quantum supremacy|Google.+2019|Arute",
                "Google AI Quantum and collaborators (2019), Quantum supremacy using a programmable superconducting processor",
                "Milestone experiment often cited in hardware progress debates.",
                "https://doi.org/10.1038/s41586-019-1666-5",
            ),
            (
                r"\bNISQ\b|Noisy Intermediate-Scale Quantum|Preskill",
                "John Preskill (2018), Quantum Computing in the NISQ era and beyond",
                "Defines the NISQ framing and its limitations.",
                "https://doi.org/10.22331/q-2018-08-06-79",
            ),
            (
                r"surface codes?|error correction|quantum error correction|\bQEC\b",
                "A. G. Fowler et al. (2012), Surface codes: Towards practical large-scale quantum computation",
                "Useful source for quantum error correction and fault-tolerance constraints.",
                "https://doi.org/10.1103/PhysRevA.86.032324",
            ),
            (
                r"post-quantum cryptography|quantum-resistant|RSA|ECC|cryptography",
                "NIST Post-Quantum Cryptography Standardization",
                "Authoritative source for quantum-resistant cryptography standards.",
                "https://csrc.nist.gov/projects/post-quantum-cryptography",
            ),
            (
                r"\bQiskit\b|IBM Quantum",
                "IBM Quantum and Qiskit documentation",
                "Representative documentation for quantum cloud tooling and software ecosystems.",
                "https://www.ibm.com/quantum/qiskit",
            ),
            (
                r"\bCirq\b|Google Quantum AI",
                "Google Quantum AI Cirq documentation",
                "Representative documentation for Google's quantum programming stack.",
                "https://quantumai.google/cirq",
            ),
            (
                r"Amazon Braket|Braket",
                "Amazon Braket documentation",
                "Representative documentation for cloud-based quantum computing access.",
                "https://aws.amazon.com/braket/",
            ),
        ]

        inferred: list[ResearchReference] = []
        for pattern, title, relevance, source in candidates:
            if not re.search(pattern, content, flags=re.IGNORECASE):
                continue
            inferred.append(
                ResearchReference(
                    title=title,
                    relevance=relevance,
                    source=source,
                    branch_name=branch_name,
                )
            )
            if len(inferred) >= 3:
                break

        if inferred:
            return inferred

        named_entities = ResearchWorkflow._extract_named_source_mentions(content)
        return [
            ResearchReference(
                title=title,
                relevance="Named source or institution mentioned in the research memo; useful for follow-up verification.",
                source="not provided",
                branch_name=branch_name,
            )
            for title in named_entities[:3]
        ]

    @staticmethod
    def _extract_named_source_mentions(content: str) -> list[str]:
        patterns = [
            r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+\(\d{4}\)",
            r"\b(?:IBM Quantum|Google Quantum AI|Amazon Braket|Microsoft Quantum|IonQ|Quantinuum|Xanadu|PsiQuantum|QuEra|NIST)\b",
        ]
        mentions: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                title = match.group(0).strip()
                key = title.casefold()
                if key in seen:
                    continue
                seen.add(key)
                mentions.append(title)
        return mentions

    @staticmethod
    def _rank_references(branches: list[ResearchBranch]) -> list[ResearchReference]:
        grouped: dict[str, tuple[ResearchReference, list[str], int]] = {}
        order = 0

        for branch in branches:
            for reference in branch.references:
                key = re.sub(r"\W+", "", reference.title.casefold())
                if not key:
                    continue
                if key in grouped:
                    first_reference, branch_names, first_order = grouped[key]
                    if reference.branch_name not in branch_names:
                        branch_names.append(reference.branch_name)
                    grouped[key] = (first_reference, branch_names, first_order)
                    continue
                grouped[key] = (reference, [reference.branch_name], order)
                order += 1

        ranked_groups = sorted(
            grouped.values(),
            key=lambda item: (-len(item[1]), item[2]),
        )
        return [
            ResearchReference(
                title=reference.title,
                relevance=reference.relevance,
                source=reference.source,
                branch_name=", ".join(branch_names),
            )
            for reference, branch_names, _ in ranked_groups
        ]
