from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from .arxiv_search import ArxivSearchError, search_arxiv
from .citations import format_apa_reference_items
from .crossref_search import CrossrefSearchError, search_crossref
from .doi import enrich_references_with_doi_metadata
from .llm import LLMClient, LLMServiceError
from .openalex_search import OpenAlexSearchError, search_openalex
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
from .pubmed_search import PubMedSearchError, search_pubmed
from .types import (
    Intent,
    ReportOutlineSection,
    ResearchBranch,
    ResearchPerspective,
    ResearchReference,
    ResearchResult,
)


class ResearchWorkflow:
    LITERATURE_POOL_LIMIT = 16
    LITERATURE_QUERY_LIMIT = 2
    LITERATURE_PER_SOURCE_LIMIT = 5

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
        self._log("1/6 Understanding intent...")
        intent = await self._understand_intent(topic)
        self._log("2/6 Planning research perspectives...")
        perspectives = await self._plan_perspectives(intent)
        self._log("3/6 Searching related literature...")
        literature_pool = self._search_related_literature(intent, perspectives)
        self._log(f"4/6 Running {len(perspectives)} research branches in parallel...")
        branches = await self._run_parallel_research(intent, perspectives, literature_pool)
        self._log("5/6 Planning final report outline...")
        outline = await self._plan_outline(intent, branches)
        self._log("6/6 Synthesizing final report...")
        final_report = await self._synthesize(intent, branches, outline)
        references = self._rank_references(branches, literature_pool)
        final_report = self.append_apa_references(final_report, references)
        self._log("Done.")
        return ResearchResult(
            intent=intent,
            branches=branches,
            final_report=final_report,
            references=references,
        )

    @staticmethod
    def append_apa_references(report: str, references: list) -> str:
        cleaned = re.sub(
            r"\n{2,}#{1,3}\s*(?:references|reference list|works cited|参考文献|参考资料)\s*\n.*\Z",
            "",
            str(report or "").strip(),
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        enriched_references = enrich_references_with_doi_metadata(
            [ResearchWorkflow._reference_to_metadata_dict(reference) for reference in references]
        )
        apa_references = format_apa_reference_items(enriched_references)
        if not apa_references:
            return cleaned
        return f"{cleaned}\n\n## References\n\n" + "\n\n".join(apa_references)

    @staticmethod
    def _reference_to_metadata_dict(reference) -> dict:
        if isinstance(reference, dict):
            return dict(reference)
        return {
            "title": getattr(reference, "title", ""),
            "relevance": getattr(reference, "relevance", ""),
            "source": getattr(reference, "source", ""),
            "branch_name": getattr(reference, "branch_name", ""),
            "authors": getattr(reference, "authors", []),
            "year": getattr(reference, "year", ""),
            "journal": getattr(reference, "journal", ""),
            "doi": getattr(reference, "doi", ""),
            "pmid": getattr(reference, "pmid", ""),
        }

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
        self,
        intent: Intent,
        perspectives: list[ResearchPerspective],
        literature_pool: list[ResearchReference],
    ) -> list[ResearchBranch]:
        tasks = [
            self._run_branch(intent, perspective, perspectives, literature_pool)
            for perspective in perspectives
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_branch(
        self,
        intent: Intent,
        perspective: ResearchPerspective,
        planned_perspectives: list[ResearchPerspective],
        literature_pool: list[ResearchReference],
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
            user_prompt=branch_user_prompt(
                intent,
                perspective,
                planned_perspectives,
                literature_pool,
            ),
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

    def _search_related_literature(
        self,
        intent: Intent,
        perspectives: list[ResearchPerspective],
    ) -> list[ResearchReference]:
        queries = self._literature_queries(intent, perspectives)
        if not queries:
            return []

        papers: list[dict] = []
        for query in queries[: self.LITERATURE_QUERY_LIMIT]:
            source_calls = [
                (
                    "PubMed",
                    lambda query=query: search_pubmed(
                        query,
                        max_results=self.LITERATURE_PER_SOURCE_LIMIT,
                    ),
                ),
                (
                    "OpenAlex",
                    lambda query=query: search_openalex(
                        query,
                        max_results=self.LITERATURE_PER_SOURCE_LIMIT,
                    ),
                ),
                (
                    "Crossref",
                    lambda query=query: search_crossref(
                        query,
                        max_results=self.LITERATURE_PER_SOURCE_LIMIT,
                    ),
                ),
                (
                    "arXiv",
                    lambda query=query: [
                        paper.to_dict()
                        for paper in search_arxiv(
                            query,
                            max_results=self.LITERATURE_PER_SOURCE_LIMIT,
                            sort_by="relevance",
                        )
                    ],
                ),
            ]
            for source_name, call in source_calls:
                try:
                    for paper in call():
                        if isinstance(paper, dict):
                            papers.append(
                                {
                                    **paper,
                                    "search_query": query,
                                    "search_source": source_name,
                                }
                            )
                except (
                    PubMedSearchError,
                    OpenAlexSearchError,
                    CrossrefSearchError,
                    ArxivSearchError,
                    OSError,
                ) as error:
                    self._log(f"{source_name} literature search unavailable: {error}")
                    continue

        ranked = self._dedupe_and_rank_papers(
            papers,
            intent,
            max_results=self.LITERATURE_POOL_LIMIT,
        )
        return [self._paper_to_reference(paper) for paper in ranked]

    @staticmethod
    def _literature_queries(
        intent: Intent,
        perspectives: list[ResearchPerspective],
    ) -> list[str]:
        focused = ResearchWorkflow._focused_literature_query(intent.original_topic)
        pieces = [
            intent.original_topic,
            intent.summary,
            " ".join(intent.key_questions[:2]),
        ]
        base = ResearchWorkflow._clean_query(" ".join(part for part in pieces if part))
        queries = []
        if focused:
            queries.append(focused)
        if base:
            queries.append(base)

        translated = ResearchWorkflow._english_keyword_query(intent.original_topic)
        if translated and translated.casefold() != base.casefold():
            queries.insert(0, translated)

        compact: list[str] = []
        seen: set[str] = set()
        for query in queries:
            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            compact.append(query)
        return compact[: ResearchWorkflow.LITERATURE_QUERY_LIMIT]

    @staticmethod
    def _focused_literature_query(text: str) -> str:
        terms: list[str] = []

        def add(*items: str) -> None:
            seen = {term.casefold() for term in terms}
            for item in items:
                if item.casefold() not in seen:
                    terms.append(item)
                    seen.add(item.casefold())

        if re.search(r"ct|computed tomography|头颅|颅脑|脑|head", text, flags=re.IGNORECASE):
            add("non-contrast CT", "brain")
        if re.search(r"脑梗死|缺血性卒中|卒中|ischemic|stroke|infarct", text, flags=re.IGNORECASE):
            add("ischemic stroke")
        if re.search(r"病灶|lesion", text, flags=re.IGNORECASE):
            add("lesion")
        if re.search(r"分割|segmentation|segment", text, flags=re.IGNORECASE):
            add("segmentation")
        return ResearchWorkflow._clean_query(" ".join(terms))

    @staticmethod
    def _english_keyword_query(text: str) -> str:
        mapping = [
            ("头颅", "head"),
            ("颅脑", "brain"),
            ("脑", "brain"),
            ("CT", "CT"),
            ("ct", "CT"),
            ("脑梗死", "ischemic stroke cerebral infarction"),
            ("梗死", "infarction"),
            ("卒中", "stroke"),
            ("病灶", "lesion"),
            ("分割", "segmentation"),
            ("医学影像", "medical imaging"),
            ("影像", "imaging"),
            ("深度学习", "deep learning"),
            ("模型", "model"),
        ]
        terms = []
        seen: set[str] = set()
        for chinese, english in mapping:
            if chinese not in text:
                continue
            for term in english.split():
                key = term.casefold()
                if key in seen:
                    continue
                seen.add(key)
                terms.append(term)
        return ResearchWorkflow._clean_query(" ".join(terms))

    @staticmethod
    def _clean_query(value: str) -> str:
        text = re.sub(r"https?://\S+", " ", value or "")
        text = re.sub(r"[^\w\u4e00-\u9fff\s-]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return " ".join(text.split()[:18])

    @staticmethod
    def _dedupe_and_rank_papers(
        papers: list[dict],
        intent: Intent,
        *,
        max_results: int,
    ) -> list[dict]:
        deduped: dict[str, dict] = {}
        order: dict[str, int] = {}
        title_alias: dict[str, str] = {}
        for index, paper in enumerate(papers):
            key = ResearchWorkflow._paper_key(paper)
            if not key:
                continue
            title_key = ResearchWorkflow._paper_title_key(paper)
            if title_key and title_key in title_alias:
                key = title_alias[title_key]
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = paper
                order[key] = index
                if title_key:
                    title_alias[title_key] = key
                continue
            if ResearchWorkflow._paper_quality_score(paper) > ResearchWorkflow._paper_quality_score(existing):
                merged = dict(existing)
                merged.update({field: value for field, value in paper.items() if value})
                deduped[key] = merged
                if title_key:
                    title_alias[title_key] = key

        topic_terms = ResearchWorkflow._keyword_set(
            " ".join([intent.original_topic, intent.summary, " ".join(intent.key_questions)])
        )
        required_concepts = ResearchWorkflow._required_concept_patterns(intent)
        filtered = [
            paper
            for paper in deduped.values()
            if ResearchWorkflow._paper_matches_required_concepts(paper, required_concepts)
        ]
        if len(filtered) < min(3, max_results):
            filtered = [
                paper
                for paper in deduped.values()
                if ResearchWorkflow._paper_relevance_score(paper, topic_terms) >= 8.0
            ]
        ranked = sorted(
            filtered,
            key=lambda paper: (
                -ResearchWorkflow._paper_relevance_score(paper, topic_terms),
                order.get(ResearchWorkflow._paper_key(paper), 999999),
            ),
        )
        return ranked[:max_results]

    @staticmethod
    def _paper_key(paper: dict) -> str:
        doi = str(paper.get("doi") or "").lower().strip()
        if doi:
            return f"doi:{doi}"
        pmid = str(paper.get("pmid") or "").lower().strip()
        if pmid:
            return f"pmid:{pmid}"
        url = str(paper.get("abs_url") or "").lower().strip().rstrip("/")
        if url:
            return f"url:{url}"
        title = re.sub(r"\W+", "", str(paper.get("title", "")).casefold())
        return f"title:{title}" if title else ""

    @staticmethod
    def _paper_title_key(paper: dict) -> str:
        title = re.sub(r"\W+", "", str(paper.get("title", "")).casefold())
        return f"title:{title}" if title else ""

    @staticmethod
    def _paper_quality_score(paper: dict) -> int:
        return sum(
            1
            for key in ["abstract", "doi", "pmid", "authors", "journal", "published", "abs_url"]
            if paper.get(key)
        )

    @staticmethod
    def _paper_relevance_score(paper: dict, topic_terms: set[str]) -> float:
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")
        source = str(paper.get("source") or paper.get("search_source") or "")
        title_terms = ResearchWorkflow._keyword_set(title)
        abstract_terms = ResearchWorkflow._keyword_set(abstract)
        source_priority = {"PubMed": 3.0, "OpenAlex": 2.0, "Crossref": 1.5, "arXiv": 1.0}
        score = 5.0 * len(topic_terms & title_terms)
        score += 2.0 * len(topic_terms & abstract_terms)
        score += ResearchWorkflow._paper_quality_score(paper) * 0.4
        score += source_priority.get(source, 0.5)
        if paper.get("doi") or paper.get("pmid"):
            score += 1.0
        return score

    @staticmethod
    def _required_concept_patterns(intent: Intent) -> list[list[str]]:
        topic = " ".join([intent.original_topic, intent.summary, " ".join(intent.key_questions)])
        groups: list[list[str]] = []
        if re.search(r"ct|computed tomography|头颅|颅脑", topic, flags=re.IGNORECASE):
            groups.append([r"\bct\b", r"computed tomography", r"\bncct\b", r"non[-\s]?contrast ct", r"头颅CT", r"颅脑CT"])
        if re.search(r"脑梗死|缺血性卒中|卒中|ischemic|stroke|infarct", topic, flags=re.IGNORECASE):
            groups.append([r"ischemi[ac]", r"\bstroke\b", r"infarct", r"脑梗死", r"缺血性卒中", r"卒中"])
        if re.search(r"病灶|lesion", topic, flags=re.IGNORECASE):
            groups.append([r"\blesions?\b", r"病灶", r"infarct"])
        if re.search(r"分割|segmentation|segment", topic, flags=re.IGNORECASE):
            groups.append([r"segment", r"segmentation", r"segmented", r"delineat", r"分割"])
        return groups

    @staticmethod
    def _paper_matches_required_concepts(paper: dict, concept_groups: list[list[str]]) -> bool:
        if not concept_groups:
            return True
        text = " ".join(
            str(paper.get(key) or "")
            for key in ["title", "abstract", "journal", "categories"]
        )
        return all(
            any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in group)
            for group in concept_groups
        )

    @staticmethod
    def _paper_to_reference(paper: dict) -> ResearchReference:
        doi = str(paper.get("doi") or "").strip()
        pmid = str(paper.get("pmid") or "").strip()
        url = (
            str(paper.get("abs_url") or "").strip()
            or (f"https://doi.org/{doi}" if doi else "")
            or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "")
            or str(paper.get("pdf_url") or "").strip()
            or "not provided"
        )
        source = str(paper.get("source") or paper.get("search_source") or "Literature search")
        details = ", ".join(
            part
            for part in [
                str(paper.get("journal") or "").strip(),
                str(paper.get("published") or "").strip(),
            ]
            if part
        )
        relevance = f"Retrieved by {source} for the research topic."
        if details:
            relevance += f" {details}."
        abstract = str(paper.get("abstract") or "").strip()
        if abstract:
            relevance += f" {abstract[:240]}"
        return ResearchReference(
            title=str(paper.get("title") or "Untitled literature").strip(),
            relevance=relevance,
            source=url,
            branch_name="Literature search",
            authors=paper.get("authors", []),
            year=str(paper.get("published") or paper.get("year") or "").strip(),
            journal=str(paper.get("journal") or "").strip(),
            doi=doi,
            pmid=pmid,
        )

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
                "参考文献",
                "参考资料",
                "主要参考文献",
                "主要来源",
                "来源",
                "资料来源",
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
    def _rank_references(
        branches: list[ResearchBranch],
        literature_pool: list[ResearchReference] | None = None,
    ) -> list[ResearchReference]:
        grouped: dict[str, tuple[ResearchReference, list[str], int]] = {}
        order = 0

        for reference in literature_pool or []:
            key = ResearchWorkflow._reference_key(reference)
            if not key:
                continue
            grouped[key] = (reference, [reference.branch_name], order)
            order += 1

        for branch in branches:
            for reference in branch.references:
                key = ResearchWorkflow._reference_key(reference)
                if not key:
                    continue
                if reference.title.strip().lower() in {"no clear source", "untitled source"}:
                    continue
                if not str(reference.source or "").strip() or str(reference.source or "").strip().casefold() in {"not provided", "未提供"}:
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
                authors=reference.authors,
                year=reference.year,
                journal=reference.journal,
                doi=reference.doi,
                pmid=reference.pmid,
            )
            for reference, branch_names, _ in ranked_groups
        ]

    @staticmethod
    def _reference_key(reference: ResearchReference) -> str:
        source = str(reference.source or "").strip().lower().rstrip("/")
        doi_match = re.search(r"(10\.\d{4,9}/\S+)", source)
        if doi_match:
            return f"doi:{doi_match.group(1).rstrip('.,;')}"
        if source and source != "not provided":
            return f"url:{source}"
        title = re.sub(r"\W+", "", str(reference.title or "").casefold())
        return f"title:{title}" if title else ""

    @staticmethod
    def _keyword_set(value: str) -> set[str]:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", (value or "").casefold())
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "研究",
            "报告",
            "文献",
            "相关",
        }
        return {token for token in tokens if len(token) >= 2 and token not in stopwords}
