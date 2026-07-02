from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime

from .arxiv_search import ArxivSearchError, extract_core_query, search_arxiv
from .citations import format_references, normalize_citation_format, year_from_paper
from .crossref_search import search_crossref
from .language_policy import contains_cjk, main_language, paper_language
from .llm import LLMClient, LLMServiceError
from .openalex_search import search_openalex
from .pubmed_search import search_pubmed


CONTEXT_DOCUMENT_EXCERPT_CHARS = 4000


@dataclass(frozen=True)
class SLRResult:
    topic: str
    search_query: str
    citation_format: str
    papers: list[dict]
    source_counts: dict[str, int]
    source_errors: dict[str, str]
    annotations: list[dict]
    synthesis: dict
    references: list[str]
    report_markdown: str


class SLRWorkflow:
    def __init__(self, llm: LLMClient | None = None, verbose: bool = False) -> None:
        self.llm = llm or LLMClient()
        self.verbose = verbose

    async def run(
        self,
        *,
        topic: str,
        max_results: int = 20,
        category: str = "",
        start_date: str = "",
        end_date: str = "",
        citation_format: str = "APA",
        user_papers: list[dict] | None = None,
        source_mode: str = "search",
    ) -> SLRResult:
        max_results = max(1, min(int(max_results or 20), 50))
        citation_format = normalize_citation_format(citation_format)
        output_language = main_language(topic, default="en")
        search_query = extract_core_query(topic)
        search_plan = await self._build_search_plan(topic, search_query)
        source_mode = normalize_source_mode(source_mode)
        normalized_user_items = [
            self._normalize_user_paper(item)
            for item in (user_papers or [])
            if isinstance(item, dict)
        ]
        user_context_documents = [
            item for item in normalized_user_items if not self._is_literature_user_paper(item)
        ]
        user_papers = [
            item for item in normalized_user_items if self._is_literature_user_paper(item)
        ]
        if user_context_documents:
            topic = self._append_user_context_documents(topic, user_context_documents)
            if source_mode == "upload" and not user_papers:
                source_mode = "search"

        self._log("1/4 Preparing literature sources...")
        source_counts: dict[str, int] = {}
        source_errors: dict[str, str] = {}
        external_papers: list[dict] = []
        if user_papers:
            source_counts["User provided"] = len(user_papers)

        if source_mode in {"search", "mixed"}:
            external_limit = max_results
            if source_mode == "mixed":
                external_limit = max(0, max_results - len(user_papers))
            if external_limit > 0:
                external_papers, search_counts, source_errors = self._search_sources(
                    topic=topic,
                    search_plan=search_plan,
                    max_results=external_limit,
                    category=category,
                    start_date=start_date,
                    end_date=end_date,
                )
                source_counts.update(search_counts)

        if source_mode == "upload":
            papers = self._dedupe_and_rank(user_papers, max_results=max(len(user_papers), 1))
        elif source_mode == "mixed":
            papers = self._dedupe_and_rank(user_papers + external_papers, max_results=max_results)
        else:
            papers = self._dedupe_and_rank(external_papers, max_results=max_results)

        if not papers:
            raise ArxivSearchError("No papers were provided or returned. Try uploading files, adding links, or broadening the search.")
        search_source = self._format_search_source(source_counts, source_errors, source_mode)

        self._log("2/4 Extracting per-paper annotations...")
        annotations = await self._extract_annotations(papers, output_language=output_language)

        self._log("3/4 Synthesizing cross-paper themes...")
        synthesis = await self._synthesize(topic, annotations, papers, output_language=output_language)

        self._log("4/4 Formatting report...")
        references = format_references(papers, citation_format)
        report_markdown = self._render_report(
            topic=topic,
            search_query=search_query,
            search_source=search_source,
            category=category,
            start_date=start_date,
            end_date=end_date,
            citation_format=citation_format,
            papers=papers,
            annotations=annotations,
            synthesis=synthesis,
            references=references,
            source_counts=source_counts,
            source_mode=source_mode,
            output_language=output_language,
        )
        self._log("Done.")
        return SLRResult(
            topic=topic,
            search_query=search_query,
            citation_format=citation_format,
            papers=papers,
            source_counts=source_counts,
            source_errors=source_errors,
            annotations=annotations,
            synthesis=synthesis,
            references=references,
            report_markdown=report_markdown,
        )

    def _search_sources(
        self,
        *,
        topic: str,
        search_plan: list[dict],
        max_results: int,
        category: str,
        start_date: str,
        end_date: str,
    ) -> tuple[list[dict], dict[str, int], dict[str, str]]:
        source_counts: dict[str, int] = {}
        source_errors: dict[str, str] = {}
        collected: list[dict] = []
        for plan in search_plan:
            query = str(plan.get("query") or "").strip()
            if not query:
                continue
            label = str(plan.get("label") or "search")
            target_language = str(plan.get("target_language") or "unknown")
            target_count = max(1, min(max_results, int(plan.get("target_count") or max_results)))
            per_source_limit = max(target_count, min(max_results, target_count * 2))
            source_calls = [
                (
                    "arXiv",
                    lambda query=query: [
                        paper.to_dict()
                        for paper in search_arxiv(
                            query,
                            max_results=per_source_limit,
                            category=category or None,
                            sort_by="relevance",
                            start_date=start_date or None,
                            end_date=end_date or None,
                        )
                    ],
                ),
                (
                    "OpenAlex",
                    lambda query=query: search_openalex(
                        query,
                        max_results=per_source_limit,
                        start_date=start_date,
                        end_date=end_date,
                    ),
                ),
                (
                    "PubMed",
                    lambda query=query: search_pubmed(
                        query,
                        max_results=per_source_limit,
                        start_date=start_date,
                        end_date=end_date,
                    ),
                ),
                (
                    "Crossref",
                    lambda query=query: search_crossref(
                        query,
                        max_results=per_source_limit,
                        start_date=start_date,
                        end_date=end_date,
                    ),
                ),
            ]
            for source_name, call in source_calls:
                source_key = f"{source_name} {label}"
                try:
                    results = call()
                except Exception as error:
                    source_errors[source_key] = f"{type(error).__name__}: {error}"
                    self._log(f"{source_key} search unavailable: {error}")
                    continue
                normalized_results = [
                    {
                        **paper,
                        "search_query": query,
                        "search_language": target_language,
                        "output_language": str(plan.get("output_language") or target_language),
                    }
                    for paper in results
                ]
                source_counts[source_key] = len(normalized_results)
                collected.extend(normalized_results)

        deduped = self._dedupe_and_rank(collected, max_results=max(max_results * 2, max_results))
        return self._select_by_search_plan(deduped, search_plan, max_results=max_results), source_counts, source_errors

    async def _build_search_plan(self, topic: str, search_query: str) -> list[dict]:
        if main_language(topic, default="en") == "zh":
            english_query = await self._translate_query_to_english(topic, search_query)
            english_query = english_query or search_query
            return [
                {
                    "label": "中文检索",
                    "query": search_query,
                    "target_language": "zh",
                    "output_language": "zh",
                    "target_ratio": 0.5,
                },
                {
                    "label": "English search",
                    "query": english_query,
                    "target_language": "en",
                    "output_language": "zh",
                    "target_ratio": 0.5,
                },
            ]
        return [
            {
                "label": "English search",
                "query": search_query,
                "target_language": "en",
                "output_language": "en",
                "target_ratio": 1.0,
            }
        ]

    async def _translate_query_to_english(self, topic: str, fallback_query: str) -> str:
        system_prompt = """
You generate concise academic search queries.
Return only valid JSON: {"query":"..."}.
Translate Chinese research topics into English scholarly keywords.
Keep the query under 8 words. Remove instructions such as review, survey, papers, literature.
""".strip()
        user_prompt = f"Topic:\n{topic}\n\nFallback query:\n{fallback_query}"
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("SLR_MODEL") or os.getenv("RESEARCH_MODEL"),
                temperature=0.1,
                max_tokens=160,
            )
            data = self._parse_json(content)
            query = str(data.get("query") or "").strip() if isinstance(data, dict) else ""
            return query[:120]
        except (LLMServiceError, ValueError, TypeError, json.JSONDecodeError):
            self._log("English query translation failed; using original query.")
            return fallback_query

    @staticmethod
    def _select_by_search_plan(papers: list[dict], search_plan: list[dict], *, max_results: int) -> list[dict]:
        if not search_plan:
            return papers[:max_results]
        buckets = {
            "zh": [paper for paper in papers if SLRWorkflow._paper_language(paper) == "zh"],
            "en": [paper for paper in papers if SLRWorkflow._paper_language(paper) == "en"],
            "other": [paper for paper in papers if SLRWorkflow._paper_language(paper) not in {"zh", "en"}],
        }
        selected: list[dict] = []
        selected_keys: set[str] = set()

        for plan in search_plan:
            language = str(plan.get("target_language") or "")
            ratio = float(plan.get("target_ratio") or 0)
            target = max(1, round(max_results * ratio))
            if len(search_plan) == 1:
                target = max_results
            for paper in buckets.get(language, [])[:target]:
                key = SLRWorkflow._paper_key(paper)
                if key in selected_keys:
                    continue
                selected.append(paper)
                selected_keys.add(key)
                if len(selected) >= max_results:
                    return selected

        for paper in papers:
            key = SLRWorkflow._paper_key(paper)
            if key in selected_keys:
                continue
            selected.append(paper)
            selected_keys.add(key)
            if len(selected) >= max_results:
                break
        return selected

    @staticmethod
    def _paper_language(paper: dict) -> str:
        return paper_language(paper)

    @staticmethod
    def _contains_cjk(value: str) -> bool:
        return contains_cjk(value)

    async def _extract_annotations(self, papers: list[dict], *, output_language: str = "en") -> list[dict]:
        batches = self._annotation_batches(papers)
        all_annotations: list[dict] = []
        for start in range(0, len(batches), 3):
            round_batches = batches[start : start + 3]
            results = await asyncio.gather(
                *[self._extract_batch(batch, output_language=output_language) for batch in round_batches],
                return_exceptions=True,
            )
            for batch, result in zip(round_batches, results):
                if isinstance(result, Exception):
                    self._log(f"Annotation batch failed; using fallback rows. {type(result).__name__}: {result}")
                    all_annotations.extend(self._fallback_annotations(batch, output_language=output_language))
                    continue
                all_annotations.extend(result)
        return self._align_annotations(all_annotations, papers, output_language=output_language)

    @staticmethod
    def _annotation_batches(papers: list[dict]) -> list[list[dict]]:
        batches: list[list[dict]] = []
        metadata_batch: list[dict] = []

        def flush_metadata_batch() -> None:
            nonlocal metadata_batch
            if metadata_batch:
                batches.append(metadata_batch)
                metadata_batch = []

        for paper in papers:
            if paper.get("content_excerpt") or paper.get("pdf_text_available"):
                flush_metadata_batch()
                batches.append([paper])
                continue
            metadata_batch.append(paper)
            if len(metadata_batch) >= 3:
                flush_metadata_batch()
        flush_metadata_batch()
        return batches

    async def _extract_batch(self, papers: list[dict], *, output_language: str = "en") -> list[dict]:
        use_chinese = any(
            output_language == "zh"
            or paper.get("output_language") == "zh"
            or self._contains_cjk(str(paper.get("search_query") or ""))
            for paper in papers
        )
        system_prompt = """
You extract structured metadata for a systematic literature review.
Return only a valid JSON array. Do not use markdown fences.
For each scholarly record, infer from the provided title, abstract, full_text_excerpt, and metadata.
When full_text_excerpt is available, prioritize it over metadata-only inference and analyze the actual article content in that excerpt.
Do not invent experiments, datasets, or results not present in the supplied context.
Do not copy section markers such as [Abstract], [Methods], [Results], [Opening Context], or page labels into any output field.
If the abstract is empty but full_text_excerpt is available, use the excerpt. Do not write placeholders such as "empty abstract" or "摘要为空".
""".strip()
        user_prompt = (
            "Papers:\n"
            + json.dumps(self._annotation_prompt_papers(papers), ensure_ascii=False, indent=2)
            + """

For each paper return:
{
  "arxiv_id": "must exactly equal the supplied paper id",
  "title": "...",
  "source": "arXiv / OpenAlex / PubMed / Crossref",
  "authors": ["..."],
  "published_date": "YYYY-MM-DD",
  "research_question": "one sentence",
  "methodology": "one or two sentences",
  "key_findings": ["3 to 5 concise bullets"],
  "innovation_point": "one concise sentence on what is novel, distinctive, or practically useful in this paper",
  "limitations": "one or two sentences"
}
""".strip()
        )
        if use_chinese:
            user_prompt += (
                "\n\nOutput language: Chinese. Translate all narrative and analytical fields into Chinese. "
                "Keep paper titles, author names, source names, journal names, DOIs, URLs, model names, dataset names, and metric names unchanged."
            )
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("SLR_MODEL") or os.getenv("RESEARCH_MODEL"),
                temperature=0.15,
                max_tokens=3200 if len(papers) > 1 else 2200,
            )
            data = self._parse_json(content)
            if not isinstance(data, list):
                raise ValueError("Expected JSON array.")
            items = [item for item in data if isinstance(item, dict)]
            self._log(f"Annotation batch returned {len(items)}/{len(papers)} structured rows.")
            return items
        except (LLMServiceError, ValueError, TypeError, json.JSONDecodeError) as error:
            self._log(f"Annotation batch parse/extraction failed: {type(error).__name__}: {error}")
            if len(papers) == 1 and str(papers[0].get("content_excerpt") or "").strip():
                retry = await self._extract_single_paper_retry(papers[0], output_language=output_language)
                if retry:
                    self._log(f"Single-paper retry succeeded for {papers[0].get('id') or papers[0].get('title')}.")
                    return [retry]
                self._log(f"Single-paper retry failed for {papers[0].get('id') or papers[0].get('title')}; using fallback.")
            return self._fallback_annotations(papers, output_language=output_language)

    async def _extract_single_paper_retry(self, paper: dict, *, output_language: str = "en") -> dict | None:
        use_chinese = (
            output_language == "zh"
            or paper.get("output_language") == "zh"
            or self._contains_cjk(str(paper.get("search_query") or ""))
        )
        language_note = (
            "Write all narrative and analytical fields in concise Chinese. Keep the original title, author names, source names, journal names, DOIs, URLs, model names, dataset names, and metric names unchanged."
            if use_chinese
            else "Write analytical fields in concise English. Keep the original title unchanged."
        )
        system_prompt = """
You analyze one uploaded scholarly PDF excerpt for a systematic literature review.
Return only one valid JSON object. Do not use markdown fences.
Ground every field in the supplied excerpt. If a detail is absent, say it is unclear from the excerpt.
Do not copy section markers such as [Abstract], [Methods], [Results], [Opening Context], or page labels into any output field.
Do not write placeholders such as "empty abstract" or "摘要为空"; use evidence from the excerpt whenever available.
""".strip()
        prompt_paper = self._annotation_prompt_papers([paper])[0]
        user_prompt = f"""
Paper:
{json.dumps(prompt_paper, ensure_ascii=False, indent=2)}

Return this JSON object:
{{
  "arxiv_id": "must exactly equal the supplied paper id",
  "title": "...",
  "source": "...",
  "authors": ["..."],
  "published_date": "YYYY-MM-DD or empty",
  "research_question": "one sentence",
  "methodology": "one or two sentences",
  "key_findings": ["2 to 4 concise bullets"],
  "innovation_point": "one concise sentence",
  "limitations": "one or two sentences"
}}

{language_note}
""".strip()
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("SLR_MODEL") or os.getenv("RESEARCH_MODEL"),
                temperature=0.1,
                max_tokens=1800,
            )
            data = self._parse_json(content)
            return data if isinstance(data, dict) else None
        except (LLMServiceError, ValueError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _annotation_prompt_papers(papers: list[dict]) -> list[dict]:
        prompt_papers = []
        for paper in papers:
            item = {
                "id": paper.get("id", ""),
                "title": paper.get("title", ""),
                "source": paper.get("source", ""),
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "doi": paper.get("doi", ""),
                "journal": paper.get("journal", ""),
                "abstract": str(paper.get("abstract") or "")[:1800],
            }
            excerpt = str(paper.get("content_excerpt") or "")
            if excerpt:
                item["full_text_excerpt"] = SLRWorkflow._high_information_excerpt(excerpt)
                item["pdf_text_available"] = bool(paper.get("pdf_text_available", False))
            prompt_papers.append(item)
        return prompt_papers

    @staticmethod
    def _high_information_excerpt(text: str, *, max_chars: int = 15000) -> str:
        cleaned = SLRWorkflow._compact_text(text)
        if len(cleaned) <= max_chars:
            return cleaned

        sections = SLRWorkflow._section_slices(cleaned)
        plan = [
            ("abstract", 2200),
            ("introduction", 2200),
            ("methods", 3600),
            ("results", 3200),
            ("discussion", 2200),
            ("limitations", 1400),
            ("conclusion", 1400),
        ]
        parts = []
        used_keys = set()
        for key, budget in plan:
            section = sections.get(key, "")
            if section:
                label = key.replace("_", " ").title()
                parts.append(f"[{label}]\n{section[:budget].strip()}")
                used_keys.add(key)

        if not parts:
            return cleaned[:max_chars]

        packed = "\n\n".join(parts)
        if len(packed) < max_chars * 0.65:
            tail_budget = max_chars - len(packed) - 16
            if tail_budget > 800:
                packed = f"{packed}\n\n[Opening Context]\n{cleaned[:tail_budget].strip()}"
        return packed[:max_chars]

    @staticmethod
    def _section_slices(text: str) -> dict[str, str]:
        heading_patterns = [
            ("abstract", r"\babstract\b"),
            ("introduction", r"\b(?:1\s*)?introduction\b"),
            ("methods", r"\b(?:materials?\s+and\s+methods|methods?|methodology|model|approach)\b"),
            ("results", r"\b(?:results?|experiments?|evaluation)\b"),
            ("discussion", r"\bdiscussion\b"),
            ("limitations", r"\b(?:limitations?|limitation\s+of\s+the\s+study)\b"),
            ("conclusion", r"\b(?:conclusions?|summary)\b"),
            ("references", r"\breferences\b"),
        ]
        matches = []
        for key, pattern in heading_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                prefix = text[max(0, match.start() - 30) : match.start()]
                if len(prefix.strip()) > 25 and not re.search(r"[\n\r.!?。！？]\s*$", prefix):
                    continue
                matches.append((match.start(), match.end(), key))
                break
        matches.sort(key=lambda item: item[0])

        sections: dict[str, str] = {}
        introduction_start = next((start for start, _, key in matches if key == "introduction"), None)
        if introduction_start and "abstract" not in {key for _, _, key in matches}:
            front_matter = text[:introduction_start].strip()
            abstract_match = re.search(r"\babstract\b[:.\s-]*(.+)", front_matter, flags=re.IGNORECASE | re.DOTALL)
            sections["abstract"] = (abstract_match.group(1) if abstract_match else front_matter)[-2600:].strip()
        for index, (start, end, key) in enumerate(matches):
            if key == "references":
                continue
            next_start = len(text)
            for later_start, _, later_key in matches[index + 1 :]:
                next_start = later_start
                break
            chunk = text[end:next_start].strip()
            if chunk and key not in sections:
                sections[key] = chunk
        return sections

    async def _synthesize(self, topic: str, annotations: list[dict], papers: list[dict], *, output_language: str = "en") -> dict:
        system_prompt = """
You are the lead agent for a systematic literature review.
Synthesize across papers instead of listing papers one by one.
Return only valid JSON with this schema:
{
  "executive_summary": "3-5 sentences",
  "themes": [{"name": "...", "summary": "...", "paper_ids": ["..."]}],
  "convergences": ["..."],
  "disagreements": ["..."],
  "gaps": ["..."],
  "methodological_patterns": ["..."],
  "limitations_of_review": "..."
}
Use the requested output language for all narrative values. Keep paper titles, author names, DOIs, source names, journal names, model names, dataset names, and metric names unchanged.
""".strip()
        user_prompt = (
            f"Topic:\n{topic}\n\n"
            f"Requested output language: {'Chinese' if output_language == 'zh' else 'English'}\n\n"
            f"Paper annotations:\n{json.dumps(annotations, ensure_ascii=False, indent=2)}"
        )
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("SLR_MODEL") or os.getenv("SYNTHESIS_MODEL"),
                temperature=0.2,
                max_tokens=3200,
            )
            data = self._parse_json(content)
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object.")
            return self._normalize_synthesis(data, annotations, papers, output_language=output_language)
        except (LLMServiceError, ValueError, TypeError, json.JSONDecodeError):
            return self._fallback_synthesis(topic, annotations, papers, output_language=output_language)

    def _render_report(
        self,
        *,
        topic: str,
        search_query: str,
        search_source: str,
        category: str,
        start_date: str,
        end_date: str,
        citation_format: str,
        papers: list[dict],
        annotations: list[dict],
        synthesis: dict,
        references: list[str],
        source_counts: dict[str, int],
        source_mode: str,
        output_language: str = "en",
    ) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        use_chinese = output_language == "zh"
        scope_parts = (
            [f"检索词 `{search_query}`", f"来源 {search_source}", "按 DOI / URL / 标题去重"]
            if use_chinese
            else [f"query `{search_query}`", f"sources {search_source}", "deduplicated by DOI/URL/title"]
        )
        if category:
            scope_parts.append(f"arXiv 分类 `{category}`" if use_chinese else f"arXiv category `{category}`")
        if start_date or end_date:
            scope_parts.append(
                f"时间范围 {start_date or '不限'} 至 {end_date or '不限'}"
                if use_chinese
                else f"date window {start_date or 'open'} to {end_date or 'open'}"
            )
        citation_label = {"apa": "APA 7th edition", "ieee": "IEEE", "bibtex": "BibTeX"}[citation_format]
        methodology = (
            f"本综述在 {today} 检索并合并了 {len(papers)} 条去重后的学术记录，检索词为 `{search_query}`。"
            "系统分别查询多个来源，随后按 DOI / URL / 标题合并去重，再通过批量语言模型抽取和跨文献综合生成综述。"
            f"最终去重前的来源统计：{self._format_source_counts(source_counts)}。"
            if use_chinese
            else (
                f"This review surveyed {len(papers)} deduplicated scholarly records retrieved on {today} "
                f"using the query `{search_query}`. Sources were searched independently, merged by DOI/URL/title, "
                "then analyzed through batch language-model extraction followed by cross-paper synthesis. "
                f"Source counts before final deduplication: {self._format_source_counts(source_counts)}."
            )
        )
        default_limits = (
            "覆盖范围取决于已连接来源返回的元数据和摘要，个别论文仍需结合全文复核。"
            if use_chinese
            else "Coverage depends on metadata and abstracts returned by the connected sources."
        )

        if use_chinese:
            sections = [
                f"# 系统性文献综述：{topic}",
                f"**日期**：{today}",
                f"**纳入文献数**：{len(papers)}",
                f"**范围**：{'，'.join(scope_parts)}",
                f"**引用格式**：{citation_label}",
                f"**来源模式**：{source_mode}",
                "## 执行摘要",
                synthesis.get("executive_summary", ""),
                "## 方法说明",
                methodology,
                f"**综述局限**：{synthesis.get('limitations_of_review') or default_limits}",
                "## 主题聚类",
                self._render_themes(synthesis.get("themes", []), use_chinese=use_chinese),
                "## 共识与分歧",
                "**共识**：\n" + self._render_bullets(synthesis.get("convergences", [])),
                "**分歧**：\n" + self._render_bullets(synthesis.get("disagreements", [])),
                "## 逐篇文献注释",
                self._render_annotations(annotations, use_chinese=use_chinese),
                "## 参考文献",
                self._render_references(references, citation_format),
            ]
            return "\n\n".join(section.strip() for section in sections if section is not None).strip() + "\n"

        sections = [
            f"# Systematic Literature Review: {topic}",
            f"**Date**: {today}",
            f"**Papers surveyed**: {len(papers)}",
            f"**Scope**: {', '.join(scope_parts)}",
            f"**Citation format**: {citation_label}",
            f"**Source mode**: {source_mode}",
            "## Executive Summary",
            synthesis.get("executive_summary", ""),
            "## Methodology",
            methodology,
            f"**Limitations of this review**: {synthesis.get('limitations_of_review') or default_limits}",
            "## Themes",
            self._render_themes(synthesis.get("themes", []), use_chinese=use_chinese),
            "## Convergences and Disagreements",
            "**Convergences**:\n" + self._render_bullets(synthesis.get("convergences", [])),
            "**Disagreements**:\n" + self._render_bullets(synthesis.get("disagreements", [])),
            "## Per-Paper Annotations",
            self._render_annotations(annotations, use_chinese=use_chinese),
            "## References",
            self._render_references(references, citation_format),
        ]
        return "\n\n".join(section.strip() for section in sections if section is not None).strip() + "\n"

    def _render_themes(self, themes: list[dict], *, use_chinese: bool = False) -> str:
        if not themes:
            return "未识别出稳定的跨文献主题。" if use_chinese else "No stable cross-paper themes were identified."
        blocks = []
        for index, theme in enumerate(themes[:6], start=1):
            paper_ids = ", ".join(str(item) for item in theme.get("paper_ids", []) if item)
            suffix = f"\n\n相关论文：{paper_ids}" if paper_ids and use_chinese else (f"\n\nRelated papers: {paper_ids}" if paper_ids else "")
            prefix = f"主题 {index}" if use_chinese else f"Theme {index}"
            fallback = "未命名主题" if use_chinese else "Untitled theme"
            blocks.append(f"### {prefix}: {theme.get('name', fallback)}\n\n{theme.get('summary', '')}{suffix}")
        return "\n\n".join(blocks)

    def _render_annotations(self, annotations: list[dict], *, use_chinese: bool = False) -> str:
        if not annotations:
            return "\u672a\u751f\u6210\u9010\u7bc7\u6587\u732e\u6ce8\u91ca\u3002" if use_chinese else "No per-paper annotations were generated."

        headers = (
            ["\u6587\u732e/\u6765\u6e90", "\u7814\u7a76\u95ee\u9898", "\u65b9\u6cd5/\u6750\u6599", "\u5173\u952e\u53d1\u73b0", "\u5c40\u9650", "\u521b\u65b0\u70b9"]
            if use_chinese
            else ["Paper / source", "Research question", "Method / material", "Key findings", "Limitations", "Innovation point"]
        )
        rows = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for annotation in sorted(annotations, key=lambda item: (year_from_paper(item), str(item.get("title", "")))):
            findings = annotation.get("key_findings", [])
            if not isinstance(findings, list):
                findings = [str(findings)]
            source_parts = [
                str(annotation.get("title") or ("\u672a\u547d\u540d\u8bba\u6587" if use_chinese else "Untitled paper")),
                str(annotation.get("source") or ("\u672a\u77e5" if use_chinese else "Unknown")),
                str(annotation.get("arxiv_id") or ""),
            ]
            cells = [
                "<br>".join(self._markdown_table_cell(part) for part in source_parts if part),
                self._markdown_table_cell(annotation.get("research_question", "")),
                self._markdown_table_cell(annotation.get("methodology", "")),
                "<br>".join(self._markdown_table_cell(item) for item in findings if str(item).strip()),
                self._markdown_table_cell(annotation.get("limitations", "")),
                self._markdown_table_cell(self._annotation_innovation(annotation, findings, use_chinese)),
            ]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    @staticmethod
    def _annotation_innovation(annotation: dict, findings: list, use_chinese: bool) -> str:
        for key in ("innovation_point", "innovation", "novelty", "contribution"):
            value = str(annotation.get(key) or "").strip()
            if value:
                return value
        for finding in findings:
            value = str(finding or "").strip()
            if value:
                return value
        return "\u9700\u7ed3\u5408\u6458\u8981\u6216\u5168\u6587\u8fdb\u4e00\u6b65\u5224\u65ad\u521b\u65b0\u70b9\u3002" if use_chinese else "Requires abstract or full-text review to identify the innovation point."

    @staticmethod
    def _markdown_table_cell(value) -> str:
        return (
            SLRWorkflow._clean_annotation_text(value)
            .replace("\r\n", "<br>")
            .replace("\n", "<br>")
            .replace("|", "\\|")
            .strip()
        )

    @staticmethod
    def _clean_annotation_text(value) -> str:
        text = str(value or "")
        if SLRWorkflow._is_low_information_annotation_text(text):
            fallback = "需结合摘要或全文进一步确认。" if SLRWorkflow._contains_cjk(text) else "Requires abstract or full-text review."
            return SLRWorkflow._with_fallback_prefix(fallback)
        text = re.sub(r"^\s*\[(?:Abstract|Introduction|Methods?|Materials?|Results?|Experiments?|Evaluation|Discussion|Limitations?|Conclusion|Summary|Opening Context)\]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?m)^\s*\[Page\s+\d+\]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*\[(?:Abstract|Introduction|Methods?|Materials?|Results?|Experiments?|Evaluation|Discussion|Limitations?|Conclusion|Summary|Opening Context)\]\s*", " ", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_low_information_annotation_text(value) -> bool:
        text = SLRWorkflow._compact_text(str(value or ""))
        if not text:
            return True
        lower = text.lower()
        patterns = [
            "摘要为空",
            "摘要缺失",
            "无法评估研究的局限性",
            "无法获取具体的研究方法",
            "无法获取具体的研究方法、详细发现和局限性信息",
            "empty abstract",
            "abstract is empty",
            "abstract-based extraction is unavailable",
            "no limitations can be inferred from the empty abstract",
            "specific limitations cannot be inferred",
            "unknown from provided information",
        ]
        if any(pattern in lower for pattern in patterns):
            return True
        if text.startswith("?? PDF") or re.search(r"\?{2,}.*(?:PDF|Discussion|Limitations).*?\?{2,}", text, flags=re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _repair_annotation_fields(item: dict, paper: dict, *, use_chinese: bool) -> dict:
        repaired = dict(item)
        excerpt = str(paper.get("content_excerpt") or paper.get("abstract") or paper.get("relevance") or "")

        if SLRWorkflow._is_low_information_annotation_text(repaired.get("research_question", "")):
            repaired["research_question"] = SLRWorkflow._with_fallback_prefix(SLRWorkflow._fallback_research_question(excerpt, use_chinese))
        if SLRWorkflow._is_low_information_annotation_text(repaired.get("methodology", "")):
            repaired["methodology"] = SLRWorkflow._with_fallback_prefix(SLRWorkflow._fallback_methodology(excerpt, use_chinese))

        findings = repaired.get("key_findings", [])
        if not isinstance(findings, list):
            findings = [str(findings)]
        findings = [
            str(finding).strip()
            for finding in findings
            if str(finding).strip()
            and not SLRWorkflow._is_low_information_annotation_text(finding)
        ]
        raw_fallback_findings = []
        if not findings:
            raw_fallback_findings = SLRWorkflow._fallback_findings_from_excerpt(excerpt, use_chinese)
            findings = [SLRWorkflow._with_fallback_prefix(finding) for finding in raw_fallback_findings]
        repaired["key_findings"] = findings

        if SLRWorkflow._is_low_information_annotation_text(repaired.get("innovation_point", "")):
            innovation_findings = raw_fallback_findings or findings
            repaired["innovation_point"] = SLRWorkflow._with_fallback_prefix(SLRWorkflow._fallback_innovation(innovation_findings, use_chinese))
        if SLRWorkflow._is_low_information_annotation_text(repaired.get("limitations", "")):
            repaired["limitations"] = SLRWorkflow._with_fallback_prefix(SLRWorkflow._fallback_limitations(excerpt, paper, use_chinese))
        return repaired

    @staticmethod
    def _with_fallback_prefix(value) -> str:
        text = SLRWorkflow._compact_text(str(value or ""))
        if not text:
            return ""
        return re.sub(r"^\[\u515c\u5e95\]\s*", "", text).strip()

    @staticmethod
    def _fallback_annotations(papers: list[dict], output_language: str = "en") -> list[dict]:
        annotations = []
        use_chinese = any(
            output_language == "zh"
            or paper.get("output_language") == "zh"
            or SLRWorkflow._contains_cjk(str(paper.get("search_query") or ""))
            for paper in papers
        )
        for paper in papers:
            excerpt = str(paper.get("content_excerpt") or paper.get("abstract") or paper.get("relevance") or "")
            abstract = excerpt
            findings = SLRWorkflow._fallback_findings_from_excerpt(excerpt, use_chinese)
            annotations.append(
                {
                    "arxiv_id": paper.get("id", ""),
                    "title": paper.get("title", ""),
                    "source": paper.get("source", ""),
                    "authors": paper.get("authors", []),
                    "published_date": paper.get("published", ""),
                    "source_origin": paper.get("source_origin", ""),
                    "research_question": abstract[:240] or ("需要结合摘要或全文进一步确认。" if use_chinese else "Requires abstract or full-text review."),
                    "methodology": "方法细节需要通过模型抽取或全文阅读进一步确认。" if use_chinese else "Method details require model extraction or full-paper review.",
                    "key_findings": ["当前只能基于元数据生成初步注释，建议进一步查看原文。"] if use_chinese else ["Abstract-based extraction is unavailable; inspect the paper directly."],
                    "limitations": "该注释仅基于可用元数据生成。" if use_chinese else "Fallback annotation based on available metadata only.",
                }
            )
            annotations[-1].update(
                {
                    "research_question": SLRWorkflow._with_fallback_prefix(SLRWorkflow._fallback_research_question(excerpt, use_chinese)),
                    "methodology": SLRWorkflow._with_fallback_prefix(SLRWorkflow._fallback_methodology(excerpt, use_chinese)),
                    "key_findings": [SLRWorkflow._with_fallback_prefix(finding) for finding in findings],
                    "innovation_point": SLRWorkflow._with_fallback_prefix(SLRWorkflow._fallback_innovation(findings, use_chinese)),
                    "limitations": SLRWorkflow._with_fallback_prefix(SLRWorkflow._fallback_limitations(excerpt, paper, use_chinese)),
                }
            )
        return annotations

    @staticmethod
    def _fallback_research_question(excerpt: str, use_chinese: bool) -> str:
        text = SLRWorkflow._compact_text(excerpt)
        if text:
            if use_chinese:
                return SLRWorkflow._summarize_excerpt_in_chinese(text, purpose="research_question")
            prefix = "该文献主要围绕：" if use_chinese else "This paper appears to address: "
            return prefix + text[:220]
        return "需要结合摘要或全文进一步确认。" if use_chinese else "Requires abstract or full-text review."

    @staticmethod
    def _fallback_methodology(excerpt: str, use_chinese: bool) -> str:
        text = SLRWorkflow._find_sentence_with_keywords(
            excerpt,
            ["method", "methods", "materials", "trained", "training", "model", "network", "cnn", "u-net", "dataset", "patients", "cohort", "segmentation"],
        )
        if text:
            if use_chinese:
                return SLRWorkflow._summarize_excerpt_in_chinese(text, purpose="methodology")
            return text[:360]
        return "已从上传 PDF 提取正文片段，但自动结构化抽取失败；建议结合方法部分复核。" if use_chinese else "Full-text excerpt was extracted, but structured method extraction failed; review the methods section for confirmation."

    @staticmethod
    def _fallback_findings_from_excerpt(excerpt: str, use_chinese: bool) -> list[str]:
        candidates = SLRWorkflow._sentences(excerpt)
        keywords = ["achieved", "dice", "dsc", "result", "results", "improved", "outperform", "performance", "accuracy", "segmentation", "propose", "proposed"]
        findings = [
            sentence[:300]
            for sentence in candidates
            if any(keyword in sentence.lower() for keyword in keywords)
        ][:3]
        if findings:
            if use_chinese:
                return [SLRWorkflow._summarize_excerpt_in_chinese(sentence, purpose="finding") for sentence in findings]
            return findings
        compact = SLRWorkflow._compact_text(excerpt)
        if compact:
            if use_chinese:
                return [SLRWorkflow._summarize_excerpt_in_chinese(compact, purpose="finding")]
            return [compact[:300]]
        return ["已上传 PDF，但可用正文片段不足以稳定提取关键发现。"] if use_chinese else ["A PDF was uploaded, but the available text excerpt was insufficient to extract stable findings."]

    @staticmethod
    def _fallback_innovation(findings: list[str], use_chinese: bool) -> str:
        if findings:
            if use_chinese:
                return "创新点主要体现在：" + SLRWorkflow._strip_chinese_lead(str(findings[0]))[:220]
            return findings[0][:260]
        return "需结合全文进一步判断创新点。" if use_chinese else "Requires full-text review to identify the innovation point."

    @staticmethod
    def _fallback_limitations(excerpt: str, paper: dict, use_chinese: bool) -> str:
        text = SLRWorkflow._find_sentence_with_keywords(
            excerpt,
            ["limitation", "limitations", "limited by", "small sample", "retrospective", "single-center", "single center", "future work"],
        )
        if text:
            if use_chinese:
                return SLRWorkflow._summarize_excerpt_in_chinese(text, purpose="limitation")
            prefix = "主要局限：" if use_chinese else "Main limitation: "
            return prefix + text[:300]
        if paper.get("pdf_text_available"):
            return "可用 PDF 片段中未识别到明确局限性表述；建议复核 Discussion 或 Limitations 部分。" if use_chinese else "No explicit limitation statement was detected in the available PDF excerpt; confirm in the Discussion or Limitations section."
        return "未提取到可读全文，局限性信息仍不确定。" if use_chinese else "Readable full text was not extracted, so limitations are uncertain."

    @staticmethod
    def _summarize_excerpt_in_chinese(text: str, *, purpose: str) -> str:
        compact = SLRWorkflow._compact_text(text)
        lower = compact.lower()
        method_terms = SLRWorkflow._extract_method_terms(compact)
        metric_summary = SLRWorkflow._extract_metric_summary(compact)
        domain = SLRWorkflow._infer_chinese_domain(compact)

        if purpose == "research_question":
            if re.search(r"\b(propose|proposed|develop|developed|automatic|automated)\b", lower):
                return f"该文献主要研究{domain}的自动化方法。"
            return f"该文献主要围绕{domain}展开。"

        if purpose == "methodology":
            if method_terms:
                return f"方法上，研究使用{method_terms}处理{domain}任务。"
            if re.search(r"\b(train|trained|training)\b", lower):
                return f"方法上，研究训练模型处理{domain}任务，并在可用数据上进行评估。"
            return f"方法细节来自 PDF 正文片段，主要涉及{domain}任务；仍建议复核原文方法部分。"

        if purpose == "finding":
            if metric_summary:
                return f"结果显示，{metric_summary}。"
            if re.search(r"\b(improv|outperform|better|significant)\w*\b", lower):
                return "结果显示，所提方法相较基线或对照方案有性能提升。"
            if re.search(r"\b(propose|proposed|develop|developed)\b", lower):
                return f"研究提出了用于{domain}的自动化分析方法。"
            return f"正文片段显示，该研究报告了与{domain}相关的实验或应用结果。"

        if purpose == "limitation":
            if "future work" in lower:
                return "主要局限：作者将进一步验证或扩展该方法，说明当前证据仍有待后续工作补充。"
            if "small sample" in lower:
                return "主要局限：样本量较小，结论的稳定性和泛化性仍需进一步验证。"
            if "single-center" in lower or "single center" in lower:
                return "主要局限：研究可能受单中心数据限制，外部泛化能力仍需验证。"
            if "retrospective" in lower:
                return "主要局限：研究设计包含回顾性因素，仍需要前瞻性或外部验证。"
            return "主要局限：PDF 片段提到研究限制或后续工作，但具体影响仍需结合原文 Discussion 或 Limitations 部分复核。"

        return compact[:220]

    @staticmethod
    def _extract_method_terms(text: str) -> str:
        terms = []
        patterns = [
            (r"(?<![A-Za-z0-9])nnU-Net(?![A-Za-z0-9])", "nnU-Net"),
            (r"(?<![A-Za-z0-9])U-Net(?![A-Za-z0-9])", "U-Net"),
            (r"\b3D\s*CNN\b|\bCNN\b", "CNN"),
            (r"\bdiffusion\b", "diffusion model"),
            (r"\badversarial\b", "adversarial learning"),
            (r"\battention\b", "attention mechanism"),
            (r"\bpre-?processing\b", "preprocessing"),
            (r"\brandom expert sampling\b", "random expert sampling"),
            (r"\bdeep learning\b", "deep learning"),
        ]
        for pattern, label in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE) and label not in terms:
                terms.append(label)
        return "、".join(terms[:4])

    @staticmethod
    def _extract_metric_summary(text: str) -> str:
        metrics = []
        number = r"([0-9]+(?:\.[0-9]+)?%?)"
        metric_patterns = [
            (rf"\bDice(?:\s+score)?s?\s*(?:of|=|:)?\s*{number}", "Dice"),
            (rf"\b(?:IoU|Intersection over Union)\s*(?:scores?\s*)?(?:of|=|:)?\s*{number}", "IoU"),
            (rf"\bF1\s*(?:scores?\s*)?(?:of|=|:)?\s*{number}", "F1"),
            (rf"\bRecall\s*(?:scores?\s*)?(?:of|=|:)?\s*{number}", "Recall"),
            (rf"\bPrecision\s*(?:scores?\s*)?(?:of|=|:)?\s*{number}", "Precision"),
            (rf"\bSurface Dice(?: at Tolerance \d+mm)?\s*(?:improvement of\s*)?{number}", "Surface Dice"),
        ]
        for pattern, label in metric_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                metrics.append(f"{label} 为 {match.group(1)}")
        improvement = re.search(r"\b(improv(?:ed|ement)?)(?:\s+\w+){0,4}\s+(?:by|of|to)\s*([0-9.]+%?|[0-9.]+)", text, flags=re.IGNORECASE)
        if improvement:
            metrics.append(f"性能提升至/提升约 {improvement.group(2)}")
        return "，".join(dict.fromkeys(metrics[:4]))

    @staticmethod
    def _infer_chinese_domain(text: str) -> str:
        lower = text.lower()
        if any(term in lower for term in ["ischemic stroke", "stroke lesion", "infarct", "ncct", "computed tomography", "ct"]):
            if "segmentation" in lower:
                return "头颅 CT/NCCT 缺血性卒中病灶分割"
            return "头颅 CT/NCCT 缺血性卒中影像分析"
        if "segmentation" in lower:
            return "医学图像分割"
        if "medical image" in lower or "imaging" in lower:
            return "医学影像分析"
        return "该研究主题"

    @staticmethod
    def _strip_chinese_lead(text: str) -> str:
        return re.sub(r"^(结果显示，|研究提出了|正文片段显示，|该文献主要研究|该文献主要围绕|创新点主要体现在：)", "", text).strip()

    @staticmethod
    def _sentences(value: str) -> list[str]:
        text = SLRWorkflow._compact_text(value)
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|[\u3002\uff01\uff1f]\s*", text) if part.strip()]

    @staticmethod
    def _compact_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _find_sentence_with_keywords(value: str, keywords: list[str]) -> str:
        for sentence in SLRWorkflow._sentences(value):
            lower = sentence.lower()
            if any(keyword in lower for keyword in keywords):
                return sentence
        return ""

    @staticmethod
    def _fallback_synthesis(topic: str, annotations: list[dict], papers: list[dict], output_language: str = "en") -> dict:
        categories = sorted({category for paper in papers for category in paper.get("categories", [])})
        sources = sorted({str(paper.get("source", "")) for paper in papers if paper.get("source")})
        use_chinese = output_language == "zh" or main_language(topic, default="en") == "zh" or any(
            paper.get("output_language") == "zh" or SLRWorkflow._contains_cjk(str(paper.get("search_query") or ""))
            for paper in papers
        )
        if use_chinese:
            return {
                "executive_summary": (
                    f"本综述检索到 {len(papers)} 条与“{topic}”相关的学术记录。"
                    "当前结果提供了可用的元数据和逐篇摘要级注释，但跨文献主题综合仍建议结合全文或更完整摘要复核。"
                ),
                "themes": [
                    {
                        "name": "基于元数据的文献集合",
                        "summary": f"来源包括：{', '.join(sources) or '未提供来源'}；分类包括：{', '.join(categories[:8]) or '未提供分类'}。",
                        "paper_ids": [str(paper.get("id", "")) for paper in papers[:8]],
                    }
                ],
                "convergences": ["检索结果在标题、摘要或元数据中与用户指定主题存在相关性。"],
                "disagreements": ["在缺少稳定综合结果时，暂不能可靠判断文献之间的明确分歧。"],
                "gaps": ["在提出强结论前，需要进一步核查全文方法、数据集和评价细节。"],
                "methodological_patterns": ["当前主要基于元数据综合，方法细节抽取尚不完整。"],
                "limitations_of_review": "该综合基于可用元数据和摘要生成。",
            }
        return {
            "executive_summary": (
                f"This review retrieved {len(papers)} scholarly records related to “{topic}”. "
                "The current result provides available metadata and per-paper annotations, but cross-paper synthesis still requires fuller abstracts or full texts."
            ),
            "themes": [
                {
                    "name": "Metadata-grounded literature set",
                    "summary": f"Sources include: {', '.join(sources) or 'not provided'}; categories include: {', '.join(categories[:8]) or 'not provided'}.",
                    "paper_ids": [str(paper.get("id", "")) for paper in papers[:8]],
                }
            ],
            "convergences": ["The retrieved records share the user-specified topic terms in title, abstract, or metadata."],
            "disagreements": ["No reliable disagreements can be inferred without successful synthesis."],
            "gaps": ["Full-text methods, datasets, and evaluation details should be checked before making strong claims."],
            "methodological_patterns": ["Metadata-only fallback; methodology extraction was not fully available."],
            "limitations_of_review": "This fallback synthesis is based on available metadata and abstracts.",
        }

    @staticmethod
    def _normalize_synthesis(data: dict, annotations: list[dict], papers: list[dict], output_language: str = "en") -> dict:
        fallback = SLRWorkflow._fallback_synthesis("", annotations, papers, output_language=output_language)
        normalized = dict(fallback)
        normalized.update({key: data.get(key) for key in normalized if data.get(key)})
        if not isinstance(normalized.get("themes"), list):
            normalized["themes"] = fallback["themes"]
        for key in ["convergences", "disagreements", "gaps", "methodological_patterns"]:
            if isinstance(normalized.get(key), str):
                normalized[key] = [normalized[key]]
            elif not isinstance(normalized.get(key), list):
                normalized[key] = fallback[key]
        return normalized

    @staticmethod
    def _align_annotations(annotations: list[dict], papers: list[dict], output_language: str = "en") -> list[dict]:
        by_key = {}
        for item in annotations:
            if not isinstance(item, dict):
                continue
            for key in SLRWorkflow._annotation_keys(item):
                by_key.setdefault(key, item)
        aligned = []
        used_annotation_indexes: set[int] = set()
        for paper_index, paper in enumerate(papers):
            source_id = str(paper.get("id", ""))
            item = {}
            for key in SLRWorkflow._paper_annotation_keys(paper):
                if key in by_key:
                    item = dict(by_key[key])
                    break
            if not item and len(annotations) == len(papers) and paper_index < len(annotations):
                candidate = annotations[paper_index]
                if isinstance(candidate, dict) and paper_index not in used_annotation_indexes:
                    item = dict(candidate)
                    used_annotation_indexes.add(paper_index)
            if not item:
                item = SLRWorkflow._fallback_annotations([paper], output_language=output_language)[0]
            if source_id and SLRWorkflow._normalize_match_key(item.get("arxiv_id", "")) != SLRWorkflow._normalize_match_key(source_id):
                item["arxiv_id"] = source_id
            item.setdefault("arxiv_id", source_id)
            item.setdefault("title", paper.get("title", ""))
            item.setdefault("source", paper.get("source", ""))
            item.setdefault("source_origin", paper.get("source_origin", ""))
            item.setdefault("authors", paper.get("authors", []))
            item.setdefault("published_date", paper.get("published", ""))
            item.setdefault("key_findings", [])
            item.setdefault(
                "innovation_point",
                SLRWorkflow._annotation_innovation(
                    item,
                    item.get("key_findings", []) if isinstance(item.get("key_findings", []), list) else [item.get("key_findings", "")],
                    paper.get("output_language") == "zh" or SLRWorkflow._contains_cjk(str(paper.get("search_query") or "")),
                ),
            )
            item = SLRWorkflow._repair_annotation_fields(
                item,
                paper,
                use_chinese=(
                    output_language == "zh"
                    or paper.get("output_language") == "zh"
                    or SLRWorkflow._contains_cjk(str(paper.get("search_query") or ""))
                ),
            )
            aligned.append(item)
        return aligned

    @staticmethod
    def _annotation_keys(item: dict) -> list[str]:
        keys = []
        for field in ("arxiv_id", "id", "doi", "title"):
            key = SLRWorkflow._normalize_match_key(item.get(field, ""))
            if key:
                keys.append(key)
        return keys

    @staticmethod
    def _paper_annotation_keys(paper: dict) -> list[str]:
        keys = []
        for field in ("id", "doi", "title", "abs_url"):
            key = SLRWorkflow._normalize_match_key(paper.get(field, ""))
            if key:
                keys.append(key)
        return keys

    @staticmethod
    def _normalize_match_key(value) -> str:
        text = SLRWorkflow._compact_text(str(value or "")).casefold()
        text = re.sub(r"^https?://(?:www\.)?", "", text)
        text = re.sub(r"\.pdf$", "", text)
        return text.strip().strip("/")

    @staticmethod
    def _normalize_user_paper(item: dict) -> dict:
        title = str(item.get("title") or item.get("source") or "User provided literature").strip()
        source_url = str(item.get("abs_url") or item.get("source") or "").strip()
        doi = str(item.get("doi") or "").strip()
        if source_url.startswith("https://doi.org/") and not doi:
            doi = source_url.removeprefix("https://doi.org/").strip()
        return {
            "id": str(item.get("id") or doi or source_url or title).strip(),
            "title": title,
            "authors": item.get("authors", []),
            "abstract": str(item.get("abstract") or item.get("content_excerpt") or item.get("relevance") or "").strip(),
            "published": str(item.get("published") or item.get("year") or "").strip(),
            "updated": "",
            "categories": ["User provided"],
            "abs_url": source_url,
            "pdf_url": "",
            "source": str(item.get("source_label") or item.get("source_origin") or "User provided").strip(),
            "source_origin": str(item.get("source_origin") or "user_provided").strip(),
            "doi": doi,
            "journal": str(item.get("journal") or "").strip(),
            "content_excerpt": str(item.get("content_excerpt") or "").strip(),
            "pdf_text_available": bool(item.get("pdf_text_available", False)),
            "document_role": str(item.get("document_role") or "literature").strip(),
            "is_literature_source": bool(item.get("is_literature_source", True)),
        }

    @staticmethod
    def _is_literature_user_paper(item: dict) -> bool:
        role = str(item.get("document_role") or "literature").strip().lower()
        return role == "literature" and bool(item.get("is_literature_source", True))

    @staticmethod
    def _append_user_context_documents(topic: str, documents: list[dict]) -> str:
        sections = [
            topic,
            "",
            "User-uploaded auxiliary documents are provided below. They may be writing requirements, rubrics, assignment prompts, or style constraints.",
            "Apply these documents as review instructions and constraints. Do not count them as surveyed papers unless they clearly contain substantive research evidence.",
        ]
        for index, document in enumerate(documents, start=1):
            title = document.get("title") or document.get("source") or f"Uploaded document {index}"
            excerpt = SLRWorkflow._high_information_excerpt(
                document.get("content_excerpt") or document.get("abstract") or "",
                max_chars=CONTEXT_DOCUMENT_EXCERPT_CHARS,
            )
            sections.append(
                f"Auxiliary document {index}: {title}\n"
                f"Role: {document.get('document_role', 'unknown')}\n"
                f"Excerpt:\n{excerpt}"
            )
        return "\n\n".join(section for section in sections if section).strip()

    @staticmethod
    def _dedupe_and_rank(papers: list[dict], *, max_results: int) -> list[dict]:
        priority = {
            "User provided": 0,
            "User link": 0,
            "user_provided": 0,
            "user_link": 0,
            "user_upload": 0,
            "arXiv": 1,
            "OpenAlex": 2,
            "PubMed": 3,
            "Crossref": 4,
        }
        deduped: dict[str, dict] = {}
        order: dict[str, int] = {}
        for index, paper in enumerate(papers):
            key = SLRWorkflow._paper_key(paper)
            if not key:
                continue
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = paper
                order[key] = index
                continue
            if SLRWorkflow._paper_quality_score(paper) > SLRWorkflow._paper_quality_score(existing):
                merged = dict(existing)
                merged.update({field: value for field, value in paper.items() if value})
                deduped[key] = merged
        grouped: dict[str, list[dict]] = {}
        for paper in sorted(
            deduped.values(),
            key=lambda item: order.get(SLRWorkflow._paper_key(item), 999999),
        ):
            grouped.setdefault(str(paper.get("source", "Unknown")), []).append(paper)

        source_order = sorted(grouped, key=lambda source: priority.get(source, 9))
        selected: list[dict] = []
        while len(selected) < max_results and any(grouped.get(source) for source in source_order):
            for source in source_order:
                if grouped.get(source):
                    selected.append(grouped[source].pop(0))
                    if len(selected) >= max_results:
                        break
        return selected

    @staticmethod
    def _paper_key(paper: dict) -> str:
        doi = str(paper.get("doi") or "").lower().strip()
        if doi:
            return f"doi:{doi}"
        url = str(paper.get("abs_url") or "").lower().strip()
        if url:
            return f"url:{url}"
        title = re.sub(r"\W+", "", str(paper.get("title", "")).casefold())
        return f"title:{title}" if title else ""

    @staticmethod
    def _paper_quality_score(paper: dict) -> int:
        return sum(1 for key in ["abstract", "doi", "authors", "journal", "published", "abs_url"] if paper.get(key))

    @staticmethod
    def _format_search_source(source_counts: dict[str, int], source_errors: dict[str, str], source_mode: str) -> str:
        active = [f"{source} ({count})" for source, count in source_counts.items() if count]
        if not active:
            return f"{source_mode}; no successful source"
        if source_errors:
            return f"{source_mode}; {', '.join(active)}; unavailable: {', '.join(source_errors)}"
        return f"{source_mode}; {', '.join(active)}"

    @staticmethod
    def _format_source_counts(source_counts: dict[str, int]) -> str:
        if not source_counts:
            return "none"
        return "; ".join(f"{source}: {count}" for source, count in source_counts.items())

    @staticmethod
    def _render_bullets(items: list | str) -> str:
        if isinstance(items, str):
            items = [items] if items.strip() else []
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        return "\n".join(f"- {item}" for item in cleaned) if cleaned else "- No clear pattern identified."

    @staticmethod
    def _render_references(references: list[str], citation_format: str) -> str:
        if citation_format == "bibtex":
            return "\n\n".join(f"```bibtex\n{reference}\n```" for reference in references)
        return "\n\n".join(references)

    @staticmethod
    def _parse_json(content: str):
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        return json.loads(cleaned)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[slr] {message}", flush=True)


def normalize_source_mode(value: str | None) -> str:
    mode = (value or "search").strip().lower()
    if mode in {"upload", "uploaded", "user"}:
        return "upload"
    if mode in {"mixed", "both", "upload_search", "upload+search"}:
        return "mixed"
    return "search"
