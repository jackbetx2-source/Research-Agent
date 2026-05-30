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
from .llm import LLMClient, LLMServiceError
from .openalex_search import search_openalex
from .pubmed_search import search_pubmed


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
        annotations = await self._extract_annotations(papers)

        self._log("3/4 Synthesizing cross-paper themes...")
        synthesis = await self._synthesize(topic, annotations, papers)

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
        if self._contains_cjk(topic):
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
                "target_ratio": 0.8,
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
        text = " ".join(
            str(paper.get(key) or "")
            for key in ("title", "abstract", "journal")
        )
        if SLRWorkflow._contains_cjk(text):
            return "zh"
        ascii_letters = len(re.findall(r"[A-Za-z]", text))
        return "en" if ascii_letters >= 12 else "other"

    @staticmethod
    def _contains_cjk(value: str) -> bool:
        return bool(re.search(r"[\u3400-\u9fff]", value or ""))

    async def _extract_annotations(self, papers: list[dict]) -> list[dict]:
        batches = [papers[index : index + 5] for index in range(0, len(papers), 5)]
        all_annotations: list[dict] = []
        for start in range(0, len(batches), 3):
            round_batches = batches[start : start + 3]
            results = await asyncio.gather(
                *[self._extract_batch(batch) for batch in round_batches],
                return_exceptions=True,
            )
            for batch, result in zip(round_batches, results):
                if isinstance(result, Exception):
                    self._log(f"Annotation batch failed; using fallback rows. {type(result).__name__}: {result}")
                    all_annotations.extend(self._fallback_annotations(batch))
                    continue
                all_annotations.extend(result)
        return self._align_annotations(all_annotations, papers)

    async def _extract_batch(self, papers: list[dict]) -> list[dict]:
        use_chinese = any(
            paper.get("output_language") == "zh" or self._contains_cjk(str(paper.get("search_query") or ""))
            for paper in papers
        )
        system_prompt = """
You extract structured metadata for a systematic literature review.
Return only a valid JSON array. Do not use markdown fences.
For each scholarly record, infer only from the provided title, abstract, and metadata.
Do not invent experiments, datasets, or results not present in the supplied context.
""".strip()
        user_prompt = (
            "Papers:\n"
            + json.dumps(papers, ensure_ascii=False, indent=2)
            + """

For each paper return:
{
  "arxiv_id": "source id, DOI, PMID, or arXiv id",
  "title": "...",
  "source": "arXiv / OpenAlex / PubMed / Crossref",
  "authors": ["..."],
  "published_date": "YYYY-MM-DD",
  "research_question": "one sentence",
  "methodology": "one or two sentences",
  "key_findings": ["3 to 5 concise bullets"],
  "limitations": "one or two sentences",
  "evidence_type": "Theoretical / Empirical / Benchmark / System / Survey / Unknown"
}
""".strip()
        )
        if use_chinese:
            user_prompt += "\n\nOutput language: Chinese. Translate analytical fields into Chinese; keep paper titles unchanged."
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("SLR_MODEL") or os.getenv("RESEARCH_MODEL"),
                temperature=0.15,
                max_tokens=3200,
            )
            data = self._parse_json(content)
            if not isinstance(data, list):
                raise ValueError("Expected JSON array.")
            return [item for item in data if isinstance(item, dict)]
        except (LLMServiceError, ValueError, TypeError, json.JSONDecodeError):
            return self._fallback_annotations(papers)

    async def _synthesize(self, topic: str, annotations: list[dict], papers: list[dict]) -> dict:
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
Use concise Chinese unless the topic is explicitly English-only.
If the topic is mainly Chinese, all narrative values must be Chinese. Keep paper titles, author names, DOIs, and source names unchanged.
""".strip()
        user_prompt = (
            f"Topic:\n{topic}\n\n"
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
            return self._normalize_synthesis(data, annotations, papers)
        except (LLMServiceError, ValueError, TypeError, json.JSONDecodeError):
            return self._fallback_synthesis(topic, annotations, papers)

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
    ) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        use_chinese = self._contains_cjk(topic)
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
                "## 研究空白与开放问题",
                self._render_bullets(synthesis.get("gaps", [])),
                "## 方法模式",
                self._render_bullets(synthesis.get("methodological_patterns", [])),
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
            "## Gaps and Open Questions",
            self._render_bullets(synthesis.get("gaps", [])),
            "## Methodological Patterns",
            self._render_bullets(synthesis.get("methodological_patterns", [])),
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
        blocks = []
        for annotation in sorted(annotations, key=lambda item: (year_from_paper(item), str(item.get("title", "")))):
            findings = annotation.get("key_findings", [])
            if not isinstance(findings, list):
                findings = [str(findings)]
            if use_chinese:
                blocks.append(
                    f"### {annotation.get('title', '未命名论文')}\n\n"
                    f"**来源**：{annotation.get('source', '未知')}\n\n"
                    f"**来源 ID**：{annotation.get('arxiv_id', '')}\n\n"
                    f"**研究问题**：{annotation.get('research_question', '')}\n\n"
                    f"**方法**：{annotation.get('methodology', '')}\n\n"
                    f"**关键发现**：\n{self._render_bullets(findings)}\n\n"
                    f"**局限**：{annotation.get('limitations', '')}\n\n"
                    f"**证据类型**：{annotation.get('evidence_type', '未知')}"
                )
            else:
                blocks.append(
                    f"### {annotation.get('title', 'Untitled paper')}\n\n"
                    f"**Source**: {annotation.get('source', 'Unknown')}\n\n"
                    f"**Source ID**: {annotation.get('arxiv_id', '')}\n\n"
                    f"**Research question**: {annotation.get('research_question', '')}\n\n"
                    f"**Methodology**: {annotation.get('methodology', '')}\n\n"
                    f"**Key findings**:\n{self._render_bullets(findings)}\n\n"
                    f"**Limitations**: {annotation.get('limitations', '')}\n\n"
                    f"**Evidence type**: {annotation.get('evidence_type', 'Unknown')}"
                )
        return "\n\n".join(blocks)

    @staticmethod
    def _fallback_annotations(papers: list[dict]) -> list[dict]:
        annotations = []
        use_chinese = any(
            paper.get("output_language") == "zh" or SLRWorkflow._contains_cjk(str(paper.get("search_query") or ""))
            for paper in papers
        )
        for paper in papers:
            abstract = str(paper.get("abstract", "") or "")
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
                    "key_findings": ["当前只能基于元数据生成兜底注释，建议进一步查看原文。"] if use_chinese else ["Abstract-based extraction is unavailable; inspect the paper directly."],
                    "limitations": "该注释仅基于可用元数据生成。" if use_chinese else "Fallback annotation based on available metadata only.",
                    "evidence_type": "未知" if use_chinese else "Unknown",
                }
            )
        return annotations

    @staticmethod
    def _fallback_synthesis(topic: str, annotations: list[dict], papers: list[dict]) -> dict:
        categories = sorted({category for paper in papers for category in paper.get("categories", [])})
        sources = sorted({str(paper.get("source", "")) for paper in papers if paper.get("source")})
        use_chinese = SLRWorkflow._contains_cjk(topic) or any(
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
                "methodological_patterns": ["当前为元数据兜底综合，方法细节抽取尚不完整。"],
                "limitations_of_review": "该兜底综合基于可用元数据和摘要生成。",
            }
        return {
            "executive_summary": (
                f"本综述检索到 {len(papers)} 条与“{topic}”相关的学术记录。"
                "当前结果提供了真实元数据和逐篇摘要级注释，但跨文献主题综合仍需要全文或更完整摘要支持。"
            ),
            "themes": [
                {
                    "name": "Metadata-grounded literature set",
                    "summary": f"来源包括：{', '.join(sources) or '未提供来源'}；分类包括：{', '.join(categories[:8]) or '未提供分类'}。",
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
    def _normalize_synthesis(data: dict, annotations: list[dict], papers: list[dict]) -> dict:
        fallback = SLRWorkflow._fallback_synthesis("", annotations, papers)
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
    def _align_annotations(annotations: list[dict], papers: list[dict]) -> list[dict]:
        by_id = {str(item.get("arxiv_id") or item.get("id")): item for item in annotations if isinstance(item, dict)}
        aligned = []
        for paper in papers:
            source_id = str(paper.get("id", ""))
            item = dict(by_id.get(source_id) or {})
            if not item:
                item = SLRWorkflow._fallback_annotations([paper])[0]
            item.setdefault("arxiv_id", source_id)
            item.setdefault("title", paper.get("title", ""))
            item.setdefault("source", paper.get("source", ""))
            item.setdefault("source_origin", paper.get("source_origin", ""))
            item.setdefault("authors", paper.get("authors", []))
            item.setdefault("published_date", paper.get("published", ""))
            item.setdefault("key_findings", [])
            aligned.append(item)
        return aligned

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
            excerpt = (document.get("content_excerpt") or document.get("abstract") or "")[:2500]
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
