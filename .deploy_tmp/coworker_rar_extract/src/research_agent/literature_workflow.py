from __future__ import annotations

import asyncio
import json
import os

from .language_policy import main_language
from .llm import LLMClient, LLMServiceError


ANALYSIS_COLUMNS = [
    "title",
    "source",
    "metadata",
    "paper_type",
    "core_claim",
    "contribution",
    "methodology",
    "evidence_strength",
    "strengths",
    "weaknesses",
    "reproducibility",
    "literature_positioning",
    "application",
    "limitations",
    "questions",
    "actionable_suggestions",
    "confidence",
    "overall_assessment",
    # Backward-compatible aliases used by the current web UI.
    "innovation",
    "method",
    "limitation",
    "next_step",
]


ACADEMIC_REVIEW_RUBRIC = """
Use peer-review standards inspired by the academic-paper-review skill:
- identify the paper's core claim and contribution
- assess methodology soundness, evidence strength, and reproducibility
- separate strengths from weaknesses with specific reasoning
- position the work relative to related literature when context is available
- provide constructive, actionable suggestions
- state uncertainty instead of inventing details when only DOI/metadata is available
""".strip()


REVIEW_ANALYST_GROUPS = [
    {
        "name": "Contribution Analyst",
        "focus": "core claim, contribution, novelty, significance, and literature positioning",
    },
    {
        "name": "Methodology Analyst",
        "focus": "method soundness, experimental design, statistics, and reproducibility",
    },
    {
        "name": "Evidence Analyst",
        "focus": "claim-evidence alignment, result support, and evidence strength",
    },
    {
        "name": "Critical Reviewer",
        "focus": "weaknesses, limitations, risks, author questions, and actionable improvements",
    },
]


class LiteratureAnalysisWorkflow:
    def __init__(self, llm: LLMClient | None = None, verbose: bool = False) -> None:
        self.llm = llm or LLMClient()
        self.verbose = verbose

    async def run(
        self,
        *,
        topic: str,
        references: list[dict],
        final_report: str = "",
    ) -> dict:
        normalized_references = self._normalize_references(references)
        clean_references = [
            reference
            for reference in normalized_references
            if self._is_literature_reference(reference)
        ]
        output_language = self._detect_output_language(topic, final_report, normalized_references)
        if not clean_references:
            return {"rows": [], "summary": self._context_only_summary(final_report, output_language)}

        self._log("1/3 Assigning literature to four analysts...")
        groups = await self._assign_references(topic, clean_references, final_report, output_language)
        self._log("2/3 Running four literature analysts in parallel...")
        analyst_outputs = await self._run_parallel_analysis(topic, groups, final_report, output_language)
        self._log("3/3 Integrating literature analysis rows and cross-paper summary...")
        result = await self._integrate(topic, clean_references, analyst_outputs, final_report, output_language)
        self._log("Done.")
        return result

    async def _assign_references(
        self, topic: str, references: list[dict], final_report: str, output_language: str
    ) -> list[dict]:
        system_prompt = """
You are LLM1, the coordinator of a literature-analysis workflow.
You will receive literature references from a research report.
Assign the references to exactly four parallel analyst groups.
Balance the workload and group related sources by contribution, methodology, evidence, and risk.
Return only valid JSON with this schema:
{
  "groups": [
    {"name": "Contribution Analyst", "focus": "...", "reference_indices": [0, 1]},
    {"name": "Methodology Analyst", "focus": "...", "reference_indices": []},
    {"name": "Evidence Analyst", "focus": "...", "reference_indices": []},
    {"name": "Critical Reviewer", "focus": "...", "reference_indices": []}
  ]
}
Use zero-based indices from the provided references array.
""".strip()
        user_prompt = self._analysis_context(topic, references, final_report, output_language)
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("ANALYSIS_MODEL") or os.getenv("RESEARCH_MODEL"),
                temperature=0.1,
                max_tokens=900,
            )
            data = self._parse_json_object(content)
            groups = self._normalize_groups(data.get("groups"), references)
            if groups:
                return groups
        except (LLMServiceError, ValueError, KeyError, TypeError):
            self._log("Assignment failed; falling back to deterministic round-robin.")
        return self._round_robin_groups(references)

    async def _run_parallel_analysis(
        self, topic: str, groups: list[dict], final_report: str, output_language: str
    ) -> list[dict]:
        tasks = [self._run_analyst(topic, group, final_report, output_language) for group in groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        analyst_outputs = []
        for group, result in zip(groups, results):
            if isinstance(result, Exception):
                self._log(f"{group['name']} failed; continuing with other analysts.")
                analyst_outputs.append(
                    {
                        "analyst": group["name"],
                        "focus": group["focus"],
                        "rows": [],
                        "error": f"{type(result).__name__}: {result}",
                    }
                )
                continue
            analyst_outputs.append(result)
        return analyst_outputs

    async def _run_analyst(self, topic: str, group: dict, final_report: str, output_language: str) -> dict:
        language_instruction = self._language_instruction(output_language)
        system_prompt = f"""
You are {group["name"]}, one of four parallel literature analysts.
Your focus: {group["focus"]}.

Analyze only the references assigned to you.
For each reference, apply this rubric:
{ACADEMIC_REVIEW_RUBRIC}

Return only valid JSON with this schema:
{{
  "analyst": "{group["name"]}",
  "rows": [
    {{
      "title": "...",
      "source": "...",
      "metadata": "authors, venue/journal, year, domain if available",
      "paper_type": "Empirical / Theoretical / Survey / Systems / Position / Unknown",
      "core_claim": "...",
      "contribution": "...",
      "methodology": "...",
      "evidence_strength": "Strong / Moderate / Weak / Unknown, with short reason",
      "strengths": "...",
      "weaknesses": "...",
      "reproducibility": "...",
      "literature_positioning": "...",
      "application": "...",
      "limitations": "...",
      "questions": "...",
      "actionable_suggestions": "...",
      "confidence": "High / Medium / Low, with short reason",
      "overall_assessment": "Landmark / Significant / Moderate / Marginal / Below threshold / Unknown"
    }}
  ]
}}
{language_instruction}
Preserve the original title and source URL when provided.
If a reference only provides a DOI and you cannot infer reliable bibliographic metadata,
use the DOI as the title and explicitly state that metadata/full text should be checked.
Do not fabricate experiments, datasets, results, or claims that are absent from the supplied metadata,
abstract, PDF excerpt, or report context.
When pdf_text_available is true, ground the analysis in content_excerpt and mention uncertainty if the
excerpt is incomplete. When pdf_text_available is false, treat the row as metadata-only and set low
confidence unless the report context supplies reliable details.
""".strip()
        user_prompt = (
            f"Research topic:\n{topic}\n\n"
            f"Assigned references:\n{json.dumps(group['references'], ensure_ascii=False, indent=2)}\n\n"
            f"Report context excerpt:\n{final_report[:4000]}"
        )
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("ANALYSIS_MODEL") or os.getenv("RESEARCH_MODEL"),
                temperature=0.2,
                max_tokens=2200,
            )
            data = self._parse_json_object(content)
            rows = data.get("rows", [])
        except (LLMServiceError, ValueError, KeyError, TypeError) as error:
            self._log(f"{group['name']} output failed; using reference-level fallback.")
            rows = self._fallback_rows_from_references(
                group["references"],
                fallback_note=f"{group['name']} failed: {type(error).__name__}",
                output_language=output_language,
            )
        return {
            "analyst": group["name"],
            "focus": group["focus"],
            "rows": rows if isinstance(rows, list) else [],
        }

    async def _integrate(
        self,
        topic: str,
        references: list[dict],
        analyst_outputs: list[dict],
        final_report: str,
        output_language: str,
    ) -> dict:
        language_instruction = self._language_instruction(output_language)
        system_prompt = """
You are LLM1, the final integrator of a literature-analysis workflow.
You will receive the original references plus four analyst outputs.
Create final table rows and a cross-literature review summary for researchers.

Requirements:
- Keep one row per important reference.
- Merge duplicate rows for the same reference.
- Preserve source URLs.
- __LANGUAGE_INSTRUCTION__
- Do not invent unsupported details; when uncertain, state a careful limitation.
- DOI-only inputs may lack title/abstract/full text. Preserve the DOI/source and state uncertainty instead of inventing.
- Preserve peer-review detail from the analyst outputs while keeping each cell short.
- Also fill the backward-compatible alias fields: innovation, method, limitation, next_step.
- The summary must synthesize across references; do not simply restate each row.
- Return only valid JSON with this schema:
{
  "rows": [
    {
      "title": "...",
      "source": "...",
      "metadata": "authors, venue/journal, year, domain if available",
      "paper_type": "Empirical / Theoretical / Survey / Systems / Position / Unknown",
      "core_claim": "...",
      "contribution": "...",
      "methodology": "...",
      "evidence_strength": "Strong / Moderate / Weak / Unknown, with short reason",
      "strengths": "...",
      "weaknesses": "...",
      "reproducibility": "...",
      "literature_positioning": "...",
      "application": "...",
      "limitations": "...",
      "questions": "...",
      "actionable_suggestions": "...",
      "confidence": "High / Medium / Low, with short reason",
      "overall_assessment": "Landmark / Significant / Moderate / Marginal / Below threshold / Unknown",
      "innovation": "short alias for contribution",
      "method": "short alias for methodology/evidence",
      "limitation": "short alias for limitations/weaknesses",
      "next_step": "short alias for actionable_suggestions"
    }
  ],
  "summary": {
    "overall_assessment": "one concise cross-literature assessment",
    "common_strengths": ["..."],
    "common_weaknesses": ["..."],
    "methodological_patterns": ["..."],
    "evidence_gaps": ["..."],
    "research_gaps": ["..."],
    "recommended_reading_order": ["title or source with short reason"],
    "next_actions": ["..."],
    "confidence": "High / Medium / Low, with short reason"
  }
}
""".strip().replace("__LANGUAGE_INSTRUCTION__", language_instruction)
        user_prompt = (
            f"Research topic:\n{topic}\n\n"
            f"Original references:\n{json.dumps(references, ensure_ascii=False, indent=2)}\n\n"
            "Parallel analyst outputs:\n"
            + json.dumps(analyst_outputs, ensure_ascii=False, indent=2)
            + f"\n\nReport context excerpt:\n{final_report[:2500]}"
        )
        try:
            content = await self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=os.getenv("ANALYSIS_MODEL") or os.getenv("SYNTHESIS_MODEL"),
                temperature=0.15,
                max_tokens=3600,
            )
            data = self._parse_json_object(content)
            rows = data.get("rows", [])
            if not isinstance(rows, list):
                raise ValueError("Expected rows list from literature analysis integrator.")
            normalized = self._normalize_rows(rows, references, output_language)
            if normalized:
                summary = self._normalize_summary(data.get("summary"), normalized, references, output_language)
                return {"rows": normalized, "summary": summary}
        except (LLMServiceError, ValueError, KeyError, TypeError) as error:
            self._log(f"Integration failed; using analyst-output fallback. {type(error).__name__}: {error}")

        fallback_rows = [
            row
            for output in analyst_outputs
            if isinstance(output, dict)
            for row in output.get("rows", [])
            if isinstance(row, dict)
        ]
        normalized = self._normalize_rows(fallback_rows, references, output_language)
        if normalized:
            return {
                "rows": normalized,
                "summary": self._fallback_summary(normalized, "LLM integration unavailable; summary derived from analyst outputs.", output_language),
            }
        fallback_rows = self._fallback_rows_from_references(
            references,
            fallback_note="LLM integration unavailable; this row only reflects supplied metadata.",
            output_language=output_language,
        )
        return {
            "rows": fallback_rows,
            "summary": self._fallback_summary(fallback_rows, "LLM integration unavailable; summary derived from supplied metadata only.", output_language),
        }

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[literature] {message}", flush=True)

    @staticmethod
    def _analysis_context(topic: str, references: list[dict], final_report: str, output_language: str) -> str:
        return (
            f"Research topic:\n{topic}\n\n"
            f"Output language: {output_language}\n\n"
            f"References:\n{json.dumps(references, ensure_ascii=False, indent=2)}\n\n"
            f"Report context excerpt:\n{final_report[:2500]}"
        )

    @staticmethod
    def _detect_output_language(topic: str, final_report: str, references: list[dict]) -> str:
        reference_text = "\n".join(
            " ".join(
                str(reference.get(key) or "")
                for key in ("relevance", "branch_name", "abstract", "content_excerpt")
            )
            for reference in references[:8]
        )
        signal = "\n\n".join(part for part in [topic, final_report[:2500], reference_text] if part)
        return main_language(signal, default="zh")

    @staticmethod
    def _language_instruction(output_language: str) -> str:
        if output_language == "zh":
            return "Write all analytical table-cell text and summary values in concise Chinese. Keep original paper titles, author names, DOI/PMID/arXiv IDs, and URLs unchanged."
        return "Write all analytical table-cell text and summary values in concise English. Keep original paper titles, author names, DOI/PMID/arXiv IDs, and URLs unchanged."

    @staticmethod
    def _normalize_references(references: list[dict]) -> list[dict]:
        normalized = []
        for index, item in enumerate(references):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            normalized.append(
                {
                    "index": index,
                    "title": title,
                    "source": str(item.get("source", "") or "").strip(),
                    "relevance": str(item.get("relevance", "") or "").strip(),
                    "branch_name": str(item.get("branch_name", "") or "").strip(),
                    "doi": str(item.get("doi", "") or "").strip(),
                    "authors": str(item.get("authors", "") or "").strip(),
                    "year": str(item.get("year", "") or "").strip(),
                    "journal": str(item.get("journal", "") or "").strip(),
                    "abstract": str(item.get("abstract", "") or "").strip(),
                    "content_excerpt": str(item.get("content_excerpt", "") or item.get("abstract", "") or "").strip(),
                    "pdf_text_available": bool(item.get("pdf_text_available", False)),
                    "pdf_page_count": str(item.get("pdf_page_count", "") or "").strip(),
                    "pdf_extracted_pages": str(item.get("pdf_extracted_pages", "") or "").strip(),
                    "pdf_extraction_note": str(item.get("pdf_extraction_note", "") or "").strip(),
                    "document_role": str(item.get("document_role", "literature") or "literature").strip(),
                    "is_literature_source": bool(item.get("is_literature_source", True)),
                }
            )
        return normalized

    @staticmethod
    def _is_literature_reference(reference: dict) -> bool:
        role = str(reference.get("document_role") or "literature").strip().lower()
        return role == "literature" and bool(reference.get("is_literature_source", True))

    @staticmethod
    def _normalize_groups(groups, references: list[dict]) -> list[dict]:
        if not isinstance(groups, list):
            return []

        normalized = []
        used_names = set()
        for fallback_index, group in enumerate(groups[:4]):
            if not isinstance(group, dict):
                continue
            name = str(group.get("name") or f"Analyst {fallback_index + 1}").strip()
            if name in used_names:
                name = f"{name} {fallback_index + 1}"
            used_names.add(name)
            focus = str(group.get("focus") or "literature contribution and research value").strip()
            indices = group.get("reference_indices", [])
            if not isinstance(indices, list):
                indices = []
            items = [
                references[index]
                for index in indices
                if isinstance(index, int) and 0 <= index < len(references)
            ]
            normalized.append({"name": name, "focus": focus, "references": items})

        while len(normalized) < 4:
            fallback_group = REVIEW_ANALYST_GROUPS[len(normalized)]
            normalized.append(
                {
                    "name": fallback_group["name"],
                    "focus": fallback_group["focus"],
                    "references": [],
                }
            )

        assigned = {
            reference["index"]
            for group in normalized
            for reference in group["references"]
            if "index" in reference
        }
        missing = [reference for reference in references if reference["index"] not in assigned]
        for offset, reference in enumerate(missing):
            normalized[offset % 4]["references"].append(reference)
        return normalized

    @staticmethod
    def _round_robin_groups(references: list[dict]) -> list[dict]:
        groups = [{**group, "references": []} for group in REVIEW_ANALYST_GROUPS]
        for index, reference in enumerate(references):
            groups[index % 4]["references"].append(reference)
        return groups

    @staticmethod
    def _normalize_rows(rows: list, references: list[dict], output_language: str = "zh") -> list[dict]:
        by_title = {item["title"].casefold(): item for item in references}
        by_source = {
            item["source"].casefold(): item
            for item in references
            if item.get("source")
        }
        normalized = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title", "")).strip()
            source = str(row.get("source", "")).strip()
            if not title and source:
                original_by_source = by_source.get(source.casefold(), {})
                title = str(original_by_source.get("title", "")).strip()
            if not title:
                continue
            key = (source or title).casefold()
            if key in seen:
                continue
            seen.add(key)
            original = by_title.get(title.casefold(), {}) or by_source.get(source.casefold(), {})
            normalized_row = {
                column: str(row.get(column) or original.get(column) or "").strip()
                for column in ANALYSIS_COLUMNS
            }
            LiteratureAnalysisWorkflow._fill_review_defaults(normalized_row, original, output_language)
            LiteratureAnalysisWorkflow._fill_legacy_aliases(normalized_row)
            normalized.append(normalized_row)
        return normalized

    @staticmethod
    def _normalize_summary(summary, rows: list[dict], references: list[dict], output_language: str = "zh") -> dict:
        if not isinstance(summary, dict):
            return LiteratureAnalysisWorkflow._fallback_summary(rows, "Integrator did not return a valid summary.", output_language)

        normalized = LiteratureAnalysisWorkflow._empty_summary()
        normalized["overall_assessment"] = str(
            summary.get("overall_assessment") or ""
        ).strip()
        normalized["confidence"] = str(summary.get("confidence") or "").strip()

        for key in [
            "common_strengths",
            "common_weaknesses",
            "methodological_patterns",
            "evidence_gaps",
            "research_gaps",
            "recommended_reading_order",
            "next_actions",
        ]:
            normalized[key] = LiteratureAnalysisWorkflow._normalize_string_list(summary.get(key))

        if not normalized["overall_assessment"]:
            normalized["overall_assessment"] = (
                f"已整合 {len(rows)} 篇/项文献；具体结论仍需结合全文和证据质量继续核验。"
                if output_language == "zh"
                else f"Integrated {len(rows)} literature item(s); conclusions should still be checked against full texts and evidence quality."
            )
        if not normalized["recommended_reading_order"]:
            normalized["recommended_reading_order"] = [
                (
                    f"{row.get('title') or row.get('source')}：优先核验核心贡献与证据"
                    if output_language == "zh"
                    else f"{row.get('title') or row.get('source')}: prioritize checking the core contribution and evidence"
                )
                for row in rows[: min(5, len(rows))]
                if row.get("title") or row.get("source")
            ]
        if not normalized["confidence"]:
            has_pdf_text = any(reference.get("pdf_text_available") for reference in references)
            if output_language == "zh":
                normalized["confidence"] = "中等；包含部分 PDF 提取文本，但可能未覆盖全文。" if has_pdf_text else "较低；主要基于元数据、摘要、链接或报告上下文。"
            else:
                normalized["confidence"] = "Medium; includes extracted PDF text but may not cover full papers." if has_pdf_text else "Low; based on metadata, abstracts, links, or report context."
        return normalized

        if not normalized["overall_assessment"]:
            normalized["overall_assessment"] = (
                f"已整合 {len(rows)} 篇/项文献；具体结论需结合全文和证据质量继续核验。"
            )
        if not normalized["recommended_reading_order"]:
            normalized["recommended_reading_order"] = [
                f"{row.get('title') or row.get('source')}：优先核验核心贡献与证据"
                for row in rows[: min(5, len(rows))]
                if row.get("title") or row.get("source")
            ]
        if not normalized["confidence"]:
            has_pdf_text = any(reference.get("pdf_text_available") for reference in references)
            normalized["confidence"] = (
                "Medium; includes extracted PDF text but may not cover full papers."
                if has_pdf_text
                else "Low; based on metadata, abstracts, links, or report context."
            )
        return normalized

    @staticmethod
    def _fallback_summary(rows: list[dict], note: str, output_language: str = "zh") -> dict:
        if output_language == "en":
            summary = LiteratureAnalysisWorkflow._empty_summary()
            summary["overall_assessment"] = f"Generated a structured review table for {len(rows)} literature item(s). {note}"
            summary["common_strengths"] = LiteratureAnalysisWorkflow._collect_unique(rows, "strengths", limit=3)
            summary["common_weaknesses"] = LiteratureAnalysisWorkflow._collect_unique(rows, "weaknesses", limit=3)
            summary["methodological_patterns"] = LiteratureAnalysisWorkflow._collect_unique(rows, "methodology", limit=3)
            summary["evidence_gaps"] = LiteratureAnalysisWorkflow._collect_unique(rows, "evidence_strength", limit=3)
            summary["research_gaps"] = LiteratureAnalysisWorkflow._collect_unique(rows, "limitations", limit=3)
            summary["recommended_reading_order"] = [
                f"{row.get('title') or row.get('source')}: check the full-text evidence and methods first"
                for row in rows[: min(5, len(rows))]
                if row.get("title") or row.get("source")
            ]
            summary["next_actions"] = [
                "Add full-text PDFs or reliable abstracts/metadata first.",
                "Verify low-confidence items against methods, datasets, baselines, and statistical evidence.",
                "Run a cross-literature comparison to separate consensus, disagreement, and research gaps.",
            ]
            summary["confidence"] = "Low; fallback summary generated without a successful integration pass."
            return summary
        summary = LiteratureAnalysisWorkflow._empty_summary()
        summary["overall_assessment"] = f"已生成 {len(rows)} 篇/项文献的结构化评审表。{note}"
        summary["common_strengths"] = LiteratureAnalysisWorkflow._collect_unique(rows, "strengths", limit=3)
        summary["common_weaknesses"] = LiteratureAnalysisWorkflow._collect_unique(rows, "weaknesses", limit=3)
        summary["methodological_patterns"] = LiteratureAnalysisWorkflow._collect_unique(rows, "methodology", limit=3)
        summary["evidence_gaps"] = LiteratureAnalysisWorkflow._collect_unique(rows, "evidence_strength", limit=3)
        summary["research_gaps"] = LiteratureAnalysisWorkflow._collect_unique(rows, "limitations", limit=3)
        summary["recommended_reading_order"] = [
            f"{row.get('title') or row.get('source')}：先核验全文证据和方法细节"
            for row in rows[: min(5, len(rows))]
            if row.get("title") or row.get("source")
        ]
        summary["next_actions"] = [
            "优先补充全文 PDF 或可靠的摘要/元数据。",
            "对低置信度条目核验实验设置、数据集、baseline 和统计显著性。",
            "再做一次跨文献对比，区分共识、分歧和真正的研究空白。",
        ]
        summary["confidence"] = "较低；该兜底总结是在集成步骤失败时生成的。"
        return summary
        summary = LiteratureAnalysisWorkflow._empty_summary()
        summary["overall_assessment"] = (
            f"已生成 {len(rows)} 篇/项文献的结构化评审表；{note}"
        )
        summary["common_strengths"] = LiteratureAnalysisWorkflow._collect_unique(rows, "strengths", limit=3)
        summary["common_weaknesses"] = LiteratureAnalysisWorkflow._collect_unique(rows, "weaknesses", limit=3)
        summary["methodological_patterns"] = LiteratureAnalysisWorkflow._collect_unique(rows, "methodology", limit=3)
        summary["evidence_gaps"] = LiteratureAnalysisWorkflow._collect_unique(rows, "evidence_strength", limit=3)
        summary["research_gaps"] = LiteratureAnalysisWorkflow._collect_unique(rows, "limitations", limit=3)
        summary["recommended_reading_order"] = [
            f"{row.get('title') or row.get('source')}：先核验全文证据和方法细节"
            for row in rows[: min(5, len(rows))]
            if row.get("title") or row.get("source")
        ]
        summary["next_actions"] = [
            "优先补充全文 PDF 或可验证的摘要/元数据。",
            "对低置信度条目核验实验设置、数据集、baseline 和统计显著性。",
            "再做一次跨文献对比，区分共识、分歧和真正研究空白。",
        ]
        summary["confidence"] = "Low; fallback summary generated without a successful integration pass."
        return summary

    @staticmethod
    def _empty_summary() -> dict:
        return {
            "overall_assessment": "",
            "common_strengths": [],
            "common_weaknesses": [],
            "methodological_patterns": [],
            "evidence_gaps": [],
            "research_gaps": [],
            "recommended_reading_order": [],
            "next_actions": [],
            "confidence": "",
        }

    @staticmethod
    def _context_only_summary(final_report: str, output_language: str = "zh") -> dict:
        if output_language == "en":
            summary = LiteratureAnalysisWorkflow._empty_summary()
            has_context = bool(final_report.strip())
            summary["overall_assessment"] = (
                "No uploaded content was identified as analyzable literature; the material is better used as writing requirements, grading criteria, task instructions, or auxiliary context."
                if has_context
                else "No analyzable paper or literature item was provided."
            )
            summary["research_gaps"] = [
                "Add DOI identifiers, paper links, full-text PDFs, or clearly mark which uploaded files are literature to analyze."
            ]
            summary["next_actions"] = [
                "Keep rubrics or assignment requirements as output constraints instead of counting them as literature.",
                "Add real paper sources, then generate the item-by-item analysis again.",
            ]
            summary["confidence"] = "High; no literature references were available after filtering auxiliary documents."
            return summary
        summary = LiteratureAnalysisWorkflow._empty_summary()
        has_context = bool(final_report.strip())
        summary["overall_assessment"] = (
            "未发现可作为论文/文献条目分析的上传内容；上传材料更适合作为写作要求、评分标准、任务说明或辅助上下文。"
            if has_context
            else "未提供可分析的论文/文献条目。"
        )
        summary["research_gaps"] = [
            "请补充 DOI、论文链接、PDF 全文，或明确哪些上传文件是需要分析的论文。"
        ]
        summary["next_actions"] = [
            "将评分标准或作业要求保留为输出约束，不要计入文献分析表。",
            "补充真正的论文来源后再生成逐篇分析。",
        ]
        summary["confidence"] = "较高；辅助文档过滤后没有可分析的文献条目。"
        return summary
        summary = LiteratureAnalysisWorkflow._empty_summary()
        has_context = bool(final_report.strip())
        summary["overall_assessment"] = (
            "未发现可作为论文/文献条目分析的上传内容；上传材料更适合作为写作要求、评分标准、任务说明或其他辅助上下文。"
            if has_context
            else "未提供可分析的论文/文献条目。"
        )
        summary["research_gaps"] = [
            "请补充 DOI、论文链接、PDF 全文，或明确哪些上传文件是需要分析的论文。"
        ]
        summary["next_actions"] = [
            "将老师要求或评分标准保留为输出约束；不要把它们计入文献分析表。",
            "补充真正的论文来源后再生成逐篇分析。"
        ]
        summary["confidence"] = "High; no literature references were available after filtering auxiliary documents."
        return summary

    @staticmethod
    def _normalize_string_list(value) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _collect_unique(rows: list[dict], key: str, limit: int) -> list[str]:
        values = []
        seen = set()
        for row in rows:
            value = str(row.get(key, "") or "").strip()
            if not value:
                continue
            dedupe_key = value.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            values.append(value)
            if len(values) >= limit:
                break
        return values

    @staticmethod
    def _fill_review_defaults(row: dict[str, str], original: dict, output_language: str = "zh") -> None:
        if not row.get("metadata"):
            row["metadata"] = LiteratureAnalysisWorkflow._format_metadata(original)
        if not row.get("paper_type"):
            row["paper_type"] = "Unknown"
        if not row.get("core_claim"):
            row["core_claim"] = original.get("abstract", "")[:300] or original.get("relevance", "")
        if not row.get("evidence_strength"):
            row["evidence_strength"] = "未知；需要阅读全文或证据审查。" if output_language == "zh" else "Unknown; requires full-text or evidence review."
        if not row.get("reproducibility"):
            row["reproducibility"] = "未知；需要代码、数据和方法细节。" if output_language == "zh" else "Unknown; requires code, data, and method details."
        if not row.get("confidence"):
            row["confidence"] = "较低；仅基于已提供的元数据/上下文。" if output_language == "zh" else "Low; based only on supplied metadata/context."
        if not row.get("overall_assessment"):
            row["overall_assessment"] = "未知" if output_language == "zh" else "Unknown"

    @staticmethod
    def _fill_legacy_aliases(row: dict[str, str]) -> None:
        if not row.get("innovation"):
            row["innovation"] = row.get("contribution", "")
        if not row.get("method"):
            method_parts = [
                value
                for value in [row.get("methodology", ""), row.get("evidence_strength", "")]
                if value
            ]
            row["method"] = "; ".join(method_parts)
        if not row.get("limitation"):
            limitation_parts = [
                value
                for value in [row.get("limitations", ""), row.get("weaknesses", "")]
                if value
            ]
            row["limitation"] = "; ".join(limitation_parts)
        if not row.get("next_step"):
            row["next_step"] = row.get("actionable_suggestions", "")

    @staticmethod
    def _fallback_rows_from_references(
        references: list[dict],
        *,
        fallback_note: str,
        output_language: str = "zh",
    ) -> list[dict]:
        rows = []
        for reference in references:
            if output_language == "zh":
                methodology = "需要阅读全文进一步确认。"
                evidence_strength = "未知；当前上下文不足以评估证据强度。"
                reproducibility = "未知；需要方法、代码、数据和实验细节。"
                questions = "全文中实际提出了哪些主张、证据和方法？"
                suggestions = "获取或上传全文，以便进行有依据的同行评审式分析。"
                confidence = "较低；该兜底结果仅基于已提供的元数据。"
                overall = "未知"
            else:
                methodology = "Requires full-text review."
                evidence_strength = "Unknown; supplied context is insufficient for evidence assessment."
                reproducibility = "Unknown; requires method, code, data, and experiment details."
                questions = "What claims, evidence, and methods are present in the full text?"
                suggestions = "Fetch or upload the full paper for a grounded peer-review analysis."
                confidence = "Low; fallback generated from supplied metadata only."
                overall = "Unknown"
            row = {
                "title": str(reference.get("title", "")).strip(),
                "source": str(reference.get("source", "")).strip(),
                "metadata": LiteratureAnalysisWorkflow._format_metadata(reference),
                "paper_type": "Unknown",
                "core_claim": str(reference.get("abstract", "") or reference.get("relevance", "")).strip()[:500],
                "contribution": str(reference.get("relevance", "")).strip(),
                "methodology": methodology,
                "evidence_strength": evidence_strength,
                "strengths": "",
                "weaknesses": fallback_note,
                "reproducibility": reproducibility,
                "literature_positioning": str(reference.get("branch_name", "")).strip(),
                "application": "",
                "limitations": fallback_note,
                "questions": questions,
                "actionable_suggestions": suggestions,
                "confidence": confidence,
                "overall_assessment": overall,
            }
            LiteratureAnalysisWorkflow._fill_legacy_aliases(row)
            rows.append(row)
        return rows

    @staticmethod
    def _format_metadata(reference: dict) -> str:
        parts = [
            str(reference.get("authors", "")).strip(),
            str(reference.get("journal", "")).strip(),
            str(reference.get("year", "")).strip(),
            str(reference.get("doi", "")).strip(),
        ]
        return "; ".join(part for part in parts if part)

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
            raise ValueError("Expected a JSON object.")
        return data
