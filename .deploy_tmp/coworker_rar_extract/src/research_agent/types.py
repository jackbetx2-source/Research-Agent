from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Intent:
    original_topic: str
    summary: str
    key_questions: list[str]
    scope: str


@dataclass(frozen=True)
class ResearchReference:
    title: str
    relevance: str
    source: str
    branch_name: str


@dataclass(frozen=True)
class ResearchPerspective:
    name: str
    focus: str
    questions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchBranch:
    name: str
    focus: str
    content: str
    references: list[ResearchReference]
    questions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReportOutlineSection:
    title: str
    purpose: str
    key_points: list[str] = field(default_factory=list)
    source_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchResult:
    intent: Intent
    branches: list[ResearchBranch]
    final_report: str
    references: list[ResearchReference]
