from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    original_topic: str
    summary: str
    key_questions: list[str]
    scope: str


@dataclass(frozen=True)
class ResearchBranch:
    name: str
    focus: str
    content: str


@dataclass(frozen=True)
class ResearchResult:
    intent: Intent
    branches: list[ResearchBranch]
    final_report: str
