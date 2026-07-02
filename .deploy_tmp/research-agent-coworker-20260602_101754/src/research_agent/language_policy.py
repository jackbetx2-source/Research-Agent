from __future__ import annotations

import re


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value or ""))


def main_language(value: str, *, default: str = "en") -> str:
    text = value or ""
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", text))
    if cjk_chars == 0:
        return default
    if latin_words == 0:
        return "zh"
    return "zh" if cjk_chars >= latin_words * 0.8 else "en"


def paper_language(paper: dict) -> str:
    text = " ".join(
        str(paper.get(key) or "")
        for key in ("title", "abstract", "journal", "metadata")
    )
    if contains_cjk(text):
        return "zh"
    ascii_letters = len(re.findall(r"[A-Za-z]", text))
    return "en" if ascii_letters >= 12 else "other"
