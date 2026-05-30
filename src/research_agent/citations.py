from __future__ import annotations

import re


def format_references(papers: list[dict], citation_format: str) -> list[str]:
    fmt = normalize_citation_format(citation_format)
    if fmt == "ieee":
        return [format_ieee_reference(index + 1, paper) for index, paper in enumerate(papers)]
    if fmt == "bibtex":
        return [format_bibtex_reference(paper) for paper in papers]
    return [format_apa_reference(paper) for paper in sorted(papers, key=apa_sort_key)]


def normalize_citation_format(value: str | None) -> str:
    fmt = (value or "APA").strip().lower()
    if "ieee" in fmt:
        return "ieee"
    if "bib" in fmt:
        return "bibtex"
    return "apa"


def year_from_paper(paper: dict) -> str:
    value = str(paper.get("published") or paper.get("published_date") or paper.get("year") or "")
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return match.group(0) if match else "n.d."


def paper_url(paper: dict) -> str:
    return str(paper.get("abs_url") or paper.get("source") or f"https://arxiv.org/abs/{paper.get('id', '')}").strip()


def arxiv_id(paper: dict) -> str:
    raw = str(paper.get("id") or paper.get("arxiv_id") or "").strip()
    if raw:
        return raw
    url = paper_url(paper)
    return url.rstrip("/").rsplit("/", 1)[-1]


def title_sentence_case(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    if not title:
        return "Untitled"
    return title[:1].upper() + title[1:]


def authors_list(paper: dict) -> list[str]:
    authors = paper.get("authors", [])
    if isinstance(authors, str):
        authors = [item.strip() for item in re.split(r";|,", authors) if item.strip()]
    if not isinstance(authors, list):
        return []
    return [str(author).strip() for author in authors if str(author).strip()]


def apa_author(name: str) -> str:
    parts = name.split()
    if not parts:
        return ""
    family = parts[-1]
    initials = " ".join(f"{part[0]}." for part in parts[:-1] if part)
    return f"{family}, {initials}".strip()


def ieee_author(name: str) -> str:
    parts = name.split()
    if not parts:
        return ""
    family = parts[-1]
    initials = " ".join(f"{part[0]}." for part in parts[:-1] if part)
    return f"{initials} {family}".strip()


def join_apa_authors(authors: list[str]) -> str:
    formatted = [apa_author(author) for author in authors if apa_author(author)]
    if not formatted:
        return "Unknown author"
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) > 20:
        formatted = formatted[:19] + ["..."] + formatted[-1:]
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def join_ieee_authors(authors: list[str]) -> str:
    formatted = [ieee_author(author) for author in authors if ieee_author(author)]
    if not formatted:
        return "Unknown author"
    if len(formatted) > 6:
        return f"{formatted[0]} et al."
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def format_apa_reference(paper: dict) -> str:
    if not is_arxiv_paper(paper):
        source = str(paper.get("journal") or paper.get("source") or "Scholarly source").strip()
        return (
            f"{join_apa_authors(authors_list(paper))} ({year_from_paper(paper)}). "
            f"{title_sentence_case(str(paper.get('title', 'Untitled')))}. {source}. {paper_url(paper)}"
        )
    return (
        f"{join_apa_authors(authors_list(paper))} ({year_from_paper(paper)}). "
        f"{title_sentence_case(str(paper.get('title', 'Untitled')))}. arXiv. {paper_url(paper)}"
    )


def format_ieee_reference(index: int, paper: dict) -> str:
    if not is_arxiv_paper(paper):
        identifier = f"doi: {paper.get('doi')}" if paper.get("doi") else paper_url(paper)
        return (
            f"[{index}] {join_ieee_authors(authors_list(paper))}, "
            f"\"{title_sentence_case(str(paper.get('title', 'Untitled')))},\" "
            f"{identifier}, {year_from_paper(paper)}."
        )
    return (
        f"[{index}] {join_ieee_authors(authors_list(paper))}, "
        f"\"{title_sentence_case(str(paper.get('title', 'Untitled')))},\" "
        f"arXiv:{arxiv_id(paper)}, {year_from_paper(paper)}."
    )


def format_bibtex_reference(paper: dict) -> str:
    key = bibtex_key(paper)
    authors = " and ".join(authors_list(paper)) or "Unknown"
    if not is_arxiv_paper(paper):
        entry_type = "article" if paper.get("journal") else "misc"
        doi_line = f"  doi = {{{paper.get('doi')}}},\n" if paper.get("doi") else ""
        journal_line = f"  journal = {{{paper.get('journal')}}},\n" if paper.get("journal") else ""
        return (
            f"@{entry_type}{{{key},\n"
            f"  title = {{{str(paper.get('title', 'Untitled')).strip()}}},\n"
            f"  author = {{{authors}}},\n"
            f"{journal_line}"
            f"  year = {{{year_from_paper(paper)}}},\n"
            f"{doi_line}"
            f"  url = {{{paper_url(paper)}}}\n"
            f"}}"
        )
    return (
        f"@misc{{{key},\n"
        f"  title = {{{str(paper.get('title', 'Untitled')).strip()}}},\n"
        f"  author = {{{authors}}},\n"
        f"  year = {{{year_from_paper(paper)}}},\n"
        f"  eprint = {{{arxiv_id(paper)}}},\n"
        f"  archivePrefix = {{arXiv}},\n"
        f"  url = {{{paper_url(paper)}}}\n"
        f"}}"
    )


def bibtex_key(paper: dict) -> str:
    authors = authors_list(paper)
    first = authors[0].split()[-1] if authors else "unknown"
    first = re.sub(r"[^A-Za-z0-9]+", "", first).lower() or "unknown"
    title_word = re.sub(r"[^A-Za-z0-9]+", "", str(paper.get("title", "paper")).split()[0]).lower()
    return f"{first}{year_from_paper(paper).replace('n.d.', 'nd')}{title_word or 'paper'}"


def apa_sort_key(paper: dict) -> tuple[str, str]:
    authors = authors_list(paper)
    first = authors[0].split()[-1].casefold() if authors else ""
    return (first, year_from_paper(paper))


def is_arxiv_paper(paper: dict) -> bool:
    source = str(paper.get("source", "") or "").casefold()
    url = paper_url(paper).casefold()
    return source == "arxiv" or "arxiv.org" in url
