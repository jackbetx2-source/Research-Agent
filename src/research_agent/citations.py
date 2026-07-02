from __future__ import annotations

import re


def format_references(papers: list[dict], citation_format: str) -> list[str]:
    fmt = normalize_citation_format(citation_format)
    if fmt == "ieee":
        return [format_ieee_reference(index + 1, paper) for index, paper in enumerate(papers)]
    if fmt == "bibtex":
        return [format_bibtex_reference(paper) for paper in papers]
    return [format_apa_reference(paper) for paper in sorted(papers, key=apa_sort_key)]


def format_apa_reference_item(reference) -> str:
    return format_apa_reference(reference_to_paper(reference))


def format_apa_reference_items(references: list) -> list[str]:
    formatted = [format_apa_reference_item(reference) for reference in references]
    return sorted(
        dict.fromkeys(item for item in formatted if item),
        key=lambda item: item.casefold(),
    )


def reference_to_paper(reference) -> dict:
    source = str(_reference_value(reference, "source") or "").strip()
    if source.casefold() in {"not provided", "未提供"}:
        source = ""
    doi = str(_reference_value(reference, "doi") or "").strip()
    pmid = str(_reference_value(reference, "pmid") or "").strip()
    if not doi:
        doi_match = re.search(r"10\.\d{4,9}/[^\s,;]+", source)
        doi = doi_match.group(0).rstrip(".)]}") if doi_match else ""
    if not pmid and "pubmed.ncbi.nlm.nih.gov" in source:
        pmid_match = re.search(r"/(\d{6,9})(?:/|$)", source)
        pmid = pmid_match.group(1) if pmid_match else ""
    return {
        "title": str(_reference_value(reference, "title") or "Untitled source").strip(),
        "authors": _reference_value(reference, "authors") or [],
        "published": str(_reference_value(reference, "published") or _reference_value(reference, "published_date") or _reference_value(reference, "year") or year_from_reference(reference) or "").strip(),
        "year": str(_reference_value(reference, "year") or "").strip(),
        "journal": str(_reference_value(reference, "journal") or "").strip(),
        "source": str(_reference_value(reference, "source_label") or _reference_value(reference, "source_origin") or "").strip(),
        "abs_url": source,
        "doi": doi,
        "pmid": pmid,
    }


def normalize_citation_format(value: str | None) -> str:
    fmt = (value or "APA").strip().lower()
    if "ieee" in fmt:
        return "ieee"
    if "bib" in fmt:
        return "bibtex"
    return "apa"


def _reference_value(reference, key: str):
    if isinstance(reference, dict):
        return reference.get(key)
    return getattr(reference, key, None)


def year_from_reference(reference, fallback_text: str = "") -> str:
    value = " ".join(
        str(item or "")
        for item in [
            _reference_value(reference, "published"),
            _reference_value(reference, "published_date"),
            _reference_value(reference, "year"),
            fallback_text,
        ]
    )
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return match.group(0) if match else "n.d."


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


def ensure_terminal_punctuation(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not text:
        return ""
    return text if text[-1] in ".?!" else f"{text}."


def authors_list(paper: dict) -> list[str]:
    authors = paper.get("authors", [])
    if isinstance(authors, str):
        authors = [item.strip() for item in re.split(r";|,", authors) if item.strip()]
    if not isinstance(authors, list):
        return []
    return [
        str(author).strip()
        for author in authors
        if str(author).strip() and not re.fullmatch(r"et\s+al\.?", str(author).strip(), flags=re.IGNORECASE)
    ]


def apa_author(name: str) -> str:
    parts = name.split()
    if not parts:
        return ""
    family = parts[-1]
    initials = " ".join(f"{part[0]}." for part in parts[:-1] if part)
    return f"{family}, {initials}".strip()


def format_apa_author_text(value: str) -> str:
    author = re.sub(r"\s+", " ", value or "").strip().rstrip(".")
    if not author:
        return "Unknown author"
    if re.search(r"\b(et al|collaborators|consortium|group|team|committee|institute|agency|organization|quantum)\b", author, flags=re.IGNORECASE):
        return author
    if "," in author:
        return author
    return join_apa_authors([author])


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
    authors = authors_list(paper)
    title = ensure_terminal_punctuation(title_sentence_case(str(paper.get("title", "Untitled"))))
    year = year_from_paper(paper)
    if not is_arxiv_paper(paper):
        source = str(paper.get("journal") or paper.get("source") or "Scholarly source").strip()
        if not authors:
            return f"{title} ({year}). {source}. {paper_url(paper)}"
        return (
            f"{join_apa_authors(authors)} ({year}). "
            f"{title} {source}. {paper_url(paper)}"
        )
    if not authors:
        return f"{title} ({year}). arXiv. {paper_url(paper)}"
    return (
        f"{join_apa_authors(authors)} ({year}). "
        f"{title} arXiv. {paper_url(paper)}"
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
