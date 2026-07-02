from __future__ import annotations

import json
import re
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CROSSREF_ENDPOINT = "https://api.crossref.org/works"


class CrossrefSearchError(RuntimeError):
    pass


def search_crossref(
    query: str,
    *,
    max_results: int = 20,
    start_date: str = "",
    end_date: str = "",
) -> list[dict]:
    max_results = max(1, min(int(max_results or 20), 50))
    filters = []
    if start_date:
        filters.append(f"from-pub-date:{start_date}")
    if end_date:
        filters.append(f"until-pub-date:{end_date}")
    params = {
        "query": query,
        "rows": min(50, max_results * 3),
        "sort": "relevance",
    }
    if filters:
        params["filter"] = ",".join(filters)

    request = Request(
        f"{CROSSREF_ENDPOINT}?{urlencode(params)}",
        headers={
            "Accept": "application/json",
            "User-Agent": "ResearchAgent-SLR/0.1 (mailto:local@example.com)",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise CrossrefSearchError(f"Crossref search failed: {error}") from error

    items = payload.get("message", {}).get("items", [])
    papers = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() in {"component", "posted-content-component"}:
            continue
        paper = normalize_crossref_item(item)
        key = paper.get("id") or paper.get("title")
        if not key or key in seen:
            continue
        seen.add(key)
        papers.append(paper)
        if len(papers) >= max_results:
            break
    return papers


def normalize_crossref_item(item: dict) -> dict:
    doi = str(item.get("DOI", "") or "").strip()
    title = first_text(item.get("title")) or (f"DOI: {doi}" if doi else "Untitled Crossref work")
    journal = first_text(item.get("container-title"))
    year = published_year(item)
    url = str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")).strip()
    authors = []
    for author in item.get("author", [])[:12]:
        if not isinstance(author, dict):
            continue
        given = str(author.get("given", "") or "").strip()
        family = str(author.get("family", "") or "").strip()
        name = " ".join(part for part in (given, family) if part)
        if name:
            authors.append(name)
    if len(item.get("author", [])) > 12:
        authors.append("et al.")
    return {
        "id": doi or url or title,
        "title": title,
        "authors": authors,
        "abstract": clean_abstract(str(item.get("abstract", "") or "")),
        "published": str(year) if year else "",
        "updated": "",
        "categories": [str(item.get("type", "") or "Crossref").strip()],
        "abs_url": url,
        "pdf_url": "",
        "source": "Crossref",
        "doi": doi,
        "journal": journal,
    }


def first_text(value) -> str:
    if isinstance(value, list) and value:
        return str(value[0]).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def published_year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued", "created"):
        value = item.get(key)
        if not isinstance(value, dict):
            continue
        date_parts = value.get("date-parts")
        if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            try:
                return int(date_parts[0][0])
            except (TypeError, ValueError):
                continue
    return None


def clean_abstract(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()
