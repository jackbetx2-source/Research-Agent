from __future__ import annotations

import json
import os
import re
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


OPENALEX_ENDPOINT = "https://api.openalex.org/works"


class OpenAlexSearchError(RuntimeError):
    pass


def search_openalex(
    query: str,
    *,
    max_results: int = 20,
    start_date: str = "",
    end_date: str = "",
) -> list[dict]:
    max_results = max(1, min(int(max_results or 20), 50))
    filters = []
    if start_date:
        filters.append(f"from_publication_date:{start_date}")
    if end_date:
        filters.append(f"to_publication_date:{end_date}")

    params = {
        "search": query,
        "per-page": max_results,
        "sort": "relevance_score:desc",
    }
    if filters:
        params["filter"] = ",".join(filters)
    if os.getenv("OPENALEX_EMAIL"):
        params["mailto"] = os.getenv("OPENALEX_EMAIL")

    request = Request(
        f"{OPENALEX_ENDPOINT}?{urlencode(params)}",
        headers={"Accept": "application/json", "User-Agent": "ResearchAgent-SLR/0.1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise OpenAlexSearchError(f"OpenAlex search failed: {error}") from error

    return [normalize_openalex_work(item) for item in payload.get("results", []) if isinstance(item, dict)]


def normalize_openalex_work(item: dict) -> dict:
    doi = str(item.get("doi") or "").replace("https://doi.org/", "").strip()
    ids = item.get("ids") if isinstance(item.get("ids"), dict) else {}
    if not doi:
        doi = str(ids.get("doi") or "").replace("https://doi.org/", "").strip()
    url = str(item.get("doi") or ids.get("openalex") or item.get("id") or "").strip()
    if doi:
        url = f"https://doi.org/{doi}"
    source = item.get("primary_location", {})
    source_info = source.get("source", {}) if isinstance(source, dict) else {}
    journal = source_info.get("display_name", "") if isinstance(source_info, dict) else ""
    authors = []
    for authorship in item.get("authorships", [])[:12]:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author", {})
        name = author.get("display_name", "") if isinstance(author, dict) else ""
        if name:
            authors.append(str(name).strip())
    if len(item.get("authorships", [])) > 12:
        authors.append("et al.")

    return {
        "id": doi or str(item.get("id") or url),
        "title": str(item.get("display_name") or "Untitled OpenAlex work").strip(),
        "authors": authors,
        "abstract": inverted_index_to_text(item.get("abstract_inverted_index")),
        "published": str(item.get("publication_date") or item.get("publication_year") or ""),
        "updated": str(item.get("updated_date") or ""),
        "categories": [str(item.get("type") or "OpenAlex").strip()],
        "abs_url": url,
        "pdf_url": best_pdf_url(item),
        "source": "OpenAlex",
        "doi": doi,
        "journal": str(journal or "").strip(),
    }


def inverted_index_to_text(value) -> str:
    if not isinstance(value, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in value.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            try:
                positions.append((int(index), str(word)))
            except (TypeError, ValueError):
                continue
    return " ".join(word for _, word in sorted(positions))


def best_pdf_url(item: dict) -> str:
    locations = []
    for key in ("best_oa_location", "primary_location"):
        value = item.get(key)
        if isinstance(value, dict):
            locations.append(value)
    for location in locations:
        pdf_url = str(location.get("pdf_url") or "").strip()
        if pdf_url:
            return pdf_url
    return ""


def clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()
