from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen


ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"
MAX_ARXIV_RESULTS = 50
ARXIV_TIMEOUT_SECONDS = 8
ARXIV_COOLDOWN_SECONDS = 20 * 60
ARXIV_COOLDOWN_UNTIL = 0.0
ARXIV_COOLDOWN_REASON = ""
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass(frozen=True)
class ArxivPaper:
    id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    updated: str
    categories: list[str]
    abs_url: str
    pdf_url: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "updated": self.updated,
            "categories": self.categories,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "source": "arXiv",
        }


class ArxivSearchError(RuntimeError):
    pass


class ArxivCooldownError(ArxivSearchError):
    pass


def search_arxiv(
    query: str,
    *,
    max_results: int = 20,
    category: str | None = None,
    sort_by: str = "relevance",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[ArxivPaper]:
    ensure_arxiv_not_in_cooldown()
    query = extract_core_query(query)
    if not query:
        return []
    if sort_by not in {"relevance", "submittedDate", "lastUpdatedDate"}:
        sort_by = "relevance"

    max_results = max(1, min(int(max_results or 20), MAX_ARXIV_RESULTS))
    params = {
        "search_query": build_search_query(query, category, start_date, end_date),
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    url = f"{ARXIV_ENDPOINT}?{urlencode(params, quote_via=quote_plus)}"
    request = Request(url, headers={"User-Agent": "ResearchAgent-SLR/0.1"})

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urlopen(request, timeout=ARXIV_TIMEOUT_SECONDS) as response:
                payload = response.read().decode("utf-8", errors="replace")
                break
        except HTTPError as error:
            last_error = error
            if error.code != 429 or attempt == 1:
                if error.code == 429:
                    mark_arxiv_cooldown("HTTP 429 rate limit")
                raise ArxivSearchError(f"arXiv search failed: HTTP {error.code}") from error
            time.sleep(3 * (attempt + 1))
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt == 1:
                mark_arxiv_cooldown(f"{type(error).__name__}: {error}")
                raise ArxivSearchError(f"arXiv search failed: {error}") from error
            time.sleep(2 * (attempt + 1))
    else:
        raise ArxivSearchError(f"arXiv search failed: {last_error}")

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ArxivSearchError("arXiv returned invalid XML.") from error

    return [parse_entry(entry) for entry in root.findall("atom:entry", ATOM_NS)]


def ensure_arxiv_not_in_cooldown() -> None:
    now = time.time()
    if now < ARXIV_COOLDOWN_UNTIL:
        remaining = int(ARXIV_COOLDOWN_UNTIL - now)
        raise ArxivCooldownError(
            f"arXiv skipped due to recent failure ({ARXIV_COOLDOWN_REASON}); retry in about {remaining} seconds."
        )


def mark_arxiv_cooldown(reason: str) -> None:
    global ARXIV_COOLDOWN_UNTIL, ARXIV_COOLDOWN_REASON
    ARXIV_COOLDOWN_UNTIL = time.time() + ARXIV_COOLDOWN_SECONDS
    ARXIV_COOLDOWN_REASON = reason


def extract_core_query(topic: str) -> str:
    cleaned = re.sub(r"\s+", " ", topic or "").strip()
    cleaned = re.sub(
        r"\b(recent|latest|survey|review|literature|systematic|papers?|研究|综述|文献|最近|最新)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:，。；：")
    words = cleaned.split()
    if len(words) > 4:
        return " ".join(words[:4])
    return cleaned


def build_search_query(
    query: str,
    category: str | None,
    start_date: str | None,
    end_date: str | None,
) -> str:
    parts = [f'all:"{query}"' if " " in query else f"all:{query}"]
    if category:
        parts.append(f"cat:{category.strip()}")
    if start_date or end_date:
        lo = normalize_date_for_arxiv(start_date) or "199101010000"
        hi = normalize_date_for_arxiv(end_date, end_of_day=True) or "299912312359"
        parts.append(f"submittedDate:[{lo} TO {hi}]")
    return " AND ".join(parts)


def normalize_date_for_arxiv(value: str | None, *, end_of_day: bool = False) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) < 8:
        return ""
    return digits[:8] + ("2359" if end_of_day else "0000")


def parse_entry(entry: ET.Element) -> ArxivPaper:
    def text(path: str) -> str:
        node = entry.find(path, ATOM_NS)
        return " ".join((node.text or "").split()) if node is not None else ""

    raw_id = text("atom:id")
    arxiv_id = normalize_arxiv_id(raw_id)
    authors = [
        " ".join((author.findtext("atom:name", default="", namespaces=ATOM_NS) or "").split())
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    authors = [author for author in authors if author]
    categories = [
        category.get("term", "")
        for category in entry.findall("atom:category", ATOM_NS)
        if category.get("term")
    ]
    abs_url = raw_id
    pdf_url = ""
    for link in entry.findall("atom:link", ATOM_NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
        elif link.get("rel") == "alternate":
            abs_url = link.get("href", abs_url)

    published_raw = text("atom:published")
    updated_raw = text("atom:updated")
    return ArxivPaper(
        id=arxiv_id,
        title=text("atom:title"),
        authors=authors,
        abstract=text("atom:summary"),
        published=published_raw.split("T", 1)[0] if published_raw else "",
        updated=updated_raw.split("T", 1)[0] if updated_raw else "",
        categories=categories,
        abs_url=abs_url or f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
    )


def normalize_arxiv_id(raw_id: str) -> str:
    tail = raw_id.split("/abs/", 1)[1] if "/abs/" in raw_id else raw_id.rsplit("/", 1)[-1]
    base, marker, suffix = tail.rpartition("v")
    if marker and suffix.isdigit():
        return base
    return tail


def papers_to_json(papers: list[ArxivPaper]) -> str:
    return json.dumps([paper.to_dict() for paper in papers], ensure_ascii=False, indent=2)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Search arXiv and return paper metadata as JSON.")
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--category")
    parser.add_argument("--sort-by", default="relevance")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()

    try:
        papers = search_arxiv(
            args.query,
            max_results=args.max_results,
            category=args.category,
            sort_by=args.sort_by,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    except ArxivSearchError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(papers_to_json(papers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
