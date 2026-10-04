"""Fetch current laws from the 국가법령정보센터 open API (law.go.kr) and convert them to ``statute_json_v1``.

The API reports errors (an unregistered key or IP, a missing parameter) as HTTP 200 with a JSON body that has ``result`` and
``msg`` instead of the data, so every response is checked before it is used. One-article laws arrive with ``조문단위`` as a dict
rather than a list, and some articles are only a pointer to where they moved; both are handled in :func:`convert_law`.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Callable

BASE_URL = "https://www.law.go.kr/DRF/"
Getter = Callable[[str], dict[str, Any]]


class LawApiError(RuntimeError):
    pass


def http_get(url: str, *, timeout: float = 60.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _checked(payload: dict[str, Any]) -> dict[str, Any]:
    if "result" in payload and "msg" in payload and "법령" not in payload and "LawSearch" not in payload:
        raise LawApiError(f"{payload['result']}: {payload['msg']}")
    return payload


def list_current_laws(oc: str, *, get: Getter = http_get, page_size: int = 100) -> list[dict[str, Any]]:
    """Every law the API lists as currently in force (법률, 대통령령, 부령, 대법원규칙, ...)."""
    laws: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"{BASE_URL}lawSearch.do?OC={urllib.parse.quote(oc)}&target=law&type=JSON&display={page_size}&page={page}"
        batch = _as_list(_checked(get(url)).get("LawSearch", {}).get("law"))
        if not batch:
            return laws
        laws += batch
        page += 1


def fetch_body(oc: str, mst: str, *, get: Getter = http_get) -> dict[str, Any]:
    return _checked(get(f"{BASE_URL}lawService.do?OC={urllib.parse.quote(oc)}&target=law&type=JSON&MST={mst}"))


def fetch_law(oc: str, entry: dict[str, Any], *, get: Getter = http_get, retries: int = 3, delay: float = 2.0) -> dict[str, Any] | None:
    """The ``statute_json_v1`` record for one list entry, or None if the law has no article structure."""
    for attempt in range(retries):
        try:
            return convert_law(fetch_body(oc, entry["법령일련번호"], get=get))
        except LawApiError:
            raise  # a rejected key will not get better by retrying
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(delay * (attempt + 1))
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def _text(value: Any) -> str:
    return " ".join(map(str, value)) if isinstance(value, list) else str(value or "")


def _strip_number(text: str, number: str) -> str:
    return re.sub(r"^\s*" + re.escape(number) + r"\s*", "", text).strip() if number else text.strip()


def _date(raw: str) -> str:
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}" if len(raw) == 8 else raw


def convert_law(body: dict[str, Any]) -> dict[str, Any] | None:
    law = body["법령"]
    base = law["기본정보"]
    articles: list[dict[str, Any]] = []
    chapter: str | None = None
    for unit in _as_list((law.get("조문") or {}).get("조문단위")):
        if unit.get("조문여부") != "조문":  # a heading such as 제1장 총칙, or unnumbered body text
            heading = re.match(r"\s*(제\d+[편장절관](?:의\d+)?.*)", _text(unit.get("조문내용")))
            if heading:
                chapter = heading.group(1).strip()
            continue
        number = f"제{unit['조문번호']}조" + (f"의{unit['조문가지번호']}" if unit.get("조문가지번호") else "")
        head = re.match(r"\s*제\d+조(?:의\d+)?\s*(?:\(([^)]*(?:\([^)]*\))?[^)]*)\))?\s*(.*)", _text(unit.get("조문내용")), re.S)
        text = (head.group(2) if head else _text(unit.get("조문내용"))).strip()
        paragraphs = []
        for paragraph in _as_list(unit.get("항")):
            numbered = str(paragraph.get("항번호") or "").strip()
            items = [{"number": str(item.get("호번호") or "").strip(),
                      "text": _strip_number(_text(item.get("호내용")), str(item.get("호번호") or "").strip())}
                     for item in _as_list(paragraph.get("호"))]
            paragraphs.append({"number": numbered, "text": _strip_number(_text(paragraph.get("항내용")), numbered), "items": items})
        if not text and not paragraphs:  # "제2조의2 [제9조로 이동]": only a pointer
            continue
        articles.append({"number": number, "title": unit.get("조문제목") or None, "text": text, "paragraphs": paragraphs, "chapter": chapter})
    if not articles:
        return None
    name = base["법령명_한글"]
    return {
        "law_name": name,
        "version": _date(base["시행일자"]),
        "effective_date": _date(base["시행일자"]),
        "promulgation_date": _date(base["공포일자"]),
        "source_url": f"https://www.law.go.kr/법령/{urllib.parse.quote(name)}",
        "legal_domain": (base.get("법종구분") or {}).get("content", ""),
        "articles": articles,
    }
