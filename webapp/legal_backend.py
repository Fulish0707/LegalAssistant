"""Client for legal_rag's /v1/chat, which writes the web chat's answers.

The service retrieves, generates with its configured model (LEGAL_RAG__LLM__BACKEND)
and verifies citations; this module only calls it and turns the structured response
into the chat text and source cards the front end shows.
"""

import re

import httpx

# A turn with one regeneration measured up to 45 s (2026-10-05); the first request
# after a start also loads the reranker.
TIMEOUT = 180.0
QUOTE_CHARS = 400

REJECTED_TEXT = (
    "근거를 확인할 수 없어 답변을 제공하지 못했습니다. "
    "질문을 더 구체적으로 적어 주시거나, 국가법령정보센터(law.go.kr)에서 직접 확인해 주세요."
)
CLARIFY_TEXT = "정확히 안내하려면 다음 내용을 알려 주세요."

# Anything that is neither a statute nor an interpretation is case-like.
_KIND_LABELS = {
    "statute": "법령",
    "legal_interpretation": "해석례",
    "admin_interpretation": "해석례",
}
# Header fields that name a source; Type, Section and Section role describe it instead.
_LABEL_FIELDS = ("Law", "Article", "Paragraph", "Court", "Authority", "Case number", "Reference number", "Decision date")
# The model writes a source as S1, [S1] or [LEGAL SOURCE S1], alone or as a parenthesised list;
# a passage of the user's own file is D1 or [USER DOCUMENT D1].
_SOURCE_ID = re.compile(r"\[?(?:LEGAL SOURCE |USER DOCUMENT )?(?<![A-Za-z0-9])([SD]\d+)(?![A-Za-z0-9])\]?")
DOCUMENT_KIND = "내 문서"
_SOURCE_GROUP = re.compile(r"\(\s*(%s(?:\s*,\s*%s)*)\s*\)" % (_SOURCE_ID.pattern, _SOURCE_ID.pattern))


class LegalBackendUnavailable(Exception):
    """The service could not be reached or did not return a chat response."""


def ask(base_url, message, conversation_id, documents=()):
    """One chat turn. `documents` are passages of the user's own files: [{name, text, ...}]."""
    endpoint = base_url.rstrip("/") + "/v1/chat"
    payload = {"message": message, "conversation_id": conversation_id}
    if documents:
        payload["user_documents"] = [{"name": document["name"], "text": document["text"]} for document in documents]
    try:
        response = httpx.post(endpoint, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise LegalBackendUnavailable(f"{type(error).__name__}: {error}") from error


def _read(base_url, path, shape, **params):
    """A GET to the service, passed through `shape`; None when the service answers 404."""
    try:
        response = httpx.get(base_url.rstrip("/") + path, params=params, timeout=TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return shape(response.json())
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
        raise LegalBackendUnavailable(f"{type(error).__name__}: {error}") from error


def document(base_url, chunk_id):
    """The whole document that holds a cited chunk, or None if not indexed: {before, cited, after}.

    The text is cut here: the service counts characters as Python does, and a browser counts a rare
    character outside the basic plane as two.
    """
    def cut(body):
        text, start, end = body["text"], body["cited_start"], body["cited_end"]
        return {"before": text[:start], "cited": text[start:end], "after": text[end:]}

    return _read(base_url, "/v1/document", cut, chunk_id=chunk_id)


def _numbered(text, numbers):
    """Replace the model's source ids (S1, S2) with the sources panel's numbers."""
    def one(match):
        source_id = match.group(1)
        return f"[{numbers[source_id]}]" if source_id in numbers else match.group(0)

    text = _SOURCE_GROUP.sub(lambda group: "".join(one(match) for match in _SOURCE_ID.finditer(group.group(1))), text)
    return _SOURCE_ID.sub(one, text)


def _kind(item):
    return _KIND_LABELS.get(item.get("document_type"), "판례")


def _label(item):
    header = item.get("header") or {}
    label = " ".join(dict.fromkeys(header[field] for field in _LABEL_FIELDS if header.get(field)))
    # Chunks often start mid-sentence, so the text is no substitute for a missing name.
    return label or f"{_kind(item)} (법원·사건번호 미기재)"


def evidence_texts(body):
    """Each evidence block in full, header values then text, for checks on the shown answer."""
    items = (body.get("evidence") or {}).get("items") or []
    return [" ".join([*map(str, (item.get("header") or {}).values()), item.get("text") or ""]) for item in items]


def questions(body):
    """What the turn asks the user: the facts an answer still needs, or a clarification's questions."""
    if body.get("kind") == "clarification":
        return list(body.get("questions") or [])
    if body.get("kind") == "answer":
        # The model sometimes writes a statement here ("...해야 합니다."); it cannot be answered.
        needed = (body.get("answer") or {}).get("additional_information_needed") or []
        return [item for item in needed if not re.search(r"다\.?$", item.strip())]
    return []


def _bullets(heading, items):
    return [heading + "\n" + "\n".join(f"- {item}" for item in items)] if items else []


def render(body, documents=()):
    """(chat text, [{source, quote, kind, score, chunk_id}]) for a /v1/chat response body.

    `documents` are the user's passages sent with the turn, in the order sent (D1, D2, ...).
    """
    kind = body.get("kind")
    if kind == "clarification":
        return CLARIFY_TEXT + "\n" + "\n".join(f"- {question}" for question in body.get("questions") or []), []
    answer = body.get("answer")
    if kind != "answer" or not answer:
        return REJECTED_TEXT, []

    parts = [answer.get("summary") or "", answer.get("analysis") or ""]
    parts += _bullets("취할 수 있는 조치", answer.get("possible_actions"))
    parts += _bullets("추가로 확인이 필요한 사실", answer.get("additional_information_needed"))
    if answer.get("uncertainty"):
        parts.append("불확실한 점\n" + answer["uncertainty"])
    prose = "\n\n".join(part.strip() for part in parts if part.strip())

    # Show only what the answer relies on: a decline or an out-of-scope question still retrieves
    # five chunks, and cards for them would read as support.
    cited = {citation.get("source_id") for citation in answer.get("citations") or []}
    cited |= set(_SOURCE_ID.findall(prose))
    items = [item for item in (body.get("evidence") or {}).get("items") or [] if item.get("source_id") in cited]
    used = [(f"D{number}", document) for number, document in enumerate(documents, 1) if f"D{number}" in cited]
    numbers = {source_id: position for position, source_id in
               enumerate([*(item["source_id"] for item in items), *(source_id for source_id, _ in used)], 1)}
    text = _numbered(prose, numbers)

    sources = [
        {
            "source": _label(item),
            "quote": (item.get("text") or "").strip()[:QUOTE_CHARS],
            "kind": _kind(item),
            "score": item.get("score"),
            # The card shows an excerpt; the reader asks for the whole document by this id.
            "chunk_id": item.get("chunk_id"),
        }
        for item in items
    ] + [
        {"source": document["name"], "quote": document["text"].strip()[:QUOTE_CHARS], "kind": DOCUMENT_KIND, "score": document.get("score")}
        for _, document in used
    ]
    return text, sources
