from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


DEFAULT_CHUNK_SIZE = 3500
DEFAULT_CHUNK_OVERLAP = 400
MIN_CHUNK_SIZE = 200

LEGAL_DOMAIN = "intellectual_property"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Utility
# ============================================================

def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    except Exception as e:
        logger.warning(
            "JSON 읽기 실패: %s | %s",
            path,
            e,
        )
        return None


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def normalize_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def normalize_sentences(
    sentences: Any,
) -> str:

    if isinstance(sentences, str):
        return normalize_text(sentences)

    if isinstance(sentences, list):

        result = []

        for item in sentences:

            if not isinstance(item, str):
                continue

            cleaned = normalize_text(item)

            if cleaned:
                result.append(cleaned)

        return "\n".join(result)

    return ""


# ============================================================
# doc_class = 1
# 판결문
# ============================================================

def extract_judgment(
    data: dict[str, Any],
) -> dict[str, Any] | None:

    info = data.get("info", {})
    taskinfo = data.get("taskinfo", {})

    doc_id = info.get("doc_id")

    if not doc_id:
        return None

    text = normalize_sentences(
        taskinfo.get("sentences", [])
    )

    if not text:
        return None

    return {
        "document_id": str(doc_id),

        "legal_domain": LEGAL_DOMAIN,

        "document_type": "judgment",

        "title": info.get("casenames"),

        "authority": info.get(
            "normalized_court"
        ),

        "date": info.get(
            "announce_date"
        ),

        "text": text,

        "metadata": {
            "source": "AIHub",

            "doc_class": info.get(
                "doc_class"
            ),

            "case_type": info.get(
                "casetype"
            ),

            "source_task_type": info.get(
                "taskType"
            ),

            "instruction_case": taskinfo.get(
                "instruction_case"
            ),
        },
    }


# ============================================================
# doc_class = 2
# 법령
# ============================================================

def extract_statute(
    data: dict[str, Any],
) -> dict[str, Any] | None:

    info = data.get("info", {})
    taskinfo = data.get("taskinfo", {})

    statute_name = info.get(
        "statute_name"
    )

    effective_date = info.get(
        "effective_date"
    )

    if not statute_name:
        return None

    text = normalize_sentences(
        taskinfo.get("sentences", "")
    )

    if not text:
        return None

    document_id = (
        f"statute::{statute_name}::"
        f"{effective_date or 'unknown'}"
    )

    return {
        "document_id": document_id,

        "legal_domain": LEGAL_DOMAIN,

        "document_type": "statute",

        "title": statute_name,

        "authority": None,

        "date": effective_date,

        "text": text,

        "metadata": {
            "source": "AIHub",

            "doc_class": info.get(
                "doc_class"
            ),

            "statute_type": info.get(
                "statute_type"
            ),

            "statute_abbrv": info.get(
                "statute_abbrv"
            ),

            "statute_category": info.get(
                "statute_category"
            ),

            "proclamation_date": info.get(
                "proclamation_date"
            ),

            "source_task_type": info.get(
                "taskType"
            ),

            "instruction_case": taskinfo.get(
                "instruction_case"
            ),
        },
    }


# ============================================================
# doc_class = 3
# 심결례 / 위원회 결정
# ============================================================

def extract_decision(
    data: dict[str, Any],
) -> dict[str, Any] | None:

    info = data.get("info", {})
    taskinfo = data.get("taskinfo", {})

    doc_id = info.get("doc_id")

    if not doc_id:
        return None

    text = normalize_sentences(
        taskinfo.get("sentences", "")
    )

    if not text:
        return None

    return {
        "document_id": str(doc_id),

        "legal_domain": LEGAL_DOMAIN,

        "document_type": "decision",

        "title": info.get("title"),

        "authority": info.get(
            "document_type"
        ),

        "date": info.get(
            "decision_date"
        ),

        "text": text,

        "metadata": {
            "source": "AIHub",

            "doc_class": info.get(
                "doc_class"
            ),

            "result": info.get(
                "result"
            ),

            "source_task_type": info.get(
                "taskType"
            ),

            "instruction_case": taskinfo.get(
                "instruction_case"
            ),
        },
    }


# ============================================================
# doc_class = 4
# 특허심판원 심결문
# ============================================================

def extract_patent_trial_decision(
    data: dict[str, Any],
) -> dict[str, Any] | None:

    info = data.get("info", {})
    taskinfo = data.get("taskinfo", {})

    doc_id = info.get("doc_id")

    if not doc_id:
        return None

    text = normalize_sentences(
        taskinfo.get("sentences", [])
    )

    if not text:
        return None

    return {
        "document_id": str(doc_id),

        "legal_domain": LEGAL_DOMAIN,

        "document_type": "patent_trial_decision",

        "title": info.get(
            "title"
        ),

        "authority": info.get(
            "document_type"
        ),

        "date": info.get(
            "decision_date"
        ),

        "text": text,

        "metadata": {
            "source": "AIHub",

            "doc_class": info.get(
                "doc_class"
            ),

            "result": info.get(
                "result"
            ),

            "source_task_type": info.get(
                "taskType"
            ),

            "instruction_case": taskinfo.get(
                "instruction_case"
            ),
        },
    }


# ============================================================
# doc_class = 5
# 유권해석 / 법령해석
# ============================================================

def extract_legal_interpretation(
    data: dict[str, Any],
) -> dict[str, Any] | None:

    info = data.get("info", {})
    taskinfo = data.get("taskinfo", {})

    doc_id = info.get("doc_id")

    if not doc_id:
        return None

    text = normalize_sentences(
        taskinfo.get("sentences", "")
    )

    if not text:
        return None

    return {
        "document_id": str(doc_id),

        "legal_domain": LEGAL_DOMAIN,

        "document_type": "legal_interpretation",

        "title": info.get(
            "title"
        ),

        "authority": info.get(
            "response_institute"
        ),

        "date": info.get(
            "response_date"
        ),

        "text": text,

        "metadata": {
            "source": "AIHub",

            "doc_class": info.get(
                "doc_class"
            ),

            "response_institute": info.get(
                "response_institute"
            ),

            "response_date": info.get(
                "response_date"
            ),

            "source_task_type": info.get(
                "taskType"
            ),

            "instruction_case": taskinfo.get(
                "instruction_case"
            ),
        },
    }


# ============================================================
# Dispatcher
# ============================================================

def extract_document(
    data: dict[str, Any],
) -> dict[str, Any] | None:

    info = data.get("info", {})

    doc_class = str(
        info.get(
            "doc_class",
            "",
        )
    )

    if doc_class == "1":

        return extract_judgment(
            data
        )

    if doc_class == "2":

        return extract_statute(
            data
        )

    if doc_class == "3":

        return extract_decision(
            data
        )

    if doc_class == "4":

        return extract_patent_trial_decision(
            data
        )

    if doc_class == "5":

        return extract_legal_interpretation(
            data
        )

    logger.warning(
        "알 수 없는 doc_class: %s",
        doc_class,
    )

    return None


# ============================================================
# QA
# ============================================================

def extract_qa(
    data: dict[str, Any],
    document: dict[str, Any] | None,
) -> dict[str, Any] | None:

    taskinfo = data.get(
        "taskinfo",
        {}
    )

    question = taskinfo.get(
        "input"
    )

    answer = taskinfo.get(
        "output"
    )

    if not isinstance(
        question,
        str,
    ):
        return None

    if not isinstance(
        answer,
        str,
    ):
        return None

    question = normalize_text(
        question
    )

    answer = normalize_text(
        answer
    )

    if not question or not answer:
        return None

    return {
        "document_id": (
            document.get("document_id")
            if document
            else None
        ),

        "legal_domain": LEGAL_DOMAIN,

        "document_type": (
            document.get("document_type")
            if document
            else None
        ),

        "question": question,

        "answer": answer,

        "metadata": {
            "source": "AIHub"
        },
    }


# ============================================================
# Chunking
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size는 0보다 커야 합니다."
        )

    if overlap < 0:
        raise ValueError(
            "overlap은 0 이상이어야 합니다."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap은 chunk_size보다 작아야 합니다."
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        chunk = text[start:end]

        if end < text_length:

            search_start = max(
                0,
                len(chunk) - 700,
            )

            tail = chunk[
                search_start:
            ]

            candidates = [
                tail.rfind("\n"),
                tail.rfind("다."),
                tail.rfind("함."),
                tail.rfind(". "),
            ]

            best_position = max(
                candidates
            )

            if best_position > 0:

                actual_end = (
                    search_start
                    + best_position
                    + 1
                )

                chunk = chunk[
                    :actual_end
                ]

                end = (
                    start
                    + actual_end
                )

        chunk = chunk.strip()

        if len(chunk) >= MIN_CHUNK_SIZE:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(
            end - overlap,
            start + 1,
        )

    return chunks


def create_chunks(
    documents: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:

    result = []

    for document in documents:

        chunks = chunk_text(
            document["text"],
            chunk_size,
            overlap,
        )

        total_chunks = len(chunks)

        for index, chunk in enumerate(
            chunks
        ):

            result.append(
                {
                    "chunk_id": (
                        f"{document['document_id']}"
                        f"_chunk_{index:04d}"
                    ),

                    "document_id": (
                        document[
                            "document_id"
                        ]
                    ),

                    "legal_domain": (
                        document[
                            "legal_domain"
                        ]
                    ),

                    "document_type": (
                        document[
                            "document_type"
                        ]
                    ),

                    "chunk_index": index,

                    "total_chunks": (
                        total_chunks
                    ),

                    "title": document.get(
                        "title"
                    ),

                    "authority": document.get(
                        "authority"
                    ),

                    "date": document.get(
                        "date"
                    ),

                    "text": chunk,

                    "metadata": document.get(
                        "metadata",
                        {},
                    ),
                }
            )

    return result


# ============================================================
# Deduplication
# ============================================================

def deduplicate_documents(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    unique = {}

    for document in documents:

        key = (
            document[
                "document_type"
            ],
            document[
                "document_id"
            ],
        )

        if key not in unique:

            unique[key] = document

            continue

        existing = unique[key]

        if (
            len(document["text"])
            >
            len(existing["text"])
        ):
            unique[key] = document

    return list(
        unique.values()
    )


def deduplicate_qa(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    seen = set()

    result = []

    for qa in records:

        key = (
            qa.get(
                "document_id"
            ),

            qa.get(
                "document_type"
            ),

            qa.get(
                "question"
            ),

            qa.get(
                "answer"
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(qa)

    return result


# ============================================================
# Preprocess
# ============================================================

def preprocess(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
):

    json_files = list(
        input_dir.rglob("*.json")
    )

    logger.info(
        "발견된 JSON 파일: %d",
        len(json_files),
    )

    documents = []
    qa_records = []

    failed_files = 0
    unknown_doc_class_files = 0

    raw_type_counts = {}

    for index, path in enumerate(
        json_files,
        start=1,
    ):

        data = read_json(path)

        if data is None:

            failed_files += 1
            continue

        info = data.get(
            "info",
            {}
        )

        doc_class = str(
            info.get(
                "doc_class",
                "",
            )
        )

        if doc_class not in {
            "1",
            "2",
            "3",
            "4",
            "5",
        }:

            unknown_doc_class_files += 1

        document = extract_document(
            data
        )

        if document:

            documents.append(
                document
            )

            document_type = (
                document[
                    "document_type"
                ]
            )

            raw_type_counts[
                document_type
            ] = (
                raw_type_counts.get(
                    document_type,
                    0,
                )
                + 1
            )

        qa = extract_qa(
            data=data,
            document=document,
        )

        if qa:
            qa_records.append(qa)

        if index % 5000 == 0:

            logger.info(
                "%d / %d 처리 완료",
                index,
                len(json_files),
            )

    raw_document_count = len(
        documents
    )

    documents = (
        deduplicate_documents(
            documents
        )
    )

    qa_records = (
        deduplicate_qa(
            qa_records
        )
    )

    chunks = create_chunks(
        documents,
        chunk_size,
        overlap,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        output_dir
        / "documents.jsonl",
        documents,
    )

    write_jsonl(
        output_dir
        / "chunks.jsonl",
        chunks,
    )

    write_jsonl(
        output_dir
        / "qa.jsonl",
        qa_records,
    )

    final_type_counts = {}

    for document in documents:

        document_type = (
            document[
                "document_type"
            ]
        )

        final_type_counts[
            document_type
        ] = (
            final_type_counts.get(
                document_type,
                0,
            )
            + 1
        )

    stats = {
        "legal_domain": (
            LEGAL_DOMAIN
        ),

        "input_json_files": (
            len(json_files)
        ),

        "failed_json_files": (
            failed_files
        ),

        "unknown_doc_class_files": (
            unknown_doc_class_files
        ),

        "raw_document_records": (
            raw_document_count
        ),

        "unique_documents": (
            len(documents)
        ),

        "qa_pairs": (
            len(qa_records)
        ),

        "chunks": (
            len(chunks)
        ),

        "chunk_size": (
            chunk_size
        ),

        "chunk_overlap": (
            overlap
        ),

        "documents_by_type": (
            final_type_counts
        ),
    }

    with (
        output_dir
        / "stats.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AI Hub 지식재산권법 QA "
            "RAG 전처리"
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        default=Path(
            "data/processed/"
            "intellectual_property"
        ),
        type=Path,
    )

    parser.add_argument(
        "--chunk-size",
        default=DEFAULT_CHUNK_SIZE,
        type=int,
    )

    parser.add_argument(
        "--chunk-overlap",
        default=DEFAULT_CHUNK_OVERLAP,
        type=int,
    )

    args = parser.parse_args()

    preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()