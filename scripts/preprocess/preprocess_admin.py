from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

# ============================================================
# Configuration
# ============================================================

LEGAL_DOMAIN = "administrative"

DEFAULT_CHUNK_SIZE = 3500
DEFAULT_CHUNK_OVERLAP = 400
MIN_CHUNK_SIZE = 200

# AI Hub administrative-law source prefix
# HJ_B = 법령
# HJ_P = 판례
# HJ_H = 법령해석
# HJ_K = 헌법재판 결정
SOURCE_TYPES = {
    "B": "statute",
    "P": "judgment",
    "H": "legal_interpretation",
    "K": "constitutional_decision",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ============================================================
# Basic utilities
# ============================================================

def normalize_text(value: Any) -> str:
    """
    문자열 정규화
    """
    if value is None:
        return ""
    if isinstance(value, str):
        value = str(value)

    value = value.strip()

    if not value:
        return ""
    if value.lower() == "nan":
        return ""
    
    value = value.replace("\x00", "")
    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return "\n".join(lines)

def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    except Exception as e:
        logger.error(f"[JSON 파일 읽기 실패] {path}: {e}")
        return None

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ============================================================
# Filename / source ID
# ============================================================

def extract_source_key(path: Path) -> str | None:
    
    stem = path.stem

    # QA suffix 제거
    if "_QA_" in stem:
        stem = stem.split("_QA_", 1)[0]
    
    pattern = re.compile(r"^HJ_([BPHK])_(.+)$", re.IGNORECASE)
    match = pattern.match(stem)

    if not match:
        return None

    source_type = match.group(1).upper()
    source_id = match.group(2)

    return f"HJ_{source_type}_{source_id}"

def parse_source_key(source_key: str) -> tuple[str, str]:

    parts = source_key.split("_", 2)

    if len(parts) != 3:
        raise ValueError(f"잘못된 source_key: {source_key}")

    source_code = parts[1].upper()
    source_id = parts[2]

    return source_code, source_id

def get_document_type(source_key: str) -> str:
    
    source_code, _ = parse_source_key(source_key)

    return SOURCE_TYPES.get(source_code, "legal_document")

def make_document_id(source_key: str) -> str:
    
    document_type = get_document_type(source_key)
    _, source_id = parse_source_key(source_key)

    return f"{LEGAL_DOMAIN}::{document_type}::{source_id}"

# ============================================================
# CSV reader
# ============================================================

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """
    AI Hub CSV 인코딩 대응
    """

    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]

    last_error = None
    
    for encoding in encodings:
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]

        except UnboundLocalError as e:
            last_error = e
            continue
        
        except Exception as e:
            logger.warning("CSV 읽기 실패: %s | %s", path, e)
            return []

    logger.warning("CSV 인코딩 확인 실패: %s | %s", path, last_error)
    
    return []

# ============================================================
# Source text reconstruction
# ============================================================

def safe_sentence_number(row: dict[str, Any]) -> int:
    
    value = normalize_text(row.get("문장번호"))
    
    try:
         return int(float(value))

    except Exception:
        return 9999999999

def reconstruct_simple_text(rows: list[dict[str, Any]]) -> str:
    """
    법령/판례/헌법재판 결정
    """
    rows = sorted(rows, key=safe_sentence_number)

    result = []

    for row in rows:
        content = normalize_text(row.get("내용"))

        if content:
            result.append(content)
    
    return "\n".join(result)

def reconstruct_interpretation_text(rows: list[dict[str, Any]]) -> str:
    """
    법령해석
    """
    rows = sorted(rows, key=safe_sentence_number)

    result = []
    previous_section = None
    
    for row in rows:
        section = normalize_text(row.get("구분"))
        content = normalize_text(row.get("내용"))

        if not content:
            continue

        if section and section != previous_section:
            result.append(f"[{section}]")

            previous_section = section
        
        result.append(content)

    return "\n".join(result)

def reconstruct_fulltext(source_key: str, rows: list[dict[str, Any]]) -> str:

    source_code, _ = parse_source_key(source_key)

    if source_code == "H":
        return reconstruct_interpretation_text(rows)

    return reconstruct_simple_text(rows)

# ============================================================
# Fulltext ID validation
# ============================================================

def get_csv_source_id(source_key: str, rows: list[dict[str, Any]]) -> str | None:
    """
    CSV 내부의 실제 ID를 추출하여 파일명과 QA metadata의 ID를 검증할 때 사용
    """

    if not rows:
        return None
    
    source_code, _ = parse_source_key(source_key)

    id_columns = {
        "B": "법령일련번호",
        "P": "판례일련번호",
        "H": "해석례일련번호",
        "K": "결정례일련번호"
    }

    column = id_columns.get(source_code)

    if not column:
        return None
    
    for row in rows:
        value = normalize_text(row.get(column))

        if value:
            return value
    
    return None

# ============================================================
# Load fulltext corpus
# ============================================================

def load_fulltext_files(fulltext_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    fulltext/*.csv 전체를 읽어 source_key index 생성
    """

    source_index = {}

    invalid_files = []

    csv_files = list(fulltext_dir.rglob("*.csv"))

    logger.info("원천데이터 csv 발견: %d", len(csv_files))

    for index, path in enumerate(csv_files, start=1):
        source_key  = extract_source_key(path)

        if not source_key:
            invalid_files.append(str(path))

            continue

        rows = read_csv_rows(path)

        if not rows:
            invalid_files.append(str(path))

            continue

        text = reconstruct_fulltext(source_key, rows)

        if not text:
            invalid_files.append(str(path))

            continue

        source_code, source_id = parse_source_key(source_key)

        csv_source_id = (get_csv_source_id(source_key, rows))

        source_index[source_key] = {
            "source_key": source_key,
            "source_code": source_code,
            "source_id": source_id,
            "csv_source_id": csv_source_id,
            "document_id": make_document_id(source_key),
            "document_type": get_document_type(source_key),
            "text": text,
            "source_path": str(path),
            "row_count": len(rows),
        }

        if index % 5000 == 0:
            logger.info("원천데이터 %d / %d 처리", index, len(csv_files))

    return source_index, invalid_files

# ============================================================
# QA metadata -> canonical metadata
# ============================================================

def get_qa_source_id(
    info: dict[str, Any],
    source_code: str,
) -> str | None:

    field_map = {
        "B": "lawId",
        "P": "precedId",
        "H": "interpreId",
        "K": "determintId",
    }

    field = field_map.get(
        source_code
    )

    if not field:
        return None

    value = info.get(
        field
    )

    if value is None:
        return None

    return str(value)


def build_metadata_from_qa(source_key: str, info: dict[str, Any]) -> dict[str, Any]:

    source_code, _ = parse_source_key(source_key)

    common_metadata = {
        "source": "AIHub",

        "source_key": source_key,

        "law_class": info.get("lawClass"),

        "docu_type": info.get("DocuType"),

        "full_text_available": info.get("fullText"),

        "section": info.get(
            "smClass"
        ),

        "sentence_type": info.get(
            "sentenceType"
        ),
    }

    # --------------------------------------------------------
    # B: 법령
    # --------------------------------------------------------

    if source_code == "B":

        return {
            "title": info.get(
                "title"
            ),

            "authority": info.get(
                "ministry"
            ),

            "date": info.get(
                "effectDate"
            ),

            "metadata": {
                **common_metadata,

                "law_id": info.get(
                    "lawId"
                ),

                "promulg_date": info.get(
                    "promulgDate"
                ),

                "effect_date": info.get(
                    "effectDate"
                ),

                "promulg_num": info.get(
                    "promulgNum"
                ),
            },
        }

    # --------------------------------------------------------
    # P: 판례
    # --------------------------------------------------------

    if source_code == "P":

        return {
            "title": info.get(
                "caseName"
            ),

            "authority": info.get(
                "courtName"
            ),

            "date": info.get(
                "sentenceDate"
            ),

            "metadata": {
                **common_metadata,

                "preced_id": info.get(
                    "precedId"
                ),

                "case_number": info.get(
                    "caseNum"
                ),

                "court_code": info.get(
                    "courtCode"
                ),

                "case_type": info.get(
                    "caseTypeName"
                ),
            },
        }

    # --------------------------------------------------------
    # H: 법령해석
    # --------------------------------------------------------

    if source_code == "H":

        return {
            "title": info.get(
                "agenda"
            ),

            "authority": info.get(
                "interpreMinName"
            ),

            "date": info.get(
                "interpreDate"
            ),

            "metadata": {
                **common_metadata,

                "interpretation_id": (
                    info.get(
                        "interpreId"
                    )
                ),

                "agenda_number": (
                    info.get(
                        "agendaNum"
                    )
                ),

                "interpretation_ministry_code": (
                    info.get(
                        "interpreMinCode"
                    )
                ),

                "question_ministry_name": (
                    info.get(
                        "questionMinName"
                    )
                ),
            },
        }

    # --------------------------------------------------------
    # K: 헌법재판 결정
    # --------------------------------------------------------

    if source_code == "K":

        return {
            "title": info.get(
                "caseName"
            ),

            # 현재 QA에서 courtCode가
            # '전원재판부'처럼 들어 있으므로 저장
            "authority": (
                "헌법재판소"
            ),

            "date": info.get(
                "finalDate"
            ),

            "metadata": {
                **common_metadata,

                "determination_id": (
                    info.get(
                        "determintId"
                    )
                ),

                "case_number": info.get(
                    "caseNum"
                ),

                "case_code": info.get(
                    "caseCode"
                ),

                "court_code": info.get(
                    "courtCode"
                ),
            },
        }

    return {
        "title": None,
        "authority": None,
        "date": None,
        "metadata": (
            common_metadata
        ),
    }


# ============================================================
# QA extraction
# ============================================================

def extract_qa_record(
    data: dict[str, Any],
    source_key: str,
    document_id: str,
    document_type: str,
) -> dict[str, Any] | None:

    label = data.get(
        "label",
        {}
    )

    question = normalize_text(
        label.get(
            "input"
        )
    )

    answer = normalize_text(
        label.get(
            "output"
        )
    )

    if not question or not answer:
        return None

    return {
        "document_id": (
            document_id
        ),

        "source_key": (
            source_key
        ),

        "legal_domain": (
            LEGAL_DOMAIN
        ),

        "document_type": (
            document_type
        ),

        "question": question,

        "answer": answer,

        "metadata": {
            "source": "AIHub",

            "instruction": label.get(
                "instruction"
            ),

            "origin_word_count": (
                label.get(
                    "originwordCnt"
                )
            ),

            "label_word_count": (
                label.get(
                    "labelwordCnt"
                )
            ),
        },
    }


# ============================================================
# Process QA and join with fulltext
# ============================================================

def join_qa_with_fulltext(
    qa_dir: Path,
    source_index: dict[
        str,
        dict[str, Any]
    ],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    """
    QA 파일을 fulltext source_index와 JOIN.

    Returns:
        documents
        qa_records
        unmatched_qa
        id_mismatches
    """

    qa_files = list(
        qa_dir.rglob(
            "*.json"
        )
    )

    logger.info(
        "QA JSON 발견: %d",
        len(qa_files),
    )

    qa_records = []

    unmatched_qa = []

    id_mismatches = []

    # source별 metadata enrichment
    enrichment = {}

    for index, path in enumerate(
        qa_files,
        start=1,
    ):

        source_key = (
            extract_source_key(
                path
            )
        )

        if not source_key:

            unmatched_qa.append(
                str(path)
            )
            continue

        source = source_index.get(
            source_key
        )

        if source is None:

            unmatched_qa.append(
                str(path)
            )
            continue

        data = read_json(
            path
        )

        if data is None:
            continue

        info = data.get(
            "info",
            {}
        )

        source_code, filename_id = (
            parse_source_key(
                source_key
            )
        )

        qa_source_id = (
            get_qa_source_id(
                info,
                source_code,
            )
        )

        # ----------------------------------------------------
        # ID validation
        # ----------------------------------------------------

        if (
            qa_source_id is not None
            and qa_source_id
            != filename_id
        ):

            # 법령 ID는 leading zero 때문에
            # 숫자 비교도 수행
            same_numeric_id = False

            try:

                same_numeric_id = (
                    int(qa_source_id)
                    == int(filename_id)
                )

            except Exception:
                pass

            if not same_numeric_id:

                id_mismatches.append(
                    {
                        "file": str(path),

                        "source_key": (
                            source_key
                        ),

                        "filename_id": (
                            filename_id
                        ),

                        "qa_source_id": (
                            qa_source_id
                        ),
                    }
                )

        # 같은 source에 QA가 여러 개 있을 수 있음.
        # 첫 QA metadata를 document metadata에 사용.
        if source_key not in enrichment:

            enrichment[
                source_key
            ] = build_metadata_from_qa(
                source_key,
                info,
            )

        qa = extract_qa_record(
            data=data,

            source_key=source_key,

            document_id=source[
                "document_id"
            ],

            document_type=source[
                "document_type"
            ],
        )

        if qa:
            qa_records.append(qa)

        if index % 5000 == 0:

            logger.info(
                "QA %d / %d 처리",
                index,
                len(qa_files),
            )

    # ========================================================
    # Canonical documents 생성
    # ========================================================

    documents = []

    for (source_key, source) in source_index.items():

        extra = enrichment.get(
            source_key,
            {
                "title": None,
                "authority": None,
                "date": None,
                "metadata": {},
            },
        )

        metadata = {
            **extra.get(
                "metadata",
                {},
            ),

            "source_key": (
                source_key
            ),

            "source_format": (
                "csv"
            ),

            "csv_source_id": (
                source.get(
                    "csv_source_id"
                )
            ),

            "source_row_count": (
                source.get(
                    "row_count"
                )
            ),
        }

        documents.append(
            {
                "document_id": (
                    source[
                        "document_id"
                    ]
                ),

                "source_key": (
                    source_key
                ),

                "legal_domain": (
                    LEGAL_DOMAIN
                ),

                "document_type": (
                    source[
                        "document_type"
                    ]
                ),

                "title": extra.get(
                    "title"
                ),

                "authority": (
                    extra.get(
                        "authority"
                    )
                ),

                "date": extra.get(
                    "date"
                ),

                "text": source[
                    "text"
                ],

                "metadata": metadata,
            }
        )

    return (
        documents,
        qa_records,
        unmatched_qa,
        id_mismatches,
    )


# ============================================================
# Deduplicate QA
# ============================================================

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
# Chunking
# ============================================================

BOUNDARIES = [
    ("\n", 1),
    ("다.", 2),
    ("함.", 2),
    (". ", 1),
]


def find_safe_boundary(
    chunk: str,
    search_window: int = 700,
) -> int | None:
    """
    chunk 후반부에서 가장 마지막 문장/문단 경계를 찾는다.

    이전 코드에서 '다.'의 마침표가 잘릴 수 있었던 문제도 수정.
    """

    search_start = max(
        0,
        len(chunk) - search_window,
    )

    tail = chunk[
        search_start:
    ]

    best_end = None

    for delimiter, length in BOUNDARIES:

        position = tail.rfind(
            delimiter
        )

        if position < 0:
            continue

        candidate_end = (
            search_start
            + position
            + length
        )

        if best_end is None or candidate_end > best_end:

            best_end = candidate_end

    return best_end


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:

    if chunk_size <= 0:

        raise ValueError("chunk_size는 0보다 커야 합니다.")

    if overlap < 0 or overlap >= chunk_size:

        raise ValueError(
            "overlap은 0 이상이며 "
            "chunk_size보다 작아야 합니다."
        )

    chunks = []

    start = 0

    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        raw_chunk = text[start:end]

        if end < text_length:

            boundary = find_safe_boundary(raw_chunk)

            if boundary is not None and boundary >= MIN_CHUNK_SIZE:

                raw_chunk = raw_chunk[:boundary]

                end = start + boundary

        chunk = raw_chunk.strip()

        if len(chunk) >= MIN_CHUNK_SIZE:

            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(documents: list[dict[str, Any]], chunk_size: int, overlap: int) -> list[dict[str, Any]]:

    result = []

    for document in documents:

        chunks = chunk_text(text=document["text"], chunk_size=chunk_size, overlap=overlap)

        total_chunks = len(chunks)

        for index, chunk in enumerate(chunks):

            # source_key 포함:
            # 다른 domain/document와 ID collision 방지
            chunk_id = (
                f"{LEGAL_DOMAIN}"
                f"::{document['source_key']}"
                f"::chunk_{index:04d}"
            )

            result.append(
                {
                    "chunk_id": chunk_id,

                    "document_id": document["document_id"],

                    "source_key": document["source_key"],

                    "legal_domain": (document["legal_domain"]),

                    "document_type": (document["document_type"]),

                    "chunk_index": index,

                    "total_chunks": total_chunks,

                    "title": document.get("title"),

                    "authority": document.get("authority"),

                    "date": document.get("date"),

                    "text": chunk,

                    "metadata": document.get("metadata", {}),
                }
            )

    return result


# ============================================================
# Stats
# ============================================================

def count_document_types(
    records: list[dict[str, Any]],
) -> dict[str, int]:

    result = {}

    for record in records:

        document_type = (record.get("document_type", "unknown"))

        result[document_type] = result.get(document_type, 0) + 1

    return result


# ============================================================
# Main preprocess
# ============================================================

def preprocess(input_dir: Path, output_dir: Path, chunk_size: int, overlap: int) -> None:

    fulltext_dir = input_dir / "fulltext"

    qa_dir = input_dir / "qa"

    if not fulltext_dir.exists():
        raise FileNotFoundError(f"fulltext 폴더 없음: {fulltext_dir}")

    if not qa_dir.exists():
        raise FileNotFoundError(f"qa 폴더 없음: {qa_dir}")

    # ========================================================
    # 1. Fulltext load
    # ========================================================

    (source_index, invalid_fulltext_files) = load_fulltext_files(fulltext_dir)

    # ========================================================
    # 2. QA JOIN
    # ========================================================

    (documents, qa_records, unmatched_qa, id_mismatches) = join_qa_with_fulltext(qa_dir=qa_dir, source_index=source_index)

    qa_records = deduplicate_qa(qa_records)

    # ========================================================
    # 3. Chunk
    # ========================================================

    chunks = create_chunks(
        documents=documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # ========================================================
    # 4. Output
    # ========================================================

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

    # ========================================================
    # 5. Match report
    # ========================================================

    matched_source_keys = {
        qa["source_key"]
        for qa in qa_records
    }

    fulltext_source_keys = set(
        source_index.keys()
    )

    unmatched_fulltext = sorted(
        fulltext_source_keys
        - matched_source_keys
    )

    match_report = {
        "unmatched_qa_files": (
            unmatched_qa
        ),

        "unmatched_fulltext_source_keys": (
            unmatched_fulltext
        ),

        "invalid_fulltext_files": (
            invalid_fulltext_files
        ),

        "id_mismatches": (
            id_mismatches
        ),
    }

    with (
        output_dir
        / "match_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            match_report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # 6. Stats
    # ========================================================

    stats = {
        "legal_domain": (
            LEGAL_DOMAIN
        ),

        "fulltext_documents": (
            len(source_index)
        ),

        "canonical_documents": (
            len(documents)
        ),

        "qa_pairs": (
            len(qa_records)
        ),

        "chunks": (
            len(chunks)
        ),

        "matched_source_documents": (
            len(matched_source_keys)
        ),

        "unmatched_fulltext_documents": (
            len(unmatched_fulltext)
        ),

        "unmatched_qa_files": (
            len(unmatched_qa)
        ),

        "invalid_fulltext_files": (
            len(invalid_fulltext_files)
        ),

        "id_mismatches": (
            len(id_mismatches)
        ),

        "documents_by_type": (
            count_document_types(
                documents
            )
        ),

        "chunks_by_type": (
            count_document_types(
                chunks
            )
        ),

        "chunk_size": (
            chunk_size
        ),

        "chunk_overlap": (
            overlap
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
        "전처리 완료"
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
            "AI Hub 행정법 fulltext + QA "
            "RAG 전처리"
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "fulltext/ 및 qa/를 포함하는 "
            "행정법 데이터 경로"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=Path(
            "data/processed/"
            "administrative"
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