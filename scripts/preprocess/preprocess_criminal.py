from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# Configuration
# ============================================================

LEGAL_DOMAIN = "criminal"

# 형사법 QA sample에서 확인된 lawClass
EXPECTED_LAW_CLASS = "02"

DEFAULT_CHUNK_SIZE = 3500
DEFAULT_CHUNK_OVERLAP = 400

# 지나치게 짧은 마지막 chunk를 이전 chunk와 병합할 때 사용
MIN_CHUNK_SIZE = 200


# 파일명의 source code
SOURCE_TYPES = {
    "B": "statute",
    "P": "judgment",
    "H": "legal_interpretation",
    "K": "constitutional_decision",
}


# QA의 DocuType과 파일명 source code 대응
DOCUTYPE_BY_SOURCE = {
    "B": "01",
    "P": "02",
    "H": "03",
    "K": "04",
}


# Fulltext CSV 내부 ID column
CSV_ID_COLUMNS = {
    "B": "법령일련번호",
    "P": "판례일련번호",
    "H": "해석례일련번호",
    "K": "결정례일련번호",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Basic utility
# ============================================================

def normalize_text(value: Any) -> str:
    """
    None, 공백, 개행 등을 정리하여 문자열로 통일.
    """

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if not value:
        return ""

    if value.lower() == "nan":
        return ""

    value = value.replace("\x00", "")
    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    return "\n".join(
        line.strip()
        for line in value.splitlines()
        if line.strip()
    )


def read_json(
    path: Path,
) -> dict[str, Any] | None:

    try:

        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as f:

            data = json.load(f)

        if not isinstance(data, dict):
            return None

        return data

    except Exception as e:

        logger.warning(
            "JSON 읽기 실패: %s | %s",
            path,
            e,
        )

        return None


def write_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Source key handling
# ============================================================

def strip_copy_suffix(
    stem: str,
) -> str:
    """
    다운로드 과정에서 생성되는:

        HS_K_238(1)
        HS_K_238(2)

    등을:

        HS_K_238

    로 정규화한다.
    """

    return re.sub(
        r"\(\d+\)$",
        "",
        stem,
    ).strip()


def extract_source_key(
    path: Path,
) -> str | None:
    """
    예:

    HS_B_000006.csv
        -> HS_B_000006

    HS_B_000006_QA_3.json
        -> HS_B_000006

    HS_K_238(1).csv
        -> HS_K_238
    """

    stem = path.stem

    # QA suffix 제거
    if "_QA_" in stem:

        stem = stem.split(
            "_QA_",
            1,
        )[0]

    stem = strip_copy_suffix(
        stem
    )

    match = re.fullmatch(
        r"HS_([BPHK])_(.+)",
        stem,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    source_code = (
        match.group(1)
        .upper()
    )

    source_id = (
        match.group(2)
        .strip()
    )

    if not source_id:
        return None

    return (
        f"HS_{source_code}_{source_id}"
    )


def parse_source_key(
    source_key: str,
) -> tuple[str, str]:
    """
    HS_B_000006
        -> ("B", "000006")
    """

    parts = source_key.split(
        "_",
        2,
    )

    if (
        len(parts) != 3
        or parts[0] != "HS"
    ):

        raise ValueError(
            f"잘못된 source_key: {source_key}"
        )

    return (
        parts[1].upper(),
        parts[2],
    )


def get_document_type(
    source_key: str,
) -> str:

    source_code, _ = (
        parse_source_key(
            source_key
        )
    )

    return SOURCE_TYPES[
        source_code
    ]


def make_document_id(
    source_key: str,
) -> str:
    """
    예:

    HS_P_64516

    -> criminal::judgment::64516
    """

    source_code, source_id = (
        parse_source_key(
            source_key
        )
    )

    document_type = (
        SOURCE_TYPES[
            source_code
        ]
    )

    return (
        f"{LEGAL_DOMAIN}"
        f"::{document_type}"
        f"::{source_id}"
    )


# ============================================================
# ID comparison
# ============================================================

def ids_equivalent(
    a: Any,
    b: Any,
) -> bool:
    """
    다음을 동일한 ID로 처리한다.

        000006
        6

    값이 한쪽에 없는 경우에는
    여기서는 mismatch로 간주하지 않는다.
    """

    a_text = normalize_text(a)
    b_text = normalize_text(b)

    if not a_text or not b_text:
        return True

    if a_text == b_text:
        return True

    try:

        return (
            int(a_text)
            == int(b_text)
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


# ============================================================
# CSV loading
# ============================================================

def read_csv_rows(
    path: Path,
) -> tuple[
    list[dict[str, str]],
    str | None,
]:
    """
    AI Hub CSV encoding 대응.
    """

    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    )

    for encoding in encodings:

        try:

            with path.open(
                "r",
                encoding=encoding,
                newline="",
            ) as f:

                reader = csv.DictReader(
                    f
                )

                if not reader.fieldnames:
                    return [], encoding

                return (
                    [
                        dict(row)
                        for row in reader
                    ],
                    encoding,
                )

        except UnicodeDecodeError:
            continue

        except Exception as e:

            logger.warning(
                "CSV 읽기 실패: %s | %s",
                path,
                e,
            )

            return [], None

    logger.warning(
        "CSV 인코딩 확인 실패: %s",
        path,
    )

    return [], None


# ============================================================
# CSV schema validation
# ============================================================

def validate_csv_schema(
    source_code: str,
    rows: list[dict[str, Any]],
) -> list[str]:

    if not rows:
        return [
            "empty_csv"
        ]

    required_columns = {
        CSV_ID_COLUMNS[
            source_code
        ],
        "구분",
        "문장번호",
        "내용",
    }

    available_columns = set(
        rows[0].keys()
    )

    missing = sorted(
        required_columns
        - available_columns
    )

    return [
        f"missing_column:{name}"
        for name in missing
    ]


# ============================================================
# Sentence ordering
# ============================================================

def safe_sentence_number(
    row: dict[str, Any],
) -> tuple[int, str]:

    raw = normalize_text(
        row.get(
            "문장번호"
        )
    )

    try:

        return (
            int(
                float(raw)
            ),
            raw,
        )

    except (
        TypeError,
        ValueError,
    ):

        return (
            10**12,
            raw,
        )


# ============================================================
# Fulltext reconstruction
# ============================================================

def reconstruct_simple_text(
    rows: list[dict[str, Any]],
) -> str:
    """
    법령 / 판례 / 헌법재판 결정.

    문장번호 순서대로 내용을 연결한다.
    """

    ordered = sorted(
        rows,
        key=safe_sentence_number,
    )

    contents = []

    for row in ordered:

        content = normalize_text(
            row.get(
                "내용"
            )
        )

        if content:
            contents.append(
                content
            )

    return "\n".join(
        contents
    )


def reconstruct_interpretation_text(
    rows: list[dict[str, Any]],
) -> str:
    """
    법령해석은:

        질의요지
        회답
        이유

    구조를 보존한다.
    """

    ordered = sorted(
        rows,
        key=safe_sentence_number,
    )

    result = []

    previous_section = None

    for row in ordered:

        section = normalize_text(
            row.get(
                "구분"
            )
        )

        content = normalize_text(
            row.get(
                "내용"
            )
        )

        if not content:
            continue

        if (
            section
            and section
            != previous_section
        ):

            result.append(
                f"[{section}]"
            )

            previous_section = (
                section
            )

        result.append(
            content
        )

    return "\n".join(
        result
    )


def reconstruct_fulltext(
    source_key: str,
    rows: list[dict[str, Any]],
) -> str:

    source_code, _ = (
        parse_source_key(
            source_key
        )
    )

    if source_code == "H":

        return (
            reconstruct_interpretation_text(
                rows
            )
        )

    return reconstruct_simple_text(
        rows
    )


# ============================================================
# CSV internal ID
# ============================================================

def get_csv_source_ids(
    source_code: str,
    rows: list[dict[str, Any]],
) -> list[str]:

    column = CSV_ID_COLUMNS[
        source_code
    ]

    values = {
        normalize_text(
            row.get(column)
        )
        for row in rows
    }

    values.discard("")

    return sorted(
        values
    )


# ============================================================
# Duplicate filename ordering
# ============================================================

def source_path_priority(
    path: Path,
) -> tuple[int, str]:
    """
    HS_K_238.csv를
    HS_K_238(1).csv보다 우선한다.
    """

    copy_suffix = re.search(
        r"\(\d+\)$",
        path.stem,
    )

    return (
        1 if copy_suffix else 0,
        str(path),
    )


# ============================================================
# Fulltext indexing
# ============================================================

def load_fulltext_files(
    fulltext_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    source_index: dict[
        str,
        dict[str, Any],
    ] = {}

    invalid_files = []

    duplicate_sources = []

    csv_files = sorted(
        fulltext_dir.rglob(
            "*.csv"
        ),
        key=source_path_priority,
    )

    logger.info(
        "Fulltext CSV 발견: %d",
        len(csv_files),
    )

    for index, path in enumerate(
        csv_files,
        start=1,
    ):

        source_key = (
            extract_source_key(
                path
            )
        )

        # ----------------------------------------------------
        # filename validation
        # ----------------------------------------------------

        if not source_key:

            invalid_files.append(
                {
                    "file": str(path),
                    "reasons": [
                        "invalid_filename"
                    ],
                }
            )

            continue

        source_code, filename_id = (
            parse_source_key(
                source_key
            )
        )

        # ----------------------------------------------------
        # CSV load
        # ----------------------------------------------------

        rows, encoding = (
            read_csv_rows(
                path
            )
        )

        # ----------------------------------------------------
        # schema validation
        # ----------------------------------------------------

        reasons = (
            validate_csv_schema(
                source_code,
                rows,
            )
        )

        if reasons:

            invalid_files.append(
                {
                    "file": str(path),
                    "source_key": (
                        source_key
                    ),
                    "reasons": reasons,
                }
            )

            continue

        # ----------------------------------------------------
        # CSV internal ID
        # ----------------------------------------------------

        csv_ids = (
            get_csv_source_ids(
                source_code,
                rows,
            )
        )

        if len(csv_ids) != 1:

            invalid_files.append(
                {
                    "file": str(path),

                    "source_key": (
                        source_key
                    ),

                    "reasons": [
                        "multiple_or_missing_internal_ids"
                    ],

                    "csv_source_ids": (
                        csv_ids
                    ),
                }
            )

            continue

        csv_source_id = (
            csv_ids[0]
        )

        # ----------------------------------------------------
        # filename ID vs CSV ID
        # ----------------------------------------------------

        if not ids_equivalent(
            filename_id,
            csv_source_id,
        ):

            invalid_files.append(
                {
                    "file": str(path),

                    "source_key": (
                        source_key
                    ),

                    "reasons": [
                        "filename_csv_id_mismatch"
                    ],

                    "filename_id": (
                        filename_id
                    ),

                    "csv_source_id": (
                        csv_source_id
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # text reconstruction
        # ----------------------------------------------------

        text = reconstruct_fulltext(
            source_key,
            rows,
        )

        if not text:

            invalid_files.append(
                {
                    "file": str(path),

                    "source_key": (
                        source_key
                    ),

                    "reasons": [
                        "empty_reconstructed_text"
                    ],
                }
            )

            continue

        candidate = {
            "source_key": (
                source_key
            ),

            "source_code": (
                source_code
            ),

            "source_id": (
                filename_id
            ),

            "csv_source_id": (
                csv_source_id
            ),

            "document_id": (
                make_document_id(
                    source_key
                )
            ),

            "document_type": (
                SOURCE_TYPES[
                    source_code
                ]
            ),

            "text": text,

            "source_path": (
                str(path)
            ),

            "row_count": (
                len(rows)
            ),

            "encoding": (
                encoding
            ),
        }

        # ----------------------------------------------------
        # duplicate source key
        # ----------------------------------------------------

        existing = source_index.get(
            source_key
        )

        if existing is not None:

            duplicate_sources.append(
                {
                    "source_key": (
                        source_key
                    ),

                    "kept_file": (
                        existing[
                            "source_path"
                        ]
                    ),

                    "duplicate_file": (
                        str(path)
                    ),

                    "same_text": (
                        existing["text"]
                        == text
                    ),
                }
            )

            # 자동 overwrite 하지 않는다.
            # 정렬상 원본 filename이 먼저 선택된다.
            continue

        source_index[
            source_key
        ] = candidate

        if index % 5000 == 0:

            logger.info(
                "Fulltext %d / %d 처리",
                index,
                len(csv_files),
            )

    return (
        source_index,
        invalid_files,
        duplicate_sources,
    )


# ============================================================
# QA source ID
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

    value = info.get(
        field_map[
            source_code
        ]
    )

    if value is None:
        return None

    return str(
        value
    )


# ============================================================
# QA validation
# ============================================================

def validate_qa_identity(
    info: dict[str, Any],
    source_key: str,
    csv_source_id: str | None,
) -> list[dict[str, Any]]:

    issues = []

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

    # --------------------------------------------------------
    # filename ID vs QA ID
    # --------------------------------------------------------

    if not ids_equivalent(
        filename_id,
        qa_source_id,
    ):

        issues.append(
            {
                "kind": (
                    "filename_qa_id_mismatch"
                ),

                "filename_id": (
                    filename_id
                ),

                "qa_source_id": (
                    qa_source_id
                ),
            }
        )

    # --------------------------------------------------------
    # filename ID vs CSV ID
    # --------------------------------------------------------

    if (
        csv_source_id
        and not ids_equivalent(
            filename_id,
            csv_source_id,
        )
    ):

        issues.append(
            {
                "kind": (
                    "filename_csv_id_mismatch"
                ),

                "filename_id": (
                    filename_id
                ),

                "csv_source_id": (
                    csv_source_id
                ),
            }
        )

    # --------------------------------------------------------
    # source code vs DocuType
    # --------------------------------------------------------

    expected_docu_type = (
        DOCUTYPE_BY_SOURCE[
            source_code
        ]
    )

    actual_docu_type = (
        normalize_text(
            info.get(
                "DocuType"
            )
        )
        .zfill(2)
    )

    if (
        actual_docu_type
        != expected_docu_type
    ):

        issues.append(
            {
                "kind": (
                    "source_docutype_mismatch"
                ),

                "source_code": (
                    source_code
                ),

                "expected_docu_type": (
                    expected_docu_type
                ),

                "actual_docu_type": (
                    actual_docu_type
                ),
            }
        )

    # --------------------------------------------------------
    # lawClass
    # --------------------------------------------------------

    law_class = normalize_text(
        info.get(
            "lawClass"
        )
    )

    if (
        law_class
        and law_class
        != EXPECTED_LAW_CLASS
    ):

        issues.append(
            {
                "kind": (
                    "unexpected_law_class"
                ),

                "expected_law_class": (
                    EXPECTED_LAW_CLASS
                ),

                "actual_law_class": (
                    law_class
                ),
            }
        )

    return issues


# ============================================================
# QA metadata
# ============================================================

def build_metadata_from_qa(
    source_key: str,
    info: dict[str, Any],
) -> dict[str, Any]:

    source_code, _ = (
        parse_source_key(
            source_key
        )
    )

    common = {
        "source": "AIHub",

        "source_key": (
            source_key
        ),

        "law_class": info.get(
            "lawClass"
        ),

        "docu_type": info.get(
            "DocuType"
        ),

        # fullText가 N이라도 실제 CSV가 존재할 수 있으므로
        # 단순 metadata flag로만 저장한다.
        "full_text_flag": info.get(
            "fullText"
        ),

        "section": info.get(
            "smClass"
        ),

        "sentence_type": info.get(
            "sentenceType"
        ),
    }

    # ========================================================
    # B: 법령
    # ========================================================

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
                **common,

                "law_id": info.get(
                    "lawId"
                ),

                "promulg_date": (
                    info.get(
                        "promulgDate"
                    )
                ),

                "effect_date": (
                    info.get(
                        "effectDate"
                    )
                ),

                "promulg_num": (
                    info.get(
                        "promulgNum"
                    )
                ),

                "ministry": info.get(
                    "ministry"
                ),
            },
        }

    # ========================================================
    # P: 판례
    # ========================================================

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
                **common,

                "preced_id": (
                    info.get(
                        "precedId"
                    )
                ),

                "case_number": (
                    info.get(
                        "caseNum"
                    )
                ),

                "court_code": (
                    info.get(
                        "courtCode"
                    )
                ),

                "case_type": (
                    info.get(
                        "caseTypeName"
                    )
                ),
            },
        }

    # ========================================================
    # H: 법령해석
    # ========================================================

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
                **common,

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

                "interpretation_ministry_name": (
                    info.get(
                        "interpreMinName"
                    )
                ),

                "question_ministry_code": (
                    info.get(
                        "questionMinCode"
                    )
                ),

                "question_ministry_name": (
                    info.get(
                        "questionMinName"
                    )
                ),
            },
        }

    # ========================================================
    # K: 헌법재판 결정
    # ========================================================

    if source_code == "K":

        return {
            "title": info.get(
                "caseName"
            ),

            "authority": (
                "헌법재판소"
            ),

            "date": info.get(
                "finalDate"
            ),

            "metadata": {
                **common,

                "determination_id": (
                    info.get(
                        "determintId"
                    )
                ),

                "case_number": (
                    info.get(
                        "caseNum"
                    )
                ),

                "case_code": (
                    info.get(
                        "caseCode"
                    )
                ),

                "court_code": (
                    info.get(
                        "courtCode"
                    )
                ),
            },
        }

    raise ValueError(
        f"지원하지 않는 source code: "
        f"{source_code}"
    )


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
        {},
    )

    if not isinstance(
        label,
        dict,
    ):
        return None

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

        "question": (
            question
        ),

        "answer": (
            answer
        ),

        "metadata": {
            "source": "AIHub",

            "instruction": (
                label.get(
                    "instruction"
                )
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
# Metadata comparison
# ============================================================

def metadata_signature(
    item: dict[str, Any],
) -> str:

    return json.dumps(
        item,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


# ============================================================
# QA + Fulltext JOIN
# ============================================================

def join_qa_with_fulltext(
    qa_dir: Path,
    source_index: dict[
        str,
        dict[str, Any],
    ],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    qa_files = sorted(
        qa_dir.rglob(
            "*.json"
        ),
        key=lambda p: str(p),
    )

    logger.info(
        "QA JSON 발견: %d",
        len(qa_files),
    )

    qa_records = []

    unmatched_qa = []

    validation_issues = []

    metadata_conflicts = []

    # source document마다 하나의 대표 metadata 저장
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
                {
                    "file": str(path),
                    "reason": (
                        "invalid_filename"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Matching fulltext
        # ----------------------------------------------------

        source = source_index.get(
            source_key
        )

        if source is None:

            unmatched_qa.append(
                {
                    "file": str(path),

                    "source_key": (
                        source_key
                    ),

                    "reason": (
                        "fulltext_not_found"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # JSON load
        # ----------------------------------------------------

        data = read_json(
            path
        )

        if data is None:

            unmatched_qa.append(
                {
                    "file": str(path),

                    "source_key": (
                        source_key
                    ),

                    "reason": (
                        "json_read_failed"
                    ),
                }
            )

            continue

        info = data.get(
            "info",
            {},
        )

        if not isinstance(
            info,
            dict,
        ):

            unmatched_qa.append(
                {
                    "file": str(path),

                    "source_key": (
                        source_key
                    ),

                    "reason": (
                        "missing_info_object"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # ID / type validation
        # ----------------------------------------------------

        issues = validate_qa_identity(
            info=info,

            source_key=source_key,

            csv_source_id=(
                source.get(
                    "csv_source_id"
                )
            ),
        )

        for issue in issues:

            validation_issues.append(
                {
                    "file": str(path),

                    "source_key": (
                        source_key
                    ),

                    **issue,
                }
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        current_metadata = (
            build_metadata_from_qa(
                source_key,
                info,
            )
        )

        existing_metadata = (
            enrichment.get(
                source_key
            )
        )

        if existing_metadata is None:

            enrichment[
                source_key
            ] = current_metadata

        elif (
            metadata_signature(
                existing_metadata
            )
            != metadata_signature(
                current_metadata
            )
        ):

            metadata_conflicts.append(
                {
                    "file": str(path),

                    "source_key": (
                        source_key
                    ),

                    "reason": (
                        "metadata_differs_between_qa_records"
                    ),
                }
            )

        # ----------------------------------------------------
        # QA pair
        # ----------------------------------------------------

        qa = extract_qa_record(
            data=data,

            source_key=source_key,

            document_id=(
                source[
                    "document_id"
                ]
            ),

            document_type=(
                source[
                    "document_type"
                ]
            ),
        )

        if qa is not None:

            qa_records.append(
                qa
            )

        if index % 5000 == 0:

            logger.info(
                "QA %d / %d 처리",
                index,
                len(qa_files),
            )

    # ========================================================
    # Canonical document 생성
    # ========================================================

    documents = []

    for (
        source_key,
        source,
    ) in source_index.items():

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

            "source_encoding": (
                source.get(
                    "encoding"
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

                "title": (
                    extra.get(
                        "title"
                    )
                ),

                "authority": (
                    extra.get(
                        "authority"
                    )
                ),

                "date": (
                    extra.get(
                        "date"
                    )
                ),

                "text": (
                    source[
                        "text"
                    ]
                ),

                "metadata": (
                    metadata
                ),
            }
        )

    documents.sort(
        key=lambda item: (
            item["source_key"]
        )
    )

    all_validation_issues = (
        validation_issues
        + metadata_conflicts
    )

    return (
        documents,
        qa_records,
        unmatched_qa,
        all_validation_issues,
    )


# ============================================================
# QA deduplication
# ============================================================

def deduplicate_qa(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    int,
]:

    seen = set()

    result = []

    duplicate_count = 0

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

            duplicate_count += 1

            continue

        seen.add(
            key
        )

        result.append(
            qa
        )

    return (
        result,
        duplicate_count,
    )


# ============================================================
# Chunking
# ============================================================

BOUNDARIES = (
    "\n",
    "다.",
    "함.",
    "니다.",
    ". ",
)


def find_safe_boundary(
    chunk: str,
    search_window: int = 900,
) -> int | None:
    """
    chunk 끝부분에서 가장 가까운
    문장/문단 경계를 찾는다.

    delimiter 길이를 len()으로 계산하여
    '다.' 등이 잘리는 문제를 방지한다.
    """

    search_start = max(
        0,
        len(chunk)
        - search_window,
    )

    tail = chunk[
        search_start:
    ]

    best_end = None

    for delimiter in BOUNDARIES:

        position = tail.rfind(
            delimiter
        )

        if position < 0:
            continue

        candidate_end = (
            search_start
            + position
            + len(delimiter)
        )

        if (
            best_end is None
            or candidate_end
            > best_end
        ):

            best_end = (
                candidate_end
            )

    return best_end


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:

    if chunk_size <= 0:

        raise ValueError(
            "chunk_size는 "
            "0보다 커야 합니다."
        )

    if (
        overlap < 0
        or overlap
        >= chunk_size
    ):

        raise ValueError(
            "overlap은 0 이상이며 "
            "chunk_size보다 작아야 합니다."
        )

    text = normalize_text(
        text
    )

    if not text:
        return []

    if len(text) <= chunk_size:

        return [
            text
        ]

    chunks = []

    start = 0

    text_length = len(
        text
    )

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length,
        )

        raw_chunk = text[
            start:end
        ]

        # 마지막 chunk가 아니라면
        # 문장/문단 경계 탐색
        if end < text_length:

            boundary = (
                find_safe_boundary(
                    raw_chunk
                )
            )

            if (
                boundary is not None
                and boundary
                >= MIN_CHUNK_SIZE
            ):

                raw_chunk = (
                    raw_chunk[
                        :boundary
                    ]
                )

                end = (
                    start
                    + boundary
                )

        chunk = (
            raw_chunk.strip()
        )

        if chunk:

            chunks.append(
                chunk
            )

        if end >= text_length:
            break

        start = max(
            end - overlap,
            start + 1,
        )

    # --------------------------------------------------------
    # 작은 마지막 chunk가 버려지지 않도록 병합
    # --------------------------------------------------------

    if (
        len(chunks) >= 2
        and len(
            chunks[-1]
        )
        < MIN_CHUNK_SIZE
    ):

        tail = chunks.pop()

        # overlap 때문에 이미 포함된 경우
        # 중복 추가 방지
        if tail not in chunks[-1]:

            chunks[-1] = (
                f"{chunks[-1]}\n{tail}"
                .strip()
            )

    return chunks


# ============================================================
# Chunk records
# ============================================================

def create_chunks(
    documents: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:

    result = []

    for document in documents:

        chunks = chunk_text(
            text=document[
                "text"
            ],

            chunk_size=chunk_size,

            overlap=overlap,
        )

        total_chunks = len(
            chunks
        )

        for index, chunk in enumerate(
            chunks
        ):

            chunk_id = (
                f"{LEGAL_DOMAIN}"
                f"::{document['source_key']}"
                f"::chunk_{index:04d}"
            )

            result.append(
                {
                    "chunk_id": (
                        chunk_id
                    ),

                    "document_id": (
                        document[
                            "document_id"
                        ]
                    ),

                    "source_key": (
                        document[
                            "source_key"
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

                    "chunk_index": (
                        index
                    ),

                    "total_chunks": (
                        total_chunks
                    ),

                    "title": (
                        document.get(
                            "title"
                        )
                    ),

                    "authority": (
                        document.get(
                            "authority"
                        )
                    ),

                    "date": (
                        document.get(
                            "date"
                        )
                    ),

                    "text": (
                        chunk
                    ),

                    "metadata": (
                        document.get(
                            "metadata",
                            {},
                        )
                    ),
                }
            )

    return result


# ============================================================
# Statistics
# ============================================================

def count_by_type(
    records: list[dict[str, Any]],
) -> dict[str, int]:

    return dict(
        Counter(
            record.get(
                "document_type",
                "unknown",
            )
            for record in records
        )
    )


# ============================================================
# Main preprocessing
# ============================================================

def preprocess(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
) -> None:

    fulltext_dir = (
        input_dir
        / "fulltext"
    )

    qa_dir = (
        input_dir
        / "qa"
    )

    # --------------------------------------------------------
    # Folder validation
    # --------------------------------------------------------

    if not fulltext_dir.is_dir():

        raise FileNotFoundError(
            f"fulltext 폴더 없음: "
            f"{fulltext_dir}"
        )

    if not qa_dir.is_dir():

        raise FileNotFoundError(
            f"qa 폴더 없음: "
            f"{qa_dir}"
        )

    # ========================================================
    # 1. Fulltext parsing
    # ========================================================

    (
        source_index,
        invalid_fulltext_files,
        duplicate_sources,
    ) = load_fulltext_files(
        fulltext_dir
    )

    # ========================================================
    # 2. QA JOIN
    # ========================================================

    (
        documents,
        qa_records,
        unmatched_qa,
        qa_validation_issues,
    ) = join_qa_with_fulltext(
        qa_dir=qa_dir,
        source_index=source_index,
    )

    # ========================================================
    # 3. QA dedup
    # ========================================================

    (
        qa_records,
        duplicate_qa_count,
    ) = deduplicate_qa(
        qa_records
    )

    # ========================================================
    # 4. Chunking
    # ========================================================

    chunks = create_chunks(
        documents=documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # ========================================================
    # 5. Write output
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
    # 6. Match report
    # ========================================================

    matched_source_keys = {
        qa[
            "source_key"
        ]
        for qa in qa_records
    }

    unmatched_fulltext = sorted(
        set(
            source_index
        )
        - matched_source_keys
    )

    match_report = {
        "unmatched_qa": (
            unmatched_qa
        ),

        "unmatched_fulltext_source_keys": (
            unmatched_fulltext
        ),

        "invalid_fulltext_files": (
            invalid_fulltext_files
        ),

        "duplicate_fulltext_sources": (
            duplicate_sources
        ),

        "qa_validation_issues": (
            qa_validation_issues
        ),
    }

    write_json(
        output_dir
        / "match_report.json",
        match_report,
    )

    # ========================================================
    # 7. Stats
    # ========================================================

    stats = {
        "legal_domain": (
            LEGAL_DOMAIN
        ),

        "fulltext_documents": (
            len(
                source_index
            )
        ),

        "canonical_documents": (
            len(
                documents
            )
        ),

        "qa_pairs": (
            len(
                qa_records
            )
        ),

        "chunks": (
            len(
                chunks
            )
        ),

        "matched_source_documents": (
            len(
                matched_source_keys
            )
        ),

        "unmatched_fulltext_documents": (
            len(
                unmatched_fulltext
            )
        ),

        "unmatched_qa_files": (
            len(
                unmatched_qa
            )
        ),

        "invalid_fulltext_files": (
            len(
                invalid_fulltext_files
            )
        ),

        "duplicate_fulltext_sources": (
            len(
                duplicate_sources
            )
        ),

        "qa_validation_issues": (
            len(
                qa_validation_issues
            )
        ),

        "duplicate_qa_removed": (
            duplicate_qa_count
        ),

        "documents_by_type": (
            count_by_type(
                documents
            )
        ),

        "chunks_by_type": (
            count_by_type(
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

    write_json(
        output_dir
        / "stats.json",
        stats,
    )

    logger.info(
        "전처리 완료"
    )

    logger.info(
        "%s",
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "AI Hub 형사법 "
            "fulltext + QA RAG 전처리"
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help=(
            "fulltext/ 및 qa/를 포함하는 "
            "형사법 데이터 경로"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=Path(
            "data/processed/criminal"
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