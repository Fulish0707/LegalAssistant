from pathlib import Path

from legal_rag.ingestion.detect import detect_schema, known_schemas
from legal_rag.ingestion.loaders import discover_raw_records

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"


def test_discover_yields_one_record_per_json_object():
    records = list(discover_raw_records(FIXTURES))
    files = {r.file for r in records}
    assert "aihub_judgment/judgment_001.json" in files
    assert all(len(r.file_sha256) == 64 for r in records)
    assert {r.dataset for r in records} == {"aihub_judgment", "statutes", "admin_interpretations", "unknown"}
    assert [r.file for r in records] == sorted(r.file for r in records)


def test_jsonl_and_list_files_yield_multiple_records(tmp_path: Path):
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "a.jsonl").write_text('{"x":1}\n\n{"x":2}\n', encoding="utf-8")
    (tmp_path / "ds" / "b.json").write_text('[{"x":3},{"x":4}, 5]', encoding="utf-8")
    records = list(discover_raw_records(tmp_path))
    assert [(r.file, r.index) for r in records] == [("ds/a.jsonl", 0), ("ds/a.jsonl", 1), ("ds/b.json", 0), ("ds/b.json", 1)]


def test_detect_schema_by_fingerprint():
    assert detect_schema({"info": {"caseNo": "x"}, "facts": {}, "dcss": {}, "close": {}}) == "aihub_judgment_v1"
    assert detect_schema({"law_name": "민법", "articles": []}) == "statute_json_v1"
    assert detect_schema({"authority": "a", "reference_number": "1", "question": "q", "interpretation": "i"}) == "admin_interpretation_v1"
    assert detect_schema({"note": "x"}) is None
    assert len(known_schemas()) == 3
