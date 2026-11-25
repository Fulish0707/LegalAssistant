import hashlib
import json

import pytest

from scripts.eval_answer_generation import answer_body, build_bundle, build_plain_messages, load_sources


def write_inputs(tmp_path, *, fictional=True, corrupt_hash=False):
    holding = "임대인은 수선 의무가 있다."
    case_text = ("[가상 판결문: 실제 판례가 아님]\n3. 판단\n" + holding) if fictional else holding
    statute = "제623조(임대인의 의무) 임대인은 목적물을 유지할 의무가 있다."
    sources = [
        {"id": "law-x", "kind": "statute", "law_name": "민법", "article": "제623조",
         "effective_date": "2026-03-17", "quote": statute, "extraction_file": "law-x.txt",
         "sha256": hashlib.sha256((statute + "\n").encode()).hexdigest()},
        {"id": "case-x", "kind": "case", "title": "수선비", "court": "가상지방법원",
         "case_number": "2026가단99001", "quote": holding, "extraction_file": "case-x.txt",
         "sha256": hashlib.sha256(case_text.encode()).hexdigest()},
    ]
    if corrupt_hash:
        sources[0]["sha256"] = "0" * 64
    (tmp_path / "law-x.txt").write_text(statute + "\n")
    (tmp_path / "case-x.txt").write_text(case_text)
    (tmp_path / "sources.json").write_text(json.dumps(sources, ensure_ascii=False))
    case = {"id": "T1", "domain": "lease", "message": "수선비를 돌려받을 수 있나요?",
            "sources": ["law-x", "case-x"], "case_source": "case-x"}
    (tmp_path / "cases.json").write_text(json.dumps([case], ensure_ascii=False))
    return case


def test_load_sources_accepts_marked_fictional_case(tmp_path):
    case = write_inputs(tmp_path)
    by_id, hashes = load_sources(tmp_path)
    assert set(by_id) == {"law-x", "case-x"}
    assert len(hashes) == 2
    bundle = build_bundle(case, by_id)
    statute, judgment = bundle.items
    assert statute.header["Law"] == "민법" and statute.header["Article"] == "제623조"
    assert judgment.header["Section role"] == "기관·법원 판단"
    assert judgment.text == "임대인은 수선 의무가 있다."


def test_load_sources_rejects_unmarked_case(tmp_path):
    write_inputs(tmp_path, fictional=False)
    with pytest.raises(ValueError, match="fictional"):
        load_sources(tmp_path)


def test_load_sources_rejects_hash_mismatch(tmp_path):
    write_inputs(tmp_path, corrupt_hash=True)
    with pytest.raises(ValueError, match="hash"):
        load_sources(tmp_path)


def test_plain_messages_use_training_format(tmp_path):
    from legal_rag.training.dataset import SYSTEM

    case = write_inputs(tmp_path)
    by_id, _ = load_sources(tmp_path)
    messages = build_plain_messages(case, by_id, case["message"])
    assert messages[0] == {"role": "system", "content": SYSTEM}
    user = messages[1]["content"]
    assert user.startswith("[법률 자료]\n제목: 민법 제623조\n")
    assert "[/법률 자료]" in user and user.endswith("질문: " + case["message"])


def test_answer_body_deduplicates_plain_wrapper():
    from legal_rag.schemas import LegalAnswer
    from legal_rag.schemas.answer import Confidence

    answer = LegalAnswer(summary="같은 본문", legal_issues=[], analysis="같은 본문",
                         possible_actions=["행동 하나"], citations=[], confidence=Confidence.medium)
    assert answer_body(answer) == "같은 본문\n\n행동 하나"
