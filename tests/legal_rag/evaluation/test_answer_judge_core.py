"""The core-conclusion grader: one yardstick per question, sentence numbers instead of copied quotes."""
import json

from legal_rag.evaluation.answer_judge_core import CoreCheck, CoreConclusion, judge_core_answers
from legal_rag.llm import FakeChatModel

ANSWER = "해고일부터 3개월 이내에 노동위원회에 구제신청을 할 수 있습니다.\n서면 통지가 없으면 해고는 효력이 없습니다."
CORE = CoreConclusion(core="부당해고 구제신청은 3개월 이내에 할 수 있다.", condition="")


def _run(tmp_path, name, answer=ANSWER, key="q1", question="두 달 전 해고, 구제신청 가능한가요?"):
    run = tmp_path / name
    run.mkdir()
    pair = {"key": key, "kind": "answer", "question": question, "reference": "3개월 이내에 구제신청을 할 수 있습니다.", "answer": answer}
    (run / "answer_pairs.jsonl").write_text(json.dumps(pair, ensure_ascii=False) + "\n", encoding="utf-8")
    return run


def _records(run):
    return [json.loads(line) for line in (run / "answer_judgments-core.jsonl").read_text(encoding="utf-8").splitlines()]


def _check(stated=True, sentences=(1,), contradicts=False):
    return CoreCheck(core_stated=stated, core_sentences=list(sentences), condition_stated=False, condition_sentences=[], contradicts=contradicts)


def test_an_answer_that_states_the_core_passes(tmp_path):
    model = FakeChatModel({"CoreConclusion": [CORE], "CoreCheck": [_check()]})
    run = _run(tmp_path, "a")
    judge_core_answers(run, model, tmp_path / "conclusions.json")
    record = _records(run)[0]
    assert record["status"] == "graded" and record["metrics"] == {"conclusion_correct": True}
    assert record["grade"]["core"] == CORE.core and record["grade"]["core_sentences"] == [1]
    assert [call.response_schema.__name__ for call in model.calls] == ["CoreConclusion", "CoreCheck"]
    # The answer reaches the grader as numbered sentences, so nothing has to be copied back.
    sent = json.loads(model.calls[1].messages[-1].content)
    assert sent["answer_sentences"][0] == {"n": 1, "text": "해고일부터 3개월 이내에 노동위원회에 구제신청을 할 수 있습니다."}


def test_a_claim_without_a_real_sentence_does_not_pass(tmp_path):
    model = FakeChatModel({"CoreConclusion": [CORE], "CoreCheck": [_check(sentences=(9,))]})
    run = _run(tmp_path, "a")
    judge_core_answers(run, model, tmp_path / "conclusions.json")
    assert _records(run)[0]["metrics"] == {"conclusion_correct": False}


def test_a_contradicting_answer_does_not_pass(tmp_path):
    model = FakeChatModel({"CoreConclusion": [CORE], "CoreCheck": [_check(contradicts=True)]})
    run = _run(tmp_path, "a")
    judge_core_answers(run, model, tmp_path / "conclusions.json")
    assert _records(run)[0]["metrics"] == {"conclusion_correct": False}


def test_every_run_is_measured_against_the_same_conclusion(tmp_path):
    cache = tmp_path / "conclusions.json"
    judge_core_answers(_run(tmp_path, "a"), FakeChatModel({"CoreConclusion": [CORE], "CoreCheck": [_check()]}), cache)
    second = FakeChatModel({"CoreCheck": [_check(stated=False, sentences=())]})  # no CoreConclusion scripted
    run_b = _run(tmp_path, "b", answer="경우에 따라 다릅니다.")
    judge_core_answers(run_b, second, cache)
    assert _records(run_b)[0]["grade"]["core"] == CORE.core
    assert [call.response_schema.__name__ for call in second.calls] == ["CoreCheck"]


def test_graded_answers_are_not_graded_again(tmp_path):
    cache, run = tmp_path / "conclusions.json", _run(tmp_path, "a")
    judge_core_answers(run, FakeChatModel({"CoreConclusion": [CORE], "CoreCheck": [_check()]}), cache)
    again = FakeChatModel({})
    judge_core_answers(run, again, cache)
    assert again.calls == [] and len(_records(run)) == 1


def test_an_empty_answer_is_recorded_without_a_model_call(tmp_path):
    model = FakeChatModel({})
    run = _run(tmp_path, "a", answer="")
    judge_core_answers(run, model, tmp_path / "conclusions.json")
    record = _records(run)[0]
    assert record["status"] == "no_answer" and record["metrics"] == {"conclusion_correct": False} and model.calls == []
