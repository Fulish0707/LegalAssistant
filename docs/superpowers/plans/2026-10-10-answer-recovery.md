# 긴 답변과 JSON 복구 개선 실행 계획

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. Execute inline. Do not spawn sub-agents. The user's request authorizes this plan document; implementation, tests and model runs start only after a subsequent execution request. Do not commit or restart user services automatically.

**Goal:** 조건과 예외를 보존하면서 답변의 잘림과 불필요한 재생성을 줄인다.

**Architecture:** 전체 답변 경로에 반복 억제와 구조화 실패 분류·조건 보존 재시도를 독립적인 실험 설정으로 추가했다. 실제 완전한 JSON 쉼표 오류가 없어 로컬 형식 수정은 구현하지 않았다. 인용 검사와 최종 표시 경로를 유지했다. 후보 코드는 격리 worktree에만 보관한다.

**Tech Stack:** Python 3.11, Pydantic, pytest, Ollama/httpx, 기존 Flask/FastAPI.

**Design:** [설계 문서](../specs/2026-10-10-answer-recovery-design.md).

**Status:** 2026-10-11 Task 1~9 실행·평가·보고와 최종 감사 완료. 진단 28턴, 제외 파일럿 56턴, 주 비교 320턴과 제외 워밍업 4턴, core/strict 640건 채점을 마쳤다. 의미 검토는 Codex 240쌍 완료, 로컬 모델 237쌍 완료·3건 형식 오류다. 세 후보 모두 채택하지 않는다. 조건부 형식 수정·역순 반복·웹 측정은 해당 조건 미충족으로 제외했다. 기본값 OFF / legacy를 유지한다. 상세 이유는 [결과 문서](../../experiments/answer-recovery-latency-2026-10-10.md)에 있다.

---

## 1. 범위와 판정 기준

- 우선순위 1: 잘린 최초 출력의 원인을 수집하고 반복 억제 지침을 비교한다.
- 우선순위 2: 실패 종류를 분리하고 형식 수정과 조건 보존 재시도를 비교한다.
- `full`, 필드 8개와 순서, 인용문, 2048 출력 토큰, context 12288, 온도 0.2를 유지한다.
- JSON 재시도 최대 1회와 인용 재생성 최대 1회를 유지한다.
- HTTP·모델 실패를 형식 오류로 처리하지 않는다.
- 후속 대화, 사용자 문서, plain-text LoRA 경로와 기존 인용 backfill을 보존한다.
- 기본 설정은 반복 억제 OFF, 복구 `legacy`다. 실험 결과 없이 기본값을 바꾸지 않는다.
- GPU 동시 요청 최적화, 모델·양자화 변경, 필드별 생성과 업로드 분석은 범위 밖이다.

| 판정 | 사전 기준 |
|---|---|
| 긴 대기 | 최종 후보의 45초 초과가 각 80문항 비교에서 2건 이하이며 같은 실행의 기준보다 50% 이상 감소 |
| 반복 확인 | 전체 80문항의 재비교에서도 위 조건 충족. 기준 0건이면 감소 입증 불가 |
| 일반 속도 | 중앙값 또는 p90의 5% 초과 악화가 두 비교에서 반복되면 채택하지 않음 |
| 의미 보존 | 새 필수 조건·예외 누락, 잘못된 단정 또는 인용 지원 악화가 확인되면 해당 후보 제외 |
| 결론 평가 | core와 strict 점수 각각 기준보다 3/80 초과 감소 시 제외. 점수가 유지돼도 의미 검토 실패를 덮지 않음 |
| 실패 정책 | 오류·거부·인용 재생성·제거 증가 시 원인 검토 전 채택 보류 |
| 표시 | 정책 처리 전 상태 문구만 전송. 최종 응답은 턴마다 1개 |

이 기준은 실험 채택 규칙이다. 통계적 유의성, 법률 정확성 인증 또는 서비스 SLA를 뜻하지 않는다. 성능 목표를 충족하지 못한 경우도 원자료와 이유를 남긴다.

## 2. 파일 책임

| 파일 | 예정 작업 |
|---|---|
| `src/legal_rag/evaluation/recovery_replay.py` | 실패 본문 수집과 독립 조합 재생. 서비스 경로에서는 사용하지 않음 |
| `src/legal_rag/llm/base.py` | 기존 오류에 실패 분류와 종료 사유 추가 |
| `src/legal_rag/llm/ollama.py` | Ollama 구조화 오류 분류 |
| `src/legal_rag/llm/structured_recovery.py` | 완전한 JSON의 마지막 쉼표 수정. 실제 사례가 있을 때만 구현 |
| `src/legal_rag/llm/prompts/answer_repetition_guard.md` | 조건을 남기는 반복 억제 지침 |
| `src/legal_rag/config/settings.py` | 기본 OFF / legacy인 실험 설정 |
| `src/legal_rag/rag/generator.py` | 지침 선택, 로컬 복구, 조건 보존 재시도 |
| `src/legal_rag/rag/orchestrator.py` | 새 설정 전달만 변경 |
| `tests/legal_rag/unit/test_answer_recovery.py` | 실패 분류, 수정 경계, 재시도, 설정 계약 |
| `tests/legal_rag/integration/test_answer_recovery.py` | 형식 수정 이후에도 인용 검사가 수행되는지 확인 |
| `docs/experiments/answer-recovery-latency-2026-10-10.md` | 구현 후 실제 결과와 채택 판단 |

`configs/local.yaml`, 기존 `answer_system.md`, `answer_concise.md`와 답변 스키마는 이번 후보 구현에서 수정하지 않는다. 기존 파일의 작업 중 변경을 먼저 확인하고, 새 수정만 분리한다.

## Task 1. 현재 기준과 측정 조건 보관

**Read:** 기존 [결과](../../experiments/verified-chat-latency-2026-10-10.md), `src/legal_rag/rag/generator.py`, `src/legal_rag/llm/ollama.py`, `src/legal_rag/schemas/answer.py`.

**Create:** `data/eval/reports/chat_pipeline/answer-recovery-2026-10-10/PREREG.md`, `baseline-manifest.json`, `AMENDMENTS.md`.

- [x] 현재 `git status --short`와 파일별 해시를 저장한다. HEAD `d03fd46`만 기준으로 삼지 않는다. 검색 재사용과 기존 미커밋 변경도 현재 기준의 일부다.
- [x] `llm_ocr.py`, `tests/test_document_citation_notice.py`, `docs/superpowers/plans/2026-10-09-four-levers.md`의 해시를 보관한다. 전체 diff와 관련 신규 파일은 `/tmp`에 별도로 보관한다. 사용자 파일을 되돌리거나 일괄 stage하지 않는다.
- [x] 격리 체크아웃을 사용할 때 현재 관련 코드의 미커밋 내용도 복사하고 해시를 대조한다. 작업 중인 `.worktrees/verified-chat-latency`를 초기화하거나 덮어쓰지 않는다.
- [x] `PREREG.md`에 위 판정 기준, 네 조합, 문항 순서, 워밍업 제외, 실패 포함 집계와 리뷰 범위를 기록한다. 첫 모델 호출 전에 완료한다.
- [x] frozen 입력과 참고 답변을 원본 그대로 사용한다.

```text
data/eval/reports/chat_pipeline/prompt-tuning-2026-10-06/frozen.jsonl
data/eval/reports/chat_pipeline/prompt-format-2026-10-07/bullets-1/answer_pairs.jsonl
```

- [ ] 모델·프롬프트·설정·입력 파일 해시, GPU 적재, Flash Attention, Ollama 버전을 기록한다. 질문·답변·키·`.env` 내용은 서비스 로그와 manifest에 넣지 않는다.
- [x] GPU 작업은 기존 잠금 아래에서 순차 실행한다. 기존 `gpu.sh`는 채팅 모델을 언로드하므로 준비된 사용자 서비스와 충돌할 수 있다. 이번 실행은 같은 lock 파일의 `flock`만 사용하고 모델을 강제로 언로드하지 않는다. 별도 사용자가 GPU를 사용하면 실행을 중지하고 조건을 기록한다.

```bash
flock /tmp/claude-1000/myllmagent-gpu.lock .venv/bin/python -m legal_rag.evaluation.recovery_replay --help
```

위 명령은 Task 2 파일을 만든 뒤 사용한다. 샌드박스에서 localhost 또는 GPU 접근이 거부되면 정해진 escalation 절차를 사용한다. 사용자 서비스를 재시작하지 않는다.

## Task 2. 실패 본문을 격리 재생에서 수집

**Create:** `src/legal_rag/evaluation/recovery_replay.py`.

아래 실행기는 처음에는 `baseline`만 실행한다. 새 설정이 추가된 뒤 네 조합을 지원한다. 각 모드의 Orchestrator와 저장소는 독립적이다. 원문은 실험 폴더에만 기록한다.

- [x] 다음 내용으로 실행기를 만든다. 기존 측정 스크립트를 수정하거나 기존 결과를 덮어쓰지 않는다.

```python
"""Isolated replay for complete-answer repetition and recovery experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

ORDER = ("baseline", "guard", "recovery", "combined")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=ORDER, default=list(ORDER))
    parser.add_argument("--indices", nargs="*", type=int)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must be unique")
    for key in list(os.environ):
        if key.startswith("LEGAL_RAG__"):
            del os.environ[key]

    from legal_rag.cli import settings_from_options
    from legal_rag.evaluation.frozen import FrozenAnalyzer, FrozenRetriever
    from legal_rag.factory import build_chat_model
    from legal_rag.rag.conversation import InMemoryConversationStore
    from legal_rag.rag.orchestrator import Orchestrator
    from legal_rag.verification import VerificationPolicy, Verifier

    recovery_events = []
    class RecoveryHandler(logging.Handler):
        def emit(self, record):
            if record.msg == "structured_recovery outcome=local_syntax_repair":
                recovery_events.append("local_syntax_repair")
    recovery_logger = logging.getLogger("legal_rag.rag.generator")
    original_level = recovery_logger.level
    recovery_handler = RecoveryHandler()

    turns = rows(args.frozen)
    references = {row["key"]: row for row in rows(args.references)}
    if len(turns) != 80 or len({turn["key"] for turn in turns}) != 80:
        parser.error("expected 80 unique frozen turns")
    indices = list(range(80)) if args.indices is None else args.indices
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= 80 for i in indices):
        parser.error("indices must be unique and between 0 and 79")
    if any(turn["key"] not in references for turn in turns):
        parser.error("reference key missing")
    if args.output.exists():
        parser.error("output directory must be new; partial runs are retained separately")
    args.output.mkdir(parents=True, mode=0o700)
    root = Path(__file__).resolve().parents[3]
    sources = sorted((root / "src/legal_rag").rglob("*.py"))
    sources += sorted((root / "src/legal_rag/llm/prompts").glob("*.md"))
    sources += [args.frozen, args.references, root / "configs/local.yaml", root / "configs/default.yaml"]
    manifest = {
        "sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "indices": indices, "modes": args.modes, "reverse": args.reverse,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "settings": {},
    }
    arms = {}
    for mode in args.modes:
        rag = {"answer_style": "full", "allow_clarification": False}
        if mode != "baseline":
            rag.update(answer_repetition_guard=mode in ("guard", "combined"),
                       structured_recovery="safe" if mode in ("recovery", "combined") else "legacy")
        cfg = settings_from_options(profile="local", extra={
            "rag": rag,
            "llm": {"backend": "ollama", "model": "qwen3:4b", "max_tokens": 2048,
                    "num_ctx": 12288, "timeout_seconds": 240.0, "temperature": 0.2},
            "api": {"warmup": False}, "trace": {"path": None},
        })
        if mode == "baseline" and hasattr(cfg.rag, "structured_recovery"):
            cfg = cfg.model_copy(update={"rag": cfg.rag.model_copy(update={
                "answer_repetition_guard": False, "structured_recovery": "legacy"})})
        manifest["settings"][mode] = cfg.model_dump(mode="json", exclude={
            "qdrant": {"api_key"}, "llm": {"api_key"}})
        model = build_chat_model(cfg)
        calls, checks = [], []
        original_post = model._client.post

        def post(*a, _post=original_post, _calls=calls, **kw):
            started = time.perf_counter()
            response = _post(*a, **kw)
            response.raise_for_status()
            body = response.json()
            _calls.append({"seconds": time.perf_counter() - started,
                           "text": (body.get("message") or {}).get("content", ""),
                           **{key: body.get(key) for key in (
                               "done_reason", "eval_count", "prompt_eval_count", "total_duration",
                               "load_duration", "prompt_eval_duration", "eval_duration")}})
            return response

        model._client.post = post
        orch = Orchestrator.from_components(settings=cfg, retriever=FrozenRetriever(), model=model,
            store=InMemoryConversationStore(), verifier=Verifier(),
            policy=VerificationPolicy(max_regenerations=cfg.rag.max_regenerations))
        orch.analyzer = FrozenAnalyzer()
        original_verify = orch.verifier.verify

        def verify(*a, _verify=original_verify, _checks=checks, **kw):
            report = _verify(*a, **kw)
            _checks.append(report.model_dump(mode="json"))
            return report

        orch.verifier.verify = verify
        arms[mode] = (orch, calls, checks)
    append(args.output / "manifest.jsonl", manifest)

    def run(index, mode, group):
        turn = turns[index]
        orch, calls, checks = arms[mode]
        orch.retriever.load(turn)
        orch.analyzer.load(turn)
        calls.clear()
        checks.clear()
        recovery_events.clear()
        started = time.perf_counter()
        row = {"key": turn["key"], "index": index, "mode": mode, "group": group}
        try:
            response = orch.handle(turn["question"], conversation_id=uuid.uuid4().hex)
            row.update(kind=response.kind, response=response.model_dump(mode="json"))
            if response.answer:
                from legal_rag.verification.citations import answer_text
                from legal_rag.verification.policy import STRIP_NOTE
                text = answer_text(response.answer).split(STRIP_NOTE)[0]
            else:
                text = ""
        except Exception as error:
            row.update(kind="error", error_type=type(error).__name__)
            text = ""
        row.update(seconds=time.perf_counter() - started, calls=list(calls), checks=list(checks),
                   local_syntax_repairs=len(recovery_events))
        append(args.output / group / mode / "verify.jsonl", row)
        if group == "main":
            pair = dict(references[turn["key"]])
            bundle = row.get("response", {}).get("evidence") or {}
            evidence = [" ".join(map(str, item["header"].values())) + " " + item["text"]
                        for item in bundle.get("items", [])]
            pair.update(answer=text, evidence=evidence, kind=row["kind"])
            append(args.output / group / mode / "answer_pairs.jsonl", pair)
        print(f"{group} index={index} mode={mode} seconds={row['seconds']:.2f} "
              f"kind={row['kind']} calls={len(calls)}", flush=True)

    recovery_logger.addHandler(recovery_handler)
    recovery_logger.setLevel(logging.INFO)
    try:
        for mode in args.modes:
            run(0, mode, "warmup")
        for index in indices:
            rotated = ORDER[index % 4:] + ORDER[:index % 4]
            order = [mode for mode in rotated if mode in args.modes]
            if args.reverse:
                order.reverse()
            for mode in order:
                run(index, mode, "main")
    finally:
        recovery_logger.removeHandler(recovery_handler)
        recovery_logger.setLevel(original_level)
        for orch, _, _ in arms.values():
            orch.generator.model._client.close()


if __name__ == "__main__":
    main()
```

- [x] 소스 확인으로 `--help`를 실행하고 필요한 인자가 표시되는지 확인한다. 추론 측정은 하지 않는다.
- [x] 기준만으로 다음 14문항을 재생한다. `diagnostic-01`, `diagnostic-02`, `diagnostic-03`의 서로 다른 출력 폴더로 최대 3회 실행한다. 해당 실패가 한 번도 재현되지 않아도 네 번째 실행을 임의로 추가하지 않는다.

```bash
flock /tmp/claude-1000/myllmagent-gpu.lock .venv/bin/python -m legal_rag.evaluation.recovery_replay --frozen data/eval/reports/chat_pipeline/prompt-tuning-2026-10-06/frozen.jsonl --references data/eval/reports/chat_pipeline/prompt-format-2026-10-07/bullets-1/answer_pairs.jsonl --output data/eval/reports/chat_pipeline/answer-recovery-2026-10-10/diagnostic-01 --modes baseline --indices 1 2 54 55 71 16 24 40 52 56 76 22 34 77
```

- [x] 각 호출의 최초 본문을 읽고 `failure-catalog.jsonl`에 문항 키, 호출 순번, 종료 사유, 8개 필드의 완성 여부, 반복 구절, 긴 인용 구절, 파싱 오류, 기존 검사 이슈를 기록한다. 원문은 private 재생 파일을 가리킨다. 과거 최초 본문을 새로 재현한 본문과 같다고 주장하지 않는다.
- [x] 마지막 쉼표 제거만으로 전체 필드와 값이 보존되는 실제 사례가 있으면 Task 5를 진행한다. 사례가 없으면 Task 5를 제외한 이유를 기록한다. 합성 사례 통과만으로 성능 기능을 추가하지 않는다.
- [x] 반복이 보이지 않으면 Task 3을 탐색 후보로 표시한다. 성공 원인으로 단정하지 않는다. 입력 자체가 긴 사례, 인용 문제와 내용 부족은 별도 원인으로 남긴다.

## Task 3. 반복 억제 후보를 독립 설정으로 추가

**Modify:** `settings.py`, `generator.py`, `orchestrator.py`.

**Create:** `answer_repetition_guard.md`, `tests/legal_rag/unit/test_answer_recovery.py`.

- [x] 먼저 새 단위 테스트 파일에 아래 코드를 추가한다.

```python
import json
import httpx
import pytest

from legal_rag.config import Settings
from legal_rag.llm import FakeChatModel, Message, ModelResponse, OllamaChatModel, StructuredOutputError
from legal_rag.llm.prompts import load_prompt
from legal_rag.rag.generator import GenerationError, Generator
from legal_rag.schemas import LegalAnswer
from tests.legal_rag.unit.test_verification import _answer, _bundle


def test_recovery_defaults_preserve_current_mode():
    settings = Settings()
    assert settings.rag.answer_style == "full"
    assert settings.rag.answer_repetition_guard is False
    assert settings.rag.structured_recovery == "legacy"


@pytest.mark.parametrize("guard", [False, True])
def test_guard_preserves_schema_budget_and_evidence(guard):
    model = FakeChatModel({"LegalAnswer": [_answer()]})
    Generator(model, answer_repetition_guard=guard, max_tokens=2048).generate("q", _bundle())
    call = model.calls[0]
    expected = load_prompt("answer_system")
    if guard:
        expected += "\n\n" + load_prompt("answer_repetition_guard")
    assert call.messages[0].content == expected
    assert call.response_schema is LegalAnswer and call.max_tokens == 2048
    assert "[LEGAL SOURCE S1]" in call.messages[1].content
    assert len(model.calls) == 1


def test_guard_cannot_be_combined_with_failed_concise_style():
    with pytest.raises(ValueError, match="full"):
        Generator(FakeChatModel({}), answer_style="concise", answer_repetition_guard=True)
```

- [x] 다음 명령을 실행하고 새 설정·인자가 없어 실패하는 것을 확인한다. 모델 서버를 사용하지 않는다.

```bash
.venv/bin/python -m pytest tests/legal_rag/unit/test_answer_recovery.py -q
```

- [x] `RagSettings`에 다음 두 필드를 추가한다.

```python
answer_repetition_guard: bool = False
structured_recovery: Literal["legacy", "safe"] = "legacy"
```

- [x] `Generator.__init__`에 keyword-only 인자 `answer_repetition_guard: bool = False`, `structured_recovery: str = "legacy"`를 추가한다. 기존 스타일 검사 뒤에 다음 코드를 추가한다.

```python
if answer_repetition_guard and answer_style != "full":
    raise ValueError("answer_repetition_guard requires full answer style")
if structured_recovery not in ("legacy", "safe"):
    raise ValueError("unknown structured recovery mode")
self.answer_repetition_guard = answer_repetition_guard
self.structured_recovery = structured_recovery
```

- [x] `_generate_structured`의 system 조립에 다음 분기를 추가한다. plain-text 함수는 변경하지 않는다.

```python
if self.answer_repetition_guard:
    system += "\n\n" + load_prompt("answer_repetition_guard")
```

- [x] `Orchestrator.from_components`의 기존 `Generator` 생성자 호출에 두 인자를 추가한다.

```python
answer_repetition_guard=settings.rag.answer_repetition_guard,
structured_recovery=settings.rag.structured_recovery,
```

- [x] 새 프롬프트 파일은 아래 내용을 후보 v1으로 기록한다. 진단 결과에 따라 변경하면 변경 이유와 새 해시를 먼저 남긴다.

```text
반복을 줄이는 작성 지침:
- 기존 근거 사용 규칙과 답변 규칙을 모두 지키세요. 전체 JSON 필드와 필드 순서를 유지하세요.
- 같은 명제나 같은 인용문을 여러 번 되풀이하지 마세요. 서로 다른 적용 범위를 같은 명제로 합치지 마세요.
- 결론이 달라지는 조건, 예외, 기간, 자격, 주체와 제출 기관은 각각 명시하세요. 이 내용을 줄여야 한다면 답변 길이보다 정확성을 우선하세요.
- analysis에는 필요한 근거와 적용 판단을 쓰세요. summary에는 그 판단의 결론과 결론을 바꾸는 중요한 전제를 남기세요. 두 필드에 필요한 제한 설명은 각각 남겨도 됩니다.
- citations의 quote는 제공 근거의 실제 구절을 그대로 쓰세요. 필요한 구절을 선택하되 조건이나 예외를 잘라 의미를 바꾸지 마세요.
- 법원·기관의 판단과 당사자의 주장을 구분하세요. 서로 다른 판단이나 청구를 하나로 합치지 마세요.
- uncertainty와 additional_information_needed에 필요한 불확실성과 사실 질문을 남기세요. 필드를 완성하기 위해 내용을 추측하지 마세요.
- 문장 수나 목록 수를 맞추기 위해 필요한 내용을 삭제하지 마세요. JSON을 끝까지 완성하세요.
```

- [x] 단위 테스트를 다시 실행한다. 새 프롬프트가 붙는 위치, 호출 수 1회, 스키마와 토큰 한도를 확인한다. 이후 Task 4를 진행한다.

## Task 4. 구조화 실패에 분류를 추가

**Modify:** `src/legal_rag/llm/base.py`, `ollama.py`.

**Test:** `tests/legal_rag/unit/test_answer_recovery.py`.

- [x] 다음 테스트를 같은 단위 테스트 파일에 추가하고 실패를 확인한다.

```python
@pytest.mark.parametrize("text,stop,kind", [
    ('{"analysis": "중간', "length", "truncated"),
    ('{"analysis": "중간', "stop", "truncated"),
    ('{"analysis": "완성",}', "stop", "syntax"),
    ('{"analysis": "완성"}', "stop", "schema"),
])
def test_ollama_failure_metadata(text, stop, kind):
    model = OllamaChatModel("m", "http://t", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"message": {"content": text}, "done_reason": stop})))
    with pytest.raises(StructuredOutputError) as error:
        model.generate([Message(role="user", content="q")], response_schema=LegalAnswer)
    assert error.value.failure_kind == kind
    assert error.value.stop_reason == stop
    assert error.value.text == text


def test_legacy_error_constructor_remains_valid():
    error = StructuredOutputError("bad", "raw")
    assert error.failure_kind == "unknown" and error.stop_reason is None
    assert error.text == "raw"


def test_http_failure_is_not_a_structured_output_error():
    model = OllamaChatModel("m", "http://t", transport=httpx.MockTransport(
        lambda request: httpx.Response(503, json={"error": "unavailable"})))
    with pytest.raises(httpx.HTTPStatusError):
        model.generate([Message(role="user", content="q")], response_schema=LegalAnswer)
```

- [x] `StructuredOutputError.__init__`을 다음으로 교체한다.

```python
def __init__(self, message: str, text: str, *, failure_kind: str = "unknown",
             stop_reason: str | None = None) -> None:
    super().__init__(message)
    self.text = text
    self.failure_kind = failure_kind
    self.stop_reason = stop_reason
```

- [x] `ollama.py`의 기존 Pydantic 파싱 except 블록을 다음으로 교체한다. 정상 JSON 파싱, fence 처리와 HTTP 처리는 그대로 둔다.

```python
except (ValidationError, json.JSONDecodeError, ValueError) as exc:
    stop_reason = body.get("done_reason")
    errors = exc.errors(include_input=False) if isinstance(exc, ValidationError) else []
    syntax_error = isinstance(exc, json.JSONDecodeError) or any(
        item["type"] == "json_invalid" for item in errors)
    eof_error = any(item["type"] == "json_invalid" and "EOF while parsing" in
                    item.get("ctx", {}).get("error", "") for item in errors)
    if stop_reason == "length" or eof_error:
        kind = "truncated"
    elif syntax_error:
        kind = "syntax"
    elif isinstance(exc, ValidationError):
        kind = "schema"
    else:
        kind = "unknown"
    raise StructuredOutputError(f"{response_schema.__name__}: {exc}", text,
                                failure_kind=kind, stop_reason=stop_reason) from exc
```

- [x] 다음 범위를 실행하고 기존 어댑터 계약도 확인한다.

```bash
.venv/bin/python -m pytest tests/legal_rag/unit/test_answer_recovery.py tests/legal_rag/unit/test_llm.py tests/legal_rag/unit/test_gemini.py -q
```

## Task 5. 실제 사례가 있을 때 완전한 JSON의 형식 수정

**Create:** `src/legal_rag/llm/structured_recovery.py`.

**Gate:** Task 2에 실제로 수정할 수 있는 사례가 있을 때만 구현한다. 사례가 없으면 이 작업을 제외하고 Task 6의 대체 분기를 사용한다.

- [ ] 같은 단위 테스트 파일에 다음 테스트를 추가한다.

```python
from legal_rag.llm.structured_recovery import repair_complete_answer


def test_complete_trailing_comma_repair_preserves_every_value():
    payload = _answer(uncertainty='인용 안의 ,} 와 ,] 는 문자입니다.').model_dump(mode="json")
    text = json.dumps(payload, ensure_ascii=False)
    broken = text[:-1] + ",}"
    recovered = repair_complete_answer(broken)
    assert recovered is not None and recovered.model_dump(mode="json") == payload


@pytest.mark.parametrize("text", [
    '{"analysis": "끝나지 않음',
    '{"analysis": "내용",}',
    '{"analysis":,}',
    '[,]',
    '{"analysis": "하나", "analysis": "둘",}',
    '{"analysis": NaN,}',
    '{"analysis": Infinity,}',
])
def test_incomplete_or_ambiguous_json_is_not_repaired(text):
    assert repair_complete_answer(text) is None


def test_missing_optional_field_is_not_filled_by_local_repair():
    payload = _answer().model_dump(mode="json")
    del payload["uncertainty"]
    text = json.dumps(payload, ensure_ascii=False)
    assert repair_complete_answer(text[:-1] + ",}") is None


def test_missing_nested_quote_is_not_filled_by_local_repair():
    payload = _answer().model_dump(mode="json")
    del payload["citations"][0]["quote"]
    text = json.dumps(payload, ensure_ascii=False)
    assert repair_complete_answer(text[:-1] + ",}") is None


def test_missing_array_value_is_not_deleted():
    text = _answer().model_dump_json().replace('"additional_information_needed":[]',
                                              '"additional_information_needed":[,]')
    assert repair_complete_answer(text) is None


def test_existing_valid_answer_needs_no_local_repair():
    assert repair_complete_answer(_answer().model_dump_json()) is None


@pytest.mark.parametrize("mode,stop,count", [
    ("legacy", "stop", 2), ("safe", "stop", 1), ("safe", "length", 2),
])
def test_local_repair_is_opt_in_and_never_accepts_length_stop(mode, stop, count):
    good = _answer().model_dump_json()
    broken = good[:-1] + ",}"
    seen = []
    def handler(request):
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"message": {
            "content": broken if len(seen) == 1 else good},
            "done_reason": stop if len(seen) == 1 else "stop"})
    model = OllamaChatModel("m", "http://t", transport=httpx.MockTransport(handler))
    answer = Generator(model, structured_recovery=mode, max_tokens=2048).generate("q", _bundle())
    assert answer == _answer() and len(seen) == count
    assert all(request["options"]["num_predict"] == 2048 for request in seen)
```

- [ ] `repair_complete_answer`가 없어 실패하는 것을 확인한다. 아래 함수 전체를 새 파일에 추가한다. 이 함수는 쉼표만 삭제하며 문자열을 바꾸지 않는다.

```python
"""One narrowly defined syntax correction; never finish a truncated answer."""
from __future__ import annotations

import json
from pydantic import ValidationError
from legal_rag.schemas import LegalAnswer


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON value")


def repair_complete_answer(text: str) -> LegalAnswer | None:
    if not text.strip().startswith("{"):
        return None
    output = []
    in_string = False
    escaped = False
    changed = False
    for index, char in enumerate(text):
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == ",":
            after = index + 1
            while after < len(text) and text[after].isspace():
                after += 1
            if after < len(text) and text[after] in "]}":
                before = index - 1
                while before >= 0 and text[before].isspace():
                    before -= 1
                if before < 0 or text[before] in "[{:,":
                    return None
                changed = True
                continue
        output.append(char)
    if in_string or not changed:
        return None
    candidate = "".join(output)
    try:
        payload = json.loads(candidate, object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant)
        if not isinstance(payload, dict) or set(payload) != set(LegalAnswer.model_fields):
            return None
        parsed = LegalAnswer.model_validate_json(candidate)
    except (ValueError, ValidationError):
        return None
    if parsed.model_dump(mode="json") != payload:
        return None
    return parsed
```

- [ ] 새 테스트를 다시 실행한다. quote·본문 안의 쉼표가 그대로인지 확인한다. 중복 키 검사는 마지막 쉼표 수정 경로에만 적용한다. 정상 JSON 파서의 기존 정책을 이 작업에서 변경하지 않는다.

## Task 6. 기존 한도 안에서 복구 경로 연결

**Modify:** `src/legal_rag/rag/generator.py`.

- [x] 먼저 다음 테스트를 단위 테스트 파일에 추가한다.

```python
@pytest.mark.parametrize("mode", ["legacy", "safe"])
def test_truncated_answer_still_requires_exactly_one_regeneration(mode):
    class Truncated:
        name = "truncated"
        def __init__(self):
            self.calls = []
        def generate(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                raise StructuredOutputError("EOF while parsing", '{"analysis":"중간',
                                            failure_kind="truncated", stop_reason="length")
            return ModelResponse(text="{}", parsed=_answer())
    model = Truncated()
    Generator(model, structured_recovery=mode, max_tokens=2048).generate("q", _bundle())
    assert len(model.calls) == 2
    assert all(kwargs["max_tokens"] == 2048 for _, kwargs in model.calls)
    retry = model.calls[1][0][-1].content
    if mode == "safe":
        assert "조건" in retry and "예외" in retry and "문장 수" in retry
        assert "다섯 문장 이내" not in retry and "세 항목 이내" not in retry
    else:
        assert "다섯 문장 이내" in retry
    assert all(message.content != '{"analysis":"중간' for message in model.calls[1][0])


def test_safe_recovery_keeps_retry_limit_on_schema_failure():
    class Dead:
        name = "dead"
        def __init__(self):
            self.count = 0
        def generate(self, messages, **kwargs):
            self.count += 1
            raise StructuredOutputError("schema error", "{}", failure_kind="schema")
    model = Dead()
    with pytest.raises(GenerationError):
        Generator(model, structured_recovery="safe").generate("q", _bundle())
    assert model.count == 2


def test_unknown_recovery_mode_is_rejected():
    with pytest.raises(ValueError, match="recovery"):
        Generator(FakeChatModel({}), structured_recovery="other")


def test_plain_text_adapter_prompt_is_preserved():
    model = FakeChatModel({"text": ["계약 종료를 전제로 합니다."]})
    answer = Generator(model, plain_text=True, answer_repetition_guard=True,
                       structured_recovery="safe").generate("q", _bundle())
    assert answer.analysis == "계약 종료를 전제로 합니다."
    assert len(model.calls) == 1 and model.calls[0].response_schema is None
    assert load_prompt("answer_repetition_guard") not in model.calls[0].messages[0].content
```

- [x] 실패를 확인한 뒤 `_repair_request`를 다음 함수로 교체한다. `safe=False`의 내용은 현재 지침과 같아야 한다.

```python
def _repair_request(error: StructuredOutputError, *, safe: bool = False) -> str:
    if safe:
        if error.failure_kind == "truncated" or error.stop_reason == "length" or "EOF while parsing" in str(error):
            return (
                "출력이 잘렸습니다. 제공된 근거와 질문으로 전체 JSON 객체를 다시 작성하세요. "
                "같은 명제와 같은 인용문의 반복만 줄이세요. 결론을 바꾸는 조건, 예외, 기간, "
                "자격, 주체, 제출 기관과 불확실성을 모두 보존하세요. 서로 다른 적용 범위와 "
                "청구를 합치지 마세요. 문장 수나 목록 수를 맞추기 위해 내용을 삭제하지 마세요. "
                "잘린 내용이나 빠진 사실을 추측하지 말고 모든 필드를 스키마에 맞게 완성하세요.")
        return (
            "출력이 JSON 스키마에 맞지 않았습니다. 제공된 근거와 질문으로 전체 JSON 객체를 "
            "다시 작성하세요. 모든 필드와 인용을 스키마에 맞게 작성하고 조건, 예외, 기간과 "
            "불확실성을 보존하세요. 빠진 내용이나 새로운 법률 근거를 추측하지 마세요.")
    if "EOF while parsing" in str(error):
        return ("출력이 너무 길어 JSON이 중간에 잘렸습니다. 같은 내용을 훨씬 짧게 쓰세요. "
                "analysis는 다섯 문장 이내로, 각 목록은 세 항목 이내로 쓰고 같은 문장을 되풀이하지 마세요. "
                "스키마에 맞는 JSON 객체만 출력하세요.")
    return f"출력이 JSON 스키마에 맞지 않았습니다: {error}. 스키마에 맞는 JSON 객체만 출력하세요."
```

- [ ] Task 5를 구현했다면 `generator.py`에 `logging`, `repair_complete_answer` import와 `logger = logging.getLogger(__name__)`를 추가한다. 기존 except 블록을 다음으로 교체한다.

```python
except StructuredOutputError as exc:
    last_error = exc
    safe = self.structured_recovery == "safe"
    if safe and exc.failure_kind == "syntax" and exc.stop_reason != "length":
        repaired = repair_complete_answer(exc.text)
        if repaired is not None:
            logger.info("structured_recovery outcome=local_syntax_repair")
            return repaired
    messages = messages + [Message(role="user", content=_repair_request(exc, safe=safe))]
    continue
```

- [x] Task 5를 제외했다면 새 helper import와 logger를 추가하지 않고 다음 블록을 사용한다.

```python
except StructuredOutputError as exc:
    last_error = exc
    messages = messages + [Message(role="user", content=_repair_request(
        exc, safe=self.structured_recovery == "safe"))]
    continue
```

- [x] 단위 테스트를 실행한다. `_generate_plain`과 모델 응답의 정상 경로가 바뀌지 않았는지 diff로 확인한다. 루프는 계속 `range(2)`여야 한다.

```bash
.venv/bin/python -m pytest tests/legal_rag/unit/test_answer_recovery.py tests/legal_rag/unit/test_generator.py tests/legal_rag/unit/test_verified_latency.py tests/legal_rag/unit/test_chat_model_context.py -q
```

## Task 7. 수정 후에도 검사가 실행되는지 확인

**Create:** `tests/legal_rag/integration/test_answer_recovery.py`.

**Gate:** Task 5를 제외했으면 아래 로컬 수정 전용 테스트 파일을 만들지 않는다. 기존 `test_chat_turn.py`의 재생성·실패·제거·거부 테스트는 두 경우 모두 실행한다.

- [ ] 로컬 수정 경로가 실제 Orchestrator를 통과하는 테스트를 추가한다. 인덱스나 실제 모델은 사용하지 않는다.

```python
import json
import httpx
import pytest

from legal_rag.config import Settings
from legal_rag.llm import OllamaChatModel
from legal_rag.rag.conversation import InMemoryConversationStore
from legal_rag.rag.orchestrator import Orchestrator
from legal_rag.schemas import QueryAnalysis
from legal_rag.verification import VerificationPolicy, Verifier
from tests.legal_rag.unit.test_verification import _answer, _bundle


@pytest.mark.parametrize("unsupported", [False, True])
def test_locally_repaired_answer_still_runs_citation_policy(unsupported):
    payload = _answer().model_dump(mode="json")
    if unsupported:
        payload["analysis"] = "민법 제839조의2에 따릅니다. 보증금은 반환되어야 합니다."
        payload["citations"][0]["reference"] = "민법 제839조의2"
        payload["citations"][0]["source_id"] = "S9"
    text = json.dumps(payload, ensure_ascii=False)
    broken = text[:-1] + ",}"
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"message": {"content": broken}, "done_reason": "stop"})
    model = OllamaChatModel("m", "http://t", transport=httpx.MockTransport(handler))
    settings = Settings().model_copy(update={"rag": Settings().rag.model_copy(update={
        "allow_clarification": False, "structured_recovery": "safe"})})
    orch = Orchestrator.from_components(settings=settings, retriever=object(), model=model,
        store=InMemoryConversationStore(), verifier=Verifier(), policy=VerificationPolicy(max_regenerations=1))
    orch.analyzer.analyze = lambda message, history: QueryAnalysis()
    orch.rewriter.rewrite = lambda question, analysis: type("Queries", (), {"search_queries": [question]})()
    orch.retriever = type("Retriever", (), {"retrieve": lambda self, query, k, filters=None: []})()
    orch.context_builder.build = lambda hits: _bundle()
    response = orch.handle("보증금을 못 받았어요", conversation_id="test")
    if unsupported:
        assert len(calls) == 2 and response.trace.regenerations == 1
        assert response.trace.policy_action == "strip"
        assert "제839조의2" not in response.answer.analysis
    else:
        assert len(calls) == 1 and response.verification.ok
        assert response.trace.policy_action == "accept"
```

- [x] unit과 integration 경계를 통과한 뒤 기존 표시 테스트를 실행한다. 서비스 프로세스를 시작하지 않는다. 다음 명령의 새 integration 파일은 Task 5를 진행했을 때만 포함한다.

```bash
.venv/bin/python -m pytest tests/legal_rag/unit/test_answer_recovery.py tests/legal_rag/integration/test_answer_recovery.py tests/legal_rag/integration/test_chat_turn.py tests/legal_rag/unit/test_verification.py tests/legal_rag/unit/test_llm.py tests/legal_rag/unit/test_generator.py tests/legal_rag/unit/test_gemini.py tests/legal_rag/unit/test_verified_latency.py -q
/home/layke/anaconda3/bin/python -m pytest tests/test_chat_streaming.py tests/test_case_documents_in_chat.py tests/test_document_citation_notice.py -q
```

기존 사용자 테스트 파일을 수정해서 통과시키지 않는다. 실제 실행한 테스트 수와 범위를 기록한다. 전체 모음을 실행하지 않았다면 전체 통과로 쓰지 않는다.

## Task 8. 독립 조합과 전체 반복 비교

**Artifacts:** `data/eval/reports/chat_pipeline/answer-recovery-2026-10-10/`의 새 폴더. 이전 실험은 읽기 전용이다.

| 모드 | 반복 억제 | 복구 |
|---|---|---|
| baseline | OFF | legacy |
| guard | ON | legacy |
| recovery | OFF | safe |
| combined | ON | safe |

- [x] 진단용 14문항으로 네 조합의 파일 수, 최종 응답, 실패 포함 기록과 설정을 먼저 확인한다. pilot 결과는 최종 속도 표에 합산하지 않는다. 프롬프트를 수정하면 후보 버전과 이유를 남긴다.
- [x] 주 비교 전에 소스와 프롬프트를 고정한다. 네 조합을 전체 80문항으로 실행한다. 각 조합의 워밍업 1개는 제외한다. 총 320개 측정 턴과 4개 워밍업이다.

```bash
flock /tmp/claude-1000/myllmagent-gpu.lock .venv/bin/python -m legal_rag.evaluation.recovery_replay --frozen data/eval/reports/chat_pipeline/prompt-tuning-2026-10-06/frozen.jsonl --references data/eval/reports/chat_pipeline/prompt-format-2026-10-07/bullets-1/answer_pairs.jsonl --output data/eval/reports/chat_pipeline/answer-recovery-2026-10-10/main-01 --modes baseline guard recovery combined
```

- [x] 각 모드의 키 80개, 동일한 최종 EvidenceBundle, 필드 8개와 같은 설정을 확인한다. 실패와 거부는 제외하지 않는다. 호출 수, length 종료, JSON 실패, 로컬 수정, 인용 재생성, strip, 거부와 오류를 각각 센다. JSON 재시도와 인용 재생성을 더해 중복된 문항 수를 만들지 않는다.
- [x] 중앙값, nearest-rank p90, 최대, 45초 초과 건수와 호출별 출력 토큰을 집계한다. 호출 1회와 여러 회를 나눠 보고한다. 새 기준 실행과 비교하며 과거 5/80만 분모로 사용하지 않는다.
- [x] AI가 각 후보 80쌍의 전체 본문과 제공 근거를 읽고 다음 항목을 `semantic-review.jsonl`에 기록한다. 프롬프트 지시가 지켜졌을 것이라는 추측만으로 통과시키지 않는다.

```json
{
  "key": "frozen turn key",
  "candidate": "guard",
  "reviewer": "AI reviewed (Codex)",
  "scope": "complete paired prose and supplied evidence",
  "conclusion_change": false,
  "lost_conditions": [],
  "lost_exceptions": [],
  "wrong_subject_or_agency": [],
  "unsupported_assertions": [],
  "citation_support_worse": false,
  "verdict": "pass",
  "reason": "전체 본문과 근거에서 확인한 구체적인 이유"
}
```

- [x] 다음 0부터 시작하는 순번은 심층 검토한다: `1, 2, 54, 55, 71, 16, 24, 40, 52, 56, 76, 22, 34, 77`. 이전 `manual_review.jsonl`의 27문항 모두와 이번 결론 평가가 바뀐 문항도 확인한다. 판단할 수 없으면 `uncertain`으로 기록하고 채택을 보류한다. 전문가의 독립 평가라고 표현하지 않는다.
- [x] 기존 core 평가와 core-v1 strict를 같은 캐시된 참고 결론으로 실행한다. 각 후보와 기준의 `answer_pairs.jsonl` 전문을 사용한다. `summary`만 읽거나 1500자로 자르지 않는다. 사용한 rubric 코드와 캐시의 해시를 기록한다.

평가 코드는 다음 기존 위치를 사용한다. `judge_core_answers`에는 모드별 결과 폴더와 기존 `conclusions-dev.json`의 사본을 전달한다. strict는 기존 `VARIANTS["core-v1"]["grade"]`를 사용한다. 기존 캐시의 해시는 평가 전후에 같아야 한다. 원본이나 참고 결론을 수정하지 않는다.

```text
src/legal_rag/evaluation/answer_judge_core.py
data/eval/reports/chat_pipeline/verified-latency-2026-10-10/grade_run.py
data/eval/reports/chat_pipeline/verified-latency-2026-10-10/conclusions-dev.json
data/eval/reports/chat_pipeline/verified-latency-2026-10-10/cache-dev/
data/eval/reports/chat_pipeline/judge-improvement-2026-10-06/variants.py
```

- [x] 새 실험 폴더에 `grade_replay.py`를 다음 내용으로 만든다. 원래 실험의 `common.HERE`를 그대로 가져오지 않는다. 다음 스크립트의 출력 대상은 명시한 재생 폴더뿐이다.

```python
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

root = Path.cwd()
sys.path[:0] = [str(root / "src"), str(root)]
from legal_rag.evaluation.answer_judge_core import judge_core_answers
from legal_rag.llm import OllamaChatModel

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--modes", nargs="+", required=True)
args = parser.parse_args()
previous = root / "data/eval/reports/chat_pipeline/verified-latency-2026-10-10"
conclusions = args.run / "conclusions-dev.json"
cache = args.run / "cache-dev"
if not conclusions.exists():
    shutil.copy2(previous / "conclusions-dev.json", conclusions)
if not cache.exists():
    shutil.copytree(previous / "cache-dev", cache)
cache_files = [conclusions, *sorted(cache.rglob("*.json"))]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in cache_files}
variant_path = root / "data/eval/reports/chat_pipeline/judge-improvement-2026-10-06/variants.py"
sys.path.insert(0, str(variant_path.parent))
spec = importlib.util.spec_from_file_location("recovery_judge_variants", variant_path)
variants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(variants)
variants._HERE = cache
chat = OllamaChatModel("qwen3:8b", "http://localhost:11434", timeout=240,
                       think=False, num_ctx=8192)
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
try:
    for mode in args.modes:
        arm = args.run / "main" / mode
        pairs = rows(arm / "answer_pairs.jsonl")
        assert len(pairs) == 80 and len({pair["key"] for pair in pairs}) == 80
        summary = judge_core_answers(arm, chat, conclusions,
                                      emit=lambda line, m=mode: print(m, line, flush=True))
        print(mode, summary, flush=True)
        evidence = {row["key"]: row.get("evidence", []) for row in pairs}
        out = arm / "strict.jsonl"
        done = {row["key"] for row in rows(out) if row.get("status") == "graded"} if out.exists() else set()
        for pair in pairs:
            if pair["key"] in done:
                continue
            try:
                grade = variants.VARIANTS["core-v1"]["grade"](chat, pair, evidence[pair["key"]])
                row = {"key": pair["key"], "status": "graded", **grade}
            except Exception as error:
                row = {"key": pair["key"], "status": "error", "error_type": type(error).__name__}
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(mode, "strict", row["status"], flush=True)
finally:
    chat._client.close()
after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in [conclusions, *sorted(cache.rglob("*.json"))]}
assert before == after, "reference cache changed; do not accept this comparison"
```

실행 예:

```bash
flock /tmp/claude-1000/myllmagent-gpu.lock .venv/bin/python data/eval/reports/chat_pipeline/answer-recovery-2026-10-10/grade_replay.py --run data/eval/reports/chat_pipeline/answer-recovery-2026-10-10/main-01 --modes baseline guard recovery combined
```

평가가 중단되면 중단된 행과 사유를 남기고 같은 입력으로 재개한다. 각 기준에 대해 모드별 80개가 모두 평가되기 전 점수를 확정하지 않는다. rubric과 참고 결론 캐시의 변경은 새로운 비교로 취급한다.

- [x] 의미 보존을 통과한 후보 중 긴 대기 감소가 가장 큰 하나를 선택한다. 주 비교가 모두 실패하면 중단하고 기준 동작을 유지한다. 실패 후 후보를 몰래 수정해서 같은 비교로 보고하지 않는다.
- [ ] 선택 후보와 baseline을 전체 80문항으로 재비교한다. `--reverse`로 각 문항의 이전 두 모드 상대 순서를 반대로 한다. 아래 명령은 `combined`가 선택된 경우다. `guard` 또는 `recovery`가 선택되면 마지막 모드 인자만 해당 실제 이름으로 바꾼다.

```bash
flock /tmp/claude-1000/myllmagent-gpu.lock .venv/bin/python -m legal_rag.evaluation.recovery_replay --frozen data/eval/reports/chat_pipeline/prompt-tuning-2026-10-06/frozen.jsonl --references data/eval/reports/chat_pipeline/prompt-format-2026-10-07/bullets-1/answer_pairs.jsonl --output data/eval/reports/chat_pipeline/answer-recovery-2026-10-10/repeat-01 --modes baseline combined --reverse
```

- [ ] 반복의 160개 턴과 워밍업 2개를 별도 집계한다. 반복의 후보 80개도 전문 의미 검토한다. 기준에서 긴 대기가 재현되지 않으면 감소 효과가 확인되지 않았다고 적는다. 추가 반복은 새 사전 계획 없이 실행하지 않는다.
- [x] 평가 모델 작업도 같은 GPU 잠금 안에서 실행한다. 생성 모델과 평가 모델을 동시에 실행하지 않는다. 모델 원문과 사용자 데이터는 외부 평가 서비스에 전송하지 않는다.

## Task 9. 웹 경로와 최종 판단

**Read/reuse:** 이전 `e2e_server.py`, `e2e_run.py`의 격리 서버·SSE 측정 방식을 새 실험 폴더의 사본에서 사용한다.

**Create:** `docs/experiments/answer-recovery-latency-2026-10-10.md`.

- [ ] 고정 근거 비교를 통과한 경우에만 scratch API와 Flask에서 baseline / 선택 후보를 비교한다. 현재 사용자 서비스와 다른 포트, 독립 DB·Chroma·업로드 경로와 InMemory 대화 저장소를 사용한다.
- [ ] 새 웹 baseline은 **현재 검색 재사용 ON + full + 반복 억제 OFF + legacy**다. 이전 `e2e_server.py`의 `old_retrieve_many()` 연결을 복사하지 않는다. 선택 후보도 같은 검색을 사용한다. 기존 `baseline/reuse/concise` 세 mount는 새 baseline/후보 두 mount로 바꾼다. API의 settings와 Orchestrator의 settings는 같아야 한다.
- [ ] 기존 질문 파일을 그대로 사용한다. 모드마다 일반 질문 12개, 후속 대화 4개 쌍의 8턴, 동일 합성 문서 질문 4개를 두 번 실행한다. 각 모드 총 48개 측정 턴이다. 두 번째 실행은 모드 순서를 반대로 한다. 문서 업로드 시간은 채팅 시간에 합산하지 않는다.
- [ ] 두 번째 실행은 모드별 DB·Chroma·업로드 경로와 대화 저장소를 새로 만든다. 후속 질문만 같은 대화 안에서 수행한다. 서로 다른 모드와 반복에 대화 상태를 공유하지 않는다. 최종 근거가 다른 웹 쌍은 표시하고 프롬프트만의 효과로 해석하지 않는다.
- [ ] SSE 측정은 POST부터 최종 응답 수신까지다. 첫 상태 시각은 따로 저장한다. API 처리, 모델 호출과 인용 정책을 응답과 연결한다. 최종 응답이 없는 실패도 턴 수에 포함한다.
- [ ] 코드 경로와 수신 이벤트에서 정책 처리 전 상태 문구만 보냈는지 확인한다. 최종 응답과 `stream_end` 개수를 대조한다. 실제 브라우저 렌더링을 측정했다고 쓰지 않는다.
- [x] 테스트 수, 후보 설정, 소스 해시, 실패 카탈로그, 주 비교와 반복, 전체 의미 검토, 참고 결론 점수, 웹 조건별 시간을 새 결과 문서에 기록한다. 고정 근거와 웹 시간을 한 표본으로 합치지 않는다.
- [x] 기준 대비 각 설정의 효과를 구분한다. 반복 억제가 실패하고 복구만 통과하면 복구만 채택 후보로 남긴다. 로컬 형식 수정이 추가 호출을 줄이지 못했으면 그 기능의 속도 효과를 주장하지 않는다.
- [x] 채택 기준을 통과한 코드만 공유 체크아웃에 적용할 수 있는 diff로 정리한다. 현재 서비스 기본값과 사용자 프로필은 실제 적용 요청까지 유지한다. 검토 결과와 rollback 설정 OFF / legacy를 함께 제시한다.
- [x] 생성한 scratch 프로세스만 종료한다. 사용자 서비스와 Ollama 데몬을 종료하지 않는다. 보호 파일의 해시와 `git diff --check`를 확인한다. 기존 변경을 stage하거나 commit하지 않는다.
- [x] 모든 기준을 충족하지 못하면 구현 실험과 실패 결과를 보고하고 기본 경로를 유지한다. 완료를 위해 목표를 낮추거나 실패 사례를 제외하지 않는다.

## 계획 자체 확인

- [x] 실행 전에 후속 구현 요청이 있는지 확인한다.
- [x] 로컬 형식 수정의 실제 사례 조건과 대체 분기가 서로 일치하는지 확인한다.
- [x] 기존 미커밋 변경을 포함한 baseline과 새 후보의 manifest를 대조한다.
- [x] 실행한 항목만 체크한다. 계획에 있는 코드와 목표를 실행 결과로 표현하지 않는다.

## 실행 결과 정리 (2026-10-11)

| Task | 실행 결과 |
|---|---|
| 1 | 현재 미커밋 기준·보호 해시·사전 기준 보관. Flash Attention 현재 상태는 미확인으로 기록 |
| 2 | 진단 2회, 28턴. 실제 2048토큰 반복 잘림 확인. 최대 3회 안에서 중단 |
| 3 | 반복 억제 플래그·프롬프트·설정 전달과 red/green 테스트 완료 |
| 4 | native 실패 분류·기존 어댑터 계약 테스트 완료 |
| 5 | 완전한 JSON 쉼표 오류 미관측. helper와 전용 테스트 조건부 제외 |
| 6 | safe 전체 재생성·native 우선 분류 구현. 기존 EOF 문자열 분기는 unknown에서만 사용 |
| 7 | 표시·plain·인용 정책 계약 확인. 전용 로컬 수정 integration은 Task 5 제외로 N/A |
| 8 | 320턴+제외 워밍업 4턴, 채점 640건, 직접 의미 검토 240쌍. 모든 후보 채택 실패 |
| 9 | 결과 문서·변경 보존 감사. 통과 후보가 없어 웹 측정과 공유 구현 적용은 N/A |

위 본문은 실행 전에 정한 계획과 예시 코드도 보관한다. 조건부 제외 항목의 빈 체크박스는 미완료 구현 요구가 아니다. Task 1의 환경 항목은 Flash Attention을 새로 확인하지 못했으므로 전체 완료로 체크하지 않았다. 실제 구현의 native 분류 우선 수정은 주 비교 전에 추가 테스트로 고정했고 파일럿에는 소급 적용하지 않았다.

로컬 의미 평가 3건은 같은 설정으로 두 번 시도해도 StructuredOutputError였다. 오류 기록 6행을 유지한다. 자동 pass가 회귀를 놓친 뒤 Codex 검토를 전체 240쌍으로 확대했으며 그 3건도 직접 읽었다. 로컬 모델 240건 완료로 기록하지 않는다. 사전 채택 기준은 변경하지 않았다.

테스트는 고정 후보 소스의 backend 632 passed / 3 skipped와 표시 관련 웹 3파일 36 passed다. 웹 전체 suite를 실행했다는 뜻은 아니다. 역순 반복 160턴과 웹 48턴/설정은 선택 후보가 없으므로 실행하지 않았다. 단일 주 비교의 중앙값 변화만으로 두 비교의 5% 규칙을 판정하지 않았다.

최종 감사 PASS: 보호 파일 5개, 기존 root 실행 상태, 생성 소스 145개, 평가 스크립트 2개와 의존 파일 5개의 해시 보존. root/worktree diff 검사·HEAD·stage 상태·보고서 링크 확인. 후보 구현은 공유 코드에 적용하지 않았다.
