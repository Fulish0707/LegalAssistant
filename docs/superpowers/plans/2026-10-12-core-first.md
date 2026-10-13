# Core First Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. No delegation or automatic commits are authorized.

**Goal:** 검증된 핵심 답변을 먼저 표시하는 후보를 격리 구현·측정한다.

**Architecture:** Orchestrator의 준비·최종 정책 처리를 분리해 기존 handle 경로와 단계 경로가 같은 검색·검사를 사용한다. 각 단계는 완성된 JSON을 받은 뒤 검사하며 core를 최종에 그대로 보관한다.

**Tech Stack:** Python,Pydantic,Flask,FastAPI,NDJSON/SSE,기존 Ollama,pytest,Node.

## T1 — 설계·baseline

- [x] baseline.json으로 root440파일·이전14상태·외부9파일을 기록하고 현재 파일을 .worktrees/core-first에 복사한다.
- [x] 기존 회귀 바인딩248파일과 이전 증거를 비교한다. 운영 root canonical mission은 별도 작업이므로 변경하지 않는다.
- [x] 설계의24턴·예열4턴·속도/입력/품질/표시 경계 기준을 상태에 고정한다.

## T2 — 구현과 계약 검사

**Files:** src/legal_rag/rag/orchestrator.py,generator.py,staged.py; src/legal_rag/schemas/staged.py; src/legal_rag/llm/prompts/answer_core.md,answer_details.md; src/legal_rag/api/routes.py; legal_backend.py,app.py,static/lc-chat.js; tests/legal_rag/unit/test_core_first.py,tests/test_core_first_transport.py,tests/core_first_ui.cjs.

- [x] 먼저 테스트에서 구현 유무를 단언해 RED를 기록한다. 모델은 FakeChatModel로 제공하며 실제 검증기를 사용한다.
- [x] _prepare_turn은 기존 분석·검색과 PreparedTurn을 반환하고 _finish_turn은 기존 strip/reject/저장 정책을 적용한다. handle은 두 메서드를 연결한다. _generate_verified는 선택 generator를 받되 기본값은 기존 generator다.
- [x] CoreAnswer는 reasoning·conclusion·conditions·exceptions·missing_facts·citations·uncertainty·confidence를 모두 필수로 둔다. core 변환은 다음 의미를 유지한다.

```python
def core_analysis(core):
    parts = [core.reasoning]
    if core.conditions:
        parts.append("적용 조건\n" + "\n".join("- " + x for x in core.conditions))
    if core.exceptions:
        parts.append("예외\n" + "\n".join("- " + x for x in core.exceptions))
    return "\n\n".join(parts)
```

- [x] CoreFirstPipeline.events는 준비1회→core 생성·검사→core_response yield→detail 생성·검사→final_response yield를 실행한다. 실패·conflict·미통과 report에서는 core를 최종으로 유지하고 detail_status unavailable을 준다. finally에서 마지막 표시 가능 답변을 한 번 저장한다.
- [x] route /v1/chat/staged는 전체 완성 이벤트만 NDJSON으로 전달한다. legal_backend.ask_events는 HTTP 스트림에서 이벤트 계약과 report.ok를 확인한다. Flask는 시험 flag에서 각 단계를 기존 표시 검사 후 SSE로 전달한다.
- [x] 화면은 core에서 agent객체1개를 추가하고 최종에서 같은 객체를 갱신한다. core에서는 EventSource를 닫지 않는다.
- [x] PYTHONPATH=src .venv/bin/python -m pytest tests/legal_rag/unit/test_core_first.py tests/test_core_first_transport.py -q 및 node tests/core_first_ui.cjs를 실행한다. 실패core·재생성·상세실패·충돌·잘린JSON·disconnect·서로 다른 사용자·자료 전달을 검사한다.
- [x] PYTHONPATH=src .venv/bin/python -m pytest tests -q로 새 전체 회귀를 기록한다. 독립 HTTP 지연 주입 시험에서 상세 시작 전에 core를 받을 수 있는지 확인한다.

## T3 — 고정 측정·전체 검토

- [x] study.py freeze로7개 입력과각 arm 첫 payload·source·prompt·schema·실행기 해시를 고정한다. 변경 전 baseline payload는 직전 original payload와 같아야 한다.
- [x] study.py run으로 새 서버4개의ABBA24턴+예열4턴을 측정한다. first_verified_seconds,total_seconds,각 실제 호출·완료 초안·검사·표시 이벤트·실패를 모두 기록한다.
- [x] 첫 답변·전체 완료 지표를 따로 계산하고12쌍의 core와최종을 새로 읽어 AI reviewed (Codex) 사유와 해시를 저장한다.
- [x] verify_measurements.py로 모든 실제 payload/파서/단계/정책/표시 답변을 새로 재생한다. parse_cache.py는 모든 실제 호출을 로그와 연결한다.

## T4 — 결과·보존

- [x] docs/core-first-2026-10-12.md에 실제 수치·core/상세 오류·실패 응답·범위·채택 여부를 기록한다.
- [x] audit_study.py로 root전체·이전상태·외부파일·baseline payload·새 회귀·manifest·재생·리뷰·지표·서버 종료·root 추가3문서를 확인한다.
- [x] canonical T1–T4를 증거와 함께 완료한다. 후보는 worktree에 보관하고 운영에는 적용하지 않는다. 부분 복구는 별도 후보로 남긴다.

[설계](../specs/2026-10-12-core-first-design.md)

## 실행 환경 기록

Backend는 기존 .venv, 웹 검사는 requirements.txt를 설치한 독립 /tmp/core-first-web 환경을 사용했다. 기존 live 임베딩 검사는 bge-m3의1024차원 출력과768차원 기대값이 달라 실패했다. 그 기록은 보존했다. 최종 웹 회귀는142개와12개 subtest가 통과했다. 이 테스트가 로드한 운영 모델의 자동 해제가 끝난 뒤 성능 측정을 시작했다. 후속 웹 검사는 운영 호출을 차단했다. 모델은 수동 해제·재시작하지 않는다.
