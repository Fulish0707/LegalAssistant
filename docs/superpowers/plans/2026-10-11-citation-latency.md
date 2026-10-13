# 인용 처리 지연 개선 실행 계획

> Execute inline with execute-protocol. User approval covers implementation and verification of P1 and a conditional P2 pilot. Preserve other work, do not commit or restart services. No subagents.

**Goal:** 사건번호 오탐을 제거하고 통과 시 인용 원문 복사 pilot을 평가한다.

**Architecture:** 공통 사건번호 extractor와 세 비교 경로를 정규화한다. P2는 격리 평가용 wrapper가 번호 선택 내부 스키마를 기존 LegalAnswer로 변환한다.

**Tech Stack:** Python, Pydantic, 기존 Ollama·Orchestrator·Verifier, pytest, private JSONL.

**Status:** T1–T4 완료. P1 반영, P2 기준 미통과로 미적용. [설계](../specs/2026-10-11-citation-latency-design.md).

## T1. P1 구현과 통과 기준

Files: `src/legal_rag/textutil/refs.py`, `src/legal_rag/verification/verifier.py`, `src/legal_rag/verification/backfill.py`, `tests/legal_rag/unit/test_case_spacing.py`.

- [x] 변경 전 root 및 기존 실행 상태 해시를 보관한 격리 worktree에서 시작한다.
- [x] 가로 공백의 정상화, 줄바꿈 비결합, 다른 사건·출처 차단, 헤더·대화·backfill 일관성을 테스트한다. 수정 전 실패를 확인한다.

```python
assert extract_case_refs("99다 40418, 99 다40418") == ["99다40418"]
assert extract_case_refs("99다\n40418") == []
```

- [x] 공백 허용 전용 정규식으로 사건번호를 추출하고 공백 없이 반환한다. 공개 CASE_PATTERN/CASE_RE는 보존한다. 헤더와 이전 대화 비교에도 extractor를 사용한다.
- [x] `env PYTHONPATH=src .venv/bin/python -m pytest tests/legal_rag/unit/test_case_spacing.py -q`와 기존 backend·웹 CPU suite를 실행한다.
- [x] 기존 80문항 및 후속 320턴·고정 실패 32턴의 파싱 답변을 전후 Verifier에 재생한다. 해제·추가 발견 이슈를 모두 기록하고 이유를 확인한다.
- [x] 알려진 오탐의 첫 복구 초안에서 불필요한 인용 재생성이 제거되는지 확인한다. 원문 본문은 변경하지 않는다.

## T2. P2 pilot 구현과 고정

Files: `src/legal_rag/evaluation/source_quotes.py`, `tests/legal_rag/unit/test_source_quotes.py`, `data/eval/reports/chat_pipeline/citation-latency-2026-10-11/*`.

- [x] T1 통과 상태를 확인한 뒤 시작한다.
- [x] 원문 복원·구간 선택·null 인용·잘못된 범위·미지 source 테스트를 먼저 작성한다.
- [x] 내부 Citation에 `quote_start: int | None`, `quote_end: int | None`를 사용한다. 기본 LegalAnswer schema를 바꾸지 않는다.
- [x] 원문 단위를 번호로 렌더링하고 검증된 연속 범위만 원문 substring으로 복사한다. wrapper는 기존 ModelResponse.parsed에 완성 LegalAnswer를 반환한다.
- [x] 지정 8문항, 참고 답변, 분석·근거, 모델 설정, 두 회차 순서·seed와 판정 기준을 고정한다. 실행기·후보·입력·payload를 해시로 보관한다.

## T3. 측정과 전체 검토

- [x] GPU lock 아래 baseline/source 방식의 32턴과 별도 4 warmup을 실행한다. 서비스 재시작·강제 unload는 하지 않는다.
- [x] 각 actual payload·raw·최종 8필드·검사·호출 시간과 token 수를 private 파일에 보관한다.
- [x] 두 회차의 최초 호출·전체 처리 median/p90, 호출·거부·오류·length 수를 집계한다. 모든 실패를 포함한다.
- [x] 16쌍을 전체 근거와 대조하여 AI reviewed (Codex), pass/fail/uncertain 및 구체 사유를 남긴다. 기준대로 후보 판정을 기록한다.

## T4. 제한된 통합과 보고

- [x] P1 통과 파일만 root의 기준 해시를 확인한 뒤 복사한다. 기존 변경은 덮어쓰지 않는다. P2 후보는 worktree에만 보관한다.
- [x] 한국어 보고서 `docs/experiments/citation-latency-2026-10-11.md`에 검증·pilot 결과와 한계를 기록한다.
- [x] root 기존 파일·prior state·HEAD·stage·고정 실험 hash·문서 링크와 path audit를 확인한다. 각 완료 상태는 관찰된 evidence로 전이한다.
