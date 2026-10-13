# 입력 처리 배치 시험 계획

**목표:** 입력 배치 512와 1024의 지연·품질을 동일 모델 조건에서 비교한다.

**실행:** 사용자 요청인 다음 테스트 범위에서 직접 실행한다. execute-protocol 상태를 사용하며 별도 worktree와 사전 기준을 고정한다. 자동 commit은 하지 않는다.

## T1 — 입력과 실행 고정

- [x] root 파일·이전 상태·외부 읽기 전용 파일 baseline을 기록한다.
- [x] study.py와 runtime.py를 분리한다. options.num_batch만 다른 7개 paired 입력, 동일 근거, ABBA 순서와 seed를 고정한다.
- [x] `PYTHONPATH=src .venv/bin/python data/eval/reports/chat_pipeline/prefill-batch-2026-10-11/study.py freeze`를 실행한다. 모델 호출 없이 해시·요청 차이·구문을 확인한다.

## T2 — 측정과 전체 검토

- [x] 동일 경로 `study.py run`을 실행한다. 새 서버 4개에서 예열 4회와 측정 24회를 기록한다. 로컬 소켓 제한은 실행 승인 절차를 사용한다.
- [x] `study.py metrics`로 중앙값·p90·입력 평가 시간·디코딩 속도·출력·호출·정책 결과를 계산한다.
- [x] 질문·참고 답변·근거 전체와 12쌍의 8필드·완료 초안을 읽고 `AI reviewed (Codex)` 이유 및 입력/답변 해시를 기록한다.
- [x] `verify_measurements.py`로 모든 요청과 기존 최종 정책을 재생한다. 이전 pytest 증거는 동일한 248개 파일 해시가 유지될 때만 재사용한다.

## T3 — 결과와 보존 감사

- [x] `docs/prefill-batch-2026-10-11.md`에 처리 범위·결과·실패·기존 오류·한계를 기록한다.
- [x] `audit_study.py`로 baseline, 원본 모델/키, 이전 상태, 지표 재계산, 검토 해시, 요청 재생, 실제 BatchSize·GPU 실행, 소유 서버 종료, 문서 링크를 확인한다.
- [x] T1–T3를 유효한 순서로 완료한다. 채택 여부와 시험 완료 상태를 구별한다.

[설계](../specs/2026-10-11-prefill-batch-design.md)
