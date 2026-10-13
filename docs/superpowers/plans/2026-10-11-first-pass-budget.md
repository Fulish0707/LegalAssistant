# 첫 생성 출력 한도 비교 실행 계획

> 현재 세션에서 execute-protocol로 순차 실행한다. 사용자는 재호출 감소 분석과 시험을 승인했다.

**목표:** 긴 응답의 원인을 분류하고 첫 출력 한도 확장이 재호출·시간·품질에 미치는 영향을 확인한다.

**구조:** 현재 root의 미커밋 파일까지 복제한 격리 worktree에서 기존 Orchestrator를 실행한다. 실험 기록기는 첫 답변 생성의 `max_tokens`만 변경한다. 서비스 코드는 수정하지 않는다.

**환경:** Python, 기존 Ollama qwen3:4b, Pydantic LegalAnswer, frozen 분석·검색.

## T1. 원인 분류와 고정

- [x] root 파일·HEAD·상태와 이전 실행 상태를 해시로 보존한다.
- [x] 현재 backend 기준 테스트를 기록한다.
- [x] 기존 full 80턴과 이전 recovery 비교 320턴의 실제 호출·잘림·인용 문제를 분류한다. 서로 다른 실행군은 합쳐 품질 지표로 쓰지 않는다.
- [x] 실험 runner `data/eval/reports/chat_pipeline/first-pass-budget-2026-10-11/study.py`를 작성한다. 기존 Generator가 쓰는 max_tokens 인수를 첫 실제 호출에만 3072로 바꾼다. 후속 호출은 2048임을 assert한다.
- [x] 4문항, 원본 근거와 두 첫 payload를 고정한다. num_predict를 제거하면 두 요청이 같아야 한다.
- [x] `env PYTHONPATH=src .venv/bin/python data/eval/reports/chat_pipeline/first-pass-budget-2026-10-11/study.py freeze`를 실행한다. 원문은 private 경로에 저장한다.

## T2. 측정과 전체 검토

- [x] `flock /tmp/claude-1000/myllmagent-gpu.lock env PYTHONPATH=src .venv/bin/python data/eval/reports/chat_pipeline/first-pass-budget-2026-10-11/study.py run`으로 16턴·4워밍업을 실행한다.
- [x] `study.py metrics`로 실패·거부·모든 실제 호출을 포함한 지표를 계산한다.
- [x] 8쌍의 전체 근거와 전체 답변을 대조하여 private `review.jsonl`에 AI reviewed 사유를 기록한다.
- [x] 속도와 품질 기준을 각각 판정한다. 작은 시험을 서비스 개선 완료로 표현하지 않는다.

## T3. 문서와 보존

- [x] 결과 문서 `docs/first-pass-budget-2026-10-11.md`에 원인·호출·시간·품질·판정·한계를 기록한다.
- [x] 전체 최초 payload, seed, 실행 순서, 후속 한도, 기존 검증 정책, manifest 해시를 감사한다.
- [x] root 422개 기준 파일과 이전 8개 상태 해시, HEAD, stage 없음과 문서 링크·코드 블록을 확인한다.
- [x] `audit_diff.py --target .worktrees/first-pass-budget`의 새 경로를 검토한다. 새 작업의 상태만 순서대로 완료한다.
- [x] root에 결과·설계·계획 문서만 복사한다. 서비스 코드와 설정은 변경하지 않는다.
