# 컨텍스트 실행 예산 시험 계획

**Goal:** 같은 근거·모델·검사 정책에서 10240 컨텍스트의 속도와 품질을 측정한다.

**Architecture:** 현재 checkout의 격리 worktree에서 고정 분석·검색을 재생한다. 요청 옵션 num_ctx만 바꾸고 기존 최종 답변 정책을 실행한다.

**Tech Stack:** Python, Ollama HTTP API, Pydantic, 기존 Orchestrator.

사용자는 다음 시험을 승인했다. `execute-protocol`로 직접 실행하며 재확인, 자동 커밋, 서비스 설정 변경은 추가하지 않는다.

## T1. 시험 고정

- [x] baseline.json으로 모든 기존 파일·상태·HEAD·index를 고정하고 worktree에 현재 내용을 복사한다.
- [x] study.py freeze를 실행해 6문항 + 별도 워밍업 입력과 첫 payload를 저장한다. 두 방식에서 num_ctx만 다른지 확인한다.
- [x] PREREG와 코드·설정·프롬프트·설계 문서 해시를 inference 전에 고정한다.

## T2. 실행과 검토

- [x] GPU lock 아래 study.py run을 실행한다. ABBA 24턴과 워밍업 4턴을 모두 보관한다.
- [x] study.py metrics로 회차별 최종 시간, p90, 생성 tokens/sec, 호출, 위험을 집계한다.
- [x] 12쌍을 read_review.py로 전문 확인하고 AI reviewed (Codex) 이유를 review.jsonl에 기록한다.
- [x] 실제 첫 payload, 컨텍스트, seed, 순서, 전체 입력·근거와 기존 정책 재생을 확인한다.

## T3. 기록과 감사

- [x] 결과를 docs/experiments/context-budget-2026-10-11.md에 한국어로 정리한다.
- [x] 백엔드 전체 회귀 확인과 최종 파일·기존 실행 상태·원자료·문서 링크 감사를 수행한다.
- [x] canonical state의 각 task gate와 최종 완료 상태를 검증하고 closure.json을 저장한다.

## 재실행 명령

worktree에서 `PYTHONPATH=src .venv/bin/python ../../data/eval/reports/chat_pipeline/context-budget-2026-10-11/study.py freeze|run|metrics`를 실행한다. run은 Python flock으로 기존 GPU lock을 획득하며 localhost 접근에 필요하면 정상 escalation을 사용한다. baseline과 manifest가 이미 있으면 freeze를 반복하지 않는다.
