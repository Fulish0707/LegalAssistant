# 같은 최초 실패 입력의 복구 비교 실행 계획

> Execute inline with execute-protocol and superpowers:executing-plans. Do not spawn agents, commit, restart services, or modify prior experiment artifacts. User `진행해줘` authorizes measurement and review.

**Goal:** 동일한 최초 실패에서 두 복구 방식의 시간과 의미 보존을 비교한다.

**Architecture:** 새 격리 worktree는 기존 후보 소스의 고정 사본을 사용한다. 평가 전용 wrapper가 첫 모델 호출을 기록된 예외로 대체하고 이후에는 실제 기존 파이프라인을 실행한다. 공유 서비스 코드를 수정하지 않는다.

**Tech Stack:** Python, 기존 Pydantic·Ollama·Orchestrator, JSONL private 실험 기록.

**Design:** [설계](../specs/2026-10-11-recovery-fixed-failure-design.md).

**Status:** T1~T4 완료. 측정 32턴과 워밍업 4턴, AI 검토 16쌍, 보존 감사를 완료했다. 후보 판정은 미채택이다. 서비스 기본값과 이전 실행 상태를 보존했다. 결과는 [비교 보고서](../../experiments/recovery-fixed-failure-2026-10-11.md)에 기록했다.

## T1. 설계와 입력 고정

- [x] `.worktrees/recovery-fixed-failure`에 기존 후보 소스·미커밋 기준을 복사하고 root/이전 worktree 해시를 보관한다.
- [x] `data/eval/reports/chat_pipeline/recovery-fixed-failure-2026-10-11/PREREG.md`에 위 설계와 판정 규칙을 기록한다.
- [x] `study.py prepare`로 이전 native-failures 중 `call == 1`인 8건을 추출한다. 실제 schema로 각 원문 오류를 확인한다. 입력에 원문·오류·질문·분석·근거·참고 답변과 초기 프롬프트를 보관한다.
- [x] CPU 사전 검사에서 실제 Orchestrator가 만든 두 payload가 마지막 복구 지시 외에는 같고, 두 EvidenceBundle이 원래 근거와 같음을 확인한다. 실제 실패 본문은 payload에 덧붙이지 않는다.
- [x] 실행기·소스·입력·payload 해시를 manifest에 고정한다. 기존 source 145개 해시도 다시 확인한다.

## T2. 두 회차 측정

- [x] `flock /tmp/claude-1000/myllmagent-gpu.lock env PYTHONPATH=src .venv/bin/python data/eval/reports/chat_pipeline/recovery-fixed-failure-2026-10-11/study.py run`을 격리 worktree에서 실행한다.
- [x] 회차별 워밍업 2턴을 별도 저장하고 16턴을 측정한다. 두 번째 회차는 쌍별 방식 순서를 반대로 한다. 완료 키는 재실행하지 않고 부분 결과를 보존한다.
- [x] 실제 호출 payload·raw 출력·파싱 결과·정책 검사·최종 응답·시간을 private 파일에 보관한다. 상태 출력에는 질문·본문을 쓰지 않는다.
- [x] 32턴의 최초 복구 호출과 전체 후속 정책 시간을 따로 집계한다. 실패도 분모에 포함한다.

## T3. 전체 의미 검토

- [x] 두 회차 16쌍 모두 질문·참고 답변·전체 근거와 전체 응답을 읽는다. 첫 복구 초안과 최종 정책 처리 답변의 차이도 기록한다.
- [x] 각 쌍에 AI reviewed (Codex), pass/fail/uncertain, 실제 문구와 근거에 따른 이유를 남긴다. 판단 불가를 pass로 바꾸지 않는다.
- [x] 지연 변화가 의미 보존 실패를 덮지 않도록 설계의 기준대로 판정한다.

## T4. 보고와 감사

- [x] `docs/experiments/recovery-fixed-failure-2026-10-11.md`에 회차별 결과·한계·사례·다음 판단을 한국어로 기록한다. 작은 세로 다이어그램을 사용한다.
- [x] root와 이전 worktree의 원래 파일·실행 상태, 새 실행 중 고정 소스·payload, HEAD와 stage 상태를 확인한다. 기존 미커밋 변경을 보존한다.
- [x] execute-protocol 상태와 계획을 실제 결과에 맞게 완료한다. 측정 완료를 서비스 지연 해결이나 자동 채택으로 표현하지 않는다.
