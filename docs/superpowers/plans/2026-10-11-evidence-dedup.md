# 입력 근거 중복 제거 시험 계획

> Execute inline using execute-protocol and executing-plans. User authorized the proposed exact-text pilot. No subagents, commit, source integration or service restart.

**Goal:** 원문·출처를 보존하는 전체 일치 중복 표기의 지연과 내용 영향을 확인한다.

**Architecture:** 평가 wrapper가 같은 전체 원문을 앞선 source의 Text 참조로 표현한다. 원본 bundle은 검사와 최종 응답에 유지한다. 대조 입력은 기존 메시지 그대로 전달한다.

**Tech Stack:** Python, Pydantic, pytest, 기존 Ollama/Generator/Orchestrator, private JSONL.

**Status:** 시험과 감사 완료. 후보는 속도·품질 기준 미통과로 미적용. [설계](../specs/2026-10-11-evidence-dedup-design.md).

## T1. 원문 보존과 고정

Files: `src/legal_rag/evaluation/evidence_dedup.py`, `tests/legal_rag/unit/test_evidence_dedup.py`, private `scan.json`, `inputs.jsonl`, `PREREG.json`, `manifest.json`, `pilot.py`.

- [x] 현 코드419개 파일과 이전7개 상태 해시를 보관하고 격리 worktree 및 새 canonical state를 만든다.
- [x] 다음 계약을 먼저 테스트하고 실패를 기록한다. `plan.restore_texts() == {e.source_id:e.text for e in bundle.items}`, 대조 `plan.render() == bundle.render()`, 서로 다른 출처의 헤더 보존, 부분/공백 차이 비변경, 여러 중복의 직접 첫 source 연결, 잘못된 source/cycle 차단, wrapper의 비근거 메시지/schema/응답/옵션 보존.
- [x] exact renderer와 wrapper를 구현한다. source metadata와 원본 bundle은 변경하지 않는다.
- [x] `env PYTHONPATH=src .venv/bin/python -m pytest tests/legal_rag/unit/test_evidence_dedup.py -q` 후 전체 backend와 웹 표시 관련3파일을 검증한다.
- [x] 80문항 roundtrip과 scan을 저장하고 지정4문항, payload, 설정, seed/순서, 기준을 추론 전에 고정한다.

## T2. 비교와 내용 검토

- [x] `flock /tmp/claude-1000/myllmagent-gpu.lock env PYTHONPATH=src .venv/bin/python data/eval/reports/chat_pipeline/evidence-dedup-2026-10-11/pilot.py run`으로 16턴과4warmup을 실행한다.
- [x] 원시 응답·실제 payload·초안·최종8필드·검사 및 전체/첫 호출 시간·token·prefill/decode 시간을 기록한다. 호출은 턴당 최대4회다.
- [x] 변경 문항을 A/A 대조와 구분해 두 회차의 시간·호출/오류·length·재생성을 집계한다.
- [x] 8쌍 전체 근거 검토에 `AI reviewed (Codex)` 및 pass/fail/uncertain 사유를 남긴다. 불확실 또는 새 중대한 오류가 있으면 적용하지 않는다.

## T3. 보고와 보존 감사

- [x] `docs/experiments/evidence-dedup-2026-10-11.md`에 변경량·시간·내용·검증·한계를 기록하고 새3문서만 root에 복사한다.
- [x] `audit_diff.py` 및 별도 hash/payload/source roundtrip/seed/order/metric 검증으로 허용 변경·기존419개 파일·이전7상태·HEAD/stage를 확인한다.
- [x] canonical task lifecycle을 관찰된 증거로 완료하고 새 mission만 COMPLETE로 기록한다.
