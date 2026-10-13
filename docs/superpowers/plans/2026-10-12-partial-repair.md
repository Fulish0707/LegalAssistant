# Partial Field Repair Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. No delegation, automatic commits or production adoption.

**Goal:** 정상 필드를 보존하는 복구 후보를 격리 구현하고 같은 최초 실패에서 비교한다.

**Architecture:** field_repair.py가 완성 필드 추출·동적 patch 스키마·엄격 병합을 담당한다. Generator는 기존 메시지 작성기를 공유하며 두 번째 JSON 호출만 선택 복구로 바꾼다. Orchestrator는 citations를 비운 본문 검사 결과가 정상일 때만 배열 복구를 선택한다.

**Tech Stack:** 기존 Python/Pydantic/Ollama/pytest, 독립 웹 테스트 환경.

## T1 — baseline·설계

- [x] root443파일·이전15상태·외부9파일을 해시로 보존하고 현재 작업 파일을 새 worktree로 복사한다.
- [x] 실행 상태를 별도로 만들고 필드 복구 계약·44턴·4예열·22리뷰 기준을 고정한다.

## T2 — 구현과 계약

**Files:** src/legal_rag/rag/field_repair.py, generator.py, orchestrator.py; src/legal_rag/llm/prompts/field_repair.md; tests/legal_rag/unit/test_field_repair.py,test_field_repair_extra.py.

- [x] RED: 정상analysis+잘린citations를 제공한 모델의 두 번째 요청이 citations/confidence 등 미완성 필드만 요구하는지 검사한다. analysis 안에서 잘리면 전체 LegalAnswer를 요구해야 한다.
- [x] 추출기는 json.JSONDecoder.raw_decode로 종료된 키·값을 순서대로 읽는다. TypeAdapter로 개별 필드를 검사한다. 미지키·중복키·잘못된구분자에서는 None을 반환한다.
- [x] RepairPlan은 fixed와targets를 보관한다. create_model로targets만 필수인extra=forbid스키마를 만들고, patch키가targets와 같을 때만 LegalAnswer.model_validate({**fixed,**patch})를 반환한다.
- [x] Generator.build_messages로 기존 메시지 작성을 그대로 추출한다. field_repair=False 기본값에서 원래 두 번 생성한다. True일 때 첫 StructuredOutputError 뒤 유효analysis가 있으면 patch 한 번, 아니면 원래 복구 지시 한 번을 사용한다. patch 실패는 GenerationError다.
- [x] Generator.repair_citations는 완성 답변에서citations만 대상인patch를 한 번 생성하고 병합한다. 호출오류는GenerationError로 기존 실패 정책에 넘긴다.
- [x] Orchestrator._generate_verified의 재생성 시점에서 citations=[] 답변을 같은검증 인자로 검사한다. report.ok일 때만 repair_citations, 아니면 기존generate를 사용한다. 전체답변은 다시backfill/verify한다.
- [x] PYTHONPATH=src .venv/bin/python -m pytest tests/legal_rag/unit/test_field_repair.py -q로 구조·호출·정책 테스트를 확인한다. 초기 요청 payload는 이전 original과 동일해야 한다.

## T3 — 측정과 검토

**Files:** data/eval/reports/chat_pipeline/partial-repair-2026-10-12/study.py,runtime.py,inputs.jsonl,PREREG.json,manifest.json,results.jsonl,warmup.jsonl,review.jsonl,metrics.json.

- [x] study.py freeze는 이전8JSON실패와 baseline3인용실패를 모두 저장하고 초기 메시지·자료·소스·스키마·실행기를 해시로 고정한다.
- [x] 첫 모델 응답만 기록된 실패로 주입하고 후속은 소유 실제Ollama서버에서 실행한다. 동일seed·ABBA44턴·예열4턴을 실행하고 모든 실제호출과정책을 저장한다.
- [x] metrics는 중앙값·nearest-rankp90·전체후속시간·출력토큰·호출수·patch대상·fallback·reject·strip을 집계한다.
- [x] 22쌍을 새로 읽어 AI reviewed (Codex) 사유와 원문/답변 해시를 저장한다. replay는 실제payload/동적스키마/파서/정책/보존필드/최종을 동일하게 재현한다.

## T4 — 회귀·보고·감사

- [x] 기존환경에서 PYTHONPATH=src .venv/bin/python -m pytest tests/legal_rag -q를 새로 실행한다. 독립 /tmp/core-first-web 환경으로 표시경계 tests/test_chat_streaming.py,tests/test_legal_backend.py,tests/test_citation_guard.py,tests/test_document_citation_notice.py를 실행한다. 웹 전체 통과로 표현하지 않는다.
- [x] docs/partial-repair-2026-10-12.md에 구조도·수치·적용분기·실패·범위·AI검토·채택판정을 기록한다.
- [x] baseline/manifest/회귀/리뷰/지표/요청재생/서버종료/기존파일/새3문서를 감사한다. root와이전상태를 유지하고 새canonical태스크만 완료한다.

[설계](../specs/2026-10-12-partial-repair-design.md)
