# 코드베이스 정리 조사 — 2026-10-04

2026-10-05 실행 내역은 [정리 실행 기록](codebase-cleanup-2026-10-05.md)과
[실제 삭제 목록](codebase-cleanup-actions-2026-10-05.json)을 참고한다.
아래 수치와 분류는 2026-10-04 조사 시점의 snapshot이다.

## 조사 결과

메인 작업 폴더의 실행 코드, 문서, 실험 스크립트, 일반 로그와 캐시를 확인했다. 현재 Git 작업 트리는 조사 시작 시 깨끗했으며 HEAD는 `470b627`이다. 이 문서와 상세 인벤토리만 추가했다. 실제 코드 병합, 파일 이동, 삭제, 테스트 실행은 수행하지 않았다.

가상환경, 다른 worktree, 모델 파일, Qdrant/Chroma/SQLite 데이터와 대규모 문서·학습 데이터는 소스 중복/캐시 검색에서 제외했다. 아래 데이터 용량은 별도로 측정했다. `.env`, API 키, 로그 본문은 목록에 수집하지 않았다.

| 항목 | 확인 결과 |
|---|---:|
| 결과 폴더 내부 실행 스크립트 | 75개 |
| 재생성 가능한 캐시 폴더 | 34개 / 3.79 MiB |
| `.log` / `.out` | 44개 / 1.03 MiB |
| 바이트 단위로 동일한 비어 있지 않은 소스/문서 | README 6개 묶음 1건; 조사 범위 내 동일 Python 소스 없음 |

중복은 파일 내용이 완전히 같은 경우와 역할이 겹치는 경우를 구분했다. 아래 통합 후보는 공통 기능을 추출할 수 있다는 뜻이며, 실험별 설정·채점 기준까지 같다는 뜻은 아니다.

## 1. 통합 후보 — 실행 코드

| 우선순위 | 현재 파일 | 통합 대상 / 방법 | 보존 조건 |
|---|---|---|
| 1 | `data/eval/reports/chat_pipeline/gemini-2026-09-30/replay.py`, `scripts/eval_golden_answers.py` | 동결된 근거에 대한 생성·검증·재시도 기능을 `src/legal_rag/evaluation/answer_replay.py`로 추출 | `SOURCE_FILES`/source hash, 요청 간격, full answer 저장, 재생성 정책 유지. 기존 `replay.py`는 현재 `runpy.run_path`로 불러오므로 추출 전 삭제 금지 |
| 2 | `data/eval/reports/chat_pipeline/structural-answer-pair-2026-10-03/{run.py,summarize.py}`, `structural-golden-80-2026-10-03/finish.py`, `gemini-2026-09-30/{judge.py,retry_judge_pair.py,report.py,report_bounds.py}` | 현행 `scripts/eval_golden_answers.py`, `scripts/judge_answers.py`, `answer_run.py`, `answer_reporting.py`로 재사용 가능한 실행/요약 기능 수용 | 과거 20문항, 80문항과 legacy/v2/v3 채점 기준을 별도 설정으로 유지. 이전 점수나 원본 판단을 새 기준으로 덮어쓰지 않음 |
| 3 | `data/eval/reports/structural-civil-test-2026-10-02/{run_sparse.py,run_hybrid.py,summarize.py}`, `structural-all-2026-10-03/{run_original_baseline.py,compare.py}`, `scripts/run_structural_all.sh` | 기존 `scripts/eval_retrieval.py`, `scripts/compare_retrieval_modes.py` 중심으로 원본/구조 청크 비교를 옵션화 | civil-only/four-index fanout, sparse/hybrid, neighbor 설정과 표본 수 구분. 공통 결과 포맷을 만든 후 이전 도우미 보관 |
| 4 | `data/eval/reports/domain-prompt-pilot-2026-10-03/{prepare.py,run.py,report.py,shared_judge.py}`, `criminal-prompt-revision-2026-10-03/report.py` | `scripts/judge_relevance.py`와 패키지 relevance 평가 기능을 사용하는 공통 prompt 비교 runner로 통합 | shared/corpus/criminal 후보 프롬프트와 frozen hash 유지. `shared_judge.py`는 과거 평가 snapshot이므로 구현 재사용과 원본 보관을 구분 |
| 5 | `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/{stage1_candidates.py,stage2_rerank.py,score_len.py,score_len2.py,guard.py}`, `diag/{embed_trunc.py,rerank_cover.py}` | 유지되는 reranker 진단 CLI로 승격; 후보 수집/재정렬/점수화 공통화 | A/B corpus 구성, 512/1024/2048, PREREG 기준 유지. `score_len.py`와 `score_len2.py`는 평가 조건이 다른 버전이라 한 파일을 단순 삭제하지 않음 |
| 6 | `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/{bulk_fetch.py,fetch_laws.py,resume_index.py,run_probe.py,run_full.py,run_control.py,score_full.py,diag_miss.py,gold_titles.py,probes_raw.py}` | 수집/재개는 `src/legal_rag/ingestion/{law_api.py,law_api_cli.py}`와 기존 index CLI 중심으로 정리; 실험·진단은 별도 평가 옵션화 | 원본 API 응답과 fetch 기록, 인덱스 resume 보호 유지. 현재 법령 수집 작업과 연결되어 있는지 확인 후 이동 |
| 7 | `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/{build_sample.py,verify_ab.py,judge_ab.py,judge_probes.py,analyze.py,extras.py,rubric.py}`, `probes/analyze_probes.py`, `gen-model-ab-2026-10-04/run.sh` | saved-answer model 비교 runner 공통화: 모델·표본·arm·예산은 명시적 설정 | 현재 model A/B 실험은 미완료. 기존 qwen/Gemini 설정 및 frozen rubric 원본 보관; 메모리 guard와 종료 신호 동작 검토 후 재사용 |
| 8 | `data/eval/reports/chat_pipeline/quick-review-2026-10-01/{prepare.py,export_completed.py}`, `formal-2026-09-28-articles/failure-breakdown/*.py`, `structural-all-2026-10-03/*ip*.py` | 공통 실패 분석/export 기능을 기존 `diagnose_retrieval.py`, `sample_calibration.py` 또는 별도 분석 모듈에 추출 | 사람의 수동 라벨과 model 판단을 구분해 저장. 새 분석이 원본 라벨·참고 답변을 수정하지 않도록 함 |

이미 완료된 `answer_run`, `answer_reporting`, `answer_rubrics`, `answer_judge_runner` 분리는 유지한다. 위 1번은 이 정리에서 남아 있는 직접 실행 의존성을 해결하므로 다음 구현 우선순위가 가장 높다.

## 2. 통합 후보 — 문서

| 현재 문서 | 정리 방향 |
|---|---|
| `docs/relevance-judge-prompt-v3.md`, `docs/relevance-judge-v3-review.md`, `docs/corpus-relevance-prompts.md` | `corpus-relevance-prompts.md`를 현재 명령·routing의 기준 문서로 유지. 프롬프트 전문과 변경 근거는 부록/실험 기록으로 연결 |
| `docs/structural-civil.md`, `docs/golden-answer-comparison.md`, `docs/answer-grader-v3-calibration.md` | 결과를 하나로 섞지 않고 `docs/evaluation-index.md` 목차로 연결. 검색 실험, 생성 답변 비교, grader calibration 역할을 명시 |
| `docs/local-qlora.md`, `docs/full-qlora.md` | 공통 환경 설정·실행 안내만 통합. pilot/full 데이터와 base+adapter 평가 조건은 별도 절 유지 |
| `docs/superpowers/plans/`, `docs/superpowers/specs/`의 완료된 기록 | 현재 실행 안내와 완료 기록을 목차로 구분. 링크·결정 이력 확인 없이 대량 삭제하지 않음 |

문서 링크를 갱신하기 전 기존 파일을 삭제하면 README와 과거 실험 기록이 끊어질 수 있다.

## 3. 불필요하거나 재생성 가능한 파일 후보

| 후보 | 분류 | 근거 / 처리 |
|---|---|
| 아래 부록의 `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` 34개 | 삭제 가능 후보 | 실행 시 재생성됨. 조사 대상 파일 내용 기준 합계 3.79 MiB |
| `package-lock.json` | 삭제 가능 후보 | `packages: {}`인 빈 npm lockfile이며 루트 `package.json`이 없다. npm workflow 정적 참조도 발견하지 못함 |
| `data/data/` | 빈 폴더 삭제 후보 | 하위 실제 파일이 없음. 의미 없는 중첩 경로 |
| `google_api.ipynb` | 수동 사용 확인 후 보관 후보 | 15셀 실험 노트북. 실행 코드/문서에서 파일명 참조를 발견하지 못했지만 수동 실행 여부는 정적 검색으로 판단할 수 없음 |
| 아래 부록의 일반 로그 44개 | 완료 확인 후 압축·보관 후보 | 합계 1.03 MiB로 공간 절약 효과는 작음. 최신 fetch/model 실험 로그는 보류 |
| `data/eval/reports/retrieval_diagnostics/ip-v1/intellectual_property/*/README.md` 중 동일한 6개 | 문서 중복 정리 후보 | SHA256이 동일함. 상위 공통 설명을 두고 각 결과 폴더에서 연결 가능하나 독립적인 결과 폴더 설명을 유지하는 편도 합리적 |

일반 `.log`와 `answer_judge_attempts-*.jsonl`, `judgments*.jsonl`, `fetch_log*.jsonl`은 다르다. 후자는 검증 실패·재개·수집 이력의 근거이므로 이름에 log/attempts가 있다는 이유로 삭제하지 않는다.

## 4. 공간 절약 효과가 큰 보관 후보

| 경로 | 파일 내용 크기 | 처리 조건 |
|---|---:|---|
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/candidates.jsonl` | 217.67 MiB | 재정렬 길이 실험의 동결된 후보. 실험 종료 및 입력 hash 보존 후 압축 보관 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/candidates_B.jsonl` | 176.04 MiB | B arm 후보. A와 corpus 구성이 다르므로 중복 파일로 처리하지 않음 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/` | 약 108 MiB (디스크 할당) | 참고 질문의 기준 데이터와 heldout/prereg 기록 보존. 완료된 replay 중간물만 별도 압축 검토 |
| `data/eval/reports/structural-all-2026-10-03/` | 약 125 MiB (디스크 할당) | baseline/diagnosis/snapshot을 구분해 보관; 결과와 입력 fingerprint 유지 |

압축 전에 파일 SHA256을 기록하고, 원본 해제 시 같은 hash가 나오는지 확인할 수 있도록 한다. 기존 runner가 경로를 사용하거나 재개를 지원하는 디렉터리는 압축 후 자동 삭제 대상으로 삼지 않는다. 두 후보 파일의 합계는 약 393.72 MiB이다.

## 5. 보존하거나 추가 확인할 대상

- `data/qdrant/`, `data/indexes/`, `data/processed/`: 실제 서비스/검색 데이터와 인덱스. 단순 로그 정리 대상으로 삭제하지 않음.
- `data/training/`, `models/adapters/`, `sft_qlora.ipynb`, `sft_qlora_full.ipynb`: 학습 데이터·어댑터·실행 흐름. 실패한 run이라도 체크포인트/학습 이력 유무를 확인한 후 개별 판단.
- `data/api_laws/`의 API 원본 JSON과 `fetch_log*.jsonl`: 법령 원문·수집 재개 근거. 다른 에이전트 작업과 연결될 수 있어 보존.
- `data/eval/reports/chat_pipeline/structural-golden-80-2026-10-03/`와 `...-calibration-v3-ctx16384/`: 원본/구조 청크·v2/v3 비교 기준. 양쪽 manifest, source/answer/evidence, 판단 기록을 함께 보존.
- `data/eval/reports/structural-all-2026-10-03/judge-repair/{frozen_agreement_cli.py,frozen_judge_cli.py,relevance_judge_before.py}`: 평가 변경을 재현하는 snapshot. 현행 코드와 겹쳐 보여도 삭제 금지.
- `evals/`와 `tests/`: 전자는 Flask/LangGraph 에이전트 평가, 후자는 앱/법률 backend 검증이다. 같은 이름의 평가 도구로 보고 합치지 않음.
- `statistics_analysis.py`: `app.py`가 실제로 import하며 관련 테스트도 있다. 실험용 이름처럼 보여도 유지.
- `.execute-protocol/`의 상태·lock 파일: 오래되었다는 이유로 삭제하지 않고 활성 mission/lock 사용 여부 확인.
- `.env`, 사용자 DB, uploads, 기존 가상환경: 이번 실험 정리 대상에서 제외.

### 진행 중이거나 미완료인 실험

`data/eval/reports/chat_pipeline/gen-model-ab-2026-10-04/`는 최종 비교 보고서가 없고 조사 시 q4b `verify.jsonl`에 7개 행만 있었다(목표 80). 조사 중 폴더 크기도 증가했다. 현재 실행·재개에 필요한 `run.sh`, `run.out`, `q4b/run.log`와 저장 답변을 유지한다. 행 수는 조사 시점의 snapshot이며 이후 달라질 수 있다.

### worktree

등록된 worktree는 13개다. `.worktrees/answer-cleanup`의 cleanup 파일 12개는 현재 메인 작업 폴더와 바이트 단위로 일치한다. 다만 worktree에는 다른 수정/미추적 파일도 있으므로 통째로 삭제하기 전에 나머지 변경과 활성 작업을 확인해야 한다. 나머지 worktree도 이름이나 생성 시점만으로 삭제 여부를 판단하지 않는다.

## 6. 저장 공간 요약

아래는 `du`로 확인한 디스크 할당량이다. 캐시/로그 표의 파일 내용 크기와 측정 방식이 다르다.

| 경로 | 디스크 할당 |
|---|---:|
| `data/qdrant/` | 27.17 GiB |
| `data/processed/` | 14.45 GiB |
| `data/training/` | 2.24 GiB |
| `data/eval/` | 1.28 GiB |
| `data/indexes/` | 298.19 MiB |
| `data/api_laws/` | 233.28 MiB |
| `models/` | 300.32 MiB |

Qdrant와 처리 문서가 대부분을 차지한다. 일반 로그와 캐시를 정리하는 목적은 용량 확보보다 파일 탐색·운영 흐름을 단순하게 만드는 데 있다.

## 7. 추천 실행 순서

1. 캐시·빈 파일/폴더 후보 정리. 일반 로그는 완료된 run부터 압축하고 미완료 run은 유지.
2. Gemini replay 직접 의존성을 패키지 코드로 옮기고 source hash 정책을 유지.
3. retrieval / prompt / reranker / model A/B runner를 각각 통합. 서로 다른 실험의 평가 조건은 설정으로 분리.
4. 현재 명령과 실험 결과를 연결하는 평가 목차를 추가하고 오래된 운영 안내를 정리.
5. 완료 실험 중간물을 hash 목록과 함께 압축 보관하고, 마지막으로 worktree 정리 여부 확인.

이 목록은 후속 정리 작업의 기준으로 사용한다. 로그·중간물·worktree는 각 항목의 보존 조건을 먼저 확인한다.

## 부록 A. 캐시 폴더 전체 목록

| 경로 | 파일 수 | 내용 크기 |
|---|---:|---:|
| `.ruff_cache/` | 7 | 124.93 KiB |
| `__pycache__/` | 48 | 526.89 KiB |
| `scripts/__pycache__/` | 12 | 166.65 KiB |
| `evals/__pycache__/` | 2 | 3.95 KiB |
| `.pytest_cache/` | 5 | 5.41 KiB |
| `src/legal_rag/ingestion/__pycache__/` | 13 | 74.22 KiB |
| `src/legal_rag/ingestion/normalize/__pycache__/` | 5 | 20.30 KiB |
| `src/legal_rag/__pycache__/` | 4 | 18.46 KiB |
| `src/legal_rag/config/__pycache__/` | 3 | 15.42 KiB |
| `src/legal_rag/verification/__pycache__/` | 5 | 19.95 KiB |
| `src/legal_rag/api/__pycache__/` | 4 | 11.88 KiB |
| `src/legal_rag/textutil/__pycache__/` | 4 | 8.50 KiB |
| `src/legal_rag/rag/__pycache__/` | 9 | 43.25 KiB |
| `src/legal_rag/training/__pycache__/` | 8 | 81.17 KiB |
| `src/legal_rag/reranking/__pycache__/` | 4 | 8.09 KiB |
| `src/legal_rag/evaluation/__pycache__/` | 30 | 345.02 KiB |
| `src/legal_rag/embeddings/__pycache__/` | 4 | 8.25 KiB |
| `src/legal_rag/schemas/__pycache__/` | 11 | 30.48 KiB |
| `src/legal_rag/chunking/__pycache__/` | 10 | 58.75 KiB |
| `src/legal_rag/llm/__pycache__/` | 6 | 21.52 KiB |
| `src/legal_rag/llm/prompts/__pycache__/` | 1 | 746.00 B |
| `src/legal_rag/retrieval/__pycache__/` | 11 | 62.50 KiB |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/__pycache__/` | 2 | 5.64 KiB |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/__pycache__/` | 1 | 2.51 KiB |
| `data/eval/reports/structural-all-2026-10-03/judge-repair/__pycache__/` | 1 | 14.16 KiB |
| `data/eval/reports/domain-prompt-pilot-2026-10-03/__pycache__/` | 1 | 19.08 KiB |
| `tests/__pycache__/` | 30 | 332.92 KiB |
| `tests/legal_rag/live/__pycache__/` | 2 | 7.73 KiB |
| `tests/legal_rag/__pycache__/` | 5 | 101.72 KiB |
| `tests/legal_rag/unit/__pycache__/` | 38 | 802.38 KiB |
| `tests/legal_rag/training/__pycache__/` | 4 | 104.67 KiB |
| `tests/legal_rag/evaluation/__pycache__/` | 18 | 727.93 KiB |
| `tests/legal_rag/integration/__pycache__/` | 6 | 87.37 KiB |
| `tests/legal_rag/llm/__pycache__/` | 2 | 16.80 KiB |

## 부록 B. 일반 로그 전체 목록

| 경로 | 내용 크기 | 분류 |
|---|---:|---|
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/res_full_existing.log` | 4.50 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/res_full_full.log` | 4.50 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/gen-model-ab-2026-10-04/run.out` | 133.00 B | 완료/활성 상태 확인 전 보류 |
| `data/eval/reports/chat_pipeline/gen-model-ab-2026-10-04/q4b/run.log` | 1.97 KiB | 완료/활성 상태 확인 전 보류 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs/gemini-B.log` | 4.85 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs/gemini-A.log` | 4.86 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs/judge-local-B.log` | 3.92 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs/judge-gemini-B.log` | 3.92 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs/local-A.log` | 4.85 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs/judge-local-A.log` | 3.87 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs/local-B.log` | 4.85 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs/judge-gemini-A.log` | 3.89 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs-probes/gemini-B.log` | 2.11 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs-probes/gemini-A.log` | 4.01 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs-probes/judge-local-B.log` | 559.00 B | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs-probes/judge-gemini-B.log` | 569.00 B | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs-probes/local-A.log` | 7.17 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs-probes/judge-local-A.log` | 553.00 B | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs-probes/local-B.log` | 10.81 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/runs-probes/judge-gemini-A.log` | 567.00 B | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/other_retrieve.log` | 2.30 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/harness-article-coverage.log` | 302.00 B | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/article_rerank.log` | 6.46 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/score_segments_other.log` | 4.89 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/queries.log` | 4.59 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/score_segments.log` | 3.77 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/retrieve.log` | 1.87 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/other_queries.log` | 6.09 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/structural-golden-80-2026-10-03/completion.log` | 474.00 B | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/stage1_B.log` | 16.36 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/stage1.log` | 16.36 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/stage2_B.log` | 1.66 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/stage2.log` | 1.68 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/structural-all-2026-10-03/criminal-index.log` | 214.15 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/structural-all-2026-10-03/civil-evaluation.log` | 1.31 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/structural-all-2026-10-03/administrative-evaluation.log` | 1.32 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/structural-all-2026-10-03/criminal-evaluation.log` | 1.31 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/structural-all-2026-10-03/ip-index.log` | 298.48 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/structural-all-2026-10-03/administrative-index.log` | 252.21 KiB | 완료 확인 후 보관 후보 |
| `data/eval/reports/structural-all-2026-10-03/ip-evaluation.log` | 1.30 KiB | 완료 확인 후 보관 후보 |
| `data/api_laws/resume_index_full.log` | 72.51 KiB | 완료/활성 상태 확인 전 보류 |
| `data/api_laws/build_index_full.log` | 50.09 KiB | 완료/활성 상태 확인 전 보류 |
| `data/api_laws/resume_index_full2.log` | 18.19 KiB | 완료/활성 상태 확인 전 보류 |
| `data/api_laws/bulk_fetch.log` | 2.35 KiB | 완료/활성 상태 확인 전 보류 |

## 부록 C. 결과 폴더 내부 실행 스크립트 전체 목록

아래는 메인 결과 폴더에서 발견한 전체 75개다. 파일명별 함수 목록과 bytes는 [상세 인벤토리](codebase-cleanup-inventory-2026-10-04.json)에 있다. 통합할 때 참조·source hash를 확인하고, 과거 재현용 원본은 보관한다.

| 경로 | 분류 |
|---|---|
| `data/eval/relevance/ip-v1-pilot-200/experiments/think-comparison/run_comparison.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/relevance/ip-v1-pilot-200/experiments/think-comparison/summarize.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/gemini-2026-09-30/replay.py` | 현재 직접 실행 의존성 — 유지 |
| `data/eval/reports/chat_pipeline/gemini-2026-09-30/report_bounds.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/gemini-2026-09-30/retry_judge_pair.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/gemini-2026-09-30/judge.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/gemini-2026-09-30/prepare.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/gemini-2026-09-30/report.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/quick-review-2026-10-01/prepare.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/quick-review-2026-10-01/export_completed.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/run_full.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/score_full.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/resume_index.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/bulk_fetch.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/gold_titles.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/run_probe.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/diag_miss.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/probes_raw.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/fetch_laws.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/statute-api-pilot-2026-10-01/scripts/run_control.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/gen-model-ab-2026-10-04/run.sh` | 미완료 실험 runner — 유지 |
| `data/eval/reports/chat_pipeline/structural-answer-pair-2026-10-03/run.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/structural-answer-pair-2026-10-03/summarize.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/rubric.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/build_sample.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/extras.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/judge_probes.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/analyze.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/verify_ab.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/judge_ab.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/full-corpus-answer-ab-2026-10-02/probes/analyze_probes.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/compare_runs.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/refuse_candidates.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/statute_targets.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/replay_retrieval.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/article_rerank.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/select_variants.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/stage_ranks.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/statute-replay/score_segments.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/failure-breakdown/article_coverage.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/failure-breakdown/where_is_key.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/failure-breakdown/k8_flip_check.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/formal-2026-09-28-articles/failure-breakdown/show_failures.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/structural-golden-80-2026-10-03/finish.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/score_len.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/stage2_rerank.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/stage1_candidates.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/score_len2.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/guard.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/diag/rerank_cover.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chat_pipeline/reranker-length-2026-10-03/diag/embed_trunc.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/chunk-size-review-2026-10-01/analyze.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/report_ip_calibration.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/diagnose_ip.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/score_ip_gold.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/prepare_ip_calibration.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/run_original_baseline.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/compare.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/label_ip_calibration.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/summarize_ip.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/judge-repair/run_frozen_relevance.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/judge-repair/prepare_fresh_calibration.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/judge-repair/compare_fresh.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/judge-repair/label_fresh_calibration.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-all-2026-10-03/judge-repair/frozen_judge_cli.py` | 동결된 과거 evaluator — 보존 |
| `data/eval/reports/structural-all-2026-10-03/judge-repair/relevance_judge_before.py` | 동결된 과거 evaluator — 보존 |
| `data/eval/reports/structural-all-2026-10-03/judge-repair/frozen_agreement_cli.py` | 동결된 과거 evaluator — 보존 |
| `data/eval/reports/domain-prompt-pilot-2026-10-03/shared_judge.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/domain-prompt-pilot-2026-10-03/run.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/domain-prompt-pilot-2026-10-03/prepare.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/domain-prompt-pilot-2026-10-03/report.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/criminal-prompt-revision-2026-10-03/report.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-civil-test-2026-10-02/run_sparse.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-civil-test-2026-10-02/summarize.py` | 공통 기능 추출 / 과거 실행본 보관 |
| `data/eval/reports/structural-civil-test-2026-10-02/run_hybrid.py` | 공통 기능 추출 / 과거 실행본 보관 |
