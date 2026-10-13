# 문서 목차

이 폴더의 문서는 네 종류다. 현재 안내는 이 폴더의 최상위에 있다. 과거 기록은 하위 폴더에 있다.

| 위치 | 내용 | 읽는 때 |
|---|---|---|
| `docs/*.md` | 현재 구현의 설치·실행·평가 안내 | 코드를 실행하거나 수정할 때 |
| `docs/experiments/` | 시험별 조건, 결과와 결정 | 같은 시험을 다시 하기 전에 |
| `docs/cleanup/` | 코드베이스 정리 조사와 실행 기록 | 파일의 이전 위치를 찾을 때 |
| `docs/superpowers/` | 기능별 설계(`specs/`)와 실행 계획(`plans/`) | 구현 이유를 확인할 때 |

## 현재 안내

- [법률 RAG 개발 안내](legal-rag-backend.md): 백엔드 설치, 자료 수집·색인, API.
- [평가 문서 목차](evaluation-index.md): 검색·답변 평가의 실행 안내와 유지되는 구현.
- [검색 진단](retrieval-diagnostics.md): 검색 설정 비교, 관련성 pool과 결과 해석.
- [구조 청크](structural-civil.md): 작은 구조 청크의 색인과 검색 비교.
- [Corpus relevance prompts](corpus-relevance-prompts.md): corpus별 관련성 채점.
- [채점기: core conclusion](answer-grader-core.md): 현재 답변 채점 방법.
- [Answer grader v3 calibration](answer-grader-v3-calibration.md): 저장 답변의 v2/v3 채점 비교.
- [Golden-answer comparison](golden-answer-comparison.md): 원본/구조 청크 답변 비교.
- [Local QLoRA](local-qlora.md), [Full QLoRA](full-qlora.md): 학습 데이터 준비와 adapter 평가.
- [Integration roadmap](integration-roadmap.md): 웹 앱과 백엔드의 통합 단계.
- `images/`: README의 화면 그림.

## 시험 기록 (`experiments/`)

원자료는 `data/eval/reports/`에 있다. 이 폴더는 Git 제외 대상이므로 아래 문서가 Git에 남는 결과 기록이다.

답변 시간과 복구 (2026-10-10 ~ 10-12). 정리 상태는 [미채택 시험 정리 기록](experiments/unsuccessful-experiments-cleanup-2026-10-13.md)에 있다.

- [검증 후 답변 시간 측정](experiments/verified-chat-latency-2026-10-10.md)
- [답변 반복 억제와 구조화 복구](experiments/answer-recovery-latency-2026-10-10.md)
- [인용 처리 지연 개선](experiments/citation-latency-2026-10-11.md)
- [컨텍스트 실행 예산](experiments/context-budget-2026-10-11.md)
- [입력 근거 중복 제거](experiments/evidence-dedup-2026-10-11.md)
- [첫 생성 한도 비교](experiments/first-pass-budget-2026-10-11.md)
- [KV 캐시 정밀도](experiments/kv-cache-2026-10-11.md)
- [입력 처리 배치](experiments/prefill-batch-2026-10-11.md)
- [같은 최초 실패 입력의 복구 비교](experiments/recovery-fixed-failure-2026-10-11.md)
- [Temperature 0](experiments/temperature-zero-2026-10-11.md)
- [검증된 핵심 답변 우선 표시](experiments/core-first-2026-10-12.md)
- [실패 필드만 복구](experiments/partial-repair-2026-10-12.md)
- [답변 의미 검증](experiments/semantic-verification-2026-10-12.md)
- [요약 필드 우선 생성](experiments/summary-first-2026-10-12.md)

관련성 채점 prompt.

- [관련성 Judge 3차 프롬프트](experiments/relevance-judge-prompt-v3.md)
- [1·2차 비교와 3차 수정 근거](experiments/relevance-judge-v3-review.md)

## 정리 기록 (`cleanup/`)

- [정리 조사 2026-10-04](cleanup/codebase-cleanup-audit-2026-10-04.md)와 [인벤토리](cleanup/codebase-cleanup-inventory-2026-10-04.json)
- [정리 실행 2026-10-05](cleanup/codebase-cleanup-2026-10-05.md)와 [삭제 목록](cleanup/codebase-cleanup-actions-2026-10-05.json)
- [Dead code 정리 2026-10-05](cleanup/dead-code-cleanup-2026-10-05.md)와 [삭제 목록](cleanup/dead-code-cleanup-actions-2026-10-05.json)
- [정리 기록 2026-10-13](cleanup/codebase-cleanup-2026-10-13.md): 테스트 위치, evaluation 폴더, `webapp/`, 문서 폴더.

조사와 인벤토리는 작성 당시의 경로를 기록한다. 현재 경로와 다를 수 있다.

## 새 문서의 위치

- 현재 구현의 안내는 최상위에 둔다.
- 날짜가 있는 시험 결과는 `experiments/`에 둔다.
- 정리 기록은 `cleanup/`에 둔다.
- 설계와 계획은 `superpowers/specs/`, `superpowers/plans/`에 둔다.
