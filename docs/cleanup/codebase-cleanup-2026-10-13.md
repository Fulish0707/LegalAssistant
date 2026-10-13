# 코드베이스 정리 기록 — 2026-10-13

정리일: 2026-10-13.

## 정리 범위

테스트 파일의 위치, `data/eval/reports/`의 낱개 결과 파일과 `src/legal_rag/evaluation/`의 폴더 구조를 정리했다. 테스트 내용, 결과 데이터와 기존 실험 폴더 이름은 바꾸지 않았다.

## 테스트 파일 이동

`tests/legal_rag/` 아래에서 폴더 규칙에 맞지 않는 파일을 옮겼다. 이동한 파일의 코드는 한 줄만 수정했다.

| 이전 위치 | 현재 위치 | 이유 |
|---|---|---|
| `test_import_processed.py` | `integration/` | factory와 index builder를 함께 사용한다 |
| `test_streaming_index.py` | `integration/` | factory와 index builder를 함께 사용한다 |
| `test_rechunk_corpus.py` | `unit/` | chunking과 rechunk 모듈만 사용한다 |
| `unit/test_qa_dataset.py` | `evaluation/` | `legal_rag.evaluation` 모듈의 테스트다 |
| `unit/test_retrieval_metrics.py` | `evaluation/` | `legal_rag.evaluation` 모듈의 테스트다 |
| `llm/test_transformers_chat.py` | `unit/` | `llm/` 폴더에 테스트가 하나뿐이었다. 다른 LLM 테스트는 `unit/`에 있다 |

- `integration/test_import_processed.py`의 저장소 루트 경로를 `parents[2]`에서 `parents[3]`으로 수정했다.
- 빈 `tests/legal_rag/llm/__init__.py`를 제거했다.
- 루트 `tests/test_*.py`(Flask 앱 테스트)는 아래 `webapp/` 이동에서 함께 옮겼다.

## 테스트 파일 이름 변경

대상 모듈과 이름이 다른 파일 3개의 이름을 바꿨다. 테스트 내용은 같다.

| 이전 이름 | 현재 이름 | 대상 |
|---|---|---|
| `evaluation/test_retrieval_diagnostics.py` | `evaluation/test_diagnostics.py` | `evaluation.diagnostics` |
| `evaluation/test_diagnostic_reports.py` | `evaluation/test_reports.py` | `evaluation.reports` |
| `unit/test_verified_latency.py` | `unit/test_embedding_reuse_and_answer_style.py` | 질의 임베딩 재사용과 답변 style 설정 |

## 중복 helper 통합

- `scripts/sweep_fusion_weights.py`와 `scripts/rerank_relevance_pool.py`의 `_needed_chunks`를 `legal_rag.evaluation.pool_extension.needed_chunks` 하나로 합쳤다. 누락 chunk의 오류 문구는 하나로 통일했다.
- `test_pool_extension.py`는 `test_relevance_pool.py`의 `make_chunk`를 사용한다.
- `test_rerank_run.py`와 `test_relevance_pool.py`는 `test_calibration.py`의 `_load_script`를 사용한다.
- `test_gemini.py`는 `test_llm.py`의 `Out`을 사용한다.

## evaluation 패키지 분리

`src/legal_rag/evaluation/`의 모듈 29개를 세 폴더로 옮겼다. 파일 이름과 로직은 같다. `src/`, `scripts/`, `tests/`의 import 경로를 직접 수정했고 이전 경로의 재수출은 두지 않았다.

| 폴더 | 모듈 |
|---|---|
| `answers/` | `answer_compare`, `answer_judge`, `answer_judge_v3`, `answer_judge_core`, `answer_judge_runner`, `answer_rubrics`, `answer_run`, `answer_replay`, `answer_reporting`, `reference_overlap`, `frozen`, `spans` |
| `relevance/` | `relevance_judge`, `relevance_pool`, `relevance_prompts`, `corpus_guidance`, `judgment_store`, `calibration`, `pool_extension`, `case_labels`, `fusion_sweep`, `rerank_run`, `graded_metrics`, `graded_report`, `fanout_summary` |
| `diagnosis/` | `diagnostics`, `diagnostic_runner`, `reports`, `mode_comparison`, `crossdomain_target` |
| 최상위 유지 | `datasets`, `qa_dataset`, `retrieval_metrics`, `retrieval_runner` |

- 구절 분할 함수와 `SPAN_MAX_CHARS`를 v2 채점기(`answer_judge.py`)에서 `answers/spans.py`로 옮겼다. 기본 채점기(`answer_judge_core.py`)는 v2 모듈을 import하지 않는다.
- v2·v3 채점기, rubric 선택, runner와 `calibrate_golden_answers.py`는 유지했다. 저장된 v2·v3 판단은 계속 보고할 수 있다.
- 저장 답변 120건에서 구절 분할 결과, v2 input fingerprint와 v2 선택 요청(`selection_request`의 schema, 목록, 지시문)이 변경 전과 같았다.
- worktree 3개(`answer-cleanup`, `rag-recall-evaluation`, `retrieval-diagnostics`)에 `src/legal_rag/evaluation/`의 미커밋 변경이 있다. 이 변경을 병합할 때 새 폴더 경로에 맞춰야 한다.
- `data/eval/reports/`의 과거 실행본 Python 46개와 manifest 42개는 이전 모듈 경로를 기록한다. 당시 snapshot이므로 수정하지 않았다. 과거 실행본을 다시 실행하려면 import 경로를 맞춰야 한다.
- 테스트 폴더 `tests/legal_rag/evaluation/`은 나누지 않았다.

## 루트 파일

- `sft_qlora.ipynb`, `sft_qlora_full.ipynb`: 저장소 루트를 작업 폴더로 요구하므로 옮기지 않았다.
- `env.yml`: `docs/local-qlora.md`가 참조하고 기본 shell 환경(`llm`)을 기술하므로 유지했다.
- `google_api.ipynb`: 10-04 조사에서 보관 후보로 표시했다. 수동 사용 여부를 확인하지 않았으므로 유지했다.

## Flask 웹 앱 폴더

루트의 Flask 앱을 `webapp/`으로 옮겼다. 백엔드(`src/legal_rag/`)와 웹 앱은 실행 환경이 다른 두 애플리케이션이다.

| 이전 위치 | 현재 위치 |
|---|---|
| 루트의 `app.py`, `legal_backend.py`, `citation_guard.py`, `langchain_config.py`, `llm_model.py`, `llm_ocr.py`, `llm_prompt.py`, `quota.py`, `html_sanitizer.py` | `webapp/` |
| `templates/`, `static/`, `requirements.txt` | `webapp/` |
| `tests/test_*.py` 17개 | `webapp/tests/` |

- 실행 명령은 저장소 루트에서 `python webapp/app.py`다. 테스트 명령은 `python -m pytest webapp/tests -k 'not Live' -q`다.
- 데이터 폴더(`sqlite_db/`, `uploads/`, `chroma_db/`)는 루트에 유지했다. `APP_STATE_DB`와 `RAG_PERSIST_DIR`의 상대 경로, 기본값과 업로드 폴더는 실행 위치가 아닌 저장소 루트를 기준으로 한다. 루트에서 실행하면 이전과 같은 파일을 사용한다.
- 앱 모듈 사이의 import와 테스트 코드는 수정하지 않았다. `webapp/`은 패키지가 아니며 `webapp/tests/`는 `tests` 패키지다.
- `scripts/eval_chat_pipeline.py`는 `citation_guard`를 `webapp/`에서 가져온다.
- 정적 파일 응답을 확인하는 테스트 1개를 `test_health_reporting.py`에 추가했다.
- `webapp/tests`와 `tests/legal_rag`는 최상위 패키지 이름이 같으므로 한 pytest 실행에서 함께 수집할 수 없다. 두 테스트는 이전에도 서로 다른 Python 환경에서 실행했다.
- `data/eval/reports/`의 과거 실행본 Python 5개는 루트의 웹 앱 모듈을 import한다. 당시 snapshot이므로 수정하지 않았다.
- worktree 3개(`local-multimodal`, `rag-recall-evaluation`, `security-2026-10-07`)에 이동한 파일의 미커밋 변경이 있다.

## 문서 폴더

`docs/` 최상위에는 현재 안내만 둔다. 과거 기록은 두 폴더로 옮겼다. 문서 내용은 수정하지 않았다.

| 폴더 | 옮긴 파일 |
|---|---|
| `docs/experiments/` | 2026-10-10 ~ 10-12 시험 결과 14개, 관련성 judge prompt 기록 2개, 미채택 시험 정리 기록 |
| `docs/cleanup/` | 정리 조사·실행 기록 4개(이 문서 포함)와 JSON 목록 4개 |

- `README.md`와 `docs/`의 Markdown 상대 링크 195개를 새 위치에 맞게 수정했다. 이동 전후 모두 깨진 링크는 0개다.
- `docs/superpowers/`의 계획·설계가 본문에 적은 시험 결과 경로를 새 위치로 수정했다.
- `docs/cleanup/`의 10-04 조사와 JSON 목록의 본문 경로는 당시 snapshot이므로 수정하지 않았다.
- [문서 목차](../README.md)를 추가했다.
- Notion 페이지의 링크는 확인하지 않았다.

## scripts 추적 규칙

`.gitignore`의 `scripts/*` 규칙과 예외 2개를 제거했다. 스크립트 35개가 이미 추적 대상이었고, 이 규칙은 새 파일만 숨겼다. 문서가 참조하는 `scripts/run_structural_all.sh`를 추적 대상에 추가했다.

## 결과 파일 이동

참조가 없는 낱개 파일 14개를 날짜 폴더로 옮겼다. `data/`는 Git 제외 대상이므로 이 기록이 유일한 이동 내역이다.

| 새 폴더 | 파일 | 이전 위치 |
|---|---|---|
| `data/eval/reports/fanout-modes-2026-09-17/` | `{criminal,ip}-{fanout,single}-{dense,hybrid,sparse}.json` 12개 | `data/eval/reports/` |
| `data/eval/reports/chat_pipeline/serving-ab-2026-09-28/` | `serving-ab-A.jsonl`, `serving-ab-B.jsonl` | `data/eval/reports/chat_pipeline/` |

이동 전에 `src/`, `scripts/`, `configs/`, `tests/`, `docs/`와 메모에서 이 파일 이름을 검색했다. 참조는 없었다.

## 삭제 내역

- `evals/`: 삭제된 패키지의 `.pyc` 파일 2개만 있었다.
- `tests/`, `scripts/`, `data/eval/` 아래의 `__pycache__` 폴더. 다시 생성되는 캐시다.

## 변경하지 않은 대상

- 초기 실행 폴더 9개(`formal-2026-09-27`, `formal-2026-09-28`, `formal-2026-09-28-k8`, `recheck-2026-09-28`, `everyday-2026-09-28`, `coverage-everyday`, `coverage-formal`, `article-coverage`, `article-heldout`). 이름은 비슷하지만 서로 다른 실행이며 문서가 각 폴더를 연결한다.
- 내용이 같은 파일은 약 60 MiB다(`data/eval` 전체 1.8 GiB). 대부분 실험별 frozen input 사본이므로 유지했다.

## 확인 범위

| 테스트 | 정리 전 | 정리 후 |
|---|---|---|
| `tests/legal_rag` (`.venv`) | 631 passed, 2 skipped | 631 passed, 2 skipped |
| evaluation import 대상 확인 (함수 내부 import 포함) | — | 이름 339개 모두 확인 |
| `scripts/*.py --help` | — | 33개 성공. conda 환경이 필요한 `probe_app_format.py`, `generate_sft_answers.py`는 실행하지 않음 |
| Flask 앱 테스트 (anaconda) | 133 passed | evaluation 분리 후 133 passed. `webapp/` 이동 후 134 passed (정적 파일 테스트 1개 추가) |

`webapp/` 이동 후 별도 폴더의 스크립트에서 앱을 import하여 루트 `.env`가 적용되고 DB·업로드·Chroma 경로가 저장소 루트를 가리키는 것을 확인했다. 실제 서버 실행과 브라우저 화면은 확인하지 않았다.
