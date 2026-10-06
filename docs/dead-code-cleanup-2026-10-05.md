# Dead code 정리 — 2026-10-05

현재 Python 실행 코드 173개를 AST로 조사하고, source·tests·notebooks·문서·설정·
과거 실행본의 참조를 검색했다. 이름 검색에서 호출이 없다는 이유만으로 삭제하지 않고
framework hook, CLI dispatch, package export와 historical caller를 따로 확인했다.
Ignored 실행본도 추가 검색했다. 다른 worktree의 동일 정의는 현재 호출로 세지 않았으며
해당 worktree를 수정하지 않았다.

## 삭제한 코드

| 파일 | 삭제 항목 | 근거 |
| --- | --- | --- |
| `app.py` | `application_setup()` | 호출·worker hook 등록 없음. 실제 `__main__`의 `init_db()` 경로는 별도 존재 |
| `scripts/build_fanout_pool.py` | `json` import | 해당 모듈에서 사용하지 않음 |
| `scripts/judge_agreement.py` | `boundary_kappa` import | 현재 CLI에서 사용하지 않음. 원래 helper는 과거 snapshot이 import하므로 유지 |
| `src/legal_rag/llm/ollama.py` | `Message` import | 구현에서 사용하지 않고 해당 모듈을 통한 import caller도 발견되지 않음. 실제 모델 schema와 package export는 유지 |
| `src/legal_rag/chunking/korean_structure.py` | `_ITEM` regex | 정의 외 참조 없음. 사용 중인 section parser regex는 유지 |
| `src/legal_rag/schemas/documents.py` | `MetadataValue` alias | 정의 외 참조 없음. 실제 metadata field는 `JsonValue`를 사용 |
| `src/legal_rag/ingestion/normalize/base.py` | `require()` | import·호출·문서 사용 없음. 현재 normalizer의 검증 코드는 별도로 동작 |
| `src/legal_rag/evaluation/answer_judge.py` | `retry_feedback()`, `RETRY_INSTRUCTION`, `MAX_RETRY_OUTPUT_CHARS` | 이전 직접 인용 재시도 경로의 잔여 코드. 현재 runner는 ID selection 경로를 사용하며 이 세 항목을 참조하지 않음 |

합계 8개 파일에서 함수 3개, assignment 4개, import 3개를 제거했다.
실제 줄 번호와 작업 전후 source hash는 [변경 목록](dead-code-cleanup-actions-2026-10-05.json)에 있다.

## 삭제하지 않은 후보

| 후보 | 보존 근거 |
| --- | --- |
| `judge_answers._retry_manifest_hashes` | 과거 quick-review 실행본이 문자열 key로 가져와 사용 |
| `calibration.boundary_kappa` | `frozen_agreement_cli.py`가 import |
| `api.app.app_for_uvicorn` | 외부에서 호출 가능한 Uvicorn factory. 내부 참조 부재만으로 사용 중단을 판단할 수 없음 |
| `citations.StatuteRef`, `extract_case_refs` | 명시적 `__all__` export. 관련 caller도 존재 |
| `eval_golden_answers.prepare` | `globals()[phase]`로 CLI가 dispatch |
| `HybridRetriever.neighbor_chunks` | Orchestrator의 `getattr` lookup과 과거 retrieval runner가 사용 |
| `legal_evidence_middleware.wrap_model_call` | LangChain hook와 기존 tests가 사용 |
| `ingest_source_folder`, `get_user_retriever` | 호출 가능한 ingestion/retriever 기능 및 기존 isolation tests |
| `load_full_splits`, `pad_batch`, `balance_encoded` | 학습 notebook와 기존 tests에서 사용 |
| `InMemoryConversationStore`, `select_review_cases`, `known_schemas`, `is_empty`, `load_generation_examples` | 기존 tests 또는 package export에서 사용 |

AST 조사에서 확실하게 도달할 수 없는 statement나 사용되지 않는 local assignment는 발견하지 못했다.
Dynamic 호출과 외부 API 사용 전체를 정적 검사로 증명할 수는 없다.
실험 당시 snapshot·채점 기록과 현재 제공하는 entry point는 보관한다.

## 확인 결과

- 변경한 8개 파일은 선언한 삭제를 제외한 AST가 작업 전 source와 동일했다. 활성 함수 본문, schema field와 prompt 문자열의 변경은 없었다.
- 현재 Python 파일 173개를 메모리에서 syntax compile했다.
- 이전 Gemini replay, frozen agreement CLI와 두 golden/calibration 디렉터리의 파일 69개는 SHA256이 작업 전후 같았다.
- `git diff --check`가 통과했다.
- 테스트 추가·suite 실행·live model 호출은 수행하지 않았다. Syntax/source 검사는 런타임 회귀 검증을 대신하지 않는다.

이전 [파일·통합 정리](codebase-cleanup-2026-10-05.md)의 변경은 그대로 유지했다.
Git staging·commit과 worktree 제거는 수행하지 않았다.
