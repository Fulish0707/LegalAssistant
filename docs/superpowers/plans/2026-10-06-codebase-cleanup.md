# 코드베이스 정리 계획 (2026-10-06)

**목표:** 쓰지 않는 코드를 지우고, 남는 코드를 짧고 명확하게 만든다. 동작은 바꾸지 않는다.

**방법:** 기준점을 커밋한다. 테스트를 전부 녹색으로 만든다. 기능 단위로 하나씩 지운다. 그다음에 긴 코드를 줄인다. 단계마다 테스트를 돌리고 커밋한다.

## 규칙

- 한 커밋은 한 가지 일만 한다. 삭제와 단축을 한 커밋에 섞지 않는다.
- 모든 커밋 뒤에 두 테스트 묶음이 실패 0건이어야 한다. 아니면 그 커밋을 되돌린다.
- "쓰지 않는다"는 근거 없이 지우지 않는다. 근거는 참조 검색 결과다: `src`, `scripts`, `tests`, 루트 `*.py`, `static`, `templates`, `docs`.
- 단축은 테스트가 덮는 코드에서만 한다. 덮지 않는 코드는 먼저 테스트를 쓰거나 건드리지 않는다.
- 줄이면서 동작이 달라질 수 있으면 멈추고 묻는다.
- `data/eval/reports/` 아래의 과거 실험 스크립트는 기록이다. 고치지 않는다.
- 원격 저장소로 push하지 않는다. 요청이 있을 때만 한다.

테스트 명령:

```
.venv/bin/python -m pytest tests/legal_rag -q -p no:cacheprovider
env APP_STATE_DB=$T/a.db RAG_PERSIST_DIR=$T/chroma APP_ENV=testing \
  LLM_PROVIDER=none EMBEDDING_PROVIDER=none LANGSMITH_TRACING=false \
  /home/layke/anaconda3/bin/python -m pytest tests --ignore=tests/legal_rag \
  --ignore=tests/test_ollama_connector_live.py -k 'not Live' -q -p no:cacheprovider
```

## 현재 크기 (정리 전)

| 영역 | Python 줄 수 |
|---|---|
| `src/legal_rag` | 10,705 |
| `tests` | 9,911 |
| `scripts` | 3,495 |
| 루트의 Flask 파일 10개 | 2,397 |
| `evals` | 491 |

60줄 이상 함수는 27개다. 가장 긴 것은 `judge_saved_answers` 159줄이다.

---

## 1단계. 기준점 커밋

작업 트리에 68개 변경이 있다. 두 세션의 작업이 섞여 있다. 묶음별로 커밋한다.

- [x] 1.1 이전 세션의 정리 작업 (10-04~05): 죽은 코드 8건, 캐시·빈 폴더, 문서 목차, `answer_replay.py`
- [x] 1.2 웹 챗을 `/v1/chat`으로 연결: `legal_backend.py`, `app.py`, 관련 테스트, README
- [x] 1.3 백엔드 수정: 검증기, 인용 점검, 프롬프트, 후속 질문 재작성, 답하면서 묻기, 검색 설정
- [x] 1.4 평가 도구: 새 채점기, 근거 고정 비교, 채점기 없는 측정값
- [x] 1.5 테스트가 `.env` 값에 의존하지 않게 한 수정
- [x] 1.6 이 계획 문서
- [x] 확인: `git status --short`가 비어 있다

`app.py`와 `README.md`에는 두 세션의 변경이 함께 있다. 1.2에 넣고 커밋 메시지에 적는다.

## 2단계. 테스트를 전부 녹색으로

`tests/legal_rag`의 실패 9건은 코드가 바뀐 뒤 테스트를 갱신하지 않은 것이다. 코드의 현재 동작이 의도이므로 테스트를 맞춘다.

- [x] 2.1 `test_imported_chunking.py` 5건: 청크 머리말에 `[역할: …]`이 붙고, 역할 경계를 넘는 병합이 없어졌다
- [x] 2.2 `test_rechunk_corpus.py` 1건: 전략 이름이 `structure_rechunk_v2_roles`가 됐다
- [x] 2.3 `test_relevance_judge.py` 2건: 채점 결과의 인용 위치 필드가 바뀌었다 (`span_index`)
- [x] 2.4 `test_pool_extension.py` 1건: 행에 `question_domain` 필드가 추가됐다
- [x] 각 건에서 코드와 테스트 중 어느 쪽이 맞는지 커밋 기록으로 확인한다. 코드가 틀린 경우는 멈추고 보고한다
- [x] 확인: 두 묶음 모두 실패 0건. 커밋 (`tests/legal_rag` 594 통과, Flask 175 통과)

2단계에서 확인한 것: 역할이 다른 절은 이제 합치지 않는다. 그래서 한 줄짜리 "주문" 절이 다시 독립 청크가 된다. 09월의 "한 줄 절은 이웃에 붙인다"는 수정과 반대 방향이다. 문서(`docs/answer-grader-v3-calibration.md`)에 의도로 적혀 있어 테스트를 맞췄다. 기존 색인에는 영향이 없고, 새로 청킹하는 코퍼스에만 적용된다.

## 3단계. 삭제 범위 확정 (2026-10-06 결정)

화면은 두 페이지다. 1번은 챗봇, 2번은 판례 검색과 사건 자료 검색이다. 둘 다 남긴다.

| # | 기능 | 결정 | 이유 |
|---|---|---|---|
| A | 통계 분석 (`/statistics/analyze`, `statistics_analysis.py`) | 삭제 | 부르는 화면이 없다 |
| B | 유사 대화 검색 (`/get_similar_chats`) | 삭제 | 부르는 화면이 없다 |
| C | 문서 분석 (`/process_document_ocr`, `llm_ocr.py`) | **유지** | 사건 자료의 파일 올리기가 쓴다 |
| D | 사건 자료 (`/knowledge_base/*`, `documents.html`, Chroma) | **유지** | 2번 페이지 |
| E | 서류 초안 패널 (`/update_statement`, "작성 중 문서") | 삭제 | 사용자 결정 |
| F | Flask 에이전트 (에이전트·도구·체크포인터·`evals/`) | 삭제 | 챗이 `/v1/chat`을 쓴 뒤로 닿는 경로가 없다. C·D가 쓰는 부분은 남긴다 |
| G | QLoRA 학습·평가 | **유지** | |
| H | 과거 채점기 (legacy, v2, v3) | **유지** | |
| I | 판례 검색 화면 (`cases.html`) | **유지** | 2번 페이지. 실제 검색 연결은 정리와 별개 작업 |

- [x] 3.1 삭제 범위 확정

정리 뒤에 따로 다룰 일: 사건 자료에 올린 문서는 지금 챗 답변에 쓰이지 않는다. 판례 검색 화면은 예시 데이터로 동작한다.

결정을 기다리는 것 (지우지 않았다):

- 왼쪽 메뉴의 "내용증명 만들기" 바로가기. 서류 초안 패널은 없어졌고, 이 버튼은 입력창에 "내용증명 초안을 만들어 주세요."를 채운다.
- `langchain_config.py`의 폴더 적재와 사용자별 검색기 (`ingest_source_folder`, `get_user_retriever`, `retrieve_for_user`, `load_rag_documents`, `get_documents_dir`, `get_document_source_folder`). 테스트는 있고 부르는 경로는 없다. 사건 자료를 챗에 다시 연결할 때 쓸 수 있다.

## 4단계. 죽은 코드 삭제

순서는 의존이 적은 것부터다. 항목마다: 참조 검색 → 삭제 → 테스트 → 커밋.

- [x] 4.1 A 통계 분석과 테스트
- [x] 4.2 B 유사 대화 검색
- [x] 4.3 E 서류 초안: 경로, 패널, `/get_initial_state`와 `/chat`의 초안 필드, 테스트. DB 테이블은 그대로 둔다
- [x] 4.4 F 에이전트: `langchain_config.py`의 에이전트·도구·미들웨어, `llm_prompt.py`의 에이전트 프롬프트, 체크포인터, `evals/`, 해당 테스트. 사건 자료와 문서 분석이 쓰는 함수는 남긴다
- [x] 4.5 의존성 정리: `requirements.txt`, `env.yml`, `.env.example`, README의 실행 안내
- [x] 4.6 화면 정리: 서버가 보내지 않는 `token_chunk` 분기를 지웠다. 세 화면과 한 턴을 브라우저로 확인했다. "내용증명 만들기" 바로가기는 결정을 기다린다
- [x] 4.7 `src/legal_rag`의 죽은 코드: 지운 것이 없다. 후보는 6개, 합쳐 22줄이다. `boundary_kappa`는 과거 실험 스크립트가 쓰고, `app_for_uvicorn`은 밖에서 부르는 진입점이고, 나머지 4개는 테스트가 쓴다
- [x] 4.8 `scripts/` 37개 분류: 32개는 문서·테스트·다른 코드가 쓴다. 나머지 5개 중 4개는 QLoRA 어댑터 평가(G, 유지)이고 1개(`run_structural_all.sh`)는 과거 실험 기록이 쓴다. 지울 것이 없다
- [x] 확인: `tests/legal_rag` 594 통과, Flask 108 통과. 실제 서버(qwen3:4b)로 `/v1/chat` 한 턴과 브라우저 웹 챗 두 턴이 답했다

## 5단계. 긴 코드 단축

대상은 60줄 이상 함수 27개와 중복이다. 동작은 바꾸지 않는다.

- [x] 5.1 대상 목록 확정: 4단계 뒤 60줄 이상 함수는 22개다
- [x] 5.2 함수마다 테스트가 실제로 실행하는 줄을 측정했다 (`coverage`). 84% 이상 10개, 70% 1개, 7% 이하 11개
- [x] 5.3 서비스 경로: `orchestrator.handle` 86 → 66줄 (생성·검증 반복을 `_generate_verified`로 분리), `process_document_ocr` 82 → 57줄
- [ ] 5.4 평가 경로: 하지 않았다. `judge_saved_answers`, `eval_chat_pipeline.verify`, `_report_versioned`은 테스트가 한 줄도 실행하지 않는다
- [ ] 5.5 적재 경로: 하지 않았다. `rechunk_corpus`, `import_processed`, `_parse_numbered`, `run_ingestion`은 읽어 보니 검사와 보고서를 한 번에 쓰는 일직선 코드다. 줄이면 줄이 옮겨질 뿐이다
- [ ] 5.6 중복 제거: 하지 않았다. JSONL 읽기 5개 중 같은 것은 2개(3줄)뿐이고, 나머지는 오류 처리가 서로 다르다. 부트스트랩은 한 곳에만 있다
- [x] 5.7 주석 정리: `app.py` 문서 올리기 경로와 `.env.example`의 에이전트 시절 설명
- [x] 각 커밋에 전후 줄 수를 적었다
- [x] 확인: `tests/legal_rag` 594 통과, Flask 108 통과

테스트가 덮지 않아 건드리지 않은 긴 함수 (11개): `judge_saved_answers` 159, `build_fanout_pool.main` 121, `probe_app_format.main` 112, `eval_chat_pipeline.verify` 104, `generate_sft_answers.main` 87, `compare_answers.main` 83, `sweep_fusion_weights.main` 78, `_report_versioned` 78, `build_sft_eval_contexts.main` 74, `judge_relevance.main` 61, `eval_golden_answers.generate` 60. 줄이려면 먼저 출력을 고정하는 테스트를 써야 한다.

학습 코드(`training/`의 143줄, 133줄 함수)는 G를 유지하는 동안 건드리지 않는다. 다시 학습해 볼 수단이 없으면 검증할 수 없기 때문이다.

## 6단계. 마무리

- [x] 6.1 실제 서버 확인: 백엔드(qwen3:4b)와 Flask를 띄우고 브라우저로 2턴 대화. 두 턴 모두 답했고 후속 질문이 앞 턴을 이어받았다
- [x] 6.2 평가 명령 확인: `freeze`, `verify --frozen`, `judge`, `report`를 4문항으로 돌렸다. 모두 끝났다
- [x] 6.3 문서 갱신: README, `.env.example`, `docs/integration-roadmap.md`에 제거 사실을 적었다. `requirements.txt`는 새로 풀어도 해결된다 (`uv pip compile`)
- [x] 6.4 결과 보고

## 결과

| 영역 | 정리 전 | 정리 후 |
|---|---|---|
| 루트의 Flask 파일 | 2,397줄 (10개) | 1,556줄 (9개) |
| `evals` | 491줄 | 0 |
| `tests` | 9,911줄 | 8,747줄 |
| `static/lc-chat.js`, `lc-shell.js` | 752줄 | 700줄 |
| `src/legal_rag` | 10,705줄 | 10,710줄 |
| `scripts` | 3,495줄 | 3,495줄 |

Flask 테스트는 175개에서 108개가 됐다. 줄어든 67개는 지운 기능(통계, 유사 대화, 서류 초안, 에이전트)의 테스트다. 남은 기능의 테스트는 지우지 않았다.

## 멈추고 묻는 경우

- 지우려는 것을 과거 실험 스크립트 외의 무언가가 참조한다
- 테스트를 고쳐야만 통과하는 단축
- 3단계 표에 없는 기능을 지워야 할 때
- 메모리 부족으로 실제 서버 확인을 못 할 때
