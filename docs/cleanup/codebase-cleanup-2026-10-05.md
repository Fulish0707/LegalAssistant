# 코드베이스 정리 실행 — 2026-10-05

[정리 조사](codebase-cleanup-audit-2026-10-04.md)의 직접 실행 의존성과
재생성 가능한 파일부터 정리했다. 원래 조사 인벤토리는 당시 snapshot으로 유지한다.

## 실행 코드 통합

- Gemini 생성·HTTP 재시도·답변 검증을 `src/legal_rag/evaluation/answer_replay.py`의 `AnswerReplay`로 추출했다.
- `scripts/eval_golden_answers.py`는 이 모듈을 직접 사용한다. 실험 폴더의 Python을 `runpy`로 실행하고 globals를 수정하던 경로를 제거했다.
- 설정과 요청 간격은 replay 객체에 명시적으로 전달한다. 두 arm의 요청은 같은 limiter를 사용한다.
- 기존 4회 HTTP 시도, quota/backoff, regeneration/strip/reject 정책, usage 기록과 결과 schema를 유지했다. Golden-answer 채점은 계속 전체 답변을 사용한다.
- 새 selection은 패키지 helper의 hash를 동결한다. 기존 manifest를 수정하지 않으므로 과거 generation 재개는 source hash 검사에서 거절될 수 있다. 새 생성은 별도 `--out`을 사용하며, 저장 답변 judge/report 명령은 유지한다.

## 삭제 내역

| 대상 | 처리 |
| --- | --- |
| 조사 목록의 cache 디렉터리 34개 | 삭제 당시 내용 합계 3,991,236 bytes (3.81 MiB) |
| `package-lock.json` | 루트 `package.json` 부재와 `packages: {}` 확인 후 삭제 |
| `data/data/` | 실제 파일·symlink가 없는 빈 중첩 디렉터리 삭제 |

전체 경로와 삭제 당시 크기는 [실제 삭제 목록](codebase-cleanup-actions-2026-10-05.json)에 있다.
삭제 전 cache 안에 Git tracked 파일이나 symlink가 없는지 확인했다.
후속 검사 시 일부 cache가 다시 생성되어 있었다. 다른 실행 과정에서 import가
계속되고 있으므로 반복 삭제하지 않았다. 삭제 크기는 지속적인 여유 공간 증가를 의미하지 않는다.

## 문서 정리

- [평가 문서 목차](../evaluation-index.md)를 추가하고 README에서 연결했다.
- 관련성 prompt 전문·검토는 과거 기록으로 표시하고 현재 corpus routing 안내로 연결했다.
- Full QLoRA 안내에서 공통 환경 설치 절을 연결했다. pilot/full 데이터와 adapter 평가 조건은 각 안내에 유지했다.
- Golden-answer 안내에 유지되는 helper와 새 출력 디렉터리 사용법을 반영했다.

## 확인 범위

- 변경한 Python 2개를 메모리에서 syntax compile했다. API 호출이나 답변 재생성은 수행하지 않았다.
- AST 비교에서 생성 함수는 설정·helper 주입을 치환한 뒤 과거 함수와 동일했다. HTTP retry/usage 처리 블록도 동일했다.
- 과거 Gemini replay와 두 golden/calibration 디렉터리의 파일 68개는 작업 전후 SHA256이 같았다.
- 현재 `scripts/`와 `src/legal_rag/`에서 과거 replay 경로·`runpy`·`__globals__` 참조가 검색되지 않았다.
- 테스트 suite와 live Gemini/Ollama 평가를 실행하지 않았다. 이 확인은 런타임 회귀 검증을 대신하지 않는다.

## 보관한 대상과 후속 통합

저장 답변, 판단·attempt 기록, frozen manifest, 실험 당시 실행본, 일반 로그,
API 원문, corpus/index/model 데이터와 worktree는 유지했다.
`gen-model-ab-2026-10-04`는 현재 `orig`와 `struct`에 각각 저장 행 80개가 있다.
행 수만으로 최종 채점·실험 종료를 판단하지 않고 로그와 결과를 보관했다.

Retrieval, prompt pilot, reranker 진단, law API prototype, model A/B,
failure/export runner의 추가 공통화는 조사 목록에 남아 있다.
이번 변경은 Gemini의 현재 실행 의존성 통합과 cache·빈 파일 정리다.
과거 75개 실행본 전체를 합치거나 제거하지 않았다. 추가 runner 통합 시
실험별 설정, snapshot과 source hash를 유지해야 한다.
