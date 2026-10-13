# 평가 문서 목차

검색 품질, 답변 참조 일치, 채점기 calibration은 서로 다른 평가다.
현재 실행 안내는 아래 문서에서 확인하고, 실험별 조건과 결과는 각 기록을 함께 읽는다.

## 현재 실행 안내

| 목적 | 안내 | 유지되는 구현 |
| --- | --- | --- |
| 검색 설정 비교·실패 분석 | [검색 진단](retrieval-diagnostics.md) | `src/legal_rag/evaluation/`의 retrieval 모듈과 `diagnosis/` |
| corpus별 관련성 채점 | [Corpus relevance prompts](corpus-relevance-prompts.md) | `relevance/relevance_prompts.py`, `judge_relevance.py`, `judge_agreement.py` |
| 원본/구조 청크 답변 비교 (저장 결과의 채점·보고만 실행 가능) | [Golden-answer comparison](golden-answer-comparison.md) | `eval_golden_answers.py`, `answers/answer_replay.py`, `answers/answer_reporting.py` |
| 저장 답변의 v2/v3 채점 비교 | [Answer grader calibration](answer-grader-v3-calibration.md) | `calibrate_golden_answers.py`, `answers/answer_rubrics.py`, `answers/answer_judge_runner.py` |

`src/legal_rag/evaluation/`은 세 폴더로 나뉜다. `relevance/`는 관련성 pool·채점·calibration, `answers/`는 답변 채점·replay·보고, `diagnosis/`는 검색 진단이다. 검색 지표와 dataset 모듈은 패키지 최상위에 있다.

관련성 채점의 corpus prompt 선택은 답변 생성 prompt를 변경하지 않는다.
Golden-answer 점수는 참고 답변과 제공 근거에 대한 일치를 측정한다.
법적 정확성은 별도의 사람 검토가 필요하다. v2/v3 점수는 rubric 변경을 명시해 읽는다.

## 학습·모델 평가

- [Local QLoRA](local-qlora.md): pilot 데이터, 환경과 adapter 평가.
- [Full QLoRA](full-qlora.md): 전체 학습 데이터와 실행·복구 안내.

`base`와 `base + adapter` 비교를 독립적으로 저장한 두 모델의 비교와 구분한다.
학습 데이터와 실행 규모가 다른 두 안내는 각각 유지한다.

## 과거 실험·결정 기록

전체 시험 기록과 정리 기록은 [문서 목차](README.md)에 있다.

- [구조 청크 시험](experiments/structural-civil.md): 작은 구조 청크의 색인·검색 비교. 채택하지 않았다. Qdrant 컬렉션은 2026-10-14에 삭제했다.
- [관련성 채점 prompt v3](experiments/relevance-judge-prompt-v3.md), [v3 검토](experiments/relevance-judge-v3-review.md): prompt 변경 근거. 현재 routing과 명령은 corpus prompt 안내를 따른다.
- `data/eval/reports/`: manifest, frozen input, 저장 답변·판단, 실험 당시 실행본. 현재 구현에서 실행하지 않는 snapshot도 재현 근거로 보관한다.
- `docs/superpowers/specs/`, `docs/superpowers/plans/`: 설계와 실행 이력. 현재 명령은 위 안내를 우선한다.
- [정리 조사](cleanup/codebase-cleanup-audit-2026-10-04.md), [정리 실행 기록](cleanup/codebase-cleanup-2026-10-05.md): 통합 후보와 실제 변경 내역.
- [정리 기록 2026-10-13](cleanup/codebase-cleanup-2026-10-13.md): 테스트 위치, 낱개 결과 파일과 evaluation 폴더 분리.
- [Dead code 정리](cleanup/dead-code-cleanup-2026-10-05.md): 제거한 코드와 dynamic caller·export 때문에 보존한 후보.

새 생성 실험에는 별도 `--out`을 사용한다. 코드나 평가 조건이 바뀐 상태에서
기존 manifest를 덮어써 재개하지 않는다. 저장 결과의 report 생성은 별도로 수행할 수 있다.
