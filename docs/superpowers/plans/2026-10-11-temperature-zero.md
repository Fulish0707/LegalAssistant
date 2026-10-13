# Temperature 0 시험 계획

**목표:** temperature 0.2와 0의 처리 시간·출력·복구 호출·상대 품질을 비교한다.

**실행:** 사용자의 계속 테스트 요청 범위에서 직접 실행한다. execute-protocol과 별도 worktree를 사용한다. 운영 소스·설정 변경과 자동 commit은 없다.

## T1 — 사전 입력 고정

- [x] root·이전 상태·외부 모델/키 파일 baseline을 보관하고 현재 파일을 worktree에 복사한다.
- [x] study.py의 Capture와 기록 wrapper에서 요청 temperature만 바꾼다. runtime.py는 양쪽 배치512·f16·전체GPU를 확인한다.
- [x] `PYTHONPATH=src .venv/bin/python data/eval/reports/chat_pipeline/temperature-zero-2026-10-11/study.py freeze`와 offline preflight를 실행해 7쌍 요청, 이전 입력 내용, source binding을 확인한다.

## T2 — 측정과 전체 검토

- [x] 같은 경로 `study.py run`으로 새 서버 4개에서 ABBA 24회와 제외 예열4회를 기록한다.
- [x] `study.py metrics`에서 두 중앙값·p90·입력 평가·디코딩 속도·출력량·실제 호출·최종 정책을 집계한다.
- [x] 12쌍의 최종 8필드와 완료 초안·검사를 새로 읽어 `AI reviewed (Codex)` 사유와 입력/답변 해시를 저장한다.
- [x] `verify_measurements.py`로 모든 실제 요청과 기존 파서·최종 정책을 새로 재생하고 `parse_cache.py`로 모든 접두부 사용 로그를 연결한다. 회귀 증거는 소스·설정·시험 파일248개가 동일할 때만 재사용한다.

## T3 — 보고와 보존 감사

- [x] `docs/temperature-zero-2026-10-11.md`에 수치·상대 오류·제한·채택 여부를 기록한다.
- [x] `audit_study.py`로 baseline, 이전 상태, 외부 모델/키, manifest, 재생, 검토 해시, 지표 재계산, 엔진과 서버 종료, root 추가3문서·링크를 확인한다.
- [x] GPU 정리 상태와 canonical T1–T3 완료를 검증한다. 시험 수행 완료와 후보 채택을 구분한다.

[설계](../specs/2026-10-11-temperature-zero-design.md)
