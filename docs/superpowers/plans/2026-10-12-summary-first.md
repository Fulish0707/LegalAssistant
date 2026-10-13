# 요약 필드 우선 생성 시험 계획

**목표:** JSON 스키마 속성 순서만 바꿔 지연·출력·복구 호출·상대 품질을 비교한다.

**실행:** 사용자의 계속 테스트 요청으로 진행한다. execute-protocol과 별도 worktree를 사용한다. 운영 소스·설정 변경과 자동 commit은 없다.

## T1 — 입력 고정

- [x] root437파일·이전13상태·외부9파일 baseline을 보관하고 현재 파일을 worktree에 복사한다.
- [x] wire_schema에서 summary를 앞으로 옮긴다. 모든8필드 정의와 기존 required3개를 유지한다.
- [x] study.py freeze와 preflight로7쌍의 직렬화 차이·기존 입력 전체·248개 회귀 증거 결합을 확인한다.

## T2 — 측정과 전체 검토

- [x] study.py run으로 새 서버4개의 ABBA24회와 제외 예열4회를 기록한다.
- [x] study.py metrics로 속도·입력·출력·호출을 계산하고 raw JSON 필드 순서를 확인한다.
- [x] 12쌍의 최종8필드·완료 초안·검사를 새로 읽어 AI reviewed (Codex) 사유와 해시를 저장한다.
- [x] verify_measurements.py로 순서까지 포함한 모든 실제 요청·기존 파서·최종 정책을 재생한다. parse_cache.py로 실제 호출을 로그에 연결한다. byte 동일248파일의 기존 회귀 결과만 재사용한다.

## T3 — 보고와 보존 감사

- [x] docs/summary-first-2026-10-12.md에 수치·필드 순서·상대 오류·채택 여부·제한을 기록한다.
- [x] audit_study.py로 baseline·이전 상태·외부 파일·manifest·재생·검토·지표·서버 종료·root 추가3문서를 확인한다.
- [x] GPU 정리와 canonical T1–T3 완료를 확인한다. 시험 완료와 후보 채택을 구분한다.

[설계](../specs/2026-10-12-summary-first-design.md)
