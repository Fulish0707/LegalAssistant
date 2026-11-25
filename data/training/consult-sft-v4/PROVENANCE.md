# consult-sft-v4 — 선별 이력

- 초안: `qwen3:8b`(사고 모드) 1,639건, `data/training/consult-sft-v2/drafts.jsonl`. 사람 법률 검토 없음.
- 선별: 강한 모델 검수자가 고르기만 하고 문장은 고치지 않음. 기준서 `review/GUIDELINES.md`(완성형 7기준), `review/GUIDELINES-template.md`(틀형 6기준).
  - 완성형 batch-1..9: claude-opus-5-5 단일 검수(보정 15건 후) → 27건.
  - 틀형 tbatch-1..13: 단일 검수. tbatch-1·2·5·6·7은 claude-opus-5-5, tbatch-3·4·8~10·11~13은 사용량 제한으로 넘어간 대체 모델 `qwen/qwen4-max:free`(감사 후 채택).
  - 틀형 tbatch-14 이후: 대체 검수자가 느슨했던 배치(39% 통과, 오도 주의점 통과)를 발견한 뒤 **독립 2중 검수, 둘 다 유지한 것만** 채택. 검수 모델은 배치마다 claude-opus-5-5 또는 qwen/qwen4-max:free.
  - 검수 통과 109건(완성형 27 + 틀형 82) 중 4건이 코드 인용 검증에서 탈락 → **105건**(학습 97, 검증 8).
- 정답 형식: 틀형은 v3(`--template-version 3`, 칸 사이 반복 축소). 검수 결과 파일: `review/full-review.jsonl`, `review/template-review.jsonl`(`second_review`, `second_reason` 필드).
- 수율: 틀형 후보 986건 중 82건(8%), 2중 검수 배치만 보면 약 4%.
