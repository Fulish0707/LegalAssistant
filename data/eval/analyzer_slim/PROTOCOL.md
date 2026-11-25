# 질의 분석 출력 축소 — 사전 등록

작성: 2026-11-23, 측정 전. 운영 모델 EXAONE 3.5 7.8B(`configs/local.yaml`).

## 바꾸는 것
질의 분석(QueryAnalyzer)은 턴마다 5~6초를 쓴다(출력 200~250토큰, `docs/spikes/006`). 현재 설정(`rag.analyzer_terms_in_search: false`, `rag.hard_filter: false`)에서 뒤 단계가 읽는 필드는 `domain`, `needs_more_info`, `missing_facts`, `statute_refs`, `case_refs`뿐이다. `subdomain`, `issues`, `key_facts`, `dates`, `keywords`, `document_types_needed`는 생성만 하고 쓰지 않는다.

- 축소안(slim): 모델에게 위 다섯 필드만 요구한다. 두 옵션 중 하나라도 켜지면 전체 스키마(full)를 쓴다.
- 결정적 참조 추출과 "사용자가 쓴 참조만 인정" 규칙은 그대로다.

## 측정
- 질문: 최종 테스트 274문항(`final-test-2026-10-06`의 formal 154 + everyday 120), 단일 턴. 이 세트는 이전에 답변 평가에 썼지만, 여기서는 품질 판정이 아니라 **동작 동등성**만 본다.
- 문항마다 full과 slim을 번갈아 실행한다(순서는 문항 번호 짝홀로 바꾼다). temperature 0, 같은 서버, 다른 모델이 GPU에 올라온 상태 유지(bge-m3 CPU).
- 기록: 단계 벽시계 시간, 출력 토큰, 스키마 실패(대체 분석으로 떨어짐), 각 필드.

## 판정 기준 (모두 만족해야 채택)
- G1 되묻기 결정(`ClarificationPolicy(allow=True).should_clarify`) 일치율 ≥ 97%, 되묻기 건수 차이 ≤ 3.
- G2 최종 `statute_refs`+`case_refs` 일치율 ≥ 97%.
- G3 답변에 넘기는 열린 질문 유무(`needs_more_info and missing_facts`) 일치율 ≥ 90%.
- G4 분석 단계 시간 중앙값 40% 이상 감소.
- G5 slim의 스키마 실패 수 ≤ full의 스키마 실패 수.

full끼리도 다를 수 있으므로 참고로 full을 50문항에 한 번 더 돌려 full-full 일치율을 기록한다. 기준은 바꾸지 않는다.

## 한계
단일 턴만 본다(이전 대화가 있는 후속 질문은 미측정). 답변 품질은 재지 않는다. 분석 필드가 답변 프롬프트에 들어가는 것은 `missing_facts`뿐이다.
