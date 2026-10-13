# 요약 필드 우선 생성 시험 설계

## 목적

내부 JSON 스키마의 properties 순서에서 summary를 analysis보다 앞에 둔다. 요약을 먼저 작성하면 긴 분석이나 복구 호출이 줄어드는지 측정한다. Ollama가 이 순서를 실제 생성에 반영하는지도 별도로 확인한다.

## 고정 조건

- qwen3:4b Q4_K_M, f16 KV, Flash Attention ON, GPU37/37, parallel1, batch512.
- temperature0.2, 컨텍스트12288, 출력2048, think false. 질문·참고 답변·분석/검색 스냅샷·근거5개·프롬프트·복구·인용 검사·최종 정책 유지.
- 스키마8개 필드의 정의·기본값·required 유지. 현재 required는 analysis, summary, confidence의3개다. 반환 답변의 직렬화 순서는 기존과 같다.
- 원본 properties: legal_issues, analysis, summary, possible_actions, additional_information_needed, citations, uncertainty, confidence.
- 후보 properties: summary, legal_issues, analysis, possible_actions, additional_information_needed, citations, uncertainty, confidence. 첫 요청과 모든 후속 요청에 동일하게 적용한다.
- 문항[0,20,40,60,43,2], 예열26. seed=2026101270+round×100+원래 문항 위치; 후속 호출은1000×호출 위치 추가.
- ABBA: 1차 원본→후보, 2차 후보→원본; 2차 문항 역순. 측정24회, 제외 예열4회.

## 실행 경계

블록마다 포트11435의 새 소유 서버를 쓴다. 전용 모델 메타데이터·원본 블롭 읽기 전용 참조·download/prune 금지·GPU 잠금을 유지한다. 운영11434에 활성 모델이 있으면 중단한다. 운영 서버와 모델은 조작하지 않는다. 실제 commit 로그와 /api/ps에서 배치512·f16·FA·37/37·컨텍스트12288을 확인한다.

## 사전 판정

- 속도: 두 회차 모두 중앙값10% 이상 감소. p90·호출·전송 오류·거절·인용 재생성·제거 증가 없음.
- 입력: 첫 요청의 직렬화 차이는 properties 순서뿐이다. paired 첫 prompt count 동일, 예열 포함 모든 호출의 보고된 prompt count+2048≤12288. 잘림·context shifting 없음. 내부 토큰 동일성의 독립 증명은 아니다.
- 순서: 완료된 모든 후보 raw JSON 호출에서 summary가 analysis보다 먼저 있어야 한다. 키 누락·역전이면 실패. 잘린 호출은 별도 기록하고 완료 순서의 증거로 쓰지 않는다. 스키마 순서 변경이 실제 생성에 반영되지 않으면 후보 효과를 주장하지 않는다.
- 품질: 전체 질문·참고 답변·근거와12쌍의 최종8필드·완료 초안·검사를 새로 대조한다. 새 중요한 오류나 uncertain이 있으면 통과하지 않는다. 기존 답변 오류는 분리한다. 입력 전체가 이전 시험과 같으면 해시 검증 뒤 기존 전체 읽기를 재사용한다.
- 입력 평가, 호출별 디코딩 tokens/s, 문항별 출력 토큰, 복구·인용 재생성, cache slot 접두부 사용량을 분리한다. prompt count/duration을 새 입력 토큰 처리량으로 표시하지 않는다.
- 범위: 고정 분석/검색 이후 생성과 기존 최종 정책까지. 검색을 포함한 전체 요청이나 브라우저 첫 표시 시간은 아니다. 개발 문항6개, 독립 holdout·법률 전문가 인증 없음. 통과해도 확대 확인 후보이며 운영에 적용하지 않는다.

Ollama v0.16.2의 [Completion 구현](https://github.com/ollama/ollama/blob/v0.16.2/llm/server.go)은 JSON 스키마를 grammar로 변환한다. 실제 필드 순서는 이번 출력에서 확인한다.
