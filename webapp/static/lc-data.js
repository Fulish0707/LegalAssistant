/* 법률 상담 데스크 — 공통 데이터
   RAG가 연결되면 이 파일의 상수를 서버 응답으로 교체하세요. */

window.LC = window.LC || {};

LC.ACCENT = '#1f4b7a';

LC.SOURCES = []; /* 최근 답변의 실제 검색 근거 — /chat final_response.sources 로 채워짐 (lc-chat.js) */

LC.DOCS = []; /* /knowledge_base/status 로부터 채워짐 (lc-shell.js LC.syncDocs) */

LC.TRACE_FALLBACK = [
  "질의 임베딩 생성",
  "지식베이스 검색 → 상위 청크",
  "도구 선택: search_documents",
  "업로드 문서 조항 대조",
  "최종 답변 생성"
];

/* 진행 중 상담 — 여러 페이지가 공유. 상담 페이지에서는 현재 대화 제목으로 바뀝니다 */
LC.MATTERS = [
  {title:"전세 보증금 반환 지연", kind:"임대차", when:"오늘", live:true}
];

/* 탭 정의 — 페이지를 추가할 때 여기에 한 줄만 넣으면 모든 페이지의 탭에 반영됩니다 */
LC.TABS = [
  {label:"상담",      href:"chatbot.html"},
  {label:"판례 검색", href:"cases.html"},
  {label:"사건 자료", href:"documents.html"}
];
