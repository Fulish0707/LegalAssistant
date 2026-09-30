# 법률 RAG 에이전트

**Flask 채팅 화면과 별도 FastAPI 법률 검색 서비스로 구성된 프로젝트**

## 1. 프로젝트 개요 (Project Overview)

-   Flask 앱은 채팅 화면과 대화 기록을 제공하고, 법률 질의가 들어오면 별도 법률 RAG 서비스에서 근거를 검색합니다.
-   법률 RAG 서비스는 법령, 판례, 행정 해석 등 색인된 자료를 검색합니다. 검색 API와 자체 채팅 API, CLI를 제공합니다.
-   현재 local 프로필은 Ollama의 qwen3:4b와 bge-m3, Qdrant 서버, 네 개 코퍼스(criminal-v1, civil-v1, administrative-v1, ip-v1)를 사용합니다.
-   Flask 앱과 법률 RAG 서비스는 별도 Python 환경에서 실행합니다. 법률 RAG 서비스의 Python 요구 버전은 3.11입니다.

## 2. 주요 기능 (Main Features)

-   **웹 채팅:** Flask 앱이 사용자 입력을 받아 LangGraph 에이전트의 응답을 SSE로 전달합니다. 대화 기록과 에이전트 체크포인트는 SQLite에 저장합니다.
-   **법률 근거 검색:** Flask 앱은 일반적인 채팅 입력에 대해 답변 생성 전에 법률 RAG 서비스의 POST /v1/search를 호출합니다. 짧은 인사나 감사 표현은 검색 대상에서 제외합니다.
-   **하이브리드 검색:** 법률 RAG 서비스는 Qdrant의 dense 검색과 BM25 검색 결과를 RRF로 합치고, 현재 local 프로필에서는 cross-encoder로 후보를 재정렬합니다.
-   **법률 답변 API와 CLI:** POST /v1/chat과 scripts/chat.py는 검색 근거를 바탕으로 답변을 만들고 인용을 검증합니다. 근거에서 확인되지 않는 조문이나 사건번호가 있으면 답변을 한 번 다시 생성하고, 그래도 남으면 해당 문장을 지웁니다. 지우고 남는 내용이 없거나 답변 생성이 실패하면 오류 대신 거절 응답을 돌려줍니다. Flask 채팅은 이 채팅 API 대신 검색 API를 사용합니다.
-   **인용 확인 표시:** Flask 채팅 답변에 검색 근거에서 찾을 수 없는 조문이나 사건번호가 있으면, 답변 끝에 직접 확인하라는 안내를 붙입니다.
-   **코퍼스별 검색:** 현재 local 프로필은 형사, 민사, 행정, 지식재산권 코퍼스를 함께 조회합니다.

## 3. 기술 스택 (Tech Stack)

| 구분 | 현재 사용 구성 |
| --- | --- |
| 웹 채팅 | Flask, LangChain/LangGraph, SSE |
| 법률 RAG API | FastAPI, Uvicorn |
| 채팅·임베딩 모델 | Ollama qwen3:4b, bge-m3 |
| 검색·재정렬 | Qdrant 서버, BM25, RRF, BAAI/bge-reranker-v2-m3 |
| 대화 상태 | SQLite, LangGraph SQLite 체크포인터 |
| 설정 | Flask의 .env, 법률 RAG의 configs/default.yaml 및 configs/local.yaml |

local 프로필의 재정렬 설정은 CUDA와 float16을 사용하며, 모델을 로컬 캐시에서 읽습니다.

## 4. 시스템 아키텍처 및 작동 방식 (System Architecture & How it Works)

~~~mermaid
%%{init: {"flowchart": {"nodeSpacing": 20, "rankSpacing": 24, "diagramPadding": 6}}}%%
flowchart TB
    User["사용자"] --> Web["Flask 웹 채팅<br/>LangGraph 에이전트"]
    Caller["API · CLI"] --> ChatAPI["FastAPI<br/>/v1/chat"]

    Web -->|필요 시 근거 검색| SearchAPI["FastAPI<br/>/v1/search"]
    ChatAPI --> Orchestrator["RAG 오케스트레이터<br/>분석 · 생성 · 인용 검증"]

    SearchAPI --> Retrieval["하이브리드 검색<br/>Dense: bge-m3 · Qdrant<br/>Sparse: BM25"]
    Orchestrator --> Retrieval
    Retrieval --> Ranking["RRF 결합<br/>Cross-encoder 재정렬"]
    Ranking --> Evidence["검색 근거"]

    Evidence --> Web
    Evidence --> Orchestrator
    Web --> LLM["Ollama<br/>qwen3:4b"]
    Orchestrator --> LLM
    Web --> SQLite["SQLite<br/>대화 기록 · 체크포인트"]
    Web -->|SSE 응답| User
~~~

1.  Flask 앱은 법률 검색이 필요한 입력을 받으면 법률 RAG 서비스에서 근거를 먼저 가져옵니다.
2.  법률 RAG 서비스는 설정된 코퍼스에서 dense와 BM25 검색을 수행하고 결과를 결합·재정렬합니다.
3.  Flask 앱은 검색 근거를 에이전트에 전달하고, 생성된 답변과 출처를 채팅 화면으로 보냅니다.
4.  법률 RAG 서비스의 자체 채팅 API와 CLI는 검색, 답변 생성, 인용 검증을 한 서비스 안에서 처리합니다.

## 5. 검색 평가 (Evaluation)

법률 RAG의 검색 품질은 scripts/eval_retrieval.py와 data/eval의 평가 자료로 확인할 수 있습니다. 현재 검색 깊이, 결합 가중치, 재정렬 설정을 선택한 근거와 측정 범위는 [검색 진단 문서](docs/retrieval-diagnostics.md)에 기록되어 있습니다. 평가 수치는 해당 자료와 설정에서의 결과이며 실제 모든 질의의 정확도를 뜻하지 않습니다.

/v1/chat의 질의 재작성과 인용 검증은 scripts/eval_chat_pipeline.py로 측정합니다. 2026-09-28 측정에서 규칙·LLM 질의 재작성은 검색 정확도를 유의하게 바꾸지 못했고, 분석기의 문서 유형 필터는 정확도를 떨어뜨려 기본으로 꺼 두었습니다. 인용 검증을 고친 뒤 같은 80문항을 다시 실행했을 때 80문항 모두 답변했고, 최종 답변에 근거 없는 인용 표시는 없었습니다. 측정 방법과 결과는 [법률 RAG 개발 안내](docs/legal-rag-backend.md)의 Chat pipeline evaluation 절에 있습니다.

## 6. 주요 코드 파일 설명 (Key Code Files)

-   app.py: Flask 라우팅, 채팅 SSE, 법률 근거 사전 검색, 대화 상태 관리.
-   langchain_config.py: LangGraph 에이전트 구성과 법률 검색 서비스 호출.
-   citation_guard.py: Flask 답변에서 검색 근거에 없는 조문·사건번호를 찾아 확인 안내 표시.
-   static/lc-chat.js: 웹 채팅의 SSE 응답 처리.
-   src/legal_rag/: 법률 자료 검색, 답변 생성, 인용 검증, API 구현.
-   scripts/serve.py, scripts/chat.py: 법률 RAG API 서버와 대화형 CLI 실행.
-   scripts/eval_chat_pipeline.py: /v1/chat의 질의 재작성·인용 검증 측정.
-   configs/local.yaml: 현재 로컬 검색 코퍼스, Qdrant, 모델, 재정렬 설정.
-   docker-compose.yml: Qdrant 서버 설정.

## 7. 실행 방법 (How to Run)

### 7.1 법률 RAG 서비스 준비

법률 RAG 서비스는 Python 3.11 환경에서 실행합니다. 프로젝트 루트에서 Qdrant와 Ollama를 준비한 뒤 local 프로필로 API를 시작합니다.

~~~bash
docker compose up -d
uv python install 3.11
uv sync --extra rerank
ollama pull qwen3:4b
ollama pull bge-m3
uv run python scripts/serve.py --profile local --port 8000
~~~

Ollama 서버가 실행 중이어야 합니다. local 프로필의 재정렬에는 CUDA와 로컬에 캐시된 BAAI/bge-reranker-v2-m3 모델이 필요합니다. 이 프로필은 data/processed와 data/indexes의 네 코퍼스 및 Qdrant 색인을 사용합니다. 새 법률 자료의 수집·색인 절차는 [법률 RAG 개발 안내](docs/legal-rag-backend.md)를 참고하십시오.

2026-09-28 이전에 만든 색인은 법령 조각의 조 번호가 비어 있습니다. 다시 색인하지 않고 조 번호만 채우려면 다음을 실행합니다.

~~~bash
uv run python scripts/backfill_statute_articles.py --profile local --apply
~~~

서비스 상태는 http://127.0.0.1:8000/health에서 확인할 수 있습니다. 주요 경로는 다음과 같습니다.

| 경로 | 용도 |
| --- | --- |
| GET /health | 코퍼스와 모델, 준비 상태 확인 |
| POST /v1/search | 법률 근거 검색 |
| POST /v1/chat | 법률 근거를 사용한 답변 생성 |

### 7.2 Flask 채팅 실행

Flask 앱은 루트 requirements.txt를 사용하는 별도 Python 환경에서 실행합니다. 프로젝트 루트의 .env에 법률 RAG 서비스 주소를 설정합니다.

~~~dotenv
APP_ENV=local
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
LEGAL_RAG_URL=http://127.0.0.1:8000
~~~

~~~bash
pip install -r requirements.txt
python app.py
~~~

웹 화면은 http://localhost:5005에서 열립니다. 운영 환경에서는 Flask 세션용 SECRET_KEY도 설정해야 합니다.

## 8. 참고 문서 (References)

-   [법률 RAG 개발 안내](docs/legal-rag-backend.md): API, CLI, 설정 및 자료 수집 방법.
-   [검색 진단 문서](docs/retrieval-diagnostics.md): 검색 설정별 측정 결과와 해석.
