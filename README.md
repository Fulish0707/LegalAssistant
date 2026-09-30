# 한국어 법률 RAG 에이전트

한국어 법률 질의에 대해 관련 자료를 검색하고, 검색 근거를 바탕으로 답변을 생성하는 프로젝트입니다. Flask 채팅 화면과 별도 FastAPI RAG 백엔드로 구성되어 있으며, 검색과 답변 평가 결과를 함께 기록합니다.

## 1. 프로젝트 개요

웹 채팅은 사용자의 법률 질의에 필요한 근거를 검색 API에서 가져와 LangGraph 에이전트에 전달합니다. 별도 RAG 백엔드는 검색 API, 자체 답변 API, CLI를 제공하며 검색·생성·인용 검증을 처리합니다.

현재 local 프로필은 형사·민사·행정·지식재산권 네 코퍼스를 조회하도록 설정되어 있습니다. 원문 데이터와 가공 데이터, Qdrant 색인은 저장소에 포함되지 않으므로 실행 전에 별도로 준비해야 합니다.

이 프로젝트는 검색과 답변 생성을 실험하는 도구입니다. 결과를 법률 자문이나 법적 판단으로 사용해서는 안 됩니다.

## 2. 주요 기능

- **웹 채팅:** Flask와 LangGraph로 대화를 처리하고, SSE로 답변을 화면에 전달합니다. 대화 기록과 에이전트 체크포인트는 SQLite에 저장합니다.
- **법률 근거 검색:** Flask 앱은 법률 검색이 필요한 질문에 대해 답변 생성 전에 FastAPI의 POST /v1/search를 호출합니다. 짧은 인사말 등은 검색하지 않습니다.
- **하이브리드 검색:** bge-m3 임베딩을 이용한 Qdrant dense 검색과 BM25 검색 결과를 RRF로 결합하고, local 프로필에서는 cross-encoder로 재정렬합니다.
- **답변 API와 CLI:** POST /v1/chat과 scripts/chat.py는 질의 분석·재작성, 검색, 근거 구성, 답변 생성과 인용 검증을 수행합니다. 검증 결과에 따라 한 번 재생성하거나, 근거가 확인되지 않는 문장을 제거하고, 답변을 거절할 수 있습니다.
- **인용 확인:** Flask 채팅은 검색 근거에서 찾지 못한 조문이나 사건번호가 답변에 있으면 확인 안내를 표시합니다.

## 3. 기술 스택

| 구분 | 사용 기술 |
| --- | --- |
| 웹 채팅과 응답 전달 | Flask, LangGraph, Server-Sent Events (SSE) |
| 법률 RAG API | FastAPI, Uvicorn |
| 답변 모델 | Ollama, qwen3:4b |
| 임베딩 모델 | Ollama, bge-m3 |
| 검색과 재정렬 | Qdrant, BM25, RRF, BAAI/bge-reranker-v2-m3 |
| 대화 상태 | SQLite, LangGraph SQLite 체크포인터 |
| 설정 | Flask 환경 변수, configs/default.yaml, configs/local.yaml |

local 프로필은 CUDA와 float16으로 cross-encoder를 실행하고, 모델을 로컬 캐시에서 읽습니다. 이 프로필에는 호환되는 CUDA 환경과 캐시된 재정렬 모델이 필요합니다.

## 4. 시스템 구조

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

웹 채팅과 자체 답변 API는 검색 결과를 공유하지만, 답변을 만드는 흐름은 다릅니다. Flask 채팅은 /v1/search가 반환한 근거를 LangGraph 에이전트에 전달합니다. /v1/chat과 CLI는 RAG 백엔드 안에서 질의 처리부터 인용 검증까지 수행합니다.

## 5. 평가 결과

### 검색 설정 비교

860개 질의에서 기존 설정과 local 프로필 설정을 비교했습니다. Success@10은 각 질의의 정답 문서가 상위 10개 결과에 하나라도 포함됐는지를 나타냅니다.

| 평가 세트 | 질의 수 | 정답 문서 Success@10 (기존 → 변경 후) |
| --- | ---: | ---: |
| 전체 | 860 | 0.440 → 0.458 |
| administrative-pilot-200 | 200 | 0.805 → 0.825 |
| civil-pilot-200 | 200 | 0.125 → 0.155 |
| criminal-pilot-200 | 200 | 0.790 → 0.805 |
| crossdomain-60 | 60 | 0.533 → 0.567 |
| ip-pilot-200 | 200 | 0.010 → 0.015 |

전체 MRR@10은 0.319에서 0.330으로 바뀌었고, Success@10 변화량의 paired bootstrap 95% 신뢰구간은 +0.006에서 +0.033입니다. 평균 검색 시간은 질의당 3.60초에서 2.14초로 측정됐습니다.

이 비교에서는 cross-encoder 연산 정밀도(fp32 → fp16), RRF leg_floor(0 → 3), k_fused(50 → 100)를 함께 변경했습니다. 따라서 전체 변화만으로 개별 설정의 효과를 구분할 수 없습니다. 지식재산권 세트의 Success@10은 변경 후에도 0.015로 낮습니다. Success@10은 정답 문서 한 건을 기준으로 한 지표이며, 모든 관련 문서를 찾았는지를 뜻하는 완전한 Recall@10은 아닙니다.

측정 조건과 세부 결과는 [검색 진단 문서](docs/retrieval-diagnostics.md)에 정리되어 있습니다.

### 답변 참조 일치 평가

AIHub의 참조 답변과 저장된 답변 80개를 비교했습니다. qwen3:8b 평가자는 답변이 참조 답변의 핵심 결론을 포함하는지와 참조 내용에 반하는지를 판정했습니다.

| 측정 항목 | 결과 |
| --- | ---: |
| 핵심 결론 전체 일치 | 0.62 (95% Wilson 신뢰구간 0.52–0.72) |
| 결론을 일부 이상 포함 | 0.71 |
| 참조 내용과 모순 | 0.15 (95% Wilson 신뢰구간 0.09–0.24) |

별도로 31개 답변을 사람이 먼저 판정한 표본에서 평가자와 사람의 일치도는 정답 판정 Gwet AC1 0.698, 모순 판정 0.879였습니다. 이 평가는 각 질문의 단일 참조 답변과의 일치도를 확인한 것이며 법률적 정답률을 측정한 것은 아닙니다. 평가 방법과 분석은 [법률 RAG 개발 안내](docs/legal-rag-backend.md)의 답변 평가 절을 참고하십시오.

## 6. 주요 코드와 설정

- [app.py](app.py): Flask 라우트, 웹 채팅 SSE, 법률 근거 검색, 대화 상태 관리.
- [langchain_config.py](langchain_config.py): Flask 채팅의 LangGraph 에이전트와 검색 서비스 호출.
- [src/legal_rag/](src/legal_rag/): 검색, 답변 생성, 인용 검증, FastAPI 구현.
- [configs/local.yaml](configs/local.yaml): 로컬 모델, Qdrant, 검색 코퍼스, 재정렬 설정.
- [scripts/serve.py](scripts/serve.py), [scripts/chat.py](scripts/chat.py): RAG API와 CLI 실행.
- [scripts/eval_retrieval.py](scripts/eval_retrieval.py), [scripts/eval_chat_pipeline.py](scripts/eval_chat_pipeline.py): 검색 및 답변 파이프라인 평가.
- [docker-compose.yml](docker-compose.yml): Qdrant 서버 설정.

## 7. 실행 방법

### 7.1 법률 RAG 백엔드

백엔드는 Python 3.11과 uv를 사용합니다. 먼저 local 프로필에 필요한 원문·가공 자료·색인을 준비합니다. 수집과 색인 절차는 [법률 RAG 개발 안내](docs/legal-rag-backend.md)를 참고하십시오. Qdrant와 Ollama도 실행 중이어야 합니다.

~~~bash
docker compose up -d
uv python install 3.11
uv sync --extra rerank
ollama pull qwen3:4b
ollama pull bge-m3
uv run python scripts/serve.py --profile local --port 8000
~~~

local 프로필의 재정렬 모델은 CUDA 환경에서 실행되며 로컬 캐시에서 읽습니다. API 상태는 http://127.0.0.1:8000/health 에서 확인할 수 있습니다.

| 경로 | 용도 |
| --- | --- |
| GET /health | 서비스와 코퍼스 상태 확인 |
| POST /v1/search | 검색 결과와 출처 반환 |
| POST /v1/chat | 검색 근거를 사용한 답변 생성 |

### 7.2 Flask 웹 채팅

Flask 앱은 Python 3.13과 루트 requirements.txt를 사용하는 별도 환경에서 실행합니다. 프로젝트 루트의 .env에 백엔드 주소와 Ollama 설정을 지정합니다.

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

웹 화면은 http://localhost:5005 에서 열립니다. 배포 환경에서는 Flask 세션을 위한 SECRET_KEY도 설정해야 합니다.

## 8. 참고 문서

- [법률 RAG 개발 안내](docs/legal-rag-backend.md): 백엔드 설치, 자료 수집·색인, API와 답변 평가.
- [검색 진단 문서](docs/retrieval-diagnostics.md): 검색 설정 비교, 측정 방법과 결과 해석.
