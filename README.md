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

### 전체 법령 코퍼스 추가

기존 코퍼스에는 국가법령정보센터가 나열한 현행 법령 5,621건 중 이름 기준 2,483건만 있었습니다(민사소송법, 상법, 주택임대차보호법, 헌법 등이 없음). 법령은 여러 조가 섞인 약 3,450자 창 하나가 한 조각이었습니다. Open API로 5,595건을 조(항) 단위 315,058개 조각의 별도 코퍼스(`statute-api-full-v1`)로 만들어 비교했습니다. 이 코퍼스는 기본 설정에 포함되어 있지 않습니다.

검색 비교는 held-out 질문 319개(형식 질문 169개, 같은 질문의 일상 표현 150개)로 했고, 값은 정답 조문이 상위 5개에 든 질문 수입니다.

| 질문 묶음 | 기존 → 추가 후 |
| --- | ---: |
| 형식 질문, 법령 정답 82개 | 43 → 78 |
| 일상 표현 질문, 법령 정답 73개 | 36 → 60 |
| 판례·유권해석 정답 질문 164개 | 밀려난 질문 0개 |

이 질문들의 법령은 이미 코퍼스에 있었으므로, 증가는 새 법령이 생긴 효과가 아니라 조 단위 조각의 효과입니다. 코퍼스에 없던 법령을 묻는 직접 쓴 질문 20개는 정답 조문이 상위 5개에 든 질문이 0개에서 14개가 됐습니다.

답변 비교는 70개 표본으로 로컬 qwen3:4b와 Gemini 두 모델에서 했습니다. 질문 분석 결과를 같게 고정해 코퍼스만 달리했고, qwen3:8b가 참고 답변과 핵심 결론이 맞는지 판정했습니다. 값은 "맞음"으로 판정된 답변 수입니다.

| 질문 묶음 | 로컬 (기존 → 추가 후) | Gemini (기존 → 추가 후) |
| --- | ---: | ---: |
| 검색이 새 조문을 찾은 법령 질문 30개 | 8 → 23 | 10 → 27 |
| 검색 변화가 없던 법령 질문 20개 | 13 → 17 | 17 → 19 |
| 판례·유권해석 정답 질문 20개 | 13 → 11 | 15 → 14 |

첫 행은 검색 결과가 좋아진 질문만 골랐으므로 효과가 가장 크게 나오는 집단이며, 전체 법령 질문의 평균 효과가 아닙니다. 둘째·셋째 행은 변화를 입증하지 못했고(95% 신뢰구간이 0을 포함), 20개로는 셋째 행의 하락을 배제하지 못합니다. 코퍼스에 아예 없던 유명 법령을 묻는 질문 20개는 답변 정확도에 차이가 없었습니다(한 사람이 눈가림으로 평가). 이 값은 개발용 표본에서 참조 답변과의 일치를 본 것이며 법률적 정답률이 아닙니다. 새 코퍼스의 Qdrant 메모리 비용은 약 1.3GiB로 측정됐습니다.

### 재정렬 입력 길이

재정렬 모델은 질의와 조각을 합쳐 512토큰까지만 읽습니다. 기존 코퍼스의 조각은 중앙값이 1,300~1,900토큰이어서 판결 조각과 법령 창의 93~100%가 잘리고, 조각의 앞 25~35%만 읽힙니다. 임베딩 모델은 조각 전체를 읽으며 잘림이 없었습니다. 같은 후보 100개를 512, 1024, 2048토큰으로 재정렬해 held-out 질문 319개에서 비교했습니다.

| 법령 정답 질문 (정답 조문이 상위 5개) | 512 | 1024 | 2048 |
| --- | ---: | ---: | ---: |
| 형식 질문 82개 | 41 | 60 | 71 |
| 일상 표현 질문 73개 | 36 | 45 | 52 |
| 질의당 재정렬 시간 | 1.6초 | 3.3초 | 7.4초 |

전체 법령 코퍼스를 더하면 512토큰에서도 81/82와 59/73이고, 더 길게 읽어도 거의 달라지지 않습니다(법령 조각이 약 122토큰이라 전부 읽힙니다). 판결·유권해석 질문은 길게 읽을수록 소폭 늘지만(2048에서 87개 중 +6/−1, 77개 중 +3/−0) 입증되지 않았습니다. 이는 정답 문서나 인용 조문이 상위 5개에 드는지로 본 검색 지표이며, 답변 모델에 전달되는 근거는 여전히 조각당 앞 1,500자입니다. 길이는 `rerank.max_length`(기본 512)로 바꿀 수 있습니다.

## 6. 주요 코드와 설정

- [app.py](app.py): Flask 라우트, 웹 채팅 SSE, 법률 근거 검색, 대화 상태 관리.
- [langchain_config.py](langchain_config.py): Flask 채팅의 LangGraph 에이전트와 검색 서비스 호출.
- [src/legal_rag/](src/legal_rag/): 검색, 답변 생성, 인용 검증, FastAPI 구현.
- [configs/local.yaml](configs/local.yaml): 로컬 모델, Qdrant, 검색 코퍼스, 재정렬 설정.
- [scripts/serve.py](scripts/serve.py), [scripts/chat.py](scripts/chat.py): RAG API와 CLI 실행.
- [scripts/eval_retrieval.py](scripts/eval_retrieval.py), [scripts/eval_chat_pipeline.py](scripts/eval_chat_pipeline.py): 검색 및 답변 파이프라인 평가.
- [src/legal_rag/ingestion/law_api_cli.py](src/legal_rag/ingestion/law_api_cli.py), [src/legal_rag/ingestion/law_api.py](src/legal_rag/ingestion/law_api.py): 국가법령정보센터 Open API에서 현행 법령을 받아 `statute_json_v1`로 변환합니다.
- `rerank.max_length` ([configs/default.yaml](configs/default.yaml)): 재정렬 모델이 한 조각에서 읽는 최대 토큰 수.
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

### 7.3 전체 법령 코퍼스 만들기 (선택)

~~~bash
uv run python -m legal_rag.ingestion.law_api_cli --out-dir data/api_laws/statute-api-full
uv run python scripts/ingest.py --raw-dir data/api_laws/statute-api-full --corpus-version statute-api-full-v1
uv run python scripts/build_index.py --corpus-version statute-api-full-v1
~~~

API 키는 open.law.go.kr에서 등록한 OC 값이며 .env의 `LAW_GO_KR_OC`에 넣습니다. 키는 등록한 서버 IP에서만 동작합니다. 키가 없으면 공개 시험 키로 동작하지만 대량 수집에는 맞지 않습니다. 5,600건 내려받기에 약 80분, 315,000개 조각의 임베딩에 수 시간이 걸립니다. 색인이 끝나면 `retrieval.corpora`에 `statute-api-full-v1`을 추가해 검색에 사용합니다. 법령은 받은 날의 현행본 하나뿐이어서 개정은 다시 받아야 반영됩니다.

호스트 메모리가 빠듯하면 earlyoom 같은 서비스가 가장 큰 프로세스인 Qdrant를 반복해서 종료할 수 있습니다. 이 환경에서도 확인됐으므로, 색인과 평가를 돌릴 때는 사용 가능한 메모리를 살펴야 합니다.

## 8. 참고 문서

- [법률 RAG 개발 안내](docs/legal-rag-backend.md): 백엔드 설치, 자료 수집·색인, API와 답변 평가.
- [검색 진단 문서](docs/retrieval-diagnostics.md): 검색 설정 비교, 측정 방법과 결과 해석.
