from dotenv import load_dotenv

# Load environment variables before importing project modules: library modules no
# longer call load_dotenv() themselves.
load_dotenv()

from flask import Flask, request, jsonify, render_template, Response, stream_with_context, g, session
from langchain_config import (
    ingest_text_to_vector_store,
    get_knowledge_base_status,
    clear_knowledge_base,
    search_knowledge_base,
    delete_knowledge_base_source,
    tracing_status,
)
import logging

from citation_guard import annotate_figures, annotate_unsupported, named_citations, unsupported_citations
import legal_backend
import sqlite3
import json
import re
import os
import secrets
import tempfile
import requests
from quota import (
    DAILY_AI_REQUEST_LIMIT,
    QuotaResult,
    client_ip_from_request,
    guest_ip_identity,
    reserve_daily_quota,
)


try:
    import llm_ocr
except ModuleNotFoundError:
    llm_ocr = None

from llm_prompt import prompt as main_chat_prompt_template
from llm_model import describe_configuration, get_chat_model

# Disable Flask's default request logging for cleaner output if not needed
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) # or logging.INFO for more details

app = Flask(__name__)
app_environment = os.getenv("APP_ENV", os.getenv("FLASK_ENV", "production")).lower()
is_local_environment = app_environment in {"development", "local", "testing"}
secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    if not is_local_environment:
        raise RuntimeError("SECRET_KEY must be configured outside development, local, or testing environments.")
    secret_key = "development-only-secret-key"

app.config.update(
    SECRET_KEY=secret_key,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=not is_local_environment,
    QUOTA_IP_SALT=os.getenv("QUOTA_IP_SALT"),
    DAILY_AI_REQUEST_LIMIT=int(os.getenv("DAILY_AI_REQUEST_LIMIT", str(DAILY_AI_REQUEST_LIMIT))),
)
app.config["QUOTA_IP_SALT"] = app.config["QUOTA_IP_SALT"] or secret_key


@app.before_request
def establish_anonymous_user():
    """Give every browser one opaque workspace id, carried in the signed cookie.

    This is the only identity in the app: SQL rows, vector filters and the
    legal_rag conversation id all key on it. Nothing submitted by the
    client -- query string, JSON, headers, or the old account cookie keys --
    can pick a different owner.
    """
    if "user_id" not in session:
        session["user_id"] = secrets.token_urlsafe(32)
    # Obsolete keys from the removed Google sign-in are dropped rather than
    # honoured: an old account cookie never recovers that account's records.
    for key in ("authenticated_user_id", "account_email", "auth_next", "oidc_nonce"):
        session.pop(key, None)

# Data folders stay at the repository root, whatever directory the app is started from:
# a relative path, from .env or the default, is resolved there and not in the working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE = os.getenv("APP_STATE_DB", "sqlite_db/app_state.db") # Define DB_FILE before use
if DB_FILE:
    DB_FILE = os.path.join(REPO_ROOT, DB_FILE)
db_dir = os.path.dirname(DB_FILE)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

def ensure_database_schema(conn):
    cursor = conn.cursor()
    chat_columns = {row[1] for row in cursor.execute("PRAGMA table_info(chat_history)")}
    if chat_columns and "user_id" not in chat_columns:
        cursor.execute("ALTER TABLE chat_history RENAME TO chat_history_legacy")
    cursor.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_user_id_id ON chat_history(user_id, id)")
    # One browser holds several consultations; rows written before 2026-10-13 belong to the first.
    chat_columns = {row[1] for row in cursor.execute("PRAGMA table_info(chat_history)")}
    if "conversation" not in chat_columns:
        cursor.execute("ALTER TABLE chat_history ADD COLUMN conversation INTEGER NOT NULL DEFAULT 0")
    # The source cards of an answer, as JSON, so a reopened consultation shows them again.
    if "sources" not in chat_columns:
        cursor.execute("ALTER TABLE chat_history ADD COLUMN sources TEXT")
    # The Google-era users table is no longer created. Existing installs keep
    # theirs untouched; nothing in the app reads it any more.
    # daily_ai_quota keeps the legacy 'user' kind so old rows stay valid; the app
    # itself only reserves 'guest_ip' entries now.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_ai_quota (
            identity_kind TEXT NOT NULL CHECK(identity_kind IN ('user', 'guest_ip')),
            identity_value TEXT NOT NULL,
            quota_day TEXT NOT NULL,
            request_count INTEGER NOT NULL DEFAULT 0 CHECK(request_count >= 0),
            PRIMARY KEY (identity_kind, identity_value, quota_day)
        )
    """)
    if cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history_legacy'").fetchone():
        cursor.execute("INSERT INTO chat_history (user_id, role, content, timestamp) SELECT 'legacy-unassigned', role, content, timestamp FROM chat_history_legacy")
        cursor.execute("DROP TABLE chat_history_legacy")
    conn.commit()

with app.app_context():
    conn = sqlite3.connect(DB_FILE)
    ensure_database_schema(conn)
    conn.close()


# --- Database Setup and Utility Functions ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_FILE, timeout=5)
        db.execute("PRAGMA busy_timeout = 5000")
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    print("Attempting to initialize database...")
    with app.app_context(): # Operations outside a request need an app context for get_db
        conn = get_db()
        ensure_database_schema(conn)
        print("Database initialized successfully.")
        # No need to close here, teardown_appcontext will handle it if get_db was called
        # For standalone init, explicitly closing is safer if not within app context management
        # but since we used `with app.app_context()`, `close_connection` will be triggered.


def get_chat_history_from_db(user_id, conversation=0):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT role, content FROM chat_history WHERE user_id=? AND conversation=? ORDER BY id ASC', (user_id, conversation))
    rows = c.fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]

def add_chat_message_to_db(user_id, role, content, conversation=0, sources=None):
    """`sources` are an answer's source cards; a message that is not an answer has none."""
    conn = get_db()
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (user_id, role, content, conversation, sources) VALUES (?, ?, ?, ?, ?)',
              (user_id, role, content, conversation, None if sources is None else json.dumps(sources, ensure_ascii=False)))
    conn.commit()

def get_latest_sources_from_db(user_id, conversation=0):
    """The source cards of the consultation's last answer, as the chat page showed them."""
    row = get_db().execute('SELECT sources FROM chat_history WHERE user_id=? AND conversation=? AND sources IS NOT NULL '
                           'ORDER BY id DESC LIMIT 1', (user_id, conversation)).fetchone()
    return json.loads(row["sources"]) if row else []

def get_conversations_from_db(user_id):
    """This browser's consultations that hold a question, newest first: {id, title, started}.

    The title is the first line of the first question; `started` is its UTC time.
    """
    rows = get_db().execute(
        'SELECT conversation, content, timestamp FROM chat_history WHERE id IN '
        "(SELECT MIN(id) FROM chat_history WHERE user_id=? AND role='user' GROUP BY conversation) "
        'ORDER BY conversation DESC', (user_id,)).fetchall()
    return [{"id": row["conversation"], "title": row["content"].strip().split("\n")[0][:80], "started": row["timestamp"]}
            for row in rows]

def clear_all_chat_history():
    """Forget every browser's consultations. The server does this each time it starts."""
    with app.app_context():
        conn = get_db()
        conn.execute('DELETE FROM chat_history')
        conn.commit()


def clear_chat_history_in_db(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM chat_history WHERE user_id=?', (user_id,))
    conn.commit()
    print("Chat history cleared for the active browser workspace.")


def reserve_ai_request():
    """Reserve one daily AI request for the direct peer's salted IP identity."""
    client_ip = client_ip_from_request(request)
    return reserve_daily_quota(
        get_db(),
        "guest_ip",
        guest_ip_identity(client_ip, app.config["QUOTA_IP_SALT"]),
        limit=app.config["DAILY_AI_REQUEST_LIMIT"],
    )


def quota_exceeded_response(result: QuotaResult):
    return jsonify({
        "error": "daily_ai_quota_exceeded",
        "limit": result.limit,
        "remaining": result.remaining,
    }), 429


# --- Messages answered without the legal service ---
_GREETING = re.compile(r"^(?:안녕|반가|ㅎㅇ|hi|hello)", re.IGNORECASE)
_THANKS = re.compile(r"^(?:고마|감사|수고|thanks?|thank you|bye)", re.IGNORECASE)
_ACKNOWLEDGED = re.compile(r"^(?:[ㅋㅎ]+|네|넵|예|응|ㅇㅇ|ㅇㅋ|오케이|ok|okay)?[\s,.!~]*"
                           r"(?:알겠(?:어요?|습니다)|알았(?:어요?|습니다)|이해했(?:어요?|습니다)|[ㅋㅎ]+)[\s.!~]*$", re.IGNORECASE)
# "고마워. 근데 변호사 없이도 할 수 있어?" thanks and then asks; the question must be searched.
_GOES_ON = re.compile(r"\?|근데|그런데|그럼|그러면|하지만")
# An answer to what the consultation asked ("계약서를 쓰셨나요?" - "네"). The legal service joins it
# to the question before it.
_YES_NO = re.compile(r"(?:네|넵|예|응|아니|아뇨|아니요|아니오)[.!]?")
GREETING_REPLY = "안녕하세요. 법률 관련 궁금한 점을 구체적으로 적어 주시면 법령과 판례를 찾아 안내해 드리겠습니다."
THANKS_REPLY = "도움이 되었다니 다행입니다. 더 궁금한 점이 있으면 이어서 적어 주세요."
ACKNOWLEDGED_REPLY = "네. 더 궁금한 점이 있으면 이어서 적어 주세요."

# Questions about the service itself. The legal service answers every message from retrieved
# passages, so "너는 뭘 할 수 있어?" came back as a legal analysis citing two unrelated sources
# (2026-10-13). A separate model call can tell such messages apart (message-kind-2026-10-13), but
# it is not wired in; these patterns cover the common ones. Each needs the service as its
# subject, so "소송 비용은 누가 내?" or "판결문은 믿어도 되는 증거야?" still go to the search.
# Checked 2026-10-14: none of 980 legal questions matches (everyday-150, the four pilot-200 sets,
# 30 general questions); with the greetings below, 15 of 18 hand-written conversational messages
# get a fixed reply. The three that do not only state a feeling ("오늘 너무 힘들다").
_SERVICE = "너|넌|당신|여기|이 ?서비스|챗봇|AI"
_SERVICE_REPLIES = [
    (re.compile(rf"(?:{_SERVICE}|이거).{{0,8}}(?:뭐|뭘|무엇|무슨|어떤).{{0,8}}(?:할 수|해 ?줄|해 ?주|하는 거|하는 곳|가능)"
                r"|^(?:너|넌|당신)은?는? ?(?:누구|뭐야|뭐니|AI|인공지능|사람|로봇)"
                r"|(?:이거|여기|이 ?서비스).{0,6}어떻게 (?:쓰|사용|이용)"
                r"|변호사(?:랑|와|하고) ?(?:뭐가|무엇이|어떻게) ?(?:달라|다른|다르)"),
     "저는 법률 상담을 돕는 AI입니다. 겪고 계신 상황이나 궁금한 점을 적어 주시면 관련 법령과 판례를 찾아 근거와 함께 "
     "설명해 드립니다. 사건 자료 탭에 올리신 계약서 같은 문서도 함께 참고합니다. 변호사가 아니므로 답변은 1차 정보로 "
     "보시고, 중요한 결정 전에는 근거 원문과 전문가의 확인을 받으시기 바랍니다."),
    (re.compile(rf"(?:{_SERVICE}|답변|네 말).{{0,10}}(?:믿어도|믿을 수|신뢰|정확한 거|맞는 거)|변호사가 (?:확인|검토|작성|쓴)"),
     "답변은 AI가 검색된 법령과 판례를 바탕으로 작성하고, 인용한 조문과 사건번호가 실제 근거에 있는지 확인한 뒤 보여 "
     "드립니다. 그래도 틀릴 수 있고, 변호사가 검토한 답변이 아닙니다. 오른쪽의 근거 원문을 직접 확인하시고, 중요한 결정 "
     "전에는 전문가와 상담하시기 바랍니다."),
    (re.compile(r"(?:대화|상담|채팅|내가 쓴).{0,8}(?:내용|기록).{0,12}(?:저장|누가|남아|지워|삭제|볼 수)"),
     "상담 내용은 이 브라우저에서 다시 볼 수 있도록 서버에 저장되며, 다른 브라우저에서는 열 수 없습니다. 서비스를 "
     "운영하는 쪽은 서버에 저장된 기록에 접근할 수 있습니다. 서버가 다시 시작되면 상담 기록은 지워집니다."),
    (re.compile(r"(?:이거|여기|이 ?서비스).{0,6}(?:무료|공짜|유료|돈 ?(?:내|들|받))|(?:이용|사용) ?(?:요금|료)"),
     "지금은 무료로 이용하실 수 있습니다. 하루에 보낼 수 있는 질문 수에는 제한이 있습니다."),
    (re.compile(r"사람.{0,8}(?:얘기|이야기|상담|연결|통화).{0,6}(?:싶|해 ?줘|할 수|가능|돼)"),
     "지금은 상담원이나 변호사와 직접 연결해 드리는 기능이 없습니다. 상황을 적어 주시면 관련 법령과 판례를 찾아 "
     "안내해 드리겠습니다."),
]
SMALL_TALK_CHARS, SERVICE_QUESTION_CHARS = 20, 30


def _fixed_reply(text, in_conversation=False):
    """The reply to a message that needs no legal search, or None when it does.

    `in_conversation`: the consultation already holds an answer, so a bare 네 answers its question.
    """
    stripped = (text or "").strip()
    if len(stripped) < 4:
        if in_conversation:
            return None if _YES_NO.fullmatch(stripped) else ACKNOWLEDGED_REPLY
        return GREETING_REPLY
    if len(stripped) <= SERVICE_QUESTION_CHARS:
        asked = next((reply for pattern, reply in _SERVICE_REPLIES if pattern.search(stripped)), None)
        if asked:
            return asked
    if len(stripped) <= SMALL_TALK_CHARS and not _GOES_ON.search(stripped):
        for pattern, reply in ((_GREETING, GREETING_REPLY), (_THANKS, THANKS_REPLY), (_ACKNOWLEDGED, ACKNOWLEDGED_REPLY)):
            if pattern.match(stripped):
                return reply
    return None


def _sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


# --- Flask Routes ---

LEGAL_BACKEND_UNSET = "법률 답변 서비스가 설정되지 않았습니다. LEGAL_RAG_URL을 설정한 뒤 다시 시작하세요."
LEGAL_BACKEND_DOWN = "법률 답변 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요."
# legal_rag refuses a longer message (ChatRequest in src/legal_rag/api/models.py).
MESSAGE_MAX_CHARS = 4000
MESSAGE_TOO_LONG = "질문이 너무 깁니다. 4,000자 이하로 줄여 주세요. 계약서 같은 긴 자료는 사건 자료 탭에 올리면 답변에 쓰입니다."


def _conversation():
    """The number of the consultation this browser has open."""
    return session.get("conversation_epoch", 0)


# Changes with every start of this server. Consultations are forgotten at a start, and a browser
# keeps its consultation number in its cookie, so the number alone would reach turns legal_rag
# still remembers from before.
SERVER_START = secrets.token_hex(4)


def _legal_conversation_id():
    """The browser's workspace id, this server start, and the number of the open consultation.

    legal_rag keeps recent turns per conversation id and has no call to forget them,
    so each consultation, and a cleared chat, continues under an id of its own.
    """
    return f"{session['user_id']}-{SERVER_START}-{_conversation()}"


def _case_key():
    return f"{SERVER_START}-{_conversation()}"


def _open_new_conversation():
    """Open a number this browser has never used: legal_rag still remembers the turns of deleted ones."""
    held = [conversation["id"] for conversation in get_conversations_from_db(session["user_id"])]
    session["conversation_epoch"] = session["conversation_top"] = \
        max([session.get("conversation_top", 0), _conversation(), *held]) + 1


def _pinned_case():
    """The case attached to the open consultation, or None: {chunk_id, title}."""
    return (session.get("cases") or {}).get(_case_key())


def _chat_state():
    """What the chat page shows: the open consultation's messages, the source cards of its last
    answer, and the list of all consultations."""
    user_id = session["user_id"]
    return {"chat_history": get_chat_history_from_db(user_id, _conversation()),
            "sources": get_latest_sources_from_db(user_id, _conversation()),
            "case": _pinned_case(),
            "conversation": _conversation(),
            "conversations": get_conversations_from_db(user_id)}


# Passages of the user's files sent with a turn. Three of 1,000 characters add about 2,700
# tokens to the answer prompt: up to 9,250 on the longest-evidence questions, which with 2,048
# output tokens fits the 12,288 context (2026-10-06). /v1/chat accepts at most three.
CASE_DOCUMENT_LIMIT = min(int(os.getenv("CASE_DOCUMENT_LIMIT", "3")), 3)
CASE_DOCUMENT_CHARS = int(os.getenv("CASE_DOCUMENT_CHARS", "1000"))
# One lease against eight questions (bge-m3, 2026-10-06): four about the lease scored 0.52,
# 0.38, 0.34 and 0.17 (a short follow-up), four about other matters 0.21, 0.21, 0.14, 0.12.
CASE_DOCUMENT_MIN_SCORE = float(os.getenv("CASE_DOCUMENT_MIN_SCORE", "0.25"))
# A message this short (characters without spaces) is taken for a follow-up and searched
# together with the question before it. One lease, 59 pairs (bge-m3, 2026-10-06): follow-ups
# of 9 to 11 characters reached the floor on 6 of 15 alone and 15 of 15 joined, and after a
# question on another matter on 8 of 20 alone and 0 of 20 joined. Full questions of 15
# characters or more are searched alone: joined, a new topic kept the lease on 10 of 12.
# data/eval/reports/chat_pipeline/context-budget-2026-10-06/followup-scores.json.
FOLLOW_UP_CHARS = 12


def _document_query(user_input):
    if len("".join(user_input.split())) > FOLLOW_UP_CHARS:
        return user_input
    asked = [turn["content"] for turn in get_chat_history_from_db(session["user_id"], _conversation()) if turn["role"] == "user"]
    if asked and asked[-1] == user_input:  # POST /chat stored this message before the stream began
        asked.pop()
    return f"{asked[-1]}\n{user_input}" if asked else user_input


def _case_documents(user_input):
    """Passages of this browser's uploaded files that are closest to the message."""
    found = search_knowledge_base(session["user_id"], _document_query(user_input), k=CASE_DOCUMENT_LIMIT)
    return [{"name": hit["source"], "text": hit["quote"][:CASE_DOCUMENT_CHARS], "score": hit["score"]}
            for hit in found.get("results", []) if hit.get("quote") and hit["score"] >= CASE_DOCUMENT_MIN_SCORE]


def _generate_legal_response_stream(user_input):
    """Stream a chat turn as SSE events; legal_rag's /v1/chat writes the answer.

    The service retrieves, generates and verifies, so the reply arrives whole and
    there are no token chunks. There is no fallback to a local model: an answer that
    skipped verification must not look like one that passed it.
    """
    try:
        legal_rag_url = os.getenv("LEGAL_RAG_URL", "").strip()
        if not legal_rag_url:
            yield _sse({'type': 'error', 'content': LEGAL_BACKEND_UNSET})
            return

        if len(user_input) > MESSAGE_MAX_CHARS:
            yield _sse({'type': 'error', 'content': MESSAGE_TOO_LONG})
            return

        # POST /chat stored this message before the stream began, so an earlier turn makes more than one.
        fixed = _fixed_reply(user_input, len(get_chat_history_from_db(session["user_id"], _conversation())) > 1)
        if fixed is None:
            # A canned reply calls no model, so only a real turn uses the daily quota.
            quota_result = reserve_ai_request()
            if not quota_result.allowed:
                yield _sse({'type': 'error', 'content': 'Daily AI request quota exceeded.', 'error': 'daily_ai_quota_exceeded', 'limit': quota_result.limit})
                return
            yield _sse({'type': 'thought', 'content': '법령·판례 검색 및 답변 작성 중'})
            documents = _case_documents(user_input)
            if documents:
                yield _sse({'type': 'thought', 'content': f'사건 자료 {len(documents)}건 참고'})
            case = _pinned_case()
            if case:
                yield _sse({'type': 'thought', 'content': f"붙인 판례 참고: {case['title']}"})
            body = legal_backend.ask(legal_rag_url, user_input, _legal_conversation_id(), documents,
                                     pinned=[case["chunk_id"]] if case else ())
            chat_response, sources = legal_backend.render(body, documents)
            questions = legal_backend.questions(body)
            yield _sse({'type': 'thought', 'content': f'근거 {len(sources)}건 확인'})
            # Second check on the shown text, whatever the service's verifier decided. It reads
            # the full evidence; the source cards carry only an excerpt. Of the user's documents it
            # reads what they cite by name; their own clause numbers vouch only for a bare 제5조.
            # What the last answers cited (prior_refs) may be repeated, as the service allows.
            document_citations = [citation for d in documents for citation in named_citations(d["text"])]
            evidence_texts = [user_input, *document_citations, *body.get("prior_refs", [])] + legal_backend.evidence_texts(body)
            missing = unsupported_citations(chat_response, evidence_texts, own_clauses=[d["text"] for d in documents])
            chat_response = annotate_figures(annotate_unsupported(chat_response, missing))
        else:
            chat_response, sources, questions = fixed, [], []

        add_chat_message_to_db(session["user_id"], "assistant", chat_response, _conversation(), sources)
        yield _sse({
            "type": "final_response",
            "chat_response": chat_response,
            "sources": sources,
            "questions": questions,
        })

    except legal_backend.LegalBackendUnavailable:
        logging.exception("legal_rag /v1/chat is unavailable")
        add_chat_message_to_db(session["user_id"], "assistant", LEGAL_BACKEND_DOWN, _conversation())
        yield _sse({'type': 'error', 'content': LEGAL_BACKEND_DOWN})
    except Exception:
        logging.exception("Error during AI processing stream (_generate_legal_response_stream)")
        error_response_content = "An error occurred while processing your request. Please check server logs."
        add_chat_message_to_db(session["user_id"], "assistant", error_response_content, _conversation())
        yield _sse({'type': 'error', 'content': error_response_content})
    finally:
        yield _sse({'type': 'stream_end'})


@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/chatbot.html') 
def chatbot_page_explicit():
    return render_template('chatbot.html')

@app.route('/documents.html')
def documents_page():
    return render_template('documents.html')

@app.route('/cases.html')
def cases_page():
    return render_template('cases.html')

LLM_PROBE_TIMEOUT_SECONDS = 1.0


def probe_llm_reachable():
    """Is the local LLM server actually answering?

    get_chat_model() builds a ChatOllama unconditionally -- no key, no
    connection -- so llm_configured reports True with Ollama stopped. This is the
    only field that can tell the difference. Opt-in via ?probe=llm, short timeout,
    never raises. Returns None when reachability is not something we can answer:
    the probe was not requested, or the provider is a hosted API we do not ping.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider != "ollama":
        return None
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=LLM_PROBE_TIMEOUT_SECONDS)
        return response.ok
    except Exception:
        return False


@app.route('/health', methods=['GET'])
def health():
    llm_available = get_chat_model() is not None
    payload = {
        "status": "ok",
        "llm_configured": llm_available,
        "llm_provider": None,
        "embedding_provider": None,
        "collection": None,
        "llm_reachable": None,
        # Whether prompts and retrieved document text are being shipped to
        # LangSmith is an operational fact worth being able to check.
        "tracing": tracing_status(),
    }

    # An endpoint whose job is to make misconfiguration visible must not 500 on
    # misconfiguration: describe_configuration() raises on an unknown provider.
    try:
        payload.update(describe_configuration())
    except Exception as error:
        payload["configuration_error"] = str(error)

    if request.args.get("probe") == "llm":
        payload["llm_reachable"] = probe_llm_reachable()

    return jsonify(payload)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        user_input = data.get('message')

        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        if len(user_input) > MESSAGE_MAX_CHARS:
            return jsonify({"error": MESSAGE_TOO_LONG}), 400

        add_chat_message_to_db(session["user_id"], "user", user_input, _conversation())
        # The POST request now just acknowledges receipt.
        # The actual processing and streaming will be triggered by the frontend's GET request.
        return jsonify({"status": "message_received_ok", "message": "Proceed to connect to SSE for response."})

    elif request.method == 'GET':
        # For SSE, data comes from query parameters
        user_input = request.args.get('message')

        if not user_input:
            # SSE requires a specific format for errors if connection is established
            def error_stream():
                yield f"data: {json.dumps({'type': 'error', 'content': 'No message provided in SSE request.'})}\n\n"
                yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
            return Response(stream_with_context(error_stream()), mimetype='text/event-stream')

        # Note: The user message for this GET request is NOT added to DB here,
        # as the POST request should have already handled it.
        # If GET is called without a preceding POST, this could lead to inconsistencies
        # in chat history if not handled carefully. The current frontend flow (POST then GET) is assumed.

        return Response(stream_with_context(_generate_legal_response_stream(user_input)), mimetype='text/event-stream')

    # Fallback to satisfy type checkers: ensure the route always returns a valid response
    return jsonify({"error": "Invalid request method."}), 405

def _from_legal_backend(lookup, asked, missing):
    """Answer a page's read of the legal corpus: `lookup(url, asked)` as JSON, or the error to show."""
    legal_rag_url = os.getenv("LEGAL_RAG_URL", "").strip()
    asked = (asked or '').strip()
    if not asked:
        return jsonify({"error": "Nothing was asked for."}), 400
    if not legal_rag_url:
        return jsonify({"error": LEGAL_BACKEND_UNSET}), 503
    try:
        found = lookup(legal_rag_url, asked)
    except legal_backend.LegalBackendUnavailable as error:
        app.logger.error(f"Legal corpus lookup failed: {error}")
        return jsonify({"error": "법률 답변 서비스에 연결할 수 없습니다."}), 502
    if found is None:
        return jsonify({"error": missing}), 404
    return jsonify(found)


@app.route('/legal_source', methods=['GET'])
def legal_source():
    """The whole document behind a source card, with the place the answer cited."""
    return _from_legal_backend(legal_backend.document, request.args.get('chunk_id'), "이 근거의 전문을 찾을 수 없습니다.")


@app.route('/law', methods=['GET'])
def law():
    """One law's articles, each with its heading and text, for showing the article a filter names."""
    return _from_legal_backend(legal_backend.law, request.args.get('name'), "이 법령을 찾을 수 없습니다.")


@app.route('/case_search', methods=['POST'])
def case_search():
    """Cases closest to what the user described, each with the statute articles it cites."""
    query = (request.get_json(silent=True) or {}).get('query')
    return _from_legal_backend(lambda url, asked: {"results": legal_backend.search_cases(url, asked)},
                               query if isinstance(query, str) else '', "")


@app.route('/get_initial_state', methods=['GET'])
def get_initial_state():
    return jsonify(_chat_state())


@app.route('/conversations/new', methods=['POST'])
def new_conversation():
    """Open an empty consultation; the earlier ones stay in the list."""
    # An open consultation that is still empty is kept.
    if _conversation() in [conversation["id"] for conversation in get_conversations_from_db(session["user_id"])]:
        _open_new_conversation()
    return jsonify(_chat_state())


# A browser keeps this many attached cases in its cookie; each is about 300 bytes.
PINNED_CASES_KEPT = 8


@app.route('/conversations/case', methods=['POST'])
def attach_case():
    """Attach a case to the open consultation, or detach it with no chunk_id.

    The service then puts that passage of the case first in the evidence of every turn of the
    consultation; without it, a question that names a case is answered from whatever a search finds.
    """
    data = request.get_json(silent=True) or {}
    chunk_id, title = data.get("chunk_id"), data.get("title")
    # Entries of earlier server starts point at consultations that no longer exist.
    cases = {key: value for key, value in (session.get("cases") or {}).items() if key.startswith(SERVER_START + "-")}
    cases.pop(_case_key(), None)
    if isinstance(chunk_id, str) and chunk_id.strip():
        cases[_case_key()] = {"chunk_id": chunk_id.strip()[:512], "title": str(title or "").strip()[:200]}
    session["cases"] = dict(list(cases.items())[-PINNED_CASES_KEPT:])
    return jsonify(_chat_state())


@app.route('/conversations/delete', methods=['POST'])
def delete_conversation():
    """Delete the open consultation with its messages, sources and attached case, and open an empty one."""
    user_id, conversation = session["user_id"], _conversation()
    conn = get_db()
    conn.execute('DELETE FROM chat_history WHERE user_id=? AND conversation=?', (user_id, conversation))
    conn.commit()
    session["cases"] = {key: value for key, value in (session.get("cases") or {}).items() if key != _case_key()}
    _open_new_conversation()
    return jsonify(_chat_state())


@app.route('/conversations/select', methods=['POST'])
def select_conversation():
    """Reopen one of this browser's consultations."""
    chosen = (request.get_json(silent=True) or {}).get("id")
    if chosen not in [conversation["id"] for conversation in get_conversations_from_db(session["user_id"])]:
        return jsonify({"error": "No such consultation."}), 404
    session["conversation_epoch"] = chosen
    return jsonify(_chat_state())


@app.route('/clear_chat_history_endpoint', methods=['POST']) # Added specific endpoint name
def clear_chat_history_endpoint():
    try:
        user_id = session["user_id"]
        _open_new_conversation()
        clear_chat_history_in_db(user_id)
        return jsonify({"message": "Chat history cleared successfully."}), 200
    except Exception as e:
        logging.error(f"Error clearing chat history: {e}")
        return jsonify({"error": "Failed to clear chat history."}), 500

@app.route('/knowledge_base/status', methods=['GET'])
def knowledge_base_status_route():
    return jsonify(get_knowledge_base_status(session["user_id"]))

@app.route('/knowledge_base/clear', methods=['POST'])
def clear_knowledge_base_route():
    result = clear_knowledge_base(session["user_id"])
    return jsonify(result), 200 if result.get("ok") else 500

@app.route('/knowledge_base/search', methods=['POST'])
def search_knowledge_base_route():
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({"ok": False, "results": [], "error": "No query provided"}), 400
    try:
        k = max(1, min(int(data.get('k', 6)), 20))
    except (TypeError, ValueError):
        k = 6
    result = search_knowledge_base(session["user_id"], query, k=k)
    return jsonify(result), 200 if result.get("ok") else 500

@app.route('/knowledge_base/source', methods=['DELETE'])
def delete_knowledge_base_source_route():
    data = request.get_json(silent=True) or {}
    source = (data.get('source') or '').strip()
    if not source:
        return jsonify({"ok": False, "message": "No source provided"}), 400
    result = delete_knowledge_base_source(session["user_id"], source)
    return jsonify(result), 200 if result.get("ok") else 500


#################################################
# --- File Upload Configuration (for OCR) --- #
#################################################
UPLOAD_FOLDER = os.path.join(REPO_ROOT, 'uploads')
ALLOWED_EXTENSIONS_OCR = {'png', 'jpg', 'jpeg', 'txt', 'pdf', 'docx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/process_document_ocr', methods=['POST'])
def process_document_ocr():
    if llm_ocr is None:
        return jsonify({"error": "OCR processing requires optional OCR dependencies such as pytesseract."}), 503
    llm = get_chat_model()
    if llm is None:
        return jsonify({"error": "Document analysis requires an LLM provider to be configured."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    analysis_mode = request.form.get('analysis_mode', 'report')
    add_to_knowledge_base = request.form.get('add_to_knowledge_base', '').lower() in {'1', 'true', 'yes', 'on'}

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS_OCR):
        return jsonify({"error": "File type not allowed for OCR"}), 400

    # The name is only a label and the temp file gets a random one, so it needs no sanitising
    # beyond dropping any path. secure_filename would drop Hangul and leave "txt".
    filename = os.path.basename(file.filename.replace("\\", "/"))
    extension = filename.rsplit('.', 1)[1].lower()
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'],
                                         suffix="." + extension) as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file)

        result_dict, status_code = llm_ocr.analyze_document(
            file_path=temp_file_path,
            file_extension=extension,
            user_instructions=request.form.get('user_instructions', ''),
            llm=llm,
            prompt_template=main_chat_prompt_template,
            logger=app.logger,
            analysis_mode=analysis_mode,
            question=request.form.get('question', ''),
            include_extracted_text=add_to_knowledge_base,
            quota_reserver=reserve_ai_request,
        )
        if status_code == 429 and result_dict.get("error") == "daily_ai_quota_exceeded":
            return quota_exceeded_response(result_dict["quota_result"])
        if status_code == 200 and add_to_knowledge_base:
            result_dict["knowledge_base"] = ingest_text_to_vector_store(
                text=result_dict.pop("extracted_text", ""),
                source_name=filename,
                metadata={"analysis_mode": analysis_mode, "uploaded_via": "document_workspace"},
                user_id=session["user_id"],
            )
        return jsonify(result_dict), status_code
    except Exception as e:
        app.logger.error(f"Unhandled error in process_document_ocr for {filename}: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred during file preparation: {str(e)}"}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_clean:
                app.logger.error(f"Error cleaning up temp file {temp_file_path}: {e_clean}")


if __name__ == '__main__':
    init_db() # Initialize the database
    clear_all_chat_history()  # consultations last until the server restarts
    app.run(host="0.0.0.0", port=5005, debug=True, use_reloader=False)
