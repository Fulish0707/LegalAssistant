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
from functools import wraps
from threading import BoundedSemaphore
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import abort, make_response
from web_security import check_browser_request, csrf_token, public_error, require_text, reserve_burst, security_headers
from document_security import DocumentRejected, validate_upload
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
from llm_model import get_chat_model

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
    secret_key = secrets.token_urlsafe(48)

if len(secret_key) < 32 or secret_key.startswith("replace-"):
    raise RuntimeError("SECRET_KEY must be a random secret of at least 32 characters.")

app.config.update(
    SECRET_KEY=secret_key,
    MAX_CONTENT_LENGTH=26 * 1024 * 1024,
    MAX_FORM_MEMORY_SIZE=128 * 1024,
    MAX_FORM_PARTS=12,
    TRUSTED_HOSTS=[host.strip() for host in os.getenv("TRUSTED_HOSTS", "localhost,127.0.0.1,[::1]").split(",") if host.strip()],
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=not is_local_environment,
    QUOTA_IP_SALT=os.getenv("QUOTA_IP_SALT"),
    DAILY_AI_REQUEST_LIMIT=int(os.getenv("DAILY_AI_REQUEST_LIMIT", str(DAILY_AI_REQUEST_LIMIT))),
)
app.config["QUOTA_IP_SALT"] = app.config["QUOTA_IP_SALT"] or secret_key
if len(app.config["QUOTA_IP_SALT"]) < 32 or app.config["QUOTA_IP_SALT"].startswith("replace-"):
    raise RuntimeError("QUOTA_IP_SALT must be random and at least 32 characters, or unset.")
trusted_proxy_count = int(os.getenv("TRUSTED_PROXY_COUNT", "0"))
if trusted_proxy_count not in {0, 1}:
    raise RuntimeError("TRUSTED_PROXY_COUNT must be 0 or 1.")
if trusted_proxy_count:
    # Trust only the scheme from an explicitly configured, restricted proxy hop.
    # IP quotas continue to use the direct peer; the proxy must enforce client budgets.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=0, x_proto=trusted_proxy_count,
                          x_host=0, x_port=0, x_prefix=0)


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
    check_browser_request()

DB_FILE = os.getenv("APP_STATE_DB", "sqlite_db/app_state.db") # Define DB_FILE before use
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
    cursor.execute("CREATE TABLE IF NOT EXISTS chat_requests (user_id TEXT NOT NULL, request_id TEXT NOT NULL, PRIMARY KEY(user_id, request_id))")
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


def get_chat_history_from_db(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT role, content FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT 100', (user_id,))
    rows = c.fetchall()[::-1]
    return [{"role": row["role"], "content": row["content"]} for row in rows]

def add_chat_message_to_db(user_id, role, content):
    conn = get_db()
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)', (user_id, role, content))
    conn.commit()

def clear_chat_history_in_db(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM chat_history WHERE user_id=?', (user_id,))
    c.execute('DELETE FROM chat_requests WHERE user_id=?', (user_id,))
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


# Controls apply before expensive work. Only the direct peer defines an IP budget.
_work_slots = BoundedSemaphore(2)
_conversation_slots = BoundedSemaphore(1)
_upload_slots = BoundedSemaphore(1)

def limited_work(function, slots=_work_slots):
    @wraps(function)
    def guarded(*args, **kwargs):
        if not slots.acquire(blocking=False):
            abort(429, description="service_busy")
        try:
            response = make_response(function(*args, **kwargs))
        except BaseException:
            slots.release()
            raise
        if response.is_streamed:
            response.call_on_close(slots.release)
        else:
            slots.release()
        return response
    return guarded

def conversation_work(function):
    return limited_work(function, _conversation_slots)

def upload_work(function):
    return limited_work(function, _upload_slots)

def json_object():
    if request.content_length and request.content_length > 32768:
        abort(413)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, description="invalid_input")
    return data

@app.before_request
def control_request_rate():
    if request.method in {"POST", "DELETE", "PUT", "PATCH"} and request.endpoint != "process_document_ocr":
        request.max_content_length = 32768
    if request.endpoint in {"chat", "process_document_ocr", "search_knowledge_base_route"} and tracing_status()["enabled"]:
        abort(403, description="external_tracing_disabled")
    if request.endpoint in {"chat", "process_document_ocr", "knowledge_base_status_route",
                            "clear_knowledge_base_route", "search_knowledge_base_route",
                            "delete_knowledge_base_source_route", "clear_chat_history_endpoint"}:
        identity = guest_ip_identity(client_ip_from_request(request), app.config["QUOTA_IP_SALT"])
        if not reserve_burst(get_db(), identity, limit=30):
            abort(429, description="rate_limit_exceeded")

app.after_request(security_headers)
app.register_error_handler(HTTPException, public_error)
app.register_error_handler(500, public_error)

# --- Text Cleaning Utility ---
_SMALL_TALK = re.compile(r"^(?:안녕|반가|고마|감사|수고|ㅎㅇ|hi|hello|thanks?|thank you|bye)", re.IGNORECASE)


def _needs_legal_search(text):
    """Everything but greetings and thanks gets the statute/case-law pre-search."""
    stripped = (text or "").strip()
    if len(stripped) < 4:
        return False
    return not (_SMALL_TALK.match(stripped) and len(stripped) <= 20)


def _sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


# --- Flask Routes ---

GREETING_REPLY = "안녕하세요. 법률 관련 궁금한 점을 구체적으로 적어 주시면 법령과 판례를 찾아 안내해 드리겠습니다."
LEGAL_BACKEND_UNSET = "법률 답변 서비스가 설정되지 않았습니다. LEGAL_RAG_URL을 설정한 뒤 다시 시작하세요."
LEGAL_BACKEND_DOWN = "법률 답변 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요."


def _legal_conversation_id():
    """The browser's workspace id plus an epoch that clearing the chat advances.

    Clearing first deletes the backend workspace history, then advances this epoch.
    """
    return f"{session['user_id']}-{session.get('conversation_epoch', 0)}"


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
    asked = [turn["content"] for turn in get_chat_history_from_db(session["user_id"]) if turn["role"] == "user"]
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

        if _needs_legal_search(user_input):
            yield _sse({'type': 'thought', 'content': '법령·판례 검색 및 답변 작성 중'})
            documents = _case_documents(user_input)
            if documents:
                yield _sse({'type': 'thought', 'content': f'사건 자료 {len(documents)}건 참고'})
            body = legal_backend.ask(legal_rag_url, user_input, _legal_conversation_id(), documents, workspace=session["user_id"])
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
            chat_response, sources, questions = GREETING_REPLY, [], []

        add_chat_message_to_db(session["user_id"], "assistant", chat_response)
        yield _sse({
            "type": "final_response",
            "chat_response": chat_response,
            "sources": sources,
            "questions": questions,
        })

    except legal_backend.LegalBackendUnavailable:
        logging.error("legal_rag chat service unavailable")
        add_chat_message_to_db(session["user_id"], "assistant", LEGAL_BACKEND_DOWN)
        yield _sse({'type': 'error', 'content': LEGAL_BACKEND_DOWN})
    except Exception as error:
        logging.error("Chat processing failed (%s)", type(error).__name__)
        error_response_content = "An error occurred while processing your request. Please check server logs."
        add_chat_message_to_db(session["user_id"], "assistant", error_response_content)
        yield _sse({'type': 'error', 'content': error_response_content})
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/chat', methods=['POST'])
@conversation_work
def chat():
    data = json_object()
    user_input = require_text(data.get('message'), limit=4000)
    request_id = require_text(data.get('request_id'), limit=64)
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", request_id):
        abort(400, description="invalid_input")
    if get_db().execute("SELECT 1 FROM chat_requests WHERE user_id=? AND request_id=?", (session["user_id"], request_id)).fetchone():
        return jsonify({"error": "duplicate_request"}), 409
    rows = get_db().execute("SELECT COUNT(*) FROM chat_history WHERE user_id=?", (session["user_id"],)).fetchone()[0]
    if rows + 2 > 500:
        return jsonify({"error": "workspace_storage_limit"}), 429
    quota_result = reserve_ai_request()
    if not quota_result.allowed:
        return quota_exceeded_response(quota_result)
    inserted = get_db().execute("INSERT OR IGNORE INTO chat_requests VALUES (?, ?)", (session["user_id"], request_id))
    if inserted.rowcount != 1:
        get_db().rollback()
        return jsonify({"error": "duplicate_request"}), 409
    add_chat_message_to_db(session["user_id"], "user", user_input)
    return Response(stream_with_context(_generate_legal_response_stream(user_input)),
                    mimetype='text/event-stream', headers={"X-Accel-Buffering": "no"})

@app.route('/get_initial_state', methods=['GET'])
def get_initial_state():
    user_id = session["user_id"]
    return jsonify({
        "csrf_token": csrf_token(),
        "chat_history": get_chat_history_from_db(user_id)
    })


@app.route('/clear_chat_history_endpoint', methods=['POST']) # Added specific endpoint name
@conversation_work
def clear_chat_history_endpoint():
    try:
        user_id = session["user_id"]
        backend_url = os.getenv("LEGAL_RAG_URL", "").strip()
        if not backend_url:
            raise legal_backend.LegalBackendUnavailable("backend_not_configured")
        legal_backend.forget(backend_url, user_id)
        clear_chat_history_in_db(user_id)
        session["conversation_epoch"] = session.get("conversation_epoch", 0) + 1
        return jsonify({"message": "Chat history cleared successfully."}), 200
    except Exception as e:
        logging.error("History deletion failed (%s)", type(e).__name__)
        return jsonify({"error": "Failed to clear chat history."}), 500

@app.route('/knowledge_base/status', methods=['GET'])
@limited_work
def knowledge_base_status_route():
    return jsonify(get_knowledge_base_status(session["user_id"]))

@app.route('/knowledge_base/clear', methods=['POST'])
@limited_work
def clear_knowledge_base_route():
    result = clear_knowledge_base(session["user_id"])
    return jsonify(result), 200 if result.get("ok") else 500

@app.route('/knowledge_base/search', methods=['POST'])
@limited_work
def search_knowledge_base_route():
    data = json_object()
    query = require_text(data.get("query"), limit=2000)
    if not query:
        return jsonify({"ok": False, "results": [], "error": "No query provided"}), 400
    try:
        k = max(1, min(int(data.get('k', 6)), 20))
    except (TypeError, ValueError):
        k = 6
    result = search_knowledge_base(session["user_id"], query, k=k)
    return jsonify(result), 200 if result.get("ok") else 500

@app.route('/knowledge_base/source', methods=['DELETE'])
@limited_work
def delete_knowledge_base_source_route():
    data = json_object()
    source = require_text(data.get("source"), limit=200)
    if not source:
        return jsonify({"ok": False, "message": "No source provided"}), 400
    result = delete_knowledge_base_source(session["user_id"], source)
    return jsonify(result), 200 if result.get("ok") else 500


#################################################
# --- File Upload Configuration (for OCR) --- #
#################################################
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_OCR = {'png', 'jpg', 'jpeg', 'txt', 'pdf', 'docx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/process_document_ocr', methods=['POST'])
@upload_work
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

    require_text(file.filename, limit=200)
    user_instructions = require_text(request.form.get("user_instructions", ""), limit=2000, allow_empty=True)
    question = require_text(request.form.get("question", ""), limit=2000, allow_empty=True)
    if analysis_mode not in llm_ocr.DOCUMENT_ANALYSIS_MODES:
        abort(400, description="invalid_input")
    if tracing_status()["enabled"]:
        return jsonify({"error": "document_tracing_disabled"}), 403

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS_OCR):
        return jsonify({"error": "File type not allowed for OCR"}), 400

    # The name is only a label and the temp file gets a random one, so it needs no sanitising
    # beyond dropping any path. secure_filename would drop Hangul and leave "txt".
    filename = os.path.basename(file.filename.replace("\\", "/"))
    extension = filename.rsplit('.', 1)[1].lower()
    try:
        validate_upload(file, extension)
    except DocumentRejected:
        return jsonify({"error": "document_rejected"}), 422
    quota_result = reserve_ai_request()
    if not quota_result.allowed:
        return quota_exceeded_response(quota_result)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'],
                                         suffix="." + extension) as temp_file:
            temp_file_path = temp_file.name
            file.save(temp_file)

        result_dict, status_code = llm_ocr.analyze_document(
            file_path=temp_file_path,
            file_extension=extension,
            user_instructions=user_instructions,
            llm=llm,
            prompt_template=main_chat_prompt_template,
            logger=app.logger,
            analysis_mode=analysis_mode,
            question=question,
            include_extracted_text=add_to_knowledge_base,
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
        app.logger.error("Document processing failed (%s)", type(e).__name__)
        return jsonify({"error": "document_processing_failed"}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_clean:
                app.logger.error(f"Error cleaning up temp file {temp_file_path}: {e_clean}")


if __name__ == '__main__':
    init_db() # Initialize the database
    app.run(host="127.0.0.1", port=5005, debug=False, use_reloader=False)
