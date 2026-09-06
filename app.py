from dotenv import load_dotenv

# Load environment variables before importing project modules: library modules no
# longer call load_dotenv() themselves.
load_dotenv()

from flask import Flask, request, jsonify, render_template, Response, stream_with_context, g, session
from langchain_config import (
    AgentContext,
    build_agent,
    extract_agent_sections,
    ingest_text_to_vector_store,
    get_knowledge_base_status,
    clear_knowledge_base,
    search_knowledge_base,
    delete_knowledge_base_source,
) # Assuming these are correctly configured
import time # Needed for a brief pause
import logging
import sqlite3
import json
import re # Added for cleaning output
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import os
import sys
import secrets
from werkzeug.utils import secure_filename
import tempfile
import requests
import auth
from quota import (
    DAILY_AI_REQUEST_LIMIT,
    QuotaResult,
    client_ip_from_request,
    guest_ip_identity,
    reserve_daily_quota,
)


# Assuming llm_ocr.py is in the same directory or accessible via PYTHONPATH
try:
    import llm_ocr
except ModuleNotFoundError:
    llm_ocr = None

from llm_prompt import prompt as main_chat_prompt_template
from llm_model import describe_configuration, get_chat_model
from statistics_analysis import (
    ALLOWED_STATISTICS_EXTENSIONS,
    analyze_dataframe,
    build_ai_statistics_prompt,
    build_statistics_knowledge_text,
    load_table,
    render_statistics_html,
)
from html_sanitizer import sanitize_report_html

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
    GOOGLE_OIDC_CLIENT_ID=os.getenv("GOOGLE_OIDC_CLIENT_ID"),
    GOOGLE_OIDC_CLIENT_SECRET=os.getenv("GOOGLE_OIDC_CLIENT_SECRET"),
    GOOGLE_OIDC_REDIRECT_URI=os.getenv("GOOGLE_OIDC_REDIRECT_URI"),
    QUOTA_IP_SALT=os.getenv("QUOTA_IP_SALT"),
    DAILY_AI_REQUEST_LIMIT=int(os.getenv("DAILY_AI_REQUEST_LIMIT", str(DAILY_AI_REQUEST_LIMIT))),
)
app.config["QUOTA_IP_SALT"] = app.config["QUOTA_IP_SALT"] or secret_key
app.config["GOOGLE_OIDC_CONFIGURED"] = all(
    app.config[name]
    for name in (
        "GOOGLE_OIDC_CLIENT_ID",
        "GOOGLE_OIDC_CLIENT_SECRET",
        "GOOGLE_OIDC_REDIRECT_URI",
    )
)
auth.init_oauth(app)


@app.before_request
def establish_anonymous_user():
    if "user_id" not in session:
        session["user_id"] = secrets.token_urlsafe(32)

DB_FILE = os.getenv("APP_STATE_DB", "sqlite_db/app_state.db") # Define DB_FILE before use
db_dir = os.path.dirname(DB_FILE)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)

# The agent's working state, kept deliberately apart from DB_FILE's chat_history
# table. chat_history is the display record the UI renders; this is what the agent
# actually reasons over between turns. Unifying them would mean the renderer and
# the model share one schema, and they do not: the checkpointer stores tool calls,
# tool results and structured-output messages the UI never shows.
CHECKPOINT_DB_FILE = os.getenv("AGENT_CHECKPOINT_DB", "sqlite_db/checkpoints.db")
checkpoint_db_dir = os.path.dirname(CHECKPOINT_DB_FILE)
if checkpoint_db_dir:
    os.makedirs(checkpoint_db_dir, exist_ok=True)


def build_checkpointer():
    """One long-lived SqliteSaver over one connection, shared by every request.

    Not SqliteSaver.from_conn_string(): that is a context manager which closes the
    connection on exit, which is right for a test and wrong for a process that must
    keep it open. check_same_thread=False is not optional either -- Flask runs the
    streaming generator on a different thread from the one that opens this
    connection, and without it the first streamed request raises a thread-affinity
    error from sqlite3.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    connection = sqlite3.connect(CHECKPOINT_DB_FILE, check_same_thread=False)
    connection.execute("PRAGMA busy_timeout = 5000")
    return SqliteSaver(connection)


CHECKPOINTER = build_checkpointer()

def ensure_database_schema(conn):
    cursor = conn.cursor()
    statement_columns = {row[1] for row in cursor.execute("PRAGMA table_info(personal_statement)")}
    if statement_columns and "user_id" not in statement_columns:
        cursor.execute("ALTER TABLE personal_statement RENAME TO personal_statement_legacy")
    chat_columns = {row[1] for row in cursor.execute("PRAGMA table_info(chat_history)")}
    if chat_columns and "user_id" not in chat_columns:
        cursor.execute("ALTER TABLE chat_history RENAME TO chat_history_legacy")
    cursor.execute("CREATE TABLE IF NOT EXISTS personal_statement (user_id TEXT PRIMARY KEY, draft TEXT NOT NULL DEFAULT '')")
    cursor.execute("CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_user_id_id ON chat_history(user_id, id)")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            google_subject TEXT NOT NULL UNIQUE,
            email TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_login_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_ai_quota (
            identity_kind TEXT NOT NULL CHECK(identity_kind IN ('user', 'guest_ip')),
            identity_value TEXT NOT NULL,
            quota_day TEXT NOT NULL,
            request_count INTEGER NOT NULL DEFAULT 0 CHECK(request_count >= 0),
            PRIMARY KEY (identity_kind, identity_value, quota_day)
        )
    """)
    if cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='personal_statement_legacy'").fetchone():
        row = cursor.execute("SELECT draft FROM personal_statement_legacy WHERE id=1").fetchone()
        if row:
            cursor.execute("INSERT OR IGNORE INTO personal_statement (user_id, draft) VALUES ('legacy-unassigned', ?)", (row[0],))
        cursor.execute("DROP TABLE personal_statement_legacy")
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


auth.register_auth_routes(app)
app.auth = auth


@app.context_processor
def inject_authenticated_account():
    return {
        "authenticated_account_email": auth.get_current_user_email(),
        "authenticated_user_id": auth.get_current_user_id(),
    }

def get_personal_statement_from_db(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT draft FROM personal_statement WHERE user_id=?', (user_id,))
    row = c.fetchone()
    return row['draft'] if row else ""

def set_personal_statement_in_db(user_id, draft):
    conn = get_db()
    c = conn.cursor()
    c.execute('INSERT INTO personal_statement (user_id, draft) VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET draft=excluded.draft', (user_id, draft))
    conn.commit()

def get_chat_history_from_db(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT role, content FROM chat_history WHERE user_id=? ORDER BY id ASC', (user_id,))
    rows = c.fetchall()
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
    conn.commit()
    print("Chat history cleared for the active anonymous session.")


def reserve_ai_request():
    """Reserve one daily AI request for the current account or anonymous client."""
    user_id = auth.get_current_user_id()
    if user_id:
        identity_kind, identity_value = "user", user_id
    else:
        client_ip = client_ip_from_request(request)
        identity_kind = "guest_ip"
        identity_value = guest_ip_identity(client_ip, app.config["QUOTA_IP_SALT"])
    return reserve_daily_quota(
        get_db(),
        identity_kind,
        identity_value,
        limit=app.config["DAILY_AI_REQUEST_LIMIT"],
    )


def quota_exceeded_response(result: QuotaResult):
    return jsonify({
        "error": "daily_ai_quota_exceeded",
        "limit": result.limit,
        "remaining": result.remaining,
    }), 429


# --- Text Cleaning Utility ---
def clean_agent_output(text_content):
    if not text_content:
        return ""
    
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned_text = ansi_escape.sub('', text_content)

    # Remove Markdown code blocks (```text``` or ```lang\ntext\n```)
    # This regex looks for ``` followed by an optional language, then newline, then content, then newline, then ```
    # Or ``` followed by content then ```
    # It captures the content in between.
    
    # Pattern for ```lang\ncontent\n``` (multiline with language)
    # and ```\ncontent\n``` (multiline without language)
    multiline_code_block_pattern = re.compile(r"^```(?:[a-zA-Z]*)?\n(.*?)\n```$", re.DOTALL | re.MULTILINE)
    
    # Pattern for ```content``` (inline)
    inline_code_block_pattern = re.compile(r"^```(.*?)```$", re.DOTALL | re.MULTILINE)

    match = multiline_code_block_pattern.match(cleaned_text)
    if match:
        # For multiline blocks, we want the content inside
        cleaned_text = match.group(1)
    else:
        # Check for inline blocks if multiline didn't match
        match = inline_code_block_pattern.match(cleaned_text)
        if match:
            cleaned_text = match.group(1)
            
    return cleaned_text.strip()

def parse_agent_output_sections(text_content):
    """Split model output into user-facing chat and writing/report suggestions."""
    section_names = [
        "CHAT",
        "STATEMENT_SUGGESTION",
        "JOB_STATEMENT_SUGGESTION",
        "REPORT_SUGGESTION",
    ]
    parsed = {name: "" for name in section_names}
    if not text_content:
        return {
            "chat_response": "",
            "statement_suggestion": "",
            "job_statement_suggestion": "",
            "report_suggestion": "",
        }

    section_pattern = re.compile(
        r"\[(CHAT|STATEMENT_SUGGESTION|JOB_STATEMENT_SUGGESTION|REPORT_SUGGESTION)\]",
        re.IGNORECASE,
    )
    matches = list(section_pattern.finditer(text_content))
    if not matches:
        return {
            "chat_response": text_content.strip(),
            "statement_suggestion": "",
            "job_statement_suggestion": "",
            "report_suggestion": "",
        }

    for index, match in enumerate(matches):
        name = match.group(1).upper()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text_content)
        parsed[name] = text_content[start:end].strip()

    chat_response = parsed["CHAT"].strip()
    if not chat_response:
        chat_response = text_content[:matches[0].start()].strip()
    if not chat_response:
        chat_response = "I prepared suggestions for your document."

    return {
        "chat_response": chat_response,
        "statement_suggestion": parsed["STATEMENT_SUGGESTION"].strip(),
        "job_statement_suggestion": parsed["JOB_STATEMENT_SUGGESTION"].strip(),
        "report_suggestion": parsed["REPORT_SUGGESTION"].strip(),
    }

# --- Streaming helpers ---

DRAFT_CONTEXT_HEADER = "[Current Draft Context]"

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _message_text(content):
    """Normalise a message's content to a plain string.

    ChatOllama with reasoning enabled can hand back a list of content blocks
    rather than a str; json.dumps-ing that straight to the browser arrives as
    "[object Object]" because the frontend does `streamed += data.content`. Only
    text blocks are visible answer; reasoning blocks are dropped here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts)
    return ""


def _tail_prefix_length(text, tag):
    """Length of the longest suffix of text that is a proper prefix of tag."""
    for size in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:size]):
            return size
    return 0


class _AnswerTokenFilter:
    """Decide once, per model call, whether a message stream is visible answer.

    Two things must never reach the browser as answer tokens. Reasoning, which
    qwen3 emits inline as <think>...</think>, is stripped wherever it appears.
    And the response_format schema JSON: a 4B rarely routes the answer through
    the structured-output tool call, so the final message content *is* the JSON,
    and the old token callback streamed `"job_statement_suggestion"` and `"}"`
    to users verbatim.

    The gate is deliberately decide-once rather than an incremental JSON parser.
    The frontend overwrites the streamed text with final_response.chat_response
    anyway, so a partial-JSON extractor would buy progressive reveal at the cost
    of a new class of leak (escapes, keys split across chunks, braces inside a
    value). Here the first non-whitespace character settles it: `{` or a code
    fence means structured output, and nothing from that message is streamed.
    The sections still arrive, once, in final_response.
    """

    def __init__(self):
        self._pending = ""
        self._in_think = False
        self._decided = False
        self._suppressed = False
        self._head = ""

    def feed(self, text):
        if not text:
            return ""
        self._pending += text
        visible = []
        while self._pending:
            if self._in_think:
                index = self._pending.find(_THINK_CLOSE)
                if index == -1:
                    keep = _tail_prefix_length(self._pending, _THINK_CLOSE)
                    self._pending = self._pending[len(self._pending) - keep:] if keep else ""
                    break
                self._pending = self._pending[index + len(_THINK_CLOSE):]
                self._in_think = False
                continue
            index = self._pending.find(_THINK_OPEN)
            if index == -1:
                keep = _tail_prefix_length(self._pending, _THINK_OPEN)
                cut = len(self._pending) - keep
                if cut:
                    visible.append(self._pending[:cut])
                self._pending = self._pending[cut:]
                break
            visible.append(self._pending[:index])
            self._pending = self._pending[index + len(_THINK_OPEN):]
            self._in_think = True
        return self._gate("".join(visible))

    def flush(self):
        """Emit whatever was held back waiting on a possible tag boundary."""
        if self._in_think:
            self._pending = ""
            return ""
        remainder, self._pending = self._pending, ""
        return self._gate(remainder)

    def _gate(self, text):
        if self._suppressed or not text:
            return ""
        if self._decided:
            return text
        self._head += text
        stripped = self._head.lstrip()
        if not stripped:
            return ""
        if stripped[0] in "{`":
            self._suppressed = True
            self._head = ""
            return ""
        self._decided = True
        emitted, self._head = self._head, ""
        return emitted


def _tool_activity(node, update):
    """One short trace line per tool result, mirroring the old on_tool_end.

    The frontend shows only the newest trace line while a turn is in flight, and
    the surrounding UI is Korean, so these are written for a reader rather than
    emitted as raw node names.
    """
    lines = []
    messages = (update or {}).get("messages") if isinstance(update, dict) else None
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        output = clean_agent_output(_message_text(message.content))
        if len(output) > 200:
            output = output[:200] + "..."
        name = getattr(message, "name", None) or "tool"
        lines.append(f"도구 결과 ({name}): {output}" if output else f"도구 실행: {name}")
    if lines:
        return lines
    return ["답변 생성 중" if node == "model" else f"단계: {node}"]


def _draft_is_already_in_thread(agent, run_config, draft):
    """True when the thread's last user turn already carried this exact draft.

    The draft used to be appended to every user message. That was harmless while
    turns were stateless, but the thread is persisted now, so turn N carried N
    copies of the draft into a 4B's context window. The fix keeps the draft in
    the message (the model needs it inline; the system prompt is static and the
    typed context is not interpolated into it) but sends it only when it changed.

    The thread itself is the source of truth, not a process-local cache: multiple
    workers would each hold a different one. Guests have no checkpointer, so
    nothing accumulates for them and get_state has nothing to read -- they always
    get the draft.
    """
    if not draft:
        return True
    if "thread_id" not in (run_config.get("configurable") or {}):
        return False
    try:
        state = agent.get_state(run_config)
    except Exception:
        logging.debug("Could not read thread state for draft de-duplication", exc_info=True)
        return False
    messages = (getattr(state, "values", None) or {}).get("messages") or []
    # Scan back to the most recent turn that actually carried a draft, not merely
    # the most recent user turn: once a draft is skipped, the latest user message
    # has no draft in it, and comparing against that would re-send on every other
    # turn -- half-rate accumulation is still accumulation.
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = _message_text(message.content)
            if DRAFT_CONTEXT_HEADER in content:
                carried = content.split(DRAFT_CONTEXT_HEADER, 1)[1].strip()
                return carried == draft.strip()
    return False


def _sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


# --- Flask Routes ---

def _generate_agent_response_stream(user_input, current_personal_statement_draft):
    """Stream a chat turn as SSE events from the LangGraph agent's own stream.

    The old producer pushed AgentExecutor callbacks (on_agent_action,
    on_tool_end, on_agent_finish) into a queue. Those callbacks never fire under
    create_agent, and the drain loop also broke on the first on_llm_end, which in
    a tool loop discarded the final turn. Iterating agent.stream() to exhaustion
    replaces both.
    """
    try:
        # thread_id is the authenticated account id and nothing else. A guest's only
        # identity is a random token in a signed cookie, with no lifecycle, no
        # deletion path and no owner to attribute the stored turns to, so guests get
        # no checkpointer and no thread: they are stateless between turns, exactly
        # as their chat history already is (add_chat_message_to_db is
        # authenticated-only). That also makes it impossible for two guests, or a
        # guest and an account, to land in one thread.
        thread_user_id = auth.get_current_user_id()
        agent = build_agent(
            thread_user_id,
            checkpointer=CHECKPOINTER if thread_user_id else None,
        )
        if agent is None:
            yield _sse({'type': 'error', 'content': 'AI chat is not configured. Set GOOGLE_API_KEY in .env and restart the app.'})
            return

        quota_result = reserve_ai_request()
        if not quota_result.allowed:
            yield _sse({'type': 'error', 'content': 'Daily AI request quota exceeded.', 'error': 'daily_ai_quota_exceeded', 'limit': quota_result.limit})
            return

        run_config = {}
        if thread_user_id:
            run_config["configurable"] = {"thread_id": thread_user_id}

        # Only the new turn goes in. The checkpointer replays the thread's earlier
        # messages, so passing history here as well would duplicate every turn.
        agent_prompt = user_input
        if current_personal_statement_draft and not _draft_is_already_in_thread(
            agent, run_config, current_personal_statement_draft
        ):
            agent_prompt += f"\n\n{DRAFT_CONTEXT_HEADER}\n{current_personal_statement_draft}"

        final_state = None
        filters = {}
        buffer = ""
        for mode, chunk in agent.stream(
            {"messages": [("user", agent_prompt)]},
            run_config,
            context=AgentContext(user_id=thread_user_id),
            stream_mode=["messages", "updates", "values"],
        ):
            if mode == "messages":
                message = chunk[0] if isinstance(chunk, tuple) else chunk
                # ToolMessages ride this same channel; RAG hits are trace, not answer.
                if not isinstance(message, AIMessage):
                    if isinstance(message, ToolMessage):
                        # A tool result always separates two model calls, so it is a
                        # safe place to retire the per-call filters. Verified that
                        # ChatOllama gives each call a distinct chunk id, but this
                        # keeps the id-less fallback key from being shared across
                        # calls, which would let prose in call 1 open the gate for
                        # the schema JSON in call 2.
                        for retired in filters.values():
                            buffer += retired.flush()
                        filters.clear()
                    continue
                key = getattr(message, "id", None) or "default"
                token_filter = filters.get(key)
                if token_filter is None:
                    token_filter = filters[key] = _AnswerTokenFilter()
                visible = token_filter.feed(_message_text(getattr(message, "content", "")))
                if visible:
                    buffer += visible
                    if len(buffer) >= 20 or any(p in visible for p in ('.', '!', '?', '\n', ' ')):
                        yield _sse({'type': 'token_chunk', 'content': buffer})
                        buffer = ""
            elif mode == "updates":
                for node, update in (chunk or {}).items():
                    for line in _tool_activity(node, update):
                        yield _sse({'type': 'thought', 'content': line})
            elif mode == "values":
                final_state = chunk

        for token_filter in filters.values():
            buffer += token_filter.flush()
        if buffer:
            yield _sse({'type': 'token_chunk', 'content': buffer})

        # Preferred route: the schema-enforced sections, from structured_response
        # (Gemini) or from JSON in the final message content (qwen3:4b). Falls back
        # to the legacy [CHAT] marker parser when neither is available.
        parsed_sections = extract_agent_sections(final_state)
        if parsed_sections is None:
            messages = (final_state or {}).get("messages") or []
            agent_output_string = getattr(messages[-1], "content", "") if messages else ""
            cleaned_agent_output_string = clean_agent_output(_message_text(agent_output_string))
            parsed_sections = parse_agent_output_sections(cleaned_agent_output_string)
            if not agent_output_string:
                print(f"Warning: Agent returned no final message. Full response object: {final_state}")
            elif not cleaned_agent_output_string:
                print(f"Warning: Agent output became empty after cleaning. Original: '{agent_output_string}'")
        parsed_chat_response = parsed_sections["chat_response"] if parsed_sections["chat_response"] else "Sorry, I couldn't generate a response."

        authenticated_user_id = auth.get_current_user_id()
        if authenticated_user_id:
            add_chat_message_to_db(authenticated_user_id, "assistant", parsed_chat_response)

        final_data = {
            "type": "final_response",
            "chat_response": parsed_chat_response,
            "statement_suggestion": parsed_sections["statement_suggestion"],
            "job_statement_suggestion": parsed_sections["job_statement_suggestion"],
            "report_suggestion": parsed_sections["report_suggestion"],
            "current_draft_processed": current_personal_statement_draft
        }
        yield _sse(final_data)

    except Exception as e:
        logging.exception("Error during AI processing stream (_generate_agent_response_stream)")
        error_response_content = "An error occurred while processing your request. Please check server logs."
        authenticated_user_id = auth.get_current_user_id()
        if authenticated_user_id:
            add_chat_message_to_db(authenticated_user_id, "assistant", error_response_content)
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
        "agent_configured": llm_available,
        "search_configured": bool(os.getenv("SERPAPI_API_KEY")),
        "google_oidc_configured": app.config["GOOGLE_OIDC_CONFIGURED"],
        "llm_provider": None,
        "embedding_provider": None,
        "collection": None,
        "llm_reachable": None,
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
        current_personal_statement_draft = data.get('personal_statement_draft', '')

        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        authenticated_user_id = auth.get_current_user_id()
        if authenticated_user_id:
            add_chat_message_to_db(authenticated_user_id, "user", user_input)
        # The POST request now just acknowledges receipt.
        # The actual processing and streaming will be triggered by the frontend's GET request.
        return jsonify({"status": "message_received_ok", "message": "Proceed to connect to SSE for response."})

    elif request.method == 'GET':
        # For SSE, data comes from query parameters
        user_input = request.args.get('message')
        current_personal_statement_draft = request.args.get('personal_statement_draft', '')

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

        return Response(stream_with_context(_generate_agent_response_stream(user_input, current_personal_statement_draft)), mimetype='text/event-stream')

    # Fallback to satisfy type checkers: ensure the route always returns a valid response
    return jsonify({"error": "Invalid request method."}), 405

@app.route('/get_initial_state', methods=['GET'])
def get_initial_state():
    authenticated_user_id = auth.get_current_user_id()
    if not authenticated_user_id:
        return jsonify({
            "personal_statement_draft": session.get("guest_draft", ""),
            "chat_history": [],
        })
    return jsonify({
        "personal_statement_draft": get_personal_statement_from_db(authenticated_user_id),
        "chat_history": get_chat_history_from_db(authenticated_user_id)
    })

@app.route('/update_statement', methods=['POST'])
def update_statement():
    data = request.get_json(silent=True) or {}
    new_draft = data.get('personal_statement_draft')
    if new_draft is None: # Check for None, as empty string is a valid draft
        return jsonify({"error": "No personal_statement_draft provided"}), 400
    
    try:
        authenticated_user_id = auth.get_current_user_id()
        if authenticated_user_id:
            set_personal_statement_in_db(authenticated_user_id, new_draft)
        else:
            session["guest_draft"] = new_draft
        return jsonify({"message": "Personal statement updated successfully."}), 200
    except Exception as e:
        logging.exception("Error updating personal statement in DB")
        return jsonify({"error": "Failed to update personal statement."}), 500

@app.route('/get_similar_chats', methods=['POST']) # This seems more like a debug/utility endpoint
def get_similar_chats_route(): # Renamed to avoid conflict with internal var
    data = request.get_json(silent=True) or {}
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    authenticated_user_id = auth.get_current_user_id()
    if not authenticated_user_id:
        return jsonify({"similar_chats": []})

    conn = get_db()
    c = conn.cursor()
    # Example: find chats containing the query text, excluding very recent ones to avoid self-match
    c.execute('''
        SELECT role, content, timestamp
        FROM chat_history
        WHERE user_id=? AND content LIKE ? AND timestamp < datetime('now', '-5 seconds')
        ORDER BY timestamp DESC
        LIMIT 5
    ''', (authenticated_user_id, f'%{query}%'))
    similar_chats = [{"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]}
                     for row in c.fetchall()]

    return jsonify({"similar_chats": similar_chats})


def clear_agent_thread(user_id):
    """Drop the agent's checkpointed thread when the display history is cleared.

    These two stores are separate, but a user clearing their history means one
    thing, not two: a user who clears the transcript and then finds the agent still
    quoting it back would call that a bug, correctly. Failures here are logged and
    swallowed -- clearing the record the user can see is this endpoint's job, and it
    must not 500 because the checkpoint store is unhappy.
    """
    try:
        CHECKPOINTER.delete_thread(user_id)
    except Exception as error:
        logging.warning(f"Could not clear the agent thread for {user_id}: {error}")


@app.route('/clear_chat_history_endpoint', methods=['POST']) # Added specific endpoint name
def clear_chat_history_endpoint():
    try:
        authenticated_user_id = auth.get_current_user_id()
        if authenticated_user_id:
            clear_chat_history_in_db(authenticated_user_id)
            clear_agent_thread(authenticated_user_id)
        return jsonify({"message": "Chat history cleared successfully."}), 200
    except Exception as e:
        logging.error(f"Error clearing chat history: {e}")
        return jsonify({"error": "Failed to clear chat history."}), 500

@app.route('/knowledge_base/status', methods=['GET'])
@auth.login_required
def knowledge_base_status_route():
    return jsonify(get_knowledge_base_status(auth.get_current_user_id()))

@app.route('/knowledge_base/clear', methods=['POST'])
@auth.login_required
def clear_knowledge_base_route():
    result = clear_knowledge_base(auth.get_current_user_id())
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


def application_setup():
    """Encapsulates setup that should happen in the worker process."""
    print("--- Worker Process Setup ---")
    print("Initializing DB...")
    init_db()
    # Add any other one-time initializations for the worker here
    print("--- Worker Process Setup Complete ---")

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


@app.route('/statistics/analyze', methods=['POST'])
@auth.login_required
def analyze_statistics_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_STATISTICS_EXTENSIONS):
        return jsonify({"error": "File type not allowed. Upload CSV, TSV, or XLSX."}), 400

    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()
    user_question = request.form.get('question', '')
    add_to_knowledge_base = request.form.get('add_to_knowledge_base', '').lower() in {'1', 'true', 'yes', 'on'}
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=app.config['UPLOAD_FOLDER'],
            suffix=os.path.splitext(filename)[1],
        ) as temp_file_obj:
            temp_file_path = temp_file_obj.name
            file.save(temp_file_obj)

        dataframe = load_table(temp_file_path, file_extension)
        analysis = analyze_dataframe(dataframe)

        llm = get_chat_model()
        ai_insight = None
        if llm is not None:
            prompt_text = build_ai_statistics_prompt(filename, analysis, user_question)
            quota_result = reserve_ai_request()
            if not quota_result.allowed:
                return quota_exceeded_response(quota_result)
            ai_response = llm.invoke(prompt_text)
            ai_insight = sanitize_report_html(getattr(ai_response, "content", str(ai_response)))

        report_html = sanitize_report_html(render_statistics_html(
            filename=filename,
            analysis=analysis,
            ai_insight=ai_insight,
            ai_enabled=llm is not None,
        ))

        response_payload = {
            "report_html": report_html,
            "analysis": analysis,
            "ai_enabled": llm is not None,
            "filename": filename,
        }

        if add_to_knowledge_base:
            response_payload["knowledge_base"] = ingest_text_to_vector_store(
                text=build_statistics_knowledge_text(filename, analysis, ai_insight),
                source_name=filename,
                metadata={
                    "analysis_type": "statistics",
                    "uploaded_via": "statistics_workspace",
                }, user_id=auth.get_current_user_id(),
            )

        return jsonify(response_payload), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unhandled error in analyze_statistics_route for {filename}: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred during statistics analysis: {str(e)}"}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e_clean:
                app.logger.error(f"Error cleaning up temp file {temp_file_path}: {e_clean}")

# --- OCR Processing Route (Existing) ---
@app.route('/process_document_ocr', methods=['POST'])
@auth.login_required
def process_document_ocr():
    if llm_ocr is None:
        return jsonify({
            "error": "OCR processing requires optional OCR dependencies such as pytesseract."
        }), 503
    llm = get_chat_model()
    if llm is None:
        return jsonify({
            "error": "Document analysis requires an LLM provider to be configured."
        }), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    user_instructions = request.form.get('user_instructions', '')
    analysis_mode = request.form.get('analysis_mode', 'report')
    question = request.form.get('question', '')
    add_to_knowledge_base = request.form.get('add_to_knowledge_base', '').lower() in {'1', 'true', 'yes', 'on'}

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS_OCR):
        return jsonify({"error": "File type not allowed for OCR"}), 400

    filename = secure_filename(file.filename)
    temp_file_path = None  # Initialize to ensure it's defined for the finally block

    try:
        # Create a temporary file to save the upload
        # Using 'with' ensures the file handle is closed even if errors occur during creation/saving
        with tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], 
                                         suffix=os.path.splitext(filename)[1]) as temp_file_obj:
            temp_file_path = temp_file_obj.name
            file.save(temp_file_obj) # Save the uploaded file object to the temp file

        # At this point, temp_file_obj is closed, and the file is saved at temp_file_path
        file_extension = filename.rsplit('.', 1)[1].lower()

        result_dict, status_code = llm_ocr.analyze_document(
            file_path=temp_file_path,
            file_extension=file_extension,
            user_instructions=user_instructions,
            llm=llm, # Passed from app.py's imports
            prompt_template=main_chat_prompt_template, # Passed from app.py's imports
            logger=app.logger, # Pass Flask app's logger
            analysis_mode=analysis_mode,
            question=question,
            include_extracted_text=add_to_knowledge_base,
            quota_reserver=reserve_ai_request,
        )

        if status_code == 429 and result_dict.get("error") == "daily_ai_quota_exceeded":
            return quota_exceeded_response(result_dict["quota_result"])

        if status_code == 200 and add_to_knowledge_base:
            extracted_text = result_dict.pop("extracted_text", "")
            ingestion_result = ingest_text_to_vector_store(
                text=extracted_text,
                source_name=filename,
                metadata={
                    "analysis_mode": analysis_mode,
                    "uploaded_via": "document_workspace",
                }, user_id=auth.get_current_user_id()
            )
            result_dict["knowledge_base"] = ingestion_result
        
        return jsonify(result_dict), status_code

    except Exception as e:
        # This catch block is for errors during file handling in app.py itself, 
        # before calling generate_report_from_document.
        app.logger.error(f"Unhandled error in process_document_ocr (file handling stage) for {filename if 'filename' in locals() else 'unknown file'}: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred during file preparation: {str(e)}"}), 500
    finally:
        # Cleanup the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                app.logger.info(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e_clean:
                app.logger.error(f"Error cleaning up temp file {temp_file_path}: {e_clean}")


if __name__ == '__main__':
    init_db() # Initialize the database
    app.run(host="0.0.0.0", port=5005, debug=True, use_reloader=False)
