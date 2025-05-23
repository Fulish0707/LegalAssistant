from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from langchain_config import retriever, memory, agent
import threading
import time # Needed for a brief pause
import logging # Optional: Configure logging for Flask
import sqlite3
import json
import asyncio
from langchain.callbacks.base import BaseCallbackHandler

# Disable Flask's default request logging in notebook for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

DB_FILE = "app_state.db"

personal_statement_draft = ""
chat_history_list = [] # For initial load and simple display rendering

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS personal_statement (
            id INTEGER PRIMARY KEY,
            draft TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Ensure a row exists for the draft
    c.execute('INSERT OR IGNORE INTO personal_statement (id, draft) VALUES (1, "")')
    conn.commit()
    conn.close()

init_db()

def get_personal_statement():
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT draft FROM personal_statement WHERE id=1')
    row = c.fetchone()
    conn.close()
    return row['draft'] if row else ""

def set_personal_statement(draft):
    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE personal_statement SET draft=? WHERE id=1', (draft,))
    conn.commit()
    conn.close()

def get_chat_history():
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT role, content FROM chat_history ORDER BY id ASC')
    rows = c.fetchall()
    conn.close()
    return [{"role": row["role"], "content": row["content"]} for row in rows]

def add_chat_message(role, content):
    conn = get_db()
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (role, content) VALUES (?, ?)', (role, content))
    conn.commit()
    conn.close()

def clear_chat_history():
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM chat_history')
    conn.commit()
    conn.close()

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        self.queue.put(None)  # 스트리밍 종료 신호

# Define Flask routes
@app.route('/')
def index():
    return render_template('html_template.html')

@app.route('/chat', methods=['POST'])
def chat():
    global personal_statement_draft
    user_input = request.json.get('message')
    current_statement_draft_from_frontend = request.json.get('personal_statement_draft', '')
    personal_statement_draft = current_statement_draft_from_frontend

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    def generate():
        # 1. Add user message to history and DB
        chat_history_list.append({"role": "user", "content": user_input})
        add_chat_message("user", user_input)
        memory.save_context({"input": user_input}, {"output": ""})

        try:
            # 2. Retrieve relevant context and similar past conversations
            retrieved_context = ""
            similar_chats = []
            if retriever:
                retrieved_docs = retriever.invoke(user_input)
                retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])
                
                conn = get_db()
                c = conn.cursor()
                c.execute('''
                    SELECT role, content 
                    FROM chat_history 
                    WHERE content LIKE ? 
                    ORDER BY timestamp DESC 
                    LIMIT 3
                ''', (f'%{user_input}%',))
                similar_chats = [{"role": row["role"], "content": row["content"]} for row in c.fetchall()]
                conn.close()

            # 3. Create streaming queue
            queue = asyncio.Queue()
            streaming_handler = StreamingCallbackHandler(queue)

            # 4. Get response from agent with streaming
            agent_response = agent.invoke(
                {
                    "input": user_input,
                    "current_statement_draft": personal_statement_draft,
                    "retrieved_context": retrieved_context,
                    "chat_history": memory.load_memory_variables({})["chat_history"],
                    "similar_chats": similar_chats
                },
                callbacks=[streaming_handler]
            )

            # 5. Stream the response
            current_chunk = ""
            while True:
                token = queue.get()
                if token is None:  # 스트리밍 종료
                    break
                
                current_chunk += token
                if len(current_chunk) >= 50 or token in ['.', '!', '?', '\n']:
                    yield f"data: {json.dumps({'chunk': current_chunk})}\n\n"
                    current_chunk = ""

            # 6. Parse final response
            llm_response_text = agent_response["output"]
            chat_response = "Sorry, I couldn't parse the response."
            statement_suggestion = ""

            try:
                if "[CHAT]" in llm_response_text:
                    parts = llm_response_text.split("[CHAT]", 1)[1].split("[STATEMENT_SUGGESTION]", 1)
                    chat_response = parts[0].strip()
                    if len(parts) > 1:
                        statement_suggestion = parts[1].strip()
                elif "[STATEMENT_SUGGESTION]" in llm_response_text:
                    statement_suggestion = llm_response_text.split("[STATEMENT_SUGGESTION]", 1)[1].strip()
                    chat_response = "I have a suggestion for your personal statement."
                else:
                    chat_response = llm_response_text.strip()

                # 7. Add agent response to history and DB
                chat_history_list.append({"role": "assistant", "content": chat_response})
                add_chat_message("assistant", chat_response)
                memory.save_context({"input": user_input}, {"output": chat_response})

                # 8. Send final response
                yield f"data: {json.dumps({'final': True, 'chat_response': chat_response, 'statement_suggestion': statement_suggestion, 'similar_chats': similar_chats, 'updated_draft': personal_statement_draft})}\n\n"

            except Exception as parse_error:
                print(f"Error parsing LLM response: {parse_error}")
                print(f"Raw LLM response: {llm_response_text}")
                error_response = "I apologize, but I had trouble processing the response. Could you please try again?"
                chat_history_list.append({"role": "assistant", "content": error_response})
                add_chat_message("assistant", error_response)
                yield f"data: {json.dumps({'error': error_response})}\n\n"

        except Exception as e:
            print(f"Error during AI processing: {e}")
            import traceback
            traceback.print_exc()
            error_response = "An error occurred while processing your request."
            chat_history_list.append({"role": "assistant", "content": error_response})
            add_chat_message("assistant", error_response)
            yield f"data: {json.dumps({'error': error_response})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/get_initial_state', methods=['GET'])
def get_initial_state():
    return jsonify({
        "personal_statement_draft": get_personal_statement(),
        "chat_history": get_chat_history()
    })

@app.route('/get_similar_chats', methods=['POST'])
def get_similar_chats():
    query = request.json.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT role, content, timestamp 
        FROM chat_history 
        WHERE content LIKE ? 
        ORDER BY timestamp DESC 
        LIMIT 5
    ''', (f'%{query}%',))
    similar_chats = [{"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]} 
                    for row in c.fetchall()]
    conn.close()

    return jsonify({"similar_chats": similar_chats})

if __name__ == '__main__':
    print("Starting Flask server...")
def run_flask():
# Use a specific port, e.g., 5000
    app.run("0.0.0.0", port=5005, debug=False, use_reloader=False) # debug=True can cause issues with threading

# Check if the Flask server is already running before starting a new thread
# (This is a basic check, might not be perfect)
try:
    threading.Thread(target=run_flask).start()
    # Give the server a moment to start
    time.sleep(2)
    print("Flask server should be running at http://127.0.0.1:5005/")
except Exception as e:
    print(f"Failed to start Flask server thread: {e}")
