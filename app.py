from flask import Flask, request, jsonify, render_template
from langchain_config import vector_store, retriever, memory, agent
from llm_prompt import prompt
from llm_model import llm
import threading
import time # Needed for a brief pause
import logging # Optional: Configure logging for Flask
import sqlite3

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

# Define Flask routes
@app.route('/')
def index():
    return render_template('html_template.html')

@app.route('/chat', methods=['POST'])
def chat():
    global personal_statement_draft
    user_input = request.json.get('message')
    current_statement_draft_from_frontend = request.json.get('personal_statement_draft', '')
    personal_statement_draft = current_statement_draft_from_frontend # Update backend's state


    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # 1. Add user message to history (for display/memory)
    chat_history_list.append({"role": "user", "content": user_input})
    memory.save_context({"input": user_input}, {"output": ""})

    try:
        # 2. (Optional) Retrieve context from RAG
        retrieved_context = ""
        if retriever:
             retrieved_docs = retriever.invoke(user_input)
             retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])

        # 3. Prepare input for the LLM
        chat_history_for_prompt = memory.load_memory_variables({})["chat_history"]

        messages_for_llm = [
            ("system", prompt.messages[0].prompt.template.format(
                current_statement_draft=personal_statement_draft,
                retrieved_context=retrieved_context
            )),
        ] + chat_history_for_prompt + [("human", user_input)]

        llm_response_object = llm.invoke(messages_for_llm)
        llm_response_text = llm_response_object.content

        # 4. Parse LLM output
        chat_response = "Sorry, I couldn't parse the response."
        statement_suggestion = ""

        chat_marker = "[CHAT]"
        statement_marker = "[STATEMENT_SUGGESTION]"

        if chat_marker in llm_response_text:
             parts = llm_response_text.split(chat_marker, 1)[1].split(statement_marker, 1)
             chat_response = parts[0].strip()
             if len(parts) > 1:
                 statement_suggestion = parts[1].strip()
        elif statement_marker in llm_response_text:
             statement_suggestion = llm_response_text.split(statement_marker, 1)[1].strip()
             chat_response = "Okay, I have a suggestion for the statement." # Default


        # 5. Add agent response to history
        chat_history_list.append({"role": "assistant", "content": chat_response})
        memory.save_context({"input": user_input}, {"output": chat_response})

        # 6. Return response
        return jsonify({
            "chat_response": chat_response,
            "statement_suggestion": statement_suggestion,
        })

    except Exception as e:
        print(f"Error during AI processing: {e}")
        import traceback
        traceback.print_exc()
        error_response = "An error occurred while processing your request."
        chat_history_list.append({"role": "assistant", "content": error_response})
        return jsonify({"chat_response": error_response}), 500

@app.route('/get_initial_state', methods=['GET'])
def get_initial_state():
     # This returns the state from the global variables, which reset on kernel restart.
     # For persistence, load from file/DB here.
     return jsonify({
         "personal_statement_draft": get_personal_statement(), # Reverted to DB
         "chat_history": get_chat_history() # Reverted to DB
     })

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
    print("Flask server should be running at http://127.0.0.1:5000/")
except Exception as e:
    print(f"Failed to start Flask server thread: {e}")
