from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from langchain_config import retriever, memory, agent
import threading
import time # Needed for a brief pause
import logging # Optional: Configure logging for Flask
import sqlite3
import json
import queue  # Add for synchronous streaming
from langchain.callbacks.base import BaseCallbackHandler

# Disable Flask's default request logging in notebook for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

DB_FILE = "sqlite_db/app_state.db"

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
    def __init__(self, queue):  # Remove loop parameter
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put({"type": "token", "content": token})

    def on_llm_end(self, *args, **kwargs) -> None:
        self.queue.put({"type": "llm_end"})

    def on_agent_action(self, action, **kwargs) -> None:
        log_text = ""
        if hasattr(action, 'log') and action.log:
            log_text = action.log.strip()

        thought_message = ""

        if log_text.startswith("Thought:"):
            thought_message = log_text
            if "Action:" not in log_text and "Tool:" not in log_text and action.tool:
                thought_message += f"\nAction: {action.tool}"
                tool_input_str = str(action.tool_input) if action.tool_input is not None else ""
                if tool_input_str.strip():
                    thought_message += f"\nAction Input: {tool_input_str}"
        elif log_text:
            thought_message = f"Thought: (Interpreted action from LLM log)\n{log_text}"
            if action.tool and action.tool not in log_text:
                tool_input_str = str(action.tool_input) if action.tool_input is not None else ""
                thought_message += f"\n(Interpreted Action: {action.tool}, Input: {tool_input_str})"
        elif action.tool:
            thought_message = f"Thought: Decided to use tool.\nAction: {action.tool}"
            tool_input_str = str(action.tool_input) if action.tool_input is not None else ""
            if tool_input_str.strip():
                thought_message += f"\nAction Input: {tool_input_str}"
        else:
            thought_message = "Thought: (Agent is processing an action without detailed log)"

        self.queue.put({"type": "thought", "content": thought_message})

    def on_tool_end(self, output: str, **kwargs) -> None:
        max_len = 200
        tool_output_display = output[:max_len] + "..." if len(output) > max_len else output
        self.queue.put({"type": "thought", "content": f"Tool Output: {tool_output_display}"})

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        chain_name = serialized.get("name", serialized.get("id", ["Unknown Chain"])[-1])
        if "AgentExecutor" not in chain_name and "RunnableSequence" not in chain_name and "LLMChain" not in chain_name:
            self.queue.put({"type": "thought", "content": f"Starting sub-chain: {chain_name}"})

    def on_agent_finish(self, finish, **kwargs) -> None:
        if hasattr(finish, 'log') and finish.log:
            final_thought_log = finish.log.strip()
            if "Final Answer:" in final_thought_log:
                final_thought_log = final_thought_log.split("Final Answer:", 1)[0].strip()
            if final_thought_log:
                self.queue.put({"type": "thought", "content": f"Final thoughts:\n{final_thought_log}"})

# Define Flask routes
@app.route('/')
def index():
    return render_template('stylish_chat_interface.html')

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

        full_streamed_llm_output = ""

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

            # 3. Create streaming queue (synchronous)
            q = queue.Queue()
            streaming_handler = StreamingCallbackHandler(q)

            # 4. Get response from agent with streaming
            agent_response = agent.invoke(
                {
                    "input": f"""Current draft: {personal_statement_draft}\nContext: {retrieved_context}\nSimilar chats: {similar_chats}\nUser input: {user_input}"""
                },
                callbacks=[streaming_handler]
            )

            # 5. Stream the response
            llm_token_buffer = ""
            while True:
                event_data = q.get()
                if event_data is None:
                    continue
                if event_data.get("type") == "token":
                    token_content = event_data["content"]
                    full_streamed_llm_output += token_content
                    llm_token_buffer += token_content
                    if len(llm_token_buffer) >= 30 or any(p in token_content for p in ['.', '!', '?', '\n']):
                        if llm_token_buffer:
                            yield f"data: {json.dumps({'chunk': llm_token_buffer})}\n\n"
                            llm_token_buffer = ""
                elif event_data.get("type") == "thought":
                    yield f"data: {json.dumps({'thought': event_data['content']})}\n\n"
                elif event_data.get("type") == "llm_end":
                    if llm_token_buffer:
                        yield f"data: {json.dumps({'chunk': llm_token_buffer})}\n\n"
                        llm_token_buffer = ""
                    break

            if llm_token_buffer:
                yield f"data: {json.dumps({'chunk': llm_token_buffer})}\n\n"

            # 6. Parse final response
            final_agent_text = agent_response.get("output", "")

            parsed_chat_response = ""
            parsed_statement_suggestion = ""

            try:
                if "[CHAT]" in final_agent_text and "[STATEMENT_SUGGESTION]" in final_agent_text:
                    if final_agent_text.index("[CHAT]") < final_agent_text.index("[STATEMENT_SUGGESTION]"):
                        parts = final_agent_text.split("[CHAT]", 1)[1].split("[STATEMENT_SUGGESTION]", 1)
                        parsed_chat_response = parts[0].strip()
                        parsed_statement_suggestion = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        parts = final_agent_text.split("[STATEMENT_SUGGESTION]", 1)[1].split("[CHAT]", 1)
                        parsed_statement_suggestion = parts[0].strip()
                        parsed_chat_response = parts[1].strip() if len(parts) > 1 else ""
                elif "[CHAT]" in final_agent_text:
                    parsed_chat_response = final_agent_text.split("[CHAT]", 1)[1].strip()
                elif "[STATEMENT_SUGGESTION]" in final_agent_text:
                    parsed_statement_suggestion = final_agent_text.split("[STATEMENT_SUGGESTION]", 1)[1].strip()
                    if not parsed_chat_response:
                        parsed_chat_response = "I have a suggestion for your personal statement."
                else:
                    if final_agent_text.startswith("Final Answer:"):
                        parsed_chat_response = final_agent_text.split("Final Answer:", 1)[1].strip()
                    else:
                        parsed_chat_response = final_agent_text.strip()

                if not parsed_chat_response.strip() and parsed_statement_suggestion.strip() and not "[STATEMENT_SUGGESTION]" in final_agent_text:
                    parsed_chat_response = "I have a suggestion for your personal statement."
                elif not parsed_chat_response.strip() and not parsed_statement_suggestion.strip():
                    print(f"Warning: Agent returned empty or unparseable final text: '{final_agent_text}'")
                    parsed_chat_response = "I'm sorry, I didn't get a valid response. Please try again."

                chat_history_list.append({"role": "assistant", "content": parsed_chat_response})
                add_chat_message("assistant", parsed_chat_response)
                memory.save_context({"input": user_input}, {"output": parsed_chat_response})

                yield f"data: {json.dumps({'final': True, 'chat_response': parsed_chat_response, 'statement_suggestion': parsed_statement_suggestion, 'similar_chats': similar_chats, 'updated_draft': personal_statement_draft})}\n\n"

            except Exception as parse_error:
                print(f"Error parsing LLM final response: {parse_error}")
                print(f"Raw final_agent_text from agent.invoke(): {final_agent_text}")
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
