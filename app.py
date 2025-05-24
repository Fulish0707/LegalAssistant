from flask import Flask, request, jsonify, render_template, Response, stream_with_context, g
# from langchain_config import retriever, memory, agent # Assuming these are correctly configured
import time # Needed for a brief pause
import logging
import sqlite3
import json
import queue
from langchain.callbacks.base import BaseCallbackHandler
import os

# --- FAKE Langchain components for standalone testing ---
# In your actual setup, you'll use: from langchain_config import retriever, memory, agent
class FakeRetriever:
    def invoke(self, query):
        print(f"[FakeRetriever] Invoked with query: {query}")
        return [{"page_content": f"Document related to '{query}' 1"}, {"page_content": "General advice document"}]

class FakeMemory:
    def __init__(self):
        self.chat_memory = {"messages": []} # Simplified in-memory store for the fake
        print("[FakeMemory] Initialized")

    def save_context(self, inputs, outputs):
        print(f"[FakeMemory] Saving context: inputs={inputs}, outputs={outputs}")
        self.chat_memory["messages"].append({"input": inputs.get("input"), "output": outputs.get("output")})

    def load_memory_variables(self, inputs=None):
        # This is a simplified representation. Real memory objects might format history differently.
        history = "\n".join([f"Human: {msg['input']}\nAI: {msg['output']}" for msg in self.chat_memory["messages"]])
        return {"history": history}

class FakeAgent:
    def invoke(self, inputs, callbacks=None):
        print(f"[FakeAgent] Invoked with inputs: {inputs}")
        user_input_detail = inputs.get("input", "") # The full input string
        
        # Simulate streaming for callbacks
        if callbacks:
            for handler in callbacks:
                if hasattr(handler, 'on_agent_action'):
                    # Simulate a thought process
                    handler.on_agent_action(
                        type('obj', (object,), {'log': 'Thought: I need to analyze the user input and draft.', 'tool': 'analysis_tool', 'tool_input': 'details'})
                    )
                    time.sleep(0.1) # simulate work
                    handler.on_tool_end(output="Analysis complete, draft seems okay.")
                    time.sleep(0.1) # simulate work
                if hasattr(handler, 'on_llm_new_token'):
                    response_chat = "This is a streamed chat response from the fake agent. "
                    response_suggestion = "Consider adding more examples to your draft."
                    
                    # Simulate streaming the structured response
                    # For demonstration, we'll stream it as if it's a JSON string
                    # In reality, the agent would produce this JSON as its final output.
                    
                    # Stream chat part
                    for char in "Okay, I've processed that. ":
                        handler.on_llm_new_token(char)
                        time.sleep(0.01)
                    
                    # Stream suggestion part
                    for char in "How about this: add more details about your leadership experience? ":
                        handler.on_llm_new_token(char)
                        time.sleep(0.01)
                        
                if hasattr(handler, 'on_llm_end'):
                    handler.on_llm_end()

        # Simulate the agent's structured output (what it *would* produce if prompted for JSON)
        # The actual prompt engineering for the real agent to produce this is key.
        structured_response = {
            "chat_response": f"Okay, I've processed your input regarding: '{user_input_detail[:30]}...'.",
            "statement_suggestion": "Based on your input, consider elaborating on your future goals in the personal statement."
        }
        # For the purpose of this fake agent, we'll return this directly.
        # The streaming part above simulates how tokens would arrive. The `invoke` should return the *final* complete output.
        return {"output": json.dumps(structured_response)} # Agent output is now a JSON string

retriever = FakeRetriever()
memory = FakeMemory() # Re-initialize for each run if it's stateful and not persisted
agent = FakeAgent()
# --- END FAKE Langchain components ---


# Disable Flask's default request logging for cleaner output if not needed
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) # or logging.INFO for more details

app = Flask(__name__)

DB_FILE = "sqlite_db/app_state.db"
# NO MORE GLOBAL DRAFT OR CHAT HISTORY LIST
# personal_statement_draft = "" # REMOVED
# chat_history_list = [] # REMOVED

# --- Database Setup and Utility Functions ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_FILE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context(): # Operations outside a request need an app context for get_db
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
        c.execute('INSERT OR IGNORE INTO personal_statement (id, draft) VALUES (1, "")')
        conn.commit()
        # No need to close here, teardown_appcontext will handle it if get_db was called
        # For standalone init, explicitly closing is safer if not within app context management
        # but since we used `with app.app_context()`, `close_connection` will be triggered.

def get_personal_statement_from_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT draft FROM personal_statement WHERE id=1')
    row = c.fetchone()
    return row['draft'] if row else ""

def set_personal_statement_in_db(draft):
    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE personal_statement SET draft=? WHERE id=1', (draft,))
    conn.commit()

def get_chat_history_from_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT role, content FROM chat_history ORDER BY id ASC')
    rows = c.fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]

def add_chat_message_to_db(role, content):
    conn = get_db()
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (role, content) VALUES (?, ?)', (role, content))
    conn.commit()

def clear_chat_history_in_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('DELETE FROM chat_history')
    conn.commit()
    # Note: You might want to clear the Langchain 'memory' as well if it's persistent
    # For the FakeMemory, we could add a clear method:
    if hasattr(memory, 'chat_memory') and 'messages' in memory.chat_memory:
        memory.chat_memory['messages'] = []
    elif hasattr(memory, 'clear'): # For more generic Langchain memory objects
        memory.clear()
    print("Chat history cleared from DB and in-memory Langchain memory (if applicable).")


# --- Langchain Callback Handler (no changes needed for backend logic refactor) ---
class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put({"type": "token", "content": token})

    def on_llm_end(self, *args, **kwargs) -> None:
        self.queue.put({"type": "llm_end"}) # Signal end of LLM token stream

    def on_agent_action(self, action, **kwargs) -> None:
        log_text = ""
        if hasattr(action, 'log') and action.log:
            log_text = action.log.strip()

        thought_message = ""
        # Simplified thought message generation for brevity, original logic was fine
        if log_text:
            thought_message = f"Thought:\n{log_text}"
        elif action.tool:
            thought_message = f"Thought: Decided to use tool.\nAction: {action.tool}"
            tool_input_str = str(action.tool_input) if action.tool_input is not None else ""
            if tool_input_str.strip():
                thought_message += f"\nAction Input: {tool_input_str}"
        else:
            thought_message = "Thought: (Agent is processing an action)"
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
            # Attempt to clean up "Final Answer:" if present in the log before the actual output
            if "Final Answer:" in final_thought_log:
                 # This assumes the actual final answer is in `finish.return_values['output']`
                 # and the log might contain preliminary final thoughts.
                final_thought_log = final_thought_log.split("Final Answer:", 1)[0].strip()
            if final_thought_log: # Only send if there's content
                self.queue.put({"type": "thought", "content": f"Final thoughts:\n{final_thought_log}"})
        # self.queue.put({"type": "agent_end"}) # Could add this if frontend needs to know agent finished before full parsing


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('stylish_chat_interface.html') # Ensure this HTML file exists

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    # The frontend must send the current state of the draft
    current_personal_statement_draft = request.json.get('personal_statement_draft', '')

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Add user message to DB
    add_chat_message_to_db("user", user_input)
    # The Langchain memory object handles its own state.
    # We save context after getting the AI's response.

    def generate_response_stream():
        full_streamed_llm_output = ""
        # Langchain memory might need to be loaded/prepared if it's not globally persistent
        # For this example, FakeMemory is simple. Real memory (e.g. ConversationBufferWindowMemory)
        # might need loading from DB or other store if not persisting across requests.
        # memory.chat_memory.add_user_message(user_input) # If using specific methods

        try:
            # 1. Retrieve context (if enabled)
            retrieved_context = ""
            if retriever: # Assuming retriever is configured
                retrieved_docs = retriever.invoke(user_input)
                retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])

            # 2. Fetch similar past conversations (optional, for agent context)
            conn = get_db()
            c = conn.cursor()
            c.execute('''
                SELECT role, content FROM chat_history
                WHERE id != (SELECT MAX(id) FROM chat_history) AND content LIKE ?
                ORDER BY timestamp DESC LIMIT 3
            ''', (f'%{user_input}%',)) # Exclude the message just added
            similar_chats_raw = c.fetchall()
            similar_chats_for_agent = [{"role": row["role"], "content": row["content"]} for row in similar_chats_raw]
            # Convert to a string format suitable for the agent prompt
            similar_chats_str = json.dumps(similar_chats_for_agent)


            # 3. Create streaming queue
            q = queue.Queue()
            streaming_handler = StreamingCallbackHandler(q)

            # 4. Prepare agent input
            # The agent's prompt needs to be designed to use these effectively
            agent_input_payload = {
                "input": user_input, # Clearer to just pass user_input directly
                "current_draft": current_personal_statement_draft,
                "retrieved_context": retrieved_context,
                "similar_historical_chats": similar_chats_str,
                # Langchain memory variables are often implicitly added by the AgentExecutor
                # or you might pass `memory.load_memory_variables({})` if needed explicitly.
            }

            # 5. Invoke agent (this now runs in the main thread of the request handler)
            # To make this non-blocking for other requests, the agent.invoke itself would need to be async
            # or run in a separate thread pool, but for streaming per *single* request, this is fine.
            agent_response_obj = agent.invoke(
                agent_input_payload,
                {"callbacks": [streaming_handler]} # Correct way to pass callbacks
            )

            # 6. Stream response parts (tokens and thoughts)
            llm_token_buffer = ""
            while True:
                try:
                    event_data = q.get(timeout=10) # Add timeout to prevent indefinite blocking
                except queue.Empty:
                    # This can happen if the agent finishes but on_llm_end wasn't the last thing, or if there's a stall
                    print("Warning: Queue empty on get. Agent might have finished or stalled.")
                    break # Exit loop if queue is empty after timeout

                if event_data.get("type") == "token":
                    token_content = event_data["content"]
                    full_streamed_llm_output += token_content
                    llm_token_buffer += token_content
                    # Adjust buffering strategy as needed
                    if len(llm_token_buffer) >= 20 or any(p in token_content for p in ['.', '!', '?', '\n', ' ']):
                        if llm_token_buffer:
                            yield f"data: {json.dumps({'type': 'token_chunk', 'content': llm_token_buffer})}\n\n"
                            llm_token_buffer = ""
                elif event_data.get("type") == "thought":
                    yield f"data: {json.dumps({'type': 'thought', 'content': event_data['content']})}\n\n"
                elif event_data.get("type") == "llm_end":
                    if llm_token_buffer: # Flush any remaining tokens
                        yield f"data: {json.dumps({'type': 'token_chunk', 'content': llm_token_buffer})}\n\n"
                        llm_token_buffer = ""
                    # This signals the end of raw token streaming from LLM, agent processing continues
                    # We break out of this loop once the *agent* is fully done, not just the LLM.
                    # The agent's final output is in agent_response_obj.
                    break # Break from the queue polling loop as LLM part is done

            # Ensure any final buffered tokens are sent if llm_end wasn't hit or loop broke early
            if llm_token_buffer:
                yield f"data: {json.dumps({'type': 'token_chunk', 'content': llm_token_buffer})}\n\n"


            # 7. Process final agent output
            # IMPORTANT: The agent's prompt MUST be updated to return a JSON string.
            # Example expected JSON: {"chat_response": "...", "statement_suggestion": "..."}
            raw_agent_output_str = agent_response_obj.get("output", "{}")

            parsed_chat_response = "Sorry, I encountered an issue processing the response."
            parsed_statement_suggestion = ""

            try:
                # The agent is now expected to output a JSON string
                parsed_output = json.loads(raw_agent_output_str)
                parsed_chat_response = parsed_output.get("chat_response", "No chat response provided.")
                parsed_statement_suggestion = parsed_output.get("statement_suggestion", "") # Can be empty

                if not parsed_chat_response and not parsed_statement_suggestion:
                    # This case indicates the agent returned valid JSON but with empty/missing fields
                    parsed_chat_response = "I don't have a specific chat reply, but I've processed your input."
                    print(f"Warning: Agent returned JSON with empty/missing 'chat_response' and 'statement_suggestion'. Raw: {raw_agent_output_str}")


            except json.JSONDecodeError:
                # Fallback if the agent DID NOT return valid JSON
                # This means the agent's prompt needs fixing.
                print(f"ERROR: Agent output was not valid JSON: {raw_agent_output_str}")
                # Attempt to use the raw output as chat response if JSON parsing fails
                parsed_chat_response = raw_agent_output_str if raw_agent_output_str.strip() else parsed_chat_response
                # No suggestion if parsing failed
                parsed_statement_suggestion = "Could not parse suggestion."


            # 8. Save assistant message and update Langchain memory
            add_chat_message_to_db("assistant", parsed_chat_response)
            memory.save_context(
                {"input": user_input}, # What the user sent
                {"output": parsed_chat_response} # What the AI responded in chat
            )

            # 9. Send final structured message
            final_data = {
                "type": "final_response",
                "chat_response": parsed_chat_response,
                "statement_suggestion": parsed_statement_suggestion,
                # "similar_chats_retrieved_for_agent": similar_chats_for_agent, # For debugging on client
                "current_draft_processed": current_personal_statement_draft # Confirm which draft was used
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            logging.exception("Error during AI processing stream") # Log full traceback
            error_response_content = "An error occurred while processing your request. Please check server logs."
            # Ensure assistant message for error is added to DB
            # Check if user message was already added; if so, add corresponding error for assistant
            # This logic might need refinement based on where the error occurs.
            # If error is very early, user_input might not be in DB yet.
            # For now, assume user message is added, and this is the assistant's error response.
            add_chat_message_to_db("assistant", error_response_content)
            memory.save_context({"input": user_input}, {"output": error_response_content})
            yield f"data: {json.dumps({'type': 'error', 'content': error_response_content})}\n\n"
        finally:
            # Signal the absolute end of the stream to the client
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"


    return Response(stream_with_context(generate_response_stream()), mimetype='text/event-stream')


@app.route('/get_initial_state', methods=['GET'])
def get_initial_state():
    # Fetches from DB, so no change needed due to global var removal
    return jsonify({
        "personal_statement_draft": get_personal_statement_from_db(),
        "chat_history": get_chat_history_from_db()
    })

@app.route('/update_statement', methods=['POST'])
def update_statement():
    new_draft = request.json.get('personal_statement_draft')
    if new_draft is None: # Check for None, as empty string is a valid draft
        return jsonify({"error": "No personal_statement_draft provided"}), 400
    
    try:
        set_personal_statement_in_db(new_draft)
        return jsonify({"message": "Personal statement updated successfully."}), 200
    except Exception as e:
        logging.exception("Error updating personal statement in DB")
        return jsonify({"error": "Failed to update personal statement."}), 500

@app.route('/get_similar_chats', methods=['POST']) # This seems more like a debug/utility endpoint
def get_similar_chats_route(): # Renamed to avoid conflict with internal var
    query = request.json.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    conn = get_db()
    c = conn.cursor()
    # Example: find chats containing the query text, excluding very recent ones to avoid self-match
    c.execute('''
        SELECT role, content, timestamp
        FROM chat_history
        WHERE content LIKE ? AND timestamp < datetime('now', '-5 seconds')
        ORDER BY timestamp DESC
        LIMIT 5
    ''', (f'%{query}%',))
    similar_chats = [{"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]}
                     for row in c.fetchall()]

    return jsonify({"similar_chats": similar_chats})


@app.route('/clear_chat_history_endpoint', methods=['POST']) # Added specific endpoint name
def clear_chat_history_endpoint():
    try:
        clear_chat_history_in_db() # This now also clears Langchain FakeMemory
        return jsonify({"message": "Chat history cleared successfully."}), 200
    except Exception as e:
        logging.error(f"Error clearing chat history: {e}")
        return jsonify({"error": "Failed to clear chat history."}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # init_db() should ideally be idempotent, but running it only once is cleaner.
    # The FakeMemory initialization is a bit trickier if you want only one instance globally
    # when using the reloader. For true singletons, other patterns are needed, but for
    # development, this often suffices to prevent double console output.

    # This check ensures these run only in the main process started by the reloader
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        print("Initializing DB (main process)...")
        init_db()
        # If FakeMemory and other fakes were meant to be initialized only once globally,
        # this is tricky with the reloader, as each process gets its own memory space.
        # The current FakeMemory will be re-initialized in the reloaded worker process.
        # This is usually fine for development fakes.
        # If these were real resources (e.g., a database connection pool),
        # you'd initialize them lazily or within the app context.

    print("Starting Flask server (Flask `app.run` call)...")
    app.run(host="0.0.0.0", port=5005, debug=True, use_reloader=True)