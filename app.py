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
    def __init__(self, queue, loop): # Add loop parameter
        self.queue = queue
        self.loop = loop # Store the loop

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Send as a dictionary to distinguish from other event types
        asyncio.run_coroutine_threadsafe(self.queue.put({"type": "token", "content": token}), self.loop)

    def on_llm_end(self, *args, **kwargs) -> None:
        # Signal end of LLM token stream
        asyncio.run_coroutine_threadsafe(self.queue.put({"type": "llm_end"}), self.loop)

    def on_agent_action(self, action, **kwargs) -> None:
        log_text = ""
        if hasattr(action, 'log') and action.log:
            log_text = action.log.strip()

        thought_message = ""

        if log_text.startswith("Thought:"):
            thought_message = log_text
            # Check if Action details are missing from the log text but an action is being taken
            if "Action:" not in log_text and "Tool:" not in log_text and action.tool: # "Tool:" check for some agent variants
                thought_message += f"\nAction: {action.tool}"
                # Ensure tool_input is a non-empty string before appending
                tool_input_str = str(action.tool_input) if action.tool_input is not None else ""
                if tool_input_str.strip(): # Only add Action Input if it's meaningful
                    thought_message += f"\nAction Input: {tool_input_str}"
        elif log_text: # Log exists but doesn't start with "Thought:" (e.g. LLM directly outputted action details)
            thought_message = f"Thought: (Interpreted action from LLM log)\n{log_text}"
            # If action.tool was parsed by Langchain and not explicitly in log_text, add it for clarity
            if action.tool and action.tool not in log_text:
                 tool_input_str = str(action.tool_input) if action.tool_input is not None else ""
                 thought_message += f"\n(Interpreted Action: {action.tool}, Input: {tool_input_str})"
        elif action.tool: # No log_text, but Langchain parsed a tool. Agent decided to use a tool.
            thought_message = f"Thought: Decided to use tool.\nAction: {action.tool}"
            tool_input_str = str(action.tool_input) if action.tool_input is not None else ""
            if tool_input_str.strip():
                thought_message += f"\nAction Input: {tool_input_str}"
        else: # Fallback: No log, no tool. This state should ideally not be reached in on_agent_action.
            thought_message = "Thought: (Agent is processing an action without detailed log)"

        asyncio.run_coroutine_threadsafe(self.queue.put({"type": "thought", "content": thought_message}), self.loop)

    def on_tool_end(self, output: str, **kwargs) -> None:
        # Send tool outputs as 'thought'
        # Limit output length if it's too long for display
        max_len = 200
        tool_output_display = output[:max_len] + "..." if len(output) > max_len else output
        asyncio.run_coroutine_threadsafe(self.queue.put({"type": "thought", "content": f"Tool Output: {tool_output_display}"}), self.loop)

    # Add other handlers if needed, e.g., on_chain_start, on_chain_end
    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        chain_name = serialized.get("name", serialized.get("id", ["Unknown Chain"])[-1])
        # Avoid overly verbose starts like the main agent executor
        if "AgentExecutor" not in chain_name and "RunnableSequence" not in chain_name and "LLMChain" not in chain_name: # Also avoid LLMChain itself
             asyncio.run_coroutine_threadsafe(self.queue.put({"type": "thought", "content": f"Starting sub-chain: {chain_name}"}), self.loop)

    def on_agent_finish(self, finish, **kwargs) -> None:
        # Stream the agent's final thoughts/log before the actual output is processed
        if hasattr(finish, 'log') and finish.log:
            final_thought_log = finish.log.strip()
            # Remove the "Final Answer:" part from this log if present, as the actual answer is handled separately
            if "Final Answer:" in final_thought_log:
                final_thought_log = final_thought_log.split("Final Answer:", 1)[0].strip()
            
            if final_thought_log: # Only send if there's content
                 asyncio.run_coroutine_threadsafe(self.queue.put({"type": "thought", "content": f"Final thoughts:\n{final_thought_log}"}), self.loop)
        # The actual 'output' from agent.invoke is handled after the streaming loop

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
        async def async_generate():
            # 1. Add user message to history and DB
            chat_history_list.append({"role": "user", "content": user_input})
            add_chat_message("user", user_input)
            memory.save_context({"input": user_input}, {"output": ""})

            full_streamed_llm_output = "" # Initialize full_streamed_llm_output

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
                current_loop = asyncio.get_event_loop() # Get the current loop
                streaming_handler = StreamingCallbackHandler(queue, current_loop) # Pass the loop

                # 4. Get response from agent with streaming
                agent_response = agent.invoke(
                    {
                        "input": f"""Current draft: {personal_statement_draft}
                        Context: {retrieved_context}
                        Similar chats: {similar_chats}
                        User input: {user_input}"""
                    },
                    callbacks=[streaming_handler]
                )

                # 5. Stream the response
                llm_token_buffer = ""
                while True:
                    event_data = await queue.get() # event_data is now a dictionary

                    if event_data is None: # Should not happen if on_llm_end sends a specific type
                        continue

                    if event_data.get("type") == "token":
                        token_content = event_data["content"]
                        full_streamed_llm_output += token_content # Accumulate token
                        llm_token_buffer += token_content
                        # Buffer tokens and send them in slightly larger chunks or on sentence breaks
                        if len(llm_token_buffer) >= 30 or any(p in token_content for p in ['.', '!', '?', '\n']):
                            if llm_token_buffer:
                                yield f"data: {json.dumps({'chunk': llm_token_buffer})}\n\n"
                                llm_token_buffer = ""
                    elif event_data.get("type") == "thought":
                        yield f"data: {json.dumps({'thought': event_data['content']})}\n\n"
                    elif event_data.get("type") == "llm_end":
                        # If there's any remaining buffered text, send it
                        if llm_token_buffer:
                            yield f"data: {json.dumps({'chunk': llm_token_buffer})}\n\n"
                            llm_token_buffer = ""
                        # This signals the end of the LLM's direct output stream.
                        # The agent's final processing and yielding of the 'final' message will follow.
                        break # Break from this loop to proceed to agent_response parsing

                # Ensure any final buffered tokens are sent before parsing agent_response
                if llm_token_buffer:
                    yield f"data: {json.dumps({'chunk': llm_token_buffer})}\n\n"

                # 6. Parse final response
                # Use full_streamed_llm_output as the primary source for parsing
                # Fallback to agent_response.get("output", "") if full_streamed_llm_output is empty
                parsing_source = full_streamed_llm_output
                if not parsing_source.strip():
                    print("Warning: full_streamed_llm_output was empty. Falling back to agent_response.get('output').")
                    parsing_source = agent_response.get("output", "")
                
                chat_response = "Sorry, I couldn't parse the response."
                statement_suggestion = ""

                try:
                    if "[CHAT]" in parsing_source:
                        parts = parsing_source.split("[CHAT]", 1)[1].split("[STATEMENT_SUGGESTION]", 1)
                        chat_response = parts[0].strip()
                        if len(parts) > 1:
                            statement_suggestion = parts[1].strip()
                    elif "[STATEMENT_SUGGESTION]" in parsing_source:
                        statement_suggestion = parsing_source.split("[STATEMENT_SUGGESTION]", 1)[1].strip()
                        chat_response = "I have a suggestion for your personal statement."
                    else:
                        # Ensure "Final Answer:" prefix is removed if present in the direct output
                        if parsing_source.startswith("Final Answer:"):
                            chat_response = parsing_source.split("Final Answer:", 1)[1].strip()
                        else:
                            chat_response = parsing_source.strip()
                        # If after stripping, chat_response is empty and agent_response.get("output") has content,
                        # it might be that the agent's final answer was not part of the token stream (e.g. direct tool output as final answer)
                        if not chat_response and agent_response.get("output", ""):
                            agent_output_fallback = agent_response.get("output", "")
                            if agent_output_fallback.startswith("Final Answer:"):
                                chat_response = agent_output_fallback.split("Final Answer:", 1)[1].strip()
                            else:
                                chat_response = agent_output_fallback.strip()


                    # 7. Add agent response to history and DB
                    chat_history_list.append({"role": "assistant", "content": chat_response})
                    add_chat_message("assistant", chat_response)
                    memory.save_context({"input": user_input}, {"output": chat_response})

                    # 8. Send final response
                    yield f"data: {json.dumps({'final': True, 'chat_response': chat_response, 'statement_suggestion': statement_suggestion, 'similar_chats': similar_chats, 'updated_draft': personal_statement_draft})}\n\n"

                except Exception as parse_error:
                    print(f"Error parsing LLM response: {parse_error}")
                    print(f"Raw parsing source (full_streamed_llm_output or fallback): {parsing_source}")
                    if parsing_source != agent_response.get("output", ""):
                        print(f"Agent response output for comparison: {agent_response.get('output', '')}")
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

        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # async_generate() returns an async generator.
            # We need a coroutine to consume it for run_until_complete.
            async def collect_chunks_from_async_gen():
                all_chunks = []
                # async_generate is defined in the outer scope of this nested function
                async for chunk_data in async_generate(): 
                    all_chunks.append(chunk_data)
                return all_chunks

            all_yielded_chunks = loop.run_until_complete(collect_chunks_from_async_gen())
            for chunk in all_yielded_chunks:
                yield chunk
        finally:
            loop.close()
            asyncio.set_event_loop(None) # Reset the event loop for the current thread

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
