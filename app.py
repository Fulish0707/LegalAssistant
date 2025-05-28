from flask import Flask, request, jsonify, render_template, Response, stream_with_context, g
from langchain_config import retriever, memory, agent # Assuming these are correctly configured
import time # Needed for a brief pause
import logging
import sqlite3
import json
import queue
from langchain.callbacks.base import BaseCallbackHandler
import os
import sys
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv
import requests
import sounddevice as sd
import soundfile as sf

# Assuming llm_ocr.py is in the same directory or accessible via PYTHONPATH
import llm_ocr
import llm_stt # Added for audio processing
# Assuming the main prompt template and llm model can be accessed
# This might require adjustments if langchain_config doesn't expose llm directly
from llm_prompt import prompt as main_chat_prompt_template
from langchain_config import llm # TRYING TO IMPORT LLM DIRECTLY

# --- FAKE Langchain components REMOVED ---
# Real components are now imported from langchain_config

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
    print("Attempting to initialize database...")
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
        print("Database initialized successfully.")
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
        chain_name = "Unknown Chain"
        if serialized: # Check if serialized is not None
            chain_name = serialized.get("name", serialized.get("id", ["Unknown Chain"])[-1])
        
        # Only log certain chain types to avoid too much noise, or adjust as needed
        if chain_name and "AgentExecutor" not in chain_name and "RunnableSequence" not in chain_name and "LLMChain" not in chain_name:
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
    return render_template('chatbot.html')

@app.route('/chatbot.html') # Explicit route for chatbot.html if users type it
def chatbot_page_explicit():
    return render_template('chatbot.html')

@app.route('/document.html')
def document_page():
    return render_template('document.html')

@app.route('/audio.html')
def audio_page():
    return render_template('audio.html')

# Note: No route for video.html yet as it doesn't exist.
# The link in chatbot.html for video should be updated.

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

        try:
            # 1. Retrieve context (REMOVED - Agent uses its own RAG tool)
            # 2. Fetch similar past conversations (REMOVED - Agent's memory handles history. If needed explicitly, prompt engineering required)
            # 3. Create streaming queue
            q = queue.Queue()
            streaming_handler = StreamingCallbackHandler(q)

            # 4. Prepare agent input
            # For conversational-react-description, "input" is the main key.
            # current_draft could be passed if the agent's prompt is designed for it.
            agent_input_payload = {
                "input": user_input,
                # "current_draft": current_personal_statement_draft, # Keep if prompt handles it
                # Langchain memory (ConversationBufferMemory) is part of the agent from langchain_config
            }
            
            # Add current_draft to input if it's not empty, assuming prompt might use it.
            # This is a placeholder; ideally, the prompt is explicitly designed for this.
            if current_personal_statement_draft:
                agent_input_payload["input"] += f"\n\n[Current Draft Context]\n{current_personal_statement_draft}"


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
            # IMPORTANT: The real agent is expected to return a dict like {"output": "response_string"}
            # The "output" field contains the direct string response.
            agent_output_string = agent_response_obj.get("output", "")

            parsed_chat_response = agent_output_string
            # Statement suggestion logic is removed as the real agent might not produce it by default.
            # If suggestions are needed, the agent's prompt (from llm_prompt.py) must be designed for it.
            parsed_statement_suggestion = "" # Default to no suggestion

            if not parsed_chat_response:
                parsed_chat_response = "Sorry, I couldn't generate a response."
                print(f"Warning: Agent returned an empty 'output'. Full response object: {agent_response_obj}")

            # 8. Save assistant message and update Langchain memory
            # The memory object used here is the one imported from langchain_config,
            # which is also used by the agent.
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


def application_setup():
    """Encapsulates setup that should happen in the worker process."""
    print("--- Worker Process Setup ---")
    print("Initializing DB...")
    init_db()
    # Add any other one-time initializations for the worker here
    print("--- Worker Process Setup Complete ---")

# --- File Upload Configuration (Shared for OCR and STT) ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS_OCR = {'png', 'jpg', 'jpeg', 'txt', 'pdf', 'docx'}
ALLOWED_EXTENSIONS_AUDIO = {'wav', 'mp3', 'ogg', 'flac', 'm4a', 'webm'} # Common audio formats

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

# --- OCR Processing Route (Existing) ---
@app.route('/process_document_ocr', methods=['POST'])
def process_document_ocr():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    user_instructions = request.form.get('user_instructions', '')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_OCR):
        filename = secure_filename(file.filename)
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], suffix=os.path.splitext(filename)[1])
        temp_file_path = temp_file.name
        file.save(temp_file_path)
        temp_file.close() # Close the file handle so OCR can open it if needed

        extracted_text = ""
        file_extension = filename.rsplit('.', 1)[1].lower()

        try:
            if file_extension in {'png', 'jpg', 'jpeg'}:
                extracted_text = llm_ocr.detect_text(temp_file_path)
            elif file_extension == 'txt':
                with open(temp_file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            elif file_extension in {'pdf', 'docx'}:
                # Placeholder: PDF/DOCX processing requires additional libraries
                # For now, we'll return a message.
                # In a real scenario, you'd integrate PyPDF2, python-docx, etc.
                # and potentially convert to images for OCR if direct text extraction is poor.
                os.remove(temp_file_path) # Clean up temp file
                return jsonify({"error": f".{file_extension} processing not yet fully implemented. Requires dedicated libraries."}), 501

            if not extracted_text.strip():
                os.remove(temp_file_path) # Clean up temp file
                return jsonify({"error": "Could not extract text from the document or the document is empty."}), 400

            # Now, use the LLM to generate the report HTML
            # We use the main_chat_prompt_template and format it for a report generation task
            # The llm_prompt.py has specific instructions for [REPORT_SUGGESTION]
            
            # Construct a focused input for the LLM based on the OCR'd text and user instructions
            # This is a simplified approach. The main prompt is complex.
            # We need to simulate a "human" turn that triggers the report generation logic within the prompt.
            report_generation_input = f"""Please generate a report based on the following document content and instructions.
The document content is:
--- DOCUMENT CONTENT START ---
{extracted_text}
--- DOCUMENT CONTENT END ---

User instructions for the report:
{user_instructions if user_instructions else "No specific instructions provided. Please generate a comprehensive summary and analysis."}

The report should be in HTML format. If relevant, include diagrams or images as described in the system prompt (e.g., using HTML/CSS for simple diagrams, or base64 encoded images if you can generate them).
Focus on clarity, logical structure, and engagement, suitable for bosses or business clients as per the system prompt's [REPORT_SUGGESTION] guidelines.
"""
            
            # We need to invoke the LLM with the main system prompt and this constructed input.
            # The main_chat_prompt_template is a ChatPromptTemplate.
            # We'll use the 'llm' instance imported from langchain_config.
            
            # Create a message sequence for the ChatPromptTemplate
            # The system prompt is already part of main_chat_prompt_template
            # We add a human message with our constructed input.
            # No chat_history is explicitly passed here, assuming a fresh interaction for report generation.
            formatted_prompt_messages = main_chat_prompt_template.format_messages(
                current_document_draft="", # Not relevant here or handled by user_instructions
                retrieved_context="", # No RAG context for this specific flow yet
                chat_history=[], # Fresh interaction for report generation
                input=report_generation_input 
            )
            
            # Invoke the LLM model directly with the formatted messages
            ai_response = llm.invoke(formatted_prompt_messages)
            
            # The response from the LLM (ai_response) will be an AIMessage object.
            # Its content should be the generated HTML report.
            report_html = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)

            # Basic check if HTML was likely generated (very naive)
            if not ("<html" in report_html.lower() or "<body" in report_html.lower() or "<div" in report_html.lower()):
                 # If not, wrap it in basic HTML structure
                 report_html = f"<html><head><title>Generated Report</title></head><body><pre>{report_html}</pre></body></html>"


            os.remove(temp_file_path) # Clean up temp file
            return jsonify({"report_html": report_html})

        except Exception as e:
            logging.exception("Error processing document for OCR report")
            if os.path.exists(temp_file_path): # Ensure cleanup even on error
                os.remove(temp_file_path)
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
        finally:
            # Ensure temp file is deleted if it still exists for some reason
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e_clean:
                    logging.error(f"Error cleaning up temp file {temp_file_path}: {e_clean}")
    else:
        return jsonify({"error": "File type not allowed for OCR"}), 400

# --- New Audio Processing Routes ---
def record_audio_to_file(filename="recorded_audio.wav", duration=5, samplerate=16000):
    """
    Records audio from the microphone and saves it to a WAV file.
    """
    print(f"Recording for {duration} seconds... Speak now!")
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        sf.write(filename, recording, samplerate)
        print(f"Recording saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error during audio recording: {e}")
        return None

# Load environment variables
load_dotenv()

# Get Naver credentials
NAVER_CLIENT_ID = os.getenv('client_id')
NAVER_CLIENT_SECRET = os.getenv('client_secret')

def process_audio_common(audio_file_path):
    """
    Transcribes and summarizes an audio file.
    Returns a dictionary with 'transcript', 'summary', or 'error'.
    """
    transcript = ""
    summary = ""
    
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        app.logger.error("Naver Cloud Platform credentials not found in .env file")
        # Ensure this returns a structure the calling function can check for "error"
        return {"error": "Naver Cloud Platform credentials not configured on server."}

    # 1. Transcribe audio using Naver STT (from llm_stt.py logic)
    lang = "Kor"
    url = f"https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang={lang}"
    try:
        with open(audio_file_path, 'rb') as audio_data_file:
            headers = {
                "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
                "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
                "Content-Type": "application/octet-stream" # Naver STT expects this for many formats
            }
            response = requests.post(url, data=audio_data_file, headers=headers)
            rescode = response.status_code
            
            if rescode == 200:
                result_json = response.json()
                transcript = result_json.get('text', '')
                app.logger.info(f"Naver STT successful: {transcript[:100]}...")
            else:
                app.logger.error(f"Naver STT API Error (HTTP {rescode}): {response.text}")
                return {"error": f"Audio transcription failed with Naver STT (HTTP {rescode}): {response.text}", "transcript": ""}
    except Exception as e:
        app.logger.error(f"Exception during Naver STT call: {str(e)}")
        return {"error": f"Exception during transcription: {str(e)}", "transcript": ""}

    if not transcript.strip():
        app.logger.info("Transcription is empty or audio was silent.")
        return {"summary": "Audio appears to be silent or could not be transcribed.", "transcript": transcript}

    # 2. Summarize transcript using LLM
    try:
        summarization_input_text = f"""Please summarize the following audio transcript concisely:
--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---

Provide a brief summary of the key points.
"""
        formatted_prompt_messages = main_chat_prompt_template.format_messages(
            current_document_draft="", 
            retrieved_context="", 
            chat_history=[], 
            input=summarization_input_text
        )
        
        ai_response = llm.invoke(formatted_prompt_messages)
        summary = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
        app.logger.info(f"LLM summarization successful: {summary[:100]}...")
        return {"transcript": transcript, "summary": summary}
    except Exception as e:
        app.logger.error(f"Exception during LLM summarization: {str(e)}")
        return {"error": f"Error during summarization: {str(e)}", "transcript": transcript, "summary": ""}

@app.route('/transcribe_and_summarize_audio_file', methods=['POST'])
def transcribe_and_summarize_audio_file_route():
    app.logger.info("Received request for /transcribe_and_summarize_audio_file")
    if 'audio_file' not in request.files:
        app.logger.warning("No 'audio_file' part in request.files")
        return jsonify({"error": "No audio_file part"}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        app.logger.warning("No selected audio file (filename is empty)")
        return jsonify({"error": "No selected audio file"}), 400

    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_AUDIO):
        filename = secure_filename(file.filename)
        app.logger.info(f"Processing uploaded file: {filename}")
        temp_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], suffix=os.path.splitext(filename)[1]) as temp_file_obj:
                temp_file_path = temp_file_obj.name
                file.save(temp_file_obj)
            # temp_file_obj is now closed, and file is saved at temp_file_path

            result = process_audio_common(temp_file_path)
            
            # Check error from process_audio_common (which now returns a dict)
            if result.get("error"):
                status_code = 500
                if "credentials not configured" in result["error"]:
                    status_code = 400 # Or 503 Service Unavailable
                elif "transcription failed" in result["error"].lower() or "naver stt api error" in result["error"].lower():
                    status_code = 502 # Bad Gateway from STT provider
                return jsonify(result), status_code
            
            return jsonify(result)
            
        except Exception as e:
            app.logger.error(f"Exception in /transcribe_and_summarize_audio_file: {str(e)}", exc_info=True)
            return jsonify({"error": f"Server error during file processing: {str(e)}"}), 500
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    app.logger.info(f"Cleaned up temp file: {temp_file_path}")
                except Exception as e_clean:
                    app.logger.error(f"Error cleaning up temp file {temp_file_path}: {e_clean}")
    else:
        app.logger.warning(f"Audio file type not allowed: {file.filename if file else 'Unknown'}")
        return jsonify({"error": "Audio file type not allowed"}), 400


@app.route('/transcribe_and_summarize_audio_blob', methods=['POST'])
def transcribe_and_summarize_audio_blob_route():
    app.logger.info(f"Received request for /transcribe_and_summarize_audio_blob. Headers: {request.headers}, Content-Type: {request.content_type}")
    app.logger.info(f"Request.files: {list(request.files.keys())}")

    if 'audio_blob' not in request.files:
        app.logger.warning("No 'audio_blob' part in request.files for blob route.")
        return jsonify({"error": "No audio_blob part"}), 400
    
    audio_blob = request.files['audio_blob']
    
    if not audio_blob.filename: # Check if filename is present and not empty
        app.logger.warning("Audio blob has no filename or filename is empty.")
        return jsonify({"error": "Audio blob has no filename"}), 400

    filename = secure_filename(audio_blob.filename)
    app.logger.info(f"Processing audio blob: {filename}")
    temp_file_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], suffix=os.path.splitext(filename)[1]) as temp_file_obj:
            temp_file_path = temp_file_obj.name
            audio_blob.save(temp_file_obj)
        # temp_file_obj is now closed

        result = process_audio_common(temp_file_path)
        
        if result.get("error"):
            status_code = 500
            if "credentials not configured" in result["error"]:
                status_code = 400
            elif "transcription failed" in result["error"].lower() or "naver stt api error" in result["error"].lower():
                status_code = 502
            return jsonify(result), status_code
        
        return jsonify(result)
            
    except Exception as e:
        app.logger.error(f"Exception in /transcribe_and_summarize_audio_blob: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error during blob processing: {str(e)}"}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                app.logger.info(f"Cleaned up temp blob file: {temp_file_path}")
            except Exception as e_clean:
                app.logger.error(f"Error cleaning up temp blob file {temp_file_path}: {e_clean}")

# Removed the /record_and_transcribe route as it's not used by the current frontend
# and llm_stt.py's record_audio_to_file is for fixed duration.

if __name__ == '__main__':
    init_db() # Initialize the database
    app.run(host="0.0.0.0", port=5005, debug=True, use_reloader=False)
