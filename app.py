from flask import Flask, request, jsonify, render_template, Response, stream_with_context, g
from langchain_config import retriever, memory, agent # Assuming these are correctly configured
import time # Needed for a brief pause
import logging
import sqlite3
import json
import queue
import re # Added for cleaning output
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

# Disable Flask's default request logging for cleaner output if not needed
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) # or logging.INFO for more details

app = Flask(__name__)

DB_FILE = "sqlite_db/app_state.db" # Define DB_FILE before use

# Ensure DB is initialized when app module is loaded
# This is a simple way for development; for production, a CLI command or more sophisticated setup might be used.
with app.app_context():
    # The init_db function already handles app_context internally for its operations.
    # To ensure it runs on app startup:
    conn = sqlite3.connect(DB_FILE) # Now DB_FILE is defined
    conn.row_factory = sqlite3.Row
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
    # Ensure the default draft row exists
    c.execute('INSERT OR IGNORE INTO personal_statement (id, draft) VALUES (1, "")')
    conn.commit()
    conn.close()
    print("Database initialized directly on app startup.")

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

# --- Langchain Callback Handler ---
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
        
        cleaned_thought_message = clean_agent_output(thought_message)
        self.queue.put({"type": "thought", "content": cleaned_thought_message})

    def on_tool_end(self, output: str, **kwargs) -> None:
        cleaned_output = clean_agent_output(output) # Clean the full output first
        max_len = 200
        tool_output_display = cleaned_output[:max_len] + "..." if len(cleaned_output) > max_len else cleaned_output
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
            if "Final Answer:" in final_thought_log:
                final_thought_log = final_thought_log.split("Final Answer:", 1)[0].strip()
            
            if final_thought_log: # Only send if there's content
                cleaned_final_thought = clean_agent_output(f"Final thoughts:\n{final_thought_log}")
                if cleaned_final_thought.startswith("Final thoughts:\n"): # Ensure prefix isn't stripped if content becomes empty
                    cleaned_final_thought_content = cleaned_final_thought[len("Final thoughts:\n"):].strip()
                    if cleaned_final_thought_content: # Only send if there's actual thought content left
                         self.queue.put({"type": "thought", "content": f"Final thoughts:\n{cleaned_final_thought_content}"})
                    # else: don't send "Final thoughts:" if there's nothing after cleaning
                elif cleaned_final_thought.strip(): # If "Final thoughts:" was stripped but other content remains
                    self.queue.put({"type": "thought", "content": cleaned_final_thought})

        # self.queue.put({"type": "agent_end"})


# --- Flask Routes ---

def _generate_agent_response_stream(user_input, current_personal_statement_draft):
    """
    Helper function to generate agent response stream.
    This function contains the core logic previously in the nested generate_response_stream.
    """
    full_streamed_llm_output = ""
    try:
        q = queue.Queue()
        streaming_handler = StreamingCallbackHandler(q)

        agent_input_payload = {
            "input": user_input,
        }
        if current_personal_statement_draft:
            agent_input_payload["input"] += f"\n\n[Current Draft Context]\n{current_personal_statement_draft}"

        # Run agent.invoke in a separate thread to allow the queue to be processed concurrently.
        # This is a common pattern for integrating synchronous Langchain calls with Flask streaming.
        # However, for simplicity and given the current structure, we'll keep it synchronous
        # within this generator function. The `stream_with_context` handles the yielding.
        # If true non-blocking for other requests is needed, agent.invoke would need to be
        # offloaded to a thread pool executor.

        agent_response_obj = agent.invoke(
            agent_input_payload,
            {"callbacks": [streaming_handler]}
        )

        llm_token_buffer = ""
        while True:
            try:
                event_data = q.get(timeout=10)
            except queue.Empty:
                print("Warning: Queue empty on get. Agent might have finished or stalled.")
                break

            if event_data.get("type") == "token":
                token_content = event_data["content"]
                full_streamed_llm_output += token_content
                llm_token_buffer += token_content
                if len(llm_token_buffer) >= 20 or any(p in token_content for p in ['.', '!', '?', '\n', ' ']):
                    if llm_token_buffer:
                        yield f"data: {json.dumps({'type': 'token_chunk', 'content': llm_token_buffer})}\n\n"
                        llm_token_buffer = ""
            elif event_data.get("type") == "thought":
                yield f"data: {json.dumps({'type': 'thought', 'content': event_data['content']})}\n\n"
            elif event_data.get("type") == "llm_end":
                if llm_token_buffer:
                    yield f"data: {json.dumps({'type': 'token_chunk', 'content': llm_token_buffer})}\n\n"
                    llm_token_buffer = ""
                break 
        
        if llm_token_buffer: # Final flush
            yield f"data: {json.dumps({'type': 'token_chunk', 'content': llm_token_buffer})}\n\n"

        agent_output_string = agent_response_obj.get("output", "")
        # Clean the final agent output string
        cleaned_agent_output_string = clean_agent_output(agent_output_string)
        
        parsed_chat_response = cleaned_agent_output_string if cleaned_agent_output_string else "Sorry, I couldn't generate a response."
        if not agent_output_string: # Log warning based on original output
             print(f"Warning: Agent returned an empty 'output'. Full response object: {agent_response_obj}")
        elif not cleaned_agent_output_string and agent_output_string: # Log if cleaning resulted in empty
             print(f"Warning: Agent output became empty after cleaning. Original: '{agent_output_string}'")


        parsed_statement_suggestion = "" # Assuming agent doesn't produce this by default. If it did, it should also be cleaned.

        # Save assistant message and update Langchain memory
        # This needs to be done carefully. If called from GET, user_input might not be in DB yet
        # if POST hasn't completed or if GET is called directly.
        # For now, we assume POST has added the user message.
        # A more robust solution might involve session management or passing a request ID.
        add_chat_message_to_db("assistant", parsed_chat_response)
        memory.save_context(
            {"input": user_input}, # This user_input is from the GET request's params
            {"output": parsed_chat_response}
        )

        final_data = {
            "type": "final_response",
            "chat_response": parsed_chat_response,
            "statement_suggestion": parsed_statement_suggestion,
            "current_draft_processed": current_personal_statement_draft
        }
        yield f"data: {json.dumps(final_data)}\n\n"

    except Exception as e:
        logging.exception("Error during AI processing stream (_generate_agent_response_stream)")
        error_response_content = "An error occurred while processing your request. Please check server logs."
        # Potentially add error to DB here as well, associated with the user_input from GET
        add_chat_message_to_db("assistant", error_response_content) # Or a more specific error message
        memory.save_context({"input": user_input}, {"output": error_response_content})
        yield f"data: {json.dumps({'type': 'error', 'content': error_response_content})}\n\n"
    finally:
        yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"


@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/chatbot.html') 
def chatbot_page_explicit():
    return render_template('chatbot.html')

@app.route('/document.html')
def document_page():
    return render_template('document.html')

@app.route('/audio.html')
def audio_page():
    return render_template('audio.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.json.get('message')
        current_personal_statement_draft = request.json.get('personal_statement_draft', '')

        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        add_chat_message_to_db("user", user_input)
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

##############################################################
# --- File Upload Configuration (Shared for OCR and STT) --- #
##############################################################
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

        # Call the refactored function in llm_ocr.py
        result_dict, status_code = llm_ocr.generate_report_from_document(
            file_path=temp_file_path,
            file_extension=file_extension,
            user_instructions=user_instructions,
            llm=llm, # Passed from app.py's imports
            prompt_template=main_chat_prompt_template, # Passed from app.py's imports
            logger=app.logger # Pass Flask app's logger
        )
        
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

#######################################
# --- New Audio Processing Routes --- #
#######################################

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

        Please provide a formal and concise summary of the key points from the transcript in Korean, suitable for a professional context.
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
