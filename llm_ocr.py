import pytesseract
from PIL import Image
import os
import pypdf # Added for PDF processing
import re # Added for improved post-processing
import docx # Added for DOCX processing

DOCUMENT_ANALYSIS_MODES = {
    "summary": {
        "label": "Document Summary",
        "instruction": """Create a concise but useful document summary.
Include:
- Executive summary
- Key points
- Important details, numbers, dates, or claims
- Suggested follow-up questions""",
    },
    "report": {
        "label": "Professional Report",
        "instruction": """Create a polished professional report for a boss or business client.
Include:
- Clear title
- Executive summary
- Main findings
- Supporting evidence
- Recommendations or next steps""",
    },
    "actions": {
        "label": "Action Items",
        "instruction": """Extract operationally useful action items from the document.
Include:
- Tasks
- Owners if mentioned
- Deadlines if mentioned
- Risks or dependencies
- Open questions""",
    },
    "qa": {
        "label": "Document Q&A",
        "instruction": """Answer the user's question using only the document content.
If the document does not contain enough evidence, say what is missing and avoid guessing.""",
    },
}

def detect_text(image_path):
    """
    Extracts text from an image file using Tesseract OCR.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The extracted text, or an error message string if extraction fails.
    """
    if not os.path.exists(image_path):
        # This case should ideally be caught before calling, but good as a safeguard.
        # Returning a more structured error might be better if this function is used widely.
        return "Error: Image file not found at path." 

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except pytesseract.TesseractNotFoundError:
        # This is a critical setup error.
        return ("Error: Tesseract is not installed or not found in your PATH. "
                "Please install Tesseract OCR and ensure it's accessible. "
                "You might also need to set pytesseract.pytesseract.tesseract_cmd "
                "at the beginning of this script if Tesseract is installed in a non-standard location.")
    except Exception as e:
        # General error during OCR.
        return f"Error during Tesseract OCR processing: {str(e)}"

def extract_text_from_document(file_path, file_extension, logger):
    if file_extension in {'png', 'jpg', 'jpeg'}:
        extracted_text = detect_text(file_path)
        if extracted_text.startswith("Error:"):
            logger.error(f"OCR error for image {file_path}: {extracted_text}")
            return ({"error": extracted_text}, 400 if "not found at path" in extracted_text else 500)
        return ({"text": extracted_text}, 200)

    if file_extension == 'txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ({"text": f.read()}, 200)
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='cp949') as f:
                return ({"text": f.read()}, 200)

    if file_extension == 'pdf':
        try:
            with open(file_path, 'rb') as pdf_file_obj:
                reader = pypdf.PdfReader(pdf_file_obj)
                text_parts = []
                if not reader.pages:
                    logger.warning(f"PDF file {file_path} has no pages.")
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                extracted_text = "\n".join(text_parts)
            if not extracted_text.strip() and reader.pages:
                logger.warning(f"No text extracted from PDF {file_path}. It might be image-based or protected.")
            return ({"text": extracted_text}, 200)
        except pypdf.errors.PdfReadError as e_pdf_read:
            logger.error(f"pypdf could not read PDF {file_path}: {e_pdf_read}")
            return ({"error": f"Invalid or corrupted PDF file. Error: {str(e_pdf_read)}"}, 400)
        except ImportError:
            logger.error("pypdf library is not installed for PDF processing.")
            return ({"error": "PDF processing library (pypdf) not available on server."}, 501)
        except Exception as e_pdf:
            logger.error(f"Error processing PDF {file_path}: {e_pdf}", exc_info=True)
            return ({"error": f"Failed to extract text from PDF: {str(e_pdf)}"}, 500)

    if file_extension == 'docx':
        try:
            doc_obj = docx.Document(file_path)
            text_parts = [para.text for para in doc_obj.paragraphs if para.text]
            return ({"text": "\n".join(text_parts)}, 200)
        except ImportError:
            logger.error("python-docx library is not installed for DOCX processing.")
            return ({"error": "DOCX processing library (python-docx) not available on server."}, 501)
        except Exception as e_docx:
            logger.error(f"Error processing DOCX {file_path}: {e_docx}", exc_info=True)
            return ({"error": f"Failed to extract text from DOCX: {str(e_docx)}"}, 500)

    return ({"error": "Unsupported document type."}, 400)

def clean_html_response(report_html):
    report_html = report_html.replace("Thought:", "").strip()
    report_html = report_html.replace("[CHAT]", "").strip()

    text_prefixes_to_remove = [
        "Thought:", "[CHAT]", "Okay, here is the HTML report:", "Okay, here's the HTML report:",
        "Here is the HTML report:", "Here's the HTML report:", "Here is your HTML report:",
        "Here's your HTML report:", "Sure, here is the HTML content:", "Sure, here's the HTML content:",
        "The HTML report is as follows:", "The report is as follows:", "Here is the report:", "Here's the report:",
        "Final Answer:", "Action Input:", "Observation:"
    ]

    current_html_processing = report_html.strip()
    for _ in range(len(text_prefixes_to_remove) + 2):
        made_change_in_pass = False
        for prefix_str_template in text_prefixes_to_remove:
            escaped_parts = [re.escape(part) for part in prefix_str_template.split(' ') if part]
            if not escaped_parts:
                continue
            prefix_regex_core = r"\s+".join(escaped_parts)
            full_pattern_for_match = rf"^\s*{prefix_regex_core}\s*"
            match = re.match(full_pattern_for_match, current_html_processing, re.IGNORECASE)
            if match:
                current_html_processing = current_html_processing[match.end():].strip()
                made_change_in_pass = True
                break
        if not made_change_in_pass:
            break

    report_html = current_html_processing
    match_wrapper = re.match(r"^\s*(?:```|''')(?:[a-zA-Z0-9]*)?\s*(.*?)\s*(?:```|''')\s*$", report_html, re.DOTALL | re.IGNORECASE)
    if match_wrapper:
        report_html = match_wrapper.group(1).strip()

    if not ("<html" in report_html.lower() or "<body" in report_html.lower() or "<div" in report_html.lower()):
        report_html = f"<html><head><title>Generated Document Analysis</title></head><body><pre>{report_html}</pre></body></html>"

    return report_html

def build_document_analysis_prompt(extracted_text, user_instructions, analysis_mode, question):
    mode_config = DOCUMENT_ANALYSIS_MODES.get(analysis_mode, DOCUMENT_ANALYSIS_MODES["report"])
    prompt_user_instructions = user_instructions if user_instructions else "No extra instructions provided."
    prompt_question = question if question else "No specific question provided."
    return f"""Analyze the following document as a work companion.

Mode: {mode_config["label"]}
Mode instructions:
{mode_config["instruction"]}

The document content is:
--- DOCUMENT CONTENT START ---
{extracted_text}
--- DOCUMENT CONTENT END ---

User instructions:
{prompt_user_instructions}

User question for Q&A mode:
{prompt_question}

The final answer must be in clean HTML format only.
Write in Korean unless the user instructions explicitly request another language.

Please adhere to the following HTML styling for the report:
- Heading elements (e.g., <h1>, <h2>).
- Subheading elements (e.g., <h3>, <h4>).
- Main body text elements (e.g., <p>, <li>, general text).
- Apply 1.5 line spacing for text content where appropriate.
- Use 'Nanum Gothic' font for Korean text and 'Times New Roman' for English text, if specified in the overall system prompt.
These styles should be embedded within the HTML, for example, using inline CSS (e.g., style="font-size: 20pt;") or a <style> block.

You are an intelligent OCR assistant. Extract all readable content from the attached image and return it in a clean, structured report format. Use the following rules:

- Bold section titles
- Use bullet points for lists
- Use tables (Markdown format) for any tabular data
- Correct formatting inconsistencies
- Remove any image noise, page numbers, or irrelevant artifacts
The result should look like a clean text report ready for review.

[Image]

Please write the report in Korean.

The final output must be *only* the HTML content itself.
- Do NOT include any markdown-style code block wrappers around the HTML (e.g., do not start with ```html or '''html and end with ``` or ''').
- Do NOT include ```tool_code```.
- Do NOT include ```Action```.
- Do NOT include any LLM-specific logging, thoughts, or chat-like prefixes (e.g., "Thought:", "[chat]", "Okay, here is the HTML report:").
The response should begin directly with <html> or a similar top-level HTML tag.
"""

def analyze_document(file_path, file_extension, user_instructions, llm, prompt_template, logger, analysis_mode="report", question="", include_extracted_text=False):
    """
    Extracts text from a document and analyzes it as a summary, report, action list, or Q&A answer.
    """
    try:
        extraction_result, extraction_status = extract_text_from_document(file_path, file_extension, logger)
        if extraction_status != 200:
            return extraction_result, extraction_status

        extracted_text = extraction_result.get("text", "")
        if not extracted_text.strip():
            logger.warning(f"Could not extract text from '{file_path}' or the document is empty/non-textual after processing.")
            return ({"error": "Could not extract text from the document, or the document is empty/non-textual."}, 400)

        analysis_mode = analysis_mode if analysis_mode in DOCUMENT_ANALYSIS_MODES else "report"
        report_generation_input = build_document_analysis_prompt(
            extracted_text=extracted_text,
            user_instructions=user_instructions,
            analysis_mode=analysis_mode,
            question=question,
        )
        
        formatted_prompt_messages = prompt_template.format_messages(
            current_document_draft="", 
            retrieved_context="", 
            chat_history=[], 
            input=report_generation_input
        )
        
        ai_response = llm.invoke(formatted_prompt_messages)
        report_html = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
        report_html = clean_html_response(report_html)
        
        result = {
            "report_html": report_html,
            "analysis_mode": analysis_mode,
            "extracted_text_preview": extracted_text[:1000],
            "extracted_text_length": len(extracted_text),
        }
        if include_extracted_text:
            result["extracted_text"] = extracted_text

        return (result, 200)

    except Exception as e:
        logger.error(f"Unhandled error in analyze_document for {file_path}: {str(e)}", exc_info=True)
        return ({"error": f"An unexpected server error occurred during report generation: {str(e)}"}, 500)

def generate_report_from_document(file_path, file_extension, user_instructions, llm, prompt_template, logger):
    return analyze_document(
        file_path=file_path,
        file_extension=file_extension,
        user_instructions=user_instructions,
        llm=llm,
        prompt_template=prompt_template,
        logger=logger,
        analysis_mode="report",
    )


if __name__ == '__main__':
    # This block is for testing the script directly.
    # print("llm_ocr.py script is being run directly for testing.")
    # ... (rest of the __main__ block remains unchanged from previous version)
    
    print("\nllm_ocr.py loaded. Functions detect_text and generate_report_from_document are available.")
    print("Ensure Tesseract OCR is installed and 'pytesseract', 'Pillow', 'pypdf', 'python-docx' Python packages are installed.")
    print("To test generate_report_from_document, you would need to call it with LLM, prompt_template, and logger instances.")
