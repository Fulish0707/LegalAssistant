import pytesseract
from PIL import Image
import os
import PyPDF2 # Added for PDF processing
import re # Added for improved post-processing
import docx # Added for DOCX processing

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

def generate_report_from_document(file_path, file_extension, user_instructions, llm, prompt_template, logger):
    """
    Processes a document file (image, txt, pdf, docx), extracts text,
    and generates an HTML report using an LLM.

    Args:
        file_path (str): Path to the temporary document file.
        file_extension (str): Lowercase extension of the file (e.g., 'pdf', 'png').
        user_instructions (str): Custom instructions from the user for the report.
        llm: The initialized language model instance.
        prompt_template: The Langchain prompt template for formatting LLM input.
        logger: The Flask app's logger instance for logging.

    Returns:
        tuple: (result_dict, status_code)
               result_dict can be {"report_html": "..."} on success or {"error": "..."} on failure.
    """
    extracted_text = ""
    
    try:
        if file_extension in {'png', 'jpg', 'jpeg'}:
            extracted_text = detect_text(file_path)
            if extracted_text.startswith("Error:"): # Check if detect_text returned an error string
                logger.error(f"OCR error for image {file_path}: {extracted_text}")
                return ({"error": extracted_text}, 400 if "not found at path" in extracted_text else 500)
        elif file_extension == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        elif file_extension == 'pdf':
            try:
                with open(file_path, 'rb') as pdf_file_obj:
                    reader = PyPDF2.PdfReader(pdf_file_obj)
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
            except PyPDF2.errors.PdfReadError as e_pdf_read:
                logger.error(f"PyPDF2 could not read PDF {file_path}: {e_pdf_read}")
                return ({"error": f"Invalid or corrupted PDF file. Error: {str(e_pdf_read)}"}, 400)
            except ImportError: 
                logger.error("PyPDF2 library is not installed for PDF processing.")
                return ({"error": "PDF processing library (PyPDF2) not available on server."}, 501)
            except Exception as e_pdf:
                logger.error(f"Error processing PDF {file_path}: {e_pdf}", exc_info=True)
                return ({"error": f"Failed to extract text from PDF: {str(e_pdf)}"}, 500)
        elif file_extension == 'docx':
            try:
                doc_obj = docx.Document(file_path)
                text_parts = [para.text for para in doc_obj.paragraphs if para.text]
                extracted_text = "\n".join(text_parts)
                if not extracted_text.strip() and doc_obj.paragraphs:
                    logger.warning(f"No text extracted from DOCX {file_path}, though it has paragraphs.")
            except ImportError: 
                logger.error("python-docx library is not installed for DOCX processing.")
                return ({"error": "DOCX processing library (python-docx) not available on server."}, 501)
            except Exception as e_docx: 
                logger.error(f"Error processing DOCX {file_path}: {e_docx}", exc_info=True)
                return ({"error": f"Failed to extract text from DOCX: {str(e_docx)}"}, 500)
        
        if not extracted_text.strip():
            logger.warning(f"Could not extract text from '{file_path}' or the document is empty/non-textual after processing.")
            return ({"error": "Could not extract text from the document, or the document is empty/non-textual."}, 400)

        # --- LLM Processing ---
        report_generation_input = f"""Please generate a report based on the following document content and instructions.
The document content is:
--- DOCUMENT CONTENT START ---
{extracted_text}
--- DOCUMENT CONTENT END ---

User instructions for the report:
{user_instructions if user_instructions else "No specific instructions provided. Please generate a comprehensive summary and analysis."}

The report should be in HTML format.
Focus on clarity, logical structure, and engagement, suitable for bosses or business clients as per the system prompt's [REPORT_SUGGESTION] guidelines.

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
        
        formatted_prompt_messages = prompt_template.format_messages(
            current_document_draft="", 
            retrieved_context="", 
            chat_history=[], 
            input=report_generation_input
        )
        
        ai_response = llm.invoke(formatted_prompt_messages)
        report_html = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)

        # Post-process to remove unwanted markers
        report_html = report_html.replace("Thought:", "").strip()
        report_html = report_html.replace("[CHAT]", "").strip() # Also handle uppercase

        # More robust post-processing to remove common textual prefixes and markdown wrappers
        
        # 1. Remove known textual prefixes (case-insensitive, strips leading/trailing space around prefix)
        text_prefixes_to_remove = [
            "Thought:", "[CHAT]", "Okay, here is the HTML report:", "Okay, here's the HTML report:",
            "Here is the HTML report:", "Here's the HTML report:", "Here is your HTML report:",
            "Here's your HTML report:", "Sure, here is the HTML content:", "Sure, here's the HTML content:",
            "The HTML report is as follows:", "The report is as follows:", "Here is the report:", "Here's the report:",
            "Final Answer:", "Action Input:", "Observation:" # Other common agent outputs
        ]
        
        current_html_processing = report_html.strip()
        for _ in range(len(text_prefixes_to_remove) + 2): # Loop a few times for multiple/nested prefixes
            made_change_in_pass = False
            for prefix_str_template in text_prefixes_to_remove:
                # Normalize the prefix_str_template to a regex pattern
                # where spaces in the template can match one or more whitespace chars (incl. newlines).
                # Special regex characters in the template are escaped.
                
                # Split the template by space, escape each part, then join with \s+
                # This makes "Okay, here" match "Okay,\n here" or "Okay,  here" etc.
                escaped_parts = [re.escape(part) for part in prefix_str_template.split(' ') if part]
                if not escaped_parts: # If prefix_str_template was empty or only spaces
                    continue
                
                # Construct regex: leading optional whitespace, then escaped parts joined by \s+, then trailing optional whitespace
                # Example: "Okay, here" -> r"^\s*Okay,\s+here\s*"
                # Example: "Thought:"   -> r"^\s*Thought:\s*" (no internal spaces to turn into \s+)
                prefix_regex_core = r"\s+".join(escaped_parts)
                full_pattern_for_match = rf"^\s*{prefix_regex_core}\s*"

                match = re.match(full_pattern_for_match, current_html_processing, re.IGNORECASE) # \s includes newlines, so DOTALL often not needed here for \s
                if match:
                    current_html_processing = current_html_processing[match.end():].strip()
                    made_change_in_pass = True
                    break # Restart checking prefixes with the modified string as it has changed
            if not made_change_in_pass:
                break # No prefixes were removed in a full pass
        report_html = current_html_processing

        # 2. Remove markdown-style code block wrappers (``` or ''') if they enclose the entire content
        match_wrapper = re.match(r"^\s*(?:```|''')(?:[a-zA-Z0-9]*)?\s*(.*?)\s*(?:```|''')\s*$", report_html, re.DOTALL | re.IGNORECASE)
        if match_wrapper:
            report_html = match_wrapper.group(1).strip()
            
        if not ("<html" in report_html.lower() or "<body" in report_html.lower() or "<div" in report_html.lower()):
             report_html = f"<html><head><title>Generated Report</title></head><body><pre>{report_html}</pre></body></html>"
        
        return ({"report_html": report_html}, 200)

    except Exception as e:
        logger.error(f"Unhandled error in generate_report_from_document for {file_path}: {str(e)}", exc_info=True)
        return ({"error": f"An unexpected server error occurred during report generation: {str(e)}"}, 500)


if __name__ == '__main__':
    # This block is for testing the script directly.
    # print("llm_ocr.py script is being run directly for testing.")
    # ... (rest of the __main__ block remains unchanged from previous version)
    
    print("\nllm_ocr.py loaded. Functions detect_text and generate_report_from_document are available.")
    print("Ensure Tesseract OCR is installed and 'pytesseract', 'Pillow', 'PyPDF2', 'python-docx' Python packages are installed.")
    print("To test generate_report_from_document, you would need to call it with LLM, prompt_template, and logger instances.")
