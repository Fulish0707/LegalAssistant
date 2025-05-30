import pytesseract
from PIL import Image
import os
import PyPDF2 # Added for PDF processing
import docx # Added for DOCX processing

# Optional: If Tesseract is not in your PATH, you might need to set the command path
# For macOS, if installed via Homebrew, it's usually in /usr/local/bin/tesseract or /opt/homebrew/bin/tesseract
# For Windows, it's typically C:\Program Files\Tesseract-OCR\tesseract.exe
# Example:
# if os.name == 'posix': # For macOS/Linux
#     # Check common paths for Tesseract
#     if os.path.exists('/opt/homebrew/bin/tesseract'):
#         pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
#     elif os.path.exists('/usr/local/bin/tesseract'):
#         pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
#     # Add other paths if necessary
# elif os.name == 'nt': # For Windows
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

The report should be in HTML format. If relevant, include diagrams or images as described in the system prompt (e.g., using HTML/CSS for simple diagrams, or base64 encoded images if you can generate them).
Focus on clarity, logical structure, and engagement, suitable for bosses or business clients as per the system prompt's [REPORT_SUGGESTION] guidelines.

Please write the report in Korean.
Please exclude '''html ''' from the report.
Please follow LaTeX formatting, especially font size.
The final output should be only the HTML report content, without any preliminary remarks, thoughts, or chat-like prefixes.
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
        report_html = report_html.replace("[chat]", "").strip()
        # Add more replacements if other variations of "thought" or unwanted markers appear

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
