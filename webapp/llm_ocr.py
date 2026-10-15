import pytesseract
from PIL import Image
import os
import pypdf
import re
import docx
from citation_guard import STATUTE_RE, unsupported_citations
from html_sanitizer import sanitize_report_html

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
        "label": "Legal Document Review",
        "instruction": """Create a Korean-law document review identifying obligations, disputed facts, clause risks, and proposed revisions.
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

    return sanitize_report_html(report_html)

UNSOURCED_NOTICE = ("<p>※ 확인 필요: {citations}은(는) 올리신 문서에 없는 조문·사건번호입니다. 이 분석은 법령을 검색하지 않고 "
                    "작성되어 실제 법령과 다를 수 있습니다. 국가법령정보센터(law.go.kr)나 대법원 종합법률정보에서 직접 확인하세요.</p>")

# Ollama drops the start of a prompt that does not fit num_ctx (12288), mode instructions included,
# and the model then reports the dropped clauses as absent. Legal text is about 0.75 tokens a character,
# so 8,000 characters leave room for the instructions and a reasoning report.
DOCUMENT_ANALYSIS_MAX_CHARS = 8000
CUT_DOCUMENT_NOTICE = ("<p>※ 문서가 길어 앞 {kept:,}자(전체 {total:,}자)만 분석했습니다. 뒷부분의 조항은 이 분석에 "
                       "반영되지 않았습니다. 필요한 부분을 나눠 올리면 따로 분석할 수 있습니다.</p>")

CUT_DOCUMENT_MARK = ("\n[여기서 문서가 잘렸습니다. 문서의 뒷부분은 길이 제한 때문에 보이지 않습니다. 찾는 조항·금액·날짜가 위에 없으면 "
                     "'문서에 없다'거나 '명시되지 않았다'고 쓰지 말고, '제공된 앞부분에서는 찾을 수 없으며 문서 뒷부분은 분석하지 "
                     "않았다'고 쓰세요.]")


def flag_unsourced_citations(report_html, source_texts):
    """Append a notice naming the statute articles and case numbers the report cites that the document does not contain.

    This route has no legal search, and qwen3:4b cited an invented article in every legal review of the
    2026-10-09 test (data/eval/reports/doc_analysis/synthetic-2026-10-09). A bare 제6조 is left alone: it is
    how a report refers to the document's own clauses.
    """
    cited = unsupported_citations(re.sub(r"<[^>]+>", " ", report_html), source_texts)
    def bare(citation):
        return (match := STATUTE_RE.fullmatch(citation)) and not match.group("law")
    missing = [citation for citation in cited if not bare(citation)]
    return report_html + UNSOURCED_NOTICE.format(citations=", ".join(missing)) if missing else report_html


def build_document_analysis_prompt(extracted_text, user_instructions, analysis_mode, question):
    mode_config = DOCUMENT_ANALYSIS_MODES.get(analysis_mode, DOCUMENT_ANALYSIS_MODES["report"])
    prompt_user_instructions = user_instructions if user_instructions else "No extra instructions provided."
    prompt_question = question if question else "No specific question provided."
    return f"""Analyze the following extracted document as a Korean-law legal assistant.

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

Use semantic HTML headings, paragraphs, lists, and HTML tables for tabular data.
Do not use Markdown tables, code fences, scripts, styles, or chat JSON.
Preserve legally meaningful dates, amounts, names, clause numbers, and negations.
Identify unreadable or ambiguous OCR text instead of guessing. Cite available
page or clause identifiers; do not invent them. Separate document statements from
legal conclusions and flag legal authorities requiring external verification.
The response should begin directly with an HTML heading or paragraph.

"""

def analyze_document(file_path, file_extension, user_instructions, llm, prompt_template, logger, analysis_mode="report", question="", include_extracted_text=False, quota_reserver=None):
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
            extracted_text=extracted_text[:DOCUMENT_ANALYSIS_MAX_CHARS] + (CUT_DOCUMENT_MARK if len(extracted_text) > DOCUMENT_ANALYSIS_MAX_CHARS else ""),
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
        
        if quota_reserver is not None:
            quota_result = quota_reserver()
            if not quota_result.allowed:
                return ({"error": "daily_ai_quota_exceeded", "quota_result": quota_result}, 429)

        ai_response = llm.invoke(formatted_prompt_messages)
        report_html = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
        report_html = clean_html_response(report_html)
        report_html = flag_unsourced_citations(report_html, [extracted_text, user_instructions, question])
        if len(extracted_text) > DOCUMENT_ANALYSIS_MAX_CHARS:
            report_html = CUT_DOCUMENT_NOTICE.format(kept=DOCUMENT_ANALYSIS_MAX_CHARS, total=len(extracted_text)) + report_html
        
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
