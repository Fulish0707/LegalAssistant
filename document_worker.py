"""Private extraction worker; no model access and no inherited application secrets."""

import json
from pathlib import Path
import resource
import sys
import zipfile

resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 ** 2, 4 * 1024 ** 2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def extract(path, extension):
    limit = 20000
    if extension == "txt":
        data = Path(path).read_bytes()
        if b"\0" in data:
            raise ValueError("unsupported_media_type")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("cp949")
    elif extension == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted or len(reader.pages) > 100:
            raise ValueError("document_too_complex")
        pieces = []
        size = 0
        for page in reader.pages:
            piece = page.extract_text() or ""
            size += len(piece) + 1
            if size > limit:
                raise ValueError("document_too_complex")
            pieces.append(piece)
        text = "\n".join(pieces)
    elif extension == "docx":
        from docx import Document
        # Archive metadata is untrusted too; inspect it inside the limited worker.
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = {member.filename for member in members}
            if len(members) > 2000 or sum(member.file_size for member in members) > 50 * 1024 * 1024:
                raise ValueError("document_too_complex")
            if {"[Content_Types].xml", "word/document.xml"} - names:
                raise ValueError("unsupported_media_type")
            if any(member.flag_bits & 1 or (member.file_size > 1024 * 1024 and member.file_size > 100 * max(1, member.compress_size)) for member in members):
                raise ValueError("document_too_complex")
        text = "\n".join(p.text for p in Document(path).paragraphs)
    elif extension in {"png", "jpg", "jpeg"}:
        from PIL import Image
        import pytesseract
        Image.MAX_IMAGE_PIXELS = 16000000
        with Image.open(path) as image:
            if image.width * image.height > 16000000:
                raise ValueError("document_too_complex")
            text = pytesseract.image_to_string(image, timeout=20)
    else:
        raise ValueError("unsupported_media_type")
    if len(text) > limit:
        raise ValueError("document_too_complex")
    return text


if __name__ == "__main__":
    try:
        print(json.dumps({"text": extract(sys.argv[1], sys.argv[2])}, ensure_ascii=True))
    except ImportError:
        raise SystemExit(69)
    except Exception:
        raise SystemExit(65)
