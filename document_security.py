"""Bound file validation and extract text in a disposable resource-limited process."""

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_EXTRACTED_CHARS = 20000


class DocumentRejected(ValueError):
    pass


def validate_upload(upload, extension):
    stream = upload.stream
    stream.seek(0, 2)
    size = stream.tell()
    stream.seek(0)
    head = stream.read(512)
    stream.seek(0)
    if not 0 < size <= MAX_FILE_BYTES:
        raise DocumentRejected("payload_too_large")
    valid = {
        "png": head.startswith(b"\x89PNG\r\n\x1a\n"),
        "jpg": head.startswith(b"\xff\xd8\xff"),
        "jpeg": head.startswith(b"\xff\xd8\xff"),
        "pdf": head.startswith(b"%PDF-"),
        "docx": head.startswith(b"PK\x03\x04"),
        "txt": b"\0" not in head,
    }
    if not valid.get(extension, False):
        raise DocumentRejected("unsupported_media_type")


def _sandbox_command(file_path, extension):
    executable = shutil.which("bwrap")
    if sys.platform != "linux" or not executable:
        return None
    command = [executable, "--unshare-all", "--die-with-parent", "--new-session",
               "--disable-userns", "--cap-drop", "ALL", "--clearenv"]
    runtime_paths = {"/usr/bin", "/usr/lib", "/usr/lib64", "/usr/local/bin", "/usr/local/lib",
                     "/lib", "/lib64", "/usr/share/tesseract-ocr", "/usr/share/tessdata",
                     "/etc/ld.so.cache", "/etc/fonts"}
    for prefix in {sys.prefix, sys.base_prefix} - {"/usr", "/usr/local"}:
        # Bind only an interpreter/venv installation, never the project or home root.
        resolved = Path(prefix).resolve()
        if resolved in {Path("/"), Path.home(), Path(__file__).resolve().parent}:
            return None
        runtime_paths.add(prefix)
    for path in sorted(runtime_paths):
        if Path(path).exists():
            command += ["--ro-bind", path, path]
    command += ["--proc", "/proc", "--dev", "/dev", "--size", "33554432", "--tmpfs", "/tmp",
                "--dir", "/input", "--ro-bind", str(Path(file_path).resolve()), "/input/document",
                "--ro-bind", str(Path(__file__).with_name("document_worker.py").resolve()), "/worker.py",
                "--setenv", "PATH", "/usr/bin:/usr/local/bin", "--setenv", "LANG", "C.UTF-8",
                "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
                "--chdir", "/tmp", "--", sys.executable, "-I", "/worker.py", "/input/document", extension]
    return command


def isolated_extract(file_path, extension):
    command = _sandbox_command(file_path, extension)
    if command is None:
        return {"error": "isolated_parser_unavailable"}, 503
    # A regular output file is covered by the worker's RLIMIT_FSIZE. Do not
    # accumulate an untrusted child's stdout in the web process memory.
    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.DEVNULL,
                                       env={"PATH": "/usr/bin:/bin"}, start_new_session=True)
        except OSError:
            return {"error": "isolated_parser_unavailable"}, 503
        try:
            process.wait(timeout=45)
            if process.returncode in {1, 69}:
                return {"error": "isolated_parser_unavailable"}, 503
            if process.returncode != 0 or os.fstat(output.fileno()).st_size > 250000:
                return {"error": "document_processing_failed"}, 422
            output.seek(0)
            result = json.loads(output.read(250001))
            text = result.get("text") if isinstance(result, dict) else None
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_EXTRACTED_CHARS:
                return {"error": "document_too_complex"}, 422
            return {"text": text}, 200
        except subprocess.TimeoutExpired:
            return {"error": "document_processing_timeout"}, 422
        except (ValueError, OSError):
            return {"error": "document_processing_failed"}, 422
        finally:
            # OCR descendants are in the same disposable process group. The PID
            # namespace also prevents them from surviving the sandbox's exit.
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
