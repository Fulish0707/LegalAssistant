"""The browser-facing controls ported from the security review: CSRF, security headers,
per-IP request bursts, upload checks and the sandboxed document parser."""

import importlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import document_security
import document_worker
from tests.support import use_browser_clients


class RequestControlTests(unittest.TestCase):
    MODULE_NAMES = ("app", "langchain_config", "llm_model", "llm_prompt")

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.env = mock.patch.dict(os.environ, {
            "APP_STATE_DB": os.path.join(cls.tmpdir.name, "app_state.db"),
            "RAG_PERSIST_DIR": os.path.join(cls.tmpdir.name, "chroma"),
            "APP_ENV": "testing", "LANGSMITH_TRACING": "false"})
        cls.env.start()
        cls.saved_modules = {name: sys.modules.pop(name, None) for name in cls.MODULE_NAMES}
        cls.app_module = importlib.import_module("app")
        use_browser_clients(cls.app_module.app)

    @classmethod
    def tearDownClass(cls):
        for name, module in cls.saved_modules.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module
        cls.env.stop()
        cls.tmpdir.cleanup()

    def test_every_unsafe_route_needs_the_csrf_token(self):
        client = self.app_module.app.test_client()
        for method, path in (("post", "/conversations/new"), ("post", "/conversations/delete"), ("post", "/case_search"),
                             ("post", "/knowledge_base/clear"), ("delete", "/knowledge_base/source"),
                             ("post", "/clear_chat_history_endpoint"), ("post", "/process_document_ocr")):
            with self.subTest(path=path):
                response = getattr(client, method)(path, json={}, without_csrf=True)
                self.assertEqual((response.status_code, response.json), (403, {"error": "request_forbidden"}))

    def test_a_cross_site_request_is_refused_even_with_the_token(self):
        client = self.app_module.app.test_client()
        for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://evil.example"}):
            with self.subTest(headers=headers):
                self.assertEqual(client.post("/conversations/new", headers=headers).status_code, 403)

    def test_pages_and_answers_carry_the_security_headers(self):
        client = self.app_module.app.test_client()
        for path in ("/", "/get_initial_state"):
            with self.subTest(path=path):
                headers = client.get(path).headers
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(headers["X-Frame-Options"], "DENY")
                self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
                self.assertEqual(headers["Cache-Control"], "no-store")

    def test_a_burst_of_work_from_one_ip_is_limited(self):
        app = self.app_module.app
        client = app.test_client()
        saved = app.config["REQUEST_BURST_LIMIT"]
        app.config["REQUEST_BURST_LIMIT"] = 2
        try:
            ip = {"REMOTE_ADDR": "203.0.113.77"}
            codes = [client.get("/law?name=", environ_base=ip).status_code for _ in range(3)]
        finally:
            app.config["REQUEST_BURST_LIMIT"] = saved
        self.assertEqual(codes[-1], 429)
        self.assertNotIn(429, codes[:2])

    def test_an_upload_whose_bytes_do_not_match_its_extension_is_rejected(self):
        client = self.app_module.app.test_client()
        with mock.patch.object(self.app_module, "get_chat_model", lambda: object()):
            response = client.post("/process_document_ocr", data={"file": (io.BytesIO(b"not a pdf"), "lease.pdf")},
                                   content_type="multipart/form-data")
        self.assertEqual((response.status_code, response.json), (422, {"error": "document_rejected"}))


class DocumentParserTests(unittest.TestCase):
    def test_the_worker_and_the_parent_agree_on_the_text_limit(self):
        self.assertEqual(document_worker.EXTRACTED_CHAR_LIMIT, document_security.MAX_EXTRACTED_CHARS)

    def test_importing_the_worker_leaves_the_importers_limits_alone(self):
        """The worker once capped open files at 64 on import, which hung the rest of a test run."""
        import resource
        self.assertNotEqual(resource.getrlimit(resource.RLIMIT_NOFILE), (64, 64))

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("bwrap"), "the sandbox needs bubblewrap on Linux")
    def test_a_long_contract_is_extracted_whole_in_the_sandbox(self):
        """llm_ocr analyses the first 8,000 characters and says the rest was cut; extraction must not
        refuse the document first (the ported limit was 20,000)."""
        text = "제1조 보증금은 1억 원으로 한다.\n" + "가" * 30000
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
            handle.write(text)
        try:
            result, status = document_security.isolated_extract(handle.name, "txt")
        finally:
            os.remove(handle.name)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["text"], text)

    def test_without_the_sandbox_nothing_is_parsed(self):
        with mock.patch.object(document_security.shutil, "which", return_value=None):
            self.assertEqual(document_security.isolated_extract("/nonexistent.txt", "txt"),
                             ({"error": "isolated_parser_unavailable"}, 503))


if __name__ == "__main__":
    unittest.main()
