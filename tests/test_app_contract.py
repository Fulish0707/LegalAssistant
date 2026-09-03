import importlib
import io
import os
import sys
import tempfile
import unittest


class AppContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        os.environ["APP_STATE_DB"] = os.path.join(cls.tmpdir.name, "app_state.db")
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("SERPAPI_API_KEY", None)

        for module_name in ["app", "langchain_config", "llm_model", "llm_prompt"]:
            sys.modules.pop(module_name, None)

        cls.app_module = importlib.import_module("app")
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_health_reports_unconfigured_services(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        self.assertFalse(response.json["llm_configured"])
        self.assertFalse(response.json["agent_configured"])

    def test_initial_state_and_draft_update_use_configured_db(self):
        update_response = self.client.post(
            "/update_statement",
            json={"personal_statement_draft": "Working draft"},
        )
        state_response = self.client.get("/get_initial_state")

        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(state_response.status_code, 200)
        self.assertEqual(state_response.json["personal_statement_draft"], "Working draft")

    def test_chat_sse_reports_missing_ai_configuration(self):
        response = self.client.get("/chat?message=hello")
        body = response.data.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("AI chat is not configured", body)
        self.assertIn("stream_end", body)

    def test_document_analysis_requires_google_key_when_not_configured(self):
        response = self.client.post("/process_document_ocr", data={})

        self.assertEqual(response.status_code, 503)
        self.assertIn("GOOGLE_API_KEY", response.json["error"])

    def test_document_upload_can_save_mocked_analysis_to_knowledge_base(self):
        original_llm = self.app_module.llm
        original_analyze_document = self.app_module.llm_ocr.analyze_document
        original_ingest = self.app_module.ingest_text_to_vector_store
        captured = {}

        def fake_analyze_document(**kwargs):
            self.assertTrue(kwargs["include_extracted_text"])
            return ({
                "report_html": "<html><body>Summary</body></html>",
                "analysis_mode": kwargs["analysis_mode"],
                "extracted_text_length": 12,
                "extracted_text": "source text",
            }, 200)

        def fake_ingest_text_to_vector_store(text, source_name, metadata=None):
            captured["text"] = text
            captured["source_name"] = source_name
            captured["metadata"] = metadata
            return {"ok": True, "message": "saved", "chunk_count": 1}

        try:
            self.app_module.llm = object()
            self.app_module.llm_ocr.analyze_document = fake_analyze_document
            self.app_module.ingest_text_to_vector_store = fake_ingest_text_to_vector_store

            response = self.client.post(
                "/process_document_ocr",
                data={
                    "file": (io.BytesIO(b"hello"), "sample.txt"),
                    "analysis_mode": "summary",
                    "add_to_knowledge_base": "true",
                },
                content_type="multipart/form-data",
            )
        finally:
            self.app_module.llm = original_llm
            self.app_module.llm_ocr.analyze_document = original_analyze_document
            self.app_module.ingest_text_to_vector_store = original_ingest

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["knowledge_base"]["chunk_count"], 1)
        self.assertNotIn("extracted_text", response.json)
        self.assertEqual(captured["text"], "source text")
        self.assertEqual(captured["source_name"], "sample.txt")
        self.assertEqual(captured["metadata"]["analysis_mode"], "summary")

    def test_statistics_analysis_handles_csv_without_ai_key(self):
        original_ingest = self.app_module.ingest_text_to_vector_store
        captured = {}

        def fake_ingest_text_to_vector_store(text, source_name, metadata=None):
            captured["text"] = text
            captured["source_name"] = source_name
            captured["metadata"] = metadata
            return {"ok": True, "message": "saved", "chunk_count": 1}

        try:
            self.app_module.ingest_text_to_vector_store = fake_ingest_text_to_vector_store
            response = self.client.post(
                "/statistics/analyze",
                data={
                    "file": (
                        io.BytesIO(b"department,sales,cost\nA,10,7\nB,20,11\nA,30,15\n"),
                        "sample.csv",
                    ),
                    "question": "Which department looks strongest?",
                    "add_to_knowledge_base": "true",
                },
                content_type="multipart/form-data",
            )
        finally:
            self.app_module.ingest_text_to_vector_store = original_ingest

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["ai_enabled"])
        self.assertEqual(response.json["analysis"]["row_count"], 3)
        self.assertEqual(response.json["analysis"]["column_count"], 3)
        self.assertIn("sales", [row["column"] for row in response.json["analysis"]["numeric_summary"]])
        self.assertIn("Local statistical analysis", response.json["report_html"])
        self.assertEqual(response.json["knowledge_base"]["chunk_count"], 1)
        self.assertEqual(captured["source_name"], "sample.csv")
        self.assertEqual(captured["metadata"]["analysis_type"], "statistics")

    def test_statistics_analysis_rejects_unsupported_file_type(self):
        response = self.client.post(
            "/statistics/analyze",
            data={"file": (io.BytesIO(b"not a table"), "sample.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("File type not allowed", response.json["error"])


if __name__ == "__main__":
    unittest.main()
