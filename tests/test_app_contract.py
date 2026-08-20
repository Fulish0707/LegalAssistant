import importlib
import io
import os
import sys
import tempfile
import unittest


class AppContractTests(unittest.TestCase):
    MODULE_NAMES = ("app", "langchain_config", "llm_model", "llm_prompt")

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.original_app_state_db = os.environ.get("APP_STATE_DB")
        cls.original_app_env = os.environ.get("APP_ENV")
        cls.original_modules = {
            name: sys.modules.get(name)
            for name in cls.MODULE_NAMES
        }
        os.environ["APP_STATE_DB"] = os.path.join(cls.tmpdir.name, "app_state.db")
        os.environ["APP_ENV"] = "testing"
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("SERPAPI_API_KEY", None)
        os.environ.pop("client_id", None)
        os.environ.pop("client_secret", None)

        for module_name in cls.MODULE_NAMES:
            sys.modules.pop(module_name, None)

        cls.app_module = importlib.import_module("app")
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        for module_name in cls.MODULE_NAMES:
            sys.modules.pop(module_name, None)
            original_module = cls.original_modules[module_name]
            if original_module is not None:
                sys.modules[module_name] = original_module
        if cls.original_app_state_db is None:
            os.environ.pop("APP_STATE_DB", None)
        else:
            os.environ["APP_STATE_DB"] = cls.original_app_state_db
        if cls.original_app_env is None:
            os.environ.pop("APP_ENV", None)
        else:
            os.environ["APP_ENV"] = cls.original_app_env
        cls.tmpdir.cleanup()

    def test_health_reports_unconfigured_services(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        self.assertFalse(response.json["llm_configured"])
        self.assertFalse(response.json["agent_configured"])

    def test_clients_receive_distinct_stable_anonymous_identities(self):
        first = self.app_module.app.test_client()
        second = self.app_module.app.test_client()

        first.get("/health")
        first.get("/health")
        second.get("/health")

        with first.session_transaction() as first_session:
            first_id = first_session["user_id"]
        with second.session_transaction() as second_session:
            second_id = second_session["user_id"]

        self.assertNotEqual(first_id, second_id)
        self.assertGreaterEqual(len(first_id), 32)

    def test_guest_draft_is_session_only_and_is_not_written_to_sqlite(self):
        guest = self.app_module.app.test_client()
        update_response = guest.post(
            "/update_statement",
            json={"personal_statement_draft": "Working draft"},
        )
        state_response = guest.get("/get_initial_state")

        with self.app_module.app.app_context():
            row = self.app_module.get_db().execute(
                "SELECT draft FROM personal_statement WHERE draft=?", ("Working draft",)
            ).fetchone()

        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(state_response.status_code, 200)
        self.assertEqual(state_response.json["personal_statement_draft"], "Working draft")
        self.assertIsNone(row)

        fresh_client = self.app_module.app.test_client()
        fresh_state = fresh_client.get("/get_initial_state")
        self.assertEqual(fresh_state.json["personal_statement_draft"], "")
        self.assertEqual(fresh_state.json["chat_history"], [])

    def test_authenticated_drafts_and_chat_histories_are_private(self):
        first = self.app_module.app.test_client()
        second = self.app_module.app.test_client()

        with first.session_transaction() as flask_session:
            flask_session["authenticated_user_id"] = "account-one"
        with second.session_transaction() as flask_session:
            flask_session["authenticated_user_id"] = "account-two"

        first.post("/update_statement", json={"personal_statement_draft": "First"})
        second.post("/update_statement", json={"personal_statement_draft": "Second"})
        first.post("/chat", json={"message": "First chat"})
        second.post("/chat", json={"message": "Second chat"})

        self.assertEqual(first.get("/get_initial_state").json["personal_statement_draft"], "First")
        self.assertEqual(second.get("/get_initial_state").json["personal_statement_draft"], "Second")
        self.assertEqual(first.get("/get_initial_state").json["chat_history"], [{"role": "user", "content": "First chat"}])
        self.assertEqual(second.get("/get_initial_state").json["chat_history"], [{"role": "user", "content": "Second chat"}])

    def test_signing_in_does_not_migrate_guest_draft_or_chat(self):
        guest = self.app_module.app.test_client()
        guest.post("/update_statement", json={"personal_statement_draft": "Guest draft"})
        guest.post("/chat", json={"message": "Guest chat"})
        with guest.session_transaction() as flask_session:
            flask_session["authenticated_user_id"] = "account-after-guest"

        state = guest.get("/get_initial_state")

        self.assertEqual(state.json["personal_statement_draft"], "")
        self.assertEqual(state.json["chat_history"], [])
        with self.app_module.app.app_context():
            row = self.app_module.get_db().execute(
                "SELECT content FROM chat_history WHERE content=?", ("Guest chat",)
            ).fetchone()
        self.assertIsNone(row)

    def test_chat_sse_reports_missing_ai_configuration(self):
        response = self.client.get("/chat?message=hello")
        body = response.data.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("AI chat is not configured", body)
        self.assertIn("stream_end", body)

    def test_document_analysis_requires_google_key_when_not_configured(self):
        response = self.client.post("/process_document_ocr", data={})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json, {"error": "authentication_required"})

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

        def fake_ingest_text_to_vector_store(text, source_name, metadata=None, user_id=None):
            captured["text"] = text
            captured["source_name"] = source_name
            captured["metadata"] = metadata
            captured["user_id"] = user_id
            return {"ok": True, "message": "saved", "chunk_count": 1}

        try:
            with self.client.session_transaction() as flask_session:
                flask_session["authenticated_user_id"] = "document-account"
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
        self.assertTrue(captured["user_id"])

    def test_statistics_analysis_handles_csv_without_ai_key(self):
        original_ingest = self.app_module.ingest_text_to_vector_store
        captured = {}

        def fake_ingest_text_to_vector_store(text, source_name, metadata=None, user_id=None):
            captured["text"] = text
            captured["source_name"] = source_name
            captured["metadata"] = metadata
            captured["user_id"] = user_id
            return {"ok": True, "message": "saved", "chunk_count": 1}

        try:
            with self.client.session_transaction() as flask_session:
                flask_session["authenticated_user_id"] = "statistics-account"
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
        self.assertTrue(captured["user_id"])

    def test_statistics_analysis_rejects_unsupported_file_type(self):
        response = self.app_module.app.test_client().post(
            "/statistics/analyze",
            data={"file": (io.BytesIO(b"not a table"), "sample.txt")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json, {"error": "authentication_required"})

    def test_guest_provider_upload_routes_require_authentication_before_body_processing(self):
        routes = (
            "/process_document_ocr",
            "/statistics/analyze",
            "/transcribe_and_summarize_audio_file",
            "/transcribe_and_summarize_audio_blob",
        )

        for route in routes:
            with self.subTest(route=route):
                response = self.app_module.app.test_client().post(
                    route,
                    data={"file": (io.BytesIO(b"body must not be processed"), "sample.txt")},
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json, {"error": "authentication_required"})

    def test_guest_cannot_clear_knowledge_base(self):
        response = self.app_module.app.test_client().post("/knowledge_base/clear")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json, {"error": "authentication_required"})


if __name__ == "__main__":
    unittest.main()
