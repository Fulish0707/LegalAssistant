"""Route-level contract for the public browser workspaces.

Every visitor owns one workspace keyed by the opaque ``user_id`` in the signed
session cookie. These tests drive independent test clients as independent
browsers and check that drafts, chat history, knowledge-base operations and
quota accounting behave as the design says: private per browser, server-side,
and never steerable by a submitted owner override.

Provider-dependent paths are mocked. Both connectors are pinned to "none" so no
test can reach Ollama or Chroma.
"""

import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import quote

from quota import guest_ip_identity, reserve_daily_quota


class ExplodingChatModel:
    """A stand-in model that fails the test if the app ever invokes it."""

    def invoke(self, *args, **kwargs):
        raise AssertionError("the model must not be invoked once the quota is exhausted")


class FakeKnowledgeBase:
    """An owner-keyed store standing in for the Chroma-backed helpers."""

    def __init__(self, documents):
        # {owner: {source: text}}
        self.documents = {owner: dict(sources) for owner, sources in documents.items()}

    def status(self, user_id):
        sources = self.documents.get(user_id, {})
        return {
            "available": bool(sources),
            "chunk_count": len(sources),
            "sources": [{"source": name, "chunk_count": 1} for name in sorted(sources)],
        }

    def clear(self, user_id):
        removed = len(self.documents.pop(user_id, {}))
        return {"ok": True, "message": "cleared", "deleted_count": removed}

    def search(self, user_id, query, k=6):
        hits = [
            {"source": name, "score": 1.0, "quote": text, "analysis_mode": ""}
            for name, text in self.documents.get(user_id, {}).items()
            if query in text
        ]
        return {"ok": True, "results": hits[:k]}

    def delete_source(self, user_id, source):
        removed = self.documents.get(user_id, {}).pop(source, None)
        return {"ok": True, "deleted_count": 0 if removed is None else 1}


class AppContractTests(unittest.TestCase):
    MODULE_NAMES = ("app", "langchain_config", "llm_model", "llm_prompt")
    PINNED_ENV = (
        "APP_STATE_DB",
        "RAG_PERSIST_DIR",
        "APP_ENV",
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "LANGSMITH_TRACING",
        "DAILY_AI_REQUEST_LIMIT",
    )

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.original_env = {name: os.environ.get(name) for name in cls.PINNED_ENV}
        cls.original_modules = {name: sys.modules.get(name) for name in cls.MODULE_NAMES}
        # Every store lives in one fresh temporary directory: the tests must never
        # touch the developer's real SQLite files or Chroma collection.
        os.environ["APP_STATE_DB"] = os.path.join(cls.tmpdir.name, "app_state.db")
        os.environ["RAG_PERSIST_DIR"] = os.path.join(cls.tmpdir.name, "chroma")
        os.environ["APP_ENV"] = "testing"
        # Pin both connectors to "none". Ollama needs no credential, so with the
        # default provider these tests would see a fully configured app; "none" is
        # the explicit off switch that exercises the unconfigured paths. Pinning
        # here also survives app.py's load_dotenv() at import.
        os.environ["LLM_PROVIDER"] = "none"
        os.environ["EMBEDDING_PROVIDER"] = "none"
        os.environ["LANGSMITH_TRACING"] = "false"
        # The quota tests exhaust 20 requests; .env may set another limit.
        os.environ["DAILY_AI_REQUEST_LIMIT"] = "20"

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
        for name, value in cls.original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls.tmpdir.cleanup()

    # --- helpers ---------------------------------------------------------

    def session_user_id(self, client):
        with client.session_transaction() as flask_session:
            return flask_session.get("user_id")

    def exhaust_ip_quota(self, remote_addr, count=20):
        identity = guest_ip_identity(remote_addr, self.app_module.app.config["QUOTA_IP_SALT"])
        with self.app_module.app.app_context():
            for _ in range(count):
                reserve_daily_quota(self.app_module.get_db(), "guest_ip", identity)
        return identity

    def quota_row(self, identity):
        with self.app_module.app.app_context():
            return self.app_module.get_db().execute(
                "SELECT request_count FROM daily_ai_quota WHERE identity_kind='guest_ip' AND identity_value=?",
                (identity,),
            ).fetchone()

    # --- health and identity ---------------------------------------------

    def test_health_reports_unconfigured_services(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        self.assertFalse(response.json["llm_configured"])
        self.assertNotIn("google_oidc_configured", response.json)

    def test_clients_receive_distinct_stable_anonymous_identities(self):
        first = self.app_module.app.test_client()
        second = self.app_module.app.test_client()

        first.get("/health")
        first.get("/health")
        second.get("/health")

        first_id = self.session_user_id(first)
        second_id = self.session_user_id(second)

        self.assertNotEqual(first_id, second_id)
        self.assertGreaterEqual(len(first_id), 32)

    # --- chat history ----------------------------------------------------

    def test_a_cleared_cookie_starts_a_fresh_empty_workspace(self):
        original = self.app_module.app.test_client()
        original.post("/chat", json={"message": "remembered"})
        original_id = self.session_user_id(original)

        fresh = self.app_module.app.test_client()
        state = fresh.get("/get_initial_state")

        self.assertEqual(state.json["chat_history"], [])
        self.assertNotEqual(self.session_user_id(fresh), original_id)
        # Losing the cookie does not delete the server-side workspace.
        with self.app_module.app.app_context():
            self.assertEqual(self.app_module.get_chat_history_from_db(original_id), [{"role": "user", "content": "remembered"}])

    def test_chat_histories_are_private_per_browser(self):
        first = self.app_module.app.test_client()
        second = self.app_module.app.test_client()

        first.post("/chat", json={"message": "First chat"})
        second.post("/chat", json={"message": "Second chat"})

        self.assertEqual(first.get("/get_initial_state").json["chat_history"], [{"role": "user", "content": "First chat"}])
        self.assertEqual(second.get("/get_initial_state").json["chat_history"], [{"role": "user", "content": "Second chat"}])


    def test_clearing_one_history_leaves_the_other_untouched(self):
        first = self.app_module.app.test_client()
        second = self.app_module.app.test_client()
        first.post("/chat", json={"message": "first keeps"})
        second.post("/chat", json={"message": "second keeps"})

        response = first.post("/clear_chat_history_endpoint")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(first.get("/get_initial_state").json["chat_history"], [])
        self.assertEqual(second.get("/get_initial_state").json["chat_history"], [{"role": "user", "content": "second keeps"}])

    def test_legacy_account_cookies_do_not_recover_account_data(self):
        legacy_id = "legacy-google-account"
        with self.app_module.app.app_context():
            self.app_module.add_chat_message_to_db(legacy_id, "user", "legacy account chat")

        browser = self.app_module.app.test_client()
        with browser.session_transaction() as flask_session:
            flask_session["authenticated_user_id"] = legacy_id
            flask_session["account_email"] = "person@example.com"
            flask_session["auth_next"] = "/chatbot.html"
            flask_session["oidc_nonce"] = "nonce"

        state = browser.get("/get_initial_state")

        self.assertEqual(state.json["chat_history"], [])
        with browser.session_transaction() as flask_session:
            self.assertNotEqual(flask_session["user_id"], legacy_id)
            for key in ("authenticated_user_id", "account_email", "auth_next", "oidc_nonce"):
                self.assertNotIn(key, flask_session)
        with self.app_module.app.app_context():
            self.assertEqual(
                self.app_module.get_chat_history_from_db(legacy_id),
                [{"role": "user", "content": "legacy account chat"}],
            )

    # --- chat streaming and quota -----------------------------------------

    def backend(self, ask):
        """/chat with legal_rag configured and its call replaced by `ask`."""
        return patch.dict(os.environ, {"LEGAL_RAG_URL": "http://legal:8000"}), \
            patch.object(self.app_module.legal_backend, "ask", ask)

    def test_chat_sse_reports_a_missing_legal_backend(self):
        with patch.dict(os.environ, {"LEGAL_RAG_URL": ""}):
            response = self.client.get("/chat?message=hello")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("LEGAL_RAG_URL", body)
        self.assertIn("stream_end", body)

    def test_chat_sse_reports_quota_exhaustion_without_calling_the_backend(self):
        self.exhaust_ip_quota("127.0.0.1")
        ask = Mock(return_value={"kind": "rejected"})
        env, stub = self.backend(ask)
        with env, stub:
            response = self.app_module.app.test_client().get("/chat?message=" + quote("전세 보증금을 못 받았어요"))

        ask.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertIn("daily_ai_quota_exceeded", response.get_data(as_text=True))
        self.assertIn("stream_end", response.get_data(as_text=True))

    def test_a_greeting_is_answered_without_using_the_daily_quota(self):
        """The quota was reserved before the canned reply was chosen (code review, 2026-10-14)."""
        identity = self.exhaust_ip_quota("127.0.0.1")
        used = self.quota_row(identity)[0]
        ask = Mock(return_value={"kind": "rejected"})
        env, stub = self.backend(ask)
        with env, stub:
            body = self.app_module.app.test_client().get("/chat?message=" + quote("안녕하세요")).get_data(as_text=True)

        ask.assert_not_called()
        self.assertIn("final_response", body)
        self.assertNotIn("daily_ai_quota_exceeded", body)
        self.assertEqual(self.quota_row(identity)[0], used)

    def test_a_message_over_the_backend_limit_is_refused_before_quota_and_backend(self):
        """legal_rag rejects a message over 4,000 characters with 422, which the app showed as
        'service unavailable' on every retry, each one using quota (code review, 2026-10-14)."""
        long_message = "보증금" * 1400
        identity = guest_ip_identity("127.0.0.1", self.app_module.app.config["QUOTA_IP_SALT"])
        used = (self.quota_row(identity) or [0])[0]
        client = self.app_module.app.test_client()
        posted = client.post("/chat", json={"message": long_message})
        ask = Mock(return_value={"kind": "rejected"})
        env, stub = self.backend(ask)
        with env, stub:
            body = client.get("/chat?message=" + quote(long_message)).get_data(as_text=True)

        events = [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]

        self.assertEqual(posted.status_code, 400)
        ask.assert_not_called()
        self.assertEqual(events, [{"type": "error", "content": self.app_module.MESSAGE_TOO_LONG}, {"type": "stream_end"}])
        self.assertEqual((self.quota_row(identity) or [0])[0], used)

    def test_quota_is_shared_per_ip_and_survives_a_cleared_cookie(self):
        shared_ip, other_ip = "203.0.113.10", "203.0.113.11"
        self.exhaust_ip_quota(shared_ip, count=19)
        ask = Mock(return_value={"kind": "rejected"})

        first = self.app_module.app.test_client()
        second = self.app_module.app.test_client()
        third = self.app_module.app.test_client()
        env, stub = self.backend(ask)
        with env, stub:
            first_body = first.get("/chat?message=보증금 문의", environ_base={"REMOTE_ADDR": shared_ip}).get_data(as_text=True)
            second_body = second.get("/chat?message=보증금 문의", environ_base={"REMOTE_ADDR": shared_ip}).get_data(as_text=True)
            third_body = third.get("/chat?message=보증금 문의", environ_base={"REMOTE_ADDR": other_ip}).get_data(as_text=True)
            cleared = self.app_module.app.test_client()
            cleared_body = cleared.get("/chat?message=보증금 문의", environ_base={"REMOTE_ADDR": shared_ip}).get_data(as_text=True)

        self.assertNotIn("daily_ai_quota_exceeded", first_body)
        self.assertIn("daily_ai_quota_exceeded", second_body)
        self.assertNotIn("daily_ai_quota_exceeded", third_body)
        self.assertIn("daily_ai_quota_exceeded", cleared_body)
        conversations = [call.args[2] for call in ask.call_args_list]
        self.assertEqual(len(conversations), 2)
        self.assertTrue(conversations[0].startswith(self.session_user_id(first)))
        self.assertTrue(conversations[1].startswith(self.session_user_id(third)))
        self.assertNotEqual(self.session_user_id(cleared), self.session_user_id(first))

    def test_guest_quota_identity_ignores_forwarded_for_headers(self):
        with self.app_module.app.test_request_context(
            "/",
            headers={"X-Forwarded-For": "198.51.100.14"},
            environ_base={"REMOTE_ADDR": "10.0.0.8"},
        ):
            result = self.app_module.reserve_ai_request()
        expected = guest_ip_identity("10.0.0.8", self.app_module.app.config["QUOTA_IP_SALT"])
        with self.app_module.app.app_context():
            row = self.app_module.get_db().execute(
                "SELECT identity_value FROM daily_ai_quota WHERE identity_kind='guest_ip' AND quota_day=?",
                (result.quota_day,),
            ).fetchone()

        self.assertEqual(row["identity_value"], expected)

    def test_quota_never_reserves_account_entries(self):
        with self.app_module.app.test_request_context("/", environ_base={"REMOTE_ADDR": "203.0.113.12"}):
            self.app_module.reserve_ai_request()
        with self.app_module.app.app_context():
            kinds = {
                row[0]
                for row in self.app_module.get_db().execute("SELECT DISTINCT identity_kind FROM daily_ai_quota")
            }
        self.assertEqual(kinds, {"guest_ip"})

    # --- uploads: validation, quota, ownership ----------------------------

    def test_empty_upload_requests_reach_validation_without_signing_in(self):
        with patch.object(self.app_module, "get_chat_model", lambda: object()):
            response = self.app_module.app.test_client().post("/process_document_ocr", data={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {"error": "No file part"})

    def test_document_analysis_reports_missing_llm_before_reading_the_upload(self):
        response = self.client.post("/process_document_ocr", data={})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json, {"error": "Document analysis requires an LLM provider to be configured."})

    def test_failed_ocr_extraction_does_not_consume_quota(self):
        remote_addr = "203.0.113.20"
        identity = guest_ip_identity(remote_addr, self.app_module.app.config["QUOTA_IP_SALT"])
        with patch.object(self.app_module, "get_chat_model", lambda: object()), patch.object(
            self.app_module.llm_ocr, "extract_text_from_document", lambda *args: ({"error": "unreadable"}, 400)
        ):
            response = self.client.post(
                "/process_document_ocr",
                data={"file": (io.BytesIO(b"invalid text"), "sample.txt")},
                content_type="multipart/form-data",
                environ_base={"REMOTE_ADDR": remote_addr},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIsNone(self.quota_row(identity))

    def test_ocr_returns_json_429_when_the_ip_quota_is_exhausted(self):
        remote_addr = "203.0.113.21"
        self.exhaust_ip_quota(remote_addr)
        with patch.object(self.app_module, "get_chat_model", lambda: ExplodingChatModel()):
            response = self.client.post(
                "/process_document_ocr",
                data={"file": (io.BytesIO(b"source text"), "sample.txt")},
                content_type="multipart/form-data",
                environ_base={"REMOTE_ADDR": remote_addr},
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json["error"], "daily_ai_quota_exceeded")


    def test_document_upload_can_save_mocked_analysis_to_the_browsers_knowledge_base(self):
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
            captured.update(text=text, source_name=source_name, metadata=metadata, user_id=user_id)
            return {"ok": True, "message": "saved", "chunk_count": 1}

        browser = self.app_module.app.test_client()
        with patch.object(self.app_module, "get_chat_model", lambda: object()), patch.object(
            self.app_module.llm_ocr, "analyze_document", fake_analyze_document
        ), patch.object(self.app_module, "ingest_text_to_vector_store", fake_ingest_text_to_vector_store):
            response = browser.post(
                "/process_document_ocr",
                data={
                    "file": (io.BytesIO(b"hello"), "sample.txt"),
                    "analysis_mode": "summary",
                    "add_to_knowledge_base": "true",
                    "user_id": "forged-owner",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["knowledge_base"]["chunk_count"], 1)
        self.assertNotIn("extracted_text", response.json)
        self.assertEqual(captured["text"], "source text")
        self.assertEqual(captured["source_name"], "sample.txt")
        self.assertEqual(captured["metadata"]["analysis_mode"], "summary")
        self.assertEqual(captured["user_id"], self.session_user_id(browser))

    def test_a_file_with_a_korean_name_is_analysed_and_keeps_its_name(self):
        """secure_filename drops Hangul, which left "txt": no extension, a 500, and a nameless source."""
        captured = {}

        def fake_analyze_document(**kwargs):
            captured["extension"] = kwargs["file_extension"]
            return ({"report_html": "<p>ok</p>", "extracted_text": "본문"}, 200)

        def fake_ingest(text, source_name, metadata=None, user_id=None):
            captured["source_name"] = source_name
            return {"ok": True, "chunk_count": 1}

        with patch.object(self.app_module, "get_chat_model", lambda: object()), patch.object(
            self.app_module.llm_ocr, "analyze_document", fake_analyze_document
        ), patch.object(self.app_module, "ingest_text_to_vector_store", fake_ingest):
            response = self.app_module.app.test_client().post(
                "/process_document_ocr",
                data={"file": (io.BytesIO("본문".encode()), "../폴더/임대차계약서.TXT"), "add_to_knowledge_base": "true"},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured, {"extension": "txt", "source_name": "임대차계약서.TXT"})


    # --- knowledge base isolation -----------------------------------------

    def test_knowledge_base_routes_answer_without_signing_in(self):
        browser = self.app_module.app.test_client()

        status = browser.get("/knowledge_base/status")
        cleared = browser.post("/knowledge_base/clear")

        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json, {"available": False, "chunk_count": 0, "sources": []})
        self.assertEqual(cleared.status_code, 200)
        self.assertTrue(cleared.json["ok"])

    def test_knowledge_base_operations_only_touch_the_callers_workspace(self):
        first = self.app_module.app.test_client()
        second = self.app_module.app.test_client()
        first.get("/health")
        second.get("/health")
        first_id, second_id = self.session_user_id(first), self.session_user_id(second)
        store = FakeKnowledgeBase({
            first_id: {"first.pdf": "shared word first", "first-extra.pdf": "only first"},
            second_id: {"second.pdf": "shared word second"},
        })

        with patch.object(self.app_module, "get_knowledge_base_status", store.status), patch.object(
            self.app_module, "clear_knowledge_base", store.clear
        ), patch.object(self.app_module, "search_knowledge_base", store.search), patch.object(
            self.app_module, "delete_knowledge_base_source", store.delete_source
        ):
            # Status and search see only the caller's documents, even with a forged owner.
            first_status = first.get(f"/knowledge_base/status?user_id={second_id}")
            self.assertEqual([s["source"] for s in first_status.json["sources"]], ["first-extra.pdf", "first.pdf"])
            first_search = first.post("/knowledge_base/search", json={"query": "shared word", "user_id": second_id})
            self.assertEqual([hit["source"] for hit in first_search.json["results"]], ["first.pdf"])
            second_search = second.post("/knowledge_base/search", json={"query": "shared word", "user_id": first_id})
            self.assertEqual([hit["source"] for hit in second_search.json["results"]], ["second.pdf"])

            # Deleting a source the caller does not own is a no-op for the owner.
            forged_delete = second.delete("/knowledge_base/source", json={"source": "first.pdf", "user_id": first_id})
            self.assertEqual(forged_delete.status_code, 200)
            self.assertEqual(forged_delete.json["deleted_count"], 0)
            self.assertIn("first.pdf", store.documents[first_id])

            own_delete = first.delete("/knowledge_base/source", json={"source": "first.pdf"})
            self.assertEqual(own_delete.json["deleted_count"], 1)
            self.assertEqual(set(store.documents[first_id]), {"first-extra.pdf"})
            self.assertEqual(set(store.documents[second_id]), {"second.pdf"})

            # Clearing affects only the caller, even with a forged owner in the query.
            cleared = second.post(f"/knowledge_base/clear?user_id={first_id}", json={"user_id": first_id})
            self.assertEqual(cleared.status_code, 200)
            self.assertEqual(cleared.json["deleted_count"], 1)
            self.assertEqual(set(store.documents.get(first_id, {})), {"first-extra.pdf"})
            self.assertNotIn(second_id, store.documents)
            self.assertEqual(second.get("/knowledge_base/status").json["chunk_count"], 0)
            self.assertEqual(first.get("/knowledge_base/status").json["chunk_count"], 1)

    def test_knowledge_base_routes_still_validate_their_payloads(self):
        browser = self.app_module.app.test_client()

        search = browser.post("/knowledge_base/search", json={})
        delete = browser.delete("/knowledge_base/source", json={})

        self.assertEqual(search.status_code, 400)
        self.assertEqual(delete.status_code, 400)


if __name__ == "__main__":
    unittest.main()
