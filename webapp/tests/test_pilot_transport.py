"""Opt-in pilot transport and bounded rendering; no live model or retrieval."""
import json
import unittest
from unittest import mock
import httpx
import legal_backend
from tests.test_chat_streaming import ChatStreamBase


WITHHELD = {"kind": "withheld", "withheld": {
    "reason_code": "evidence_gap", "known_facts": ["가상 계약의 중도 퇴거 조항"],
    "unresolved_items": ["종료 합의 여부"], "explanation": "결론을 보류합니다."},
    "answer": {"summary": "노출하면 안 되는 결론"}, "questions": ["다시 질문?"],
    "evidence": {"items": [{"source_id": "S1", "text": "not support"}]}}


class PilotRenderingTests(unittest.TestCase):
    def test_withheld_uses_only_bounded_details_and_no_cards(self):
        text, sources = legal_backend.render(WITHHELD)
        for value in ("결론을 보류합니다.", "evidence_gap", "가상 계약의 중도 퇴거 조항", "종료 합의 여부"):
            self.assertIn(value, text)
        self.assertNotIn("노출하면", text)
        self.assertEqual(sources, [])
        self.assertEqual(legal_backend.questions(WITHHELD), [])

    def test_answer_question_suppression(self):
        self.assertEqual(legal_backend.questions({"kind": "answer", "suppress_followup_questions": True,
            "answer": {"additional_information_needed": ["종료 합의 여부?"]}}), [])

    def test_transport_sends_identity_only_with_scope(self):
        sent = []
        def handler(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, text=json.dumps({"type": "response", "body": WITHHELD}) + "\n")
        scope = {"profile": "lease_review_v1", "matter_id": "fictional-lease-1", "issue_id": "rent-deduction"}
        with httpx.Client(transport=httpx.MockTransport(handler)) as client, mock.patch.object(legal_backend.httpx, "stream", client.stream):
            list(legal_backend.ask("http://legal", "q", "c", request_id="ignored"))
            list(legal_backend.ask("http://legal", "q", "c", policy_scope=scope, request_id="r1"))
        self.assertNotIn("request_id", sent[0])
        self.assertNotIn("policy_scope", sent[0])
        self.assertEqual(sent[1]["policy_scope"], scope)
        self.assertEqual(sent[1]["request_id"], "r1")


class PilotWebTests(ChatStreamBase):
    def test_default_post_does_not_opt_in_even_with_uploaded_documents(self):
        self.as_browser("pilot-default")
        self.assertEqual(self.client.post("/chat", json={"message": "계약 검토"}).status_code, 200)
        with mock.patch.object(self.app_module, "_case_documents", return_value=[{"name": "실제 계약", "text": "실제 자료"}]):
            self.collect(WITHHELD, query="message=계약 검토")
        self.assertNotIn("policy_scope", self.ask.call_args.kwargs)

    def test_opt_in_replaces_uploads_and_keeps_request_identity_on_retry(self):
        self.as_browser("pilot-explicit")
        response = self.client.post("/chat", json={"message": "이 특약 설명", "pilot": True,
            "issue_id": "clause-explanation", "request_id": "pilot-r1"})
        self.assertEqual(response.status_code, 200)
        with mock.patch.object(self.app_module, "_case_documents", side_effect=AssertionError("must not use uploads")):
            events = self.collect(WITHHELD, query="message=이 특약 설명&request_id=pilot-r1")
            first = self.ask.call_args
            self.collect(WITHHELD, query="message=이 특약 설명&request_id=pilot-r1")
        self.assertEqual(first.kwargs["request_id"], "pilot-r1")
        self.ask.assert_not_called()
        self.assertEqual(first.kwargs["policy_scope"], {"profile": "lease_review_v1", "matter_id": "fictional-lease-1", "issue_id": "clause-explanation"})
        self.assertIn("가상 주택 임대차계약", first.args[3][0]["text"])
        self.assertEqual(first.kwargs["pinned"], ())
        self.assertEqual(self.final(events)["sources"], [])
        self.assertFalse(any("근거 " in e.get("content", "") and "건 확인" in e.get("content", "") for e in events))

    def test_invalid_pilot_identity_is_rejected_before_history_write(self):
        self.as_browser("pilot-invalid")
        for payload in ({"pilot": True}, {"pilot": True, "issue_id": "other", "request_id": "r"},
                        {"pilot": True, "issue_id": "rent-deduction", "request_id": " "}):
            response = self.client.post("/chat", json={"message": "질문", **payload})
            self.assertEqual(response.status_code, 400)

    def test_nonanswers_do_not_claim_verified_sources(self):
        self.as_browser("pilot-status")
        for body in (WITHHELD, {"kind": "clarification", "questions": ["합의 여부?"]}, {"kind": "rejected"}):
            events = self.collect(body)
            self.assertFalse(any("근거 0건 확인" in e.get("content", "") for e in events))

    def test_pilot_identity_cannot_be_activated_by_get_or_a_different_consultation(self):
        self.as_browser("pilot-get")
        self.assertEqual(self.client.get("/chat?message=q&request_id=unknown").status_code, 409)
        self.client.post("/chat", json={"message": "q", "pilot": True, "issue_id": "rent-deduction", "request_id": "r1"})
        with self.client.session_transaction() as stored:
            self.assertNotIn("text", stored["pilot_turn"])
            stored["conversation_epoch"] = 99
        self.assertEqual(self.client.get("/chat?message=q&request_id=r1").status_code, 409)

    def test_default_post_clears_previous_pilot_activation(self):
        self.as_browser("pilot-clear")
        self.client.post("/chat", json={"message": "q", "pilot": True, "issue_id": "rent-deduction", "request_id": "r1"})
        self.client.post("/chat", json={"message": "일반 질문"})
        with self.client.session_transaction() as stored:
            self.assertNotIn("pilot_turn", stored)
        self.assertEqual(self.client.get("/chat?message=q&request_id=r1").status_code, 409)


    def test_retry_does_not_charge_quota_call_backend_or_store_reply_twice(self):
        self.as_browser("pilot-retry-once")
        self.client.post("/chat", json={"message": "q", "pilot": True,
            "issue_id": "rent-deduction", "request_id": "r1"})
        with mock.patch.object(self.app_module, "reserve_ai_request", wraps=self.app_module.reserve_ai_request) as quota:
            first = self.collect(WITHHELD, query="message=q&request_id=r1")
            second = self.collect(WITHHELD, query="message=q&request_id=r1")
            self.assertEqual(quota.call_count, 1)
            self.ask.assert_not_called()
        self.assertEqual(self.final(first), self.final(second))
        with self.app_module.app.test_request_context():
            rows = self.app_module.get_db().execute(
                "SELECT role FROM chat_history WHERE user_id=? ORDER BY id", ("pilot-retry-once",)).fetchall()
        self.assertEqual([row["role"] for row in rows], ["user", "assistant"])

    def test_changed_get_message_is_rejected_before_backend(self):
        self.as_browser("pilot-bound-message")
        self.client.post("/chat", json={"message": "original", "pilot": True,
            "issue_id": "rent-deduction", "request_id": "r1"})
        response = self.client.get("/chat?message=altered&request_id=r1")
        self.assertEqual(response.status_code, 409)

    def test_duplicate_post_does_not_store_user_twice(self):
        self.as_browser("pilot-post-once")
        payload = {"message": "q", "pilot": True, "issue_id": "rent-deduction", "request_id": "r1"}
        self.assertEqual(self.client.post("/chat", json=payload).status_code, 200)
        self.assertEqual(self.client.post("/chat", json=payload).status_code, 200)
        self.assertEqual(self.client.post("/chat", json={**payload, "message": "changed"}).status_code, 409)
        with self.app_module.app.test_request_context():
            rows = self.app_module.get_chat_history_from_db("pilot-post-once", 0)
        self.assertEqual([row["role"] for row in rows], ["user"])


    def test_disconnect_finishes_once_and_retry_replays_saved_result(self):
        self.as_browser("pilot-disconnect")
        self.client.post("/chat", json={"message": "q", "pilot": True,
            "issue_id": "rent-deduction", "request_id": "r1"})
        backend = mock.Mock(return_value=[{"type": "response", "body": WITHHELD}])
        with mock.patch.dict("os.environ", {"LEGAL_RAG_URL": "http://legal"}), \
                mock.patch.object(self.app_module.legal_backend, "ask", backend):
            response = self.client.get("/chat?message=q&request_id=r1", buffered=False)
            next(iter(response.response))
            response.close()
        self.assertEqual(backend.call_count, 1)
        replay = self.collect(WITHHELD, query="message=q&request_id=r1")
        self.ask.assert_not_called()
        self.assertIn("결론을 보류합니다", self.final(replay)["chat_response"])

    def test_running_request_does_not_call_backend_again(self):
        self.as_browser("pilot-running")
        self.client.post("/chat", json={"message": "q", "pilot": True,
            "issue_id": "rent-deduction", "request_id": "r1"})
        with self.client.session_transaction() as stored:
            key = stored["pilot_turn"]["key"]
        with self.app_module.app.app_context():
            self.app_module.pilot_requests.claim(self.app_module.get_db(), "pilot-running", key, "r1")
        events = self.collect(WITHHELD, query="message=q&request_id=r1")
        self.ask.assert_not_called()
        self.assertEqual(events[0]["error"], "policy_request_in_progress")

    def test_failed_web_request_is_cached_not_reexecuted(self):
        self.as_browser("pilot-error-cache")
        self.client.post("/chat", json={"message": "q", "pilot": True,
            "issue_id": "rent-deduction", "request_id": "r1"})
        first = self.collect(legal_backend.LegalBackendUnavailable("test error"), query="message=q&request_id=r1")
        second = self.collect(WITHHELD, query="message=q&request_id=r1")
        self.ask.assert_not_called()
        self.assertEqual([e for e in first if e["type"] == "error"], [e for e in second if e["type"] == "error"])
