"""The web chat's client for legal_rag's /v1/chat: one call, and a renderer.

The service is stubbed with an httpx transport; nothing here starts a server or a model.
"""

import unittest
from unittest import mock

import httpx

import legal_backend

EVIDENCE = {
    "items": [
        {"source_id": "S1", "chunk_id": "c1", "document_id": "d1", "document_type": "statute",
         "header": {"Type": "Statute", "Law": "근로기준법", "Article": "제28조"}, "text": "구제신청은 3개월 이내에 하여야 한다.", "score": 0.91},
        {"source_id": "S2", "chunk_id": "c2", "document_id": "d2", "document_type": "judgment",
         "header": {"Type": "Case", "Court": "대법원", "Case number": "2011두1234", "Section role": "법원의 판단"}, "text": "가" * 900, "score": 0.74},
    ]
}
ANSWER = {
    "kind": "answer",
    "conversation_id": "browser-1",
    "answer": {
        "summary": "해고일부터 3개월 안에 구제신청을 할 수 있습니다(S1).",
        "legal_issues": ["구제신청 기간"],
        "analysis": "근로기준법 제28조는 기간을 3개월로 정합니다(S1, S2).",
        "possible_actions": ["노동위원회에 구제신청"],
        "additional_information_needed": ["해고 통보일"],
        "citations": [{"source_id": "S1", "kind": "statute", "reference": "근로기준법 제28조"}],
        "uncertainty": "사업장 규모에 따라 달라질 수 있습니다.",
        "confidence": "medium",
    },
    "evidence": EVIDENCE,
}


def served(handler):
    """Route legal_backend's httpx.post through `handler` instead of the network."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return mock.patch.object(legal_backend.httpx, "post", client.post)


class AskTests(unittest.TestCase):
    def test_posts_the_message_and_conversation_to_the_chat_endpoint(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["json"] = __import__("json").loads(request.content)
            return httpx.Response(200, json=ANSWER)

        with served(handler):
            body = legal_backend.ask("http://legal:8000/", "해고당했어요", "browser-1")

        self.assertEqual(seen["url"], "http://legal:8000/v1/chat")
        self.assertEqual(seen["json"], {"message": "해고당했어요", "conversation_id": "browser-1"})
        self.assertEqual(body["kind"], "answer")

    def test_an_unreachable_service_is_unavailable(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with served(handler), self.assertRaises(legal_backend.LegalBackendUnavailable):
            legal_backend.ask("http://legal:8000", "q", "c")

    def test_a_timeout_is_unavailable(self):
        def handler(request):
            raise httpx.ReadTimeout("slow")

        with served(handler), self.assertRaises(legal_backend.LegalBackendUnavailable):
            legal_backend.ask("http://legal:8000", "q", "c")

    def test_an_error_status_is_unavailable(self):
        with served(lambda request: httpx.Response(500, text="boom")), \
                self.assertRaises(legal_backend.LegalBackendUnavailable):
            legal_backend.ask("http://legal:8000", "q", "c")

    def test_a_body_that_is_not_json_is_unavailable(self):
        with served(lambda request: httpx.Response(200, text="<html>")), \
                self.assertRaises(legal_backend.LegalBackendUnavailable):
            legal_backend.ask("http://legal:8000", "q", "c")


class RenderTests(unittest.TestCase):
    def test_answer_sections_appear_in_reading_order(self):
        text, _ = legal_backend.render(ANSWER)

        order = [text.index(part) for part in (
            "해고일부터 3개월", "근로기준법 제28조는", "취할 수 있는 조치", "노동위원회에 구제신청",
            "추가로 확인이 필요한 사실", "해고 통보일", "불확실한 점", "사업장 규모")]
        self.assertEqual(order, sorted(order))

    def test_source_ids_become_the_numbers_of_the_sources_panel(self):
        text, _ = legal_backend.render(ANSWER)

        self.assertIn("할 수 있습니다[1].", text)
        self.assertIn("정합니다[1][2].", text)
        self.assertNotIn("S1", text)

    def test_empty_sections_are_left_out(self):
        answer = dict(ANSWER["answer"], possible_actions=[], additional_information_needed=[], uncertainty="")
        text, _ = legal_backend.render(dict(ANSWER, answer=answer))

        for heading in ("취할 수 있는 조치", "추가로 확인이 필요한 사실", "불확실한 점"):
            self.assertNotIn(heading, text)

    def test_each_evidence_item_becomes_a_source_card(self):
        _, sources = legal_backend.render(ANSWER)

        self.assertEqual(sources[0], {"source": "근로기준법 제28조", "quote": "구제신청은 3개월 이내에 하여야 한다.",
                                      "kind": "법령", "score": 0.91})
        self.assertEqual(sources[1]["source"], "대법원 2011두1234")
        self.assertEqual(sources[1]["kind"], "판례")
        self.assertEqual(len(sources[1]["quote"]), 400)

    def test_only_the_sources_the_answer_cites_are_shown_and_renumbered(self):
        third = dict(EVIDENCE["items"][1], source_id="S3", header={"Type": "Case", "Court": "헌법재판소", "Case number": "93헌마13"})
        evidence = {"items": [EVIDENCE["items"][0], EVIDENCE["items"][1], third]}
        answer = dict(ANSWER["answer"], summary="신청할 수 있습니다(S3).", analysis="근거는 S3입니다.", citations=[])
        text, sources = legal_backend.render(dict(ANSWER, answer=answer, evidence=evidence))

        self.assertEqual([s["source"] for s in sources], ["헌법재판소 93헌마13"])
        self.assertIn("신청할 수 있습니다[1].", text)
        self.assertIn("근거는 [1]입니다.", text)

    def test_a_source_listed_only_in_citations_is_still_shown(self):
        answer = dict(ANSWER["answer"], summary="신청할 수 있습니다.", analysis="3개월 이내입니다.")  # citations name S1
        _, sources = legal_backend.render(dict(ANSWER, answer=answer))

        self.assertEqual([s["source"] for s in sources], ["근로기준법 제28조"])

    def test_an_answer_that_cites_nothing_shows_no_sources(self):
        answer = dict(ANSWER["answer"], summary="제공된 근거로는 답할 수 없습니다.", analysis="", citations=[])
        _, sources = legal_backend.render(dict(ANSWER, answer=answer))

        self.assertEqual(sources, [])

    def test_bracketed_and_spelled_out_source_ids_are_numbered_too(self):
        answer = dict(ANSWER["answer"], summary="신청할 수 있습니다[S1], [S2].", analysis="[LEGAL SOURCE S2]에 따르면 그렇습니다([S1], [S2]).")
        text, _ = legal_backend.render(dict(ANSWER, answer=answer))

        self.assertIn("신청할 수 있습니다[1], [2].", text)
        self.assertIn("[2]에 따르면 그렇습니다[1][2].", text)

    def test_a_source_without_naming_fields_gets_a_generic_label(self):
        item = dict(EVIDENCE["items"][1], header={"Type": "Case", "Section role": "역할 미상"}, text="에 해당하므로 적법하다.")
        _, sources = legal_backend.render(dict(ANSWER, evidence={"items": [item]}))

        self.assertEqual(sources[0]["source"], "판례 (법원·사건번호 미기재)")

    def test_evidence_texts_are_whole_blocks_not_excerpts(self):
        texts = legal_backend.evidence_texts(ANSWER)

        self.assertEqual(texts[0], "Statute 근로기준법 제28조 구제신청은 3개월 이내에 하여야 한다.")
        self.assertIn("가" * 900, texts[1])

    def test_clarification_lists_the_questions(self):
        text, sources = legal_backend.render({"kind": "clarification", "questions": ["언제 해고되었나요?", "서면 통지를 받았나요?"]})

        self.assertIn("언제 해고되었나요?", text)
        self.assertIn("서면 통지를 받았나요?", text)
        self.assertEqual(sources, [])

    def test_rejection_says_the_answer_could_not_be_grounded(self):
        text, sources = legal_backend.render({"kind": "rejected", "reason": "unsupported citations", "evidence": EVIDENCE})

        self.assertIn("근거를 확인할 수 없어", text)
        self.assertEqual(sources, [])


if __name__ == "__main__":
    unittest.main()
