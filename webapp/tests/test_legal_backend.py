"""The web chat's client for legal_rag's /v1/chat: one call, and a renderer.

The service is stubbed with an httpx transport; nothing here starts a server or a model.
"""

import unittest
from unittest import mock

import json

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


class DocumentTests(unittest.TestCase):
    def fetched(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return mock.patch.object(legal_backend.httpx, "get", client.get)

    def test_the_whole_document_is_requested_by_the_chunk_the_card_shows(self):
        def handler(request):
            self.assertEqual((request.url.path, request.url.params["chunk_id"]), ("/v1/document", "law#art:제3조"))
            return httpx.Response(200, json={"document_id": "law", "text": "제2조 \U00020000\n\n제3조 ...\n\n제4조", "cited_start": 7, "cited_end": 14})

        with self.fetched(handler):
            document = legal_backend.document("http://legal:8000/", "law#art:제3조")

        self.assertEqual(document, {"before": "제2조 \U00020000\n\n", "cited": "제3조 ...", "after": "\n\n제4조"})

    def test_a_chunk_the_service_does_not_hold_has_no_document(self):
        with self.fetched(lambda request: httpx.Response(404, json={"detail": "none"})):
            self.assertIsNone(legal_backend.document("http://legal:8000", "gone"))

    def test_an_unreachable_or_broken_service_is_reported_as_unavailable(self):
        def refused(request):
            raise httpx.ConnectError("refused")

        for handler in (refused, lambda request: httpx.Response(500, text="boom"), lambda request: httpx.Response(200, json={})):
            with self.subTest(handler=handler), self.fetched(handler), self.assertRaises(legal_backend.LegalBackendUnavailable):
                legal_backend.document("http://legal:8000", "c1")


class LawTests(unittest.TestCase):
    def fetched(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return mock.patch.object(legal_backend.httpx, "get", client.get)

    def test_a_law_is_read_as_its_articles_and_an_unknown_law_is_none(self):
        article = {"article": "제3조", "title": "대항력 등", "text": "제①항 ...", "chunk_id": "law#art:제3조"}

        def handler(request):
            if request.url.params["name"] != "주택임대차보호법":
                return httpx.Response(404, json={"detail": "none"})
            return httpx.Response(200, json={"name": "주택임대차보호법", "articles": [article]})

        with self.fetched(handler):
            self.assertEqual(legal_backend.law("http://legal:8000", "주택임대차보호법"), {"name": "주택임대차보호법", "articles": [article]})
            self.assertIsNone(legal_backend.law("http://legal:8000", "없는법"))


def hit(document_id, score, text="본문", chunk=0, title="", **metadata):
    return {"score": score, "chunk": {"id": f"{document_id}_chunk_{chunk:04d}", "document_id": document_id,
                                      "document_type": "judgment", "title": title, "text": text, "metadata": metadata}}


class CaseSearchTests(unittest.TestCase):
    def search(self, hits, labels=None, statutes=None):
        def handler(request):
            self.path, self.sent = request.url.path, json.loads(request.content)
            return httpx.Response(200, json={"hits": hits, "labels": labels or {}, "statutes": statutes or {}})

        with served(handler):
            return legal_backend.search_cases("http://legal:8000", "임차권등기명령 보증금 반환")

    def test_the_case_search_of_the_service_is_asked(self):
        self.search([])

        self.assertEqual((self.path, self.sent), ("/v1/cases", {"query": "임차권등기명령 보증금 반환", "k": 20}))

    def test_a_card_carries_the_articles_its_case_cites(self):
        cited = [{"law": "주택임대차보호법", "article": "제3조의3", "paragraph": "제1항"},
                 {"law": "민사집행법", "article": "제283조", "paragraph": None}]

        cards = self.search([hit("대법원-2019다1", 0.9), hit("서울고등법원-2020나2", 0.8)], statutes={"대법원-2019다1": cited})

        self.assertEqual([card["statutes"] for card in cards], [cited, []])

    def test_a_case_named_only_in_its_document_id_shows_court_number_and_date(self):
        """The civil corpus keeps no case number in its metadata (광주고등법원-2014나2029)."""
        card = self.search([hit("광주고등법원-2014나2029", 0.91, text="피고는 원고에게 지급하라.", title="공사대금",
                                court="광주고등법원", original={"date": "2015-04-17T09:00:00.000+09:00"})])[0]

        self.assertEqual(card, {"title": "광주고등법원 2014나2029", "subject": "공사대금", "date": "2015.04.17", "kind": "판례",
                                "quote": "피고는 원고에게 지급하라.", "score": 0.91,
                                "chunk_id": "광주고등법원-2014나2029_chunk_0000", "statutes": []})

    def test_metadata_names_a_case_before_its_id_does(self):
        card = self.search([hit("criminal::judgment::100029", 0.8, court="대법원", case_number="84도2229",
                                original={"date": "1984.12.26"})])[0]

        self.assertEqual((card["title"], card["date"]), ("대법원 84도2229", "1984.12.26"))

    def test_a_case_with_no_name_uses_the_services_label_or_says_so(self):
        cards = self.search([hit("administrative::judgment::1", 0.7), hit("administrative::judgment::2", 0.6)],
                            labels={"administrative::judgment::1": "서울행정법원 2019구합1234"})

        self.assertEqual([card["title"] for card in cards], ["서울행정법원 2019구합1234", "판례 (법원·사건번호 미기재)"])
        self.assertEqual(cards[0]["date"], "")

    def test_a_broken_reply_is_reported_as_unavailable(self):
        with served(lambda request: httpx.Response(200, json={"results": []})), \
                self.assertRaises(legal_backend.LegalBackendUnavailable):
            legal_backend.search_cases("http://legal:8000", "x")


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
                                      "kind": "법령", "score": 0.91, "chunk_id": "c1"})
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
