"""One chat turn: analyze → clarify? → rewrite → retrieve → evidence → generate → verify → policy."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import threading

from legal_rag.config import Settings
from legal_rag.llm import ChatModel
from legal_rag.retrieval.hybrid import HybridRetriever, Retriever
from legal_rag.schemas import (ChatResponse, Chunk, Confidence, EvidenceBundle, LegalAnswer, QueryAnalysis, ScoredChunk, Turn,
                               UserDocument, VerificationAction, VerificationReport)
from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs
from legal_rag.verification import VerificationPolicy, Verifier, backfill_citations

from .analyzer import QueryAnalyzer
from .clarify import ClarificationPolicy
from .condenser import QueryCondenser
from .context import ContextBuilder
from .conversation import ConversationStore
from .generator import GenerationError, Generator
from .rewriter import QueryRewriter
from .trace import TraceRecorder


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _unconfirmed_case_sources(bundle: EvidenceBundle, support_roles: dict[str, str]) -> list[str]:
    """Case/interpretation passages not confirmed as reasoning or a legal rule at a cited quote:
    unknown roles and passages the assessment never cited. Statutes need no role."""
    return [item.source_id for item in bundle.items
            if item.header.get("Type") != "Statute" and support_roles.get(item.source_id) not in {"finding", "legal_rule"}]


class Orchestrator:
    def __init__(
        self,
        *,
        analyzer: QueryAnalyzer,
        clarifier: ClarificationPolicy,
        rewriter: QueryRewriter,
        retriever: Retriever,
        context_builder: ContextBuilder,
        generator: Generator,
        verifier: Verifier,
        policy: VerificationPolicy,
        store: ConversationStore,
        settings: Settings,
        condenser: QueryCondenser | None = None,
    ) -> None:
        self.condenser = condenser
        self.analyzer = analyzer
        self.clarifier = clarifier
        self.rewriter = rewriter
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator
        self.verifier = verifier
        self.policy = policy
        self.store = store
        self.settings = settings
        self._pilot_state = None
        self._pilot_state_lock = threading.Lock()

    @classmethod
    def from_components(cls, *, settings: Settings, retriever: Retriever, model: ChatModel, store: ConversationStore, verifier: Verifier, policy: VerificationPolicy) -> "Orchestrator":
        return cls(
            analyzer=QueryAnalyzer(model, history_turns=settings.rag.history_turns),
            clarifier=ClarificationPolicy(settings.rag.allow_clarification),
            condenser=QueryCondenser(model, history_turns=settings.rag.history_turns),
            rewriter=QueryRewriter(model if settings.rag.llm_rewrite else None, analyzer_terms=settings.rag.analyzer_terms_in_search),
            retriever=retriever,
            context_builder=ContextBuilder(
                settings.rag.max_evidence_chars,
                neighbor_radius=settings.rag.context_neighbor_radius,
                max_context_chars=settings.rag.max_context_chars,
                neighbor_corpora=settings.rag.context_neighbor_corpora,
                neighbor_lookup=getattr(retriever, "neighbor_chunks", None),
            ),
            generator=Generator(model, temperature=settings.llm.temperature, max_tokens=settings.llm.max_tokens,
                                plain_text=settings.llm.backend == "transformers", mask_case_figures=settings.rag.mask_case_figures,
                                answer_style=settings.rag.answer_style),
            verifier=verifier,
            policy=policy,
            store=store,
            settings=settings,
        )

    def handle(self, message: str, *, conversation_id: str, user_documents: Sequence[UserDocument] = (),
               pinned: Sequence[Chunk] = (), on_draft=None, on_stage=None,
               policy_scope=None, request_id: str | None = None) -> ChatResponse:
        """Default path is unchanged; the explicit pilot adds persistent request/budget control."""
        kwargs = dict(conversation_id=conversation_id, user_documents=user_documents,
                      pinned=pinned, on_draft=on_draft, on_stage=on_stage)
        if policy_scope is None:
            return self._run_turn(message, **kwargs)
        from legal_rag.schemas.action import PolicyScope
        scope = PolicyScope.model_validate(policy_scope)
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
            raise ValueError("An explicit request_id is required for the policy pilot")
        payload = {"message": message, "scope": scope.model_dump(mode="json"),
                   "documents": [doc.model_dump(mode="json") for doc in user_documents],
                   "pinned": [chunk.model_dump(mode="json") for chunk in pinned]}
        fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        state_store = self._policy_state()
        cached = state_store.claim_request(conversation_id, request_id, fingerprint)
        if cached is not None:
            return ChatResponse.model_validate_json(cached)
        try:
            response = self._run_turn(message, **kwargs, policy_scope=scope, request_id=request_id)
            state_store.complete_request(conversation_id, request_id, response.model_dump_json())
            return response
        except Exception:
            state_store.fail_request(conversation_id, request_id)
            raise

    def _policy_state(self):
        # Creating an orchestrator with the feature off creates no policy tables.
        from .policy_state import InMemoryPolicyStateStore, SQLitePolicyStateStore
        with self._pilot_state_lock:
            if self._pilot_state is None:
                path = getattr(self.store, "path", None)
                self._pilot_state = SQLitePolicyStateStore(path) if path is not None else InMemoryPolicyStateStore()
            return self._pilot_state

    def _run_turn(self, message: str, *, conversation_id: str, user_documents: Sequence[UserDocument] = (),
                  pinned: Sequence[Chunk] = (), on_draft=None, on_stage=None,
                  policy_scope=None, request_id: str | None = None) -> ChatResponse:
        """One chat turn. `pinned` are passages the user chose to ask about; they lead the evidence.
        `on_draft` receives the analysis as the model writes it, and None before each new attempt (see Generator).
        `on_stage` receives each stage as it starts: analyze, search, write, verify, and rewrite for a regeneration."""
        announce = on_stage or (lambda stage: None)
        recorder = TraceRecorder(
            conversation_id=conversation_id,
            user_text=message,
            include_text=self.settings.trace.include_text,
            path=self.settings.trace.path,
            model=getattr(self.generator.model, "name", None),
        )
        trace = recorder.trace
        history_id = conversation_id
        if policy_scope is not None:
            history_id = "policy:" + hashlib.sha256(json.dumps(
                [conversation_id, policy_scope.matter_id, policy_scope.issue_id], ensure_ascii=False,
            ).encode()).hexdigest()
        history = self.store.get(history_id).turns
        self.store.append(history_id, Turn(role="user", content=message, created_at=_now()))

        announce("analyze")
        with recorder.stage("analyze"):
            analysis = self.analyzer.analyze(message, history)
        trace.analysis = analysis

        if policy_scope is None and self.clarifier.should_clarify(analysis):
            recorder.finish(policy_action="clarify")
            self._remember(history_id, "\n".join(analysis.missing_facts), [])
            return ChatResponse(kind="clarification", conversation_id=conversation_id, questions=list(analysis.missing_facts), trace=trace)

        # Retrieval and generation work on a question that stands alone; the user's own words
        # (and so the verifier's user_refs) stay `message`.
        recent = history[-self.settings.rag.history_turns :] if self.settings.rag.history_turns > 0 else []
        announce("search")
        with recorder.stage("condense"):
            question = self.condenser.condense(message, history) if self.condenser else message

        # Facts the analyzer wanted but the turn is answered without; the answer states them.
        open_questions = list(analysis.missing_facts) if analysis.needs_more_info else []

        with recorder.stage("rewrite"):
            queries = self.rewriter.rewrite(question, analysis)
            filters = self.rewriter.filters(analysis) if self.settings.rag.hard_filter else None
        trace.search_queries = list(queries.search_queries)

        k = self.settings.retrieval.k_final
        with recorder.stage("retrieve"):
            if isinstance(self.retriever, HybridRetriever):
                hits = self.retriever.retrieve_many(queries.search_queries, k, filters, rerank_query=question)
            else:
                hits = self.retriever.retrieve(question, k, filters)
        trace.retrieved = dict(getattr(self.retriever, "last_trace", {}) or {})
        if pinned:
            # The chosen passages take the first places; the evidence stays k passages long, so the
            # prompt stays inside the context the model was measured with.
            chosen = [ScoredChunk(chunk=chunk, score=1.0, retriever="pinned") for chunk in pinned]
            held = {chunk.id for chunk in pinned}
            hits = (chosen + [hit for hit in hits if hit.chunk.id not in held])[:k]

        with recorder.stage("context"):
            bundle = self.context_builder.build(hits)
        trace.evidence_ids = [item.chunk_id for item in bundle.items]

        if policy_scope is not None:
            from .action_policy import decide
            from .sufficiency import SufficiencyAssessor
            from legal_rag.schemas.trace import PostRetrievalDecision
            assert request_id is not None  # checked before claiming the pilot request
            state_store = self._policy_state()
            # Two assessments at most: first evidence, then one research result.
            for revision in range(2):
                announce("assess")
                with recorder.stage(f"assess[{revision}]"):
                    assessed = SufficiencyAssessor(
                        self.generator.model, document_text=getattr(self.retriever, "document_text", None),
                    ).assess(
                        message, history, user_documents, bundle,
                        issue_id=policy_scope.issue_id, matter_id=policy_scope.matter_id,
                        mask_case_figures=self.generator.mask_case_figures,
                    )
                policy_input = assessed.policy_input.model_copy(update={
                    "clarification_used": state_store.clarification_used(conversation_id, policy_scope.matter_id, policy_scope.issue_id),
                    "research_used": state_store.research_used(conversation_id, request_id),
                })
                decision = decide(policy_input)
                if decision.action == "clarify" and not state_store.reserve_clarification(
                    conversation_id, policy_scope.matter_id, policy_scope.issue_id,
                ):
                    decision = decide(policy_input.model_copy(update={"clarification_used": True}))
                if decision.action == "research" and not state_store.reserve_research(conversation_id, request_id):
                    decision = decide(policy_input.model_copy(update={"research_used": True}))
                trace.post_retrieval.append(PostRetrievalDecision(
                    action=decision.action, reason=decision.reason, bundle_revision=revision,
                    evidence_ids=[item.chunk_id for item in bundle.items], support_roles=assessed.support_roles,
                ))
                if decision.action == "clarify":
                    recorder.finish(policy_action="clarify")
                    self._remember(history_id, decision.question, [])
                    return ChatResponse(kind="clarification", conversation_id=conversation_id,
                                        questions=[decision.question], evidence=bundle, trace=trace)
                if decision.action == "withhold":
                    return self._withhold_response(decision, assessed, bundle, recorder, conversation_id, history_id)
                if decision.action == "answer":
                    break
                unmet = set(decision.unresolved_keys)
                search_intent = " ".join(item.proposition for item in assessed.assessment.evidence if item.key in unmet)
                research_query = question + "\n" + search_intent
                announce("research")
                with recorder.stage("research"):
                    if isinstance(self.retriever, HybridRetriever):
                        extra_hits = self.retriever.retrieve_many([research_query], k, filters, rerank_query=question)
                    else:
                        extra_hits = self.retriever.retrieve(research_query, k, filters)
                    # New evidence gets priority after pinned passages; keep old unique sources.
                    seen = set()
                    merged = []
                    for hit in [*(hit for hit in hits if hit.retriever == "pinned"), *extra_hits, *hits]:
                        key = (hit.chunk.metadata.corpus_version, hit.chunk.id)
                        if key not in seen:
                            seen.add(key)
                            merged.append(hit)
                    hits = merged[:k]
                    bundle = self.context_builder.build(hits)
                    trace.search_queries.append(research_query)
                    trace.evidence_ids = [item.chunk_id for item in bundle.items]
            else:
                raise RuntimeError("Policy exhausted its bounded decision loop without a terminal action")
            open_questions = []
            unconfirmed = _unconfirmed_case_sources(bundle, assessed.support_roles)
        else:
            unconfirmed = []

        # A stored answer has passed verification and stripping, so what it still cites was grounded
        # in its own turn. A clarification is the analyzer's unchecked text; it has no evidence ids.
        said = " ".join(turn.content for turn in recent if turn.role == "assistant" and turn.evidence_ids)
        prior_refs = [*(ref.render() for ref in extract_statute_refs(said) if ref.law_name), *extract_case_refs(said)]

        asked = message if self.settings.rag.generate_from_message else question
        answer, report, action = self._generate_verified(asked, bundle, analysis, recent, open_questions, recorder, user_documents, prior_refs,
                                                         on_draft, announce, suppress_followup_questions=policy_scope is not None,
                                                         unconfirmed_case_sources=unconfirmed)

        if answer is None or report is None:
            recorder.finish(policy_action=VerificationAction.reject.value)
            self._remember(history_id, "답변을 생성하지 못했습니다.", trace.evidence_ids)
            return ChatResponse(kind="rejected", conversation_id=conversation_id, reason="answer generation failed", evidence=bundle, trace=trace)

        if action == VerificationAction.strip:
            answer = self.policy.strip(answer, report)
            action = self.policy.decide_after_strip(answer, evidence_required=True)
        trace.verification = report
        recorder.finish(policy_action=action.value)

        if action == VerificationAction.reject:
            self._remember(history_id, "근거를 확인할 수 없어 답변을 제공하지 못했습니다.", trace.evidence_ids)
            return ChatResponse(kind="rejected", conversation_id=conversation_id, reason="unsupported citations", evidence=bundle, verification=report, trace=trace)

        cited = sorted({citation.source_id for citation in answer.citations} & set(unconfirmed))
        if cited:
            # Deterministic, whatever the writer did with the instruction: say it, and never "high".
            notice = (f"인용한 판례·결정례 중 {', '.join(cited)}은 법원·기관이 판단한 부분인지 자동으로 확인되지 않았습니다. "
                      "비슷한 사건에서 같은 결과가 나온다고 단정할 수 없으니, 판결문 원문을 확인하거나 전문가와 상담하세요.")
            answer = answer.model_copy(update={
                "uncertainty": "\n".join(part for part in (notice, answer.uncertainty) if part),
                "confidence": Confidence.medium if answer.confidence == Confidence.high else answer.confidence,
            })
            trace.unverified_case_sources = cited
        self._remember(history_id, f"{answer.summary}\n{answer.analysis}".strip(), trace.evidence_ids)
        return ChatResponse(kind="answer", conversation_id=conversation_id, answer=answer, evidence=bundle, verification=report,
                            prior_refs=prior_refs, trace=trace, suppress_followup_questions=policy_scope is not None)

    def _withhold_response(self, decision, assessed, bundle, recorder, conversation_id, history_id):
        from legal_rag.schemas.action import WithheldDetails
        explanations = {
            "facts_unresolved": "같은 쟁점에서 사실 확인을 한 번 요청했지만, 결론을 좌우하는 사실이 아직 확인되지 않아 결론을 보류합니다.",
            "evidence_unresolved": "추가 검색에서도 핵심 법률 근거를 충분히 확보하지 못해 결론을 보류합니다. 이는 공제가 불가능하다는 뜻이 아닙니다.",
            "assessment_failed": "사실·근거 판정 결과를 검증하지 못해 결론을 보류합니다. 사용자 사실 부족이나 법률 근거 부재가 확인된 것은 아닙니다.",
        }
        labels = {"termination_agreement": "계약 후 별도 종료 합의 여부",
                  "deduction_basis": "종료 합의·특약 아래 차임 공제의 법률 근거",
                  "clause_meaning": "해당 특약의 의미를 설명할 법률 근거"}
        known = []
        if assessed.policy_input.assessment_status == "valid":
            known = list(dict.fromkeys(observation.quote for fact in assessed.assessment.facts
                                       if fact.status == "known" for observation in fact.observations))
        details = WithheldDetails(reason_code=decision.reason, known_facts=known,
                                  unresolved_items=[labels.get(key, key) for key in decision.unresolved_keys],
                                  explanation=explanations[decision.reason])
        recorder.finish(policy_action="withheld")
        self._remember(history_id, details.explanation, [])
        return ChatResponse(kind="withheld", conversation_id=conversation_id, withheld=details,
                            evidence=bundle, trace=recorder.trace)

    def _generate_verified(
        self, question: str, bundle: EvidenceBundle, analysis: QueryAnalysis, recent: list[Turn],
        open_questions: list[str], recorder: TraceRecorder, user_documents: Sequence[UserDocument] = (),
        prior_refs: Sequence[str] = (), on_draft=None, announce=lambda stage: None, suppress_followup_questions: bool = False,
        unconfirmed_case_sources: Sequence[str] = (),
    ) -> tuple[LegalAnswer | None, VerificationReport | None, VerificationAction]:
        """Generate, verify, and regenerate with the verifier's feedback while the policy asks for it."""
        answer: LegalAnswer | None = None
        report: VerificationReport | None = None
        action = VerificationAction.accept
        no_followup = (
            "이 쟁점은 검색 후 정책이 필요한 사실을 확인했습니다. 후속 질문을 다시 요청하지 마세요. "
            "additional_information_needed는 빈 목록으로 두고, 남은 한계는 uncertainty에 설명하세요."
        ) if suppress_followup_questions else ""
        if unconfirmed_case_sources:
            no_followup += (
                f" 다음 판례·결정례 출처는 인용 부분이 법원·기관의 판단인지 확인되지 않았습니다: {', '.join(unconfirmed_case_sources)}. "
                "이 출처만으로 사용자 사건의 결론을 단정하지 마세요. '비슷한 사건에서 …라고 본 사례가 있다'처럼 쓰고, "
                "사건마다 결과가 달라질 수 있음을 uncertainty에 적으세요."
            )
        feedback: str | None = no_followup or None
        # A document is not a legal source: listing one under citations is dropped, not treated as
        # a fabricated source. An article it names with its law, or a case number, counts as the
        # user's own words. Its bare 제4조 is the contract's own clause: the answer may point to
        # it without a law, and it must not let any law's 제4조 through.
        document_ids = {f"D{number}" for number in range(1, len(user_documents) + 1)}
        refs = [ref for document in user_documents for ref in extract_statute_refs(document.text)]
        own_clauses = {ref.article for ref in refs if ref.law_name is None}
        user_refs = [*analysis.statute_refs, *analysis.case_refs, *(ref.render() for ref in refs if ref.law_name),
                     *(case for document in user_documents for case in extract_case_refs(document.text))]
        for attempt in range(self.settings.rag.max_regenerations + 1):
            try:
                announce("rewrite" if attempt else "write")
                with recorder.stage(f"generate[{attempt}]"):
                    draft = self.generator.generate(question, bundle, feedback=feedback, history=recent, open_questions=open_questions,
                                                    user_documents=user_documents, on_draft=on_draft)
            except GenerationError:
                # The model broke the JSON twice, e.g. by looping past max_tokens. Strip the
                # draft that asked for this regeneration, or refuse; either beats a 500.
                action = VerificationAction.strip if answer is not None else VerificationAction.reject
                break
            if suppress_followup_questions:
                draft = draft.model_copy(update={"additional_information_needed": []})
            announce("verify")
            with recorder.stage(f"verify[{attempt}]"):
                draft = draft.model_copy(update={"citations": [c for c in draft.citations if c.source_id not in document_ids]})
                answer = backfill_citations(draft, bundle)
                report = self.verifier.verify(answer, bundle, user_refs=user_refs, own_clauses=own_clauses, prior_refs=prior_refs)
            action = self.policy.decide(report, attempt)
            if action != VerificationAction.regenerate:
                break
            recorder.trace.regenerations += 1
            feedback = "\n".join(part for part in (no_followup, self.policy.feedback(report)) if part)
        return answer, report, action

    def _remember(self, conversation_id: str, content: str, evidence_ids: list[str]) -> None:
        self.store.append(conversation_id, Turn(role="assistant", content=content, created_at=_now(), evidence_ids=list(evidence_ids)))
