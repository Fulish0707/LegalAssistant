"""One chat turn: analyze → clarify? → rewrite → retrieve → evidence → generate → verify → policy."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from legal_rag.config import Settings
from legal_rag.llm import ChatModel
from legal_rag.retrieval.hybrid import HybridRetriever, Retriever
from legal_rag.schemas import ChatResponse, EvidenceBundle, LegalAnswer, QueryAnalysis, Turn, UserDocument, VerificationAction, VerificationReport
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

    def handle(self, message: str, *, conversation_id: str, user_documents: Sequence[UserDocument] = ()) -> ChatResponse:
        recorder = TraceRecorder(
            conversation_id=conversation_id,
            user_text=message,
            include_text=self.settings.trace.include_text,
            path=self.settings.trace.path,
            model=getattr(self.generator.model, "name", None),
        )
        trace = recorder.trace
        history = self.store.get(conversation_id).turns
        self.store.append(conversation_id, Turn(role="user", content=message, created_at=_now()))

        with recorder.stage("analyze"):
            analysis = self.analyzer.analyze(message, history)
        trace.analysis = analysis

        if self.clarifier.should_clarify(analysis):
            recorder.finish(policy_action="clarify")
            self._remember(conversation_id, "\n".join(analysis.missing_facts), [])
            return ChatResponse(kind="clarification", conversation_id=conversation_id, questions=list(analysis.missing_facts), trace=trace)

        # Retrieval and generation work on a question that stands alone; the user's own words
        # (and so the verifier's user_refs) stay `message`.
        recent = history[-self.settings.rag.history_turns :] if self.settings.rag.history_turns > 0 else []
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

        with recorder.stage("context"):
            bundle = self.context_builder.build(hits)
        trace.evidence_ids = [item.chunk_id for item in bundle.items]

        # A stored answer has passed verification and stripping, so what it still cites was grounded
        # in its own turn. A clarification is the analyzer's unchecked text; it has no evidence ids.
        said = " ".join(turn.content for turn in recent if turn.role == "assistant" and turn.evidence_ids)
        prior_refs = [*(ref.render() for ref in extract_statute_refs(said) if ref.law_name), *extract_case_refs(said)]

        answer, report, action = self._generate_verified(question, bundle, analysis, recent, open_questions, recorder, user_documents, prior_refs)

        if answer is None or report is None:
            recorder.finish(policy_action=VerificationAction.reject.value)
            self._remember(conversation_id, "답변을 생성하지 못했습니다.", trace.evidence_ids)
            return ChatResponse(kind="rejected", conversation_id=conversation_id, reason="answer generation failed", evidence=bundle, trace=trace)

        if action == VerificationAction.strip:
            answer = self.policy.strip(answer, report)
            action = self.policy.decide_after_strip(answer, evidence_required=True)
        trace.verification = report
        recorder.finish(policy_action=action.value)

        if action == VerificationAction.reject:
            self._remember(conversation_id, "근거를 확인할 수 없어 답변을 제공하지 못했습니다.", trace.evidence_ids)
            return ChatResponse(kind="rejected", conversation_id=conversation_id, reason="unsupported citations", evidence=bundle, verification=report, trace=trace)

        self._remember(conversation_id, f"{answer.summary}\n{answer.analysis}".strip(), trace.evidence_ids)
        return ChatResponse(kind="answer", conversation_id=conversation_id, answer=answer, evidence=bundle, verification=report,
                            prior_refs=prior_refs, trace=trace)

    def _generate_verified(
        self, question: str, bundle: EvidenceBundle, analysis: QueryAnalysis, recent: list[Turn],
        open_questions: list[str], recorder: TraceRecorder, user_documents: Sequence[UserDocument] = (),
        prior_refs: Sequence[str] = (),
    ) -> tuple[LegalAnswer | None, VerificationReport | None, VerificationAction]:
        """Generate, verify, and regenerate with the verifier's feedback while the policy asks for it."""
        answer: LegalAnswer | None = None
        report: VerificationReport | None = None
        action = VerificationAction.accept
        feedback: str | None = None
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
                with recorder.stage(f"generate[{attempt}]"):
                    draft = self.generator.generate(question, bundle, feedback=feedback, history=recent, open_questions=open_questions,
                                                    user_documents=user_documents)
            except GenerationError:
                # The model broke the JSON twice, e.g. by looping past max_tokens. Strip the
                # draft that asked for this regeneration, or refuse; either beats a 500.
                action = VerificationAction.strip if answer is not None else VerificationAction.reject
                break
            with recorder.stage(f"verify[{attempt}]"):
                draft = draft.model_copy(update={"citations": [c for c in draft.citations if c.source_id not in document_ids]})
                answer = backfill_citations(draft, bundle)
                report = self.verifier.verify(answer, bundle, user_refs=user_refs, own_clauses=own_clauses, prior_refs=prior_refs)
            action = self.policy.decide(report, attempt)
            if action != VerificationAction.regenerate:
                break
            recorder.trace.regenerations += 1
            feedback = self.policy.feedback(report)
        return answer, report, action

    def _remember(self, conversation_id: str, content: str, evidence_ids: list[str]) -> None:
        self.store.append(conversation_id, Turn(role="assistant", content=content, created_at=_now(), evidence_ids=list(evidence_ids)))
