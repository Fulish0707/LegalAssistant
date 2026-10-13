"""Generate and verify answers from frozen evidence for Gemini evaluations.

The historical replay under data/eval remains a frozen experiment snapshot.
Settings and request pacing belong to a replay instance, shared across its rows.
"""
from __future__ import annotations

import threading
import time

import httpx

from legal_rag.config.settings import Settings
from legal_rag.factory import build_chat_model
from legal_rag.rag.generator import Generator, GenerationError
from legal_rag.schemas import EvidenceBundle, VerificationAction
from legal_rag.verification import Verifier, VerificationPolicy, backfill_citations
from legal_rag.verification.citations import answer_text, extract_statute_refs, extract_case_refs
from legal_rag.verification.policy import STRIP_NOTE


class _RetryModel:
    def __init__(self, model, wait_for_request):
        self.model = model
        self.wait_for_request = wait_for_request
        self.usage = []

    def generate(self, *args, **kwargs):
        for attempt in range(4):
            self.wait_for_request()
            try:
                result = self.model.generate(*args, **kwargs)
                self.usage.append(result.raw.get('usageMetadata', {}))
                return result
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                retry = isinstance(error, httpx.TransportError) or error.response.status_code in (429, 500, 502, 503, 504)
                if not retry or attempt == 3:
                    raise
                delay = 2 ** (attempt + 1)
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429:
                    details = error.response.json().get('error', {}).get('details', [])
                    if any('PerDay' in str(v.get('quotaId', '')) for d in details for v in d.get('violations', [])):
                        raise
                    delay = 60
                    for detail in details:
                        if 'retryDelay' in detail:
                            delay = max(delay, float(detail['retryDelay'].rstrip('s')))
                time.sleep(delay)


class AnswerReplay:
    """One Gemini run's settings and shared request-start rate limiter.

    Results retain the historical replay schema, including full_text and the
    1500-character preview. Golden-answer scoring uses full_text explicitly.
    """

    def __init__(self, settings: Settings, *, request_interval: float = 15.0):
        if settings.llm.backend != 'gemini':
            raise ValueError('Answer replay requires the Gemini backend')
        if request_interval < 0:
            raise ValueError('Request interval must be nonnegative')
        self.settings = settings
        self.request_interval = request_interval
        self._request_lock = threading.Lock()
        self._last_request = 0.0

    def _wait_for_request(self):
        with self._request_lock:
            delay = self.request_interval - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()

    def generate(self, row):
        settings = self.settings
        model = build_chat_model(settings)
        retry_model = _RetryModel(model, self._wait_for_request)
        generator = Generator(retry_model, temperature=settings.llm.temperature, max_tokens=settings.llm.max_tokens)
        bundle = EvidenceBundle.model_validate(row['evidence'])
        verifier, policy = Verifier(), VerificationPolicy(settings.rag.max_regenerations)
        user_refs = [r.render() for r in extract_statute_refs(row['question'])] + extract_case_refs(row['question'])
        started, attempts, answer, report, feedback = time.perf_counter(), [], None, None, None
        action = VerificationAction.accept
        result = {'key': row['key'], 'model': model.name}
        try:
            for attempt in range(settings.rag.max_regenerations + 1):
                try:
                    draft = generator.generate(row['question'], bundle, feedback=feedback)
                except GenerationError:
                    action = VerificationAction.strip if answer is not None else VerificationAction.reject
                    break
                answer = backfill_citations(draft, bundle)
                report = verifier.verify(answer, bundle, user_refs=user_refs)
                action = policy.decide(report, attempt)
                attempts.append({'ok': report.ok, 'issues': [i.model_dump() for i in report.issues], 'action': action.value})
                if action != VerificationAction.regenerate:
                    break
                feedback = policy.feedback(report)
            if answer is not None and report is not None and action == VerificationAction.strip:
                answer = policy.strip(answer, report)
                action = policy.decide_after_strip(answer, evidence_required=True)
            if answer is None or report is None or action == VerificationAction.reject:
                result.update(kind='rejected', final_text='', final_ok=False)
            else:
                full = answer_text(answer).split(STRIP_NOTE)[0]
                result.update(kind='answer', answer=answer.model_dump(mode='json'), full_text=full,
                              final_text=full[:1500], stored_text_truncated=len(full) > 1500, final_ok=report.ok)
            result.update(attempts=attempts, action=action.value)
        except httpx.HTTPStatusError as error:
            result.update(kind='error', error=f'Gemini HTTP {error.response.status_code}', final_text='')
            key = settings.llm.api_key.get_secret_value()
            result['error_detail'] = error.response.text.replace(key, '[REDACTED]')[:2000]
        except httpx.TransportError as error:
            result.update(kind='error', error=type(error).__name__, final_text='')
        finally:
            model._client.close()
        result.update(seconds=round(time.perf_counter() - started, 2), usage=retry_model.usage)
        return result
