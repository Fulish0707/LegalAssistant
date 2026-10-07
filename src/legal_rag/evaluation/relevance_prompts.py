"""Question-corpus guidance and routing for relevance evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .corpus_guidance import CORPUS_GUIDANCE

DEFAULT_PROMPT_PROFILE = 'corpus'
ROUTING_VERSION = 2
ACTIVE_CORPUS_DOMAINS = frozenset({'civil', 'administrative', 'ip'})
GUIDANCE = dict(CORPUS_GUIDANCE)
GUIDANCE['criminal'] = """[형사 자료의 평가 유의사항: 쟁점과 적용 대상 확인]
자료 묶음의 이름은 법적 쟁점을 결정하지 않습니다. 질문이 행정·노동·연금 등 다른 쟁점을 묻는다면 그 질문에 맞춰 공통 기준을 적용하십시오.
먼저 질문이 묻는 대상(기관·위원회·행위·권리)과 요구하는 내용(요건·허용 예외·금지·적용 법조·개별 위반 이유)을 확인하십시오. 참조 답변은 대상을 식별하는 힌트로 사용하되 잘못된 조문 번호를 정답 조건으로 강제하지 마십시오.
다른 기관이나 다른 법률의 위원회에 관한 규정은 임기 등 수치가 같아도 질문의 직접 근거가 아니며 0입니다. 같은 법률 안에서도 거짓 보고서 작성과 검사원 질문에 대한 거짓 진술처럼 별개의 행위·책임을 혼동하지 마십시오.
허용되는 경우나 예외를 묻는 질문에 금지 규정·위반 사례만 있으면 같은 쟁점의 부분 근거인 1입니다. 제한의 시간·횟수를 설명한다는 이유만으로 요청된 허용 예외를 모두 설명한 것으로 보지 마십시오.
공소사실·검사나 피고인의 주장과 법원의 독립적인 판단을 구별하십시오. 법적 요건이나 적용 법조를 묻는다면 해당 요건·조문의 직접 명시가 2의 근거가 됩니다. 특정 인물의 개별 위반 이유를 묻는다면 일반적인 처벌 조문만으로 그 사건의 구체적 원인을 추정하지 마십시오.
같은 범죄의 양형 사유만으로 구성요건을 설명했다고 보지 마십시오. 같은 쟁점의 사실·적용만 있으면 1이며, 다른 대상·행위·절차이면 0입니다. 번호가 없더라도 법리가 직접 서술되어 있으면 공통 2 기준을 적용할 수 있습니다.
evidence_quote는 오직 문서 조각에서 그대로 복사하십시오. 참조 답변을 원문 인용으로 사용하지 마십시오. 문서에서 짧은 완결 문장을 먼저 찾고, 원하는 문장이 없으면 만들어 쓰지 마십시오."""


def canonical_domain(value: str | None) -> str | None:
    value = (value or '').strip().lower()
    if value == 'intellectual_property':
        return 'ip'
    for domain in GUIDANCE:
        if value == domain or value.startswith(domain + '-'):
            return domain
    return None


def validate_profile(profile: str, domain: str | None = None) -> None:
    if profile not in {'shared', 'corpus', 'corpus-experimental'}:
        raise ValueError('prompt profile must be shared, corpus or corpus-experimental')
    if domain is not None and canonical_domain(domain) is None and domain not in {'mixed', 'unknown'}:
        raise ValueError('judge domain must be criminal, civil, administrative, ip, mixed or unknown')


def question_domain(pair: dict, override: str | None = None) -> str | None:
    if override is not None:
        return canonical_domain(override)
    for field in ('question_domain', 'evaluation_domain', 'legal_domain'):
        if field in pair and pair[field] is not None:
            return canonical_domain(pair[field])
    return None


def system_prompt(shared: str, pair: dict, *, profile: str, domain: str | None = None) -> str:
    validate_profile(profile, domain)
    selected = guidance_domain(pair, profile=profile, domain=domain)
    return shared + '\n\n' + GUIDANCE[selected] if selected else shared


def guidance_domain(pair: dict, *, profile: str, domain: str | None = None) -> str | None:
    selected = question_domain(pair, domain)
    if profile == 'corpus-experimental' or (profile == 'corpus' and selected in ACTIVE_CORPUS_DOMAINS):
        return selected
    return None


def attach_question_domains(pairs: list[dict], pool_dir: Path) -> list[dict]:
    """Recover old pools' question domains from verified target metadata, never passage IDs."""
    meta_path = Path(pool_dir) / 'pool.json'
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if all(any(pair.get(field) is not None for field in ('question_domain', 'evaluation_domain', 'legal_domain'))
           for pair in pairs):
        return [dict(pair) for pair in pairs]
    mapping: dict[str, str | None] = {}
    for label, source in (metadata.get('case_sources') or {}).items():
        domain = canonical_domain(label)
        if domain is None:
            continue
        path = Path(source['target'])
        if not path.is_absolute() and not path.exists():
            path = Path(pool_dir) / path
        data = path.read_bytes()
        expected = source.get('target_sha256')
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'question-domain target hash mismatch: {path}')
        for line in data.decode('utf-8').splitlines():
            row = json.loads(line)
            case_id = row.get('case_id', row.get('id'))
            if case_id in mapping and mapping[case_id] != domain:
                mapping[case_id] = None
            else:
                mapping[case_id] = domain
    single_domain = canonical_domain(metadata.get('domain') or (metadata.get('filters') or {}).get('legal_domain'))
    output = []
    for pair in pairs:
        if any(pair.get(field) is not None for field in ('question_domain', 'evaluation_domain', 'legal_domain')):
            output.append(dict(pair))
        else:
            output.append(dict(pair, question_domain=mapping.get(pair['case_id'], single_domain)))
    return output
