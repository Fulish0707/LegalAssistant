"""Constrained legal-term query expansion for the C5 development experiment."""
from __future__ import annotations

import hashlib
import json
import re
from pydantic import BaseModel
from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs

MODEL="qwen3:4b"
TEMPERATURE=0.0
MAX_TOKENS=256
NUM_CTX=2048
SYSTEM=("당신은 한국 법률 검색 질의 확장기입니다. 답을 작성하지 마세요. "
        "legal_query에는 원래 질문을 반복하지 말고 검색에 도움이 되는 추가 법률 용어만 넣으세요. "
        "질문에 없는 조문·사건번호나 사실은 만들지 마세요. 최대 10개 어절로 짧게 쓰세요. "
        "프로그램이 원래 질문 뒤에 이 용어를 붙입니다. JSON만 반환하세요.")
USER="원래 질문: {query}\nJSON 형식: {{\"legal_query\": \"추가 법률 용어\"}}"

class ExpansionOutput(BaseModel):
    legal_query: str

def prompt_sha256():
    return hashlib.sha256(json.dumps([SYSTEM,USER,ExpansionOutput.model_json_schema(),
                                      MODEL,TEMPERATURE,MAX_TOKENS,NUM_CTX,False],
                                     ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def _identifiers(text):
    articles={ref.article for ref in extract_statute_refs(text)}
    cases=set(extract_case_refs(text))
    other=set(re.findall(r"제\s*\d+\s*(?:항|호)",text))
    return articles | cases | {re.sub(r"\s+","",item) for item in other}

def validate_expansion(original, candidate):
    original=" ".join(original.split())
    candidate=" ".join(candidate.split()) if isinstance(candidate,str) else ""
    if not candidate:
        return None,"empty"
    if not candidate.startswith(original) or (len(candidate)>len(original) and
                                               not candidate[len(original)].isspace()):
        return None,"original_not_preserved"
    if len(candidate.split())>30:
        return None,"too_many_terms"
    if _identifiers(candidate)-_identifiers(original):
        return None,"novel_identifier"
    if candidate==original:
        return None,"no_expansion"
    return candidate,None

def generate_expansion(model, original):
    from legal_rag.llm.base import Message, StructuredOutputError
    try:
        response=model.generate([Message(role="system",content=SYSTEM),
                                 Message(role="user",content=USER.format(query=original))],
                                response_schema=ExpansionOutput,temperature=TEMPERATURE,
                                max_tokens=MAX_TOKENS)
    except StructuredOutputError as exc:
        return {"legal_query":None,"raw":exc.text,"fallback_reason":"malformed"}
    terms=" ".join(response.parsed.legal_query.split())
    candidate=original+(" "+terms if terms else "")
    accepted,reason=validate_expansion(original,candidate)
    return {"legal_query":accepted,"proposed_terms":terms,
            "raw":response.text,"fallback_reason":reason}
