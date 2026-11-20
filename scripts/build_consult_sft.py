"""Build the consulting-format SFT pilot (C plan): keep the reference answer, add grounded preparations/cautions.

For each sampled training row (real judgments, interpretations and statutes, from legal-qa-full-v2/train),
a local Apache-2.0 model (Qwen3-8B via Ollama) extracts preparations and cautions, each with a verbatim
quote from the row's passages. Code keeps only items whose quote is literally in the passages and whose
text names no statute or case absent from them. The target is a production LegalAnswer JSON built from
the reference answer plus the surviving items, and the prompt is the production answer prompt, captured
from the real Generator so training and serving see the same messages.

Usage:
  python scripts/build_consult_sft.py extract --candidates 450 --out data/training/consult-sft-v1
  python scripts/build_consult_sft.py assemble --out data/training/consult-sft-v1 --train 300 --validation 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SOURCE = ROOT / "data/training/legal-qa-full-v2/train.jsonl"
SEPARATOR = "\n\n[발췌 구분]\n\n"
DOMAIN_QUOTA = {"civil": 0.35, "criminal": 0.25, "administrative": 0.25, "intellectual_property": 0.15}
SEED = 20261120
EXTRACTOR = "qwen3:8b"


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=2, max_length=90)
    quote: str = Field(min_length=6, max_length=80)


class Preparation(BaseModel):
    """A material the user can gather, and the fact it shows. Rendered as an Item for the answer."""
    model_config = ConfigDict(extra="forbid")
    material: str = Field(min_length=2, max_length=30)
    proves: str = Field(min_length=2, max_length=50)
    quote: str = Field(min_length=6, max_length=80)

    def item(self) -> Item:
        return Item(text=f"{self.material.strip()}: {self.proves.strip()}", quote=self.quote)


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preparations: list[Preparation] = Field(default_factory=list, max_length=3)
    cautions: list[Item] = Field(default_factory=list, max_length=3)


class SupportCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[bool]


EXTRACT_SYSTEM = (
    "당신은 판결문에서 법률 상담에 필요한 정보를 뽑는 도우미입니다. 아래 자료와 질문, 참고 답변을 읽고 두 가지를 뽑으세요.\n"
    "1. preparations: 같은 쟁점으로 다투는 사람이 모아야 할 증거·자료.\n"
    "   - material은 문서·기록·증거물의 이름입니다(예: 계약서, 영수증, 문자 기록, 진단서, 감정 결과). 법률 요건이나 결론 문장을 쓰지 마세요.\n"
    "   - proves는 그 자료로 보여 줄 사실입니다. 법원이 결론을 내릴 때 기준으로 삼은 사실이어야 합니다.\n"
    "   - quote는 법원이 그 사실을 기준으로 삼았다는 것을 보여 주는 자료 문장입니다.\n"
    "2. cautions: 결론을 바꿀 수 있는 예외·기한·조건·요건. text는 자료에 적힌 내용과 같은 방향이어야 합니다.\n"
    "규칙:\n"
    "- quote는 자료 문장을 한 글자도 바꾸지 말고 그대로 복사하세요(80자 이내).\n"
    "- 법원이 배척한 주장('이유 없다', '받아들일 수 없다')을 해야 할 일처럼 쓰지 마세요.\n"
    "- 자료에 근거 문장이 없으면 그 항목을 쓰지 마세요. 해당 없으면 빈 목록으로 두세요. 억지로 채우지 마세요.\n"
    "- 자료에 없는 서류·절차 이름이나 조문·사건번호를 만들어 내지 마세요. 일반인이 이해할 수 있는 짧은 한국어로 쓰세요."
)

CHECK_SYSTEM = (
    "각 항목이 함께 제시된 자료 문장으로 뒷받침되는지 판정하세요. 문장과 같은 내용이거나 문장에서 바로 도출되면 true, "
    "문장과 반대이거나(예: 법원이 배척한 주장을 해야 할 일로 씀), 문장과 무관하거나, 문장에 없는 내용을 덧붙였으면 false입니다. "
    "준비물 항목은 자료 이름이 문서·기록·증거물이 아니고 법률 요건이면 false입니다. 항목 순서대로 verdicts에 답하세요."
)

COURT_CASE_ID = re.compile(r"\d{4}[가-힣]+\d+")
# Questions that only ask what a party argued or how the court ruled on an appeal ground are not consultations.
ARGUMENT_QUESTION = re.compile(r"주장한 이유|주장은 무엇|주장하였|주장한 내용|항소이유|상고이유|어떻게 판단")


def is_court_judgment(document_id: str) -> bool:
    """Court judgments only: statutes, constitutional decisions, interpretations and patent-office decisions are out."""
    if document_id.count("::") >= 2:
        return document_id.split("::")[1] == "judgment"
    return bool(COURT_CASE_ID.search(document_id))


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def passages(context: str) -> list[str]:
    return [p.strip() for p in context.split(SEPARATOR) if p.strip()]


def validate_items(items: list[Item], context: str) -> tuple[list[Item], list[dict]]:
    """Keep items whose quote is literally in the passages and whose text adds no foreign authority."""
    from legal_rag.textutil.refs import extract_case_refs, extract_statute_refs

    haystack = norm(context)
    context_statutes = {r.article for r in extract_statute_refs(context)}
    context_cases = set(extract_case_refs(context))
    kept, dropped = [], []
    seen = set()
    for item in items:
        quote = norm(item.quote).strip("\"'“”‘’")
        if len(quote) < 6 or quote not in haystack:
            dropped.append({"text": item.text, "reason": "quote_not_in_passages"})
            continue
        foreign = [r.render() for r in extract_statute_refs(item.text) if r.article not in context_statutes]
        foreign += [c for c in extract_case_refs(item.text) if c not in context_cases]
        if foreign:
            dropped.append({"text": item.text, "reason": f"foreign_authority:{foreign}"})
            continue
        key = norm(item.text)
        if key in seen:
            dropped.append({"text": item.text, "reason": "duplicate"})
            continue
        seen.add(key)
        kept.append(Item(text=item.text.strip(), quote=quote))
    return kept, dropped


LAW_BEFORE_ARTICLE = re.compile(r"((?:[가-힣·]+ ){0,5}[가-힣·]*(?:법률|법|령|규칙))\s*제\s*\d+\s*조")
LEADING_NOISE = {"구", "현행", "개정", "같은", "동", "및", "또는", "즉"}
NOT_MATERIAL = re.compile(r"판결|발췌|법률|자료$|주장|판단|요지|규정|헌법|조문|법$|제\s*\d+\s*[조항]|대법원|선고|판$|관계$|계산$|기준$|요건|수행$")


def law_phrases(text: str) -> list[str]:
    """Law names written right before an article number, with leading connectors dropped ('...따르면 민원 처리에 관한 법률')."""
    phrases = []
    for match in LAW_BEFORE_ARTICLE.finditer(text):
        words = match.group(1).split()
        while len(words) > 1 and (words[0] in LEADING_NOISE or words[0].endswith(("면", "는", "은", "고", "서", "며", "고"))):
            words = words[1:]
        phrases.append("".join(words))
    return phrases


def answer_laws_grounded(answer: str, context: str, title: str) -> bool:
    """Reject reference answers naming a law or article absent from the passages and title (known label noise)."""
    from legal_rag.textutil.refs import extract_statute_refs

    haystack = norm(context + " " + title).replace(" ", "")
    if any(phrase not in haystack for phrase in law_phrases(answer)):
        return False
    context_articles = {r.article for r in extract_statute_refs(context + " " + title)}
    return all(r.article in context_articles for r in extract_statute_refs(answer))


def is_material(name: str) -> bool:
    """A preparation must name a document, record or exhibit, not a statute, the judgment itself or a legal test."""
    return bool(name.strip()) and not NOT_MATERIAL.search(name.strip())


def first_sentence(text: str) -> str:
    match = re.search(r"^(.+?(?:다|요|음)\.)(?:\s|$)", text.strip(), flags=re.S)
    return (match.group(1) if match else text.strip()).strip()


def sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "?", "!")) else text + "."


WHAT_TAIL = re.compile(r"\s*(?:은|는)?\s*(?:무엇인가요|무엇입니까|무엇인가|무엇이며|어떤 것들이 있나요|어떤 것이 있나요)$")
YES_NO_TAIL = [(re.compile(r"(한|인|은|운)가요$"), r"\1지"), (re.compile(r"입니까$"), "인지"), (re.compile(r"할까요$"), "할지"),
               (re.compile(r"([가-힣])나요$"), r"\1는지"), (re.compile(r"습니까$"), "는지")]


def issue_from_question(question: str) -> str:
    """The question restated as an issue phrase by changing only the interrogative ending (no new content)."""
    text = question.strip().rstrip("?").strip()
    issue = WHAT_TAIL.sub("", text)
    if issue != text:
        return issue.strip(" ,")
    for pattern, replacement in YES_NO_TAIL:
        if pattern.search(text):
            return pattern.sub(replacement, text)
    return text


def parse_case(document_id: str) -> tuple[str | None, str | None]:
    """'서울중앙지방법원-2017가단27240' -> (court, case number); other ids carry no number."""
    match = re.fullmatch(r"(.+?)-(\d{4}[가-힣]+\d+)", document_id)
    return (match.group(1), match.group(2)) if match else (None, None)


def bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    return {compact[i:i + 2] for i in range(len(compact) - 1)}


def best_sentence(context: str, reference: str) -> tuple[int, str]:
    """(passage number, sentence) in the passages closest to the reference answer; used as a fallback citation."""
    target = bigrams(reference)
    best = (0.0, 1, "")
    for number, passage in enumerate(passages(context), 1):
        for piece in re.split(r"(?<=[.다])\s+", norm(passage)):
            if 15 <= len(piece) <= 200:
                score = len(bigrams(piece) & target) / (len(bigrams(piece)) or 1)
                best = max(best, (score, -number, piece), key=lambda x: x[0])
    return abs(best[1]), best[2][:80]


def build_citations(row: dict, quotes: list[str]):
    """Cite every passage a kept quote comes from; with no items, the passage closest to the reference answer."""
    from legal_rag.schemas.answer import Citation, CitationKind

    court, number = parse_case(row["document_id"])
    reference = f"{court} {number}" if number else f"{row['title']} 판결"
    texts = [norm(p) for p in passages(row["context"])]
    cited: dict[int, str] = {}
    for quote in quotes:
        for index, text in enumerate(texts, 1):
            if quote in text and index not in cited:
                cited[index] = quote
                break
    if not cited:
        index, quote = best_sentence(row["context"], row["answer"])
        if quote:
            cited[index] = quote
    return [Citation(source_id=f"S{i}", kind=CitationKind.case, reference=reference, quote=q) for i, q in sorted(cited.items())]


def assemble_answer(reference: str, preparations: list[Item], cautions: list[Item], *, question: str = "", citations=()):
    """Production LegalAnswer: reference answer as the legal core, grounded items as actions and cautions."""
    from legal_rag.schemas import LegalAnswer
    from legal_rag.schemas.answer import Confidence

    analysis = reference.strip()
    caution_text = " ".join(sentence(c.text) for c in cautions)
    if caution_text:
        analysis += "\n\n주의할 점: " + caution_text
    return LegalAnswer(legal_issues=[issue_from_question(question)] if question else [], analysis=analysis,
                       summary=first_sentence(reference), possible_actions=[p.text.strip() for p in preparations],
                       additional_information_needed=[], citations=list(citations), uncertainty=caution_text,
                       confidence=Confidence.medium)


def evidence_bundle(row: dict):
    from legal_rag.schemas import Evidence, EvidenceBundle
    from legal_rag.schemas.documents import DocumentType

    statute = row["document_id"].count("::") >= 2 and row["document_id"].split("::")[1] == "statute"
    court, case_number = parse_case(row["document_id"])
    items = []
    for number, text in enumerate(passages(row["context"]), 1):
        header = ({"Type": "Statute", "Law": row["title"]} if statute else
                  {"Type": "Case", "Case name": row["title"], **({"Court": court, "Case number": case_number} if case_number else {}),
                   "Section role": "역할 미상 (판단·주장 구분 미확인)"})
        items.append(Evidence(source_id=f"S{number}", chunk_id=f"{row['id']}#p{number}", document_id=row["document_id"],
                              document_type=DocumentType.statute if statute else DocumentType.judgment,
                              header=header, text=text, score=1.0))
    return EvidenceBundle(items=items)


class _Capture:
    """A ChatModel that records the production messages and returns the prepared target."""
    name = "capture"

    def __init__(self, target):
        self.target, self.messages = target, None

    def generate(self, messages, *, response_schema=None, temperature=0.2, max_tokens=1024, on_text=None):
        from legal_rag.llm.base import ModelResponse
        self.messages = [{"role": m.role, "content": m.content} for m in messages]
        return ModelResponse(text=self.target.model_dump_json(), parsed=self.target)


def production_messages(question: str, row: dict, target) -> list[dict]:
    from legal_rag.rag.generator import Generator

    capture = _Capture(target)
    Generator(capture).generate(question, evidence_bundle(row))
    if capture.messages is None:
        raise RuntimeError("generator did not call the model")
    return capture.messages


def sample_rows(count: int, *, exclude_hashes: set[str]) -> list[dict]:
    by_domain: dict[str, list[dict]] = {d: [] for d in DOMAIN_QUOTA}
    with SOURCE.open() as f:
        for line in f:
            row = json.loads(line)
            if (row["domain"] in by_domain and is_court_judgment(row["document_id"]) and len(row["context"]) >= 300
                    and not ARGUMENT_QUESTION.search(row["question"])
                    and row["question_hash"] not in exclude_hashes and answer_laws_grounded(row["answer"], row["context"], row["title"])):
                by_domain[row["domain"]].append({k: row[k] for k in ("id", "domain", "document_id", "title", "question",
                                                                     "answer", "context", "question_hash", "group_id")})
    rng = random.Random(SEED)
    picked = []
    for domain, share in DOMAIN_QUOTA.items():
        rows = by_domain[domain]
        rng.shuffle(rows)
        groups, chosen = set(), []
        for row in rows:  # one row per source group, so near-duplicates do not repeat
            if row["group_id"] in groups:
                continue
            groups.add(row["group_id"])
            chosen.append(row)
            if len(chosen) >= round(count * share):
                break
        picked += chosen
    rng.shuffle(picked)
    return picked


def extract(args) -> None:
    from legal_rag.llm import Message
    from legal_rag.llm.ollama import OllamaChatModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "extracted.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    model = OllamaChatModel(model=EXTRACTOR, base_url="http://localhost:11434", timeout=600, think=args.think, num_ctx=8192)
    rows = sample_rows(args.candidates, exclude_hashes=set())
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in sample_rows(10**6, exclude_hashes=set()) if r["id"] in wanted]
    print(json.dumps({"candidates": len(rows), "already_done": len(done)}), flush=True)
    with path.open("a") as stream:
        for row in rows:
            if row["id"] in done:
                continue
            user = (f"[법률 자료]\n제목: {row['title']}\n{row['context']}\n[/법률 자료]\n\n"
                    f"질문: {row['question']}\n\n참고 답변: {row['answer']}")
            started, record = time.perf_counter(), {"id": row["id"], "row": row, "extractor": EXTRACTOR, "think": args.think}
            try:
                response = model.generate([Message(role="system", content=EXTRACT_SYSTEM), Message(role="user", content=user)],
                                          response_schema=Extraction, temperature=0.0, max_tokens=4096 if args.think else 1024)
                record["raw"] = response.text
                parsed = response.parsed if isinstance(response.parsed, Extraction) else Extraction.model_validate_json(response.text)
                record["extraction"] = parsed.model_dump()
                items = [("preparation", p.item()) for p in parsed.preparations] + [("caution", c) for c in parsed.cautions]
                if items:
                    listing = "\n".join(f"{n}. [{kind}] 항목: {it.text}\n   자료 문장: {it.quote}"
                                        for n, (kind, it) in enumerate(items, 1))
                    check = model.generate([Message(role="system", content=CHECK_SYSTEM),
                                            Message(role="user", content=listing)],
                                           response_schema=SupportCheck, temperature=0.0, max_tokens=3072 if args.think else 256)
                    verdicts = (check.parsed if isinstance(check.parsed, SupportCheck)
                                else SupportCheck.model_validate_json(check.text)).verdicts
                    record["support"] = verdicts if len(verdicts) == len(items) else None
            except Exception as exc:  # recorded and skipped at assembly
                record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"id": row["id"], "ok": "error" not in record, "ms": record["elapsed_ms"]}), flush=True)


def assemble(args) -> None:
    from legal_rag.verification import Verifier

    out = Path(args.out)
    records = [json.loads(line) for line in (out / "extracted.jsonl").read_text().splitlines()]
    examples, report = [], {"records": len(records), "extraction_errors": 0, "verifier_rejected": 0,
                            "items_kept": 0, "items_dropped": 0, "drop_reasons": {}, "with_preparations": 0,
                            "with_cautions": 0, "with_neither": 0}
    for record in records:
        if "extraction" not in record:
            report["extraction_errors"] += 1
            continue
        row, ext = record["row"], Extraction.model_validate(record["extraction"])
        support = record.get("support")
        if support is None and (ext.preparations or ext.cautions):
            report["support_check_missing"] = report.get("support_check_missing", 0) + 1
            continue
        flags = list(support or [])
        prep_items = [p.item() for p in ext.preparations]
        prep_ok = [it for it, p, ok in zip(prep_items, ext.preparations, flags[:len(prep_items)]) if ok and is_material(p.material)]
        not_material = [{"text": it.text, "reason": "not_a_material"} for it, p, ok in
                        zip(prep_items, ext.preparations, flags[:len(prep_items)]) if ok and not is_material(p.material)]
        caut_ok = [it for it, ok in zip(ext.cautions, flags[len(prep_items):]) if ok]
        unsupported = [{"text": it.text, "reason": "not_supported_by_quote"}
                       for it, ok in zip(prep_items + list(ext.cautions), flags) if not ok]
        preps, dropped_p = validate_items(prep_ok, row["context"])
        cauts, dropped_c = validate_items(caut_ok, row["context"])
        dropped_p = unsupported + not_material + dropped_p
        for drop in dropped_p + dropped_c:
            reason = drop["reason"].split(":")[0]
            report["drop_reasons"][reason] = report["drop_reasons"].get(reason, 0) + 1
        report["items_kept"] += len(preps) + len(cauts)
        report["items_dropped"] += len(dropped_p) + len(dropped_c)
        citations = build_citations(row, [i.quote for i in preps + cauts])
        target = assemble_answer(row["answer"], preps, cauts, question=row["question"], citations=citations)
        bundle = evidence_bundle(row)
        if not Verifier().verify(target, bundle).ok:
            report["verifier_rejected"] += 1
            continue
        report["with_preparations"] += bool(preps)
        report["with_cautions"] += bool(cauts)
        report["with_neither"] += not preps and not cauts
        messages = production_messages(row["question"], row, target)
        examples.append({"id": row["id"], "domain": row["domain"], "document_id": row["document_id"],
                         "messages": messages + [{"role": "assistant", "content": target.model_dump_json()}],
                         "preparations": [p.model_dump() for p in preps], "cautions": [c.model_dump() for c in cauts],
                         "dropped": dropped_p + dropped_c, "reference_answer": row["answer"]})
    # Prefer examples that teach the new behaviour; keep some empty ones so "none" stays a valid answer.
    rng = random.Random(SEED)
    rng.shuffle(examples)
    rich = [e for e in examples if e["preparations"] or e["cautions"]]
    empty = [e for e in examples if not e["preparations"] and not e["cautions"]]
    total = args.train + args.validation
    n_empty = min(len(empty), round(total * 0.1))
    chosen = rich[: total - n_empty] + empty[:n_empty]
    if len(chosen) < total:
        raise SystemExit(f"only {len(chosen)} usable examples for {total}; extract more candidates")
    rng.shuffle(chosen)
    splits = {"validation": chosen[: args.validation], "train": chosen[args.validation: total]}
    hashes = {}
    for name, rows in splits.items():
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (out / f"{name}.jsonl").write_text(text)
        hashes[f"{name}.jsonl"] = hashlib.sha256(text.encode()).hexdigest()
    report.update({"usable": len(examples), "rich": len(rich), "empty": len(empty), "chosen_empty": n_empty,
                   "train": len(splits["train"]), "validation": len(splits["validation"]), "hashes": hashes,
                   "source": str(SOURCE.relative_to(ROOT)), "seed": SEED, "extractor": EXTRACTOR,
                   "human_legal_review": False})
    (out / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--candidates", type=int, default=450)
    e.add_argument("--out", required=True)
    e.add_argument("--think", action="store_true", help="let the extractor reason before answering (slower)")
    e.add_argument("--ids", default="", help="comma-separated row ids to extract instead of the sample")
    a = sub.add_parser("assemble")
    a.add_argument("--out", required=True)
    a.add_argument("--train", type=int, default=300)
    a.add_argument("--validation", type=int, default=30)
    args = parser.parse_args()
    {"extract": extract, "assemble": assemble}[args.command](args)


if __name__ == "__main__":
    main()
