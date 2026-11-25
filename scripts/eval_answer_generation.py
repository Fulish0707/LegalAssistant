"""Answer-writing stage comparison: current serving model vs base vs legal adapter.

Bypasses the action policy: every case is an answer situation with fixed evidence
(official statute extracts + fictional judgment windows). Arms:

  structured  production Generator (JSON schema, production answer prompt)
  plain       the adapter's exact training format (training.dataset.SYSTEM +
              training.adapter_eval.retrieved_messages), free text wrapped in LegalAnswer

Every answer goes through the production Verifier. Statute/case references not in the
evidence are fatal (A2). Run one model+arm per invocation; repeat runs get new --out dirs.

Run: env -u PYTHONPATH -u PYTHONHOME PYTHONPATH=src LEGAL_RAG__LLM__TIMEOUT_SECONDS=300 \
     .venv/bin/python scripts/eval_answer_generation.py --model exaone3.5:7.8b --arm structured --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FICTIONAL_CASE_MARK = "실제 판례가 아님"


def load_sources(inputs: Path):
    sources = json.loads((inputs / "sources.json").read_text())
    by_id, hashes = {}, {}
    for source in sources:
        content = (inputs / source["extraction_file"]).read_text()
        digest = hashlib.sha256(content.encode()).hexdigest()
        if digest != source["sha256"] or source["quote"] not in content:
            raise ValueError(f"source hash or literal quote mismatch: {source['id']}")
        if source["kind"] == "case" and FICTIONAL_CASE_MARK not in content:
            raise ValueError(f"case source must be marked fictional: {source['id']}")
        hashes[source["extraction_file"]] = digest
        by_id[source["id"]] = source
    return by_id, hashes


def build_bundle(case, by_id):
    from legal_rag.schemas import Evidence, EvidenceBundle
    from legal_rag.schemas.documents import DocumentType

    items = []
    for number, sid in enumerate(case["sources"], 1):
        source = by_id[sid]
        if source["kind"] == "statute":
            header = {"Type": "Statute", "Law": source["law_name"], "Article": source["article"],
                      "Effective date": source["effective_date"]}
            chunk_id = f"official:{source['law_name']}:{source['effective_date']}#art:{source['article']}"
            document_type = DocumentType.statute
        else:
            header = {"Type": "Case", "Court": source["court"], "Case number": source["case_number"],
                      "Section": "판단", "Section role": "기관·법원 판단"}
            chunk_id = f"fictional:{sid}#window:0"
            document_type = DocumentType.judgment
        items.append(Evidence(source_id=f"S{number}", chunk_id=chunk_id, document_id=sid,
                              document_type=document_type, header=header, text=source["quote"], score=1.0))
    return EvidenceBundle(items=items)


def build_plain_messages(case, by_id, message):
    from legal_rag.training.adapter_eval import retrieved_messages

    blocks = []
    for sid in case["sources"]:
        source = by_id[sid]
        title = (f"{source['law_name']} {source['article']}" if source["kind"] == "statute" else source["title"])
        blocks.append((title, [source["quote"]]))
    return retrieved_messages(message, blocks)


def answer_body(answer) -> str:
    """The prose a blind scorer reads: no citations, confidence or code-added fields."""
    parts = [answer.summary, answer.analysis]
    parts += list(answer.possible_actions)
    seen, out = set(), []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return "\n\n".join(out)


def run_case(case, by_id, model, settings, arm):
    from legal_rag.llm import Message
    from legal_rag.rag.generator import GenerationError, Generator
    from legal_rag.schemas import LegalAnswer
    from legal_rag.schemas.answer import Confidence
    from legal_rag.verification import Verifier

    bundle = build_bundle(case, by_id)
    started = time.perf_counter()
    error, prompt = None, None
    try:
        if arm == "structured":
            generator = Generator(model, temperature=settings.llm.temperature, max_tokens=settings.llm.max_tokens,
                                  mask_case_figures=settings.rag.mask_case_figures, answer_style=settings.rag.answer_style)
            answer = generator.generate(case["message"], bundle)
            prompt = "production answer_system prompt + EvidenceBundle.render()"
        else:
            messages = build_plain_messages(case, by_id, case["message"])
            response = model.generate([Message(**m) for m in messages],
                                      temperature=settings.llm.temperature, max_tokens=settings.llm.max_tokens)
            answer = LegalAnswer(summary=response.text, legal_issues=[], analysis=response.text,
                                 possible_actions=[], citations=[], confidence=Confidence.medium)
            prompt = messages
    except GenerationError as exc:
        answer, error = None, str(exc)
    except Exception as exc:  # transport/timeout: record, keep going
        answer, error = None, f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    row = {"case_id": case["id"], "domain": case["domain"], "arm": arm, "model": model.name,
           "status": "completed" if answer is not None else "failed", "elapsed_ms": elapsed_ms, "error": error}
    if answer is not None:
        report = Verifier().verify(answer, bundle)
        row["answer"] = answer.model_dump(mode="json")
        row["body"] = answer_body(answer)
        row["verification"] = report.model_dump(mode="json")
        row["fatal"] = [issue.model_dump(mode="json") for issue in report.issues
                        if issue.kind in ("statute_not_in_evidence", "case_not_in_evidence")]
    row["prompt"] = prompt
    return row


def main() -> int:
    from legal_rag.config import load_settings
    from legal_rag.factory import build_chat_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/eval/answer_generation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("structured", "plain"), required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--cases", default=None, help="comma-separated case ids (default: all)")
    args = parser.parse_args()

    settings = load_settings(profile=args.profile)
    llm = settings.llm.model_copy(update={"backend": "ollama", "model": args.model,
                                          **({"base_url": args.base_url} if args.base_url else {})})
    settings = settings.model_copy(update={"llm": llm, "trace": settings.trace.model_copy(update={"path": None})})
    model = build_chat_model(settings)

    inputs, out = args.inputs, args.out
    cases = json.loads((inputs / "cases.json").read_text())
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [case for case in cases if case["id"] in wanted]
    by_id, hashes = load_sources(inputs)
    for name in ("cases.json", "sources.json", "comparison.json"):
        hashes[name] = hashlib.sha256((inputs / name).read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    frozen = out / "inputs"
    frozen.mkdir()
    for name in hashes:
        (frozen / name).write_bytes((inputs / name).read_bytes())
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "model_name": args.model, "arm": args.arm,
                "settings": settings.model_dump(mode="json", exclude={"llm": {"api_key"}, "qdrant": {"api_key"}}),
                "input_hashes": hashes,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                "protocol": "answer stage only, fixed evidence; structured=production prompt, plain=training format",
                "human_legal_review": False}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    rows = []
    with (out / "results.jsonl").open("x") as stream:
        for case in cases:
            row = run_case(case, by_id, model, settings, args.arm)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({key: row.get(key) for key in ("case_id", "arm", "status", "elapsed_ms")}), flush=True)
    report = {"attempted": len(rows), "completed": sum(row["status"] == "completed" for row in rows),
              "failed": sum(row["status"] == "failed" for row in rows),
              "verification_ok": sum(bool(row.get("verification", {}).get("ok")) for row in rows),
              "fatal_answers": sum(bool(row.get("fatal")) for row in rows), "human_legal_review": False}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
