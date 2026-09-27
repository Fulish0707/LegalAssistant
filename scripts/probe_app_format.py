#!/usr/bin/env python
"""Probe: can the adapter produce valid LegalAnswer JSON under the app's prompt?

Runs in the conda environment (needs torch, transformers, peft, bitsandbytes).
Reads contexts.jsonl from stage 1, rebuilds the app-style prompt (answer_system.md
+ [LEGAL SOURCE S1] blocks), generates with base and adapter, and reports parse rate.

Usage:
    /home/layke/anaconda3/envs/legal-qlora/bin/python scripts/probe_app_format.py \
      --contexts-dir data/eval/sft/full-v2-200 \
      --adapter models/adapters/qwen3-4b-legal-full/20260924T105613457564Z/adapter \
      --n 20
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

# ── app prompt (answer_system.md) ───────────────────────────────────────────
ANSWER_SYSTEM = (Path(__file__).resolve().parent.parent / "src/legal_rag/llm/prompts/answer_system.md").read_text("utf-8")

# ── LegalAnswer JSON schema (inline, conda has no pydantic) ─────────────────
LEGAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "legal_issues": {"type": "array", "items": {"type": "string"}},
        "analysis": {"type": "string"},
        "possible_actions": {"type": "array", "items": {"type": "string"}},
        "additional_information_needed": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "kind": {"type": "string", "enum": ["statute", "case", "interpretation"]},
                    "reference": {"type": "string"},
                    "quote": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["source_id", "kind", "reference"],
            },
        },
        "uncertainty": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["summary", "legal_issues", "analysis", "possible_actions", "citations", "confidence"],
}

FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def render_evidence(chunks: list[dict], question: str) -> str:
    """Render retrieved chunks in the app's [LEGAL SOURCE Sn] format."""
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        sid = f"S{i}"
        lines = [f"[LEGAL SOURCE {sid}]", ""]
        meta = chunk.get("metadata", {})
        for key in ("법령명", "조문", "사건번호", "제목"):
            if key in meta:
                lines.append(f"{key}: {meta[key]}")
        lines += [f"Source ID: {sid}", "", "Text:", chunk.get("text", chunk.get("passage", "")), ""]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(검색된 법률 근거가 없습니다. 근거 없이 조문이나 판례를 언급하지 마세요.)"


def build_app_messages(question: str, chunks: list[dict]) -> list[dict]:
    """Build app-style messages matching Generator.generate()."""
    evidence_text = render_evidence(chunks, question)
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": "다음은 검색된 법률 근거입니다.\n\n" + evidence_text},
        {"role": "user", "content": "질문: " + question},
    ]


def try_parse_legal_answer(text: str) -> tuple[bool, str]:
    """Try to parse text as LegalAnswer JSON. Returns (success, reason)."""
    cleaned = FENCE.sub(r"\1", text)
    # Try to find JSON object
    # Sometimes model outputs text before/after JSON
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return False, "no JSON object found"
    candidate = cleaned[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    # Check required fields
    required = ["summary", "legal_issues", "analysis", "possible_actions", "citations", "confidence"]
    missing = [f for f in required if f not in obj]
    if missing:
        return False, f"missing fields: {missing}"
    # Check citations structure
    if not isinstance(obj["citations"], list):
        return False, "citations is not a list"
    for cit in obj["citations"]:
        if not isinstance(cit, dict):
            return False, "citation is not an object"
        if "source_id" not in cit:
            return False, "citation missing source_id"
    # Check confidence enum
    if obj["confidence"] not in ("low", "medium", "high"):
        return False, f"invalid confidence: {obj['confidence']}"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts-dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    import random
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    contexts_dir = Path(args.contexts_dir)

    # Load contexts and sample N cases with retrieved context
    cases = []
    with open(contexts_dir / "contexts.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if "retrieved" in row.get("conditions", {}):
                cases.append(row)

    rng = random.Random(args.seed)
    cases = rng.sample(cases, min(args.n, len(cases)))
    print(f"probing {len(cases)} cases")

    # Load model
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", quantization_config=bnb, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    results = []

    for ci, case in enumerate(cases):
        question = case["question"]
        # Build app-style evidence from the retrieved hits
        hits = case["conditions"]["retrieved"].get("hits", [])
        chunks = []
        for hit in hits:
            chunks.append({
                "text": hit.get("text", hit.get("passage", "")),
                "metadata": hit.get("metadata", {}),
            })

        app_messages = build_app_messages(question, chunks)

        # Add schema instruction
        schema_instruction = f"\n\n출력은 반드시 다음 JSON 스키마를 따르는 객체 하나여야 합니다:\n```json\n{json.dumps(LEGAL_ANSWER_SCHEMA, ensure_ascii=False)}\n```"
        app_messages[0]["content"] += schema_instruction

        prompt = tokenizer.apply_chat_template(
            app_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        for model_name in ("adapter", "base"):
            if model_name == "base":
                model.disable_adapter_layers()
            else:
                model.enable_adapter_layers()

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            elapsed = time.time() - t0
            gen_tokens = out.shape[1] - prompt_len
            text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

            ok, reason = try_parse_legal_answer(text)
            results.append({
                "case_id": case["case_id"],
                "model": model_name,
                "parse_ok": ok,
                "reason": reason,
                "gen_tokens": gen_tokens,
                "seconds": round(elapsed, 1),
                "prompt_tokens": prompt_len,
            })
            status = "✓" if ok else f"✗ {reason}"
            print(f"  [{ci+1}/{len(cases)}] {model_name:8s} {gen_tokens:4d}tok {elapsed:5.1f}s {status}")

    # Summary
    for arm in ("adapter", "base"):
        arm_results = [r for r in results if r["model"] == arm]
        parsed = sum(1 for r in arm_results if r["parse_ok"])
        total = len(arm_results)
        avg_tok = sum(r["gen_tokens"] for r in arm_results) / total if total else 0
        print(f"\n{arm}: {parsed}/{total} parsed ({parsed/total*100:.0f}%), avg {avg_tok:.0f} tokens")

    # Write results
    out_path = contexts_dir / "probe_app_format.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
