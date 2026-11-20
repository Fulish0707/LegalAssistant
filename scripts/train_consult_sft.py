"""Continue the legal QLoRA adapter on the consulting-format pilot (data/training/consult-sft-v1).

Same base, quantization and tokenization as the original run (Qwen3-4B @1cfa9a72, NF4 double-quant,
prompt = production messages with the non-thinking generation prefix, loss on answer + EOS only).
To fit 8 GB, logits are computed only for the answer span (`logits_to_keep`), which is where all
the loss is (spike 003: full logits OOM at 1,536 tokens; answer-only fits 4,096).

Run in the legal-qlora env with PYTHONPATH=src. `--dry-run` tokenizes and reports without the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
BASE_ADAPTER = ROOT / "models/adapters/qwen3-4b-legal-full/20260924T105613457564Z/adapter"
DATA = ROOT / "data/training/consult-sft-v1"
OUT_ROOT = ROOT / "models/adapters/qwen3-4b-legal-consult"


def encode(row: dict, tokenizer, max_length: int) -> dict | None:
    from legal_rag.training.tokenization import generation_prompt

    messages = row["messages"]
    if messages[-1]["role"] != "assistant":
        raise ValueError("last turn must be the assistant target")
    prompt = tokenizer(generation_prompt(messages[:-1], tokenizer), add_special_tokens=False)["input_ids"]
    answer = tokenizer(messages[-1]["content"].strip(), add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt) + len(answer) > max_length:
        return None
    return {"id": row["id"], "input_ids": prompt + answer, "answer_len": len(answer)}


def load_split(data: Path, name: str, tokenizer, max_length: int) -> tuple[list[dict], int]:
    rows = [json.loads(line) for line in (data / f"{name}.jsonl").read_text().splitlines()]
    encoded = [encode(row, tokenizer, max_length) for row in rows]
    kept = [e for e in encoded if e is not None]
    return kept, len(rows) - len(kept)


def example_loss(model, example: dict):
    """Mean NLL over answer tokens; logits only for the last answer_len + 1 positions."""
    import torch

    ids = torch.tensor([example["input_ids"]], device="cuda")
    keep = example["answer_len"] + 1
    labels = ids[:, -keep:].clone()
    labels[:, 0] = -100  # last prompt token: only its prediction of the first answer token counts (shifted away)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(input_ids=ids, labels=labels, logits_to_keep=keep).loss


def evaluate(model, examples: list[dict]) -> float:
    import torch

    model.eval()
    total = 0.0
    with torch.no_grad():
        for example in examples:
            total += example_loss(model, example).item()
    model.train()
    return total / len(examples)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    train, train_over = load_split(args.data, "train", tokenizer, args.max_length)
    validation, val_over = load_split(args.data, "validation", tokenizer, args.max_length)
    lengths = [len(e["input_ids"]) for e in train]
    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    plan = {"train": len(train), "train_overlength": train_over, "validation": len(validation),
            "validation_overlength": val_over, "max_tokens": max(lengths), "max_answer_tokens": max(e["answer_len"] for e in train),
            "optimizer_steps": steps_per_epoch * args.epochs, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return

    import bitsandbytes as bnb
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup, set_seed

    set_seed(args.seed)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, quantization_config=quant,
                                                torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, str(BASE_ADAPTER), is_trainable=True)
    model.train()  # from_pretrained returns eval mode, which disables gradient checkpointing (spike 003)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=0.01)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps * args.warmup_ratio), total_steps)

    run_dir = OUT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True)
    metrics = {"plan": plan, "base_adapter": str(BASE_ADAPTER.relative_to(ROOT)), "model": MODEL_ID, "revision": REVISION,
               "data_hashes": {n: file_hash(args.data / n) for n in ("train.jsonl", "validation.jsonl", "manifest.json")},
               "validation_loss": {"before": evaluate(model, validation)}, "log": []}
    print(json.dumps({"validation_loss_before": metrics["validation_loss"]["before"]}), flush=True)

    rng = random.Random(args.seed)
    step, started = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        order = train[:]
        rng.shuffle(order)
        running, count = 0.0, 0
        for index, example in enumerate(order, 1):
            loss = example_loss(model, example)
            (loss / args.grad_accum).backward()
            running, count = running + loss.item(), count + 1
            if index % args.grad_accum == 0 or index == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or index == len(order):
                    entry = {"epoch": epoch, "step": step, "train_loss": round(running / count, 4),
                             "lr": scheduler.get_last_lr()[0], "elapsed_min": round((time.time() - started) / 60, 1),
                             "peak_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
                    metrics["log"].append(entry)
                    print(json.dumps(entry), flush=True)
                    running, count = 0.0, 0
        val = evaluate(model, validation)
        metrics["validation_loss"][f"epoch_{epoch}"] = val
        print(json.dumps({"epoch": epoch, "validation_loss": val}), flush=True)
        model.save_pretrained(str(run_dir / f"adapter-epoch{epoch}"))
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"run_dir": str(run_dir.relative_to(ROOT)), "minutes": round((time.time() - started) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
