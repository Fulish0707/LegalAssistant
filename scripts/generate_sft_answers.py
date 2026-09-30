#!/usr/bin/env python
"""Generate base and adapter answers for every prompt in a contexts.jsonl.

Stage 2 of the adapter evaluation. Runs in the legal-qlora conda environment (bitsandbytes,
peft), which has no typer and does not install this package, hence argparse and the path insert.
One 4-bit model is loaded; disable_adapter() gives the base answer, so the adapter is the only
difference within a pair. Answers are appended as they finish; --resume continues a run.

Example:
    conda activate legal-qlora
    python scripts/generate_sft_answers.py --contexts-dir data/eval/sft/full-v2-200 \
      --adapter models/adapters/qwen3-4b-legal-full/20260924T105613457564Z/adapter
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from legal_rag.training.adapter_eval import CONDITIONS, MODELS, pending_generations  # noqa: E402
from legal_rag.training.tokenization import generation_prompt  # noqa: E402

MODEL_ID = 'Qwen/Qwen3-4B'
MODEL_REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contexts-dir', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True, help='Directory holding adapter_model.safetensors.')
    parser.add_argument('--output', type=Path, help='Defaults to <contexts-dir>/answers.jsonl.')
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--max-new-tokens', type=int, default=384)
    parser.add_argument('--limit', type=int, help='Only the first N cases; for a smoke run.')
    parser.add_argument('--resume', action='store_true', help='Keep answers already in --output.')
    args = parser.parse_args()

    output = args.output or args.contexts_dir / 'answers.jsonl'
    provenance_path = output.with_suffix('.json')
    if output.exists() and not args.resume:
        sys.exit(f'{output} exists; pass --resume or choose another --output')
    weights = args.adapter / 'adapter_model.safetensors'
    if not weights.is_file():
        sys.exit(f'no adapter_model.safetensors in {args.adapter}')
    contexts_meta = json.loads((args.contexts_dir / 'contexts.json').read_text(encoding='utf-8'))
    contexts_path = args.contexts_dir / 'contexts.jsonl'
    if _sha256(contexts_path) != contexts_meta['contexts_sha256']:
        sys.exit(f'{contexts_path} changed since it was written')
    contexts = _read_jsonl(contexts_path)[:args.limit]

    provenance = {
        'contexts_sha256': contexts_meta['contexts_sha256'], 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
        'adapter': str(args.adapter.resolve()), 'adapter_sha256': _sha256(weights), 'quantization': '4-bit nf4, double quant',
        'decoding': {'do_sample': False, 'max_new_tokens': args.max_new_tokens, 'enable_thinking': False},
    }
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding='utf-8'))
        if previous != provenance:
            sys.exit(f'{provenance_path} records different settings; resume only the same run')
    done = set()
    if output.exists():
        done = {(r['case_id'], r['condition'], r['model']) for r in _read_jsonl(output) if not r.get('error')}
    todo = pending_generations(contexts, done, conditions=args.conditions, models=MODELS)
    print(f'{len(contexts)} cases, {len(done)} answers on disk, {len(todo)} to generate', file=sys.stderr)
    if not todo:
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(contexts_meta['tokenizer_dir'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True, torch_dtype=dtype, device_map={'': 0},
        attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    provenance_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')

    started = time.perf_counter()
    with output.open('a', encoding='utf-8') as handle:
        for number, (context, condition, name) in enumerate(todo, 1):
            messages = context['conditions'][condition]['messages']
            inputs = tokenizer(generation_prompt(messages, tokenizer), return_tensors='pt',
                               add_special_tokens=False).to('cuda:0')
            row = {'case_id': context['case_id'], 'domain': context['domain'], 'condition': condition, 'model': name}
            began = time.perf_counter()
            try:
                with (nullcontext() if name == 'adapter' else model.disable_adapter()), torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                        temperature=None, top_p=None, top_k=None,
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
                completion = generated[0, inputs['input_ids'].shape[1]:].tolist()
                row.update(text=tokenizer.decode(completion, skip_special_tokens=True).strip(),
                           generated_tokens=len(completion),
                           hit_length_limit=len(completion) == args.max_new_tokens
                           and completion[-1] != tokenizer.eos_token_id)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row['error'] = 'CUDA OOM'
            row['seconds'] = round(time.perf_counter() - began, 2)
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            if number % 20 == 0 or number == len(todo):
                rate = (time.perf_counter() - started) / number
                print(f'  {number}/{len(todo)} generated, {rate:.1f} s each, '
                      f'~{rate * (len(todo) - number) / 60:.0f} min left', file=sys.stderr)


if __name__ == '__main__':
    main()
