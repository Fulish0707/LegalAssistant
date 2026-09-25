# Does the legal QLoRA adapter help once evidence comes from retrieval?

## Question

The full-data adapter (`models/adapters/qwen3-4b-legal-full/20260924T105613457564Z`) lowered
validation loss from 1.453 to 0.475. That loss is measured with the oracle source passage in the
prompt. In service the evidence comes from retrieval, so the question is whether the adapter still
writes better answers than its base model when the passages are the ones retrieval found.

## Models

- **base**: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`, loaded in 4-bit
  NF4 exactly as training loaded it.
- **adapter**: the same loaded model with the LoRA adapter enabled.

One model is loaded; `disable_adapter()` toggles between the two, so the adapter is the only
difference between paired answers. The served `qwen3:8b` (Ollama) is not part of this comparison.

## Prompt format

The adapter was trained on `training.dataset.messages_for`: the fixed Korean `SYSTEM` prompt, a
`[법률 자료] 제목: … [/법률 자료]` block of verbatim passages separated by `[발췌 구분]`, then
`질문: …`, answered in plain text. The RAG `Generator` instead asks for JSON `LegalAnswer` with
`S1`-style citations, a format the adapter never saw. Measuring the adapter through `Generator`
would confound the adapter with the format change, so every condition here uses the training
format and only the source of the passages changes.

## Conditions

200 questions from the SFT `test.jsonl`, 50 per domain, sampled with a seed. Test groups never
appear in training.

| Condition | Passages |
|---|---|
| `none` | none (`messages_for(question, '')`) |
| `retrieved` | the served retriever's top `k_final` chunks, in its domain |
| `oracle` | the test row's own stored prompt, identical to training |

`retrieved` uses the served configuration (four corpora, sparse weight 0.75, cross-encoder rerank
over the fused top 50) with `legal_domain` set to the question's domain, since a filter is required
whenever the domain is known. The raw question is the query; the service's LLM query rewriter is
not used.

The top five chunks of ip-v1 run to about 24k characters, far longer than any training prompt.
They are therefore fitted into a **768-token** prompt (1024 training length minus a 256-token
answer reserve) with the training passage selector: chunks in rank order, each taken whole if it
fits, otherwise its `select_passages` windows with a positive question score, best first, while
the templated prompt stays within budget. Passages from one document form one `[법률 자료]` block,
in order of their source offset; documents appear in rank order. A single-document prompt is
byte-identical to `messages_for`. Each case records its retrieved chunk ids and ranks, the passages
used, the prompt token count, and whether the gold document was retrieved.

## Generation

Greedy decoding (`do_sample=False`), `max_new_tokens` 384, thinking disabled in the chat template
as in training. Each answer records `generated_tokens` and `hit_length_limit`. Answers are appended
as they finish and a run resumes from the file.

## Judging

The primary comparison is adapter vs base under `retrieved`, 200 blind pairs, judged by Codex with
the question and the reference answer visible and the side assignment hidden, using
`answer_compare.pairwise_rows` (which gains an optional `reference_answer` column) and
`scripts/score_answers.py`. The other conditions are generated but judged only if the primary
result calls for it. Automatic overlap with the reference is not used as a verdict: the adapter was
trained to imitate those references' style and length.

## Pipeline

Retrieval runs in the project `uv` environment and 4-bit generation in the `legal-qlora` conda
environment, so the stages exchange files.

1. `scripts/build_sft_eval_contexts.py` (uv): sample, retrieve, write `contexts.jsonl` holding the
   finished messages for all three conditions, plus `contexts.json` provenance (test split hash,
   seed, settings hash, token budget, tokenizer).
2. `scripts/generate_sft_answers.py` (conda, GPU): read `contexts.jsonl`, write `answers.jsonl`
   (one row per case, condition and model) and `answers.json` provenance (adapter path and weight
   hash, base revision, decoding settings). Uses argparse; the conda environment has no typer.
3. `scripts/compare_sft_answers.py` (uv): write `pairwise.csv` and `pairwise_key.json` for one
   condition, adapter vs base.

Pure logic lives in `src/legal_rag/evaluation/sft_eval.py` and imports neither torch nor
transformers, so it is tested offline with a fake tokenizer.

## Testing

- A single-document retrieved prompt equals `messages_for`; several documents give one block each,
  in rank order.
- The fitted prompt never exceeds the budget; a chunk that fits whole is used whole; passages with
  score 0 are never used; an empty result falls back to an empty context and is recorded.
- Sampling is deterministic per seed and balanced across domains.
- `pairwise_rows` without references is unchanged; with references the column is filled.
- Generation rows are keyed by (case, condition, model); resume skips existing keys.
