# Prepare the full legal QA corpus

Use the shared [training environment setup](local-qlora.md#separate-training-environment)
to create `legal-qlora` and register its notebook kernel. The full-corpus data and
notebook steps below remain separate from the pilot workflow.

For a complete, commented Qwen3-4B training workflow, open
[`sft_qlora_full.ipynb`](../sft_qlora_full.ipynb). It loads the full preparation
directly and keeps GPU training disabled until `RUN_TRAINING=True`. The older
`sft_qlora.ipynb` is the manually edited pilot notebook; the replacement-cell
instructions below are retained for anyone continuing in that file.

The CPU command below processes all four `qa.jsonl` files. It removes the pilot caps and
does not downsample domains. It does not train or load model weights. The existing local
Qwen tokenizer is required. The user's handwritten notebook is left unchanged.

```bash
cd /home/layke/Documents/MyLLMAgent
conda activate legal-qlora
python -u scripts/prepare_full_sft.py \
  --source-root /home/layke/Documents/legal-rag/data/processed \
  --output data/training/legal-qa-full-v1 \
  --tokenizer-dir data/training/qwen3-4b-tokenizer \
  --max-length 1024
```

Use a fresh directory for each preparation. The manifest remains `preparing` if the process
fails; such outputs cannot be loaded for training. Rerun into a new version directory.

The builder indexes source offsets, groups related sources and duplicate questions globally,
then splits groups 80/10/10. It selects verbatim overlapping windows within the QA's known
source document using question-only lexical ranking. It measures the actual Qwen prompt and
complete answer together, preserving both the question and answer. It never takes the last
N tokens of a training example. Candidate passage boundaries and scores are not legal proof.

The answer is used AFTER selection for lexical coverage and article-presence screening.
Default minimum unique-term coverage is 0.5; this is a tunable triage heuristic, not a
validated grounding threshold. Common words can pass the screen despite incorrect reasoning,
and supported paraphrases can fail. Missing source article references and low-coverage rows
are routed to `review.jsonl`. Rank candidates are ordered using the question, not the answer.
Answer length affects how many source passages fit the budget.

Outputs:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: every eligible example in its split, with
  source offsets, original answer, messages, tokens, answer-only labels, and provenance.
- `review.jsonl`: excluded records and reasons. Inspect a sample of both kept and excluded
  examples before spending time on training. Review is not automatically converted to training.
- `manifest.json`: settings, source/output hashes, retained counts and all exclusion counts.

"Full" means scanning every QA record and keeping every eligible record, not training on
held-out rows, duplicates, or examples whose complete answer cannot fit. The output is a
candidate dataset with heuristic support screening; it is not legally verified data. It is
an oracle-source experiment, not full-corpus retrieval evaluation. Exact grouping does not
catch every partially overlapping source or paraphrase. Prior inspected test rows are not a
pristine final benchmark.

## Notebook changes

Restart the kernel, keeping the old adapter/checkpoints out of this new experiment.

In configuration:

```python
DATA_DIR = PROJECT_ROOT / "data/training/legal-qa-full-v1"
MAX_LENGTH = 1024
RUN_GPU = False
RUN_MODE = "train"
```

Replace the entire old dataset preparation/verification cell with:

```python
from legal_rag.training.full_sft import load_full_splits

raw_splits, tokenized, manifest = load_full_splits(DATA_DIR)
selected_raw = raw_splits

if manifest["max_length"] != MAX_LENGTH:
    raise RuntimeError("MAX_LENGTH differs from the prepared dataset.")
if manifest["tokenizer_revision"] != MODEL_REVISION:
    raise RuntimeError("Model/tokenizer revision differs from preparation.")

token_report = {
    "format": "pretokenized-full-v2",
    "max_length": manifest["max_length"],
    "dataset_files": manifest["files"],
    "counts": manifest["counts"],
}

for split, dataset in raw_splits.items():
    print(split, len(dataset))
print(json.dumps(manifest["counts"], indent=2))
print("Hashes and split isolation passed; data is loaded lazily from disk.")
```

The raw and encoded datasets share the same file offsets and order. They are lazy datasets,
not lists, so examples and token arrays are not all held in RAM. Do not convert them to lists
or run the old `balance_encoded` loop afterward: that would restore downsampling.

Replace the old tokenizer/tokenization cell in its entirety with:

```python
from transformers import AutoTokenizer

for name, expected in manifest["tokenizer_files"].items():
    if file_hash(TOKENIZER_DIR / name) != expected:
        raise RuntimeError(f"Tokenizer file changed: {name}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
tokenizer.padding_side = "right"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

sample = tokenized["train"][0]
print(tokenizer.decode([t for t in sample["labels"] if t != -100]))
print("Already tokenized during preparation; no resampling or truncation applied.")
```

Keep the existing source inspection cell if desired. The model, training and comparison cells
can use these datasets. Set `RUN_GPU=True` only when ready to load the model and train. Keep
`RESUME_FROM_CHECKPOINT=None` for the first corrected run; never resume a leaked adapter.
Later resumption is only from a checkpoint of this same prepared dataset and configuration.

## Remove the unsafe OOM retry

In the training cell, replace the entire `try: trainer.train(...)` / OOM retry section, up to
but excluding `metrics["training"] = train_result.metrics`, with:

```python
try:
    train_result = trainer.train(
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT
    )
except torch.cuda.OutOfMemoryError:
    metrics["training"] = {
        "status": "interrupted",
        "reason": "CUDA out of memory",
        "global_step": trainer.state.global_step,
    }
    save_json("metrics.json", metrics)
    print("Training stopped. Preserve the checkpoints; do not truncate examples in place.")
    raise
```

This does not guarantee 1,024-token GPU fit. If a longer example still triggers OOM, stop,
restart the kernel, and prepare a new dataset version at `--max-length 768`; change the notebook
length/path accordingly. Re-prepare the source passages within that budget, rather than
slicing away evidence. A checkpoint from another dataset/length is not an exact resume point.

The existing periodic checkpoint settings protect only completed optimizer steps. Saving
an adapter immediately after training and before final evaluation still applies.

## Verification performed

Focused CPU tests cover no caps/downsampling, duplicate handling, disjoint splits, hash
checking, complete-answer retention, review routing, and source offsets. A synthetic
four-domain fixture was processed with the real pinned Qwen tokenizer at 1,024 tokens.
The full corpus and GPU training are left for the user to run. Training output quality and
support-screen accuracy are not established by these software tests.
