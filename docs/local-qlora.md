# Local Qwen3-4B legal QLoRA pilot

Open [`sft_qlora.ipynb`](../sft_qlora.ipynb). Preparation is implemented; no training has been started.
The first adapter targets natural Korean answers grounded in supplied source text across
civil, criminal, administrative and intellectual-property law.

## Prepared data

Source: `/home/layke/Documents/legal-rag/data/processed/<domain>/{qa,documents}.jsonl`.
The source corpus is read-only. Generated files live in this project's ignored
`data/training/legal-qa-pilot-v1/` directory:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: original answers, full short source context,
  chat messages, document IDs, source/question hashes and split-group IDs.
- `manifest.json`: source/output SHA-256 hashes, preparation settings and exclusions.
- `tokenization-report.json`: measured token exclusions and selected IDs for this pilot.

| Stage | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Complete-source pilot | 4,000 | 400 | 400 |
| Fits 1,536 tokens, balanced again | 2,612 | 248 | 252 |
| Per domain after token filtering | 653 | 62 | 63 |

All 235,901 QA records had matching document IDs in the audited corpus. Matching does not
establish that an answer is supported by its source. Inspect the notebook's examples before
training. Short-source candidates with duplicate questions but different normalized answers
were excluded: civil 259, criminal 454, administrative 608, intellectual property 6,114.
These counts represent excluded rows, not necessarily incorrect answers; context-dependent
answers can legitimately differ.

Sources longer than 2,400 characters are excluded from this first pilot, rather than cropped
using the answer. Remaining overlength examples are dropped whole at tokenization. This
biases the pilot toward short-source questions; it is not representative of all legal work.

Splitting links documents globally by document ID, normalized exact text, matching statute
titles across versions, and duplicate questions. Connected groups are assigned 80/10/10 by
seeded hash before sampling. Equal per-domain samples prevent larger corpora dominating.
Exact grouping does not catch every paraphrase, partially overlapping source, or differently
named version. The new held-out test is for generation; older retrieval evaluation sets are
not guaranteed disjoint from the new training set and should not be presented as clean SFT
test sets.

Rebuild in a new output directory with the existing project environment:

```bash
.venv/bin/python scripts/prepare_sft.py \
  --source-root /home/layke/Documents/legal-rag/data/processed \
  --output data/training/legal-qa-pilot-v2
```

The builder uses Python's standard library and refuses to overwrite an existing directory.
It does not call an LLM or touch an index. Data and model artifacts are excluded from Git.

## Separate training environment

The current `env.yml` and application environment are unchanged. When ready, provision the
pinned environment and register its notebook kernel:

```bash
conda env create -f env-qlora.yml
conda run -n legal-qlora python -m ipykernel install --user \
  --name legal-qlora --display-name 'Python (legal-qlora)'
```

Select **Python (legal-qlora)** in VS Code. The notebook adds this repository's `src` directory
to its import path; an editable installation of the application is unnecessary.
No training dependencies were installed as part of preparation. The environment pins are a
reproducible starting stack, not a GPU-validated compatibility claim. CUDA availability is
checked before model loading; the PyTorch wheel must work with the host NVIDIA driver.

Only the public tokenizer/config files were downloaded for CPU verification, under
`data/training/qwen3-4b-tokenizer/`, at model revision
`1cfa9a7208912126459214e8b04321603b3df60c`. No model weights were downloaded by this work.
The notebook falls back to downloading the pinned tokenizer if this directory is absent.

## Notebook sequence

1. Configure paths; leave `RUN_GPU = False` while testing continues.
2. Verify dataset hashes and split isolation; review source/answer samples.
3. Tokenize on CPU, exclude overlength examples, rebalance, and inspect supervised text.
4. The GPU boundary intentionally raises an error with the default settings. This is the
   expected end of preparation, including when using **Run All**.
5. After other testing finishes, inspect `nvidia-smi` and `ollama ps`. Unload serving models
   only when you no longer need them. Set `RUN_GPU = True`, keep `RUN_MODE = "smoke"`, and
   run one optimizer step to validate CUDA, quantization, backpropagation and memory use.
6. Restart the kernel and use `RUN_MODE = "train"` for the one-epoch pilot. A new output
   directory keeps smoke and real runs separate. Compare the base and adapter on held-out
   questions per domain before considering deployment.

Initial settings: NF4 with double quantization, BF16 where supported (FP16 otherwise),
LoRA rank 8 / alpha 16 / dropout 0.05 on all linear layers, microbatch 1, gradient accumulation
16, learning rate 1e-4, gradient checkpointing and a paged 8-bit optimizer. GPU fit on the
8 GB card remains untested. On OOM, restart the kernel, reduce `MAX_LENGTH` to 1,024, and
rerun preparation and smoke mode. Counts will change; the run records the selected IDs.

The Qwen non-thinking generation prefix is used consistently for training and inference.
Loss includes only the original answer and end-of-turn token. Padding and all prompt tokens
are masked. Validation is used during training; test is reserved for the final comparison.

## Outputs and limitations

Runs write to `models/adapters/qwen3-4b-legal/<timestamp>-<mode>/`: settings, package versions,
model revision, dataset hashes, token-selected IDs, checkpoints, metrics and the final PEFT
adapter/tokenizer. Full mode adds per-domain test loss and twelve paired base/adapter answers.
Generation length limits are recorded so incomplete answers can be identified.

The comparison supplies the correct source document, so it measures oracle-context generation,
not end-to-end retrieval. Loss is not a legal correctness score. Review answer correctness,
source support, unsupported claims and completeness. Existing QA does not supply deliberate
abstention/clarification examples or validated structured citations. This work does not change
the application, export to Ollama, merge weights, or establish legal reliability.

Preparation verification: focused CPU tests for joins, splits, duplicate handling, deterministic
sampling, no overwrite, masking, token limits and padding; real-tokenizer checks; notebook
schema and code compilation; execution through the default GPU stop. Model loading, training,
GPU memory fit and answer quality remain for after the user's testing finishes.

Sources used for the implementation: [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B),
[PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization), and
[Transformers 4.57.1 Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).
