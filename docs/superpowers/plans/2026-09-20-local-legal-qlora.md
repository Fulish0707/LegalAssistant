# Local legal QLoRA preparation

**Goal:** Prepare, but do not launch, local Qwen3-4B training on natural Korean answers across civil, criminal, administrative and intellectual-property QA.

**Accepted design:** One adapter, evidence plus question as input, original answer as target. Use a balanced pilot and evaluate against the unchanged checkpoint. Read the sibling `legal-rag/data/processed` corpus without modifying it. Keep installation and GPU work separate from current testing.

**Architecture:** A standard-library dataset builder writes reproducible JSONL splits and a manifest. Group connected documents by ID, normalized exact text, statute title, and duplicate question before splitting. Select only complete short source documents for this pilot; never crop evidence or answers silently. Tokenization later excludes overlength examples and balances each split again. A dedicated training environment and notebook use Transformers Trainer, PEFT and bitsandbytes, with prompt labels masked and non-thinking Qwen chat formatting.

## Tasks

- [x] Test dataset joins, cross-domain split isolation, duplicate handling, deterministic sampling, missing sources, and no overwrite.
- [x] Implement `src/legal_rag/training/dataset.py` and CPU-only `scripts/prepare_sft.py`; build a real balanced pilot and record exclusions/hashes.
- [x] Test answer-only token labels, token limits and padding without requiring a model or GPU; implement `src/legal_rag/training/tokenization.py`.
- [x] Populate `sft_qlora.ipynb`: configuration, CPU data inspection, optional tokenizer preparation, manual GPU gate, baseline evaluation, one-step smoke mode, training/checkpoints, adapter save and held-out comparison.
- [x] Add isolated environment instructions and a run guide in `docs/local-qlora.md`. Validate notebook schema, compile its cells and run focused CPU tests. Do not install training packages into the running test environment or load model weights.

## Pilot choices

Aim for 1,000 train, 100 validation and 100 test examples per domain, subject to available eligible source groups. Start at 1,536 total tokens, microbatch 1, gradient accumulation 16, rank 8, NF4 with double quantization, gradient checkpointing, one epoch. These are starting settings, not a verified 8 GB fit. Keep a fixed test set; use validation only for tuning. Existing QA does not automatically teach abstention or verified citation formatting. Those require separately reviewed examples in a later phase.

## Verification

Run `.venv/bin/python -m pytest tests/legal_rag/training -q`. Run the builder on the real corpus, then audit split groups and per-domain counts. Validate notebook with nbformat if available and compile every code cell. GPU memory, CUDA compatibility, loss/backpropagation and generated answer quality remain checks for after the user's tests finish.

## Recorded result

Prepared 4,000/400/400 examples; the real Qwen tokenizer retains a balanced 2,612/248/252 at 1,536 tokens. Nine focused CPU tests pass. Notebook schema and all code cells validate; CPU cells execute and stop at the default GPU boundary. The pinned training environment has not been installed or GPU-tested. No model weights were loaded and no training was launched.
