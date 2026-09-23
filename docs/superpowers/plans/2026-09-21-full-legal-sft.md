# Full-data legal SFT preparation

User approved question-selected evidence passages and requested a command to run themselves.

- [x] Implement CPU-only full-corpus preparation with no pilot caps or domain downsampling.
- [x] Keep source text on disk using offsets; group documents and duplicate questions globally before split assignment.
- [x] Select verbatim windows within the known source using question-only lexical ranking. Reserve space for the complete answer; never slice tokenized examples to recover from OOM.
- [x] Record exclusions and lexical/reference screening diagnostics. These are heuristics, not verified grounding labels.
- [x] Write disk-backed train/validation/test datasets and validate split identities/hashes on loading.
- [x] Add CLI and focused tests, including an actual Qwen tokenizer fixture check.
- [x] Provide notebook integration instructions; preserve the user's manually edited notebook.

Validation uses synthetic fixtures only. The user runs full corpus preparation and training. No model weights are loaded for this work. Existing leaked adapters are not valid resume points.
