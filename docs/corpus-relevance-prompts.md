# Corpus-specific relevance grading

The main relevance evaluation commands now default to `--prompt-profile corpus`. The 0/1/2 grading rubric is shared; known question corpora add guidance from `src/legal_rag/evaluation/relevance/relevance_prompts.py`. Civil, administrative and IP guidance matches the 2026-10-03 pilot. Criminal uses the shared rubric by default because both criminal guidance variants regressed. `--prompt-profile corpus-experimental` enables the revised criminal candidate as well as the other three blocks.

This changes relevance grading. Chat answer generation uses its existing separate prompt.

## Routing

An explicit `--judge-domain` takes precedence. Otherwise the grader uses `question_domain`, `evaluation_domain` or `legal_domain` on each pair. Older pools can recover domains from hash-verified `pool.json` case-source targets, or a single declared question domain. New pool rows preserve the question domain. Retrieved document/chunk prefixes do not select the prompt. Unknown or mixed domains use shared instructions.

Corpus membership is not guaranteed to equal the actual legal issue. For a mixed question, use `--judge-domain mixed` or store `question_domain: mixed`. The run prints route counts and records them in `judge_runs.jsonl`; successful judgments record the question domain, applied guidance domain and profile. Routing version 2 includes the active-domain allowlist in the prompt hash. Corpus-family hashes include the guidance and routing version; changing a guidance block creates a new family key for all corpora. Old judgments are not rewritten. Domain overrides also affect the corpus-family key. Cached judgments with different recorded routing are refused.

## Commands

Run from the repository root with local Ollama serving the judge model. Use matching profile, thinking and domain options for grading and reporting.

```bash
# Existing pool with question-domain metadata: corpus guidance is the default.
PYTHONPATH=src .venv/bin/python scripts/judge_relevance.py \
  --pool-dir data/eval/relevance/crossdomain-60-fanout --calibration-only --think

PYTHONPATH=src .venv/bin/python scripts/judge_agreement.py \
  --pool-dir data/eval/relevance/crossdomain-60-fanout --think \
  --reference-source 'Existing AI reference labels; not human validation' \
  --pass-boundary 1 --pass-metric ac1

# Archived pool without question-domain metadata: declare the known question corpus.
PYTHONPATH=src .venv/bin/python scripts/judge_relevance.py \
  --pool-dir data/eval/relevance/ip-struct800-calibration-fresh-2026-10-03 \
  --calibration-only --think --judge-domain ip

# Reproduce the preceding shared grader and its existing judgments.
PYTHONPATH=src .venv/bin/python scripts/judge_agreement.py \
  --pool-dir data/eval/relevance/ip-struct800-calibration-fresh-2026-10-03 \
  --think --prompt-profile shared --pass-boundary 1 --pass-metric ac1
```

`eval_graded.py`, `build_fanout_pool.py`, `rerank_relevance_pool.py` and `sweep_fusion_weights.py` expose the same `--prompt-profile` and `--judge-domain` selectors when counting or reading judgments. Old shared results require the shared selector. Corpus-profile results must not be compared with shared-profile results without naming both keys.

## Evidence and limits

The [shared v3 prompt snapshot](experiments/relevance-judge-prompt-v3.md) and
[v1/v2 review](experiments/relevance-judge-v3-review.md) retain the earlier design evidence.
Use this document for current profile selection and routing.

The 48-pair pilot increased AI-reference agreement from 33/48 to 36/48. Civil improved, administrative had very limited positive coverage, IP was unchanged, and the original criminal guidance regressed. The criminal revision scored 7/12 (58.3%), versus shared 8/12 (66.7%), with all three partial passages overgraded as direct support. It had no quote extraction failures. Criminal therefore retains shared instructions. This was the same 12 development pairs, not an independent holdout. Exact quotes do not establish entailment, and these scores do not establish legal correctness or golden-answer improvement.

Artifacts: `data/eval/reports/domain-prompt-pilot-2026-10-03/` and `data/eval/reports/criminal-prompt-revision-2026-10-03/`.

## Experimental criminal candidate

```bash
PYTHONPATH=src .venv/bin/python scripts/judge_relevance.py \
  --pool-dir data/eval/reports/criminal-prompt-revision-2026-10-03 \
  --calibration-only --think --prompt-profile corpus-experimental
```

This starts judgments under the current experimental key. The completed revision experiment used routing version 1 before the criminal fallback decision. Reproduce its stored comparison with the frozen key in its pre-registration:

```bash
PYTHONPATH=src .venv/bin/python \
  data/eval/reports/criminal-prompt-revision-2026-10-03/report.py
```

The current default and experimental profiles have new keys; their agreement commands do not select the historical revision key. The next criminal investigation should focus on requested-rule matching and partial-versus-direct support, using fresh reviewed examples.
