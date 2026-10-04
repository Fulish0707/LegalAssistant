# Golden-answer structural comparison

Compare original four-index fanout with structural four-index fanout, neighbors
disabled, on all 80 historical answer questions (20 administrative, 20 civil,
20 criminal, 20 cross-domain). This sample has no dedicated IP stratum. Neither
the retrieved passage's corpus nor the relevance grader profile modifies generation.

Retrieve fresh top-five anchors for both arms with identical retrieval settings.
Freeze complete evidence bundles and reference selection before Gemini calls.
Generate fresh answers for both arms in alternating order, without references in
generation inputs. Hold model, temperature, thinking, token limit, verification
policy and 7500-character rendered evidence ceiling fixed. Score complete answers
using the same v2 answer grader (Qwen3:8B, no thinking, 16384 context).

Primary metric: grounded reference agreement. Also report conclusion agreement,
reference agreement including basis, per-group outcomes, paired wins/losses and
bootstrap intervals. Keep failed model calls unresolved; never treat them as wrong
answers. Validate source quotes and input hashes. This reuses development questions;
it measures AI-reference agreement, not legal correctness or a human-calibrated gain.

Preserve original labels, indexes and production prompts. Freeze run manifests and
reject resume when settings, source code or contexts change. Stop after consecutive
API failures, retain outputs and expose independent resumable phases.
