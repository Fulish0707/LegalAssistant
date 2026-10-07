# Main corpus grader integration and criminal revision

User authorizes applying the pilot guidance to the main relevance grader and revising criminal guidance. This updates relevance grading only.

Make corpus guidance the main evaluation CLI default, with an explicit shared-profile option for old judgments. A shared 0/1/2 rubric remains unchanged. Append civil, administrative and IP blocks from the frozen pilot. Keep criminal on shared instructions in the default profile after the revision scored 7/12 versus shared 8/12. Expose the candidate under `corpus-experimental`. Revise criminal guidance to check the queried entity and legal predicate, distinguish allowed exceptions from prohibitions, and select quotes from passage text rather than reference answers. Do not change quote retry while comparing the criminal prompt.

Route using an explicit domain override, question-domain fields or verified pool case-source metadata; never route from the retrieved passage's corpus. Unknown or mixed domains use the shared instructions. Preserve question domain on new pool rows. Include profile, override, all guidance and routing version in the corpus-family hash. Shared profile keeps the old hash. Store chosen domain/profile in judgment details and provenance. Refuse cache reuse if recorded routing differs.

Expose matching profile/domain options in judging, agreement, graded evaluation and pending-count tools. Archive the old CLI entrypoints used by the frozen historical comparison so its commands remain reproducible after main scripts change.

Validate the main entrypoints and routing, then grade the same frozen 12 criminal pilot pairs with the revised prompt. Compare with both prior shared and original criminal-guidance results. This is development-set regression checking, not fresh calibration; report that limit. Preserve unrelated changes, original labels and indexes; no commits or answer-generation edits.
