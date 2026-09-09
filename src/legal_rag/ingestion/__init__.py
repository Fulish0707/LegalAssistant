"""Raw legal data → canonical documents.

``PIPELINE_VERSION`` is recorded in every corpus manifest. Bump it whenever
normalization, validation, deduplication, or chunking changes behavior, so a
processed corpus can be traced to the code that produced it.
"""

PIPELINE_VERSION = "2026.09.07"
