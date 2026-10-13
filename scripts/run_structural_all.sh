#!/usr/bin/env bash
# Build the three prepared structural corpora, then evaluate four-index retrieval.
# The collections were deleted on 2026-10-14; this is the way to build them again (hours of
# embedding). civil-struct800-v1 is built first, as docs/experiments/structural-civil.md describes.
set -euo pipefail
cd /home/layke/Documents/MyLLMAgent

report_dir=data/eval/reports/structural-all-2026-10-03
mkdir -p "$report_dir"
exec 9>"$report_dir/run.lock"
flock -n 9 || { echo "Another structural-all run is active." >&2; exit 1; }

# Pin the tested settings even if the calling shell exported older overrides.
export LEGAL_RAG__EMBEDDING__BACKEND=ollama
export LEGAL_RAG__EMBEDDING__MODEL=bge-m3
export LEGAL_RAG__QDRANT__MODE=server
export LEGAL_RAG__QDRANT__URL=http://localhost:6333
export LEGAL_RAG__RAG__CONTEXT_NEIGHBOR_RADIUS=0
export LEGAL_RAG__RETRIEVAL__CORPORA='["criminal-struct800-v1","civil-struct800-v1","administrative-struct800-v1","ip-struct800-v1"]'

for domain in criminal administrative ip; do
  version="${domain}-struct800-v1"
  collection="legal_${domain}_struct800_v1_ollama_bge_m3"
  status=$(curl --silent --show-error --output /dev/null --write-out '%{http_code}' \
    "$LEGAL_RAG__QDRANT__URL/collections/$collection")
  resume=()
  case "$status" in
    200)
      if [[ ! -f "data/indexes/$version/bm25.json" ]]; then
        echo "$collection exists without saved BM25 statistics; refusing to rebuild it." >&2
        exit 1
      fi
      resume=(--resume)
      ;;
    404) ;;
    *) echo "Qdrant returned HTTP $status for $collection." >&2; exit 1 ;;
  esac
  echo "Indexing $version ${resume[*]}"
  LEGAL_RAG__CORPUS__VERSION="$version" .venv/bin/python scripts/build_index.py \
    --profile structural-all --root "$PWD" --corpus-version "$version" \
    --on-disk "${resume[@]}" 2>&1 | tee -a "$report_dir/$domain-index.log"
done

# Every question searches all four indexes; no domain filter is supplied.
for domain in criminal civil administrative ip; do
  echo "Evaluating $domain: 200 questions across all four indexes"
  LEGAL_RAG__CORPUS__VERSION=civil-struct800-v1 .venv/bin/python scripts/eval_retrieval.py \
    --profile structural-all --root "$PWD" \
    --dataset "data/eval/$domain-pilot-200.jsonl" \
    --json "$report_dir/$domain-retrieval.json" 2>&1 | tee "$report_dir/$domain-evaluation.log"
done
echo "Finished. Reports: $report_dir"
