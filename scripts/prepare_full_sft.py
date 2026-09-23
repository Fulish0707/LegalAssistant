"""Prepare all eligible legal QA with question-selected passages, on CPU only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from legal_rag.training.full_sft import prepare_full_dataset
from legal_rag.training.dataset import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, default=Path('data/training/qwen3-4b-tokenizer'))
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--window-chars', type=int, default=480)
    parser.add_argument('--max-passages', type=int, default=3)
    parser.add_argument('--min-answer-coverage', type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists. Choose a new version directory.')
    revision_path = args.tokenizer_dir/'revision.txt'
    if not revision_path.is_file():
        parser.error('Use the previously downloaded tokenizer directory containing revision.txt.')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir,local_files_only=True)
    report = prepare_full_dataset(
        args.source_root,args.output,tokenizer,max_length=args.max_length,seed=args.seed,
        window_chars=args.window_chars,max_passages=args.max_passages,
        min_answer_coverage=args.min_answer_coverage,
        tokenizer_revision=revision_path.read_text().strip())
    report['tokenizer_files'] = {
        name:file_hash(args.tokenizer_dir/name)
        for name in ('tokenizer.json','tokenizer_config.json','revision.txt')
    }
    (args.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['counts'],ensure_ascii=False,indent=2))
    print('Prepared:',args.output)
    print('Inspect manifest.json and review.jsonl before training. No GPU work was run.')


if __name__ == '__main__':
    main()
