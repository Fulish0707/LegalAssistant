"""CPU-only preparation. No model downloads or training dependencies required."""
import argparse
import json
from pathlib import Path

from legal_rag.training.dataset import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-per-domain', type=int, default=1000)
    parser.add_argument('--eval-per-domain', type=int, default=100)
    parser.add_argument('--max-source-chars', type=int, default=2400)
    args = parser.parse_args()
    result = prepare_dataset(args.source_root, args.output, seed=args.seed,
                             max_source_chars=args.max_source_chars,
                             per_domain={'train': args.train_per_domain,
                                         'validation': args.eval_per_domain,
                                         'test': args.eval_per_domain})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
