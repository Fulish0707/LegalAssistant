from pathlib import Path

from legal_rag.evaluation import load_retrieval_examples, run_retrieval_eval

EVAL = Path(__file__).resolve().parents[1] / "fixtures" / "eval" / "retrieval.jsonl"


def test_retrieval_eval_on_fixtures(indexed, tmp_path):
    _, _, retriever = indexed
    examples = load_retrieval_examples(EVAL)
    assert len(examples) == 3 and examples[2].filters is not None
    report = run_retrieval_eval(retriever, examples, ks=(5, 10))
    assert report.aggregates["recall@5"] >= 0.5 and 0 < report.aggregates["mrr"] <= 1
    assert "recall@5" in report.table()
    report.write(tmp_path / "r.json")
    assert (tmp_path / "r.json").exists()
