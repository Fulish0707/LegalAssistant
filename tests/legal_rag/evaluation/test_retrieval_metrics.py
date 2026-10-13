import math

from legal_rag.evaluation.retrieval_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

RANKED = ["a", "b", "c", "d"]
REL = {"b", "d", "z"}


def test_recall_and_precision():
    assert recall_at_k(RANKED, REL, 2) == 1 / 3
    assert recall_at_k(RANKED, REL, 4) == 2 / 3
    assert precision_at_k(RANKED, REL, 4) == 0.5
    assert recall_at_k(RANKED, set(), 4) == 0.0


def test_mrr():
    assert mrr(RANKED, REL) == 0.5
    assert mrr(["x"], REL) == 0.0


def test_ndcg():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(RANKED, REL, 4) - dcg / idcg) < 1e-9
    assert ndcg_at_k(["b", "d"], {"b", "d"}, 2) == 1.0
