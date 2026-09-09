from legal_rag.retrieval.sparse import BM25Stats, term_id


def test_fit_and_weights():
    docs = {"a": ["보증금", "반환", "보증금"], "b": ["불법행위", "손해"], "c": ["보증금", "손해"]}
    stats = BM25Stats.fit(docs, k1=1.2, b=0.75)
    assert stats.n_docs == 3 and stats.avg_len == (3 + 2 + 2) / 3
    assert stats.idf("보증금") < stats.idf("불법행위")
    doc_vec = stats.document_vector(["보증금", "반환", "보증금"])
    weights = dict(zip(doc_vec.indices, doc_vec.values))
    assert weights[term_id("보증금")] > weights[term_id("반환")]
    query_vec = stats.query_vector(["보증금", "없는단어"])
    assert query_vec.indices == [term_id("보증금")]


def test_dot_product_equals_bm25_ranking():
    docs = {"a": ["보증금", "반환"], "b": ["보증금", "보증금", "반환", "반환", "임대차", "계약"], "c": ["손해"]}
    stats = BM25Stats.fit(docs, k1=1.2, b=0.75)
    q = stats.query_vector(["보증금"])
    def score(tokens):
        d = dict(zip(*stats.document_vector(tokens).__dict__.values()))
        return sum(d.get(i, 0.0) * v for i, v in zip(q.indices, q.values))
    assert score(docs["a"]) > 0 and score(docs["c"]) == 0


def test_round_trip_json(tmp_path):
    stats = BM25Stats.fit({"a": ["x", "y"], "b": ["y"]}, k1=1.2, b=0.75)
    stats.save(tmp_path / "bm25.json")
    loaded = BM25Stats.load(tmp_path / "bm25.json")
    assert loaded.idf("y") == stats.idf("y") and loaded.avg_len == stats.avg_len and loaded.k1 == 1.2
