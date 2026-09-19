import pytest

from legal_rag.evaluation.fanout_summary import corpus_mix, gold_document_metrics


def run(*rankings):
    return [{'case_id': f'c{i}', 'ranked': [{'chunk_id': f'{d}#0', 'document_id': d} for d in ranked]}
            for i, ranked in enumerate(rankings)]


def test_gold_document_metrics_count_a_case_once_however_many_chunks_hit():
    runs = {'a': run(['gold', 'gold', 'x'], ['y', 'z', 'gold'])}
    summary = gold_document_metrics(runs, {'c0': {'gold'}, 'c1': {'gold'}}, depth=10)['a']
    assert summary['cases'] == 2
    assert summary['success@1'] == 0.5 and summary['success@5'] == 1.0 and summary['success@10'] == 1.0
    assert summary['mrr@10'] == pytest.approx((1.0 + 1 / 3) / 2)


def test_gold_document_metrics_stop_at_the_depth():
    summary = gold_document_metrics({'a': run(['x', 'gold'])}, {'c0': {'gold'}}, depth=1)['a']
    assert summary['success@1'] == 0.0 and summary['mrr@1'] == 0.0
    assert 'success@5' not in summary


def test_gold_document_metrics_refuse_a_case_absent_from_the_target():
    with pytest.raises(ValueError, match='no gold documents'):
        gold_document_metrics({'a': run(['x'])}, {}, depth=1)


def test_corpus_mix_attributes_each_slot_and_keeps_unknowns():
    runs = {'fanout': run(['ip1', 'crim1', 'stray'], ['ip1', 'ip2', 'crim1'])}
    mix = corpus_mix(runs, {'ip-v1': {'ip1', 'ip2'}, 'criminal-v1': {'crim1'}}, depth=3)['fanout']
    assert mix['slots'] == 6
    assert mix['ip-v1'] == pytest.approx(0.5) and mix['criminal-v1'] == pytest.approx(1 / 3)
    assert mix['unknown'] == pytest.approx(1 / 6)
    assert sum(value for key, value in mix.items() if key != 'slots') == pytest.approx(1.0)
