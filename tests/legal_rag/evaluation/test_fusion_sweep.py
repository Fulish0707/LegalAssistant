import json
from pathlib import Path

import pytest

from legal_rag.evaluation.relevance.fusion_sweep import pair_ranks, pending_pairs, refuse
from legal_rag.evaluation.relevance.relevance_pool import read_jsonl

REAL_POOL = Path('data/eval/relevance/ip-v1-pilot-200-top10')


def run(case_id, chunk_ids):
    return {'case_id': case_id, 'ranked': [{'chunk_id': c, 'document_id': c.split('#')[0]} for c in chunk_ids]}


def ranked(rows, case_id):
    return [hit['chunk_id'] for row in rows if row['case_id'] == case_id for hit in row['ranked']]


def test_refuse_sums_weighted_reciprocal_ranks_and_deduplicates_on_chunk_id():
    legs = {'dense': [run('c1', ['a#0', 'b#0'])], 'sparse': [run('c1', ['b#0', 'z#0'])]}
    fused = refuse(legs, {'dense': 1.0, 'sparse': 1.0}, rrf_k=60, depth=10)
    # b#0 scores 1/61 + 1/62 and wins; a#0 and z#0 tie at 1/61 and 1/62 respectively.
    assert ranked(fused, 'c1') == ['b#0', 'a#0', 'z#0']
    assert fused == [{'case_id': 'c1', 'ranked': [
        {'chunk_id': 'b#0', 'document_id': 'b'}, {'chunk_id': 'a#0', 'document_id': 'a'},
        {'chunk_id': 'z#0', 'document_id': 'z'}]}]


def test_ties_break_on_chunk_id_ascending_like_rrffuser():
    legs = {'dense': [run('c1', ['b#0'])], 'sparse': [run('c1', ['a#0'])]}
    assert ranked(refuse(legs, {'dense': 1.0, 'sparse': 1.0}, rrf_k=60, depth=10), 'c1') == ['a#0', 'b#0']


def test_zero_weight_leg_contributes_nothing_and_its_own_chunks_drop_out():
    """RRFFuser keeps only ``score > 0``, so a chunk seen by the zero-weighted leg alone vanishes."""
    legs = {'dense': [run('c1', ['a#0'])], 'sparse': [run('c1', ['z#0', 'a#0'])]}
    fused = refuse(legs, {'dense': 1.0, 'sparse': 0.0}, rrf_k=60, depth=10)
    assert ranked(fused, 'c1') == ['a#0']


def test_zero_weight_sweep_variant_reproduces_the_dense_leg():
    dense = [run('c1', ['a#0', 'b#0', 'c#0'])]
    legs = {'dense': dense, 'sparse': [run('c1', ['z#0', 'y#0'])]}
    assert refuse(legs, {'dense': 1.0, 'sparse': 0.0}, rrf_k=60, depth=3) == dense


def test_weighting_reorders_towards_the_heavier_leg():
    legs = {'dense': [run('c1', ['a#0', 'b#0'])], 'sparse': [run('c1', ['b#0', 'a#0'])]}
    assert ranked(refuse(legs, {'dense': 1.0, 'sparse': 0.1}, rrf_k=60, depth=2), 'c1') == ['a#0', 'b#0']
    assert ranked(refuse(legs, {'dense': 0.1, 'sparse': 1.0}, rrf_k=60, depth=2), 'c1') == ['b#0', 'a#0']


def test_depth_truncates_after_scoring_not_before():
    legs = {'dense': [run('c1', ['a#0', 'b#0', 'c#0'])], 'sparse': [run('c1', ['c#0'])]}
    assert ranked(refuse(legs, {'dense': 1.0, 'sparse': 1.0}, rrf_k=60, depth=1), 'c1') == ['c#0']


def test_missing_weight_defaults_to_one_like_rrffuser():
    legs = {'dense': [run('c1', ['a#0'])], 'sparse': [run('c1', ['z#0'])]}
    assert refuse(legs, {}, rrf_k=60, depth=2) == refuse(legs, {'dense': 1.0, 'sparse': 1.0}, rrf_k=60, depth=2)


def test_case_order_is_kept_and_every_leg_must_cover_the_same_cases():
    legs = {'dense': [run('c2', ['a#0']), run('c1', ['b#0'])], 'sparse': [run('c2', ['a#0']), run('c1', ['b#0'])]}
    assert [row['case_id'] for row in refuse(legs, {}, rrf_k=60, depth=1)] == ['c2', 'c1']
    with pytest.raises(ValueError, match='same cases'):
        refuse({'dense': [run('c1', ['a#0'])], 'sparse': [run('c2', ['a#0'])]}, {}, rrf_k=60, depth=1)


def test_refuse_rejects_empty_legs_and_bad_depth():
    with pytest.raises(ValueError, match='no leg runs'):
        refuse({}, {}, rrf_k=60, depth=1)
    with pytest.raises(ValueError, match='depth'):
        refuse({'dense': [run('c1', ['a#0'])]}, {}, rrf_k=60, depth=0)


def test_pair_ranks_records_each_mode_rank_within_depth():
    runs = {'dense': [run('c1', ['a#0', 'b#0', 'c#0'])], 'hybrid_s0.50': [run('c1', ['b#0', 'a#0'])]}
    assert pair_ranks(runs, 2) == {('c1', 'a#0'): {'dense': 1, 'hybrid_s0.50': 2},
                                   ('c1', 'b#0'): {'dense': 2, 'hybrid_s0.50': 1}}


def test_pending_pairs_are_the_unjudged_pairs_within_depth_sorted_and_distinct():
    runs = {'dense': [run('c1', ['a#0', 'b#0'])], 'hybrid_s0.50': [run('c1', ['b#0', 'z#0'])]}
    assert pending_pairs(runs, {('c1', 'a#0')}, 2) == [('c1', 'b#0'), ('c1', 'z#0')]
    assert pending_pairs(runs, {('c1', 'a#0'), ('c1', 'b#0'), ('c1', 'z#0')}, 2) == []


@pytest.mark.skipif(not (REAL_POOL / 'runs' / 'hybrid.jsonl').exists(), reason=f'no pool at {REAL_POOL}')
def test_equal_weight_refusion_reproduces_the_recorded_hybrid_run():
    """The pool filters to one corpus, so dense.jsonl and sparse.jsonl are the whole fusion input.

    If this fails, the recorded leg runs or RRFFuser are not what the sweep assumes and no
    re-weighted variant can be trusted.
    """
    rrf_k = json.loads((REAL_POOL / 'pool.json').read_text(encoding='utf-8')).get('rrf_k', 60)
    legs = {leg: read_jsonl(REAL_POOL / 'runs' / f'{leg}.jsonl') for leg in ('dense', 'sparse')}
    recorded = read_jsonl(REAL_POOL / 'runs' / 'hybrid.jsonl')
    depth = max(len(row['ranked']) for row in recorded)
    assert refuse(legs, {'dense': 1.0, 'sparse': 1.0}, rrf_k=rrf_k, depth=depth) == recorded
