import json

import pytest

from legal_rag.training.dataset import prepare_dataset


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def corpus(root):
    for domain in ('civil', 'criminal'):
        docs = [{'document_id': f'{domain}-{i}', 'title': f'case {domain} {i}',
                 'document_type': 'judgment', 'text': f'evidence {domain} {i}'} for i in range(300)]
        qa = [{'document_id': d['document_id'], 'question': f'question {domain} {i}',
               'answer': f'answer {i}'} for i, d in enumerate(docs)]
        # Same statute title across versions and domains must stay together.
        docs[0].update(title='Same Law', document_type='statute')
        # Exact source copy with different ID must stay together.
        docs[1]['text'] = 'shared source'
        # Duplicate question links two otherwise distinct sources.
        qa[2]['question'] = 'shared question'
        qa += [{'document_id': 'missing', 'question': 'unmatched', 'answer': 'answer'}]
        docs[3]['text'] = 'x' * 3000
        write(root / domain / 'documents.jsonl', docs)
        write(root / domain / 'qa.jsonl', qa)


def read_all(path):
    return {s: [json.loads(x) for x in (path / f'{s}.jsonl').read_text().splitlines()]
            for s in ('train', 'validation', 'test')}


def test_preparation_balances_splits_and_is_repeatable(tmp_path):
    root = tmp_path / 'corpus'
    corpus(root)
    args = dict(domains=('civil', 'criminal'), per_domain={'train': 12, 'validation': 3, 'test': 3})
    a = prepare_dataset(root, tmp_path / 'a', **args)
    b = prepare_dataset(root, tmp_path / 'b', **args)
    for split, rows in read_all(tmp_path / 'a').items():
        assert len(rows) == 2 * args['per_domain'][split]
        assert (tmp_path / 'a' / f'{split}.jsonl').read_bytes() == (tmp_path / 'b' / f'{split}.jsonl').read_bytes()
        for row in rows:
            assert row['messages'][-1]['role'] == 'assistant'
            assert row['context'] in row['messages'][1]['content']
            assert row['answer'] == row['messages'][-1]['content']
            assert row['document_id'] != 'missing'
            assert len(row['context']) <= 2400
    assert a['domains']['civil']['unmatched'] == 1
    assert a['domains']['criminal']['source_too_long'] == 1
    assert a['files'] == b['files']
    with pytest.raises(FileExistsError):
        prepare_dataset(root, tmp_path / 'a', **args)


def test_grouping_before_sampling_keeps_related_sources_together(tmp_path):
    root = tmp_path / 'corpus'
    corpus(root)
    prepare_dataset(root, tmp_path / 'out', domains=('civil', 'criminal'),
                    per_domain={'train': 1000, 'validation': 1000, 'test': 1000})
    rows = read_all(tmp_path / 'out')
    seen_groups = {}
    seen_questions = set()
    shared = {}
    for split, examples in rows.items():
        for row in examples:
            group = row['group_id']
            assert seen_groups.setdefault(group, split) == split
            assert row['question'] not in seen_questions
            seen_questions.add(row['question'])
            if row['document_id'].endswith('-0'):
                assert shared.setdefault('statute', group) == group
            if row['document_id'].endswith('-1'):
                assert shared.setdefault('text', group) == group


def test_missing_domain_fails_before_writing(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_dataset(tmp_path, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_conflicting_answers_are_excluded_across_domains(tmp_path):
    root = tmp_path / 'corpus'
    corpus(root)
    path = root / 'criminal' / 'qa.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[2]['answer'] = 'a different answer to the shared question'
    write(path, rows)
    manifest = prepare_dataset(root, tmp_path / 'out', domains=('civil', 'criminal'),
                               per_domain={'train': 1000, 'validation': 1000, 'test': 1000})
    assert manifest['domains']['civil']['conflicting_question_answers'] == 1
    assert manifest['domains']['criminal']['conflicting_question_answers'] == 1
    assert all(row['question'] != 'shared question'
               for rows in read_all(tmp_path / 'out').values() for row in rows)


def test_duplicate_document_ids_fail_instead_of_choosing_arbitrary_evidence(tmp_path):
    root = tmp_path / 'corpus'
    corpus(root)
    path = root / 'civil' / 'documents.jsonl'
    with path.open('a') as handle:
        handle.write(json.dumps({'document_id': 'civil-0', 'text': 'different source'}) + '\n')
    with pytest.raises(ValueError, match='duplicate document_id'):
        prepare_dataset(root, tmp_path / 'out', domains=('civil', 'criminal'))
    assert not (tmp_path / 'out').exists()
