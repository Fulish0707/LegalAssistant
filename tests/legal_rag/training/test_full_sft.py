import json

import pytest

from legal_rag.training.full_sft import prepare_full_dataset, load_full_splits, select_passages


class CharTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return ''.join(m['content'] for m in messages) + '<assistant>'

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text]}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))


def fixture(root):
    for domain, count in [('civil', 150), ('criminal', 190)]:
        docs, qa = [], []
        for i in range(count):
            text = f'The remedy for defect {domain} {i} is repair.'
            docs.append(dict(document_id=f'{domain}-{i}', text=text, document_type='judgment'))
            qa.append(dict(document_id=f'{domain}-{i}', question=f'What is the remedy for defect {domain} {i}?', answer=text))
        # A duplicate across corpora cannot end up on both sides of a split.
        docs[0]['text'] = 'The common remedy is repair.'
        qa[0].update(question='What is the common remedy?', answer=docs[0]['text'])
        write(root/domain/'documents.jsonl', docs)
        write(root/domain/'qa.jsonl', qa)


def test_passages_use_question_and_preserve_source_offsets():
    text = 'Unrelated weather information.\n' * 80 + 'The remedy for a defect is repair.\n' + 'Unrelated weather information.\n' * 80
    passages = select_passages('What is the remedy for a defect?', text, window_chars=120)
    assert 'remedy' in passages[0]['text']
    assert all(text[p['start']:p['end']] == p['text'] for p in passages)


def test_full_data_no_caps_no_downsampling_and_disjoint(tmp_path):
    fixture(tmp_path/'input')
    report = prepare_full_dataset(tmp_path/'input', tmp_path/'out', CharTokenizer(),
                                  domains=('civil','criminal'), max_length=600)
    raw, encoded, loaded_report = load_full_splits(tmp_path/'out')
    assert sum(map(len, raw.values())) == 339  # 340 rows minus one exact duplicate
    assert report['counts']['criminal']['kept'] != report['counts']['civil']['kept']
    seen = set()
    for split in raw:
        assert len(raw[split]) == len(encoded[split])
        for r, e in zip(raw[split], encoded[split]):
            assert r['id'] not in seen
            seen.add(r['id'])
            assert len(e['input_ids']) <= 600
            assert e['labels'][-1] == 1
            assert r['answer'] == r['messages'][-1]['content']
    assert loaded_report['status'] == 'complete'
    with pytest.raises(FileExistsError):
        prepare_full_dataset(tmp_path/'input',tmp_path/'out',CharTokenizer())


def test_overlong_answers_and_unsupported_candidates_go_to_review(tmp_path):
    fixture(tmp_path/'input')
    p=tmp_path/'input/civil/qa.jsonl'
    rows=[json.loads(line) for line in p.read_text().splitlines()]
    rows[1]['answer']='x'*900
    rows[2]['answer']='zzzzzzzzzzzzzzzz'
    write(p,rows)
    report=prepare_full_dataset(tmp_path/'input',tmp_path/'out',CharTokenizer(),domains=('civil','criminal'), max_length=600)
    reviews=[json.loads(line) for line in (tmp_path/'out/review.jsonl').read_text().splitlines()]
    assert any(r['reason']=='answer_or_prompt_too_long' for r in reviews)
    assert any(r['reason']=='low_answer_coverage' for r in reviews)
    assert report['counts']['civil']['kept'] < 150


def test_loader_rejects_changed_output(tmp_path):
    fixture(tmp_path/'input')
    prepare_full_dataset(tmp_path/'input',tmp_path/'out',CharTokenizer(),domains=('civil','criminal'),max_length=600)
    with (tmp_path/'out/train.jsonl').open('a') as f:
        f.write('{}\n')
    with pytest.raises(ValueError, match='hash'):
        load_full_splits(tmp_path/'out')
