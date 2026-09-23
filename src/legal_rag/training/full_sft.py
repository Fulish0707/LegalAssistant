"""CPU-only, uncapped SFT preparation with question-selected source passages.

Lexical support screening is a triage heuristic, not an entailment verifier.
Outputs retain exact source offsets and are read lazily during training.
"""
from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re

from .dataset import DOMAINS, SPLITS, Groups, digest, file_hash, messages_for, normalized
from .tokenization import encode_example


def terms(text):
    words = re.findall(r'[a-z0-9]+|[가-힣]+', text.casefold())
    result = set(words)
    for word in words:
        if re.search('[가-힣]', word):
            result.update(word[i:i+2] for i in range(len(word)-1))
    return result


def select_passages(question, source, *, window_chars=480):
    """Rank overlapping verbatim windows using question-only lexical relevance.

    Character windows limit candidate size, not final training length. Final
    prompt + complete answer is measured with the actual model tokenizer.
    """
    if window_chars < 32:
        raise ValueError('window_chars must be at least 32')
    query = terms(question)
    passages = []
    step = max(1, window_chars * 3 // 4)
    for start in range(0, len(source), step):
        end = min(len(source), start + window_chars)
        # Prefer a nearby sentence/newline boundary at the end of the window.
        if end < len(source):
            boundary = max(source.rfind('\n', start + step, end),
                           source.rfind('. ', start + step, end))
            if boundary >= 0:
                end = boundary + 1
        text = source[start:end]
        passages.append(dict(start=start, end=end, text=text, terms=terms(text)))
        if end == len(source):
            break
    frequencies = Counter(term for p in passages for term in p['terms'] & query)
    for p in passages:
        p['score'] = sum(math.log(1 + len(passages) / frequencies[t]) for t in p['terms'] & query)
        del p['terms']
    return sorted(passages, key=lambda p: (-p['score'], p['start']))


def _rows_with_offsets(path):
    with Path(path).open('rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode('utf-8-sig'))
                if not isinstance(row, dict):
                    raise ValueError('expected object')
            except ValueError as exc:
                raise ValueError(f'{path} byte {offset}: {exc}') from exc
            yield offset, row


def _candidate(row, source, title, tokenizer, max_length, window_chars, max_passages):
    def assemble(context):
        messages = messages_for(row['question'], context, title)
        messages.append({'role': 'assistant', 'content': row['answer']})
        return dict(row, title=title, context=context, messages=messages)

    # Account for the full answer before trying any source passages.
    if encode_example(assemble(''), tokenizer, max_length=max_length) is None:
        return None, 'answer_or_prompt_too_long'
    example = assemble(source)
    encoded = encode_example(example, tokenizer, max_length=max_length)
    if encoded is not None:
        spans = [dict(start=0, end=len(source), text=source)]
    else:
        # Only the question influences candidate ranking. The answer's length
        # affects the available budget; its content is not a retrieval query.
        ranked = select_passages(row['question'], source, window_chars=window_chars)
        spans = []
        encoded = None
        for passage in ranked:
            if passage['score'] <= 0:
                continue
            if any(passage['start'] < p['end'] and p['start'] < passage['end'] for p in spans):
                continue
            proposal = sorted(spans + [passage], key=lambda p: p['start'])
            candidate = assemble('\n\n[발췌 구분]\n\n'.join(p['text'] for p in proposal))
            tokens = encode_example(candidate, tokenizer, max_length=max_length)
            if tokens is not None:
                spans, example, encoded = proposal, candidate, tokens
            if len(spans) == max_passages:
                break
        if encoded is None:
            return None, 'no_question_passage_fits'
    answer_terms = terms(row['answer'])
    coverage = len(answer_terms & terms(example['context'])) / max(1, len(answer_terms))
    # Reference presence catches some omitted evidence; it does not prove that
    # an article has the right statute, interpretation, or applicability.
    pattern = r'제\s*\d+\s*조(?:의\s*\d+)?'
    answer_refs = {normalized(x) for x in re.findall(pattern, row['answer'])}
    source_refs = {normalized(x) for x in re.findall(pattern, example['context'])}
    example.update(encoded)
    example['evidence_spans'] = [dict(start=p['start'], end=p['end']) for p in spans]
    example['answer_lexical_coverage'] = coverage
    example['missing_article_refs'] = sorted(answer_refs - source_refs)
    example['grounding_status'] = 'lexically_screened_not_verified'
    return example, None


def prepare_full_dataset(source_root, output_dir, tokenizer, *, domains=DOMAINS,
                         max_length=1024, seed=42, window_chars=480, max_passages=3,
                         min_answer_coverage=0.5, tokenizer_revision=None,
                         log=print):
    """Process every QA row, group globally, keep all eligible rows without caps.

    Source texts stay on disk; metadata and QA are kept in RAM. Splits use the
    same 80/10/10 connected-group policy as the pilot. review.jsonl records every
    exclusion. Only a completed manifest can be loaded for training.
    """
    root, out = Path(source_root).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'{out} already exists; choose a new output version')
    if not domains or len(set(domains)) != len(domains):
        raise ValueError('domains must be nonempty and unique')
    if max_length < 64 or window_chars < 32 or max_passages < 1 or not 0 <= min_answer_coverage <= 1:
        raise ValueError('invalid length, passage, or coverage settings')
    paths = [root/d/name for d in domains for name in ('documents.jsonl', 'qa.jsonl')]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    signatures = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    hashes = {str(p.relative_to(root)): file_hash(p) for p in paths}
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps({'status': 'preparing'})+'\n')

    groups, aliases, docs = Groups(), {}, {}
    stats = {d: Counter() for d in domains}
    for domain in domains:
        log(f'Indexing sources: {domain}', flush=True)
        for offset, doc in _rows_with_offsets(root/domain/'documents.jsonl'):
            identifier, source = doc.get('document_id'), doc.get('text')
            if not isinstance(identifier, str) or not isinstance(source, str):
                raise ValueError(f'{domain}: document_id/text must be strings')
            key = (domain, identifier)
            if key in docs:
                raise ValueError(f'duplicate document_id: {key}')
            node = f'{domain}:{identifier}'
            title = doc.get('title') or ''
            source_hash = digest(normalized(source))
            keys = ['id:'+identifier]
            if source.strip():
                keys.append('text:'+source_hash)
            if doc.get('document_type') == 'statute' and title.strip():
                keys.append('statute:'+normalized(title))
            groups.find(node)
            for alias in keys:
                groups.union(node, aliases.setdefault(alias, node))
            docs[key] = dict(offset=offset, node=node, title=title, source_hash=source_hash)
            stats[domain]['documents'] += 1

    rows, question_nodes, answer_hashes = [], {}, {}
    for domain in domains:
        log(f'Indexing QA: {domain}', flush=True)
        for _, qa in _rows_with_offsets(root/domain/'qa.jsonl'):
            stats[domain]['qa_rows'] += 1
            question, answer, identifier = qa.get('question'), qa.get('answer'), qa.get('document_id')
            row = dict(domain=domain, document_id=identifier, question=question, answer=answer)
            if not all(isinstance(x, str) and x.strip() for x in (question, answer, identifier)):
                row['reason'] = 'invalid_qa'
            elif (domain, identifier) not in docs:
                row['reason'] = 'unmatched_source'
            else:
                qhash = digest(normalized(question))
                node = docs[(domain, identifier)]['node']
                groups.union(node, question_nodes.setdefault(qhash, node))
                ahash = digest(normalized(answer))
                old = answer_hashes.setdefault(qhash, ahash)
                if old != ahash:
                    answer_hashes[qhash] = None  # conservative conflict exclusion
                row.update(question=question.strip(), answer=answer.strip(), question_hash=qhash,
                           id=digest(f'{domain}:{identifier}:{qhash}')[:24])
            rows.append(row)

    # Stable order also selects a deterministic representative of duplicates.
    rows.sort(key=lambda r: digest(f"{seed}:{r.get('id', str(r))}"))
    seen = set()
    with ExitStack() as stack:
        sources = {d: stack.enter_context((root/d/'documents.jsonl').open('rb')) for d in domains}
        outputs = {s: stack.enter_context((out/f'{s}.jsonl').open('w',encoding='utf-8')) for s in SPLITS}
        review = stack.enter_context((out/'review.jsonl').open('w',encoding='utf-8'))
        for number, row in enumerate(rows, 1):
            domain = row['domain']
            reason = row.get('reason')
            example = None
            if not reason:
                qhash = row['question_hash']
                if answer_hashes[qhash] is None:
                    reason = 'conflicting_question_answers'
                elif qhash in seen:
                    reason = 'duplicate_question'
                else:
                    seen.add(qhash)
                    doc = docs[(domain,row['document_id'])]
                    handle = sources[domain]
                    handle.seek(doc['offset'])
                    source = json.loads(handle.readline().decode('utf-8-sig'))['text']
                    if not source.strip():
                        reason = 'empty_source'
                    else:
                        example, reason = _candidate(row,source,doc['title'],tokenizer,max_length,window_chars,max_passages)
                        if not reason and example['answer_lexical_coverage'] < min_answer_coverage:
                            reason = 'low_answer_coverage'
                        if not reason and example['missing_article_refs']:
                            reason = 'missing_answer_article_refs'
                    if not reason:
                        group = digest(groups.find(doc['node']))
                        value = int(digest(f'{seed}:{group}')[:8],16) / 2**32
                        split = 'train' if value < .8 else ('validation' if value < .9 else 'test')
                        example.update(group_id=group, source_hash=doc['source_hash'])
                        outputs[split].write(json.dumps(example,ensure_ascii=False)+'\n')
                        stats[domain]['kept'] += 1
                        stats[domain][split] += 1
            if reason:
                record = dict(example or row, reason=reason)
                # Review files retain evidence/answers, but not large token arrays.
                record.pop('input_ids',None)
                record.pop('labels',None)
                review.write(json.dumps(record,ensure_ascii=False)+'\n')
                stats[domain][reason] += 1
            if number % 1000 == 0:
                log(f'Prepared {number:,}/{len(rows):,} QA rows', flush=True)

    for path in paths:
        if signatures[str(path)] != (path.stat().st_size,path.stat().st_mtime_ns):
            raise RuntimeError(f'Source changed during preparation: {path}; discard this output')
    for split in SPLITS:
        if any(stats[d][split] == 0 for d in domains):
            raise ValueError(f'{split}: an empty domain remains; inspect review.jsonl before training')
    report = dict(status='complete', format_version=2, domains=list(domains), source_root=str(root),
                  max_length=max_length, seed=seed, window_chars=window_chars, max_passages=max_passages,
                  min_answer_coverage=min_answer_coverage, tokenizer_revision=tokenizer_revision,
                  source_files=hashes, counts={d:dict(c) for d,c in stats.items()},
                  selection='all eligible rows, no sample caps or domain downsampling',
                  split_policy='80/10/10 hash of global connected source/question groups',
                  limitations=['Lexical coverage and reference presence do not prove answer support.',
                               'Passages are selected within the known source, not the full retrieval corpus.',
                               'Exact grouping does not detect every paraphrase or partial source overlap.',
                               'Previously inspected test questions are not a pristine final benchmark.'])
    report['files'] = {f'{s}.jsonl':file_hash(out/f'{s}.jsonl') for s in SPLITS}
    report['review_sha256'] = file_hash(out/'review.jsonl')
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


class JsonlDataset:
    """Random access by byte offset; do not load the entire token corpus into RAM."""
    def __init__(self, path, *, encoded=False):
        self.path, self.encoded = Path(path), encoded
        self.offsets = []
        with self.path.open('rb') as f:
            while True:
                offset=f.tell()
                line=f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index,slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        with self.path.open('rb') as f:
            f.seek(self.offsets[index])
            row=json.loads(f.readline())
        if self.encoded:
            return {key:row[key] for key in ('input_ids','labels')}
        row.pop('input_ids',None)
        row.pop('labels',None)
        return row


def load_full_splits(directory):
    """Verify hashes and split identities before exposing disk-backed datasets."""
    directory=Path(directory)
    report=json.loads((directory/'manifest.json').read_text())
    if report.get('status') != 'complete' or report.get('format_version') != 2:
        raise ValueError('full-data preparation is not complete')
    raw, encoded = {}, {}
    seen = {field:{} for field in ('id','group_id','question_hash')}
    for split in SPLITS:
        path=directory/f'{split}.jsonl'
        if file_hash(path) != report['files'][path.name]:
            raise ValueError(f'dataset hash mismatch: {path}')
        for _,row in _rows_with_offsets(path):
            for field,values in seen.items():
                value=row[field]
                if value in values and (values[value] != split or field != 'group_id'):
                    raise ValueError(f'data leakage or duplicate {field}: {split}')
                values[value]=split
            if len(row['input_ids']) > report['max_length'] or len(row['input_ids']) != len(row['labels']):
                raise ValueError('invalid token lengths')
            if not any(label != -100 for label in row['labels']):
                raise ValueError('no supervised answer tokens')
        raw[split]=JsonlDataset(path)
        encoded[split]=JsonlDataset(path,encoded=True)
    return raw,encoded,report
