"""Human-searchable case numbers for pooled chunks.

The corpus leaves ``metadata.case_number`` empty, but ``document_id`` carries the number.
A label always falls back to ``document_id`` so a reviewer never gets a blank cell.
"""
from __future__ import annotations

import re

# '대법원-2017후905-1': a trailing '-N' separates rulings that share one case number.
_AUTHORITY_ID = re.compile(r'^(?P<authority>[^-\d]+)-(?P<number>\d+(?:[가-힣]+\d+)?)(?:-\d+)?$')
_AGENDA = re.compile(r'(\d{2}-\d{4})$')
_ARTICLE = re.compile(r'^\s*(제\d+조(?:의\d+)?)\s*\(')
_LATER_ARTICLE = re.compile(r'(?m)^\s*(제\d+조(?:의\d+)?)\s*\(')


def case_label(chunk: dict) -> dict:
    document_id, document_type = chunk['document_id'], chunk['document_type']
    authority = chunk.get('authority')
    if document_type == 'statute':
        law = chunk.get('title') or document_id.split('::')[1]
        text = chunk.get('text') or ''
        article = _ARTICLE.match(text)
        if article:
            number = query = f'{law} {article.group(1)}'
        elif later := _LATER_ARTICLE.search(text):
            # The chunk opens mid-article: point the reviewer at the next heading and scroll up.
            query = f'{law} {later.group(1)}'
            number = f'{query} 앞'
        else:
            number = query = law
        return {'case_number': number, 'authority': authority, 'search_query': query}
    prefixed = _AUTHORITY_ID.match(document_id)
    if prefixed:
        authority = authority or prefixed['authority']
        number = prefixed['number']
    elif document_type == 'legal_interpretation' and _AGENDA.search(document_id):
        number = _AGENDA.search(document_id).group(1)
    else:
        number = document_id
    query = f'{authority} {number}' if authority else number
    return {'case_number': number, 'authority': authority, 'search_query': query}
