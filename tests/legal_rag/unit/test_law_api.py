import pytest

from legal_rag.ingestion.law_api import LawApiError, convert_law, fetch_law, fetch_body, list_current_laws
from legal_rag.ingestion.normalize.statute_json import normalize_statute_json


def _base(name='테스트법', effective='20260102', promulgated='20251001'):
    return {'법령명_한글': name, '시행일자': effective, '공포일자': promulgated, '법종구분': {'content': '법률'}}


def _body(units, **base):
    return {'법령': {'기본정보': _base(**base), '조문': {'조문단위': units}}}


def _article(number, content, *, branch=None, title=None, paragraphs=None):
    unit = {'조문번호': str(number), '조문여부': '조문', '조문내용': content, '조문제목': title or ''}
    if branch:
        unit['조문가지번호'] = str(branch)
    if paragraphs:
        unit['항'] = paragraphs
    return unit


def test_articles_paragraphs_items_and_branch_numbers_are_converted():
    body = _body([
        {'조문번호': '0', '조문여부': '전문', '조문내용': '제1장 총칙'},
        _article(1, '제1조(목적) 이 법은 임대차를 규율한다.', title='목적'),
        _article(3, '제3조의2(보증금의 회수)', branch=2, title='보증금의 회수', paragraphs=[
            {'항번호': '①', '항내용': '① 임차인은 경매를 신청할 수 있다.'},
            {'항번호': '②', '항내용': '② 다음 각 호의 사항을 적어야 한다.', '호': [{'호번호': '1.', '호내용': '1. 신청의 취지'}, {'호번호': '2.', '호내용': '2. 이유'}]},
        ]),
    ])
    record = convert_law(body)
    assert record['law_name'] == '테스트법' and record['effective_date'] == '2026-01-02' and record['promulgation_date'] == '2025-10-01'
    first, second = record['articles']
    assert (first['number'], first['title'], first['text'], first['chapter']) == ('제1조', '목적', '이 법은 임대차를 규율한다.', '제1장 총칙')
    assert second['number'] == '제3조의2' and second['text'] == ''
    # the leading ①, ② and 1. are numbering, not text: the renderer prints them again
    assert [p['text'] for p in second['paragraphs']] == ['임차인은 경매를 신청할 수 있다.', '다음 각 호의 사항을 적어야 한다.']
    assert [i['text'] for i in second['paragraphs'][1]['items']] == ['신청의 취지', '이유']


def test_a_one_article_law_arrives_as_a_dict_not_a_list():
    """26 laws were skipped at first because 조문단위 was one dict and the code iterated its keys."""
    record = convert_law(_body(_article(1, '제1조(목적) 이 규칙은 목적을 정한다.', title='목적')))
    assert [a['number'] for a in record['articles']] == ['제1조']


def test_a_pointer_to_a_moved_article_is_skipped_but_its_neighbours_are_kept():
    record = convert_law(_body([_article(2, '제2조의2', branch=2), _article(3, '제3조(정의) 정의한다.', title='정의')]))
    assert [a['number'] for a in record['articles']] == ['제3조']


def test_unnumbered_body_text_has_no_article_structure():
    assert convert_law(_body({'조문번호': '0', '조문여부': '전문', '조문내용': '경상북도 군위군을 대구광역시의 관할구역에 편입한다.'})) is None


def test_the_converted_record_passes_the_ingestion_normalizer():
    record = convert_law(_body([_article(3, '제3조의2(회수)', branch=2, title='회수', paragraphs=[{'항번호': '①', '항내용': '① 본문이다.'}])]))
    statute = normalize_statute_json(record, dataset='statute-api', file='1.json')
    assert statute.law_name == '테스트법' and statute.articles[0].number == '제3조의2'
    assert '본문이다.' in statute.text


def test_an_error_body_with_http_200_is_an_error_not_an_empty_result():
    rejected = {'result': '사용자 정보 검증에 실패하였습니다.', 'msg': 'OPEN API 호출 시 사용자 검증을 위하여 정확한 서버장비의 IP주소 및 도메인주소를 등록해 주세요.'}
    with pytest.raises(LawApiError, match='검증'):
        fetch_body('bad', '1', get=lambda url: rejected)
    with pytest.raises(LawApiError):
        list_current_laws('bad', get=lambda url: rejected)


def test_listing_follows_pages_until_one_is_empty():
    pages = {1: [{'법령명한글': '가법'}, {'법령명한글': '나법'}], 2: [{'법령명한글': '다법'}], 3: None}
    urls = []

    def get(url):
        urls.append(url)
        page = int(url.rsplit('page=', 1)[1])
        batch = pages[page]
        return {'LawSearch': {'totalCnt': '3', **({'law': batch} if batch else {})}}

    assert [law['법령명한글'] for law in list_current_laws('key', get=get)] == ['가법', '나법', '다법']
    assert len(urls) == 3 and 'OC=key' in urls[0]


def test_a_single_result_page_with_one_law_is_a_dict():
    assert len(list_current_laws('k', get=lambda url: {'LawSearch': {'law': {'법령명한글': '유일법'}}} if url.endswith('page=1') else {'LawSearch': {}})) == 1


def test_fetch_law_retries_network_errors_but_not_a_rejected_key():
    calls = []

    def flaky(url):
        calls.append(url)
        raise OSError('connection reset')

    with pytest.raises(OSError):
        fetch_law('k', {'법령일련번호': '9'}, get=flaky, retries=3, delay=0)
    assert len(calls) == 3

    calls.clear()

    def rejected(url):
        calls.append(url)
        return {'result': '실패', 'msg': '키 오류'}

    with pytest.raises(LawApiError):
        fetch_law('k', {'법령일련번호': '9'}, get=rejected, retries=3, delay=0)
    assert len(calls) == 1
