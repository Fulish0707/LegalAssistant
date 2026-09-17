from legal_rag.evaluation.case_labels import case_label


def chunk(document_id, document_type, *, title='', text='본문', authority=None, date=None):
    return dict(document_id=document_id, document_type=document_type, title=title, text=text,
                authority=authority, date=date)


def test_court_judgment_splits_court_from_number():
    label = case_label(chunk('서울중앙지방법원-2016노4825', 'judgment', authority='서울중앙지방법원'))
    assert label == {'case_number': '2016노4825', 'authority': '서울중앙지방법원',
                     'search_query': '서울중앙지방법원 2016노4825'}


def test_patent_court_judgment():
    assert case_label(chunk('특허법원-2010허8443', 'judgment'))['search_query'] == '특허법원 2010허8443'


def test_interpretation_keeps_agenda_number():
    label = case_label(chunk('법제처-법령해석사례-법제처-09-0381', 'legal_interpretation', authority='법제처'))
    assert label['case_number'] == '09-0381'
    assert label['search_query'] == '법제처 09-0381'


def test_statute_uses_law_name_and_opening_article():
    label = case_label(chunk('statute::상표법::2024-08-07 00:00:00', 'statute', title='상표법',
                             text='제73조의2(상표등록의 취소심판)\n...'))
    assert label['case_number'] == '상표법 제73조의2'
    assert label['search_query'] == '상표법 제73조의2'
    older = case_label(chunk('statute::특허보상심의회규정::1970', 'statute', title='특허보상심의회규정',
                             text='제1조 (설치)\n...'))
    assert older['case_number'] == '특허보상심의회규정 제1조'


def test_statute_without_article_falls_back_to_law_name():
    label = case_label(chunk('statute::상표법::2024', 'statute', title='상표법', text='부칙 ...'))
    assert label['case_number'] == '상표법'


def test_statute_opening_mid_article_points_before_next_heading():
    label = case_label(chunk('statute::상표법::2024', 'statute', title='상표법',
                             text='리인을 포함한다.\n1. 등록출원\n제18조의2(절차의 정지)\n...'))
    assert label['case_number'] == '상표법 제18조의2 앞'
    assert label['search_query'] == '상표법 제18조의2'


def test_statute_ignores_in_sentence_article_reference():
    label = case_label(chunk('statute::상표법::2024', 'statute', title='상표법',
                             text='제73조제1항에 따라\n제73조제1항에 따라 취소한다.\n1. 제36조(출원)에 따른'))
    assert label['case_number'] == '상표법'


def test_other_types_use_document_id_never_blank():
    label = case_label(chunk('2011101003196', 'patent_trial_decision', authority='특허심판원'))
    assert label == {'case_number': '2011101003196', 'authority': '특허심판원',
                     'search_query': '특허심판원 2011101003196'}
    assert case_label(chunk('2022기정1485', 'decision'))['search_query'] == '2022기정1485'


def test_constitutional_court_and_high_court_branch():
    assert case_label(chunk('헌법재판소-2015헌마545', 'judgment'))['search_query'] == '헌법재판소 2015헌마545'
    label = case_label(chunk('대전고등법원청주-2019누2344', 'judgment', authority='대전고등법원'))
    assert label['search_query'] == '대전고등법원 2019누2344'


def test_judgment_drops_ruling_suffix():
    assert case_label(chunk('대법원-2017후905-1', 'judgment'))['case_number'] == '2017후905'


def test_procurement_interpretation_number():
    label = case_label(chunk('조달청-1802010026', 'legal_interpretation', authority='조달청'))
    assert label['search_query'] == '조달청 1802010026'
