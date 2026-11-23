from legal_rag.verification.quantities import check_quantities, extract_quantities


def rules(findings):
    return sorted((f.rule, f.quantity) for f in findings)


def test_extracts_numbers_spelled_counts_and_units():
    found = {q.value for q in extract_quantities("두 달 전에 3개월분 월세 2기를 밀렸고 1년 6개월이 지났다. 연 5%, 300만 원.")}
    assert {"2개월", "3개월", "2기", "1년", "6개월", "5%", "300만원"} <= found


def test_article_numbers_are_not_quantities():
    assert extract_quantities("민법 제640조 제2항에 따라") == []


def test_reach_rewritten_as_exceed_is_flagged():
    evidence = ["건물 기타 공작물의 임대차에는 임차인의 차임연체액이 2기의 차임액에 달하는 때에는 임대인은 계약을 해지할 수 있다."]
    answer = "차임 연체가 2기를 초과해야 해지가 유효합니다."
    assert rules(check_quantities(answer, evidence, "월세를 두 번 밀렸어요")) == [("R1", "2기")]


def test_reach_restated_as_reach_passes():
    evidence = ["차임연체액이 2기의 차임액에 달하는 때에는 임대인은 계약을 해지할 수 있다."]
    answer = "연체액이 2기의 차임액에 이르면 임대인은 해지할 수 있습니다. 2기 이상 밀린 경우입니다."
    assert check_quantities(answer, evidence, "") == []


def test_deemed_total_rewritten_as_additional_is_flagged():
    evidence = ["기간을 정하지 아니하거나 2년 미만으로 정한 임대차는 그 기간을 2년으로 본다."]
    answer = "계약 만료 후 추가로 2년 더 거주할 수 있습니다."
    assert ("R1", "2년") in rules(check_quantities(answer, evidence, "1년 계약했어요"))


def test_deemed_total_restated_passes():
    evidence = ["그 기간을 2년으로 본다."]
    answer = "1년으로 계약했더라도 임대차 기간은 2년으로 봅니다."
    assert check_quantities(answer, evidence, "1년으로 계약했어요") == []


def test_user_point_in_time_rewritten_as_window_is_flagged():
    answer = "2개월 이내에 통지했다면 해지할 수 있습니다."
    assert rules(check_quantities(answer, [], "두 달 전에 해지 통지를 보냈어요")) == [("R2", "2개월")]


def test_user_point_in_time_as_a_predicate_is_read():
    user = "상담의 질문: 해고 통보를 받은 날짜가 언제인가요?\n답변: 두 달 전이요."
    answer = "해고 통보를 받은 날로부터 2개월 이내에 부당해고 구제신청을 할 수 있습니다."
    assert rules(check_quantities(answer, [], user)) == [("R2", "2개월")]
    assert check_quantities("3개월 전세 계약입니다.", [], "3개월 전세로 살아요") == []


def test_user_point_in_time_restated_passes():
    answer = "두 달 전에 통지하셨으므로, 통지 후 3개월이 지나야 효력이 생깁니다."
    evidence = ["임대인이 그 통지를 받은 날부터 3개월이 지나면 그 효력이 발생한다."]
    assert check_quantities(answer, evidence, "두 달 전에 해지 통지를 보냈어요") == []


def test_quantity_in_no_source_is_flagged():
    answer = "24시간 이내에 보증금을 돌려받아야 합니다."
    assert rules(check_quantities(answer, ["임대차가 종료한 경우 보증금을 반환하여야 한다."], "")) == [("R3", "24시간")]


def test_quantity_computed_from_sources_is_not_flagged():
    evidence = ["손해 및 가해자를 안 날부터 3년간 행사하지 아니하면 시효로 인하여 소멸한다.", "수선비의 75%는 임대인이 부담한다."]
    answer = "사고 후 4년이 지났고 3년 기간을 1년 넘겼습니다. 75%는 임대인 부담이고 나머지 25%는 임차인 부담입니다."
    assert check_quantities(answer, evidence, "4년 전에 사고가 났어요") == []


def test_hedge_percentages_are_ignored():
    assert check_quantities("100% 확정할 수는 없습니다.", [], "") == []


def test_additional_period_flagged_only_against_a_deemed_total():
    evidence = ["부득이한 사유가 있으면 10일의 범위에서 연장할 수 있다.", "그 통지를 받은 날부터 3개월이 지나면 효력이 생긴다."]
    answer = "부득이하면 추가 10일 연장이 가능합니다. 통지를 받은 날로부터 추가로 3개월이 지나야 합니다."
    assert check_quantities(answer, evidence, "") == []


def test_traditional_rates_and_fractions_are_read_as_percent():
    evidence = ["이자 있는 채권의 이율은 연 5푼으로 한다.", "증액청구는 약정한 차임의 20분의 1의 금액을 초과하지 못한다."]
    assert check_quantities("연 5%의 법정이율이 적용되고, 증액은 5%를 넘을 수 없습니다.", evidence, "") == []
    flagged = rules(check_quantities("증액 한도는 20%이고 이율은 연 0.5%입니다.", evidence, "월세 60만 원"))
    assert flagged == [("R3", "0.5%"), ("R3", "20%")]


def test_hypothetical_premise_counts_as_a_source_but_wrong_results_do_not():
    evidence = ["차임연체액이 2기의 차임액에 달하는 때에는 해지할 수 있다."]
    ok = "예: 한 달 차임이 50만 원이라면, 2기 차임은 100만 원입니다."
    assert check_quantities(ok, evidence, "") == []
    wrong = "예를 들어, 월세가 60만 원인 경우 증액은 12만 원을 초과할 수 없습니다."
    assert rules(check_quantities(wrong, ["증액은 차임의 20분의 1을 초과하지 못한다."], "")) == [("R3", "12만원")]


def test_dates_inside_examples_are_skipped():
    answer = "예를 들어, 변제기일이 2024년 1월 1일이었다면, 6개월 후인 2024년 7월 1일부터 청구할 수 있습니다."
    assert check_quantities(answer, ["변제기로부터 6개월"], "") == []


def test_terms_and_counts_are_the_same_unit():
    assert check_quantities("'3기'는 3번의 연체를 뜻합니다.", ["3기의 차임액에 달하도록 연체한 경우"], "") == []


def test_monthly_rate_from_an_annual_rate_is_computed():
    assert check_quantities("연 5%는 월 약 0.417%입니다.", ["연 5푼"], "") == []


def test_rate_followed_by_a_particle_is_read():
    assert check_quantities("법정이율은 연 5%입니다.", ["이율은 연 5푼으로 한다."], "") == []
