from app.rules import load_rules, scan


def test_quantified_shape_claim_is_blocking():
    matches = scan("姐妹们，上身显瘦十几斤，正面看是蚂蚁腰。")
    rule_ids = {match.rule.id for match in matches}
    assert "AP-01" in rule_ids
    assert "AP-02" in rule_ids
    assert all(match.rule.severity == "blocking" for match in matches if match.rule.id in {"AP-01", "AP-02"})


def test_before_after_claim():
    matches = scan("穿前小肚子，穿后曲线明显。")
    assert any(match.rule.id == "AP-03" for match in matches)


def test_neutral_belly_fat_mention_is_not_blocking():
    text = "你如果肚子没有赘肉，不要拍。我们肚子上只要有赘肉的姐妹，可以看看这条。"
    matches = scan(text)
    assert not [match for match in matches if match.rule.id == "AP-05"]


def test_belly_fat_efficacy_claim_is_blocking():
    matches = scan("能够让你的赘肉慢慢紧致起来，脂肪一整天都被收紧夹住。")
    assert any(match.rule.id == "AP-05" and match.rule.severity == "blocking" for match in matches)


def test_flat_stomach_claim_is_blocking():
    matches = scan("不管你肚子有多大，它都能够给你整理平整了。")
    assert any(match.rule.id == "AP-15" for match in matches)


def test_lift_and_tuck_claim_is_blocking():
    matches = scan("同时提拉紧致，给你该收的地方给你收进去，臀部又给你往上托。")
    assert any(match.rule.id == "AP-15" for match in matches)


def test_body_temperature_tech_claim_is_blocking():
    matches = scan("模拟人体体温，更好的呵护我们的腹部。")
    assert any(match.rule.id == "AP-08" for match in matches)


def test_size_range_claim_with_digits_and_chinese_numbers():
    matches = scan("一百五的穿，一百二的穿，九十斤的还是穿。")
    assert any(match.rule.id == "AP-06" for match in matches)


def test_size_range_claim_across_subtitle_lines_keeps_full_quote():
    text = "150的穿北极绒，\n120的穿北极绒，\n90斤的还是穿北极绒。"
    matches = [match for match in scan(text) if match.rule.id == "AP-06"]
    assert matches
    quote = matches[0].quote
    assert "150的穿北极绒" in quote
    assert "120的穿北极绒" in quote
    assert "90斤的还是穿北极绒" in quote


def test_weight_loss_context_is_not_suppressed_by_bare_negative():
    matches = scan("到现在还没有减肥成功呢，穿上它你的肚子就解决了。")
    assert any(match.rule.id == "AP-05" for match in matches)


def test_wearing_for_months_claim():
    matches = scan("连续穿它三个月，你就可以不用再穿了。")
    assert any(match.rule.id == "AP-04" for match in matches)


def test_wearing_for_months_claim_across_subtitle_lines():
    matches = scan("连续穿它三个月，\n你就可以不用再穿了。")
    assert any(match.rule.id == "AP-04" for match in matches)


def test_wearing_for_digit_months_claim_across_subtitle_lines():
    matches = scan("连续穿了3个月，\n你就可以不用再穿了。")
    assert any(match.rule.id == "AP-04" for match in matches)


def test_small_quantified_shape_change():
    matches = scan("一上身就能穿出显瘦一二斤的效果。")
    assert any(match.rule.id == "AP-01" for match in matches)


def test_quantified_shape_range():
    matches = scan("一上身就能穿出显瘦10~20斤的效果。")
    assert any(match.rule.id == "AP-01" for match in matches)


def test_shape_fixing_claim():
    matches = scan("身材能塑形定型，还能把保暖和塑形做到二合一。")
    assert any(match.rule.id == "AP-02" for match in matches)


def test_stomach_solution_claim():
    matches = scan("姐妹，你现在穿上这件塑形裤，把你的肚子完美的解决了。")
    assert any(match.rule.id == "AP-15" for match in matches)


def test_thigh_fat_compression_claim():
    matches = scan("收压紧致大腿根部的赘肉。")
    assert any(match.rule.id == "AP-05" for match in matches)


def test_antibacterial_needs_material():
    matches = scan("底裆是抗菌抑菌面料。")
    assert any(match.rule.id == "AP-09" and match.rule.action == "request_material" for match in matches)


def test_clean_copy_has_no_blocking_match():
    matches = scan("这款保暖内衣面料柔软，日常穿着很舒服，搭配高腰裤更利落。")
    assert not [match for match in matches if match.rule.severity == "blocking"]


def test_duplicate_ap01_is_deduplicated():
    matches = [match for match in scan("上身显瘦十几斤。") if match.rule.id == "AP-01"]
    assert len(matches) == 1
    assert matches[0].match_text == "显瘦十几斤"


def test_rule_pack_loads():
    rules = load_rules()
    assert len([rule for rule in rules if rule.id.startswith("AP-")]) == 14
    assert {rule.id for rule in rules} >= {"AP-01", "AP-03", "AP-09", "AP-15"}
