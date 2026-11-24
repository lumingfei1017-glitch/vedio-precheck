from app.review import review_transcript
from app.srt import Segment, align_quote, parse_srt


SRT = """1
00:00:00,000 --> 00:00:03,000
姐妹们，上身显瘦十几斤。

2
00:00:03,000 --> 00:00:06,000
正面看是蚂蚁腰。
"""


def test_srt_parse_and_align():
    segments = parse_srt(SRT)
    assert len(segments) == 2
    start, end, confidence = align_quote("上身显瘦十几斤", segments)
    assert start == 0
    assert end == 3000
    assert confidence >= 0.9


def test_align_quote_across_three_segments():
    segments = [
        Segment(1, 640, 1960, "150的穿北极绒，"),
        Segment(2, 2520, 3840, "120的穿北极绒，"),
        Segment(3, 4000, 5520, "90斤的还是穿北极绒。"),
    ]
    start, end, confidence = align_quote(
        "150的穿北极绒，120的穿北极绒，90斤的还是穿北极绒",
        segments,
    )
    assert start == 640
    assert end == 5520
    assert confidence >= 0.9


def test_contained_size_range_finding_is_dropped():
    transcript = "150的穿北极绒，\n120的穿北极绒，\n90斤的还是穿北极绒。"
    segments = [
        Segment(1, 640, 1960, "150的穿北极绒，"),
        Segment(2, 2520, 3840, "120的穿北极绒，"),
        Segment(3, 4000, 5520, "90斤的还是穿北极绒。"),
    ]
    result = review_transcript({"has_face": 0, "category": "shapewear"}, transcript, segments)
    ap06 = [finding for finding in result.findings if finding["rule_id"] == "AP-06"]
    assert len(ap06) == 1
    assert "150的穿北极绒" in ap06[0]["quote"]
    assert ap06[0]["start_ms"] == 640
    assert ap06[0]["end_ms"] == 5520


def test_review_without_face_routes_to_a():
    material = {"has_face": 0, "category": "shapewear"}
    result = review_transcript(material, "姐妹们，上身显瘦十几斤。", parse_srt(SRT))
    assert result.findings
    assert result.findings[0]["rule_id"] == "AP-01"
    assert result.suggestions[0]["abc_class"] == "A"
    assert result.suggestions[0]["method"] == "TTS_SUBTITLE"
    assert result.suggestions[0]["estimated_duration_ms"] is not None


def test_review_with_face_routes_to_c():
    material = {"has_face": 1, "category": "shapewear"}
    result = review_transcript(material, "姐妹们，上身显瘦十几斤。", parse_srt(SRT))
    assert result.suggestions[0]["abc_class"] == "C"
    assert result.suggestions[0]["method"] == "WAN_LIPSYNC"


def test_recheck_passes_after_safe_rewrite():
    material = {"has_face": 0, "category": "shapewear"}
    result = review_transcript(material, "姐妹们，上身显瘦十几斤。", parse_srt(SRT))
    assert result.recheck["status"] == "passed"
    assert result.suggestions[0]["after_text"] != result.suggestions[0]["before_text"]
