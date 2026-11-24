from __future__ import annotations

import pytest

from app.asr import AsrError, AsrWord, parse_response, segments_from_text, segments_from_words
from app.srt import Segment, render_srt


def test_segments_from_words_split_on_punctuation_and_gap():
    words = [
        AsrWord(text="hello", start_ms=600, end_ms=1000),
        AsrWord(text=" world", start_ms=1000, end_ms=1280, punctuation="，"),
        AsrWord(text="这里是", start_ms=1400, end_ms=1920),
        AsrWord(text="阿里巴巴", start_ms=1920, end_ms=2520),
        AsrWord(text="语音实验室", start_ms=2520, end_ms=3760, punctuation="。"),
    ]
    segments = segments_from_words(words, duration_ms=4000)
    assert [segment.text for segment in segments] == [
        "hello world，",
        "这里是阿里巴巴语音实验室。",
    ]
    assert segments[0].start_ms == 600
    assert segments[0].end_ms == 1280
    assert segments[1].end_ms == 3760


def test_parse_response_reads_word_timestamps():
    body = {
        "request_id": "req-1",
        "output": {
            "text": "这里是阿里巴巴语音实验室。",
            "sentence": {
                "begin_time": 600,
                "end_time": 3760,
                "text": "这里是阿里巴巴语音实验室。",
                "words": [
                    {"begin_time": 600, "end_time": 1920, "text": "这里是", "punctuation": ""},
                    {
                        "begin_time": 1920,
                        "end_time": 3760,
                        "text": "阿里巴巴语音实验室",
                        "punctuation": "。",
                    },
                ],
            },
            "usage": {"duration": 4},
        },
        "usage": {"duration": 4},
    }
    result = parse_response(body, duration_ms=4000, model="qwen-audio-3.0-asr-flash")
    assert result.request_id == "req-1"
    assert result.text.startswith("这里是")
    assert len(result.words) == 2
    assert result.segments[-1].end_ms == 3760


def test_parse_response_estimates_when_only_text_is_returned():
    body = {"output": {"text": "第一句话。第二句话。", "usage": {"duration": 6}}}
    result = parse_response(body, duration_ms=0, model="fake")
    assert len(result.segments) == 2
    assert result.segments[0].start_ms == 0
    assert result.segments[-1].end_ms == 6000
    assert result.duration_ms == 6000


def test_parse_response_raises_when_no_speech():
    with pytest.raises(AsrError) as excinfo:
        parse_response({"output": {"text": ""}}, duration_ms=6000, model="fake")
    assert excinfo.value.code == "ASR_RESPONSE_HAVE_NO_WORDS"


def test_text_fallback_segments_do_not_overlap():
    segments = segments_from_text("第一句。第二句。第三句。", duration_ms=9000)
    assert len(segments) == 3
    for previous, current in zip(segments, segments[1:]):
        assert current.start_ms == previous.end_ms


def test_render_srt_format():
    content = render_srt(
        [
            Segment(index=1, start_ms=0, end_ms=1500, text="第一句"),
            Segment(index=2, start_ms=1500, end_ms=3661, text="第二句"),
        ]
    )
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert "00:00:01,500 --> 00:00:03,661" in content
    assert content.startswith("1\n")
