"""Dashboard endpoint tests: summary shape, message filters, missing-artifact guard."""

import json

import main


def test_summary_shape() -> None:
    s = main.dashboard_summary(sentiment=None, topic=None, q="")
    assert s["total"] == 2000
    assert s["loaded"] == 2000
    assert set(s["counts"]) == {"pos", "neu", "neg", "q"}
    assert len(s["per_topic"]) >= 3
    first = s["per_topic"][0]
    assert first["total"] == first["pos"] + first["neu"] + first["neg"] + first["q"]
    assert set(s["sentiments"]) >= {"pos", "neu", "neg"}
    assert -1 in s["topics"]


def test_summary_filters_narrow() -> None:
    s = main.dashboard_summary(sentiment=["q"], topic=[1], q="ราคา")
    assert s["total"] == 1
    assert s["counts"]["q"] == 1


def test_messages_filter_and_page() -> None:
    m = main.dashboard_messages(sentiment=["pos"], topic=None, q="ราคา", limit=5, offset=0)
    assert m["total"] > 0
    assert len(m["items"]) <= 5
    assert all("ราคา" in r["texts"] for r in m["items"])
    assert all(r["predicted_sentiment"] == "pos" for r in m["items"])
    m2 = main.dashboard_messages(sentiment=None, topic=None, q="", limit=3, offset=1)
    assert len(m2["items"]) == 3
    assert m2["offset"] == 1


def test_messages_json_serializable_full_range() -> None:
    """กัน regression: NaN (topic_keywords ว่างของ topic -1) ต้องไม่ทำ JSON แตก."""
    m = main.dashboard_messages(sentiment=None, topic=None, q="", limit=1000, offset=0)
    json.dumps(m)  # ต้องไม่ raise ValueError: Out of range float
    s = main.dashboard_summary(sentiment=None, topic=None, q="")
    json.dumps(s)
