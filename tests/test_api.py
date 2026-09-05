"""API tests: direct calls (no httpx/TestClient needed)."""

from main import AnalyzeRequest, analyze, health


def test_health_ok() -> None:
    body = health()
    assert body["status"] == "ok"
    assert "sentiment_model" in body


def test_analyze_positive_thai() -> None:
    res = analyze(AnalyzeRequest(text="อาหารอร่อยมาก ประทับใจสุดๆ"))
    assert res.predicted_sentiment == "pos"
    assert res.sentiment_score > 0.0
    assert "อร่อย" in res.cleaned_text


def test_analyze_short_text_constraint() -> None:
    res = analyze(AnalyzeRequest(text="ก"))
    assert res.predicted_sentiment == "neu"
    assert res.sentiment_score == 0.0
    assert res.topic_id == -1
    assert res.tokens == []
