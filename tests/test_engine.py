"""Engine tests: rule fallbacks, question gate, pos threshold, short-text constraint."""

from model_engine import (
    SentimentEngine,
    is_question_like,
    rule_based_sentiment,
    rule_based_topic,
)


def test_rule_sentiment_buckets() -> None:
    assert rule_based_sentiment("อาหารอร่อยมาก ประทับใจ")[0] == "pos"
    assert rule_based_sentiment("ห่วยมาก ผิดหวัง")[0] == "neg"
    assert rule_based_sentiment("บัตรสมาชิกลดได้อีกไหมคับ")[0] == "q"


def test_rule_short_text_neu_zero() -> None:
    assert rule_based_sentiment("ก") == ("neu", 0.0)
    assert rule_based_sentiment("") == ("neu", 0.0)


def test_question_gate() -> None:
    assert is_question_like("บัตรสมาชิกลดได้อีกไหมคับ") is True
    assert is_question_like("อาหารอร่อยมาก") is False
    # ขั้วบวกเหนือกว่า -> ไม่ใช่คำถามล้วน
    assert is_question_like("อร่อยไหม ชอบมาก") is False


def test_pos_threshold() -> None:
    se = SentimentEngine(pos_threshold=0.6)
    se._has_q = True
    se.pipe = lambda texts: [{"label": "pos", "score": 0.5}]  # type: ignore[assignment]
    assert se.predict("ข้อความทดสอบจ้า") == ("neu", 0.5)
    se.pipe = lambda texts: [{"label": "pos", "score": 0.9}]  # type: ignore[assignment]
    assert se.predict("ข้อความทดสอบจ้า") == ("pos", 0.9)
    se.pipe = lambda texts: [{"label": "neg", "score": 0.4}]  # type: ignore[assignment]
    assert se.predict("ข้อความทดสอบจ้า") == ("neg", 0.4)


def test_hybrid_q_for_3label_model() -> None:
    se = SentimentEngine()
    se._has_q = False
    se.pipe = lambda texts: [{"label": "NEU", "score": 0.9}]  # type: ignore[assignment]
    assert se.predict("บัตรสมาชิกลดได้อีกไหมคับ") == ("q", 0.65)


def test_engine_short_text_constraint() -> None:
    se = SentimentEngine()
    assert se.predict("ก") == ("neu", 0.0)


def test_rule_topic() -> None:
    docs = ["ข้าวผัดอร่อยมาก", "พนักงานบริการช้ามาก", "ราคาแพงเกินไป", "x"]
    ids, kw = rule_based_topic(docs)
    assert ids == [0, 1, 2, -1]
    assert set(kw) >= {0, 1, 2}
