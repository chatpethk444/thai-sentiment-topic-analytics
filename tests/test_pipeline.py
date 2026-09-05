"""Preprocessing tests: URL/mention stripping, newmm tokenization, short-text rule."""

from pipeline import clean_raw_text, load_wisesight_zip, preprocess_dataframe


def test_clean_removes_url_and_mention() -> None:
    out = clean_raw_text("โปรดีมาก https://example.com/x @shop แนะนำเลย!")
    assert "http" not in out
    assert "@shop" not in out
    assert "โปรดีมาก" in out


def test_clean_keeps_thai_english_digits() -> None:
    out = clean_raw_text("Mazda2 ราคา 250000 😍")
    assert "Mazda2" in out
    assert "250000" in out


def test_preprocess_adds_columns_and_tokens() -> None:
    import pandas as pd

    df = pd.DataFrame([{"texts": "อาหารอร่อยมาก ประทับใจ", "category": "pos"}])
    out = preprocess_dataframe(df)
    assert {"cleaned_text", "clean_text", "tokens"} <= set(out.columns)
    assert len(out.loc[0, "tokens"]) > 0
    assert "อร่อย" in out.loc[0, "cleaned_text"]


def test_short_text_skips_topic_input() -> None:
    import pandas as pd

    df = pd.DataFrame([{"texts": "ก", "category": "neu"}])
    out = preprocess_dataframe(df)
    assert out.loc[0, "cleaned_text"] == ""
    assert out.loc[0, "tokens"] == []


def test_zip_loads_without_extracting() -> None:
    df = load_wisesight_zip(limit=5)
    assert {"texts", "category"} <= set(df.columns)
    assert len(df) == 5
