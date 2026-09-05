"""Phase 3b: Streamlit Dashboard (light pastel theme).

Layout ตาม reference: KPI ring x3 บน, Statistics line + Performance radar กลาง,
Activity area + Sentiment bar ล่าง, ตาราง Raw Text ท้าย.

Data source (ลำดับ):
    1. outputs/enriched.csv / .parquet (ถ้ามี — สร้างจาก model_engine.py)
    2. ไม่งั้นรัน pipeline+engine สด (limit แถวผ่าน sidebar เพื่อกันช้า)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
CANDIDATE_ARTIFACTS = [
    BASE_DIR / "outputs" / "enriched.parquet",
    BASE_DIR / "outputs" / "enriched.csv",
    BASE_DIR / "outputs" / "cleaned.parquet",
    BASE_DIR / "outputs" / "cleaned.csv",
]

SENT_COLORS = {"pos": "#7C3AED", "neu": "#14B8A6", "neg": "#60A5FA", "q": "#F87171"}
SENT_ORDER = ["pos", "neu", "neg", "q"]

st.set_page_config(page_title="Wisesight Sentiment & Topic Dashboard", layout="wide")

THEME_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #EDE9FE 0%, #FDF2F8 45%, #E0E7FF 100%);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.75);
}
.kpi {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 8px 24px rgba(124,92,255,0.10);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}
.kpi-title { color: #1E1B4B; font-weight: 700; font-size: 15px; }
.kpi-sub { color: #9CA3AF; font-size: 12px; margin-top: 2px; }
.kpi-num { color: #111827; font-weight: 800; font-size: 26px; margin-top: 4px; }
.kpi-foot { font-size: 12px; margin-top: 4px; color: #6B7280; }
.kpi-foot .up { color: #10B981; font-weight: 700; }
.kpi-foot .down { color: #EF4444; font-weight: 700; }
.page-title { color: #1E1B4B; font-weight: 800; font-size: 26px; }
.page-sub { color: #8B8FA3; font-size: 13px; margin-top: -8px; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(124,92,255,0.10);
    background: #FFFFFF;
}
</style>
"""


def _ring_svg(pct: float, color: str, icon: str) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    arc = pct / 100.0 * 2 * math.pi * 40
    return (
        f'<svg width="92" height="92" viewBox="0 0 96 96">'
        f'<circle cx="48" cy="48" r="40" stroke="#EEF2FF" stroke-width="10" fill="none"/>'
        f'<circle cx="48" cy="48" r="40" stroke="{color}" stroke-width="10" fill="none" '
        f'stroke-linecap="round" stroke-dasharray="{arc:.1f} 251.4" transform="rotate(-90 48 48)"/>'
        f'<text x="48" y="56" text-anchor="middle" font-size="22">{icon}</text></svg>'
    )


def _kpi_card(title: str, sub: str, num: str, foot: str, pct: float, color: str, icon: str) -> None:
    st.markdown(
        f'<div class="kpi"><div><div class="kpi-title">{title}</div>'
        f'<div class="kpi-sub">{sub}</div><div class="kpi-num">{num}</div>'
        f'<div class="kpi-foot">{foot}</div></div>'
        f'<div>{_ring_svg(pct, color, icon)}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading data...")
def load_dataframe(limit: int = 2000) -> tuple[pd.DataFrame, dict]:
    """Fallback 2 ชั้นตาม spec. คืน (df, meta) เพื่อโชว์ badge ใน UI.

    1. ไม่มี outputs/enriched.* -> รัน Pipeline/Engine สดด้วย limit จาก Slider
    2. ไม่มี GPU/Model -> Engine ใช้ rule-based keyword ชั่วคราว (ดู mode ใน meta)
    """
    for p in CANDIDATE_ARTIFACTS:
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            if limit and len(df) > limit:
                df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
            df = _ensure_columns(df)
            return df, {"source": f"artifact:{p.name}", "sentiment": _infer_sent_mode(df), "topic": _infer_topic_mode(df)}
    # fallback 1: รันสดจาก zip ด้วย limit จาก Slider
    from model_engine import SentimentEngine, TopicEngine, enrich_dataframe
    from pipeline import load_and_preprocess

    df, _ = load_and_preprocess(BASE_DIR / "data" / "wisesight-sentiment-1.1.zip", limit=limit)
    se, te = SentimentEngine(), TopicEngine()
    enriched = enrich_dataframe(df, sentiment=se, topic=te)
    enriched = _ensure_columns(enriched)
    return enriched, {"source": f"live-pipeline(limit={limit})", "sentiment": se.mode, "topic": te.mode}


def _infer_sent_mode(df: pd.DataFrame) -> str:
    if "predicted_sentiment" not in df.columns:
        return "rule-based"
    # artifact จาก engine จริงมักมี score กระจาย; rule-based มักมีแค่ {0.5,0.6,0.65..}
    return "artifact"


def _infer_topic_mode(df: pd.DataFrame) -> str:
    if "topic_id" not in df.columns:
        return "rule-based"
    return "artifact"


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """เติมคอลัมน์ที่ขาด + alias ให้ตรงทั้ง spec และ SKILL.md."""
    out = df.copy()
    if "texts" not in out.columns and "text" in out.columns:
        out["texts"] = out["text"]
    if "predicted_sentiment" not in out.columns and "sentiment_pred" in out.columns:
        out["predicted_sentiment"] = out["sentiment_pred"]
    if "sentiment_pred" not in out.columns and "predicted_sentiment" in out.columns:
        out["sentiment_pred"] = out["predicted_sentiment"]
    for col, default in [("category", "neu"), ("predicted_sentiment", "neu"),
                         ("sentiment_score", 0.0), ("topic_id", -1), ("topic_keywords", "")]:
        if col not in out.columns:
            out[col] = default
    # ถ้ายังไม่มี prediction (โหลด cleaned อย่างเดียว) -> fallback 2: rule-based keyword ชั่วคราว
    need_sent = (
        "predicted_sentiment" not in out.columns
        or (out["predicted_sentiment"] == "neu").all() and (out["sentiment_score"] == 0.0).all()
    )
    if need_sent:
        try:
            from model_engine import rule_based_sentiment

            preds = [rule_based_sentiment(str(t)) for t in out["texts"].tolist()]
            out["predicted_sentiment"] = [p[0] for p in preds]
            out["sentiment_pred"] = out["predicted_sentiment"]
            out["sentiment_score"] = [p[1] for p in preds]
        except Exception:
            pass
    # ถ้ายังไม่มี topic (โหลด cleaned อย่างเดียว / ไม่มี sklearn) -> rule-based topic ชั่วคราว
    need_topic = "topic_id" not in out.columns or (out["topic_id"] == -1).all()
    if need_topic and "cleaned_text" in out.columns:
        try:
            from model_engine import rule_based_topic

            tids, kw_map = rule_based_topic(out["cleaned_text"].astype(str).tolist())
            out["topic_id"] = tids
            out["topic_keywords"] = [", ".join(kw_map.get(int(t), [])) if int(t) != -1 else "" for t in tids]
        except Exception:
            pass
    return out


def _base_layout(fig, title: str):
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": "#1E1B4B"}},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#4B5563"},
        margin={"l": 40, "r": 16, "t": 48, "b": 40},
        legend={"orientation": "h", "y": 1.08, "x": 1.0, "xanchor": "right"},
    )
    return fig


def main() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Filters")
        limit = st.slider("Max rows", 200, 20000, 2000, step=200)
        df, meta = load_dataframe(limit)
        if meta["source"].startswith("live-pipeline"):
            st.warning(f"ไม่มี artifact -> รันสด ({meta['source']}). sentiment={meta['sentiment']} topic={meta['topic']}")
        else:
            st.caption(f"source={meta['source']} sentiment={meta['sentiment']} topic={meta['topic']}")

        sentiments = sorted(df["predicted_sentiment"].dropna().unique().tolist())
        topics = sorted(df["topic_id"].dropna().unique().tolist())
        sel_sent = st.multiselect("Sentiment class", sentiments, default=sentiments)
        sel_topic = st.multiselect("Topic ID", topics, default=topics)
        query = st.text_input("Search raw text (substring)")

    f = df[df["predicted_sentiment"].isin(sel_sent) & df["topic_id"].isin(sel_topic)]
    if query.strip():
        f = f[f["texts"].astype(str).str.contains(query.strip(), na=False)]

    total = len(f)
    n_pos = int((f["predicted_sentiment"] == "pos").sum()) if total else 0
    n_neg = int((f["predicted_sentiment"] == "neg").sum()) if total else 0
    n_neu = int((f["predicted_sentiment"] == "neu").sum()) if total else 0
    pct_pos = n_pos / total * 100 if total else 0.0
    pct_neg = n_neg / total * 100 if total else 0.0
    coverage = total / len(df) * 100 if len(df) else 0.0

    st.markdown('<div class="page-title">Dashboard</div><div class="page-sub">Wisesight Thai Sentiment &amp; Topic</div>', unsafe_allow_html=True)

    k1, k2, k3 = st.columns(3)
    with k1:
        _kpi_card("Messages", "Total filtered", f"{total:,}",
                  f'<span class="up">▲ {n_pos} pos</span> &nbsp; <span class="down">▼ {n_neg} neg</span>',
                  coverage, "#60A5FA", "💬")
    with k2:
        _kpi_card("Positive", f"{pct_pos:.1f}% of filtered", f"{n_pos:,}",
                  f'<span class="up">▲ {pct_pos:.1f}% share</span> &nbsp; {n_neu} neu',
                  pct_pos, "#7C3AED", "😊")
    with k3:
        _kpi_card("Negative", f"{pct_neg:.1f}% of filtered", f"{n_neg:,}",
                  f'<span class="down">▼ {pct_neg:.1f}% share</span> &nbsp; coverage {coverage:.0f}%',
                  pct_neg, "#14B8A6", "🛡️")

    if total == 0:
        st.warning("No rows match filter.")
        return

    import plotly.express as px
    import plotly.graph_objects as go

    g = f.groupby(["topic_id", "predicted_sentiment"]).size().reset_index(name="count")
    g = g.sort_values("topic_id")

    mid1, mid2 = st.columns([2, 1])
    with mid1, st.container(border=True):
        order = [s for s in SENT_ORDER if s in g["predicted_sentiment"].unique().tolist()]
        fig = px.line(g, x="topic_id", y="count", color="predicted_sentiment",
                      category_orders={"predicted_sentiment": order},
                      color_discrete_map=SENT_COLORS, markers=True,
                      labels={"topic_id": "Topic ID", "count": "count", "predicted_sentiment": "Sentiment"})
        fig.update_traces(line_shape="spline", line_width=2.5, marker_size=7)
        st.plotly_chart(_base_layout(fig, "Statistics — sentiment flow across topics"), use_container_width=True)
    with mid2, st.container(border=True):
        cats = [s for s in SENT_ORDER if s in set(sel_sent)]
        filt_share = [(f["predicted_sentiment"] == s).mean() * 100 for s in cats]
        all_share = [(df["predicted_sentiment"] == s).mean() * 100 for s in cats]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=filt_share + filt_share[:1], theta=cats + cats[:1],
                                      fill="toself", name="Filtered",
                                      line_color="#7C3AED", fillcolor="rgba(124,58,237,0.35)"))
        fig.add_trace(go.Scatterpolar(r=all_share + all_share[:1], theta=cats + cats[:1],
                                      fill="toself", name="Overall",
                                      line_color="#14B8A6", fillcolor="rgba(20,184,166,0.25)"))
        fig.update_layout(polar={"bgcolor": "white", "radialaxis": {"visible": True, "range": [0, max(10, max(filt_share + all_share + [10]))]}})
        st.plotly_chart(_base_layout(fig, "Performance — filtered vs overall (%)"), use_container_width=True)

    bot1, bot2 = st.columns(2)
    with bot1, st.container(border=True):
        vol = f["topic_id"].value_counts().sort_index()
        fig = px.area(x=vol.index.astype(str), y=vol.values,
                      labels={"x": "Topic ID", "y": "messages"})
        fig.update_traces(line_color="#3B82F6", fillcolor="rgba(147,197,253,0.45)", line_width=2.5)
        st.plotly_chart(_base_layout(fig, "Activity — volume per topic"), use_container_width=True)
    with bot2, st.container(border=True):
        fig = px.bar(g, x="topic_id", y="count", color="predicted_sentiment",
                     category_orders={"predicted_sentiment": order},
                     color_discrete_map=SENT_COLORS, barmode="group",
                     labels={"topic_id": "Topic ID", "predicted_sentiment": "Sentiment"})
        st.plotly_chart(_base_layout(fig, "Sentiment count by Topic"), use_container_width=True)

    with st.container(border=True):
        st.markdown("**ตัวอย่างข้อความ (Raw Text)**")
        show_cols = [c for c in ["texts", "category", "predicted_sentiment", "sentiment_score", "topic_id", "topic_keywords"] if c in f.columns]
        st.dataframe(f[show_cols].head(100), use_container_width=True, height=360)

    with st.expander("Topic keywords"):
        kw = f[["topic_id", "topic_keywords"]].drop_duplicates().sort_values("topic_id")
        st.table(kw)


if __name__ == "__main__":
    main()
