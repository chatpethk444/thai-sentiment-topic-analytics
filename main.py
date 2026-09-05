"""Phase 3a: FastAPI Backend.

Endpoints:
    GET  /health            -> สถานะ + โมเดลที่โหลดได้
    POST /analyze {text}    -> {sentiment, score, topic, keywords, cleaned_text}
    GET  /topics            -> map topic_id -> keywords (ถ้ามี artifacts)
    GET  /dashboard/summary -> metric + distribution ราย topic (ให้ Next.js/ECharts)
    GET  /dashboard/messages-> ตารางข้อความแบบ filter + paginate (ให้ Next.js/ECharts)

Constraint: text ยาว < 2 ตัวอักษร -> sentiment neu 0.0, ข้าม topic (topic_id=-1).

Run:
    uvicorn main:app --reload --port 8000
    curl -X POST localhost:8000/analyze -H "Content-Type: application/json" -d "{\"text\": \"อาหารอร่อยมาก ประทับใจ\"}"
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Wisesight Thai Sentiment & Topic API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="ข้อความภาษาไทย")


class AnalyzeResponse(BaseModel):
    text: str
    cleaned_text: str
    tokens: List[str]
    predicted_sentiment: str
    sentiment_pred: str
    sentiment_score: float
    topic_id: int
    topic_keywords: str
    topic_keywords_list: List[str] = []


@lru_cache(maxsize=1)
def _get_resources():
    """โหลด pipeline helpers + engines ครั้งเดียว (lazy import หนัก)."""
    from model_engine import SentimentEngine, TopicEngine  # noqa: E402
    from pipeline import clean_raw_text, get_stopwords, tokenize_thai  # noqa: E402

    sentiment = SentimentEngine()
    sentiment.load()  # ล้มเหลว -> fallback rule-based อัตโนมัติใน .predict
    # Topic: พยายามโหลด keywords สำเร็จรูป (outputs/topic_keywords.json) เพื่อ inference เบา
    topic = TopicEngine()
    kw_path = BASE_DIR / "outputs" / "topic_keywords.json"
    if kw_path.exists():
        import json

        raw = json.loads(kw_path.read_text(encoding="utf-8"))
        topic.topic_keywords_ = {int(k): v for k, v in raw.items()}
        # ไม่มีเวกเตอร์โมเดล -> โหมด keyword-lookup ผ่าน transform fallback จะคืน -1;
        # เก็บ map ไว้ตอบ /topics
    return {
        "sentiment": sentiment,
        "topic": topic,
        "clean": clean_raw_text,
        "tokenize": tokenize_thai,
        "stopwords": get_stopwords(),
    }


@app.get("/health")
def health():
    r = _get_resources()
    se = r["sentiment"]
    return {
        "status": "ok",
        "sentiment_model": se.mode if se.ready else f"fallback rule-based ({se._load_error})",
        "has_q_label": se._has_q,
        "policy": "thai-only, len<2 -> neu/0.0 skip-topic; 3-label model -> hybrid rule-q",
    }


@app.get("/topics")
def list_topics() -> Dict[str, list]:
    r = _get_resources()
    return {str(k): v for k, v in r["topic"].topic_keywords_.items()}


@lru_cache(maxsize=1)
def _get_dashboard_df():
    """DataFrame สำหรับ dashboard endpoints. ต้องมี artifact ก่อน (เร็ว, ไม่รันโมเดล)."""
    import pandas as pd

    for name in ("enriched.parquet", "enriched.csv"):
        p = BASE_DIR / "outputs" / name
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            if "sentiment_pred" not in df.columns and "predicted_sentiment" in df.columns:
                df["sentiment_pred"] = df["predicted_sentiment"]
            return df, name
    raise HTTPException(
        status_code=404,
        detail="No dashboard artifact. Run: python model_engine.py --limit 2000 --out outputs/enriched.csv",
    )


def _apply_filters(df, sentiments: Optional[List[str]], topics: Optional[List[int]], q: str):
    f = df
    if sentiments:
        f = f[f["predicted_sentiment"].isin(sentiments)]
    if topics is not None:
        f = f[f["topic_id"].isin(topics)]
    if q.strip():
        f = f[f["texts"].astype(str).str.contains(q.strip(), na=False)]
    return f


def _json_safe_records(page: "pd.DataFrame") -> List[dict]:
    """ล้าง NaN/inf (เช่น topic_keywords ว่างของ topic -1) ก่อนส่ง JSON."""
    import math

    import pandas as pd

    page = page.copy()
    if "sentiment_score" in page.columns:
        page["sentiment_score"] = pd.to_numeric(page["sentiment_score"], errors="coerce").fillna(0.0)
    if "topic_id" in page.columns:
        page["topic_id"] = pd.to_numeric(page["topic_id"], errors="coerce").fillna(-1).astype(int)
    for c in ("texts", "category", "predicted_sentiment", "topic_keywords"):
        if c in page.columns:
            page[c] = page[c].fillna("")
    records = page.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and not math.isfinite(v):
                r[k] = 0.0 if k == "sentiment_score" else None
    return records


@app.get("/dashboard/summary")
def dashboard_summary(
    sentiment: Optional[List[str]] = Query(default=None, description="filter pos/neg/neu/q (repeatable)"),
    topic: Optional[List[int]] = Query(default=None, description="filter topic_id (repeatable)"),
    q: str = Query(default="", description="substring search ใน texts"),
):
    df, source = _get_dashboard_df()
    f = _apply_filters(df, sentiment, topic, q)
    total = int(len(f))
    counts = {s: int((f["predicted_sentiment"] == s).sum()) for s in ("pos", "neu", "neg", "q")}
    overall = {s: int((df["predicted_sentiment"] == s).sum()) for s in ("pos", "neu", "neg", "q")}
    dist = (
        f.groupby(["topic_id", "predicted_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for s in ("pos", "neu", "neg", "q"):
        if s not in dist.columns:
            dist[s] = 0
    per_topic = [
        {"topic_id": int(r["topic_id"]), "total": int(r["pos"] + r["neu"] + r["neg"] + r["q"]),
         "pos": int(r["pos"]), "neu": int(r["neu"]), "neg": int(r["neg"]), "q": int(r["q"])}
        for r in dist.sort_values("topic_id").to_dict(orient="records")
    ]
    kw = (
        f[["topic_id", "topic_keywords"]]
        .drop_duplicates()
        .sort_values("topic_id")
        .copy()
    )
    kw["topic_keywords"] = kw["topic_keywords"].fillna("")
    kw_records = kw.to_dict(orient="records")
    return {
        "source": f"artifact:{source}",
        "total": total,
        "loaded": int(len(df)),
        "counts": counts,
        "overall_counts": overall,
        "pct": {s: (counts[s] / total * 100 if total else 0.0) for s in counts},
        "per_topic": per_topic,
        "topic_keywords": [{"topic_id": int(r["topic_id"]), "keywords": str(r["topic_keywords"])} for r in kw_records],
        "sentiments": sorted(df["predicted_sentiment"].dropna().unique().tolist()),
        "topics": sorted(int(t) for t in df["topic_id"].dropna().unique().tolist()),
    }


@app.get("/dashboard/messages")
def dashboard_messages(
    sentiment: Optional[List[str]] = Query(default=None),
    topic: Optional[List[int]] = Query(default=None),
    q: str = Query(default=""),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    df, _ = _get_dashboard_df()
    f = _apply_filters(df, sentiment, topic, q)
    cols = [c for c in ["texts", "category", "predicted_sentiment", "sentiment_score", "topic_id", "topic_keywords"] if c in f.columns]
    page = f[cols].iloc[offset : offset + limit]
    return {"total": int(len(f)), "limit": limit, "offset": offset, "items": _json_safe_records(page)}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    r = _get_resources()
    text = (req.text or "").strip()

    # Strict constraint
    if len(text) < 2:
        return AnalyzeResponse(
            text=req.text,
            cleaned_text="",
            tokens=[],
            predicted_sentiment="neu",
            sentiment_pred="neu",
            sentiment_score=0.0,
            topic_id=-1,
            topic_keywords="",
            topic_keywords_list=[],
        )

    cleaned = r["clean"](text)
    tokens = r["tokenize"](cleaned, stopwords=r["stopwords"])
    joined = " ".join(tokens)
    label, score = r["sentiment"].predict(text)
    tids = r["topic"].transform([joined or cleaned])
    tid = int(tids[0]) if tids else -1
    kws: List[str] = r["topic"].keywords_for(tid) if tid != -1 else []
    return AnalyzeResponse(
        text=req.text,
        cleaned_text=joined,
        tokens=tokens,
        predicted_sentiment=label,
        sentiment_pred=label,
        sentiment_score=float(score),
        topic_id=tid,
        topic_keywords=", ".join(kws),
        topic_keywords_list=kws,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
