"""Phase 2: ML/NLP Engine — Sentiment (WangchanBERTa) + Topic (BERTopic).

Output DataFrame columns:
    [texts, category, predicted_sentiment, sentiment_score, topic_id, topic_keywords]
+ aliases: sentiment_pred (=predicted_sentiment), clean_text/cleaned_text, tokens

Constraints (SKILL.md):
- ข้อความยาว < 2 ตัวอักษร -> sentiment=(neu, 0.0), ข้าม topic (topic_id=-1, keywords=[])
- รองรับเฉพาะภาษาไทย (ฟังก์ชัน contains_thai ไว้กรอง/log)

Design:
- lazy import torch/transformers/bertopic -> ถ้าไม่มี ใช้ fallback (keyword rule / TF-IDF+KMeans)
  เพื่อให้ main.py / app.py import ได้เสมอ และรัน demo ได้โดยไม่ต้องมี GPU
- SentimentEngine โหลดตาม SENTIMENT_CANDIDATES (verified):
    1. airesearch/wangchanberta-base-att-spm-uncased @ finetuned@wisesight_sentiment (official 4-class)
    2. poom-sci/WangchanBERTa-finetuned-sentiment (3-class + hybrid rule ดัก q)
    3. phoner45/wangchan-sentiment-thai-text-model (3-class + hybrid rule ดัก q)
    4. base ตาม spec (ไม่มี head — ตัวเลือกสุดท้าย)
  map label -> {pos, neg, neu, q}

Run:
    python model_engine.py --zip data/wisesight-sentiment-1.1.zip --limit 2000 --out outputs/enriched.csv
    python model_engine.py --model poom-sci/WangchanBERTa-finetuned-sentiment --limit 500
    python model_engine.py --model airesearch/wangchanberta-base-att-spm-uncased --revision finetuned@wisesight_sentiment
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

VALID_LABELS = ("pos", "neg", "neu", "q")
THAI_RE = re.compile(r"[ก-๙]")

# HF candidates (verified 2026-09): official 4-class ก่อน, community 3-class ตาม, base ท้ายสุด
# revision รองรับ branch เช่น finetuned@wisesight_sentiment ของ airesearch
SENTIMENT_CANDIDATES: List[Dict[str, Optional[str]]] = [
    # Official: 4 class pos/neg/neu/q ตรงสเปก Wisesight (PyThaiNLP tutorial)
    {"model": "airesearch/wangchanberta-base-att-spm-uncased", "revision": "finetuned@wisesight_sentiment"},
    # Community: 3 class (pos/neu/neg, ไม่มี q) — ใช้ hybrid rule ดัก q ก่อนยิงโมเดล
    {"model": "poom-sci/WangchanBERTa-finetuned-sentiment", "revision": None},
    {"model": "phoner45/wangchan-sentiment-thai-text-model", "revision": None},
    # Base ตาม spec: ไม่มี head (MISSING) — ตัวเลือกสุดท้าย, score มั่ว, ต้อง fine-tune
    {"model": "airesearch/wangchanberta-base-att-spm-uncased", "revision": None},
]

POS_KEYWORDS = ["ดี", "ชอบ", "รัก", "เยี่ยม", "ยอด", "ประทับใจ", "คุ้ม", "อร่อย", "สวย", "ขอบคุณ", "ดีมาก", "เลิศ", "ปัง", "แนะนำ"]
NEG_KEYWORDS = ["แย่", "ห่วย", "ผิดหวัง", "ไม่ดี", "ไม่ชอบ", "เกลียด", "แพง", "ช้า", "พัง", "เสียใจ", "แย่มาก", "กาก", "หงุดหงิด", "โกรธ"]
Q_KEYWORDS = ["ไหม", "หรือ", "อะไร", "ยังไง", "อย่างไร", "ที่ไหน", "เมื่อไหร่", "?", "ช่วยแนะนำ", "ขอถาม", "สงสัย"]

# Rule-based topic ชั่วคราว (ไม่มี BERTopic/sklearn): จับ keyword โดเมน -> topic_id คงที่
TOPIC_RULES: Dict[int, List[str]] = {
    0: ["กิน", "อร่อย", "อาหาร", "ร้าน", "เมนู", "รสชาติ"],
    1: ["บริการ", "พนักงาน", "รอ", "ช้า", "ดูแล", "ประทับใจ"],
    2: ["ราคา", "แพง", "ถูก", "คุ้ม", "ลด", "โปร", "บาท"],
    3: ["รถ", "ขับ", "น้ำมัน", "เครื่อง", "ศูนย์", "ซ่อม"],
    4: ["ถาม", "ไหม", "หรือ", "ยังไง", "ที่ไหน", "แนะนำ", "?"],
}
TOPIC_RULE_DEFAULT = 5  # เบ็ดเตล็ด (มี keyword ไม่เข้า rules ใดเลยแต่ยาวพอ)
TOPIC_RULE_LABELS: Dict[int, List[str]] = {
    **TOPIC_RULES,
    TOPIC_RULE_DEFAULT: ["ทั่วไป", "เบ็ดเตล็ด"],
}


def contains_thai(text: str) -> bool:
    return bool(THAI_RE.search(text or ""))


def map_label(raw_label: str) -> str:
    """Map HF label string -> pos/neg/neu/q."""
    s = (raw_label or "").strip().lower()
    if s in VALID_LABELS:
        return s
    if "positive" in s or "5 star" in s or "4 star" in s or s in ("label_2", "label 2", "2"):
        return "pos"
    if "negative" in s or "1 star" in s or "2 star" in s or s in ("label_0", "label 0", "0"):
        return "neg"
    if "neutral" in s or "3 star" in s or s in ("label_1", "label 1", "1"):
        return "neu"
    if "question" in s or s in ("label_3", "label 3", "3"):
        return "q"
    # WangchanBERTa fine-tuned บางเวอร์ชันใช้ LABEL_0..3 = neg/neu/pos/q ตามลำดับ train wisesight
    # (ถ้าไม่แน่ใจ ให้คง mapping กลาง: 0=neg, 1=neu, 2=pos, 3=q)
    return "neu"


def rule_based_sentiment(text: str) -> Tuple[str, float]:
    """Fallback เมื่อไม่มี transformers/torch/GPU: นับ keyword แบบง่าย."""
    t = text or ""
    if len(t.strip()) < 2:
        return "neu", 0.0
    pos = sum(1 for k in POS_KEYWORDS if k in t)
    neg = sum(1 for k in NEG_KEYWORDS if k in t)
    q = sum(1 for k in Q_KEYWORDS if k in t)
    if "?" in t:
        q += 1
    if q > 0 and pos == 0 and neg == 0:
        return "q", 0.6
    if pos > neg and pos > 0:
        return "pos", min(0.55 + 0.1 * pos, 0.9)
    if neg > pos and neg > 0:
        return "neg", min(0.55 + 0.1 * neg, 0.9)
    return "neu", 0.5


def is_question_like(text: str) -> bool:
    """Hybrid gate สำหรับโมเดล 3-label (ไม่มี q): คำถามล้วน ไม่มีขั้ว pos/neg."""
    t = text or ""
    if len(t.strip()) < 2:
        return False
    pos = sum(1 for k in POS_KEYWORDS if k in t)
    neg = sum(1 for k in NEG_KEYWORDS if k in t)
    if pos > 0 or neg > 0:
        return False
    return any(k in t for k in Q_KEYWORDS) or "?" in t


def rule_based_topic(docs: List[str], top_n: int = 5) -> Tuple[List[int], Dict[int, List[str]]]:
    """Fallback สุดท้ายเมื่อไม่มี BERTopic/sklearn: แมตช์ TOPIC_RULES ตรงๆ ไม่ต้องพึ่ง lib ใด.

    - doc < 2 chars -> -1 (ข้ามตาม SKILL constraint)
    - นับ hits ต่อ rule, เลือก rule คะแนนสูงสุด, เสมอ -> id น้อยสุด
    - ไม่เจอ rule ใดแต่ยาวพอ -> TOPIC_RULE_DEFAULT
    """
    ids: List[int] = []
    for d in docs:
        t = d or ""
        if len(t.strip()) < 2:
            ids.append(-1)
            continue
        best_id, best_hit = TOPIC_RULE_DEFAULT, 0
        for tid, kws in TOPIC_RULES.items():
            hit = sum(1 for k in kws if k in t)
            if hit > best_hit:
                best_id, best_hit = tid, hit
        ids.append(best_id)
    kw_map = {tid: kws[:top_n] for tid, kws in TOPIC_RULE_LABELS.items()}
    return ids, kw_map


def resolve_device(prefer: Optional[int] = None) -> int:
    """GPU-safe: คืน device id ให้ transformers pipeline. -1 = CPU/fallback."""
    if prefer is not None:
        return prefer
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


class SentimentEngine:
    """WangchanBERTa inference + fallback. ใช้ .predict(text) -> (label, score)."""

    # Threshold: pos ต้องมั่นใจเกินนี้ ไม่งั้นตกเป็น neu (ลด false-pos จาก pos คะแนนต่ำ)
    POS_MIN_SCORE: float = 0.6

    def __init__(self, model_name: Optional[str] = None, device: Optional[int] = None, revision: Optional[str] = None, pos_threshold: Optional[float] = None):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.pos_threshold = self.POS_MIN_SCORE if pos_threshold is None else float(pos_threshold)
        self.pipe = None
        self.loaded_model: Optional[str] = None
        self.loaded_revision: Optional[str] = None
        self._has_q: bool = True  # False เมื่อโมเดลมีแค่ 3 label -> เปิด hybrid ดัก q
        self._load_error: Optional[str] = None

    def _detect_labels(self) -> None:
        """ตรวจ config ว่าโมเดลมี q หรือไม่. พลาด -> คง True (ไม่ดัก)."""
        try:
            labels = self.pipe.model.config.id2label or {}
            vals = {str(v).strip().lower() for v in labels.values()}
            self._has_q = ("q" in vals) or ("question" in " ".join(vals)) or len(vals) >= 4
        except Exception:  # noqa: BLE001
            self._has_q = True

    def _build_kwargs(self, model: str, revision: Optional[str], device: int) -> dict:
        kwargs: dict = {"task": "sentiment-analysis", "model": model, "tokenizer": model, "truncation": True, "device": device}
        if revision:
            kwargs["revision"] = revision
        return kwargs

    def _import_pipeline(self):
        """Import transformers pipeline หลายทาง + traceback เต็มเมื่อพัง."""
        try:
            from transformers import pipeline

            return pipeline
        except ImportError:
            pass
        try:
            from transformers.pipelines import pipeline

            return pipeline
        except ImportError:
            pass
        try:
            import transformers

            fn = getattr(transformers, "pipeline", None)
            if callable(fn):
                return fn
        except Exception:  # noqa: BLE001
            pass
        import traceback

        raise ImportError(f"transformers pipeline unavailable:\n{traceback.format_exc(limit=5)}")

    def load(self) -> bool:
        try:
            pipeline = self._import_pipeline()
        except ImportError as e:
            self._load_error = f"transformers missing: {e}"
            return False
        if self.model_name:
            candidates = [{"model": self.model_name, "revision": self.revision}]
        else:
            candidates = list(SENTIMENT_CANDIDATES)
        last_err = ""
        device = resolve_device(self.device)
        for cand in candidates:
            name, rev = cand["model"], cand.get("revision")
            try:
                kwargs = self._build_kwargs(name, rev, device)
                self.pipe = pipeline(**kwargs)
                self.loaded_model = name
                self.loaded_revision = rev
                self._detect_labels()
                return True
            except Exception as e:  # noqa: BLE001 — ลอง candidate ถัดไป (เช่น ไม่มี GPU/เน็ต/model ไม่มี head)
                last_err = f"{name}@{rev}: {e}" if rev else f"{name}: {e}"
                # ลองซ้ำบน CPU หนึ่งครั้งถ้า GPU พัง
                if device != -1:
                    try:
                        kwargs["device"] = -1
                        self.pipe = pipeline(**kwargs)
                        self.loaded_model = name
                        self.loaded_revision = rev
                        self._detect_labels()
                        return True
                    except Exception as e2:  # noqa: BLE001
                        last_err = f"{name}@{rev} (cpu-retry): {e2}" if rev else f"{name} (cpu-retry): {e2}"
                continue
        self._load_error = last_err
        return False

    @property
    def ready(self) -> bool:
        return self.pipe is not None

    @property
    def is_fallback(self) -> bool:
        return self.pipe is None

    @property
    def mode(self) -> str:
        if self.pipe is None:
            return "rule-based"
        base = self.loaded_model or "model"
        if self.loaded_revision:
            base += f"@{self.loaded_revision}"
        if not self._has_q:
            base += " (3-label+rule-q)"
        return base

    def predict(self, text: str) -> Tuple[str, float]:
        # Strict constraint: < 2 chars -> neu 0.0
        if not isinstance(text, str) or len(text.strip()) < 2:
            return "neu", 0.0
        if self.pipe is None and not self.load():
            label, score = rule_based_sentiment(text)
            return self._apply_threshold(label, score)
        # Hybrid: โมเดล 3-label ไม่มี q -> ดักคำถามล้วนก่อนยิงโมเดล
        if not self._has_q and is_question_like(text):
            return "q", 0.65
        try:
            out = self.pipe(text[:512])[0]
            return self._apply_threshold(map_label(str(out.get("label", ""))), float(out.get("score", 0.0)))
        except Exception:  # noqa: BLE001 — inference พลาด -> fallback
            label, score = rule_based_sentiment(text)
            return self._apply_threshold(label, score)

    def _apply_threshold(self, label: str, score: float) -> Tuple[str, float]:
        """pos คะแนนต่ำกว่า pos_threshold -> ตกเป็น neu (คง score เดิมไว้ดู)."""
        if label == "pos" and float(score) < self.pos_threshold:
            return "neu", float(score)
        return label, float(score)

    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Tuple[str, float]]:
        if self.pipe is None and not self.load():
            return [rule_based_sentiment(t) for t in texts]
        results: List[Tuple[str, float]] = []
        try:
            from transformers.pipelines.pt_utils import KeyDataset  # lazy, optional
            del KeyDataset
        except Exception:
            pass
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for t in chunk:
                results.append(self.predict(t))
        return results


def thai_tokenizer_for_bertopic(stopwords: Optional[set] = None):
    """Tokenizer callable สำหรับ CountVectorizer <- PyThaiNLP newmm."""
    try:
        from pythainlp import word_tokenize
        from pythainlp.corpus import thai_stopwords
    except ImportError:
        return None
    sw = stopwords if stopwords is not None else set(thai_stopwords())

    def _tok(doc: str) -> List[str]:
        toks = word_tokenize(doc or "", engine="newmm", keep_whitespace=False)
        return [t.strip() for t in toks if t.strip() and t.strip() not in sw and len(t.strip()) > 1]

    return _tok


class TopicEngine:
    """BERTopic หลัก -> TF-IDF/KMeans -> rule-based keyword. .fit_transform(docs) -> (ids, keywords_map)."""

    def __init__(self, min_topic_size: int = 20, top_n_words: int = 5, embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.min_topic_size = min_topic_size
        self.top_n_words = top_n_words
        self.embedding_model = embedding_model
        self.model = None  # BERTopic instance
        self.fallback = None  # dict with kmeans/vectorizer
        self.mode: str = "unfitted"  # bertopic | sklearn-fallback | rule-based
        self.topic_keywords_: Dict[int, List[str]] = {}

    def fit_transform(self, docs: List[str]) -> Tuple[List[int], Dict[int, List[str]]]:
        docs = [(d or "") for d in docs]
        # ทิ้ง doc ว่าง (<2 chars) ออกจาก training แต่คืน id=-1 ให้ภายหลัง
        train_idx = [i for i, d in enumerate(docs) if len(d.strip()) >= 2]
        train_docs = [docs[i] for i in train_idx]
        if not train_docs:
            self.mode = "rule-based"
            return [-1] * len(docs), {}

        if self._fit_bertopic(train_docs):
            from numpy import ndarray  # lazy

            topics, _ = self.model.transform(train_docs)
            if isinstance(topics, ndarray):
                topics = topics.tolist()
            self.topic_keywords_ = self._bertopic_keywords()
            self.mode = "bertopic"
        else:
            try:
                topics = self._fit_sklearn(train_docs)
                self.mode = "sklearn-fallback"
            except Exception:  # noqa: BLE001 — ไม่มี sklearn/RAM ไม่พอ -> rule-based สุดท้าย
                rule_ids, self.topic_keywords_ = rule_based_topic(train_docs, top_n=self.top_n_words)
                topics = rule_ids
                self.mode = "rule-based"

        full_ids = [-1] * len(docs)
        for pos, tidx in enumerate(train_idx):
            full_ids[tidx] = int(topics[pos])
        return full_ids, self.topic_keywords_

    def transform(self, docs: List[str]) -> List[int]:
        docs = [(d or "") for d in docs]
        out = [-1] * len(docs)
        valid = [(i, d) for i, d in enumerate(docs) if len(d.strip()) >= 2]
        if not valid:
            return out
        if self.mode == "bertopic" and self.model is not None:
            try:
                idx = [i for i, _ in valid]
                dd = [d for _, d in valid]
                topics, _ = self.model.transform(dd)
                for i, t in zip(idx, topics):
                    out[i] = int(t)
                return out
            except Exception:  # noqa: BLE001 — ตกไป fallback predict
                pass
        if self.mode == "sklearn-fallback" and self.fallback is not None:
            try:
                vec = self.fallback["vectorizer"].transform([d for _, d in valid])
                pred = self.fallback["kmeans"].predict(vec)
                for (i, _), t in zip(valid, pred):
                    out[i] = int(t)
                return out
            except Exception:  # noqa: BLE001 — ตกไป rule-based
                pass
        rule_ids, _ = rule_based_topic([d for _, d in valid], top_n=self.top_n_words)
        for (i, _), t in zip(valid, rule_ids):
            out[i] = int(t)
        return out

    def keywords_for(self, topic_id: int) -> List[str]:
        return self.topic_keywords_.get(int(topic_id), [])

    # -- internal --
    def _fit_bertopic(self, train_docs: List[str]) -> bool:
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
        except ImportError:
            return False
        try:
            tok = thai_tokenizer_for_bertopic()
            vectorizer = CountVectorizer(tokenizer=tok, token_pattern=None, max_features=5000) if tok else CountVectorizer(max_features=5000)
            # embedding: multilingual รองรับไทย; ถ้าโหลดไม่ได้ BERTopic จะ fallback ภายใน -> ให้ exception แล้วไป sklearn
            self.model = BERTopic(
                embedding_model=self.embedding_model,
                vectorizer_model=vectorizer,
                min_topic_size=min(self.min_topic_size, max(2, len(train_docs) // 10)),
                top_n_words=self.top_n_words,
                verbose=False,
            )
            self.model.fit(train_docs)
            return True
        except Exception:  # noqa: BLE001
            self.model = None
            return False

    def _bertopic_keywords(self) -> Dict[int, List[str]]:
        kw: Dict[int, List[str]] = {}
        try:
            for tid in self.model.get_topic_info()["Topic"].tolist():
                if tid == -1:
                    continue
                words = [w for w, _ in (self.model.get_topic(int(tid)) or [])[: self.top_n_words]]
                kw[int(tid)] = words
        except Exception:  # noqa: BLE001
            pass
        return kw

    def _fit_sklearn(self, train_docs: List[str]) -> List[int]:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        tok = thai_tokenizer_for_bertopic()
        vec = TfidfVectorizer(tokenizer=tok, token_pattern=None, max_features=3000) if tok else TfidfVectorizer(max_features=3000)
        X = vec.fit_transform(train_docs)
        n = min(max(2, len(train_docs) // 200), 10)
        km = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = km.fit_predict(X).tolist()
        # top-5 keywords ต่อ cluster จาก centroid
        terms = vec.get_feature_names_out()
        order = km.cluster_centers_.argsort()[:, ::-1]
        self.topic_keywords_ = {
            c: [terms[i] for i in order[c, : self.top_n_words]] for c in range(n)
        }
        self.fallback = {"vectorizer": vec, "kmeans": km}
        return [int(x) for x in labels]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.mode == "bertopic" and self.model is not None:
            self.model.save(str(path / "bertopic_model"))
        import json

        (path / "topic_keywords.json").write_text(
            __import__("json").dumps({str(k): v for k, v in self.topic_keywords_.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def enrich_dataframe(
    df: "pd.DataFrame",
    sentiment: Optional[SentimentEngine] = None,
    topic: Optional[TopicEngine] = None,
    text_col: str = "texts",
    clean_col: str = "cleaned_text",
) -> "pd.DataFrame":
    """เพิ่ม predicted_sentiment, sentiment_score, topic_id, topic_keywords (+aliases)."""
    import pandas as pd  # noqa: F401

    sentiment = sentiment or SentimentEngine()
    topic = topic or TopicEngine()

    texts = df[text_col].astype(str).tolist()
    cleans = df[clean_col].astype(str).tolist() if clean_col in df.columns else texts

    preds = sentiment.predict_batch(texts)
    topic_ids, kw_map = topic.fit_transform(cleans)

    out = df.copy()
    out["predicted_sentiment"] = [p[0] for p in preds]
    out["sentiment_pred"] = out["predicted_sentiment"]  # alias SKILL.md
    out["sentiment_score"] = [float(p[1]) for p in preds]
    out["topic_id"] = [int(t) for t in topic_ids]
    out["topic_keywords"] = [", ".join(kw_map.get(int(t), [])) if int(t) != -1 else "" for t in topic_ids]
    out["topic_keywords_list"] = [kw_map.get(int(t), []) if int(t) != -1 else [] for t in topic_ids]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Wisesight Phase 2 engine")
    ap.add_argument("--zip", dest="zip_path", default="data/wisesight-sentiment-1.1.zip")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--out", default="outputs/enriched.csv")
    ap.add_argument("--model", default=None, help="HF model id (default: autoตาม SENTIMENT_CANDIDATES)")
    ap.add_argument("--revision", default=None, help="HF revision/branch เช่น finetuned@wisesight_sentiment")
    ap.add_argument("--pos-threshold", type=float, default=0.6, help="pos ต้อง score>=นี้ ไม่งั้นตกเป็น neu")
    ap.add_argument("--min-topic-size", type=int, default=20)
    args = ap.parse_args()

    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from pipeline import load_and_preprocess

    print(f"[engine] preprocess limit={args.limit}")
    df, _ = load_and_preprocess(args.zip_path, limit=args.limit)
    print(f"[engine] sentiment model: {args.model or 'auto'} revision={args.revision or 'auto-per-candidate'} pos_thr={args.pos_threshold}")
    se = SentimentEngine(model_name=args.model, revision=args.revision, pos_threshold=args.pos_threshold)
    te = TopicEngine(min_topic_size=args.min_topic_size)
    enriched = enrich_dataframe(df, sentiment=se, topic=te)
    print(f"[engine] sentiment engine: mode={se.mode} has_q={se._has_q} err={se._load_error or '-'}")
    print(f"[engine] topic engine: mode={te.mode} topics={sorted(enriched['topic_id'].unique())[:10]}")
    try:
        print(enriched[["texts", "category", "predicted_sentiment", "sentiment_score", "topic_id", "topic_keywords"]].head().to_string(index=False))
    except UnicodeEncodeError:
        print(enriched[["category", "predicted_sentiment", "sentiment_score", "topic_id"]].head().to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".parquet":
            enriched.to_parquet(out, index=False)
        else:
            enriched.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[engine] saved -> {out}")


if __name__ == "__main__":
    main()
