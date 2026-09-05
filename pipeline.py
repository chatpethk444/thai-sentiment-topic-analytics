"""Phase 1: Data Ingestion & Preprocessing (PyThaiNLP).

- อ่าน train.jsonl จาก wisesight-sentiment-1.1.zip โดยไม่แตกไฟล์ลงดิสก์ (zipfile + memory buffer)
- รองรับ nested zip (huggingface/data.zip/data/train.jsonl) + fallback (pos/neg/neu/q.txt, kaggle train.txt)
- ทำความสะอาด: ลบ URL, Mention, สัญลักษณ์พิเศษ -> ตัดคำ newmm -> ลบ Thai stopwords
- Output columns: texts, category, cleaned_text, clean_text (alias), tokens

Run:
    python pipeline.py --zip data/wisesight-sentiment-1.1.zip --head 5
    python pipeline.py --zip data/wisesight-sentiment-1.1.zip --out outputs/cleaned.parquet --limit 5000
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
# เก็บ ก-๙, a-z, A-Z, 0-9, whitespace; อย่างอื่น -> space
KEEP_RE = re.compile(r"[^ก-๙a-zA-Z0-9\s]+", re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")


def _read_nested_jsonl(outer_zip: zipfile.ZipFile, target: str = "train.jsonl") -> Optional[List[dict]]:
    """ค้นหา target (.jsonl) ทั้ง direct + nested .zip (memory only). คืน list[dict] หรือ None."""
    # 1) direct match (suffix match เพื่อรองรับ path prefix)
    for name in outer_zip.namelist():
        if name.endswith(target) and not name.startswith("__MACOSX"):
            with outer_zip.open(name) as f:
                return _parse_jsonl_bytes(f.read())
    # 2) nested .zip (เช่น huggingface/data.zip)
    for name in outer_zip.namelist():
        if name.endswith(".zip") and not name.startswith("__MACOSX"):
            with outer_zip.open(name) as f:
                inner_buf = io.BytesIO(f.read())
            try:
                with zipfile.ZipFile(inner_buf) as inner:
                    for iname in inner.namelist():
                        if iname.endswith(target) and not iname.startswith("__MACOSX"):
                            with inner.open(iname) as ff:
                                return _parse_jsonl_bytes(ff.read())
            except zipfile.BadZipFile:
                continue
    return None


def _parse_jsonl_bytes(raw: bytes) -> List[dict]:
    text = raw.decode("utf-8", errors="ignore")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _fallback_from_txt_labels(z: zipfile.ZipFile) -> Optional[List[dict]]:
    """Fallback 1: pos.txt / neg.txt / neu.txt / q.txt (1 บรรทัด = 1 texts)."""
    mapping = {
        "wisesight-sentiment-1.1/pos.txt": "pos",
        "wisesight-sentiment-1.1/neg.txt": "neg",
        "wisesight-sentiment-1.1/neu.txt": "neu",
        "wisesight-sentiment-1.1/q.txt": "q",
    }
    rows: List[dict] = []
    found = False
    for name, label in mapping.items():
        try:
            raw = z.read(name).decode("utf-8", errors="ignore").splitlines()
        except KeyError:
            continue
        found = True
        for line in raw:
            line = line.strip()
            if line:
                rows.append({"texts": line, "category": label})
    return rows if found and rows else None


def _fallback_from_kaggle(z: zipfile.ZipFile) -> Optional[List[dict]]:
    """Fallback 2: kaggle-competition/train.txt + train_label.txt."""
    try:
        texts = z.read("wisesight-sentiment-1.1/kaggle-competition/train.txt").decode("utf-8", errors="ignore").splitlines()
        labels = z.read("wisesight-sentiment-1.1/kaggle-competition/train_label.txt").decode("utf-8", errors="ignore").splitlines()
    except KeyError:
        return None
    rows = [
        {"texts": t.strip(), "category": l.strip()}
        for t, l in zip(texts, labels)
        if t.strip() and l.strip() in ("pos", "neg", "neu", "q")
    ]
    return rows or None


def load_wisesight_zip(
    zip_path: str | Path = "data/wisesight-sentiment-1.1.zip",
    target: str = "train.jsonl",
    limit: Optional[int] = None,
) -> "pd.DataFrame":
    """โหลด DataFrame [texts, category] จาก zip โดยไม่แตกไฟล์ลงดิสก์."""
    import pandas as pd

    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path.resolve()}")

    with zipfile.ZipFile(zip_path) as z:
        rows = _read_nested_jsonl(z, target)
        if not rows:
            rows = _fallback_from_kaggle(z)
        if not rows:
            rows = _fallback_from_txt_labels(z)
        if not rows:
            raise ValueError(
                f"ไม่พบ {target} หรือ fallback ใน {zip_path}. namelist: {z.namelist()[:10]}"
            )

    df = pd.DataFrame(rows)
    # normalize columns: รองรับ text/texts, label/category
    colmap = {}
    if "texts" not in df.columns and "text" in df.columns:
        colmap["text"] = "texts"
    if "category" not in df.columns and "label" in df.columns:
        colmap["label"] = "category"
    if colmap:
        df = df.rename(columns=colmap)
    df = df[df["texts"].notna()]
    df["texts"] = df["texts"].astype(str)
    if "category" not in df.columns:
        df["category"] = "neu"
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def clean_raw_text(text: str) -> str:
    """ลบ URL, Mention, สัญลักษณ์พิเศษ. คืน string ที่ยังไม่ตัดคำ."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = KEEP_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def get_stopwords() -> set:
    from pythainlp.corpus import thai_stopwords

    try:
        return set(thai_stopwords())
    except Exception:
        return set()


def tokenize_thai(text: str, stopwords: Optional[set] = None, engine: str = "newmm") -> List[str]:
    """ตัดคำด้วย PyThaiNLP engine=newmm + ลบ stopwords."""
    from pythainlp import word_tokenize

    if not text or len(text.strip()) < 1:
        return []
    if stopwords is None:
        stopwords = get_stopwords()
    tokens = word_tokenize(text, engine=engine, keep_whitespace=False)
    return [t.strip() for t in tokens if t.strip() and t.strip() not in stopwords]


def preprocess_dataframe(df: "pd.DataFrame", engine: str = "newmm") -> "pd.DataFrame":
    """เพิ่ม cleaned_text (str ต่อด้วย space), clean_text (alias), tokens (list)."""
    stopwords = get_stopwords()
    cleaned: List[str] = []
    token_lists: List[List[str]] = []
    for raw in df["texts"].tolist():
        base = clean_raw_text(raw)
        # SKILL constraint: ข้อความ < 2 ตัวอักษร -> tokens ว่าง, cleaned ว่าง
        if len(base.strip()) < 2:
            cleaned.append("")
            token_lists.append([])
            continue
        toks = tokenize_thai(base, stopwords=stopwords, engine=engine)
        token_lists.append(toks)
        cleaned.append(" ".join(toks))
    out = df.copy()
    out["cleaned_text"] = cleaned
    out["clean_text"] = cleaned  # alias ตาม SKILL.md
    out["tokens"] = token_lists
    return out


def load_and_preprocess(
    zip_path: str | Path = "data/wisesight-sentiment-1.1.zip",
    limit: Optional[int] = None,
    engine: str = "newmm",
) -> Tuple["pd.DataFrame", set]:
    df = load_wisesight_zip(zip_path, limit=limit)
    return preprocess_dataframe(df, engine=engine), get_stopwords()


def main() -> None:
    ap = argparse.ArgumentParser(description="Wisesight Phase 1 pipeline")
    ap.add_argument("--zip", dest="zip_path", default="data/wisesight-sentiment-1.1.zip")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--head", type=int, default=5)
    ap.add_argument("--out", default=None, help="เช่น outputs/cleaned.parquet หรือ .csv")
    ap.add_argument("--engine", default="newmm")
    args = ap.parse_args()

    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    try:
        pd = __import__("pandas")
    except ImportError:
        raise SystemExit("need: pip install pandas pythainlp")
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 200)

    print(f"[pipeline] loading: {args.zip_path}")
    df = load_wisesight_zip(args.zip_path, limit=args.limit)
    print(f"[pipeline] raw rows={len(df)} | category dist:\n{df['category'].value_counts()}")
    df = preprocess_dataframe(df, engine=args.engine)
    print(f"[pipeline] head({args.head}):")
    try:
        print(df[["texts", "category", "cleaned_text", "tokens"]].head(args.head).to_string(index=False))
    except UnicodeEncodeError:
        # Windows console cp1252 -> dump ascii-safe preview
        preview = df[["texts", "category", "cleaned_text"]].head(args.head)
        for i, r in preview.iterrows():
            print(f"[{i}] cat={r['category']} ntok={len(df.loc[i, 'tokens'])} texts={str(r['texts'])[:80].encode('ascii', 'replace').decode()}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[pipeline] saved -> {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
