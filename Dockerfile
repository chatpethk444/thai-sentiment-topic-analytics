# Single image for both services (compose overrides the command).
# CPU torch to keep the image small; GPU users can swap the index URL.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for hdbscan/umap/sentencepiece builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install -r requirements.txt

COPY pipeline.py model_engine.py main.py app.py SKILL.md ./
COPY data/ ./data/

EXPOSE 8000 8501

# Default: API. Compose overrides dashboard with `streamlit run`.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
