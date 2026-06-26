# syntax=docker/dockerfile:1
# ── Streamlit frontend ────────────────────────────────────────────────────────
# Intentionally minimal: the app never imports mmrag pipeline code directly —
# all data comes from the FastAPI backend over HTTP.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only the three packages the Streamlit app imports at module level.
# pandas is a transitive dep of streamlit but listed explicitly for clarity.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        "streamlit>=1.35.0,<2.0.0" \
        "requests>=2.31.0,<3.0.0" \
        "pandas>=2.0.0,<3.0.0"

COPY app/ ./app/

ENV PYTHONUNBUFFERED=1 \
    # API base URL — overridden in docker-compose to point at the api service
    MMRAG_API_BASE=http://api:8000

EXPOSE 8501

# headless=true   — suppresses "open browser" attempts inside the container
# gatherUsageStats=false — no outbound telemetry calls (offline-friendly)
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
