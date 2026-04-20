# Python 3.14 matches the training environment (venv/pyvenv.cfg).
# If python:3.14-slim is unavailable on your registry, use python:3.13-slim —
# sklearn models pickled in 3.14 are compatible with 3.13.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# build-essential + curl: shap compiles native extensions; curl used by HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (maximise layer cache re-use)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and all pre-trained model artifacts
COPY dashboard.py \
     model.pkl \
     shap_model.pkl \
     scaler.pkl \
     feature_cols.pkl \
     ./

# Streamlit server config — headless, fixed port, disable XSRF for Azure ingress proxy
RUN mkdir -p /app/.streamlit && \
    printf "[server]\nheadless = true\nport = 8501\nenableXsrfProtection = false\n" \
    > /app/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0"]
