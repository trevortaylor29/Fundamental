FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# NOTE: use /bin/sh -c so ${PORT} expands; default to 8501 if unset
CMD ["/bin/sh","-c","streamlit run streamlit_app.py --server.headless=true --server.address=0.0.0.0 --server.enableCORS=false --server.port=${PORT:-8501}"]
