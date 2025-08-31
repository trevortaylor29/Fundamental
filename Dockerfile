FROM python:3.12-slim

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Streamlit env + port
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
    PYTHONUNBUFFERED=1 \
    PORT=8501
EXPOSE 8501

# Start Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.headless=true", "--server.port=${PORT}", "--server.enableCORS=false"]
