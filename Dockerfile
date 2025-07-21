FROM python:3.11-slim

# OS tools
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright + Chromium
RUN playwright install --with-deps chromium

# App code
COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "app.job_entrypoint"]