FROM python:3.11-slim

# 1. basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg && \
    rm -rf /var/lib/apt/lists/*

# 2. python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. playwright ⬇ (installs Chromium + deps ~120 MB)
RUN playwright install --with-deps chromium

# 4. copy code
COPY . .

# 5. expose   (Railway respects $PORT env but we’ll default to 8000)
ENV PORT=8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
