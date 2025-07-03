############################  Base image
FROM python:3.11-slim

############################  1. minimal OS tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg && \
    rm -rf /var/lib/apt/lists/*

############################  2. Python deps
WORKDIR /app
COPY requirements.txt .
# install your project deps + speedtest-cli in one go
RUN pip install --no-cache-dir -r requirements.txt speedtest-cli

############################  3. Playwright + Chromium
RUN playwright install --with-deps chromium

############################  4. copy source code
COPY . .

############################  5. entry-point that does the speed test first
COPY docker-entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint

############################  6. start
ENV PORT=8000
ENTRYPOINT ["/usr/local/bin/entrypoint"]