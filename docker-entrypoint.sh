#!/usr/bin/env sh
echo "📡  Running speed test…"
speedtest-cli --simple || echo "⚠️  Speed test failed, continuing"

echo "🚀  Starting FastAPI on port ${PORT:-8000}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"