FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "app.job_entrypoint"]