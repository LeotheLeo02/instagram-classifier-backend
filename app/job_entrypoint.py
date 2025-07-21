# app/job_entrypoint.py
"""
Entry-point used when JOB_MODE=job.
Every parameter arrives via env-variables that the Cloud-Run Job
injects (TARGET, TARGET_YES, BATCH_SIZE, STATE_B64 …).
"""

import os, base64, tempfile, json, asyncio
from pathlib import Path
from google.cloud import storage

from playwright.async_api import async_playwright
from app.scraper import scrape_followers

def _download_state_from_gcs(gs_uri: str) -> str:
    if not gs_uri.startswith("gs://"):
        raise ValueError("STATE_GCS_URI must start with gs://")
    bucket_name, blob_name = gs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "wb") as fp:
        blob.download_to_file(fp)
    return path

async def main() -> None:
    print("Job entrypoint started", flush=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])

        state_gcs_uri = os.environ["STATE_GCS_URI"]
        target      = os.environ["TARGET"]
        yes         = int(os.getenv("TARGET_YES", 10))
        batch_size  = int(os.getenv("BATCH_SIZE", 30))
        state_path = Path(_download_state_from_gcs(state_gcs_uri))

        results = await scrape_followers(
            browser     = browser,
            state_path  = state_path,
            target      = target,
            target_yes  = yes,
            batch_size  = batch_size,
        )

        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())