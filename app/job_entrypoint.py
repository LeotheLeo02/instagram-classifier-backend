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
        target       = os.environ["TARGET"]
        yes          = int(os.getenv("TARGET_YES", 10))
        batch_size   = int(os.getenv("BATCH_SIZE", 30))
        num_bio_pages = int(os.getenv("NUM_BIO_PAGES", 3))

        print(f"STATE_GCS_URI={state_gcs_uri}", flush=True)
        print(f"TARGET={target}", flush=True)
        print(f"TARGET_YES={yes}", flush=True)
        print(f"BATCH_SIZE={batch_size}", flush=True)
        print(f"NUM_BIO_PAGES={num_bio_pages}", flush=True)

        state_path = Path(_download_state_from_gcs(state_gcs_uri))

        print("Starting scrape_followers", flush=True)

        results = await scrape_followers(
            browser     = browser,
            state_path  = state_path,
            target      = target,
            target_yes  = yes,
            batch_size  = batch_size,
            num_bio_pages = num_bio_pages,
        )

        print("scrape_followers finished", flush=True)

        print(json.dumps(results, indent=2), flush=True)

if __name__ == "__main__":
    asyncio.run(main())