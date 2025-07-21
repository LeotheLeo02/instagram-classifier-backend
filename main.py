# main.py – Railway “start command” →  uvicorn main:app --host 0.0.0.0 --port 8080

import base64, os, tempfile, asyncio, json, shutil
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, Browser
from app.scraper import scrape_followers     # ← your helper from app/scraper.py

###############################################################################
# 1. FastAPI app with a single /scrape endpoint
###############################################################################

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    pw = await async_playwright().start()
    app.state.playwright = pw
    app.state.browser   = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
    # optional: cap concurrency
    app.state.sema      = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT", 2)))
    print("✅ headless Chromium started.")
    print("Lets go!")
    
    yield
    
    # Shutdown
    await app.state.browser.close()
    await app.state.playwright.stop()
    print("🛑   browser closed.")

app = FastAPI(title="IG Christian-Bio Scraper", version="1.0", lifespan=lifespan)


###############################################################################
# 2. Request / Response models
###############################################################################

class ScrapeRequest(BaseModel):
    target      : str              = Field(...,  example="utmartin")
    target_yes  : int              = Field(10,   ge=1, le=200)
    batch_size  : int              = Field(30,   ge=5, le=50)
    state_b64   : str | None = Field(
        None,
        description="Base-64 encoded Playwright storage_state.json "
                    "(omit if sending it as a file upload)."
    )

class ScrapeResponse(BaseModel):
    results: List[dict]            # [{username,url}, …]


###############################################################################
# 3.  /scrape  – one-shot “upload state & get results” endpoint
###############################################################################

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(body: ScrapeRequest):          # ❶ ← only one parameter now
    # ---------------------------------------------------------------- safety
    if not body.state_b64:                      # ❷
        raise HTTPException(422, "state_b64 is required (send cookies in base-64)")

    # ---------------------------------------------------------------- save tmp
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir="/tmp")
    tmp.write(base64.b64decode(body.state_b64))
    tmp.close()

    try:
        async with app.state.sema:              # respect MAX_CONCURRENT
            print(f"🔍 starting scrape for target={body.target}, desired_yes={body.target_yes}")
            yes_rows = await scrape_followers(
                browser    = app.state.browser,
                state_path = tmp.name,
                target     = body.target,
                target_yes = body.target_yes,
                batch_size = body.batch_size,
            )
    except Exception:
        import traceback, sys
        traceback.print_exc()
        raise
    finally:
        os.unlink(tmp.name)
    return ScrapeResponse(results=yes_rows)

