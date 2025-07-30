# backend/scraper.py
"""
Reusable Playwright helper for:
1.  logging in (once, cached cookies)
2.  scrolling a target account's follower list
3.  pulling bios for each follower
"""

import asyncio
from google.cloud import storage
from pathlib import Path
from typing import List, Dict
# Playwright timeout error is used in a few places; import it explicitly
from playwright.async_api import Browser, TimeoutError as PlayTimeout
import httpx
import os
import time

# Try to import playwright-stealth for enhanced anti-detection
try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Timeout constants
# ---------------------------------------------------------------------------
# Timeouts for Playwright operations are in milliseconds
PAGE_NAVIGATION_TIMEOUT_MS = 60_000  # 60 seconds
DIALOG_SELECTOR_TIMEOUT_MS = 60_000  # 60 seconds
FIRST_LINK_WAIT_TIMEOUT_MS = 30_000  # 30 seconds
# NEW: much shorter timeout for per-profile bio page loads – we don’t
#      want a single problematic profile to block the run for a minute
BIO_PAGE_TIMEOUT_MS = 10_000        # 10 seconds

# Scroll timing constants (in seconds)
BASE_SCROLL_WAIT = 1.0  # Base wait time after each scroll
IDLE_SCROLL_WAIT = 3.0  # Base wait time when no new followers detected
PROGRESSIVE_WAIT = True # If True, wait time increases with each idle loop
MAX_IDLE_LOOPS = 5      # Number of idle loops before giving up (was 3)

# Timeout for HTTP requests
HTTPX_LONG_TIMEOUT = httpx.Timeout(connect=30.0, write=30.0, read=10_000.0, pool=None)

GCS_BUCKET = os.getenv("SCREENSHOT_BUCKET")


def upload_to_gcs(local_path: str, destination_blob: str) -> None:
    """
    Upload a local file to GCS.
      - local_path:  path on disk, e.g. "shots/no_link.png"
      - destination_blob: path in bucket, e.g. "shots/no_link.png"
    """
    client = storage.Client()  # uses default credentials from Cloud Run
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(destination_blob)
    blob.upload_from_filename(local_path)
    print(f"✅ Screenshot uploaded to gs://{GCS_BUCKET}/{destination_blob}")



def chunks(seq, size: int = 30):
    """Yield successive `size`-sized slices from seq."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]



async def classify_remote(bio_texts: list[str], client: httpx.AsyncClient) -> list[str]:
    try:
        resp = await client.post(
            "https://bio-classifier-672383441505.us-central1.run.app/classify",
            json={"bios": bio_texts},
            timeout=HTTPX_LONG_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("results", [])
        # normalise ("yes yes no…" vs ["0","3"])
        if isinstance(data, str):
            data = data.strip().split()
        if len(data) != len(bio_texts):
            data = (data + [""] * len(bio_texts))[: len(bio_texts)]
        return data
    except Exception as e:
        print("remote classify failed → heuristic fallback", e)
        return [""] * len(bio_texts)


async def get_bio(page, username: str) -> str:
    """Return the profile bio (may be empty) with stealth measures.
    
    A dedicated (shorter) timeout is used so that profiles that fail to load
    don't freeze the whole scraping loop for the default Playwright timeout
    (60 s). If the navigation hits that timeout (or any other error), we
    immediately return an empty string so the caller can continue.
    """
    try:
        # Small random delay before navigation
        
        await page.goto(
            f"https://www.instagram.com/{username}/",
            timeout=BIO_PAGE_TIMEOUT_MS,
            wait_until="domcontentloaded",  # Don't wait for all network requests
        )
        
        # Check for security challenges
        challenge_indicators = [
            'text="Help us confirm it\'s you"',
            'text="challenge"',
            'text="unusual activity"',
            'text="temporarily restricted"',
            'text="reCAPTCHA"',
            'text="Something went wrong"',
            'text="Page not found"',
            '[data-testid="challenge-page"]'
        ]
        
        for indicator in challenge_indicators:
            if await page.locator(indicator).count() > 0:
                print(f"❌ CAPTCHA or challenge detected for {username} – aborting scraping.")
                raise RuntimeError("CAPTCHA or challenge detected")
        
        # Small random delay after navigation
        
    except PlayTimeout:
        # Profile failed to load within BIO_PAGE_TIMEOUT_MS – skip it quickly.
        return ""
    except Exception:
        # Any other unexpected error while navigating → skip
        return ""

    try:
        desc = await page.get_attribute("head meta[name='description']", "content")
        if desc and " on Instagram: " in desc:
            return desc.split(" on Instagram: ")[1].strip().strip('"')
    except Exception:
        pass
    return ""


async def send_notification(message: str, title: str = "Instagram Scraper"):
    """Send push notification using Pushover."""
    pushover_token = os.getenv("PUSHOVER_TOKEN")
    pushover_user = os.getenv("PUSHOVER_USER")
    
    if not pushover_token or not pushover_user:
        print("⚠️ Pushover credentials not configured - skipping notification")
        return
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": pushover_token,
                    "user": pushover_user,
                    "title": title,
                    "message": message,
                    "priority": 0,  # Normal priority
                },
                timeout=10.0
            )
            response.raise_for_status()
            print("📱 Notification sent successfully")
    except Exception as e:
        print(f"❌ Failed to send notification: {e}")


async def scrape_followers(
    browser: Browser,
    state_path: Path,
    target: str,
    target_yes: int = 10,
    batch_size: int = 20,
    num_bio_pages: int = 3,  # New parameter for number of bio pages
) -> List[Dict]:
    """
    Reuses a running `browser` (passed from FastAPI lifespan),
    logs in with given creds (if not cached), scrolls follower list,
    returns a list of {'username': str, 'bio': str}.
    
    Args:
        num_bio_pages: Number of parallel bio pages to use for fetching bios
    """
    
    context = await browser.new_context(
        storage_state=state_path,
        viewport={"width": 1280, "height": 800},
        user_agent="Mozilla/5.0 (X11; Linux x86_64)",
        record_video_dir="videos/"
    )
    
    # Create multiple bio pages for parallel processing
    followers_page = await context.new_page()
    bio_pages = [await context.new_page() for _ in range(num_bio_pages)]
    
    # Apply stealth measures if available
    if STEALTH_AVAILABLE:
        await stealth_async(followers_page)
        for bio_page in bio_pages:
            await stealth_async(bio_page)
    
    async def get_bios_parallel(usernames: list[str]) -> list[dict]:
        """
        Fetch bios for multiple usernames in parallel using multiple bio pages.
        Distributes the workload across available bio pages.
        """
        if not usernames:
            return []
        
        # Split usernames across bio pages
        chunk_size = max(1, len(usernames) // num_bio_pages)
        chunks_list = [usernames[i:i + chunk_size] for i in range(0, len(usernames), chunk_size)]
        
        # Ensure we don't have more chunks than bio pages
        while len(chunks_list) > num_bio_pages:
            # Merge the last two chunks
            chunks_list[-2].extend(chunks_list[-1])
            chunks_list.pop()
        
        # Create tasks for each chunk
        tasks = []
        for i, chunk in enumerate(chunks_list):
            if chunk:  # Only create task if chunk is not empty
                task = asyncio.create_task(
                    get_bios_for_chunk(bio_pages[i % num_bio_pages], chunk)
                )
                tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle any exceptions
        all_bios = []
        for result in results:
            if isinstance(result, Exception):
                print(f"⚠️ Error in bio fetching: {result}")
                # Add empty bios for failed chunk to maintain alignment
                all_bios.extend([{"username": "", "bio": ""}] * len(chunk))
            else:
                all_bios.extend(result)
        
        return all_bios
    
    async def get_bios_for_chunk(bio_page, usernames: list[str]) -> list[dict]:
        """Fetch bios for a chunk of usernames using a single bio page."""
        bios = []
        for username in usernames:
            bio = await get_bio(bio_page, username)
            bios.append({"username": username, "bio": bio})
        return bios
    
    
    try:
        # -- open the target followers overlay on followers tab --
        t0 = time.perf_counter()
        await followers_page.goto(f"https://www.instagram.com/{target}/")
        nav_time = time.perf_counter() - t0
        print(f"🏁 page.goto took {nav_time*1000:.0f} ms")
        
        
            # 1) Use a more tolerant locator
        link = followers_page.locator('header a[href*="/followers"]')

        # 2) Give Playwright a generous default so every locator inherits it
        followers_page.set_default_timeout(25_000)       # 25 s

        # 3) Wait for *existence* first, then for visibility, then click
        await followers_page.wait_for_selector('header a[href*="/followers"]', state="attached")
        await link.wait_for(state="visible")              # respects default timeout
        await link.click()
        await followers_page.wait_for_selector(
            'div[role="dialog"]',
            timeout=DIALOG_SELECTOR_TIMEOUT_MS,
        )  # Increased from 25_000 to 60_000 (60 seconds)
        dialog = followers_page.locator('div[role="dialog"]').last
        first_link = dialog.locator('a[href^="/"]').first
        await first_link.wait_for(
            state="attached",
            timeout=FIRST_LINK_WAIT_TIMEOUT_MS,
        )  # Increased from 15_000 to 30_000 (30 seconds)

        user_links = dialog.locator('a[href^="/"]')

        previous_count = 0
        yes_rows: list[dict] = []  # final result
        seen_handles: set[str] = set()  # so we don't process duplicates
        batch_handles: list[str] = []  # fills up to `batch_size`

        # keep scrolling until we EITHER: (a) have enough yes's OR (b) time is up
        idle_loops = 0
        previous_count = 0
        start_time = time.perf_counter()
        timeout_seconds = 3600  # Cloud Run timeout is 3600 seconds

        scroll_timeout = False
        async with httpx.AsyncClient(timeout=HTTPX_LONG_TIMEOUT) as client:
            while len(yes_rows) < target_yes:
                # Check if we're approaching timeout (leave 30 seconds buffer for cleanup)
                elapsed_time = time.perf_counter() - start_time
                if elapsed_time > (timeout_seconds - 30):
                    print(f"⏰ Timeout approaching ({elapsed_time:.1f}s elapsed, {timeout_seconds}s limit). Returning partial results.")
                    break
                
                # 1️⃣  Collect any handles currently visible *before* we test for growth
                for h in await user_links.all_inner_texts():
                    h = h.strip()
                    if h and h not in seen_handles:
                        seen_handles.add(h)
                        batch_handles.append(h)

                # 2️⃣  Check if we made progress
                new_total = len(seen_handles)
                if new_total == previous_count:
                    idle_loops += 1
                    if idle_loops >= MAX_IDLE_LOOPS:          # three scrolls with zero growth ⇒ stop
                        print(f"No new followers after {MAX_IDLE_LOOPS} scrolls – quitting.")
                        break
                else:
                    idle_loops = 0               # reset because we *did* add something
                previous_count = new_total

                # 3️⃣  Scroll the last visible link into view
                try:
                    await user_links.nth(-1).scroll_into_view_if_needed()
                except PlayTimeout:
                    print("⚠️ Scroll timeout encountered – likely end of list.")
                    scroll_timeout = True
                    break
                
                # Dynamic wait time: longer when idle, shorter when making progress
                if idle_loops > 0:
                    if PROGRESSIVE_WAIT:
                        # Progressive wait: 3s, 5s, 7s for idle loops 1, 2, 3
                        wait_time = IDLE_SCROLL_WAIT + (idle_loops - 1) * 2
                    else:
                        wait_time = IDLE_SCROLL_WAIT
                    print(f"⏳ Idle loop {idle_loops}/{MAX_IDLE_LOOPS}, waiting {wait_time}s...")
                else:
                    wait_time = BASE_SCROLL_WAIT
                await asyncio.sleep(wait_time)

                # 4️⃣  Batch → classify exactly as you already do
                if len(batch_handles) >= batch_size:
                    try:
                        # Use parallel bio fetching with multiple pages
                        current_batch = batch_handles[:batch_size]
                        bios = await get_bios_parallel(current_batch)
                        
                        # Filter out any failed bio fetches (empty usernames)
                        valid_bios = [b for b in bios if b["username"]]
                        
                        batch_handles = batch_handles[batch_size:]     # trim

                        if valid_bios:
                            # Process bios in chunks of 10 for classification
                            all_flags = []
                            for i in range(0, len(valid_bios), 10):
                                chunk = valid_bios[i:i+10]
                                chunk_flags = await classify_remote(
                                    [b["bio"] for b in chunk], client
                                )
                                all_flags.extend(chunk_flags)
                            
                            # Process results from all chunks
                            yes_idx = {int(f) for f in all_flags if f.isdigit()}
                            for idx in yes_idx:
                                if idx < len(valid_bios) and valid_bios[idx]["bio"]:  # make sure bio isn't empty
                                    yes_rows.append({
                                        "username": valid_bios[idx]["username"],
                                        "url": f"https://www.instagram.com/{valid_bios[idx]['username']}/",
                                        "bio": valid_bios[idx]["bio"]
                                    })

                        print(f"✅ gathered {len(yes_rows)}/{target_yes} so far…")
                    except RuntimeError as e:
                        print(str(e))
                        print("🛑 Aborting scraping due to challenge.")
                        break
            
            # out of the while loop -> either enough yes's or time ran out
            if batch_handles:  # flush leftovers
                async with httpx.AsyncClient(timeout=HTTPX_LONG_TIMEOUT) as client:
                    # Use parallel bio fetching for leftovers too
                    bios = await get_bios_parallel(batch_handles)
                    valid_bios = [b for b in bios if b["username"]]
                    
                    if valid_bios:
                        # Process leftover bios in chunks of 10 for classification
                        all_flags = []
                        for i in range(0, len(valid_bios), 10):
                            chunk = valid_bios[i:i+10]
                            chunk_flags = await classify_remote([b["bio"] for b in chunk], client)
                            all_flags.extend(chunk_flags)
                        
                        # Process results from all chunks
                        yes_idx = {int(f) for f in all_flags if f.isdigit()}
                        # Add "yes" results
                        for idx in yes_idx:
                            if idx < len(valid_bios) and valid_bios[idx]["bio"]:  # make sure bio isn't empty
                                yes_rows.append({
                                    "username": valid_bios[idx]["username"],
                                    "url": f"https://www.instagram.com/{valid_bios[idx]['username']}/",
                                    "bio": valid_bios[idx]["bio"]
                                })
        # Check if we got partial results due to scrolling or timeout
        elapsed_time = time.perf_counter() - start_time
        if scroll_timeout:
            print(
                f"⚠️ Returning partial results: {len(yes_rows)}/{target_yes} (scroll timeout)"
            )
            await send_notification(
                f"Partial results: {len(yes_rows)}/{target_yes} followers found for @{target} (scroll timeout)",
                "Instagram Scraper - Partial Results",
            )
            return yes_rows
        elif len(yes_rows) < target_yes and elapsed_time >= (timeout_seconds - 30):
            print(f"⚠️ Returning partial results: {len(yes_rows)}/{target_yes} (timeout reached after {elapsed_time:.1f}s)")
            # Send notification about partial results
            await send_notification(
                f"Partial results: {len(yes_rows)}/{target_yes} followers found for @{target}",
                f"Instagram Scraper - Partial Results"
            )
            return yes_rows
        else:
            print(f"✅ Completed successfully: {len(yes_rows)}/{target_yes} results in {elapsed_time:.1f}s")
            # Send notification about successful completion
            await send_notification(
                f"Successfully found {len(yes_rows)}/{target_yes} followers for @{target} in {elapsed_time:.1f}s",
                f"Instagram Scraper - Complete"
            )
            return yes_rows

    except Exception as e:
        # Capture the page state on failure
        os.makedirs("shots", exist_ok=True)
        not_visible_path = "shots/not_visible.png"
        try:
            await followers_page.screenshot(path=not_visible_path, full_page=True)
        except PlayTimeout:
            print("Screenshot timed out – continuing without it")
        upload_to_gcs(local_path=not_visible_path, destination_blob=not_visible_path)
        # Finalize and upload video files
        await context.close()
        # for p in (followers_page, *bio_pages):
        #     try:
        #         await p.close()
        #         video_path = await p.video.path()
        #         if video_path:
        #             upload_to_gcs(local_path=video_path, destination_blob=f"videos/{os.path.basename(video_path)}")
        #     except Exception as ve:
        #         print(f"⚠️ Could not upload video: {ve}")
        # Send notification about failure
        await send_notification(
            f"Scraping failed for @{target}: {str(e)[:100]}...",
            f"Instagram Scraper - Error"
        )
        raise
    finally:
        # Always close the context so the video is flushed to disk
        await context.close()
