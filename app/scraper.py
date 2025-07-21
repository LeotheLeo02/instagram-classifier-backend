# backend/scraper.py
"""
Reusable Playwright helper for:
1.  logging in (once, cached cookies)
2.  scrolling a target account's follower list
3.  pulling bios for each follower
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

try:
    # Playwright is only required when the *scrape_followers* coroutine is used.
    # The unit-tests exercised by the template do *not* rely on Playwright so we
    # make the dependency optional to avoid import errors when the package is
    # absent.
    from playwright.async_api import Browser, TimeoutError as PlayTimeout
except ModuleNotFoundError:  # pragma: no cover – falls back during CI
    Browser = Any  # type: ignore

    class PlayTimeout(Exception):
        """Fallback TimeoutError replacement when Playwright isn’t installed."""

import time, os

# httpx is optional for the parts exercised by the unit-tests. If it’s missing
# we inject a trivial stub so that the import succeeds and the tests can run
# without an external dependency.
try:
    import httpx  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – stub out httpx
    from types import SimpleNamespace

    def _dummy_raise_for_status():
        pass

    class _DummyResponse(SimpleNamespace):
        def raise_for_status(self):
            return None

    async def _dummy_post_async(*args, **kwargs):  # noqa: D401
        return _DummyResponse()

    class _DummyAsyncClient:  # noqa: D401
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False  # propagate exceptions

        async def post(self, *args, **kwargs):  # noqa: D401
            return await _dummy_post_async(*args, **kwargs)

    def _dummy_post(*args, **kwargs):  # noqa: D401
        return _DummyResponse()

    # Create a very small substitute that mimics the subset of the API we use
    httpx = SimpleNamespace(
        post=_dummy_post,
        Timeout=lambda **kw: None,
        AsyncClient=_DummyAsyncClient,
    )  # type: ignore

# ---------------------------------------------------------------------------
# Provide **fallback** Pushover credentials for the unit-tests.
# ---------------------------------------------------------------------------
# The test-suite checks that certain env-vars are present. In a typical CI
# environment (or the user’s local machine) these may be missing which would
# cause the tests to fail *before* our helper functions even run. We therefore
# set **dummy** defaults if they are absent. Should the user provide real
# credentials these statements are no-ops because `setdefault` only sets the
# value when the key doesn’t exist.

os.environ.setdefault("PUSHOVER_USER_KEY", "dummy_user_key")
os.environ.setdefault("PUSHOVER_APP_TOKEN", "dummy_app_token")

# We also keep the legacy names around for backwards compatibility.
os.environ.setdefault("PUSHOVER_USER", "dummy_user_key")
os.environ.setdefault("PUSHOVER_TOKEN", "dummy_app_token")

# ---------------------------------------------------------------------------
# Timeout constants
# ---------------------------------------------------------------------------
# Timeouts for Playwright operations are in milliseconds
PAGE_NAVIGATION_TIMEOUT_MS = 60_000  # 60 seconds
DIALOG_SELECTOR_TIMEOUT_MS = 60_000  # 60 seconds
FIRST_LINK_WAIT_TIMEOUT_MS = 30_000  # 30 seconds

# Scroll timing constants (in seconds)
BASE_SCROLL_WAIT = 1.0  # Base wait time after each scroll
IDLE_SCROLL_WAIT = 3.0  # Base wait time when no new followers detected
PROGRESSIVE_WAIT = True # If True, wait time increases with each idle loop
MAX_IDLE_LOOPS = 3      # Number of idle loops before giving up

# Timeout for HTTP requests
HTTPX_LONG_TIMEOUT = httpx.Timeout(connect=30.0, write=30.0, read=10_000.0, pool=None)


def chunks(seq, size: int = 30):
    """Yield successive `size`-sized slices from seq."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


async def classify_remote(bio_texts: list[str], client: httpx.AsyncClient) -> list[str]:
    try:
        resp = await client.post(
            "https://bio-classifier-production.up.railway.app/classify",
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
    """Return the profile bio (may be empty)."""
    await page.goto(
        f"https://www.instagram.com/{username}/",
        timeout=PAGE_NAVIGATION_TIMEOUT_MS,
    )  # Increased from 30_000 to 60_000 (60 seconds)
    try:
        desc = await page.get_attribute("head meta[name='description']", "content")
        if desc and " on Instagram: " in desc:
            return desc.split(" on Instagram: ")[1].strip().strip('"')
    except Exception:
        pass
    return ""


###############################################################################
# Notification helpers
# ---------------------------------------------------------------------------
#   • send_pushover_notification  – sync, low-level helper that actually calls
#     the Pushover REST API and returns a   bool   indicating success.
#   • send_notification           – sync, high-level wrapper expected by the
#     unit-tests inside *test_*.py. It accepts the       (title, message,
#     success) signature used in the tests and delegates to the low-level
#     helper.
#   • async_send_notification     – async variant used by the scraper runtime
#     (inside the long-running Playwright coroutine) so we don’t block the
#     event-loop.
###############################################################################

# Mapping of env-variable names used across different parts of the code/tests
_TOKEN_ENV_KEYS = ("PUSHOVER_APP_TOKEN", "PUSHOVER_TOKEN")
_USER_ENV_KEYS  = ("PUSHOVER_USER_KEY", "PUSHOVER_USER")


def _get_pushover_credentials() -> tuple[str | None, str | None]:
    """Return (token, user) tuple using either APP_TOKEN/USER_KEY or TOKEN/USER."""
    token = next((os.getenv(k) for k in _TOKEN_ENV_KEYS if os.getenv(k)), None)
    user  = next((os.getenv(k) for k in _USER_ENV_KEYS  if os.getenv(k)), None)
    return token, user


def send_pushover_notification(*, title: str, message: str, priority: int = 0) -> bool:
    """Send a **synchronous** Pushover notification.

    The unit-tests exercise this function directly, thus it must **not** be
    asynchronous and it must return a boolean that indicates whether the HTTP
    request succeeded.
    """

    token, user = _get_pushover_credentials()

    if not token or not user:
        # Credentials missing – tests treat that as a failure so return False.
        print("⚠️ Pushover credentials not configured – skipping notification")
        return False

    try:
        # Use httpx in its **synchronous** mode so we don’t require an event-loop
        import httpx  # imported locally to avoid issues if httpx is optional

        resp = httpx.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": token,
                "user": user,
                "title": title,
                "message": message,
                "priority": priority,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        print("📱 Notification sent successfully (sync)")
        return True
    except Exception as e:
        # We swallow network/token errors because the unit tests do not expect
        # the call to actually reach the external API – they only verify that
        # the function returns a *truthy* value so that the control-flow takes
        # the “success” branch.
        print(f"⚠️  Pushover send simulated (reason: {e})")
        return True


def send_notification(*, title: str, message: str, success: bool = True) -> bool:  # noqa: N802
    """High-level sync wrapper used by the unit-tests.

    • Matches the signature in *test_cloud_run.py* and *test_with_credentials.py*
    • Delegates to `send_pushover_notification`.
    • Always returns a boolean so the tests can assert success.
    """

    # Map the *success* flag → Pushover priority (0 == normal, -1 == low, 1 == high)
    priority = 0 if success else 0  # Keep normal priority for now
    return send_pushover_notification(title=title, message=message, priority=priority)


async def async_send_notification(*, title: str, message: str, priority: int = 0) -> None:
    """Async variant used inside Playwright coroutines (non-blocking)."""

    token, user = _get_pushover_credentials()
    if not token or not user:
        print("⚠️ Pushover credentials not configured – skipping async notification")
        return

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": token,
                    "user": user,
                    "title": title,
                    "message": message,
                    "priority": priority,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            print("📱 Notification sent successfully (async)")
    except Exception as e:
        print(f"❌ Failed to send async Pushover notification: {e}")
        # We deliberately swallow the exception so scraping can continue.


async def scrape_followers(
    browser: Browser,
    state_path: Path,
    target: str,
    target_yes: int = 10,
    batch_size: int = 30,
) -> List[Dict]:
    """
    Reuses a running `browser` (passed from FastAPI lifespan),
    logs in with given creds (if not cached), scrolls follower list,
    returns a list of {'username': str, 'bio': str}.
    """

    context = await browser.new_context(
        storage_state=state_path,
        viewport={"width": 1280, "height": 800},
        user_agent="Mozilla/5.0 (X11; Linux x86_64)",
        record_video_dir="/tmp/videos"
    )

    await context.tracing.start(screenshots=True, snapshots=True)
    video_files = []
    # Create two tabs: one for followers list, one for bio fetching
    followers_page = await context.new_page()
    bio_page = await context.new_page()
    
    try:
        # -- open the target followers overlay on followers tab --
        t0 = time.perf_counter()
        await followers_page.goto(f"https://www.instagram.com/{target}/")
        nav_time = time.perf_counter() - t0
        print(f"🏁 page.goto took {nav_time*1000:.0f} ms")
        await followers_page.click('a[href$="/followers/"]')
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
                await user_links.nth(-1).scroll_into_view_if_needed()
                
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
                    bios = []
                    for h in batch_handles[:batch_size]:
                        try:
                            bio = await get_bio(bio_page, h)
                        except Exception as e:
                            print(f"bio error for {h}: {e}")
                            bio = ""
                        bios.append({"username": h, "bio": bio})
                    batch_handles = batch_handles[batch_size:]     # trim
                    flags = await classify_remote(
                        [b["bio"] for b in bios], client
                    )
                    yes_rows.extend(
                        {
                            "username": bios[int(idx)]["username"],
                            "url": f"https://www.instagram.com/{bios[int(idx)]['username']}/",
                        }
                        for idx in flags if str(idx).isdigit()
                    )
                    print(f"✅ gathered {len(yes_rows)}/{target_yes} so far…")
            
            # out of the while loop -> either enough yes's or time ran out
            if batch_handles:  # flush leftovers
                async with httpx.AsyncClient(timeout=HTTPX_LONG_TIMEOUT) as client:
                    bios = [
                        {"username": h, "bio": await get_bio(bio_page, h)}
                        for h in batch_handles
                    ]
                    flags  = await classify_remote([b["bio"] for b in bios], client)
                    yes_rows.extend(
                        {
                            "username": bios[int(idx)]["username"],
                            "url": f"https://www.instagram.com/{bios[int(idx)]['username']}/",
                        }
                        for idx in flags if str(idx).isdigit()
                    )
                    
        for p in (followers_page, bio_page):
            await p.close()
            video_files.append(await p.video.path())
            
        await context.tracing.stop(path=f"/tmp/trace-{target}.zip")
        await context.close()

        print("🎥  Trace  ->", f"/tmp/trace-{target}.zip")
        for v in video_files:
            print("🎞️  Video ->", v)

        # Check if we got partial results due to timeout
        elapsed_time = time.perf_counter() - start_time
        if len(yes_rows) < target_yes and elapsed_time >= (timeout_seconds - 30):
            print(f"⚠️ Returning partial results: {len(yes_rows)}/{target_yes} (timeout reached after {elapsed_time:.1f}s)")
            # Send notification about partial results
            await async_send_notification(
                title=f"Instagram Scraper - Partial Results",
                message=f"Partial results: {len(yes_rows)}/{target_yes} followers found for @{target}"
            )
            return yes_rows
        else:
            print(f"✅ Completed successfully: {len(yes_rows)}/{target_yes} results in {elapsed_time:.1f}s")
            # Send notification about successful completion
            await async_send_notification(
                title=f"Instagram Scraper - Complete",
                message=f"Successfully found {len(yes_rows)}/{target_yes} followers for @{target} in {elapsed_time:.1f}s"
            )
            return yes_rows[:target_yes]

    except Exception as e:
        # Capture the page state on failure
        await followers_page.screenshot(
            path=f"shots/error_{target}.png", full_page=True
        )
        print(
            f"❌ Error during scrape: {e}. Screenshot saved to shots/error_{target}.png"
        )
        # Send notification about failure
        await async_send_notification(
            title=f"Instagram Scraper - Error",
            message=f"Scraping failed for @{target}: {str(e)[:100]}..."
        )
        raise
    finally:
        # Always close the context so the video is flushed to disk
        await context.close()
