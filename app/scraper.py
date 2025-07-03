# backend/scraper.py
"""
Reusable Playwright helper for:
1.  logging in (once, cached cookies)
2.  scrolling a target account's follower list
3.  pulling bios for each follower
"""

import asyncio
from pathlib import Path
from typing import List, Dict
from playwright.async_api import Browser, TimeoutError as PlayTimeout
import httpx


from typing import List, Dict
from playwright.async_api import Browser
import httpx, asyncio, time

long_timeout = httpx.Timeout(connect=10.0, write=10.0, read=300.0, pool=None)


def chunks(seq, size: int = 30):
    """Yield successive `size`-sized slices from seq."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


async def classify_remote(bio_texts: list[str], client: httpx.AsyncClient) -> list[str]:
    try:
        resp = await client.post(
            "https://bio-classifier-production.up.railway.app/classify",
            json={"bios": bio_texts},
            timeout=long_timeout,
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
    await page.goto(f"https://www.instagram.com/{username}/", timeout=30_000)
    try:
        desc = await page.get_attribute("head meta[name='description']", "content")
        if desc and " on Instagram: " in desc:
            return desc.split(" on Instagram: ")[1].strip().strip('"')
    except Exception:
        pass
    return ""


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
    )

    # Create two tabs: one for followers list, one for bio fetching
    followers_page = await context.new_page()
    bio_page = await context.new_page()
    # Below is code that needs to be operated online
    await followers_page.route(
        "**/*",
        lambda r: (
            r.abort()
            if r.request.resource_type in ("image", "stylesheet", "font")
            else r.continue_()
        ),
    )
    await bio_page.route(
        "**/*",
        lambda r: (
            r.abort()
            if r.request.resource_type in ("image", "stylesheet", "font")
            else r.continue_()
        ),
    )
    try:

        # -- open the target followers overlay on followers tab --
        t0 = time.perf_counter()
        await followers_page.goto(f"https://www.instagram.com/{target}/")
        nav_time = time.perf_counter() - t0
        print(f"🏁 page.goto took {nav_time*1000:.0f} ms")
        await followers_page.click('a[href$="/followers/"]')
        await followers_page.wait_for_selector('div[role="dialog"]', timeout=25_000)
        dialog = followers_page.locator('div[role="dialog"]').last
        first_link = dialog.locator('a[href^="/"]').first
        await first_link.wait_for(state="attached", timeout=15_000)

        user_links = dialog.locator('a[href^="/"]')

        previous_count = 0
        yes_rows: list[dict] = []  # final result
        seen_handles: set[str] = set()  # so we don't process duplicates
        batch_handles: list[str] = []  # fills up to `batch_size`

        # keep scrolling until we EITHER: (a) have enough yes's OR (b) time is up
        async with httpx.AsyncClient(timeout=long_timeout) as client:
            while len(yes_rows) < target_yes:
                # ------------- SCROLL one "page" -------------
                await user_links.nth(-1).scroll_into_view_if_needed()
                await asyncio.sleep(1)

                # -------- after we add any newly-seen handles --------------
                new_total = len(seen_handles)

                if new_total == previous_count:  # nothing new this pass
                    await asyncio.sleep(20)  # short grace period
                    # try one more time
                    for h in await user_links.all_inner_texts():
                        h = h.strip()
                        if h and h not in seen_handles:
                            seen_handles.add(h)
                            batch_handles.append(h)

                    if len(seen_handles) == previous_count:  # still nothing
                        print("No new followers after two checks – stopping.")
                        break  # bail out of the while-loop

                previous_count = new_total  # update for next round

                # ------------- when we have a full batch, classify it -------------
                if len(batch_handles) >= batch_size:
                    # fetch bios for this batch using the bio tab
                    bios = []
                    for h in batch_handles[:batch_size]:
                        try:
                            bio = await get_bio(bio_page, h)
                            bios.append({"username": h, "bio": bio})
                        except Exception as e:
                            print(f"bio error for {h}: {e}")
                            bios.append({"username": h, "bio": ""})

                    batch_handles = batch_handles[batch_size:]  # remove processed ones

                    # -------- call the remote classifier exactly for this slice --------
                    flags = await classify_remote([b["bio"] for b in bios], client)
                    # flags is already length-matched to bios

                    # keep the winners
                    yes_rows.extend(
                        {
                            "username": bios[int(idx)]["username"],
                            "url": f"https://www.instagram.com/{bios[int(idx)]['username']}/",
                        }
                        for idx in flags
                        if str(idx).isdigit()
                    )

                    print(
                        f"✅ gathered {len(yes_rows)}/{target_yes} yes-profiles so far…"
                    )

        # out of the while loop -> either enough yes's or time ran out
        if batch_handles:  # flush leftovers
            async with httpx.AsyncClient(timeout=long_timeout) as client:
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
        return yes_rows[:target_yes]

    except Exception as e:
        # Capture the page state on failure
        await followers_page.screenshot(
            path=f"shots/error_{target}.png", full_page=True
        )
        print(
            f"❌ Error during scrape: {e}. Screenshot saved to shots/error_{target}.png"
        )
        raise
    finally:
        # Always close the context so the video is flushed to disk
        await context.close()
