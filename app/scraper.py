# backend/scraper.py
"""
Instagram follower scraper with bio classification.

This module provides functionality to:
1. Log into Instagram (with cached cookies)
2. Scroll through a target account's follower list
3. Extract and classify follower bios
4. Save results to CSV and upload to Google Cloud Storage
"""

import asyncio
import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import httpx
from google.cloud import storage
from playwright.async_api import Browser, Page, TimeoutError as PlayTimeout, Error as PlaywrightError

# Try to import playwright-stealth for enhanced anti-detection
try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

# Import our modular classes
from .config import ScraperConfig
from .validators import BioValidator
from .exporters import CSVExporter
from .classifiers import BioClassifier
from .notifications import NotificationService
from .ratelimit import RateLimiter, with_retry


class InstagramScraper:
    """Main Instagram scraper class."""
    
    def __init__(self, gcs_bucket: Optional[str] = None, exec_id: Optional[str] = None):
        self.gcs_bucket = gcs_bucket or os.getenv("SCREENSHOT_BUCKET")
        self.csv_exporter = CSVExporter(self.gcs_bucket)
        self.bio_classifier = BioClassifier()
        self.notification_service = NotificationService()
        # Preferred exec_id comes from caller/env so backend/launcher can pre-select folder
        # Fallback to None; we'll generate if still missing when needed
        self.exec_id = exec_id or os.getenv("EXEC_ID") or os.getenv("SCRAPE_EXEC_ID")
        self._rl_graphql = RateLimiter(
            ScraperConfig.GRAPHQL_QPS,
            ScraperConfig.GRAPHQL_BURST,
            "graphql",
        )
        self._rl_profile = RateLimiter(
            ScraperConfig.PROFILE_QPS,
            ScraperConfig.PROFILE_BURST,
            "profile",
        )
        self._last_429_at: Optional[float] = None
        self._last_5xx_at: Optional[float] = None
        self._throttle_events: int = 0
        self._server_error_events: int = 0
        self._last_metrics_report: float = 0.0
        self._rolling_latency: float = 0.0
        self._last_latency_report: float = 0.0

    def _gcs_prefix(self, target: str, operation_id: str) -> str:
        """Stable prefix for result artifacts in GCS."""
        # e.g., scrapes/<target>/<op_1722980000>/
        return f"scrapes/{target}/{operation_id}/"

    def _update_latency_metric(self, previous: float, latest: float) -> float:
        smoothing = ScraperConfig.LATENCY_SMOOTHING
        if previous <= 0:
            return latest
        return (1.0 - smoothing) * previous + smoothing * latest

    def _log_latency(self, name: str, latest_value: float) -> None:
        now = time.time()
        if now - self._last_latency_report < 15.0:
            return
        print(f"[metrics] {name} rolling latency ~ {latest_value:.2f}s")
        self._last_latency_report = now

    def _note_throttle(self, name: str, attempt: int) -> None:
        self._throttle_events += 1
        self._last_429_at = time.time()
        print(f"[metrics] {name} received 429 on attempt {attempt+1}")

    def _note_server_error(self, name: str, attempt: int) -> None:
        self._server_error_events += 1
        self._last_5xx_at = time.time()
        print(f"[metrics] {name} server error on attempt {attempt+1}")

    def _upload_json_to_gcs(self, obj, destination_blob: str) -> Optional[str]:
        """Serialize obj to a temp JSON file and upload to GCS via CSVExporter.
        Returns the gs:// path if uploaded, else None.
        """
        if not self.gcs_bucket:
            print(f"⚠️ No GCS bucket configured; skipping upload to {destination_blob}")
            return None
        # Create a temp JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(obj, tf, ensure_ascii=False)
            tmp_path = tf.name
        try:
            self.csv_exporter.upload_to_gcs(local_path=tmp_path, destination_blob=destination_blob)
            return f"gs://{self.gcs_bucket}/{destination_blob}"
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _ensure_start_artifacts(self, target: str, target_yes: int, started_at_epoch: Optional[int] = None) -> str:
        """Ensure a stable exec_id and upload START/meta(status=running) before heavy work.
        Returns the effective exec_id used.
        """
        if not self.exec_id:
            # Generate only if not provided by env or caller
            self.exec_id = f"op_{int(time.time())}"
        if started_at_epoch is None:
            started_at_epoch = int(time.time())

        prefix = self._gcs_prefix(target, self.exec_id)
        # Upload a START marker and running meta to avoid 404s for early polling
        try:
            self._upload_json_to_gcs({"ok": True, "ts": started_at_epoch}, prefix + "START")
            running_meta = {
                "status": "running",
                "target": target,
                "target_yes": target_yes,
                "started_at": started_at_epoch,
                "operation_id": self.exec_id,
            }
            self._upload_json_to_gcs(running_meta, prefix + "meta.json")
        except Exception:
            # Non-fatal; scraping can still proceed even if pre-create fails
            pass
        return self.exec_id

    def _followers_page_blob(self, target: str) -> str:
        return f"followers-cache/{target}/page.json"

    async def _load_cached_followers_page(self, target: str) -> Optional[Dict[str, Any]]:
        if not self.gcs_bucket:
            return None
        try:
            loop = asyncio.get_running_loop()
            client = storage.Client()
            bucket = client.bucket(self.gcs_bucket)
            blob = bucket.blob(self._followers_page_blob(target))
            exists = await loop.run_in_executor(None, blob.exists)
            if not exists:
                return None
            data = await loop.run_in_executor(None, blob.download_as_text)
            return json.loads(data)
        except Exception as exc:
            print(f"⚠️ Failed to load cached followers page for {target}: {exc}")
            return None

    async def _save_followers_page(self, target: str, entries: Iterable[str], next_cursor: Optional[str]) -> None:
        if not self.gcs_bucket:
            return
        payload = {
            "followers": list(entries),
            "next_cursor": next_cursor,
            "updated_at": int(time.time()),
        }
        blob_name = self._followers_page_blob(target)
        loop = asyncio.get_running_loop()
        tmp_path: Optional[str] = None
        try:
            client = storage.Client()
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as tf:
                json.dump(payload, tf, ensure_ascii=False)
                tmp_path = tf.name
            await loop.run_in_executor(
                None,
                lambda: client.bucket(self.gcs_bucket).blob(blob_name).upload_from_filename(tmp_path),
            )
        finally:
            try:
                if tmp_path:
                    os.remove(tmp_path)
            except OSError:
                pass
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in seconds to a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"

    @staticmethod
    def log_classification_stats(total_bios: int, valid_bios: int, yes_count: int, target_yes: int) -> None:
        """Log comprehensive classification statistics for current batch."""
        print(f"📊 Batch Classification Statistics:")
        print(f"   - Total bios in batch: {total_bios}")
        print(f"   - Valid bios after filtering: {valid_bios}")
        print(f"   - Bios classified as 'yes' in this batch: {yes_count}")
        print(f"   - Target 'yes' count: {target_yes}")
        if valid_bios > 0:
            success_rate = (yes_count / valid_bios) * 100
            print(f"   - Batch success rate: {success_rate:.1f}%")
        if yes_count > 0:
            progress = (yes_count / target_yes) * 100
            print(f"   - Batch contribution to target: {progress:.1f}%")

    def _parse_username_from_href(self, href: Optional[str]) -> Optional[str]:
        """Parse a username from an anchor href.
        Supports relative like "/user/" and absolute like "https://www.instagram.com/user/".
        Filters out non-profile paths such as /p/, /explore/, etc.
        """
        if not href:
            return None
        try:
            if href.startswith("http"):
                from urllib.parse import urlparse
                path = urlparse(href).path
            else:
                path = href

            if not path or path == "/":
                return None

            disallowed_prefixes = (
                "/p/", "/explore/", "/reels/", "/stories/", "/direct/", "/accounts/",
            )
            if any(path.startswith(pref) for pref in disallowed_prefixes):
                return None

            candidate = path.strip("/").split("/")[0]
            # Instagram usernames: letters, digits, dot, underscore (1..30)
            if re.match(r"^[A-Za-z0-9._]{1,30}$", candidate):
                return candidate
        except Exception:
            return None
        return None
    
    async def _get_bio(self, page: Page, username: str) -> str:
        """Extract bio from a user's profile page."""
        try:
            await self._rl_profile.acquire()
            await page.goto(
                f"https://www.instagram.com/{username}/",
                timeout=ScraperConfig.BIO_PAGE_TIMEOUT_MS,
                wait_until="domcontentloaded",
            )
            await asyncio.sleep(0.2 + random.random() * 0.4)
            print(f"[bio] navigated to profile for {username}")
            
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
                    
        except PlayTimeout:
            print(f"[bio] timeout while loading profile for {username}")
            return ""
        except Exception as exc:
            print(f"[bio] error loading profile for {username}: {exc}")
            return ""

        try:
            desc = await page.get_attribute("head meta[name='description']", "content")
            if desc and " on Instagram: " in desc:
                bio_text = desc.split(" on Instagram: ")[1].strip().strip('"')
                preview = bio_text[:80].replace("\n", " ")
                print(f"[bio] extracted bio for {username}: {preview}")
                return bio_text
            else:
                print(f"[bio] meta description missing expected pattern for {username}: {desc}")
        except Exception as exc:
            print(f"[bio] error extracting meta description for {username}: {exc}")
        return ""
    
    async def _get_bio_api_first(self, page: Page, username: str) -> str:
        """Attempt lightweight web_profile_info before falling back to full page navigation."""
        try:
            url = f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}"
            resp = await with_retry(
                lambda: page.request.get(
                    url,
                    headers={
                        "X-IG-App-ID": ScraperConfig.IG_APP_ID,
                        "Referer": f"https://www.instagram.com/{username}/",
                    },
                ),
                limiter=self._rl_graphql,
                what=f"web_profile_info for {username}",
                hard_penalty_factor=ScraperConfig.RATE_LIMIT_PENALTY_FACTOR,
                hard_penalty_seconds=ScraperConfig.RATE_LIMIT_PENALTY_SECONDS,
                soft_penalty_factor=ScraperConfig.RATE_LIMIT_PENALTY_SOFT_FACTOR,
                soft_penalty_seconds=ScraperConfig.RATE_LIMIT_PENALTY_SOFT_SECONDS,
                on_429=lambda attempt: self._note_throttle("web_profile_info", attempt),
                on_5xx=lambda attempt: self._note_server_error("web_profile_info", attempt),
            )
            if resp.status == 200:
                data = await resp.json()
                bio = ((data.get("data") or {}).get("user") or {}).get("biography") or ""
                if bio:
                    return bio
        except Exception:
            pass

        return await self._get_bio(page, username)

    async def _ensure_bio_pages_alive(self, context, bio_pages: List[Page]) -> List[Page]:
        alive: List[Page] = []
        total_expected = len(bio_pages)

        for page in bio_pages:
            try:
                if page and not page.is_closed():
                    alive.append(page)
            except PlaywrightError:
                continue

        context_open = False
        if context:
            is_closed_method = getattr(context, "is_closed", None)
            if callable(is_closed_method):
                try:
                    context_open = not context.is_closed()
                except PlaywrightError:
                    context_open = False
            else:
                try:
                    _ = context.pages
                    context_open = True
                except PlaywrightError:
                    context_open = False

        if context_open and total_expected > len(alive):
            for _ in range(total_expected - len(alive)):
                try:
                    new_page = await context.new_page()
                except PlaywrightError as e:
                    print(f"[fetcher] failed to create replacement bio page ({type(e).__name__}: {e})")
                    break
                if STEALTH_AVAILABLE:
                    try:
                        await stealth_async(new_page)
                    except Exception as e:
                        print(f"[fetcher] stealth setup failed ({e}); continuing without stealth")
                alive.append(new_page)

        return alive
    
    async def _scrape_followers_parallel(
        self,
        context,
        followers_page: Page,
        bio_pages: List[Page],
        target: str,
        target_yes: int,
        batch_size: int,
        criteria_text: Optional[str] = None,
    ) -> List[Dict]:
        """
        Parallel scraping implementation using producer-consumer pattern.
        Fetches bios and classifies them concurrently for maximum speed.
        """
        from asyncio import Queue
        
        # Initialize bounded queues to provide backpressure and prevent memory growth
        username_queue = Queue(maxsize=ScraperConfig.USERNAME_QUEUE_MAXSIZE)
        bio_queue = Queue(maxsize=ScraperConfig.BIO_QUEUE_MAXSIZE)
        
        # Thread-safe results storage
        yes_results = []
        all_bios = []  # Store all processed bios
        results_lock = asyncio.Lock()
        
        # Control flags
        stop_event = asyncio.Event()
        
        # Statistics
        stats = {
            "total_scraped": 0,
            "total_fetched": 0,   # attempted bios (including empty)
            "total_nonempty": 0,  # non-empty bios
            "total_classified": 0,
            "total_yes": 0
        }

        cached_followers_page = await self._load_cached_followers_page(target)
        persisted_followers: List[str] = []
        persisted_followers_set: Set[str] = set()
        stored_next_cursor: Optional[str] = None
        rolling_latency: float = 0.0

        if cached_followers_page:
            cached_list = [
                handle.strip()
                for handle in cached_followers_page.get("followers", []) or []
                if isinstance(handle, str) and handle.strip()
            ]
            persisted_followers.extend(cached_list)
            persisted_followers_set.update(cached_list)
            stored_next_cursor = cached_followers_page.get("next_cursor") or None
            print(
                f"[cache] Loaded {len(cached_list)} cached followers for @{target}"
                + (" with next_cursor" if stored_next_cursor else "")
            )

        followers_page_state = {
            "dirty": False,
            "last_save_ts": 0.0,
        }

        def should_feed_followers() -> bool:
            return (
                username_queue.qsize() <= ScraperConfig.USERNAME_QUEUE_LOW_WATERMARK
                and bio_queue.qsize() <= ScraperConfig.BIO_QUEUE_LOW_WATERMARK
            )

        async def persist_followers_page_if_needed(force: bool = False) -> None:
            if not self.gcs_bucket:
                return
            now = time.time()
            if not force and not followers_page_state["dirty"]:
                return
            if not force and (now - followers_page_state["last_save_ts"]) < 5.0:
                return
            try:
                await self._save_followers_page(target, persisted_followers, stored_next_cursor)
                followers_page_state["dirty"] = False
                followers_page_state["last_save_ts"] = now
            except Exception as exc:
                print(f"⚠️ Failed to persist followers page for {target}: {exc}")
            else:
                print(f"[cache] persisted {len(persisted_followers)} cached followers for @{target} cursor={'set' if stored_next_cursor else 'none'}")

        async def _flush_followers_cache_final() -> None:
            await persist_followers_page_if_needed(force=True)

        # Progress meta update throttling and helpers
        last_meta_update_ts: float = 0.0
        last_yes_reported: int = -1
        started_at_epoch: int = int(time.time())
        effective_exec_id = self.exec_id or os.getenv("EXEC_ID") or os.getenv("SCRAPE_EXEC_ID") or ""
        prefix_for_meta: str = self._gcs_prefix(target, effective_exec_id) if effective_exec_id else self._gcs_prefix(target, self.exec_id or "")

        # cooldown handling disabled

        def _maybe_update_progress_meta(force: bool = False) -> None:
            nonlocal last_meta_update_ts, last_yes_reported
            try:
                now = time.time()
                yes_count = int(stats.get("total_yes", 0))
                # Update if forced, yes_count changed, or 5s elapsed
                if not force and yes_count == last_yes_reported and (now - last_meta_update_ts) < 5.0:
                    return
                meta_running = {
                    "status": "running",
                    "target": target,
                    "target_yes": int(target_yes),
                    "yes_count": yes_count,
                    "total_scraped": int(stats.get("total_scraped", 0)),
                    "total_fetched": int(stats.get("total_fetched", 0)),
                    "total_nonempty": int(stats.get("total_nonempty", 0)),
                    "total_classified": int(stats.get("total_classified", 0)),
                    "started_at": started_at_epoch,
                    "operation_id": self.exec_id,
                    "updated_at": int(now),
                }
                # Best-effort upload; ignore failures
                self._upload_json_to_gcs(meta_running, prefix_for_meta + "meta.json")
                last_meta_update_ts = now
                last_yes_reported = yes_count
            except Exception:
                # Never fail the scrape due to progress updates
                pass
        
        async def _resolve_user_id(page: Page, username: str) -> Optional[str]:
            """Resolve numeric user id for a username using the profile JSON or web_profile_info endpoint."""
            try:
                print(f"[api] resolving user id for {username} via profile page")
                await page.goto(f"https://www.instagram.com/{username}/", wait_until="domcontentloaded")
                handle = await page.query_selector('script[type="application/json"][id="__NEXT_DATA__"]')
                if handle:
                    try:
                        txt = await handle.text_content()
                        obj = json.loads(txt or "{}")
                        def find_id(o):
                            if isinstance(o, dict):
                                uname = o.get("username") or o.get("user_name")
                                if uname and str(uname).lower() == username.lower() and "id" in o:
                                    return o["id"]
                                for v in o.values():
                                    r = find_id(v)
                                    if r:
                                        return r
                            elif isinstance(o, list):
                                for it in o:
                                    r = find_id(it)
                                    if r:
                                        return r
                            return None
                        uid = find_id(obj)
                        if uid:
                            print(f"[api] resolved user id from profile JSON: {uid}")
                            return str(uid)
                    except Exception as exc:
                        print(f"[api] profile JSON parse failed; continuing to alternate lookup. error={exc}")
            except Exception as exc:
                print(f"[api] error loading profile page for id resolve: {exc}")

            # Query web_profile_info with cookies/headers for a direct lookup
            try:
                url = f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}"
                resp = await with_retry(
                    lambda: page.request.get(
                        url,
                        headers={
                            "X-IG-App-ID": ScraperConfig.IG_APP_ID,
                            "Referer": f"https://www.instagram.com/{username}/",
                        },
                    ),
                    limiter=self._rl_graphql,
                    what=f"web_profile_info for {username}",
                    hard_penalty_factor=ScraperConfig.RATE_LIMIT_PENALTY_FACTOR,
                    hard_penalty_seconds=ScraperConfig.RATE_LIMIT_PENALTY_SECONDS,
                    soft_penalty_factor=ScraperConfig.RATE_LIMIT_PENALTY_SOFT_FACTOR,
                    soft_penalty_seconds=ScraperConfig.RATE_LIMIT_PENALTY_SOFT_SECONDS,
                    on_429=lambda attempt: self._note_throttle("web_profile_info", attempt),
                            on_5xx=lambda attempt: self._note_server_error("web_profile_info", attempt),
                )
                if resp.status != 200:
                    try:
                        snippet = (await resp.text() or "")[:500].replace("\n", " ")
                    except Exception:
                        snippet = "<unreadable>"
                    print(f"[api] web_profile_info non-200: {resp.status} snippet={snippet}")
                    return None
                data = await resp.json()
                user_obj = (data.get("data", {}) or {}).get("user", {})
                uid = user_obj.get("id")
                if uid:
                    print(f"[api] resolved user id from web_profile_info: {uid}")
                    return str(uid)
            except Exception as exc:
                print(f"[api] error in web_profile_info lookup: {exc}")
            return None

        async def username_producer_api(user_id: str):
            """Produce usernames by paging Instagram's followers GraphQL API using page cookies.
            Caps total produced usernames at target_yes (API limit), batching by batch_size.
            """
            print("Calling API Consumer")
            nonlocal stored_next_cursor
            cache_drained_reported = False
            after = stored_next_cursor
            cache_count = len(persisted_followers)
            print(
                f"[api] cache entries available: {cache_count} stored_cursor={'set' if stored_next_cursor else 'none'}"
            )
            seen_handles: Set[str] = set()
            first = ScraperConfig.API_PAGE_SIZE
            # Buffer of usernames from the last fetched API page; we trickle them into the queue
            buffered_usernames: List[str] = []
            try:
                cached_idx = 0
                rolling_latency: float = 0.0
                paused = False
                last_pause_log = 0.0
                PAUSE_LOG_INTERVAL = 5.0

                def log_pause_once() -> None:
                    nonlocal paused, last_pause_log
                    now = time.monotonic()
                    if (not paused) or (now - last_pause_log) >= PAUSE_LOG_INTERVAL:
                        print(
                            f"[api][cache] paused feed: cached_idx={cached_idx} "
                            f"cache_len={len(persisted_followers)} "
                            f"username_qsize={username_queue.qsize()} "
                            f"bio_qsize={bio_queue.qsize()} "
                            f"(low_wm_u={ScraperConfig.USERNAME_QUEUE_LOW_WATERMARK}, "
                            f"low_wm_b={ScraperConfig.BIO_QUEUE_LOW_WATERMARK})"
                        )
                        paused = True
                        last_pause_log = now

                def log_resume_once() -> None:
                    nonlocal paused
                    if paused:
                        print(f"[api][cache] resumed feed at idx={cached_idx}")
                        paused = False

                async def _feed_cached() -> bool:
                    nonlocal cached_idx, cache_drained_reported
                    added = False
                    while cached_idx < len(persisted_followers):
                        if not should_feed_followers():
                            break
                        chunk = persisted_followers[cached_idx:cached_idx + batch_size]
                        if not chunk:
                            break
                        cached_idx += len(chunk)
                        print(
                            f"[api][cache] enqueued {len(chunk)} cached usernames (remaining={len(persisted_followers) - cached_idx})"
                        )
                        for username_val in chunk:
                            await username_queue.put(username_val)
                            stats["total_scraped"] += 1
                            added = True
                    if not added and cached_idx >= len(persisted_followers) and not cache_drained_reported:
                        print("[api][cache] cache exhausted; waiting on fresh API data")
                        cache_drained_reported = True
                    return added

                while not stop_event.is_set():
                    if not should_feed_followers():
                        log_pause_once()
                        await asyncio.sleep(0.5)
                        continue

                    log_resume_once()

                    while buffered_usernames and not stop_event.is_set():
                        if not should_feed_followers():
                            log_pause_once()
                            break
                        username_val = buffered_usernames.pop(0)
                        await username_queue.put(username_val)
                        stats["total_scraped"] += 1

                    if stop_event.is_set():
                        break

                    if buffered_usernames:
                        await asyncio.sleep(0)
                        continue

                    if not should_feed_followers():
                        log_pause_once()
                        await asyncio.sleep(0.5)
                        continue

                    if await _feed_cached():
                        await asyncio.sleep(0)
                        continue

                    if cached_idx >= len(persisted_followers) and not cache_drained_reported:
                        print(
                            f"[api][cache] no cached followers available (cached_idx={cached_idx})"
                        )
                        cache_drained_reported = True

                    if not should_feed_followers():
                        log_pause_once()
                        await asyncio.sleep(0.5)
                        continue

                    variables = {
                        "id": user_id,
                        "include_reel": True,
                        "fetch_mutual": False,
                        "first": first,
                    }
                    if after:
                        variables["after"] = after
                    print("Using API to fetch followers")
                    url = (
                        f"https://www.instagram.com/graphql/query/?query_hash={ScraperConfig.FOLLOWERS_QUERY_HASH}"
                        f"&variables={json.dumps(variables, separators=(',',':'))}"
                    )
                    print(f"[api] GraphQL GET: size={first} after={'yes' if after else 'no'} qsize={username_queue.qsize()}" )
                    self._log_latency("graphql", rolling_latency)
                    try:
                        start_req = time.monotonic()
                        resp = await with_retry(
                            lambda: followers_page.request.get(
                                url,
                                headers={
                                    "Referer": f"https://www.instagram.com/{target}/",
                                    "X-IG-App-ID": ScraperConfig.IG_APP_ID
                                },
                            ),
                            limiter=self._rl_graphql,
                            what="GraphQL followers page",
                            hard_penalty_factor=ScraperConfig.RATE_LIMIT_PENALTY_FACTOR,
                            hard_penalty_seconds=ScraperConfig.RATE_LIMIT_PENALTY_SECONDS,
                            soft_penalty_factor=ScraperConfig.RATE_LIMIT_PENALTY_SOFT_FACTOR,
                            soft_penalty_seconds=ScraperConfig.RATE_LIMIT_PENALTY_SOFT_SECONDS,
                            on_429=lambda attempt: self._note_throttle("graphql", attempt),
                            on_5xx=lambda attempt: self._note_server_error("graphql", attempt),
                        )
                        duration = time.monotonic() - start_req
                        rolling_latency = self._update_latency_metric(
                            rolling_latency,
                            duration,
                        )
                    except Exception as api_error:
                        print(f"[api] ❌ Error calling GraphQL API: {api_error}")
                        await persist_followers_page_if_needed(force=True)
                        await self.notification_service.send_notification(
                            f"GraphQL followers fetch failed for @{target}: {api_error}",
                            "Instagram Scraper - GraphQL Error",
                        )
                        raise

                    try:
                        data = await resp.json()
                    except json.JSONDecodeError as decode_err:
                        try:
                            raw_body = await resp.text()
                            snippet = (raw_body or "")[:500].replace("\n", " ")
                        except Exception:
                            snippet = "<unreadable>"
                        print(
                            f"[api] ❌ GraphQL JSON decode error: {decode_err} body_snippet={snippet}"
                        )
                        await persist_followers_page_if_needed(force=True)
                        raise RuntimeError("GraphQL returned a non-JSON response") from decode_err
                        
                    try:
                        edge_followed_by = data["data"]["user"]["edge_followed_by"]
                        edges = edge_followed_by["edges"]
                        page_info = edge_followed_by["page_info"]
                    except Exception as exc:
                        try:
                            snippet = json.dumps(data)[:500]
                        except Exception:
                            snippet = str(data)[:500]
                        print(f"[api] ❌ JSON structure error: {exc} data={snippet}")
                        await persist_followers_page_if_needed(force=True)
                        raise RuntimeError("Unexpected GraphQL response structure") from exc

                    if not edges:
                        print("[api] no edges returned from GraphQL")
                        # Don't trigger password change for empty edges - might be end of followers
                        raise RuntimeError("GraphQL returned no edges")

                    # Fill buffer with the entire page (deduped)
                    for edge in edges:
                        node = edge.get("node") or {}
                        username_val = node.get("username")
                        if username_val and username_val not in seen_handles:
                            seen_handles.add(username_val)
                            if username_val not in persisted_followers_set:
                                persisted_followers.append(username_val)
                                persisted_followers_set.add(username_val)
                                followers_page_state["dirty"] = True
                            buffered_usernames.append(username_val)

                    if not page_info.get("has_next_page") or not page_info.get("end_cursor"):
                        print("[api] end of followers pages; stopping fetch; buffer will be drained")
                        # No more pages; we'll exit loop after buffer drains
                        after = None
                        stored_next_cursor = None
                        # Keep looping to drain buffer
                    else:
                        after = page_info["end_cursor"]
                        stored_next_cursor = after
                        followers_page_state["dirty"] = True
                    await asyncio.sleep(0.2)

            except Exception as e:
                print(f"[api] username producer error: {e}")
                stop_event.set()
                raise
            finally:
                # Do not prematurely signal end; let shutdown phase send sentinels
                pass
        
        async def bio_worker(worker_id: int, page: Page, context):
            """Worker that consumes usernames and fetches bios one at a time."""
            try:
                while True:
                    try:
                        username = await asyncio.wait_for(username_queue.get(), timeout=2.0)
                    except asyncio.TimeoutError:
                        if stop_event.is_set():
                            break
                        continue

                    if username is None:
                        username_queue.task_done()
                        break

                    try:
                        if not page or page.is_closed():
                            if context and hasattr(context, "is_closed") and not context.is_closed():
                                page = await context.new_page()
                                if STEALTH_AVAILABLE:
                                    try:
                                        await stealth_async(page)
                                    except Exception as e:
                                        print(f"[bio-worker {worker_id}] stealth setup failed: {e}")
                                if worker_id < len(bio_pages):
                                    bio_pages[worker_id] = page
                                else:
                                    bio_pages.append(page)
                            else:
                                print(f"[bio-worker {worker_id}] context closed; stopping")
                                stop_event.set()
                                break

                        active_pages = await self._ensure_bio_pages_alive(context, bio_pages)
                        bio_pages[:] = active_pages
                        if not active_pages:
                            print(f"[bio-worker {worker_id}] no bio pages available; stopping")
                            stop_event.set()
                            break
                        if page not in active_pages or not page:
                            idx = worker_id if worker_id < len(active_pages) else 0
                            page = active_pages[idx]

                        bio = await self._get_bio_api_first(page, username)
                        stats["total_fetched"] += 1
                        if bio:
                            stats["total_nonempty"] += 1
                            await bio_queue.put({"username": username, "bio": bio})
                        else:
                            print(f"[bio] empty bio returned for {username}")

                        async with results_lock:
                            all_bios.append({
                                "username": username,
                                "url": f"https://www.instagram.com/{username}/",
                                "bio": bio or "",
                            })

                    except Exception as e:
                        print(f"[bio-worker {worker_id}] error for {username}: {e}")
                    finally:
                        username_queue.task_done()
            finally:
                print(f"[bio-worker {worker_id}] exiting")
        
        async def bio_classifier():
            """Continuously classify bios from the queue."""
            client = httpx.AsyncClient(timeout=ScraperConfig.HTTPX_LONG_TIMEOUT)
            batch_buffer: List[Dict] = []

            async def _flush_batch(reason: str = "") -> None:
                nonlocal batch_buffer
                if not batch_buffer:
                    return
                if reason:
                    print(
                        f"[classifier] starting batch (size={len(batch_buffer)}) reason={reason}"
                    )
                try:
                    await asyncio.wait_for(
                        process_classification_batch(batch_buffer, client, criteria_text),
                        timeout=30,
                    )
                    print("[classifier] finished batch")
                except asyncio.TimeoutError:
                    print("[classifier] batch timed out; skipping")
                finally:
                    batch_buffer = []

            try:
                loop = asyncio.get_running_loop()

                while True:
                    if stop_event.is_set() and bio_queue.qsize() == 0 and not batch_buffer:
                        print("[classifier] stop_event set and queue empty; exiting")
                        break

                    # 1) Wait for the FIRST item OR stop, whichever comes first
                    bio_get = asyncio.create_task(bio_queue.get())
                    stop_wait = asyncio.create_task(stop_event.wait())
                    done, pending = await asyncio.wait(
                        {bio_get, stop_wait},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if stop_wait in done:
                        print("[classifier] stop_event set during wait; flushing and exiting")
                        bio_get.cancel()
                        await _flush_batch("stop")
                        break

                    bio = bio_get.result()

                    if bio is None:
                        print("[classifier] got sentinel; processing remaining batch and exiting")
                        await _flush_batch("sentinel")
                        stop_event.set()
                        break

                    username = bio.get("username") if isinstance(bio, dict) else "<unknown>"
                    batch_buffer = [bio]
                    print(
                        f"[classifier] dequeued bio for {username} queue_size_after_get={bio_queue.qsize()}"
                    )
                    print(
                        f"[classifier] batch size now {len(batch_buffer)}; usernames={[b.get('username') for b in batch_buffer]}"
                    )

                    target = ScraperConfig.CLASSIFICATION_CHUNK_SIZE
                    deadline = loop.time() + ScraperConfig.CLASS_MAX_WAIT_SEC
                    min_required = ScraperConfig.CLASS_MIN_CHUNK_SIZE

                    saw_sentinel = False

                    # Drain immediate queue contents
                    while len(batch_buffer) < target:
                        try:
                            nxt = bio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if nxt is None:
                            saw_sentinel = True
                            break
                        batch_buffer.append(nxt)

                    # Wait up to the deadline for more items
                    while not saw_sentinel and len(batch_buffer) < target:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            break
                        try:
                            nxt = await asyncio.wait_for(bio_queue.get(), timeout=remaining)
                        except asyncio.TimeoutError:
                            break
                        if nxt is None:
                            saw_sentinel = True
                            break
                        batch_buffer.append(nxt)

                    full = len(batch_buffer) >= target
                    min_ok = len(batch_buffer) >= min_required
                    time_up = loop.time() >= deadline

                    if full or saw_sentinel or time_up or min_ok:
                        reason_parts = []
                        if full:
                            reason_parts.append("full")
                        if time_up and not full:
                            reason_parts.append("deadline")
                        if saw_sentinel:
                            reason_parts.append("sentinel")
                        if not reason_parts:
                            reason_parts.append("min")
                        await _flush_batch("+".join(reason_parts))

                        if saw_sentinel:
                            print("[classifier] sentinel observed during coalesce; stopping")
                            stop_event.set()
                            break

            except Exception as e:
                print(f"Bio classifier error: {e}")
            finally:
                await client.aclose()
        
        async def process_classification_batch(bios: List[Dict], client, criteria_text_param: Optional[str]):
            """Process a batch of bios for classification."""
            try:
                # Extract just the bio text for classification
                bio_texts = [b["bio"] for b in bios]
                usernames = [b.get("username") for b in bios]
                print(f"[classifier] process_classification_batch usernames={usernames}")

                # Classify the batch
                print(f"🧪 [DEBUG] process_classification_batch: size={len(bio_texts)} custom_criteria={bool(criteria_text_param and criteria_text_param.strip())}")
                flags = await self.bio_classifier.classify_bios(bio_texts, client, criteria_text=criteria_text_param)
                print(f"[classifier] processed batch of {len(bio_texts)} bios; got {len(flags)} flags")

                # Process results
                async with results_lock:
                    for i, (bio, flag) in enumerate(zip(bios, flags)):
                        stats["total_classified"] += 1

                        if flag and flag.isdigit() and int(flag) == i:
                            yes_results.append({
                                "username": bio["username"],
                                "url": f"https://www.instagram.com/{bio['username']}/",
                                "bio": bio["bio"]
                            })
                            stats["total_yes"] += 1
                            print(f"✅ Found match #{stats['total_yes']}: {bio['username']}")

                            # Check if we've reached our target
                            if stats["total_yes"] >= target_yes:
                                print(f"[stop] classifier: target met ({stats['total_yes']}/{target_yes}); setting stop_event")
                                stop_event.set()
                                # Final progress update before returning
                                _maybe_update_progress_meta(force=True)
                                return

                # Update running progress after processing batch
                _maybe_update_progress_meta(force=False)

            except Exception as e:
                print(f"Classification batch error: {e}")
        
        # Resolve user id for API-based producer
        user_id_val: Optional[str] = await _resolve_user_id(followers_page, target)

        # Start all tasks concurrently with names for better debugging
        tasks = []
        if user_id_val:
            tasks.append(asyncio.create_task(username_producer_api(user_id_val), name="username_producer_api"))
        else:
            raise RuntimeError("Unable to resolve target user id for API scraping")
        for i, bio_page in enumerate(bio_pages):
            tasks.append(asyncio.create_task(bio_worker(i, bio_page, followers_page.context), name=f"bio_worker_{i}"))
        tasks.append(asyncio.create_task(bio_classifier(), name="bio_classifier"))

        start_time = time.perf_counter()

        try:
            # Wait until classifier signals stop_event (target_yes met) or timeout
            wait_task = asyncio.create_task(stop_event.wait(), name="stop_event_wait")
            done, pending = await asyncio.wait(
                {wait_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=ScraperConfig.SCRAPE_MAX_SECONDS,
            )
            print(f"🔄 Stop condition met or timeout; proceeding to shutdown")

        except asyncio.TimeoutError:
            still_pending = [t.get_name() for t in tasks if not t.done()]
            print(f"⏰ Scraping timeout reached; pending tasks: {still_pending}")
        finally:
            # Ensure shutdown is prompt and graceful
            stop_event.set()

            # Proactively send sentinels to unblock queues in case producers/consumers are waiting
            for _ in range(len(bio_pages)):
                try:
                    username_queue.put_nowait(None)
                except Exception:
                    pass
            try:
                bio_queue.put_nowait(None)
            except Exception:
                pass

            # Allow fetcher to drain remaining usernames into bios and classifier to finish
            done2, pending2 = await asyncio.wait(tasks, timeout=30)
            for t in pending2:
                t.cancel()
                print(f"[cancel] cancelled task {t.get_name()}")
            # Ensure all tasks are gathered
            await asyncio.gather(*tasks, return_exceptions=True)
            print("[shutdown] all tasks have been gathered")

            # Also cancel the wait_task if it still exists
            try:
                if 'wait_task' in locals() and not wait_task.done():
                    wait_task.cancel()
                    print("[cancel] cancelled wait_task")
            except Exception:
                pass

            # Ensure latest follower cache state is persisted before saving results
            try:
                await _flush_followers_cache_final()
            except Exception as exc:
                print(f"⚠️ Failed final follower cache flush: {exc}")

            # Calculate elapsed time
            elapsed_time = time.perf_counter() - start_time
            formatted_time = self.format_duration(elapsed_time)

            # Print final statistics
            print(f"\n📊 Final Statistics:")
            print(f"  • Total usernames scraped: {stats['total_scraped']}")
            print(f"  • Total bios fetched (attempted): {stats['total_fetched']}")
            print(f"  • Total non-empty bios: {stats['total_nonempty']}")
            print(f"  • Total bios classified: {stats['total_classified']}")
            print(f"  • Total matches found: {stats['total_yes']}")
            print(f"  • Queue status: usernames={username_queue.qsize()} bios={bio_queue.qsize()}")
            print(f"  • Time elapsed: {formatted_time}")

            # Save results with all processed bios
            timeout_seconds = ScraperConfig.SCRAPE_MAX_SECONDS
            await self._save_results(all_bios, yes_results, target, target_yes, start_time, timeout_seconds)

        return yes_results

    async def scrape_followers(
        self,
        browser: Browser,
        state_path: Path,
        target: str,
        target_yes: int = 10,
        batch_size: int = 20,
        num_bio_pages: int = 3,
        criteria_text: Optional[str] = None,
    ) -> List[Dict]:
        """
        Scrape followers from a target Instagram account.
        
        Args:
            browser: Playwright browser instance
            state_path: Path to browser state file
            target: Target Instagram username
            target_yes: Number of "yes" classifications to target
            batch_size: Size of bio processing batches
            num_bio_pages: Number of parallel bio pages to use
            
        Returns:
            List of dictionaries with follower data
        """
        context = await browser.new_context(
            storage_state=state_path,
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (X11; Linux x86_64)",
            record_video_dir="videos/"
        )

        async def _request_shaper(route, request):
            resource_type = request.resource_type
            if resource_type in {"image", "media", "font", "stylesheet"}:
                return await route.abort()
            return await route.continue_()

        await context.route("**/*", _request_shaper)
        
        # Create pages
        followers_page = await context.new_page()
        bio_pages = [await context.new_page() for _ in range(num_bio_pages)]
        
        # Apply stealth measures if available
        if STEALTH_AVAILABLE:
            await stealth_async(followers_page)
            for bio_page in bio_pages:
                await stealth_async(bio_page)
        
        try:
            # Pre-create artifacts only if an exec_id was NOT provided by the orchestrator
            # When EXEC_ID is passed via env/argument, the backend already created START/meta
            if not self.exec_id:
                self._ensure_start_artifacts(target=target, target_yes=target_yes, started_at_epoch=int(time.time()))
            return await self._scrape_followers_parallel(
                context, followers_page, bio_pages, target, target_yes, batch_size, criteria_text
            )
        finally:
            await self._cleanup(context, followers_page)
    
    
    async def _save_results(
        self, 
        all_scraped_bios: List[Dict], 
        yes_rows: List[Dict], 
        target: str, 
        target_yes: int,
        start_time: float, 
        timeout_seconds: int
    ) -> None:
        """Save scraping results to CSV/JSON, upload artifacts, and send notifications."""
        # Use provided exec_id if available; otherwise generate one
        if not self.exec_id:
            self.exec_id = f"op_{int(time.time())}"
        operation_id = self.exec_id
        prefix = self._gcs_prefix(target, operation_id)

        # CSV saving disabled by request; keep meta keys as None
        if all_scraped_bios:
            print(f"📊 Total bios scraped: {len(all_scraped_bios)}")
            print(f"📊 Total bios classified as YES: {len(yes_rows)}")
        
        # Build JSON artifacts (results + meta), then upload a DONE marker last as the completion signal
        finished_at_epoch = int(time.time())
        elapsed_time = time.perf_counter() - start_time
        duration_seconds = float(elapsed_time)

        # Results JSON (list of yes rows)
        results_blob = prefix + "results.json"
        results_uri = self._upload_json_to_gcs(yes_rows, results_blob)
        if results_uri:
            print(f"📝 results.json uploaded to: {results_uri}")

        # Meta JSON (status + pointers)
        meta = {
            "status": "completed",  # this reflects scraper completion of save phase
            "target": target,
            "target_yes": target_yes,
            "yes_count": len(yes_rows),
            "finished_at": finished_at_epoch,
            "duration_seconds": duration_seconds,
            "operation_id": operation_id,
        }
        meta_blob = prefix + "meta.json"
        meta_uri = self._upload_json_to_gcs(meta, meta_blob)
        if meta_uri:
            print(f"📝 meta.json uploaded to: {meta_uri}")

        # DONE marker (must be uploaded LAST)
        done_blob = prefix + "DONE"
        done_uri = self._upload_json_to_gcs({"ok": True, "ts": finished_at_epoch}, done_blob)
        if done_uri:
            print(f"✅ DONE marker uploaded to: {done_uri}")

        # Send notifications
        formatted_time = self.format_duration(elapsed_time)
        if len(yes_rows) < target_yes and elapsed_time >= (timeout_seconds - 30):
            print(f"⚠️ Returning partial results: {len(yes_rows)}/{target_yes} (timeout reached after {formatted_time})")
            await self.notification_service.send_notification(
                f"Partial results: {len(yes_rows)}/{target_yes} followers found for @{target} in {formatted_time}",
                "Instagram Scraper - Partial Results"
            )
        else:
            print(f"✅ Completed successfully: {len(yes_rows)}/{target_yes} results in {formatted_time}")
            await self.notification_service.send_notification(
                f"Successfully found {len(yes_rows)}/{target_yes} followers for @{target} in {formatted_time}",
                "Instagram Scraper - Complete"
            )
    
    async def _cleanup(self, context, followers_page: Page) -> None:
        """Clean up resources and save final screenshot."""
        os.makedirs("shots", exist_ok=True)
        not_visible_path = "shots/not_visible.png"
        try:
            if followers_page and not followers_page.is_closed():
                await followers_page.screenshot(path=not_visible_path, full_page=True)
            else:
                print("[cleanup] page already closed; skipping screenshot")
        except (PlayTimeout, PlaywrightError) as e:
            print(f"[cleanup] screenshot skipped ({type(e).__name__}: {e})")

        try:
            if self.gcs_bucket and os.path.exists(not_visible_path):
                self.csv_exporter.upload_to_gcs(local_path=not_visible_path, destination_blob=not_visible_path)
        except Exception as e:
            print(f"[cleanup] upload skip/error: {e}")

        try:
            if context:
                is_closed_method = getattr(context, "is_closed", None)
                should_close = True
                if callable(is_closed_method):
                    try:
                        should_close = not context.is_closed()
                    except PlaywrightError:
                        should_close = False
                if should_close:
                    await context.close()
                else:
                    print("[cleanup] context already closed; nothing to do")
        except PlaywrightError as e:
            print(f"[cleanup] context close ignored ({type(e).__name__}: {e})")


# Backward compatibility function
async def scrape_followers(
    browser: Browser,
    state_path: Path,
    target: str,
    target_yes: int = 10,
    batch_size: int = 20,
    num_bio_pages: int = 3,
    criteria_text: Optional[str] = None,
) -> List[Dict]:
    """
    Backward compatibility function for the old API.
    
    This function creates a scraper instance and calls the new method.
    """
    scraper = InstagramScraper()
    return await scraper.scrape_followers(
        browser, state_path, target, target_yes, batch_size, num_bio_pages, criteria_text
    )
