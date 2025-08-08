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
import math
import os
import time
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Set

import httpx
from playwright.async_api import Browser, Page, TimeoutError as PlayTimeout

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

    def _gcs_prefix(self, target: str, operation_id: str) -> str:
        """Stable prefix for result artifacts in GCS."""
        # e.g., scrapes/<target>/<op_1722980000>/
        return f"scrapes/{target}/{operation_id}/"

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
    
    async def _get_bio(self, page: Page, username: str) -> str:
        """Extract bio from a user's profile page."""
        try:
            await page.goto(
                f"https://www.instagram.com/{username}/",
                timeout=ScraperConfig.BIO_PAGE_TIMEOUT_MS,
                wait_until="domcontentloaded",
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
                    
        except PlayTimeout:
            return ""
        except Exception:
            return ""

        try:
            desc = await page.get_attribute("head meta[name='description']", "content")
            if desc and " on Instagram: " in desc:
                return desc.split(" on Instagram: ")[1].strip().strip('"')
        except Exception:
            pass
        return ""
    
    async def _get_bios_for_chunk(self, bio_page: Page, usernames: List[str]) -> List[Dict]:
        """Fetch bios for a chunk of usernames using a single bio page."""
        bios = []
        for username in usernames:
            bio = await self._get_bio(bio_page, username)
            bios.append({"username": username, "bio": bio})
        return bios
    
    async def _get_bios_parallel(self, bio_pages: List[Page], usernames: List[str]) -> List[Dict]:
        """Fetch bios for multiple usernames in parallel using multiple bio pages."""
        if not usernames:
            return []
        
        num_bio_pages = len(bio_pages)
        
        # Split usernames across bio pages (more even distribution)
        chunk_size = max(1, math.ceil(len(usernames) / num_bio_pages))
        chunks_list = [usernames[i:i + chunk_size] for i in range(0, len(usernames), chunk_size)]
        
        # Create tasks for each chunk with proper error handling
        tasks = []
        for i, chunk in enumerate(chunks_list):
            if chunk:
                task = asyncio.create_task(
                    self._get_bios_for_chunk(bio_pages[i % num_bio_pages], chunk)
                )
                tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle any exceptions with proper chunk sizes
        all_bios = []
        for result, chunk in zip(results, chunks_list):
            if isinstance(result, Exception):
                print(f"⚠️ Error in bio fetching: {result}")
                # Skip this chunk; do not enqueue empty placeholders
                continue
            else:
                all_bios.extend(result)
        
        return all_bios
    
    async def _process_bio_batch(
        self, 
        bios: List[Dict], 
        client: httpx.AsyncClient
    ) -> List[Dict]:
        """Process a batch of bios through validation and classification."""
        # Validate and clean bios
        valid_bios = []
        for bio in bios:
            cleaned_bio = BioValidator.validate_and_clean_bio(bio)
            if cleaned_bio:
                valid_bios.append(cleaned_bio)
        
        # Log filtering results
        original_count = len(bios)
        filtered_count = len(valid_bios)
        if original_count != filtered_count:
            print(f"📊 Bio filtering: {original_count} -> {filtered_count} valid bios (removed {original_count - filtered_count} invalid)")
        
        if not valid_bios:
            return []
        
        # Process bios in chunks for classification
        all_flags = []
        chunk_success_count = 0
        
        for i in range(0, len(valid_bios), ScraperConfig.CLASSIFICATION_CHUNK_SIZE):
            chunk = valid_bios[i:i + ScraperConfig.CLASSIFICATION_CHUNK_SIZE]
            chunk_num = i // ScraperConfig.CLASSIFICATION_CHUNK_SIZE + 1
            
            try:
                chunk_flags = await self.bio_classifier.classify_bios(
                    [b["bio"] for b in chunk], client
                )
                all_flags.extend(chunk_flags)
                chunk_success_count += 1
            except Exception as e:
                print(f"⚠️ Chunk {chunk_num} classification failed: {e}")
                all_flags.extend([""] * len(chunk))
        
        print(f"✅ Successfully classified {chunk_success_count} chunks")
        
        # Process classification results using chunk-based tracking
        yes_idx = set()
        
        # Calculate how many complete chunks we have
        total_chunks = (len(valid_bios) + ScraperConfig.CLASSIFICATION_CHUNK_SIZE - 1) // ScraperConfig.CLASSIFICATION_CHUNK_SIZE
        
        for chunk_index in range(total_chunks):
            # Calculate the global offset for this chunk
            chunk_offset = chunk_index * ScraperConfig.CLASSIFICATION_CHUNK_SIZE
            
            # Calculate actual chunk size (handles last chunk that might be smaller)
            chunk_start_pos = chunk_index * ScraperConfig.CLASSIFICATION_CHUNK_SIZE
            chunk_end_pos = min(chunk_start_pos + ScraperConfig.CLASSIFICATION_CHUNK_SIZE, len(valid_bios))
            actual_chunk_size = chunk_end_pos - chunk_start_pos
            
            # Get flags for this chunk
            flags_start = chunk_index * ScraperConfig.CLASSIFICATION_CHUNK_SIZE
            flags_end = flags_start + actual_chunk_size
            chunk_flags = all_flags[flags_start:flags_end]
            
            # Process each flag in the chunk
            for flag in chunk_flags:  # Fixed: removed enumerate
                if flag and flag.isdigit():
                    # Convert local index to global index using chunk offset
                    global_idx = chunk_offset + int(flag)
                    if 0 <= global_idx < len(valid_bios):
                        yes_idx.add(global_idx)
        
        # Return "yes" results
        yes_results = []
        for idx in yes_idx:
            if (idx < len(valid_bios) and 
                valid_bios[idx]["bio"] and 
                valid_bios[idx]["username"]):
                yes_results.append({
                    "username": valid_bios[idx]["username"],
                    "url": f"https://www.instagram.com/{valid_bios[idx]['username']}/",
                    "bio": valid_bios[idx]["bio"]
                })
        
        return yes_results
    
    async def _scrape_followers_parallel(
        self,
        context,
        followers_page: Page,
        bio_pages: List[Page],
        target: str,
        target_yes: int,
        batch_size: int,
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
            "total_fetched": 0,
            "total_classified": 0,
            "total_yes": 0
        }
        
        async def username_producer():
            """Continuously scrape usernames from the followers dialog."""
            try:
                # Navigate to target profile
                await followers_page.goto(f"https://www.instagram.com/{target}/")

                # Open followers dialog
                link = followers_page.locator('header a[href*="/followers"]')
                followers_page.set_default_timeout(25_000)

                await followers_page.wait_for_selector('header a[href*="/followers"]', state="attached")
                await link.wait_for(state="visible")
                await link.click()
                await followers_page.wait_for_selector(
                    'div[role="dialog"]',
                    timeout=ScraperConfig.DIALOG_SELECTOR_TIMEOUT_MS,
                )

                dialog = followers_page.locator('div[role="dialog"]').last
                first_link = dialog.locator('a[href^="/"]').first
                await first_link.wait_for(
                    state="attached",
                    timeout=ScraperConfig.FIRST_LINK_WAIT_TIMEOUT_MS,
                )

                user_links = dialog.locator('a[href^="/"]')

                seen_handles = set()
                idle_loops = 0
                previous_count = 0

                while not stop_event.is_set():
                    # Collect visible handles
                    current_batch = []
                    for h in await user_links.all_inner_texts():
                        h = h.strip()
                        if h and h not in seen_handles:
                            seen_handles.add(h)
                            current_batch.append(h)

                            # Put usernames in queue as we find them
                            if len(current_batch) >= batch_size:
                                # Backpressure: await if the queue is full
                                await username_queue.put(current_batch[:batch_size])
                                current_batch = current_batch[batch_size:]
                                stats["total_scraped"] += batch_size

                    # Check progress
                    new_total = len(seen_handles)
                    if new_total == previous_count:
                        idle_loops += 1
                        if idle_loops >= ScraperConfig.MAX_IDLE_LOOPS:
                            print(f"[stop] producer: no new followers after {ScraperConfig.MAX_IDLE_LOOPS} scrolls; setting stop_event and exiting")
                            # Flush any remaining current_batch before exiting
                            if current_batch:
                                await username_queue.put(current_batch)
                                stats["total_scraped"] += len(current_batch)
                                current_batch = []
                            stop_event.set()
                            break
                    else:
                        idle_loops = 0
                    previous_count = new_total

                    # Scroll
                    try:
                        await user_links.nth(-1).scroll_into_view_if_needed()
                    except PlayTimeout:
                        print("[stop] producer: scroll timeout - likely end of list; setting stop_event and exiting")
                        # Flush any remaining current_batch before exiting
                        if current_batch:
                            await username_queue.put(current_batch)
                            stats["total_scraped"] += len(current_batch)
                            current_batch = []
                        stop_event.set()
                        break

                    # Dynamic wait
                    wait_time = ScraperConfig.IDLE_SCROLL_WAIT if idle_loops > 0 else ScraperConfig.BASE_SCROLL_WAIT
                    await asyncio.sleep(wait_time)

                    # Put remaining batch if any
                    if current_batch:
                        await username_queue.put(current_batch)
                        stats["total_scraped"] += len(current_batch)

            except Exception as e:
                print(f"Username producer error: {e}")
            finally:
                # Signal end of production
                await username_queue.put(None)
                print("[producer] sent sentinel None to username_queue")
        
        async def bio_fetcher():
            """Continuously fetch bios for usernames from the queue."""
            try:
                while True:
                    try:
                        usernames = await asyncio.wait_for(username_queue.get(), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue

                    if usernames is None:  # End signal
                        print("[fetcher] got sentinel from producer; exiting fetch loop")
                        break

                    try:
                        # Fetch bios in parallel using multiple pages
                        bios = await self._get_bios_parallel(bio_pages, usernames)

                        # Record all scraped bios (including empty bios) for parity with linear path
                        if bios:
                            async with results_lock:
                                for b in bios:
                                    if b and b.get("username"):
                                        all_bios.append({
                                            "username": b["username"],
                                            "url": f"https://www.instagram.com/{b['username']}/",
                                            "bio": b.get("bio", "")
                                        })

                        # Put only non-empty bios into the classification queue
                        for bio in bios:
                            if bio and bio.get("bio"):
                                # Backpressure: await if the queue is full
                                await bio_queue.put(bio)
                                stats["total_fetched"] += 1

                    except Exception as e:
                        print(f"Bio fetch error: {e}")
                    finally:
                        username_queue.task_done()

            except Exception as e:
                print(f"Bio fetcher error: {e}")
            finally:
                # Signal end of fetching
                await bio_queue.put(None)
                print("[fetcher] sent sentinel None to classifier")
        
        async def bio_classifier():
            """Continuously classify bios from the queue."""
            client = httpx.AsyncClient(timeout=ScraperConfig.HTTPX_LONG_TIMEOUT)
            batch_buffer = []

            try:
                while True:
                    # Collect bios for batch classification
                    try:
                        bio = await asyncio.wait_for(bio_queue.get(), timeout=ScraperConfig.BATCH_FLUSH_TIMEOUT_SEC)
                    except asyncio.TimeoutError:
                        # Process partial batch if we have any
                        if batch_buffer:
                            await process_classification_batch(batch_buffer, client)
                            batch_buffer = []
                        continue

                    if bio is None:  # End signal
                        print("[classifier] got sentinel; processing remaining batch and exiting")
                        if batch_buffer:
                            await process_classification_batch(batch_buffer, client)
                        break

                    batch_buffer.append(bio)

                    # Process when we have a full batch
                    if len(batch_buffer) >= ScraperConfig.CLASSIFICATION_CHUNK_SIZE:
                        await process_classification_batch(batch_buffer, client)
                        batch_buffer = []

            except Exception as e:
                print(f"Bio classifier error: {e}")
            finally:
                await client.aclose()
        
        async def process_classification_batch(bios: List[Dict], client):
            """Process a batch of bios for classification."""
            try:
                # Extract just the bio text for classification
                bio_texts = [b["bio"] for b in bios]

                # Classify the batch
                flags = await self.bio_classifier.classify_bios(bio_texts, client)
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
                                return

            except Exception as e:
                print(f"Classification batch error: {e}")
        
        # Start all tasks concurrently with names for better debugging
        tasks = [
            asyncio.create_task(username_producer(), name="username_producer"),
            asyncio.create_task(bio_fetcher(), name="bio_fetcher"),
            asyncio.create_task(bio_classifier(), name="bio_classifier"),
        ]

        start_time = time.perf_counter()

        try:
            # Wait for EITHER target is met OR workers finish - whichever comes first
            wait_task = asyncio.create_task(stop_event.wait(), name="stop_event_wait")
            done, pending = await asyncio.wait(
                {wait_task, *tasks},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=ScraperConfig.SCRAPE_MAX_SECONDS,
            )

            print(f"🔄 Done: {[t.get_name() for t in done]} | Pending: {[t.get_name() for t in pending]}")

        except asyncio.TimeoutError:
            still_pending = [t.get_name() for t in tasks if not t.done()]
            print(f"⏰ Scraping timeout reached; pending tasks: {still_pending}")
        finally:
            # Ensure shutdown is prompt and graceful
            stop_event.set()

            # Let the fetcher and classifier finish draining queues; then cancel if needed
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

            # Calculate elapsed time
            elapsed_time = time.perf_counter() - start_time
            formatted_time = self.format_duration(elapsed_time)

            # Print final statistics
            print(f"\n📊 Final Statistics:")
            print(f"  • Total usernames scraped: {stats['total_scraped']}")
            print(f"  • Total bios fetched: {stats['total_fetched']}")
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
                context, followers_page, bio_pages, target, target_yes, batch_size
            )
        finally:
            await self._cleanup(context, followers_page)
    
    async def _scrape_followers_impl(
        self,
        context,
        followers_page: Page,
        bio_pages: List[Page],
        target: str,
        target_yes: int,
        batch_size: int,
    ) -> List[Dict]:
        """Main scraping implementation."""
        # Navigate to target profile
        t0 = time.perf_counter()
        await followers_page.goto(f"https://www.instagram.com/{target}/")
        nav_time = time.perf_counter() - t0
        print(f"🏁 page.goto took {nav_time*1000:.0f} ms")
        
        # Open followers dialog
        link = followers_page.locator('header a[href*="/followers"]')
        followers_page.set_default_timeout(25_000)

        await followers_page.wait_for_selector('header a[href*="/followers"]', state="attached")
        await link.wait_for(state="visible")
        await link.click()
        await followers_page.wait_for_selector(
            'div[role="dialog"]',
            timeout=ScraperConfig.DIALOG_SELECTOR_TIMEOUT_MS,
        )
        
        dialog = followers_page.locator('div[role="dialog"]').last
        first_link = dialog.locator('a[href^="/"]').first
        await first_link.wait_for(
            state="attached",
            timeout=ScraperConfig.FIRST_LINK_WAIT_TIMEOUT_MS,
        )

        user_links = dialog.locator('a[href^="/"]')

        # Initialize tracking variables
        yes_rows: List[Dict] = []
        seen_handles: Set[str] = set()
        batch_handles: List[str] = []
        total_classified = 0
        all_scraped_bios: List[Dict] = []
        
        # Scraping loop
        idle_loops = 0
        previous_count = 0
        start_time = time.perf_counter()
        timeout_seconds = 3600

        async with httpx.AsyncClient(timeout=ScraperConfig.HTTPX_LONG_TIMEOUT) as client:
            while len(yes_rows) < target_yes:
                # Check timeout
                elapsed_time = time.perf_counter() - start_time
                if elapsed_time > (timeout_seconds - 30):
                    print(f"⏰ Timeout approaching ({elapsed_time:.1f}s elapsed, {timeout_seconds}s limit). Returning partial results.")
                    break
                
                # Collect visible handles
                for h in await user_links.all_inner_texts():
                    h = h.strip()
                    if h and h not in seen_handles:
                        seen_handles.add(h)
                        batch_handles.append(h)

                # Check progress
                new_total = len(seen_handles)
                if new_total == previous_count:
                    idle_loops += 1
                    if idle_loops >= ScraperConfig.MAX_IDLE_LOOPS:
                        print(f"No new followers after {ScraperConfig.MAX_IDLE_LOOPS} scrolls – quitting.")
                        break
                else:
                    idle_loops = 0
                previous_count = new_total

                # Scroll
                try:
                    await user_links.nth(-1).scroll_into_view_if_needed()
                except PlayTimeout:
                    print("⚠️ Scroll timeout encountered – likely end of list.")
                    break
                
                # Dynamic wait time
                if idle_loops > 0:
                    if ScraperConfig.PROGRESSIVE_WAIT:
                        wait_time = ScraperConfig.IDLE_SCROLL_WAIT + (idle_loops - 1) * 2
                    else:
                        wait_time = ScraperConfig.IDLE_SCROLL_WAIT
                    print(f"⏳ Idle loop {idle_loops}/{ScraperConfig.MAX_IDLE_LOOPS}, waiting {wait_time}s...")
                else:
                    wait_time = ScraperConfig.BASE_SCROLL_WAIT
                await asyncio.sleep(wait_time)

                # Process batch
                if len(batch_handles) >= batch_size:
                    try:
                        current_batch = batch_handles[:batch_size]
                        bios = await self._get_bios_parallel(bio_pages, current_batch)
                        
                        all_scraped_bios.extend(bios)
                        batch_handles = batch_handles[batch_size:]
                        
                        if bios:
                            total_classified += len(bios)
                            yes_results = await self._process_bio_batch(bios, client)
                            yes_rows.extend(yes_results)
                        
                        # Log statistics for current batch only
                        self.log_classification_stats(
                            total_bios=len(bios),
                            valid_bios=len(bios),
                            yes_count=len(yes_results),  # Use current batch results, not cumulative
                            target_yes=target_yes
                        )
                        
                    except RuntimeError as e:
                        print(str(e))
                        print("🛑 Aborting scraping due to challenge.")
                        break
            
            # Process leftover handles
            if batch_handles:
                async with httpx.AsyncClient(timeout=ScraperConfig.HTTPX_LONG_TIMEOUT) as client:
                    bios = await self._get_bios_parallel(bio_pages, batch_handles)
                    
                    if bios:
                        print(f"📊 Processing {len(bios)} leftover bios")
                        all_scraped_bios.extend(bios)
                        total_classified += len(bios)
                        
                        yes_results = await self._process_bio_batch(bios, client)
                        yes_rows.extend(yes_results)
        
        # Log overall classification statistics
        if all_scraped_bios:
            print(f"📊 Overall Classification Summary:")
            print(f"   - Total bios scraped: {len(all_scraped_bios)}")
            print(f"   - Total bios classified as 'yes': {len(yes_rows)}")
            print(f"   - Target 'yes' count: {target_yes}")
            if len(all_scraped_bios) > 0:
                overall_success_rate = (len(yes_rows) / len(all_scraped_bios)) * 100
                print(f"   - Overall success rate: {overall_success_rate:.1f}%")
            if len(yes_rows) > 0:
                overall_progress = (len(yes_rows) / target_yes) * 100
                print(f"   - Overall progress toward target: {overall_progress:.1f}%")
        
        # Save results
        await self._save_results(all_scraped_bios, yes_rows, target, target_yes, start_time, timeout_seconds)
        
        return yes_rows
    
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
            await followers_page.screenshot(path=not_visible_path, full_page=True)
        except PlayTimeout:
            print("Screenshot timed out – continuing without it")
        
        if self.gcs_bucket:
            self.csv_exporter.upload_to_gcs(local_path=not_visible_path, destination_blob=not_visible_path)
        
        await context.close()


# Backward compatibility function
async def scrape_followers(
    browser: Browser,
    state_path: Path,
    target: str,
    target_yes: int = 10,
    batch_size: int = 20,
    num_bio_pages: int = 3,
) -> List[Dict]:
    """
    Backward compatibility function for the old API.
    
    This function creates a scraper instance and calls the new method.
    """
    scraper = InstagramScraper()
    return await scraper.scrape_followers(
        browser, state_path, target, target_yes, batch_size, num_bio_pages
    )
