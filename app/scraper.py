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
import os
import time
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
    
    def __init__(self, gcs_bucket: Optional[str] = None):
        self.gcs_bucket = gcs_bucket or os.getenv("SCREENSHOT_BUCKET")
        self.csv_exporter = CSVExporter(self.gcs_bucket)
        self.bio_classifier = BioClassifier()
        self.notification_service = NotificationService()
    
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
        """Log comprehensive classification statistics."""
        print(f"📊 Classification Statistics:")
        print(f"   - Total bios processed: {total_bios}")
        print(f"   - Valid bios after filtering: {valid_bios}")
        print(f"   - Bios classified as 'yes': {yes_count}")
        print(f"   - Target 'yes' count: {target_yes}")
        if valid_bios > 0:
            success_rate = (yes_count / valid_bios) * 100
            print(f"   - Success rate: {success_rate:.1f}%")
        if yes_count > 0:
            progress = (yes_count / target_yes) * 100
            print(f"   - Progress toward target: {progress:.1f}%")
    
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
        
        # Split usernames across bio pages
        chunk_size = max(1, len(usernames) // num_bio_pages)
        chunks_list = [usernames[i:i + chunk_size] for i in range(0, len(usernames), chunk_size)]
        
        # Ensure we don't have more chunks than bio pages
        while len(chunks_list) > num_bio_pages:
            chunks_list[-2].extend(chunks_list[-1])
            chunks_list.pop()
        
        # Create tasks for each chunk
        tasks = []
        for i, chunk in enumerate(chunks_list):
            if chunk:
                task = asyncio.create_task(
                    self._get_bios_for_chunk(bio_pages[i % num_bio_pages], chunk)
                )
                tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle any exceptions
        all_bios = []
        for result in results:
            if isinstance(result, Exception):
                print(f"⚠️ Error in bio fetching: {result}")
                all_bios.extend([{"username": "", "bio": ""}] * len(chunk))
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
        
        # Process classification results
        yes_idx = set()
        chunk_start = 0
        for chunk_num in range(0, len(valid_bios), ScraperConfig.CLASSIFICATION_CHUNK_SIZE):
            chunk_end = min(chunk_num + ScraperConfig.CLASSIFICATION_CHUNK_SIZE, len(valid_bios))
            chunk_size = chunk_end - chunk_num
            
            chunk_flags = all_flags[chunk_start:chunk_start + chunk_size]
            
            for flag in chunk_flags:
                if flag and flag.isdigit():
                    global_idx = chunk_num + int(flag)
                    if 0 <= global_idx < len(valid_bios):
                        yes_idx.add(global_idx)
            
            chunk_start += chunk_size
        
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
            return await self._scrape_followers_impl(
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
                        
                        # Log statistics
                        self.log_classification_stats(
                            total_bios=len(bios),
                            valid_bios=len(bios),
                            yes_count=len(yes_rows),
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
        
        # Save results
        await self._save_results(all_scraped_bios, yes_rows, target, start_time, timeout_seconds)
        
        return yes_rows
    
    async def _save_results(
        self, 
        all_scraped_bios: List[Dict], 
        yes_rows: List[Dict], 
        target: str, 
        start_time: float, 
        timeout_seconds: int
    ) -> None:
        """Save scraping results to CSV and send notifications."""
        if all_scraped_bios:
            operation_id = f"op_{int(time.time())}"
            
            # Save bios to CSV
            csv_gcs_path = self.csv_exporter.save_bios_to_csv_and_upload(
                all_scraped_bios, target, operation_id
            )
            print(f"📊 Total bios scraped: {len(all_scraped_bios)}")
            print(f"📊 Total bios classified as YES: {len(yes_rows)}")
            print(f"📊 CSV saved to: {csv_gcs_path}")
            
            # Save classification results
            all_classifications = []
            for bio_dict in all_scraped_bios:
                is_yes = any(yes_row["username"] == bio_dict["username"] for yes_row in yes_rows)
                classification = "yes" if is_yes else "no"
                all_classifications.append(classification)
            
            classification_csv_path = self.csv_exporter.save_classification_results_to_csv_and_upload(
                all_scraped_bios, all_classifications, target, operation_id
            )
            print(f"📊 Classification results saved to: {classification_csv_path}")
        
        # Send notifications
        elapsed_time = time.perf_counter() - start_time
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
