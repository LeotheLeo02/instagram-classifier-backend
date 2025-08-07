"""
Configuration constants for the Instagram scraper.
"""

import httpx


class ScraperConfig:
    """Configuration constants for the Instagram scraper."""
    
    # Timeouts (in milliseconds)
    PAGE_NAVIGATION_TIMEOUT_MS = 60_000
    DIALOG_SELECTOR_TIMEOUT_MS = 60_000
    FIRST_LINK_WAIT_TIMEOUT_MS = 30_000
    BIO_PAGE_TIMEOUT_MS = 10_000
    
    # Scroll timing constants (in seconds)
    BASE_SCROLL_WAIT = 1.0
    IDLE_SCROLL_WAIT = 3.0
    PROGRESSIVE_WAIT = True
    MAX_IDLE_LOOPS = 10
    
    # HTTP timeout
    HTTPX_LONG_TIMEOUT = httpx.Timeout(connect=30.0, write=30.0, read=10_000.0, pool=None)
    
    # Classification settings
    CLASSIFICATION_CHUNK_SIZE = 10
    CLASSIFICATION_MAX_RETRIES = 2
    # When the classifier has a partial batch buffered, flush it after this many seconds
    BATCH_FLUSH_TIMEOUT_SEC = 1.0

    # Parallelism/backpressure
    # Use bounded queues to prevent unbounded memory growth under backpressure
    USERNAME_QUEUE_MAXSIZE = 20
    BIO_QUEUE_MAXSIZE = 200

    # Overall safety timeout for a single scrape run (seconds)
    SCRAPE_MAX_SECONDS = 3600
    
    # Christian symbols for bio validation
    CHRISTIAN_SYMBOLS = {
        '✝️', '✞', '⛪', '🙏', '📖', '✝', '⛪️', '🕊️', '🕊', '✟', '☦️', '☦'
    } 