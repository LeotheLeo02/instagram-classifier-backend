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
    # Reduce read timeout to avoid long stalls on remote classifier
    HTTPX_LONG_TIMEOUT = httpx.Timeout(connect=30.0, write=30.0, read=30.0, pool=None)
    
    # Classification settings
    CLASSIFICATION_CHUNK_SIZE = 10
    CLASSIFICATION_MAX_RETRIES = 2
    # When the classifier has a partial batch buffered, flush it after this many seconds
    BATCH_FLUSH_TIMEOUT_SEC = 1.0

    # Parallelism/backpressure
    # Use bounded queues to prevent unbounded memory growth under backpressure
    USERNAME_QUEUE_MAXSIZE = 20
    BIO_QUEUE_MAXSIZE = 200
    # Optional watermarks for bio queue to avoid overproduction upstream
    BIO_QUEUE_HIGH_WATERMARK = 100
    BIO_QUEUE_LOW_WATERMARK = 20
    # Watermarks to control API fetch cadence
    USERNAME_QUEUE_LOW_WATERMARK = 2   # fetch another page when qsize <= LOW
    USERNAME_QUEUE_HIGH_WATERMARK = 8  # optional: slow down if qsize >= HIGH

    # Overall safety timeout for a single scrape run (seconds)
    SCRAPE_MAX_SECONDS = 3600
    
    # Instagram web API constants
    IG_APP_ID = "936619743392459"
    # Followers GraphQL query hash used by Instagram web to list followers
    FOLLOWERS_QUERY_HASH = "5aefa9893005572d237da5068082d8d5"
    # Typical page size returned by the followers GraphQL API (12 or 24)
    API_PAGE_SIZE = 24
    
    # Christian symbols for bio validation
    CHRISTIAN_SYMBOLS = {
        '✝️', '✞', '⛪', '🙏', '📖', '✝', '⛪️', '🕊️', '🕊', '✟', '☦️', '☦'
    } 