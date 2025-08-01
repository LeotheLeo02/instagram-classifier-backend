# Instagram Scraper - Modular Architecture

This directory contains the modular Instagram scraper implementation, organized into separate files for better maintainability and readability.

## File Structure

### Core Files
- **`scraper.py`** - Main scraper class and backward compatibility functions
- **`config.py`** - Configuration constants and settings
- **`validators.py`** - Bio validation and cleaning logic
- **`exporters.py`** - CSV export and Google Cloud Storage functionality
- **`classifiers.py`** - Remote bio classification service
- **`notifications.py`** - Push notification service

### Supporting Files
- **`job_entrypoint.py`** - Job entry point for the scraper
- **`__init__.py`** - Python package initialization

## Class Overview

### `InstagramScraper` (scraper.py)
Main scraper class that orchestrates the entire scraping process.

**Key Methods:**
- `scrape_followers()` - Main scraping method
- `_get_bio()` - Extract bio from profile page
- `_get_bios_parallel()` - Parallel bio fetching
- `_process_bio_batch()` - Bio validation and classification
- `_save_results()` - Save results and send notifications

### `ScraperConfig` (config.py)
Centralized configuration constants including:
- Timeouts for various operations
- Scroll timing settings
- Classification parameters
- Christian symbols for bio validation

### `BioValidator` (validators.py)
Handles bio validation and cleaning:
- Validates username and bio format
- Filters out invalid content
- Handles special character validation
- Supports Christian symbol detection

### `CSVExporter` (exporters.py)
Manages data export functionality:
- Uploads files to Google Cloud Storage
- Saves bios to CSV format
- Exports classification results
- Handles file naming and organization

### `BioClassifier` (classifiers.py)
Remote bio classification service:
- Makes API calls to classification service
- Handles retry logic and error recovery
- Processes classification responses
- Maps results back to original indices

### `NotificationService` (notifications.py)
Push notification functionality:
- Sends notifications via Pushover
- Handles credential management
- Provides error handling for notification failures

## Usage

### Basic Usage (Backward Compatible)
```python
from app.scraper import scrape_followers

results = await scrape_followers(
    browser=browser,
    state_path=state_path,
    target="target_username",
    target_yes=10
)
```

### Advanced Usage (New API)
```python
from app.scraper import InstagramScraper

scraper = InstagramScraper(gcs_bucket="my-bucket")
results = await scraper.scrape_followers(
    browser=browser,
    state_path=state_path,
    target="target_username",
    target_yes=10,
    batch_size=20,
    num_bio_pages=3
)
```

## Benefits of Modular Structure

1. **Maintainability** - Each class has a single responsibility
2. **Testability** - Individual components can be tested in isolation
3. **Reusability** - Classes can be used independently
4. **Readability** - Smaller, focused files are easier to understand
5. **Extensibility** - New features can be added without affecting existing code

## Dependencies

- `playwright` - Browser automation
- `httpx` - HTTP client for API calls
- `google-cloud-storage` - Google Cloud Storage integration
- `playwright-stealth` - Anti-detection measures (optional) 