#!/usr/bin/env python3
"""
Test script for Cloud Run notification setup.
This simulates the environment variables that would be set in Cloud Run.
"""

import os
import sys
from app.scraper import send_notification, send_pushover_notification

def test_cloud_run_notifications():
    """Test notifications as they would work in Cloud Run."""
    
    print("☁️ Testing Cloud Run Notification Setup")
    print("=" * 50)
    
    # Check environment variables
    user_key = os.getenv('PUSHOVER_USER_KEY')
    app_token = os.getenv('PUSHOVER_APP_TOKEN')
    
    print(f"PUSHOVER_USER_KEY: {'✅ Set' if user_key else '❌ Not set'}")
    print(f"PUSHOVER_APP_TOKEN: {'✅ Set' if app_token else '❌ Not set'}")
    
    if not user_key or not app_token:
        print("\n❌ Missing environment variables!")
        print("Please set them before running this test:")
        print("export PUSHOVER_USER_KEY='uons5rx8amqeutz9qmjh5dgtes5nke'")
        print("export PUSHOVER_APP_TOKEN='your_app_token_here'")
        return False
    
    print("\n📱 Testing different notification types...")
    
    # Test success notification (like when scraping completes)
    print("\n1. Testing success notification...")
    success = send_notification(
        title="Instagram Scraper - Complete",
        message="Scraping for @testuser completed successfully: 10/10 results found in 45.2s",
        success=True
    )
    
    # Test partial results notification (like when timeout is reached)
    print("\n2. Testing partial results notification...")
    partial = send_notification(
        title="Instagram Scraper - Partial Results", 
        message="Scraping for @testuser completed with partial results: 7/10 found in 3600.0s (timeout reached)",
        success=False
    )
    
    # Test error notification
    print("\n3. Testing error notification...")
    error = send_notification(
        title="Instagram Scraper - Error",
        message="Scraping for @testuser failed with error: Timeout waiting for page to load...",
        success=False
    )
    
    if success and partial and error:
        print("\n✅ All notification tests passed!")
        print("Check your phone for 3 test notifications.")
        return True
    else:
        print("\n❌ Some notification tests failed.")
        return False

if __name__ == "__main__":
    success = test_cloud_run_notifications()
    sys.exit(0 if success else 1) 