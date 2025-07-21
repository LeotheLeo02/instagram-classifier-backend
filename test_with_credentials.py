#!/usr/bin/env python3
"""
Test script with actual Pushover credentials.
This will send a real notification to your phone.
"""

import os
import sys
from app.scraper import send_pushover_notification, send_notification

def test_with_real_credentials():
    """Test with the actual Pushover credentials."""
    
    print("📱 Testing Pushover with Real Credentials")
    print("=" * 50)
    
    # Set the credentials
    os.environ['PUSHOVER_USER_KEY'] = 'uons5rx8amqeutz9qmjh5dgtes5nke'
    os.environ['PUSHOVER_APP_TOKEN'] = 'a1p8p3dz84safu7vcsh4emdo7nk26i'
    
    print("✅ Credentials set in environment")
    print(f"User Key: {os.environ['PUSHOVER_USER_KEY'][:10]}...")
    print(f"App Token: {os.environ['PUSHOVER_APP_TOKEN'][:10]}...")
    
    print("\n📤 Sending test notification...")
    
    # Test direct Pushover notification
    success = send_pushover_notification(
        title="Instagram Scraper - Test",
        message="🎉 Your Pushover setup is working! This notification was sent from your local test.",
        priority=0
    )
    
    if success:
        print("✅ Test notification sent successfully!")
        print("Check your phone for the notification.")
        
        # Test the main notification function
        print("\n📤 Testing main notification function...")
        send_notification(
            title="Instagram Scraper - Ready for Cloud Run",
            message="Your notification system is ready! When you deploy to Cloud Run, you'll get notifications like this when scraping completes.",
            success=True
        )
        
        print("\n✅ All tests passed!")
        print("Your Pushover setup is working correctly.")
        return True
    else:
        print("❌ Failed to send notification.")
        print("Please check your credentials and internet connection.")
        return False

if __name__ == "__main__":
    success = test_with_real_credentials()
    sys.exit(0 if success else 1) 