"""
Push notification functionality.
"""

import os

import httpx


class NotificationService:
    """Handles push notifications."""
    
    def __init__(self):
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.pushover_user = os.getenv("PUSHOVER_USER")
    
    async def send_notification(self, message: str, title: str = "Instagram Scraper") -> None:
        """Send push notification using Pushover."""
        if not self.pushover_token or not self.pushover_user:
            print("⚠️ Pushover credentials not configured - skipping notification")
            return
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.pushover.net/1/messages.json",
                    data={
                        "token": self.pushover_token,
                        "user": self.pushover_user,
                        "title": title,
                        "message": message,
                        "priority": 0,
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                print("📱 Notification sent successfully")
        except Exception as e:
            print(f"❌ Failed to send notification: {e}") 