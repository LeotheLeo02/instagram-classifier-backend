"""
Bio validation and cleaning functionality.
"""

from typing import Optional

from .config import ScraperConfig


class BioValidator:
    """Handles bio validation and cleaning."""
    
    @staticmethod
    def validate_and_clean_bio(bio_dict: dict) -> Optional[dict]:
        """
        Validate and clean a bio dictionary to ensure data quality.
        
        Args:
            bio_dict: Dictionary with 'username' and 'bio' keys
            
        Returns:
            Cleaned bio dictionary or None if validation fails
        """
        if not bio_dict or not isinstance(bio_dict, dict):
            return None
        
        username = bio_dict.get("username", "").strip()
        bio = bio_dict.get("bio", "").strip()
        
        # Basic validation
        if not username or not bio:
            return None
        
        # Username validation
        if len(username) < 1 or len(username) > 30:
            return None
        
        # Bio validation
        if len(bio) < 1 or len(bio) > 1000:
            return None
        
        # Filter out obviously invalid content
        if bio.lower() in ["", "n/a", "none", "null", "undefined"]:
            return None
        
        # Check for excessive special characters, but allow Christian symbols
        special_chars = sum(
            1 for c in bio 
            if not c.isalnum() and not c.isspace() and c not in ScraperConfig.CHRISTIAN_SYMBOLS
        )
        
        special_char_ratio = special_chars / len(bio)
        if special_char_ratio > 0.5:
            return None
        
        # Additional check: if bio contains Christian symbols, be more lenient
        has_christian_symbols = any(symbol in bio for symbol in ScraperConfig.CHRISTIAN_SYMBOLS)
        if has_christian_symbols:
            return {"username": username, "bio": bio}
        
        return {"username": username, "bio": bio} 