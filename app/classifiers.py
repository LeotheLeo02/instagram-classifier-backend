"""
Remote bio classification functionality.
"""

import asyncio
from typing import List

import httpx

from .config import ScraperConfig


class BioClassifier:
    """Handles remote bio classification."""
    
    def __init__(self, api_url: str = "https://bio-classifier-672383441505.us-central1.run.app/classify"):
        self.api_url = api_url
    
    async def classify_bios(
        self, 
        bio_texts: List[str], 
        client: httpx.AsyncClient, 
        max_retries: int = ScraperConfig.CLASSIFICATION_MAX_RETRIES
    ) -> List[str]:
        """
        Classify bios using the remote API with retry logic.
        
        Args:
            bio_texts: List of bio texts to classify
            client: HTTP client for API requests
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of classification results (empty string for 'no', index string for 'yes')
        """
        for attempt in range(max_retries + 1):
            try:
                # Filter out empty or invalid bios
                valid_bios = []
                valid_indices = []
                for i, bio in enumerate(bio_texts):
                    if bio and bio.strip():
                        valid_bios.append(bio.strip())
                        valid_indices.append(i)
                
                if not valid_bios:
                    print("⚠️ No valid bios to classify")
                    return [""] * len(bio_texts)
                
                print(f"🔍 Classifying {len(valid_bios)} valid bios out of {len(bio_texts)} total (attempt {attempt + 1}/{max_retries + 1})")
                
                # Make API request
                request_payload = {"bios": valid_bios}
                resp = await client.post(
                    self.api_url,
                    json=request_payload,
                    timeout=ScraperConfig.HTTPX_LONG_TIMEOUT,
                )
                resp.raise_for_status()
                
                response_data = resp.json()
                data = response_data.get("results", [])
                
                # Process response
                if isinstance(data, list):
                    yes_indices = set()
                    for item in data:
                        if isinstance(item, str) and item.isdigit():
                            yes_indices.add(int(item))
                        elif isinstance(item, int):
                            yes_indices.add(item)
                    
                    # Map back to original bio indices
                    result = [""] * len(bio_texts)
                    for i, original_idx in enumerate(valid_indices):
                        if i in yes_indices:
                            result[original_idx] = str(i)
                    
                    print(f"✅ Classification successful: {len(yes_indices)} bios classified as 'yes'")
                    return result
                else:
                    print(f"⚠️ Unexpected response format: {type(data)}")
                    if attempt < max_retries:
                        print(f"🔄 Retrying... (attempt {attempt + 2}/{max_retries + 1})")
                        await asyncio.sleep(1)
                        continue
                    else:
                        return [""] * len(bio_texts)
                        
            except Exception as e:
                print(f"❌ Classification failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}")
                if attempt < max_retries:
                    print(f"🔄 Retrying in 2 seconds... (attempt {attempt + 2}/{max_retries + 1})")
                    await asyncio.sleep(2)
                    continue
                else:
                    print("🔄 All retries exhausted, falling back to 'no' classification for all bios")
                    return [""] * len(bio_texts) 