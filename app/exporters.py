"""
CSV export and Google Cloud Storage upload functionality.
"""

import csv
import os
from datetime import datetime
from typing import List, Dict, Optional

from google.cloud import storage


class CSVExporter:
    """Handles CSV export and GCS upload functionality."""
    
    def __init__(self, gcs_bucket: str):
        self.gcs_bucket = gcs_bucket
    
    def upload_to_gcs(self, local_path: str, destination_blob: str) -> None:
        """Upload a local file to Google Cloud Storage."""
        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(local_path)
        print(f"✅ Screenshot uploaded to gs://{self.gcs_bucket}/{destination_blob}")
    
    def save_bios_to_csv_and_upload(
        self, 
        bios: List[Dict], 
        target: str, 
        operation_id: Optional[str] = None
    ) -> str:
        """Save scraped bios to CSV and upload to GCS."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        operation_suffix = f"_{operation_id}" if operation_id else ""
        filename = f"scraped_bios_{target}_{timestamp}{operation_suffix}.csv"
        
        local_path = f"data/{filename}"
        os.makedirs("data", exist_ok=True)
        
        with open(local_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['username', 'bio', 'url', 'scraped_at', 'target_account', 'operation_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for bio in bios:
                writer.writerow({
                    'username': bio.get('username', ''),
                    'bio': bio.get('bio', ''),
                    'url': f"https://www.instagram.com/{bio.get('username', '')}/",
                    'scraped_at': datetime.now().isoformat(),
                    'target_account': target,
                    'operation_id': operation_id or ''
                })
        
        gcs_path = f"scraped_bios/{filename}"
        self.upload_to_gcs(local_path, gcs_path)
        
        print(f"✅ Bios saved to CSV: {local_path}")
        print(f"✅ CSV uploaded to GCS: gs://{self.gcs_bucket}/{gcs_path}")
        
        return f"gs://{self.gcs_bucket}/{gcs_path}"
    
    def save_classification_results_to_csv_and_upload(
        self, 
        bios: List[Dict], 
        classifications: List[str], 
        target: str, 
        operation_id: Optional[str] = None
    ) -> str:
        """Save scraped bios with their classifications to CSV and upload to GCS."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        operation_suffix = f"_{operation_id}" if operation_id else ""
        filename = f"scraped_bios_with_classifications_{target}_{timestamp}{operation_suffix}.csv"
        
        local_path = f"data/{filename}"
        os.makedirs("data", exist_ok=True)
        
        with open(local_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['username', 'bio', 'classification', 'timestamp', 'source', 'operation_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for bio_dict, classification in zip(bios, classifications):
                writer.writerow({
                    'username': bio_dict['username'],
                    'bio': bio_dict['bio'],
                    'classification': classification,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'scraper',
                    'operation_id': operation_id
                })
        
        gcs_path = f"scraped_bios/{filename}"
        self.upload_to_gcs(local_path, gcs_path)
        
        return f"gs://{self.gcs_bucket}/{gcs_path}" 