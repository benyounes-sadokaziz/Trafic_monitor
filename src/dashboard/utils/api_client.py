"""
API Client Utilities
Helper functions for communicating with FastAPI backend
"""

import requests
from typing import Optional, Dict, Any


class APIClient:
    """Client for interacting with Traffic Monitor API."""
    
    def __init__(self, base_url: str):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API (e.g., http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Optional[Dict[str, Any]]:
        """Check API health status."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a processing job.
        
        Args:
            job_id: Unique job identifier
        
        Returns:
            Job status dict or None if error
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/jobs/{job_id}",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return None
    
    def list_all_jobs(self) -> Optional[Dict[str, Any]]:
        """Get list of all jobs."""
        try:
            response = requests.get(f"{self.base_url}/api/jobs", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job and its associated files.
        
        Args:
            job_id: Job to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.delete(
                f"{self.base_url}/api/jobs/{job_id}",
                timeout=5
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def get_job_tracks(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch latest tracks for a job (with speed/violation/screenshot)."""
        try:
            response = requests.get(
                f"{self.base_url}/api/jobs/{job_id}/tracks",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None