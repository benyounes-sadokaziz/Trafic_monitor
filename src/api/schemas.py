"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class JobResponse(BaseModel):
    """Response when creating a new job."""
    job_id: str
    status: str
    message: str
    created_at: datetime


class ProcessingStats(BaseModel):
    """Real-time processing statistics."""
    processed_frames: int = 0
    total_frames: int = 0
    total_detections: int = 0
    total_tracks: int = 0
    plate_detections: int = 0
    screenshots_saved: int = 0
    violations_count: int = 0


class JobStatusResponse(BaseModel):
    """Detailed job status response."""
    job_id: str
    status: str  # queued, processing, completed, failed
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0
    stats: Optional[ProcessingStats] = None
    error: Optional[str] = None


class VehicleTrack(BaseModel):
    """Single vehicle track information."""
    track_id: int
    class_name: str
    bbox: List[int]
    confidence: float
    speed_kmh: Optional[float] = None
    is_violation: bool = False


class FrameUpdate(BaseModel):
    """Real-time frame processing update (for WebSocket)."""
    job_id: str
    frame_number: int
    timestamp: float
    progress: float
    tracks: List[VehicleTrack]
    detections_count: int
    violations_count: int


class SpeedViolation(BaseModel):
    """Speed violation record."""
    track_id: int
    vehicle_class: str
    max_speed: float
    speed_limit: float
    timestamp: float
    frame_number: int