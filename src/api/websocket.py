"""
WebSocket endpoint for real-time video processing updates
"""

import logging
import json
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Active WebSocket connections per job
active_connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept new WebSocket connection for a job."""
        await websocket.accept()
        
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        
        self.active_connections[job_id].add(websocket)
        logger.info(f"WebSocket connected for job {job_id}. "
                   f"Total connections: {len(self.active_connections[job_id])}")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove WebSocket connection."""
        if job_id in self.active_connections:
            self.active_connections[job_id].discard(websocket)
            
            # Clean up empty sets
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
            
            logger.info(f"WebSocket disconnected for job {job_id}")
    
    async def send_message(self, job_id: str, message: dict):
        """Send message to all connections for a job."""
        if job_id not in self.active_connections:
            return
        
        # Convert to JSON
        message_json = json.dumps(message)
        
        # Send to all connections
        disconnected = set()
        for connection in self.active_connections[job_id]:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                disconnected.add(connection)
        
        # Remove failed connections
        for connection in disconnected:
            self.disconnect(connection, job_id)
    
    async def broadcast_frame_update(
        self,
        job_id: str,
        frame_number: int,
        progress: float,
        tracks: list,
        stats: dict,
        frame_base64: str = None,
        
    ):
        """Broadcast frame processing update."""
        message = {
            "type": "frame_update",
            "job_id": job_id,
            "frame_number": frame_number,
            "progress": progress,
            "tracks": tracks,
            "stats": stats,
            "timestamp": asyncio.get_event_loop().time(),
            "frame": frame_base64
        }
        # Include frame image only if provided to reduce bandwidth when not needed
        if frame_base64 is not None:
            message["frame_base64"] = frame_base64
        
        await self.send_message(job_id, message)
    
    async def broadcast_status_update(
        self,
        job_id: str,
        status: str,
        message: str = ""
    ):
        """Broadcast job status change."""
        update = {
            "type": "status_update",
            "job_id": job_id,
            "status": status,
            "message": message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await self.send_message(job_id, update)
    
    async def send_heartbeat(self, job_id: str):
        """Send heartbeat to keep connection alive."""
        message = {
            "type": "heartbeat",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await self.send_message(job_id, message)


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job updates.
    
    Usage:
        ws://localhost:8000/ws/{job_id}
    """
    # Accept connection FIRST (don't validate job yet)
    await manager.connect(websocket, job_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "message": "WebSocket connection established"
        })
        
        # Keep connection alive and listen for client messages
        while True:
            try:
                # Wait for messages from client (with timeout)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                # Handle client messages
                try:
                    message = json.loads(data)
                    
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {data}")
                
            except asyncio.TimeoutError:
                # Send heartbeat if no message received
                await manager.send_heartbeat(job_id)
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from job {job_id}")
        manager.disconnect(websocket, job_id)
    
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}", exc_info=True)
        manager.disconnect(websocket, job_id)


# Integration helper for pipeline
async def send_frame_update(job_id: str, frame_result, progress: float):
    """
    Helper function to send frame updates from pipeline.
    
    Call this from your pipeline progress callback.
    """
    tracks = [
        {
            "track_id": t["track_id"],
            "class": t["class"],
            "bbox": t["bbox"],
            "confidence": t["confidence"]
        }
        for t in frame_result.tracks
    ]
    
    stats = {
        "detections": frame_result.detections_count,
        "tracked": frame_result.tracked_count,
        "active_tracks": frame_result.active_tracks,
        "plates": frame_result.plate_detections,
        "violations": frame_result.violations_count
    }
    
    await manager.broadcast_frame_update(
        job_id=job_id,
        frame_number=frame_result.frame_number,
        progress=progress,
        tracks=tracks,
        stats=stats
    )