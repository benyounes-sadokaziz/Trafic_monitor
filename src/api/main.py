"""
FastAPI Backend for Traffic Monitor
Provides REST API endpoints for video processing
"""
import json
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, Form
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pathlib import Path
import uuid
from datetime import datetime
from typing import Optional

from config.settings import settings
from src.api.schemas import (
    JobResponse,
    JobStatusResponse,
    ProcessingStats
)
from src.api.websocket import websocket_endpoint, manager as ws_manager
from src.monitoring.metrics import (
    prometheus_metrics,
    track_request,
    track_processing_time
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Traffic Monitoring System API - Vehicle Detection, Tracking, and Speed Estimation",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (will be replaced with Redis/Database)
jobs_storage = {}


@app.on_event("startup")
async def startup_event():
    """Initialize app on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {'Development' if settings.debug else 'Production'}")
    settings.ensure_directories()
    logger.info("✅ Application started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application...")


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "process_video": "/api/process",
            "job_status": "/api/jobs/{job_id}",
            "websocket": "/ws/{job_id}"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis": "connected",  
        "storage": "available"  
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/api/process", response_model=JobResponse)
@track_request
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    frame_width_meters: float = Form(50.0),
    frame_height_meters: float = Form(30.0),
    max_frames: Optional[int] = Form(None),
    save_output_video: bool = Form(False)
):
    """
    Upload and process a video file.
    
    Args:
        file: Video file to process (MP4, AVI, MOV)
        frame_width_meters: Real-world width camera sees (for speed estimation)
        frame_height_meters: Real-world height camera sees
        max_frames: Maximum frames to process (None = all frames)
        save_output_video: Whether to save annotated output video
    
    Returns:
        JobResponse with job_id and status
    """
    logger.info(f"Received video upload: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: MP4, AVI, MOV"
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    input_path = settings.input_dir / f"{job_id}_{file.filename}"
    
    try:
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Saved video to: {input_path}")
        
        # Create job entry
        job_data = {
            "job_id": job_id,
            "filename": file.filename,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "video_path": str(input_path),
            "frame_width_meters": frame_width_meters,
            "frame_height_meters": frame_height_meters,
            "max_frames": max_frames,
            "save_output_video": save_output_video
        }
        
        jobs_storage[job_id] = job_data
        # Metrics: a new job has been created
        try:
            from src.monitoring.metrics import prometheus_metrics
            prometheus_metrics.jobs_created.inc()
        except Exception:
            pass
        
        # Add processing task to background
        background_tasks.add_task(
            process_video_task,
            job_id=job_id,
            video_path=input_path,
            frame_width_meters=frame_width_meters,
            frame_height_meters=frame_height_meters,
            max_frames=max_frames,
            save_output_video=save_output_video
        )
        
        logger.info(f"✅ Job {job_id} queued for processing")
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            message="Video uploaded successfully and queued for processing",
            created_at=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status and results of a processing job.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        JobStatusResponse with current status and statistics
    """
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_data = jobs_storage[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_data["status"],
        created_at=job_data["created_at"],
        progress=job_data.get("progress", 0.0),
        stats=job_data.get("stats"),
        error=job_data.get("error")
    )


@app.get("/api/jobs/{job_id}/tracks")
async def get_job_tracks(job_id: str):
    """Return the latest known tracks with speed/violation/screenshot info."""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = jobs_storage[job_id]
    # Prefer cumulative tracks if available, else fall back to latest snapshot
    if "all_tracks" in job and isinstance(job["all_tracks"], dict):
        tracks = list(job["all_tracks"].values())
    else:
        tracks = job.get("latest_tracks", [])

    # Fallback: if a track has no plate_screenshot yet, scan the screenshots folder
    # Structure: settings.output_dir / job_id / f"{class}_{track_id}" / files like frame_XXXXXX_q0.95.jpg
    try:
        from pathlib import Path
        import re
        import base64 as _b64
        root = Path(settings.output_dir) / job_id
        for t in tracks:
            if t.get("plate_screenshot"):
                continue
            cls = t.get("class")
            tid = t.get("track_id")
            if cls is None or tid is None:
                continue
            track_dir = root / f"{cls}_{tid}"
            if not track_dir.exists() or not track_dir.is_dir():
                continue
            best_file = None
            best_q = -1.0
            for img in track_dir.glob("*.jpg"):
                m = re.search(r"_q([0-9]+\.[0-9]+)\.jpg$", img.name)
                if not m:
                    continue
                q = float(m.group(1))
                if q > best_q:
                    best_q = q
                    best_file = img
            if best_file is not None:
                try:
                    with open(best_file, "rb") as _f:
                        data_url = "data:image/jpeg;base64," + _b64.b64encode(_f.read()).decode("ascii")
                    t["plate_screenshot"] = data_url
                    if "all_tracks" in job and isinstance(job["all_tracks"], dict) and tid in job["all_tracks"]:
                        job["all_tracks"][tid]["plate_screenshot"] = data_url
                except Exception:
                    pass
    except Exception as _e:
        # Non-fatal: if scan fails, we just return what we have
        pass
    # Ensure appearance fields exist in each track for UI formatting
    for t in tracks:
        t.setdefault("first_frame", None)
        t.setdefault("first_ts", None)
        t.setdefault("last_frame", None)
        t.setdefault("last_ts", None)
    return {"tracks": tracks}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files.
    
    Args:
        job_id: Job identifier to delete
    """
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_data = jobs_storage[job_id]
    
    # Delete video file
    video_path = Path(job_data["video_path"])
    if video_path.exists():
        video_path.unlink()
    
    # Delete output directory
    output_dir = settings.output_dir / job_id
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    
    # Remove from storage
    del jobs_storage[job_id]
    
    logger.info(f"Deleted job {job_id}")
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs with their current status."""
    return {
        "total": len(jobs_storage),
        "jobs": [
            {
                "job_id": job_id,
                "filename": data["filename"],
                "status": data["status"],
                "created_at": data["created_at"],
                "progress": data.get("progress", 0.0)
            }
            for job_id, data in jobs_storage.items()
        ]
    }

@app.websocket("/ws/{job_id}")
async def websocket_route(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    
    # Accept connection immediately
    await websocket.accept()
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "message": "WebSocket connected successfully"
        })
        
        logger.info(f"WebSocket connected: {job_id}")
        
        # Register connection with manager
        if job_id not in ws_manager.active_connections:
            ws_manager.active_connections[job_id] = set()
        ws_manager.active_connections[job_id].add(websocket)
        
        # Keep connection alive
        while True:
            try:
                # Wait for message or timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                # Handle client messages
                import json
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        if job_id in ws_manager.active_connections:
            ws_manager.active_connections[job_id].discard(websocket)
            if not ws_manager.active_connections[job_id]:
                del ws_manager.active_connections[job_id]
        logger.info(f"WebSocket disconnected: {job_id}")

# Background processing task
# Background processing task
@track_processing_time
async def process_video_task(
    job_id: str,
    video_path: Path,
    frame_width_meters: float,
    frame_height_meters: float,
    max_frames: Optional[int],
    save_output_video: bool
):
    """Background task to process video through pipeline."""
    from src.pipeline.orchestrator import TrafficMonitorPipeline
    from src.api.websocket import manager as ws_manager
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    logger.info(
        f"Starting processing for job {job_id} | "
        f"calibration: width={frame_width_meters}m, height={frame_height_meters}m | "
        f"max_frames={max_frames} | save_output_video={save_output_video}"
    )
    
    jobs_storage[job_id]["status"] = "processing"
    jobs_storage[job_id]["started_at"] = datetime.now().isoformat()
    
    # Create event loop for this thread
    loop = asyncio.get_event_loop()
    
    def run_pipeline():
        """Run pipeline in separate thread."""
        try:
            pipeline = TrafficMonitorPipeline(
                vehicle_model_path=str(settings.yolo_vehicle_model),
                plate_model_path=str(settings.yolo_plate_model),
                output_dir=str(settings.output_dir),
                speed_output_dir=str(settings.output_dir / "speed_data"),
                frame_width_meters=frame_width_meters,
                frame_height_meters=frame_height_meters
            )
            
            # Progress callback with WebSocket support
            def progress_callback(job, frame_result, progress_percentage):
                # Update job storage (thread-safe)
                jobs_storage[job_id]["progress"] = progress_percentage
                jobs_storage[job_id]["current_frame"] = frame_result.frame_number
                
                # Update stats
                jobs_storage[job_id]["stats"] = {
                    "processed_frames": job.processed_frames,
                    "total_frames": job.total_frames,
                    "total_detections": job.total_detections,
                    "total_tracks": job.total_tracks,
                    "plate_detections": job.plate_detections,
                    "screenshots_saved": job.screenshots_saved,
                    "violations_count": job.violations_count
                }
                # Store a lightweight snapshot of tracks for REST retrieval
                try:
                    latest = []
                    from pathlib import Path as _Path
                    import base64 as _b64
                    for t in frame_result.tracks:
                        shot = t.get("plate_screenshot")
                        shot_data = None
                        if isinstance(shot, str) and _Path(shot).exists():
                            try:
                                with open(shot, "rb") as _f:
                                    shot_data = "data:image/jpeg;base64," + _b64.b64encode(_f.read()).decode("ascii")
                            except Exception:
                                shot_data = None
                        latest.append({
                            "track_id": t.get("track_id"),
                            "class": t.get("class"),
                            "speed": t.get("speed"),
                            "is_violation": t.get("is_violation", False),
                            "plate_screenshot": shot_data,
                            # For latest snapshot, last appearance is the current frame; first unknown here
                            "last_frame": frame_result.frame_number,
                            "last_ts": frame_result.timestamp
                        })
                    jobs_storage[job_id]["latest_tracks"] = latest
                except Exception:
                    pass

                # Maintain cumulative tracks so rows persist once a vehicle appears
                try:
                    all_tracks = jobs_storage[job_id].setdefault("all_tracks", {})
                    for t in frame_result.tracks:
                        tid = t.get("track_id")
                        if tid is None:
                            continue
                        existing = all_tracks.get(tid, {})
                        # Convert screenshot file path to data URL if possible
                        shot = t.get("plate_screenshot")
                        shot_data = existing.get("plate_screenshot")
                        if isinstance(shot, str):
                            from pathlib import Path as _Path
                            import base64 as _b64
                            p = _Path(shot)
                            if p.exists():
                                try:
                                    with open(p, "rb") as _f:
                                        shot_data = "data:image/jpeg;base64," + _b64.b64encode(_f.read()).decode("ascii")
                                except Exception:
                                    pass
                        # Initialize first appearance if not already set
                        first_frame = existing.get("first_frame")
                        first_ts = existing.get("first_ts")
                        if first_frame is None:
                            first_frame = frame_result.frame_number
                        if first_ts is None:
                            first_ts = frame_result.timestamp

                        # Always update last appearance to current
                        last_frame = frame_result.frame_number
                        last_ts = frame_result.timestamp

                        all_tracks[tid] = {
                            "track_id": tid,
                            # Keep class once set; update if previously unknown
                            "class": existing.get("class", t.get("class")),
                            # Always update latest speed and violation state
                            "speed": t.get("speed"),
                            "is_violation": t.get("is_violation", False),
                            # Prefer the newest non-null screenshot data URL, else keep existing
                            "plate_screenshot": shot_data,
                            # Appearance tracking
                            "first_frame": first_frame,
                            "first_ts": first_ts,
                            "last_frame": last_frame,
                            "last_ts": last_ts
                        }
                except Exception:
                    pass

                
                if frame_result.frame_number % 3 == 0:
                    try:
                        import base64
                        
                        # Encode frame as base64 if available
                        frame_base64 = None
                        if hasattr(frame_result, 'frame_data') and frame_result.frame_data:
                            frame_base64 = base64.b64encode(frame_result.frame_data).decode('utf-8')
                        
                        asyncio.run_coroutine_threadsafe(
                            ws_manager.broadcast_frame_update(
                                job_id=job_id,
                                frame_number=frame_result.frame_number,
                                progress=progress_percentage,
                                tracks=[
                                    {
                                        "track_id": t["track_id"],
                                        "class": t["class"],
                                        "bbox": t["bbox"],
                                        "confidence": t.get("confidence", 0.0)
                                    }
                                    for t in frame_result.tracks
                                ],
                                stats={
                                    "detections": frame_result.detections_count,
                                    "tracked": frame_result.tracked_count,
                                    "plates": frame_result.plate_detections,
                                    "violations": frame_result.violations_count
                                },
                                frame_base64=frame_base64  # ← ADD FRAME DATA
                            ),
                            loop
                        )
                    except Exception as e:
                        logger.warning(f"WebSocket broadcast failed: {e}")
                # Send WebSocket update (schedule in main loop)
                try:
                    # Optionally include an annotated, downscaled frame in base64 to display in dashboard
                    frame_b64 = None
                    try:
                        # Import here to avoid global dependency if not used elsewhere
                        import cv2
                        import base64
                        import numpy as np
                        # We will draw on a copy and downscale to reduce bandwidth
                        if "last_frame" in frame_result.__dict__ and frame_result.last_frame is not None:
                            annotated = frame_result.last_frame.copy()
                        else:
                            annotated = None
                        # If annotated frame not provided by frame_result, skip image streaming
                        if annotated is not None:
                            # Downscale to width ~640 while keeping aspect ratio
                            h, w = annotated.shape[:2]
                            target_w = 640
                            if w > target_w:
                                scale = target_w / float(w)
                                annotated = cv2.resize(annotated, (target_w, int(h * scale)))
                            # Encode JPEG
                            success, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            if success:
                                frame_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
                    except Exception as _img_e:
                        frame_b64 = None
                        # Do not fail WS on image issues; image is optional

                    asyncio.run_coroutine_threadsafe(
                        ws_manager.broadcast_frame_update(
                            job_id=job_id,
                            frame_number=frame_result.frame_number,
                            progress=progress_percentage,
                            tracks=[
                                {
                                    "track_id": t["track_id"],
                                    "class": t["class"],
                                    "bbox": t["bbox"],
                                    "confidence": t.get("confidence", 0.0)
                                }
                                for t in frame_result.tracks
                            ],
                            stats={
                                "detections": frame_result.detections_count,
                                "tracked": frame_result.tracked_count,
                                "plates": frame_result.plate_detections,
                                "violations": frame_result.violations_count
                            },
                            frame_base64=frame_b64,
                        ),
                        loop
                    )
                except Exception as e:
                    logger.warning(f"WebSocket broadcast failed: {e}")
            
            # Process video (this blocks, but in separate thread!)
            result = pipeline.process_video(
                video_path=video_path,
                job_id=job_id,
                max_frames=max_frames,
                progress_callback=progress_callback,
                save_output_video=save_output_video
            )
            
            # Job completed
            jobs_storage[job_id]["status"] = "completed"
            jobs_storage[job_id]["completed_at"] = datetime.now().isoformat()
            jobs_storage[job_id]["progress"] = 100.0
            jobs_storage[job_id]["result"] = {
                "processed_frames": result.processed_frames,
                "total_tracks": result.total_tracks,
                "plate_detections": result.plate_detections,
                "screenshots_saved": result.screenshots_saved,
                "violations_count": result.violations_count,
                "speed_data_path": result.speed_data_path,
                "duration_seconds": (result.end_time - result.start_time).total_seconds()
            }
            
            # Send completion message via WebSocket
            try:
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast_status_update(
                        job_id=job_id,
                        status="completed",
                        message="Video processing completed successfully!"
                    ),
                    loop
                )
            except Exception as e:
                logger.warning(f"WebSocket completion broadcast failed: {e}")
            
            logger.info(f"✅ Job {job_id} completed successfully")
            prometheus_metrics.job_completed.inc()
            
        except Exception as e:
            logger.error(f"❌ Job {job_id} failed: {e}", exc_info=True)
            
            jobs_storage[job_id]["status"] = "failed"
            jobs_storage[job_id]["error"] = str(e)
            jobs_storage[job_id]["failed_at"] = datetime.now().isoformat()
            
            # Send error via WebSocket
            try:
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast_status_update(
                        job_id=job_id,
                        status="failed",
                        message=f"Processing failed: {str(e)}"
                    ),
                    loop
                )
            except Exception as e:
                logger.warning(f"WebSocket error broadcast failed: {e}")
            
            prometheus_metrics.job_failed.inc()
    
    # Run pipeline in thread pool (non-blocking!)
    with ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(executor, run_pipeline)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers
    )