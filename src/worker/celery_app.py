"""
Celery worker for async video processing
"""

import logging
from celery import Celery
from pathlib import Path

from config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    'traffic_monitor',
    broker=settings.redis_url,
    backend=settings.redis_url
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks
)

logger.info("✅ Celery app initialized")


@celery_app.task(bind=True, name='process_video_task')
def process_video_task(
    self,
    job_id: str,
    video_path: str,
    frame_width_meters: float = 50.0,
    frame_height_meters: float = 30.0,
    max_frames: int = None,
    save_output_video: bool = False
):
    """
    Celery task to process video through pipeline.
    
    This task runs in the background worker, freeing up the API.
    
    Args:
        job_id: Unique job identifier
        video_path: Path to video file
        frame_width_meters: Camera calibration parameter
        frame_height_meters: Camera calibration parameter
        max_frames: Max frames to process (None = all)
        save_output_video: Whether to save annotated video
    """
    from src.pipeline.orchestrator import TrafficMonitorPipeline
    
    logger.info(f"[Celery] Starting task for job {job_id}")
    
    # Update task state
    self.update_state(
        state='PROCESSING',
        meta={'job_id': job_id, 'status': 'initializing'}
    )
    
    try:
        # Initialize pipeline
        pipeline = TrafficMonitorPipeline(
            vehicle_model_path=str(settings.yolo_vehicle_model),
            plate_model_path=str(settings.yolo_plate_model),
            output_dir=str(settings.output_dir),
            speed_output_dir=str(settings.output_dir / "speed_data"),
            frame_width_meters=frame_width_meters,
            frame_height_meters=frame_height_meters
        )
        
        # Progress callback for Celery
        def progress_callback(job, frame_result, progress_percentage):
            self.update_state(
                state='PROCESSING',
                meta={
                    'job_id': job_id,
                    'status': 'processing',
                    'progress': progress_percentage,
                    'frame': frame_result.frame_number,
                    'total_frames': job.total_frames,
                    'tracks': frame_result.active_tracks,
                    'violations': frame_result.violations_count
                }
            )
        
        # Process video
        result = pipeline.process_video(
            video_path=Path(video_path),
            job_id=job_id,
            max_frames=max_frames,
            progress_callback=progress_callback,
            save_output_video=save_output_video
        )
        
        logger.info(f"[Celery] ✅ Job {job_id} completed successfully")
        
        # Return final results
        return {
            'job_id': job_id,
            'status': 'completed',
            'processed_frames': result.processed_frames,
            'total_tracks': result.total_tracks,
            'plate_detections': result.plate_detections,
            'screenshots_saved': result.screenshots_saved,
            'violations_count': result.violations_count,
            'speed_data_path': result.speed_data_path,
            'duration_seconds': (result.end_time - result.start_time).total_seconds()
        }
    
    except Exception as e:
        logger.error(f"[Celery] ❌ Job {job_id} failed: {e}", exc_info=True)
        
        self.update_state(
            state='FAILURE',
            meta={
                'job_id': job_id,
                'status': 'failed',
                'error': str(e)
            }
        )
        
        raise