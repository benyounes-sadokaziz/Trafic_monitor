"""
Production-Ready Traffic Monitor Pipeline
Adapted from your working test pipeline for async processing
"""

import cv2
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Generator, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

# from shapely import buffer  # Removed: not used and may conflict with local variable names

# Your existing imports
from src.detection.vehicle_detector import VehicleDetector
from src.tracking.bytetrack_tracker import VehicleTracker
from src.ocr.plate_detector import LicensePlateDetector
from src.image_quality.quality_assessor import PlateQualityAssessor
from src.ocr.plate_screenshot_manager import PlateScreenshotManager
from src.speed.speed_estimator import SpeedEstimator

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame_number: int
    timestamp: float
    detections_count: int
    tracked_count: int
    active_tracks: int
    plate_detections: int
    screenshots_saved: int
    violations_count: int
    tracks: List[Dict] = field(default_factory=list)
    violations: List[Dict] = field(default_factory=list)
    # Optional annotated frame for live preview (not persisted)
    last_frame: Optional[Any] = None
    frame_data: Optional[bytes] = None  # ← ADD THIS


@dataclass
class ProcessingJob:
    """Complete video processing job with all metadata."""
    job_id: str
    video_path: Path
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, processing, completed, failed
    
    # Video info
    fps: int = 0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    processed_frames: int = 0
    duration_seconds: float = 0.0
    
    # Processing stats
    total_detections: int = 0
    total_tracks: int = 0
    plate_detections: int = 0
    screenshots_saved: int = 0
    violations_count: int = 0
    
    # Results
    frame_results: List[FrameResult] = field(default_factory=list)
    speed_data_path: Optional[str] = None
    error: Optional[str] = None


class TrafficMonitorPipeline:
    """
    Production-ready traffic monitoring pipeline.
    
    This wraps your existing modules with production features:
    - Progress callbacks for real-time updates
    - Streaming results for WebSocket
    - Database integration hooks
    - Error handling and recovery
    """
    
    def __init__(
        self,
        vehicle_model_path: str = 'yolov8n.pt',
        plate_model_path: str = 'best.pt',
        output_dir: str = 'data/output/plate_screenshots',
        speed_output_dir: str = 'data/output/speed_data',
        frame_width_meters: float = 50.0,
        frame_height_meters: float = 30.0,
        speed_limits: Optional[Dict[str, float]] = None,
        vehicle_confidence: float = 0.5,
        plate_confidence: float = 0.5,
        quality_threshold: float = 0.6,
        device: str = "cuda"
    ):
        """
        Initialize pipeline with all components.
        
        Args:
            vehicle_model_path: Path to vehicle detection model
            plate_model_path: Path to plate detection model
            output_dir: Directory for plate screenshots
            speed_output_dir: Directory for speed data exports
            frame_width_meters: Real-world width camera sees
            frame_height_meters: Real-world height camera sees
            speed_limits: Speed limits per vehicle class (km/h)
            vehicle_confidence: Confidence threshold for vehicles
            plate_confidence: Confidence threshold for plates
            quality_threshold: Minimum quality for plate screenshots
            device: Device for inference (cuda/cpu)
        """
        logger.info("Initializing Traffic Monitor Pipeline...")
        
        self.output_dir = Path(output_dir)
        self.speed_output_dir = Path(speed_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.speed_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Camera calibration
        self.frame_width_meters = frame_width_meters
        self.frame_height_meters = frame_height_meters
        
        # Speed limits
        self.speed_limits = speed_limits or {
            'car': 120,
            'truck': 90,
            'bus': 90,
            'motorcycle': 120,
            'bicycle': 30
        }
        
        # Initialize vehicle detector
        logger.info("Loading vehicle detector...")
        self.vehicle_detector = VehicleDetector(
            model_path=vehicle_model_path,
            confidence_threshold=vehicle_confidence
        )
        self.vehicle_detector.load_model()
        logger.info("✓ Vehicle detector ready")
        
        # Initialize tracker
        logger.info("Loading tracker...")
        self.vehicle_tracker = VehicleTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8
        )
        logger.info("✓ Tracker ready")
        
        # Initialize plate detector
        logger.info("Loading plate detector...")
        try:
            self.plate_detector = LicensePlateDetector(
                model_path=plate_model_path,
                confidence_threshold=plate_confidence,
                device=device
            )
            logger.info("✓ Plate detector ready")
        except Exception as e:
            logger.warning(f"Could not load plate detector: {e}")
            self.plate_detector = None
        
        # Initialize quality assessor
        logger.info("Loading quality assessor...")
        self.quality_assessor = PlateQualityAssessor(
            overall_threshold=quality_threshold
        )
        logger.info("✓ Quality assessor ready")
        
        logger.info("✅ Pipeline initialized successfully")
    
    def process_video(
        self,
        video_path: Path,
        job_id: str,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        save_output_video: bool = False
    ) -> ProcessingJob:
        """
        Process complete video through pipeline.
        
        Args:
            video_path: Path to input video
            job_id: Unique job identifier
            max_frames: Maximum frames to process (None = all)
            progress_callback: Function called with progress updates
            save_output_video: Whether to save annotated video
        
        Returns:
            ProcessingJob with all results and statistics
        """
        job = ProcessingJob(
            job_id=job_id,
            video_path=video_path,
            status="processing"
        )
        
        # Initialize screenshot manager per job
        screenshot_manager = PlateScreenshotManager(
            output_dir=str(self.output_dir / job_id),
            perfect_quality_threshold=0.95,
            max_perfect_quality_shots=3,
            min_quality_threshold=0.6,
            verbose=False
        )
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            job.fps = int(cap.get(cv2.CAP_PROP_FPS))
            job.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            job.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            job.duration_seconds = job.total_frames / job.fps if job.fps > 0 else 0
            
            logger.info(f"Video info: {job.width}x{job.height} @ {job.fps}fps, "
                       f"{job.total_frames} frames ({job.duration_seconds:.1f}s)")
            
            # Initialize speed estimator
            speed_estimator = SpeedEstimator(
                frame_width_meters=self.frame_width_meters,
                frame_height_meters=self.frame_height_meters,
                fps=job.fps,
                speed_limits=self.speed_limits,
                min_frames_for_speed=10,
                smoothing_window=5,
                ignore_edge_frames=3,
                min_distance_threshold=0.5,
                speed_unit='kmh',
                output_dir='.',  # ✅ Fixed - let SpeedEstimator handle its own path
                outlier_rejection=True,
                verbose=False
            )
            speed_estimator.set_frame_dimensions(job.width, job.height)
            
            # Setup output video if requested
            out = None
            if save_output_video:
                output_path = self.output_dir / job_id / "output.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, job.fps, 
                                     (job.width, job.height))
            
            # Process frames
            frame_count = 0
            active_tracks = set()
            lost_tracks = set()
            
            max_frames_to_process = max_frames or job.total_frames
            
            while frame_count < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                frame_result = self._process_frame(
                    frame=frame,
                    frame_number=frame_count,
                    fps=job.fps,
                    width=job.width,
                    height=job.height,
                    screenshot_manager=screenshot_manager,
                    speed_estimator=speed_estimator,
                    active_tracks=active_tracks,
                    lost_tracks=lost_tracks
                )
                
                # Update job statistics
                job.processed_frames = frame_count
                job.total_detections += frame_result.detections_count
                job.total_tracks = frame_result.active_tracks
                job.plate_detections += frame_result.plate_detections
                job.screenshots_saved += frame_result.screenshots_saved
                job.violations_count = frame_result.violations_count
                job.frame_results.append(frame_result)
                
                # Prepare annotated frame for live preview (downstream WS encoding)
                annotated_frame = self._draw_frame(
                    frame.copy(),
                    frame_result,
                    screenshot_manager,
                    speed_estimator
                )
                frame_result.last_frame = annotated_frame

                # Call progress callback (for WebSocket streaming)
                if progress_callback:
                    progress_percentage = (frame_count / max_frames_to_process) * 100
                    progress_callback(job, frame_result, progress_percentage)
                # Clear heavy data after callback to free memory
                frame_result.last_frame = None
                
                # Save output frame if requested
                if out:
                    out.write(annotated_frame)
                
                # Log progress
                if frame_count % 100 == 0:
                    logger.info(f"[{job_id}] Processed {frame_count}/{max_frames_to_process} "
                              f"| Tracks: {len(frame_result.tracks)} "
                              f"| Violations: {job.violations_count}")
            
            # Finalize remaining tracks
            for track_id in active_tracks:
                if track_id not in lost_tracks:
                    speed_estimator.finalize_track(track_id)
            
            # Export speed data
            speed_data_filename = f"{job_id}_speed_data.json"
            speed_data_path = self.speed_output_dir / speed_data_filename
            speed_estimator.export_json(str(speed_data_path))
            job.speed_data_path = str(speed_data_path)
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            
            # Mark job as completed
            job.status = "completed"
            job.end_time = datetime.now()
            
            logger.info(f"✅ Job {job_id} completed successfully")
            logger.info(f"   Frames: {job.processed_frames}")
            logger.info(f"   Tracks: {job.total_tracks}")
            logger.info(f"   Plates: {job.plate_detections}")
            logger.info(f"   Violations: {job.violations_count}")
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.end_time = datetime.now()
            logger.error(f"❌ Job {job_id} failed: {e}", exc_info=True)
            raise
        
        return job
    
    def process_video_stream(
        self,
        video_path: Path,
        job_id: str,
        max_frames: Optional[int] = None
    ) -> Generator[FrameResult, None, None]:
        """
        Stream processing results frame-by-frame (for WebSocket).
        
        This is a generator that yields results as they're processed.
        Perfect for real-time dashboard updates.
        
        Args:
            video_path: Path to video
            job_id: Job identifier
            max_frames: Max frames to process
        
        Yields:
            FrameResult for each processed frame
        """
        # Initialize components
        screenshot_manager = PlateScreenshotManager(
            output_dir=str(self.output_dir / job_id),
            perfect_quality_threshold=0.95,
            max_perfect_quality_shots=3,
            min_quality_threshold=0.6,
            verbose=False
        )
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize speed estimator
        speed_estimator = SpeedEstimator(
            frame_width_meters=self.frame_width_meters,
            frame_height_meters=self.frame_height_meters,
            fps=fps,
            speed_limits=self.speed_limits,
            min_frames_for_speed=10,
            smoothing_window=5,
            ignore_edge_frames=3,
            min_distance_threshold=0.5,
            speed_unit='kmh',
            output_dir='.',  # ✅ Fixed - let SpeedEstimator handle its own path
            outlier_rejection=True,
            verbose=False
        )
        speed_estimator.set_frame_dimensions(width, height)
        
        frame_count = 0
        active_tracks = set()
        lost_tracks = set()
        max_frames_to_process = max_frames or total_frames
        
        try:
            while frame_count < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process and yield result
                result = self._process_frame(
                    frame=frame,
                    frame_number=frame_count,
                    fps=fps,
                    width=width,
                    height=height,
                    screenshot_manager=screenshot_manager,
                    speed_estimator=speed_estimator,
                    active_tracks=active_tracks,
                    lost_tracks=lost_tracks
                )
                
                yield result
        
        finally:
            cap.release()
    
    def _process_frame(
        self,
        frame,
        frame_number: int,
        fps: int,
        width: int,
        height: int,
        screenshot_manager: PlateScreenshotManager,
        speed_estimator: SpeedEstimator,
        active_tracks: set,
        lost_tracks: set
    ) -> FrameResult:
        """
        Process single frame through complete pipeline.
        
        This is adapted from your test_full_pipeline logic.
        """
        timestamp = frame_number / fps if fps > 0 else 0
        
        # Step 1: Detect vehicles
        detections = self.vehicle_detector.detect_vehicles(frame)
        
        # Step 2: Update tracker
        tracks = self.vehicle_tracker.update(detections, frame_number)
        
        # Track active IDs
        current_frame_tracks = set()
        for track in tracks:
            track_id = track['track_id']
            current_frame_tracks.add(track_id)
            active_tracks.add(track_id)
            
            # Update speed estimator
            speed_estimator.update(
                track_id=track_id,
                bbox=track['bbox'],
                class_name=track['class'],
                frame_id=frame_number,
                confidence=track['confidence']
            )
        
        # Detect lost tracks
        if frame_number > 1:
            previous_tracks = active_tracks - lost_tracks
            newly_lost = previous_tracks - current_frame_tracks
            
            for track_id in newly_lost:
                speed_estimator.finalize_track(track_id)
                lost_tracks.add(track_id)
        
        # Step 3: Process plates (every 3 frames to save resources)
        plate_detections_count = 0
        screenshots_saved_count = 0
        
        if self.plate_detector and frame_number % 3 == 0:
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                class_name = track['class']
                
                # Extract vehicle crop
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                vehicle_crop = frame[y1:y2, x1:x2]
                
                if vehicle_crop.size == 0:
                    continue
                
                # Detect plate
                plate_detection = self.plate_detector.detect(vehicle_crop)
                
                if plate_detection:
                    plate_detections_count += 1
                    
                    # Extract plate crop
                    plate_crop = self.plate_detector.extract_plate_crop(
                        vehicle_crop,
                        plate_detection,
                        padding=5
                    )
                    
                    # Assess quality
                    metrics = self.quality_assessor.assess(plate_crop)
                    
                    # Save if quality is good
                    was_saved, _ = screenshot_manager.save_if_better(
                        track_id=track_id,
                        class_name=class_name,
                        plate_crop=plate_crop,
                        quality_score=metrics.overall_score,
                        frame_id=frame_number
                    )
                    
                    if was_saved:
                        screenshots_saved_count += 1

        
        # Get violations
        violations = speed_estimator.get_violations()
        
        # Create result
        result = FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections_count=len(detections),
            tracked_count=len(tracks),
            active_tracks=len(active_tracks),
            plate_detections=plate_detections_count,
            screenshots_saved=screenshots_saved_count,
            violations_count=len(violations),
            tracks=tracks,
            violations=violations
        )
        
        # Draw annotated frame and JPEG-encode for lightweight transport
        annotated_frame = self._draw_frame(
            frame.copy(),
            result,
            screenshot_manager,
            speed_estimator
        )
        success, buf = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            result.frame_data = buf.tobytes()
        
        return result
    
    def _draw_frame(
        self,
        frame,
        frame_result: FrameResult,
        screenshot_manager: PlateScreenshotManager,
        speed_estimator: SpeedEstimator
    ):
        """Draw tracks with info on frame (adapted from your code)."""
        colors = {
            'car': (0, 255, 0),
            'bus': (255, 0, 0),
            'truck': (0, 255, 255),
            'motorcycle': (0, 0, 255),
            'bicycle': (255, 0, 255)
        }
        
        for track in frame_result.tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            class_name = track['class']
            confidence = track['confidence']
            
            color = colors.get(class_name, (255, 255, 255))
            
            # Check violation
            is_violation = any(v['track_id'] == track_id 
                             for v in frame_result.violations)
            
            if is_violation:
                color = (0, 0, 255)
            
            thickness = 3 if is_violation else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Get plate info
            plate_info = ""
            screenshots = screenshot_manager.get_track_screenshots(track_id)
            if screenshots:
                plate_info = f" | {len(screenshots)}P"
            
            # Get speed info
            speed_info = ""
            current_speed = speed_estimator.get_current_speed(track_id)
            if current_speed is not None:
                speed_info = f" | {current_speed:.0f}km/h"
                if is_violation:
                    speed_info = f" | ⚠️{current_speed:.0f}km/h"
            
            id_text = f"ID:{track_id}{plate_info}{speed_info}"
            
            # Draw background
            (text_w, text_h), _ = cv2.getTextSize(id_text, 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-text_h-40), 
                         (x1+text_w+10, y1-35), (0, 0, 0), -1)
            
            text_color = (0, 0, 255) if is_violation else (255, 255, 255)
            cv2.putText(frame, id_text, (x1+5, y1-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Draw class
            label = f"{class_name} {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-h-10), (x1+w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame