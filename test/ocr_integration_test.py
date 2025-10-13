"""
Complete Vehicle Detection Pipeline:
- Vehicle Detection (YOLOv8)
- Multi-object Tracking (ByteTrack)
- License Plate Detection & Screenshot Saving
- Speed Estimation with Violation Detection
"""

import cv2
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detection.vehicle_detector import VehicleDetector
from tracking.bytetrack_tracker import VehicleTracker
from ocr.plate_detector import LicensePlateDetector
from ocr.quality_assessor import PlateQualityAssessor
from ocr.plate_screenshot_manager import PlateScreenshotManager
from speed.speed_estimator import SpeedEstimator


def test_full_pipeline(
    video_path,
    frame_width_meters=50.0,
    frame_height_meters=30.0,
    speed_limits=None
):
    """
    Test complete pipeline on video with speed estimation.
    
    Args:
        video_path: Path to input video
        frame_width_meters: Real-world width the camera sees (meters)
        frame_height_meters: Real-world height the camera sees (meters)
        speed_limits: Dict of speed limits per vehicle class (km/h)
    """
    
    print("\n" + "="*70)
    print("TESTING COMPLETE PIPELINE")
    print("DETECTION + TRACKING + PLATES + SPEED ESTIMATION")
    print("="*70 + "\n")
    
    # Default speed limits if not provided
    if speed_limits is None:
        speed_limits = {
            'car': 120,
            'truck': 90,
            'bus': 90,
            'motorcycle': 120,
            'bicycle': 30
        }
    
    # Check video exists
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        print(f"Available files in data/input:")
        input_dir = Path("data/input")
        if input_dir.exists():
            for file in input_dir.iterdir():
                print(f"  - {file.name}")
        return False
    
    # Create output directories
    output_dir = Path("data/output/plate_screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plate screenshots will be saved to: {output_dir}\n")
    
    # Initialize detector
    print("Step 1: Initializing vehicle detector...")
    detector = VehicleDetector(
        model_path='yolov8n.pt',
        confidence_threshold=0.5
    )
    detector.load_model()
    print("✓ Vehicle detector ready\n")
    
    # Initialize tracker
    print("Step 2: Initializing tracker...")
    try:
        tracker = VehicleTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8
        )
        print("✓ Tracker ready\n")
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install supervision")
        return False
    
    # Initialize plate detector
    print("Step 3: Initializing plate detector...")
    try:
        plate_detector = LicensePlateDetector(
            model_path="best.pt",  # Update path if needed
            confidence_threshold=0.5,
            device="cuda"
        )
        print("✓ Plate detector ready\n")
    except Exception as e:
        print(f"Warning: Could not load plate detector: {e}")
        print("Continuing without plate detection...")
        plate_detector = None
    
    # Initialize quality assessor
    print("Step 4: Initializing quality assessor...")
    quality_assessor = PlateQualityAssessor(
        overall_threshold=0.6
    )
    print("✓ Quality assessor ready\n")
    
    # Initialize screenshot manager
    print("Step 5: Initializing screenshot manager...")
    screenshot_manager = PlateScreenshotManager(
        output_dir="data/output/plate_screenshots",
        perfect_quality_threshold=0.95,
        max_perfect_quality_shots=3,
        min_quality_threshold=0.6,
        verbose=True
    )
    print("✓ Screenshot manager ready\n")
    
    # Open video
    print("Step 6: Opening video...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f}s\n")
    
    # Initialize speed estimator
    print("Step 7: Initializing speed estimator...")
    speed_estimator = SpeedEstimator(
        frame_width_meters=frame_width_meters,
        frame_height_meters=frame_height_meters,
        fps=fps,
        speed_limits=speed_limits,
        min_frames_for_speed=10,
        smoothing_window=5,
        ignore_edge_frames=3,
        min_distance_threshold=0.5,
        speed_unit='kmh',
        output_dir='data/output/speed_data',
        outlier_rejection=True,
        verbose=True
    )
    # Set frame dimensions for speed calibration
    speed_estimator.set_frame_dimensions(width, height)
    print("✓ Speed estimator ready\n")
    
    # Setup output video
    output_path = "output_full_pipeline.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Step 8: Processing video...")
    print("Press 'q' to stop\n")
    
    frame_count = 0
    total_detections = 0
    total_tracks = 0
    active_tracks = set()
    lost_tracks = set()  # Tracks that have left the frame
    plate_detections = 0
    screenshots_saved = 0

    while frame_count < 600:  # Limit to first 600 frames for testing
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Step 1: Detect vehicles
        detections = detector.detect_vehicles(frame)
        total_detections += len(detections)
        
        # Step 2: Update tracker
        tracks = tracker.update(detections, frame_count)
        
        # Track currently active tracks
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
                frame_id=frame_count,
                confidence=track['confidence']
            )
        
        # Detect lost tracks (vehicles that left the frame)
        if frame_count > 1:
            previous_tracks = active_tracks - lost_tracks
            newly_lost = previous_tracks - current_frame_tracks
            
            for track_id in newly_lost:
                # Finalize speed for lost track
                speed_estimator.finalize_track(track_id)
                lost_tracks.add(track_id)
        
        total_tracks = len(active_tracks)
        
        # Step 3: Process plate detection and save screenshots
        if plate_detector:
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                class_name = track['class']
                
                # Process every 3 frames to save GPU
                if frame_count % 3 != 0:
                    continue
                
                # Extract vehicle crop
                x1, y1, x2, y2 = bbox
                # Ensure coordinates are within frame bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                vehicle_crop = frame[y1:y2, x1:x2]
                
                if vehicle_crop.size == 0:
                    continue
                
                # Detect plate in vehicle crop
                plate_detection = plate_detector.detect(vehicle_crop)
                
                if plate_detection:
                    plate_detections += 1
                    
                    # Extract plate crop
                    plate_crop = plate_detector.extract_plate_crop(
                        vehicle_crop, 
                        plate_detection,
                        padding=5
                    )
                    
                    # Assess quality
                    metrics = quality_assessor.assess(plate_crop)
                    
                    # Save screenshot using screenshot manager
                    was_saved, reason = screenshot_manager.save_if_better(
                        track_id=track_id,
                        class_name=class_name,
                        plate_crop=plate_crop,
                        quality_score=metrics.overall_score,
                        frame_id=frame_count
                    )
                    
                    if was_saved:
                        screenshots_saved += 1
        
        # Draw results with speed info
        output_frame = draw_tracks_with_info(
            frame.copy(), 
            tracks, 
            screenshot_manager, 
            speed_estimator
        )
        
        # Add info overlay
        info_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Detections: {len(detections)}",
            f"Tracked: {len(tracks)}",
            f"Total IDs: {total_tracks}",
            f"Plates: {plate_detections}",
            f"Screenshots: {screenshots_saved}",
            f"Violations: {len(speed_estimator.get_violations())}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(
                output_frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            y_offset += 35
        
        # Write and display
        out.write(output_frame)
        cv2.imshow('Complete Pipeline: Detection + Tracking + Plates + Speed', output_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"Tracks: {len(tracks)} | Screenshots: {screenshots_saved} | "
                  f"Violations: {len(speed_estimator.get_violations())}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
    
    # Finalize any remaining tracks
    print("\n" + "="*70)
    print("FINALIZING REMAINING TRACKS...")
    print("="*70)
    for track_id in active_tracks:
        if track_id not in lost_tracks:
            speed_estimator.finalize_track(track_id)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Processed frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Unique vehicle IDs tracked: {total_tracks}")
    print(f"Plate detections: {plate_detections}")
    print(f"Output video: {output_path}")
    
    # Get detailed statistics
    screenshot_manager.print_summary()
    speed_estimator.print_summary()
    
    # Export speed data to JSON
    speed_estimator.export_json("speed_data.json")
    
    return True


def draw_tracks_with_info(frame, tracks, screenshot_manager, speed_estimator):
    """Draw tracked vehicles with IDs, plate info, and speed"""
    
    colors = {
        'car': (0, 255, 0),
        'bus': (255, 0, 0),
        'truck': (0, 255, 255),
        'motorcycle': (0, 0, 255),
        'bicycle': (255, 0, 255)
    }
    
    for track in tracks:
        x1, y1, x2, y2 = track['bbox']
        track_id = track['track_id']
        class_name = track['class']
        confidence = track['confidence']
        
        # Get color
        color = colors.get(class_name, (255, 255, 255))
        
        # Check for speed violation
        violations = speed_estimator.get_violations()
        is_violation = any(v['track_id'] == track_id for v in violations)
        
        # Use red color for violators
        if is_violation:
            color = (0, 0, 255)  # Red
        
        # Draw bounding box
        thickness = 3 if is_violation else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Get plate info from screenshot manager
        plate_info = ""
        screenshots = screenshot_manager.get_track_screenshots(track_id)
        if screenshots:
            num_screenshots = len(screenshots)
            plate_info = f" | {num_screenshots}P"
        
        # Get speed info
        speed_info = ""
        current_speed = speed_estimator.get_current_speed(track_id)
        if current_speed is not None:
            speed_info = f" | {current_speed:.0f}km/h"
            if is_violation:
                speed_info = f" | ⚠️{current_speed:.0f}km/h"
        
        # Draw track ID, plate, and speed info
        id_text = f"ID:{track_id}{plate_info}{speed_info}"
        
        # Background for better visibility
        (text_w, text_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-text_h-40), (x1+text_w+10, y1-35), (0, 0, 0), -1)
        
        text_color = (0, 0, 255) if is_violation else (255, 255, 255)
        cv2.putText(
            frame, id_text, (x1+5, y1-40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2
        )
        
        # Draw class and confidence
        label = f"{class_name} {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        cv2.rectangle(frame, (x1, y1-h-10), (x1+w, y1), color, -1)
        cv2.putText(
            frame, label, (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
        
        # Draw center point
        if 'center' in track:
            cx, cy = track['center']
            cv2.circle(frame, (cx, cy), 4, color, -1)
    
    return frame


if __name__ == "__main__":
    # Video path
    video_path = "data/input/test1.mp4"
    
    # Camera calibration (adjust these for your video!)
    # These values represent the real-world area the camera sees
    FRAME_WIDTH_METERS = 50.0   # Width of road/area in frame (meters)
    FRAME_HEIGHT_METERS = 30.0  # Height of road/area in frame (meters)
    
    # Speed limits per vehicle class (km/h)
    SPEED_LIMITS = {
        'car': 120,
        'truck': 90,
        'bus': 90,
        'motorcycle': 120,
        'bicycle': 30
    }
    
    # Run pipeline
    success = test_full_pipeline(
        video_path=video_path,
        frame_width_meters=FRAME_WIDTH_METERS,
        frame_height_meters=FRAME_HEIGHT_METERS,
        speed_limits=SPEED_LIMITS
    )
    
    sys.exit(0 if success else 1)