"""
Test VehicleDetector + ByteTrack Tracker + Plate Screenshot Pipeline
Saves high-quality plate images instead of running OCR
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


def test_full_pipeline(video_path):
    """Test detector + tracker + plate screenshot pipeline on video"""
    
    print("\n" + "="*70)
    print("TESTING PIPELINE: DETECTION + TRACKING + PLATE SCREENSHOTS")
    print("="*70 + "\n")
    
    # Check video exists
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        print(f"Available files in data/input:")
        input_dir = Path("data/input")
        if input_dir.exists():
            for file in input_dir.iterdir():
                print(f"  - {file.name}")
        return False
    
    # Create output directory for plate screenshots
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
    
    # Open video
    print("Step 5: Opening video...")
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
    
    # Setup output video
    output_path = "output_full_pipeline.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Track data storage
    track_data = {}  # {track_id: {'screenshots': [], 'class': str, 'best_quality': float}}
    
    print("Step 6: Processing video...")
    print("Press 'q' to stop\n")
    
    frame_count = 0
    total_detections = 0
    total_tracks = 0
    active_tracks = set()
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
        
        # Track statistics
        for track in tracks:
            active_tracks.add(track['track_id'])
        total_tracks = len(active_tracks)
        
        # Step 3: Process plate detection and save screenshots
        if plate_detector:
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                class_name = track['class']
                
                # Initialize track data
                if track_id not in track_data:
                    track_data[track_id] = {
                        'screenshots': [],
                        'class': class_name,
                        'best_quality': 0.0,
                        'best_screenshot': None
                    }
                
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
                    
                    # Save screenshot if quality is good
                    if metrics.is_good_quality:
                        # Create track-specific directory
                        track_dir = output_dir / f"{class_name}_{track_id}"
                        track_dir.mkdir(exist_ok=True)
                        
                        # Generate filename
                        filename = f"frame_{frame_count:06d}_q{metrics.overall_score:.2f}.jpg"
                        filepath = track_dir / filename
                        
                        # Save screenshot
                        cv2.imwrite(str(filepath), plate_crop)
                        screenshots_saved += 1
                        
                        # Store info
                        track_data[track_id]['screenshots'].append({
                            'filepath': str(filepath),
                            'quality': metrics.overall_score,
                            'frame_id': frame_count
                        })
                        
                        print(f"  Track {track_id} ({class_name}): Saved plate screenshot "
                              f"(quality: {metrics.overall_score:.2f}, frame: {frame_count})")
                        
                        # Update best quality screenshot
                        if metrics.overall_score > track_data[track_id]['best_quality']:
                            track_data[track_id]['best_quality'] = metrics.overall_score
                            track_data[track_id]['best_screenshot'] = str(filepath)
        
        # Draw results
        output_frame = draw_tracks_with_info(frame.copy(), tracks, track_data)
        
        # Add info overlay
        info_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Detections: {len(detections)}",
            f"Tracked: {len(tracks)}",
            f"Total IDs: {total_tracks}",
            f"Plates: {plate_detections}",
            f"Screenshots: {screenshots_saved}"
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
        cv2.imshow('Pipeline: Detection + Tracking + Plate Screenshots', output_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"Tracks: {len(tracks)} | Screenshots: {screenshots_saved}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
    
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
    print(f"Screenshots saved: {screenshots_saved}")
    print(f"\nPlate screenshots per track:")
    print("-" * 70)
    
    for track_id, data in sorted(track_data.items()):
        num_screenshots = len(data['screenshots'])
        if num_screenshots > 0:
            best_quality = data['best_quality']
            print(f"  Track {track_id} ({data['class']}): {num_screenshots} screenshots "
                  f"(best quality: {best_quality:.2f})")
            print(f"    Best: {data['best_screenshot']}")
    
    print("-" * 70)
    print(f"Total tracks with plates: {sum(1 for d in track_data.values() if d['screenshots'])}/{total_tracks}")
    print(f"Output video: {output_path}")
    print(f"Screenshots directory: {output_dir}")
    print("="*70 + "\n")
    
    return True


def draw_tracks_with_info(frame, tracks, track_data):
    """Draw tracked vehicles with IDs and plate detection status"""
    
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
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Get plate info
        plate_info = ""
        if track_id in track_data:
            data = track_data[track_id]
            num_screenshots = len(data['screenshots'])
            if num_screenshots > 0:
                plate_info = f" | {num_screenshots} plates"
                if data['best_quality'] > 0:
                    plate_info += f" (Q:{data['best_quality']:.2f})"
        
        # Draw track ID and plate info
        id_text = f"ID:{track_id}{plate_info}"
        
        # Background for better visibility
        (text_w, text_h), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1-text_h-40), (x1+text_w+10, y1-35), (0, 0, 0), -1)
        
        cv2.putText(
            frame, id_text, (x1+5, y1-40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
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
    # Point to your video
    video_path = "data/input/test1.mp4"
    
    # Auto-detect first video in data/input
    """input_dir = Path("data/input")
    if input_dir.exists():
        videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
        if videos:
            video_path = str(videos[0])
            print(f"Using video: {video_path}")"""
    
    success = test_full_pipeline(video_path)
    sys.exit(0 if success else 1)