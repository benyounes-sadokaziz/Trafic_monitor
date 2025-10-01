"""
Test VehicleDetector + ByteTrack Tracker together on video
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detection.vehicle_detector import VehicleDetector
from tracking.bytetrack_tracker import VehicleTracker


def test_detection_and_tracking(video_path):
    """Test detector + tracker on video"""
    
    print("\n" + "="*70)
    print("TESTING VEHICLE DETECTION + TRACKING")
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
    
    # Initialize detector
    print("Step 1: Initializing detector...")
    detector = VehicleDetector(
        model_path='yolov8n.pt',
        confidence_threshold=0.5
    )
    detector.load_model()
    print("Detector ready\n")
    
    # Initialize tracker
    print("Step 2: Initializing tracker...")
    try:
        tracker = VehicleTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8
        )
        print("Tracker ready\n")
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install supervision")
        return False
    
    # Open video
    print("Step 3: Opening video...")
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
    output_path = "output_detection_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Step 4: Processing video...")
    print("Press 'q' to stop\n")
    
    frame_count = 0
    total_detections = 0
    total_tracks = 0
    active_tracks = set()
    
    while True:
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
        
        # Draw results
        output_frame = draw_tracks(frame.copy(), tracks)
        
        # Add info overlay
        info_text = [
            f"Frame: {frame_count}/{total_frames}",
            f"Detections: {len(detections)}",
            f"Tracked: {len(tracks)}",
            f"Total IDs: {total_tracks}"
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
        cv2.imshow('Detection + Tracking', output_frame)
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"Detections: {len(detections)} | Tracks: {len(tracks)}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Final summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Processed frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Unique vehicle IDs tracked: {total_tracks}")
    print(f"Average detections/frame: {total_detections/frame_count:.1f}")
    print(f"Average tracks/frame: {len(active_tracks)/frame_count if frame_count > 0 else 0:.1f}")
    print(f"Output saved: {output_path}")
    print("="*70 + "\n")
    
    return True


def draw_tracks(frame, tracks):
    """Draw tracked vehicles with IDs"""
    
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
        
        # Draw track ID (larger, on the box)
        id_text = f"ID:{track_id}"
        cv2.putText(
            frame, id_text, (x1, y1 - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
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
    video_path = "data/input/your_video.mp4"
    
    # Auto-detect first video in data/input
    input_dir = Path("data/input")
    if input_dir.exists():
        videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
        if videos:
            video_path = str(videos[0])
            print(f"Using video: {video_path}")
    
    success = test_detection_and_tracking(video_path)
    sys.exit(0 if success else 1)