"""
Test VehicleDetector on video file
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detection.vehicle_detector import VehicleDetector


def test_on_video(video_path):
    """Test detector on video file"""
    
    print("\n" + "="*60)
    print("TESTING VEHICLE DETECTOR ON VIDEO")
    print("="*60 + "\n")
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return False
    
    # Initialize detector
    print("Step 1: Initializing detector...")
    detector = VehicleDetector(
        model_path='yolov8n.pt',
        confidence_threshold=0.5
    )
    detector.load_model()
    print("Model loaded\n")
    
    # Open video
    print("Step 2: Opening video...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds\n")
    
    # Setup output video
    output_path = "output_detector_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Step 3: Processing video...")
    print("(Press 'q' to stop early)\n")
    
    frame_count = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect vehicles
        detections = detector.detect_vehicles(frame)
        total_detections += len(detections)
        
        # Draw detections
        output_frame = draw_detections(frame.copy(), detections)
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames} | Vehicles: {len(detections)}"
        cv2.putText(
            output_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Write to output
        out.write(output_frame)
        
        # Display progress
        if frame_count % 30 == 0:  # Every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Show frame (optional, can slow down processing)
        cv2.imshow('Detection', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Processed frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {total_detections/frame_count:.1f}")
    print(f"Output saved: {output_path}")
    print("="*60 + "\n")
    
    return True


def draw_detections(frame, detections):
    """Draw boxes on frame"""
    
    colors = {
        'car': (0, 255, 0),
        'bus': (255, 0, 0),
        'truck': (0, 255, 255),
        'motorcycle': (0, 0, 255),
        'bicycle': (255, 0, 255)
    }
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        color = colors.get(class_name, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        cv2.rectangle(frame, (x1, y1-h-10), (x1+w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    
    return frame


if __name__ == "__main__":
    # Update this path to your video
    video_path = "data/input/your_video.mp4"
    
    success = test_on_video(video_path)
    sys.exit(0 if success else 1)