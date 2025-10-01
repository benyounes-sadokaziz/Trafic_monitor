# step4_bounding_boxes.py

from ultralytics import YOLO
import urllib.request

print("Step 4: Understanding Bounding Boxes")
print("="*50)
print()

# Load model and image (same as before)
model = YOLO('yolov8n.pt')
url = "https://ultralytics.com/images/bus.jpg"
image_path = "test_image.jpg"
urllib.request.urlretrieve(url, image_path)

# Detect
results = model(image_path)
detections = results[0].boxes

print(f"Found {len(detections)} objects")
print()

# Show detailed information for each detection
for i, box in enumerate(detections, 1):
    print(f"Object #{i}")
    print("-" * 40)
    
    # Get class information
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    
    print(f"  Class: {class_name}")
    print(f"  Confidence: {confidence:.2%}")
    print()
    
    # Get bounding box coordinates
    # Method 1: xyxy format (x1, y1, x2, y2)
    x1, y1, x2, y2 = box.xyxy[0]
    print(f"  📦 Bounding Box (xyxy format):")
    print(f"     Top-left corner: ({int(x1)}, {int(y1)})")
    print(f"     Bottom-right corner: ({int(x2)}, {int(y2)})")
    print()
    
    

print("="*50)
print("✅ Step 4 Complete!")