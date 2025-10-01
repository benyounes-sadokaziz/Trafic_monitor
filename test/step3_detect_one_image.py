# step3_detect_one_image.py

from ultralytics import YOLO
import torch
import cv2
import urllib.request

print("Step 3: Detect Vehicles in Image")
print("="*50)
print()

# 1. Load model
print("📦 Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Model loaded on {device}")
print()

# 2. Download a test image
print("🌐 Downloading test image...")
url = "https://ultralytics.com/images/bus.jpg"
image_path = "test_image.jpg"
urllib.request.urlretrieve(url, image_path)
print(f"✅ Image saved as: {image_path}")
print()

# 3. Detect vehicles
print("🔍 Detecting vehicles...")
results = model(image_path)
print("✅ Detection complete!")
print()

# 4. Show what was found
print("📊 Results:")
print("-" * 50)

# Get the detections
detections = results[0].boxes
print(detections[0])

if len(detections) > 0:
    print(f"Found {len(detections)} objects:")
    print()
    
    for i, box in enumerate(detections, 1):
        # Get class ID and name
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        
        print(f"  {i}. {class_name}")
        print(f"     Confidence: {confidence:.2%}")
        print()
else:
    print("No objects detected")

print("="*50)
print("✅ Step 3 Complete!")