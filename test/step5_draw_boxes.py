# step5_draw_boxes.py

from ultralytics import YOLO
import cv2
import urllib.request

print("Step 5: Draw Bounding Boxes")
print("="*50)
print()

# 1. Load model and image
print("📦 Loading model...")
model = YOLO('yolov8n.pt')

print("🌐 Downloading test image...")
url = "https://ultralytics.com/images/bus.jpg"
image_path = "test_image.jpg"
urllib.request.urlretrieve(url, image_path)

# 2. Read image with OpenCV
print("📸 Reading image...")
image = cv2.imread(image_path)
print(f"✅ Image size: {image.shape[1]}×{image.shape[0]} pixels")
print()

# 3. Detect objects
print("🔍 Detecting objects...")
results = model(image_path)
detections = results[0].boxes
print(f"✅ Found {len(detections)} objects")
print()

# 4. Draw boxes on image
print("🎨 Drawing bounding boxes...")

for box in detections:
    # Get coordinates
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Get class name and confidence
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    confidence = float(box.conf[0])
    
    # Choose color based on object type
    # BGR format (Blue, Green, Red)
    if class_name in ['car', 'truck']:
        color = (0, 255, 0)      # Green for cars
    elif class_name == 'bus':
        color = (255, 0, 0)      # Blue for buses
    elif class_name in ['motorcycle', 'bicycle']:
        color = (0, 0, 255)      # Red for 2-wheelers
    else:
        color = (255, 255, 255)  # White for others
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Create label text
    label = f"{class_name} {confidence:.2f}"
    
    # Draw label background (filled rectangle)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(
        image,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0], y1),
        color,
        -1  # -1 means filled
    )
    
    # Draw label text
    cv2.putText(
        image,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),  # White text
        2
    )
    
    print(f"  ✓ Drew box for: {class_name}")

print()

# 5. Save result
output_path = "output_with_boxes.jpg"
cv2.imwrite(output_path, image)
print(f"💾 Saved result: {output_path}")

# 6. Display image
print("👁️  Displaying image...")
print("   Press ANY KEY to close the window")
cv2.imshow('Vehicle Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print()
print("="*50)
print("✅ Step 5 Complete!")