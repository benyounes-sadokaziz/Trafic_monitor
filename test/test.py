# step2_load_yolo.py

from ultralytics import YOLO
import torch

print("Step 2: Loading YOLOv8 Model")
print("="*50)
print()

# Show which device we'll use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print()

# Load the smallest YOLOv8 model (nano version)
print("Loading YOLOv8 nano model...")
print("(First time: will download ~6MB)")
print()

model = YOLO('yolov8n.pt')

print("✅ Model loaded successfully!")
print()

# Show model info
print("Model information:")
print(f"  - Model type: YOLOv8 Nano")
print(f"  - Size: ~6 MB")
print(f"  - Speed: Fastest")
print(f"  - Can detect: 80 different object types")
print(f"  - Including: car, bus, truck, motorcycle, bicycle")