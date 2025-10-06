"""
License Plate Detector
Uses YOLOv8 model (yasirfaizahmed/license-plate-object-detection) 
to detect license plates within vehicle crops
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class PlateDetection:
    """Container for plate detection results"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) relative to vehicle crop
    confidence: float
    center: Tuple[int, int]  # (cx, cy) center point
    width: int
    height: int
    area: int
    
    def to_dict(self) -> Dict:
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'center': self.center,
            'width': self.width,
            'height': self.height,
            'area': self.area
        }


class LicensePlateDetector:
    """
    Detects license plates within vehicle crops using YOLOv8
    Model: yasirfaizahmed/license-plate-object-detection
    """
    
    def __init__(
        self,
        model_path: str = "best.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        device: str = "cuda",
        img_size: int = 640,
        min_plate_area: int = 500,
        max_detections: int = 1
    ):
        """
        Initialize License Plate Detector
        
        Args:
            model_path: Path to model weights or HuggingFace model ID
            confidence_threshold: Minimum confidence for detections (0-1)
            iou_threshold: IoU threshold for NMS (Non-Maximum Suppression)
            device: Device to run inference on ('cuda' or 'cpu')
            img_size: Input image size for model (640 is standard for YOLOv8)
            min_plate_area: Minimum plate area in pixels (filters tiny detections)
            max_detections: Maximum number of plates to return per vehicle (usually 1)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.img_size = img_size
        self.min_plate_area = min_plate_area
        self.max_detections = max_detections
        
        # Load YOLOv8 model
        print(f"Loading license plate detection model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            print(f"✓ Model loaded successfully on {device}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
        
        # Statistics tracking
        self.stats = {
            'total_inferences': 0,
            'detections_found': 0,
            'detections_filtered': 0
        }
    
    def detect(self, vehicle_crop: np.ndarray) -> Optional[PlateDetection]:
        """
        Detect license plate in a vehicle crop
        
        Args:
            vehicle_crop: BGR image of vehicle (cropped from full frame)
            
        Returns:
            PlateDetection object if plate found, None otherwise
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None
        
        self.stats['total_inferences'] += 1
        
        # Run YOLOv8 inference
        results = self.model.predict(
            vehicle_crop,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
            device=self.device
        )
        
        # Extract detections from results
        detections = self._parse_results(results, vehicle_crop.shape)
        
        if not detections:
            return None
        
        # Return the best detection (highest confidence)
        return detections[0]
    
    def detect_batch(self, vehicle_crops: List[np.ndarray]) -> List[Optional[PlateDetection]]:
        """
        Detect license plates in multiple vehicle crops (batch processing)
        More efficient for processing multiple vehicles
        
        Args:
            vehicle_crops: List of vehicle crop images
            
        Returns:
            List of PlateDetection objects (None for vehicles without plates)
        """
        if not vehicle_crops:
            return []
        
        self.stats['total_inferences'] += len(vehicle_crops)
        
        # Run batch inference
        results = self.model.predict(
            vehicle_crops,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
            device=self.device
        )
        
        # Parse results for each crop
        detections_list = []
        for i, result in enumerate(results):
            crop_shape = vehicle_crops[i].shape
            detections = self._parse_results([result], crop_shape)
            detections_list.append(detections[0] if detections else None)
        
        return detections_list
    
    def _parse_results(
        self, 
        results, 
        image_shape: Tuple[int, int, int]
    ) -> List[PlateDetection]:
        """
        Parse YOLOv8 results into PlateDetection objects
        
        Args:
            results: YOLOv8 prediction results
            image_shape: Shape of input image (H, W, C)
            
        Returns:
            List of PlateDetection objects, sorted by confidence (descending)
        """
        detections = []
        
        # Get first result (single image inference)
        result = results[0]
        
        # Check if any detections exist
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        # Extract bounding boxes and confidences
        boxes = result.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2) format
        confidences = result.boxes.conf.cpu().numpy()
        
        # Process each detection
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            # Calculate additional properties
            width = x2 - x1
            height = y2 - y1
            area = width * height
            center = (x1 + width // 2, y1 + height // 2)
            
            # Filter by minimum area
            if area < self.min_plate_area:
                self.stats['detections_filtered'] += 1
                continue
            
            # Ensure bbox is within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_shape[1], x2)
            y2 = min(image_shape[0], y2)
            
            detection = PlateDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                center=center,
                width=width,
                height=height,
                area=area
            )
            
            detections.append(detection)
            self.stats['detections_found'] += 1
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        # Return top N detections
        return detections[:self.max_detections]
    
    def extract_plate_crop(
        self, 
        vehicle_crop: np.ndarray, 
        plate_detection: PlateDetection,
        padding: int = 5
    ) -> np.ndarray:
        """
        Extract plate crop from vehicle crop using detection bbox
        
        Args:
            vehicle_crop: Original vehicle crop image
            plate_detection: PlateDetection object with bbox
            padding: Extra pixels to add around plate (helps with OCR)
            
        Returns:
            Cropped plate image
        """
        x1, y1, x2, y2 = plate_detection.bbox
        h, w = vehicle_crop.shape[:2]
        
        # Add padding (with bounds checking)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Extract crop
        plate_crop = vehicle_crop[y1:y2, x1:x2]
        
        return plate_crop
    
    def visualize_detection(
        self, 
        vehicle_crop: np.ndarray, 
        plate_detection: PlateDetection,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection bbox on vehicle crop for visualization
        
        Args:
            vehicle_crop: Original vehicle crop image
            plate_detection: PlateDetection object
            color: BGR color for bbox
            thickness: Line thickness
            
        Returns:
            Image with drawn bbox
        """
        vis_img = vehicle_crop.copy()
        x1, y1, x2, y2 = plate_detection.bbox
        
        # Draw rectangle
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw confidence label
        label = f"Plate: {plate_detection.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            vis_img, 
            (x1, y1 - label_size[1] - 5), 
            (x1 + label_size[0], y1), 
            color, 
            -1
        )
        cv2.putText(
            vis_img, 
            label, 
            (x1, y1 - 3), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1
        )
        
        return vis_img
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        stats = self.stats.copy()
        if stats['total_inferences'] > 0:
            stats['detection_rate'] = stats['detections_found'] / stats['total_inferences']
        else:
            stats['detection_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'total_inferences': 0,
            'detections_found': 0,
            'detections_filtered': 0
        }


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = LicensePlateDetector(
        model_path="best.pt",
        confidence_threshold=0.5,
        device="cuda"  # Use "cpu" if no GPU
    )
    
    # Test with a sample vehicle crop (load your actual image here)
    # For demonstration, create a synthetic vehicle image
    test_vehicle = np.ones((300, 400, 3), dtype=np.uint8) * 100
    
    # Simulate a plate region (white rectangle)
    cv2.rectangle(test_vehicle, (150, 200), (330, 250), (255, 255, 255), -1)
    cv2.putText(test_vehicle, "ABC-1234", (160, 235), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Detect plate
    plate_detection = detector.detect(test_vehicle)
    print(f"aw lenna{plate_detection}")
    
    if plate_detection:
        print("✓ Plate detected!")
        print(f"  BBox: {plate_detection.bbox}")
        print(f"  Confidence: {plate_detection.confidence:.3f}")
        print(f"  Size: {plate_detection.width}x{plate_detection.height}")
        
        # Extract plate crop
        plate_crop = detector.extract_plate_crop(test_vehicle, plate_detection)
        print(f"  Plate crop shape: {plate_crop.shape}")
        
        # Visualize
        vis_img = detector.visualize_detection(test_vehicle, plate_detection)
        cv2.imwrite("plate_detection_result.jpg", vis_img)
        print("  Visualization saved to: plate_detection_result.jpg")
    else:
        print("✗ No plate detected")
    
    # Print statistics
    print("\nDetector Statistics:")
    stats = detector.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")