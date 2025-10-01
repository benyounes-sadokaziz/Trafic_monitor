"""
Vehicle Detection Module
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict
import logging


class VehicleDetector:
    """Detects vehicles in video frames using YOLOv8."""
    
    # Correct COCO class IDs
    VEHICLE_CLASSES = {
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        6: 'bus',
        8: 'truck'
    }
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = self._get_device()
        self.model = None
        
    def _get_device(self) -> str:
        """Get computation device."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.warning("GPU not available, using CPU (slower)")
            
        return device
    
       
    def load_model(self):
        """Load YOLOv8 model."""
        if self.model is not None:
            return  # Already loaded
            
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.logger.info(f"Model loaded: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in frame.
        
        Returns:
            List of detections with format:
            {
                'bbox': (x1, y1, x2, y2),  # Kept consistent for tracking
                'confidence': float,
                'class': str,
                'class_id': int
            }
        """
        if self.model is None:
            self.load_model()
            
        if frame is None or len(frame.shape) != 3:
            self.logger.error("Invalid frame")
            return []
            
        try:
            # Run inference with only vehicle classes
            results = self.model(
                frame,
                classes=list(self.VEHICLE_CLASSES.keys()),  # Filter at inference time
                conf=self.confidence_threshold,
                verbose=False
            )
            
            return self._process_results(results[0])
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
        
    def _process_results(self, results) -> List[Dict]:
        """Process YOLOv8 results."""
        detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return detections
            
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            detection = {
                'bbox': (x1, y1, x2, y2),  # Standard format
                'confidence': float(conf),
                'class': self.VEHICLE_CLASSES.get(class_id, 'unknown'),
                'class_id': int(class_id)
            }
            
            detections.append(detection)
                
        return detections
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "Not loaded"}
            
        return {
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "supported_classes": list(self.VEHICLE_CLASSES.values())
        }