"""
License Plate Quality Assessor
Evaluates plate crop quality before OCR to optimize processing
"""

import cv2
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Container for quality assessment metrics"""
    sharpness: float
    brightness: float
    contrast: float
    size_score: float
    aspect_ratio_score: float
    overall_score: float
    is_good_quality: bool
    
    def to_dict(self) -> Dict:
        return {
            'sharpness': self.sharpness,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'size_score': self.size_score,
            'aspect_ratio_score': self.aspect_ratio_score,
            'overall_score': self.overall_score,
            'is_good_quality': self.is_good_quality
        }


class PlateQualityAssessor:
    """
    Assesses license plate crop quality using multiple metrics:
    - Sharpness (Laplacian variance)
    - Brightness (mean intensity)
    - Contrast (standard deviation)
    - Size (minimum dimensions)
    - Aspect ratio (width/height ratio)
    """
    
    def __init__(
        self,
        sharpness_threshold: float = 100.0,
        brightness_range: Tuple[float, float] = (40, 220),
        contrast_threshold: float = 30.0,
        min_width: int = 80,
        min_height: int = 25,
        aspect_ratio_range: Tuple[float, float] = (2.0, 6.0),
        overall_threshold: float = 0.6,
        weights: Dict[str, float] = None
    ):
        """
        Initialize Quality Assessor
        
        Args:
            sharpness_threshold: Minimum Laplacian variance (higher = sharper)
            brightness_range: Acceptable brightness range (min, max)
            contrast_threshold: Minimum standard deviation (higher = more contrast)
            min_width: Minimum plate width in pixels
            min_height: Minimum plate height in pixels
            aspect_ratio_range: Expected width/height ratio range (min, max)
            overall_threshold: Minimum overall score to pass (0-1)
            weights: Custom weights for each metric (defaults provided)
        """
        self.sharpness_threshold = sharpness_threshold
        self.brightness_min, self.brightness_max = brightness_range
        self.contrast_threshold = contrast_threshold
        self.min_width = min_width
        self.min_height = min_height
        self.aspect_ratio_min, self.aspect_ratio_max = aspect_ratio_range
        self.overall_threshold = overall_threshold
        
        # Default weights (can be tuned based on your data)
        self.weights = weights or {
            'sharpness': 0.35,
            'brightness': 0.15,
            'contrast': 0.20,
            'size': 0.15,
            'aspect_ratio': 0.15
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def assess(self, plate_crop: np.ndarray) -> QualityMetrics:
        """
        Assess quality of a license plate crop
        
        Args:
            plate_crop: BGR image of license plate crop
            
        Returns:
            QualityMetrics object with all scores and final decision
        """
        if plate_crop is None or plate_crop.size == 0:
            return self._create_failed_metrics("Empty or None image")
        
        # Get dimensions
        height, width = plate_crop.shape[:2]
        
        # Calculate individual metrics
        sharpness = self._calculate_sharpness(plate_crop)
        brightness = self._calculate_brightness(plate_crop)
        contrast = self._calculate_contrast(plate_crop)
        size_score = self._calculate_size_score(width, height)
        aspect_ratio_score = self._calculate_aspect_ratio_score(width, height)
        
        # Calculate weighted overall score (0-1)
        overall_score = (
            self.weights['sharpness'] * sharpness +
            self.weights['brightness'] * brightness +
            self.weights['contrast'] * contrast +
            self.weights['size'] * size_score +
            self.weights['aspect_ratio'] * aspect_ratio_score
        )
        
        # Final decision
        is_good_quality = overall_score >= self.overall_threshold
        
        return QualityMetrics(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            size_score=size_score,
            aspect_ratio_score=aspect_ratio_score,
            overall_score=overall_score,
            is_good_quality=is_good_quality
        )
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate sharpness using Laplacian variance
        Higher values = sharper image
        Returns normalized score (0-1)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range using threshold
        # Score = 1.0 if variance >= threshold, proportional below
        score = min(variance / self.sharpness_threshold, 1.0)
        
        return score
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """
        Calculate brightness (mean intensity)
        Returns score based on optimal range (0-1)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mean_brightness = gray.mean()
        
        # Score based on proximity to optimal range
        if mean_brightness < self.brightness_min:
            # Too dark
            score = mean_brightness / self.brightness_min
        elif mean_brightness > self.brightness_max:
            # Too bright
            score = 1.0 - (mean_brightness - self.brightness_max) / (255 - self.brightness_max)
        else:
            # Within optimal range
            score = 1.0
        
        return max(0.0, min(score, 1.0))
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate contrast using standard deviation
        Higher values = more contrast
        Returns normalized score (0-1)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        std_dev = gray.std()
        
        # Normalize to 0-1 range using threshold
        score = min(std_dev / self.contrast_threshold, 1.0)
        
        return score
    
    def _calculate_size_score(self, width: int, height: int) -> float:
        """
        Calculate size adequacy score
        Returns 1.0 if above minimum, proportional below
        """
        width_score = min(width / self.min_width, 1.0) if self.min_width > 0 else 1.0
        height_score = min(height / self.min_height, 1.0) if self.min_height > 0 else 1.0
        
        # Use minimum of both dimensions
        score = min(width_score, height_score)
        
        return score
    
    def _calculate_aspect_ratio_score(self, width: int, height: int) -> float:
        """
        Calculate aspect ratio score
        Returns 1.0 if within expected range, proportional outside
        """
        if height == 0:
            return 0.0
        
        aspect_ratio = width / height
        
        # Score based on proximity to expected range
        if aspect_ratio < self.aspect_ratio_min:
            # Too narrow
            score = aspect_ratio / self.aspect_ratio_min
        elif aspect_ratio > self.aspect_ratio_max:
            # Too wide
            score = 1.0 - (aspect_ratio - self.aspect_ratio_max) / self.aspect_ratio_max
        else:
            # Within expected range
            score = 1.0
        
        return max(0.0, min(score, 1.0))
    
    def _create_failed_metrics(self, reason: str = "") -> QualityMetrics:
        """Create a failed quality metrics object"""
        return QualityMetrics(
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            size_score=0.0,
            aspect_ratio_score=0.0,
            overall_score=0.0,
            is_good_quality=False
        )
    
    def batch_assess(self, plate_crops: list) -> list:
        """
        Assess quality for multiple plate crops
        
        Args:
            plate_crops: List of plate crop images
            
        Returns:
            List of QualityMetrics objects
        """
        return [self.assess(crop) for crop in plate_crops]
    
    def get_best_crop(self, plate_crops: list) -> Tuple[int, QualityMetrics]:
        """
        Find the best quality crop from a list
        
        Args:
            plate_crops: List of plate crop images
            
        Returns:
            Tuple of (best_index, best_metrics)
        """
        if not plate_crops:
            return -1, self._create_failed_metrics("Empty list")
        
        metrics_list = self.batch_assess(plate_crops)
        
        best_idx = max(range(len(metrics_list)), 
                      key=lambda i: metrics_list[i].overall_score)
        
        return best_idx, metrics_list[best_idx]


