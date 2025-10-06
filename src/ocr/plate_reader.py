"""
License Plate Reader (OCR)
Uses EasyOCR to read text from license plate crops
"""

import cv2
import numpy as np
import easyocr
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class OCRReading:
    """Container for OCR reading results"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    raw_text: str  # Original text before cleaning
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'raw_text': self.raw_text
        }


class LicensePlateReader:
    """
    Reads license plate text using EasyOCR
    Includes preprocessing and post-processing for better accuracy
    """
    
    def __init__(
        self,
        languages: List[str] = ['en'],
        gpu: bool = True,
        confidence_threshold: float = 0.3,
        min_text_length: int = 4,
        max_text_length: int = 15,
        allowed_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        preprocessing: bool = True,
        paragraph: bool = False,
        batch_size: int = 1
    ):
        """
        Initialize License Plate Reader
        
        Args:
            languages: List of language codes for OCR (e.g., ['en', 'ar'])
            gpu: Use GPU for OCR if available
            confidence_threshold: Minimum confidence for OCR results (0-1)
            min_text_length: Minimum valid plate text length
            max_text_length: Maximum valid plate text length
            allowed_chars: Valid characters for license plates
            preprocessing: Apply image preprocessing before OCR
            paragraph: If False, treats image as single line of text (better for plates)
            batch_size: Number of images to process in parallel
        """
        self.confidence_threshold = confidence_threshold
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.allowed_chars = allowed_chars
        self.preprocessing = preprocessing
        
        # Initialize EasyOCR Reader
        print(f"Initializing EasyOCR with languages: {languages}")
        try:
            self.reader = easyocr.Reader(
                languages,
                gpu=gpu,
                verbose=False
            )
            self.paragraph = paragraph
            self.batch_size = batch_size
            print(f"✓ EasyOCR initialized successfully (GPU: {gpu})")
            
        except Exception as e:
            print(f"✗ Failed to initialize EasyOCR: {e}")
            raise
        
        # Statistics tracking
        self.stats = {
            'total_reads': 0,
            'successful_reads': 0,
            'failed_reads': 0,
            'low_confidence_reads': 0
        }
    
    def read(self, plate_crop: np.ndarray) -> Optional[OCRReading]:
        """
        Read text from a license plate crop
        
        Args:
            plate_crop: BGR image of license plate
            
        Returns:
            OCRReading object if text found, None otherwise
        """
        if plate_crop is None or plate_crop.size == 0:
            return None
        
        self.stats['total_reads'] += 1
        
        # Preprocess image if enabled
        processed_img = self._preprocess(plate_crop) if self.preprocessing else plate_crop
        
        # Run EasyOCR
        try:
            results = self.reader.readtext(
                processed_img,
                paragraph=self.paragraph,
                batch_size=self.batch_size
            )
        except Exception as e:
            print(f"OCR Error: {e}")
            self.stats['failed_reads'] += 1
            return None
        
        # Parse results
        ocr_reading = self._parse_results(results, plate_crop.shape)
        
        if ocr_reading is None:
            self.stats['failed_reads'] += 1
        elif ocr_reading.confidence < self.confidence_threshold:
            self.stats['low_confidence_reads'] += 1
        else:
            self.stats['successful_reads'] += 1
        
        return ocr_reading
    
    def read_batch(self, plate_crops: List[np.ndarray]) -> List[Optional[OCRReading]]:
        """
        Read text from multiple plate crops
        
        Args:
            plate_crops: List of license plate crop images
            
        Returns:
            List of OCRReading objects (None for failed reads)
        """
        return [self.read(crop) for crop in plate_crops]
    
    def _preprocess(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR accuracy
        
        Args:
            plate_crop: Original plate crop
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop.copy()
        
        # Resize if too small (helps OCR)
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50), interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _parse_results(
        self,
        results: List,
        image_shape: Tuple[int, int]
    ) -> Optional[OCRReading]:
        """
        Parse EasyOCR results and extract best reading
        
        Args:
            results: EasyOCR results [(bbox, text, confidence), ...]
            image_shape: Shape of input image (H, W) or (H, W, C)
            
        Returns:
            OCRReading object or None
        """
        if not results:
            return None
        
        # EasyOCR returns: [(bbox, text, confidence), ...]
        # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        
        # Find result with highest confidence
        best_result = max(results, key=lambda x: x[2])
        bbox_points, raw_text, confidence = best_result
        
        # Convert bbox format
        x_coords = [p[0] for p in bbox_points]
        y_coords = [p[1] for p in bbox_points]
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        bbox = (x1, y1, x2, y2)
        
        # Clean text
        cleaned_text = self._clean_text(raw_text)
        
        # Validate text
        if not self._is_valid_plate_text(cleaned_text):
            return None
        
        return OCRReading(
            text=cleaned_text,
            confidence=float(confidence),
            bbox=bbox,
            raw_text=raw_text
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean OCR text output
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Convert to uppercase
        text = text.upper()
        
        # Remove spaces
        text = text.replace(" ", "")
        
        # Common OCR mistakes - character substitutions
        substitutions = {
            'O': '0',  # Letter O -> Zero
            'I': '1',  # Letter I -> One
            'Z': '2',  # Sometimes Z -> 2
            'S': '5',  # Sometimes S -> 5
            'B': '8',  # Sometimes B -> 8
            'G': '6',  # Sometimes G -> 6
            'Q': '0',  # Letter Q -> Zero
        }
        
        # Apply substitutions intelligently
        # Keep letters at the beginning, numbers at the end (common pattern)
        cleaned = ""
        for i, char in enumerate(text):
            if char in self.allowed_chars:
                cleaned += char
            elif char in substitutions:
                # Apply substitution if it makes sense
                cleaned += substitutions[char]
        
        # Remove any remaining invalid characters
        cleaned = re.sub(f'[^{re.escape(self.allowed_chars)}]', '', cleaned)
        
        return cleaned
    
    def _is_valid_plate_text(self, text: str) -> bool:
        """
        Validate if text looks like a license plate
        
        Args:
            text: Cleaned text
            
        Returns:
            True if valid, False otherwise
        """
        # Check length
        if len(text) < self.min_text_length or len(text) > self.max_text_length:
            return False
        
        # Check if contains at least one letter and one number
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        if not (has_letter and has_number):
            return False
        
        # Check if all characters are allowed
        if not all(c in self.allowed_chars for c in text):
            return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Get OCR statistics"""
        stats = self.stats.copy()
        if stats['total_reads'] > 0:
            stats['success_rate'] = stats['successful_reads'] / stats['total_reads']
        else:
            stats['success_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'total_reads': 0,
            'successful_reads': 0,
            'failed_reads': 0,
            'low_confidence_reads': 0
        }
    
    def visualize_reading(
        self,
        plate_crop: np.ndarray,
        ocr_reading: OCRReading,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw OCR results on plate image for visualization
        
        Args:
            plate_crop: Original plate crop
            ocr_reading: OCRReading object
            color: BGR color for text/bbox
            thickness: Line thickness
            
        Returns:
            Image with drawn results
        """
        vis_img = plate_crop.copy()
        
        # Draw bbox
        x1, y1, x2, y2 = ocr_reading.bbox
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw text label
        label = f"{ocr_reading.text} ({ocr_reading.confidence:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw background for text
        cv2.rectangle(
            vis_img,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            vis_img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        return vis_img


