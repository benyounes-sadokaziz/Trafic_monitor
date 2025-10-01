"""
ByteTrack Tracker Module
"""

import numpy as np
from typing import List, Dict
import logging


try:
    from supervision import ByteTrack, Detections
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    

class VehicleTracker:
    """Tracks vehicles across frames using ByteTrack algorithm."""
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8
    ):
        """
        Initialize tracker.
        
        Args:
            track_thresh: Detection confidence threshold
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IOU threshold for matching
        """
        self.logger = logging.getLogger(__name__)
        
        if not SUPERVISION_AVAILABLE:
            raise ImportError(
                "supervision library not installed. "
                "Install with: pip install supervision"
            )
        
        self.tracker = ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=30,
            minimum_consecutive_frames=1
        )
        
        self.track_history = {}  # Store track trajectories
        
        self.logger.info("ByteTrack tracker initialized")
        
    def update(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List from VehicleDetector
            frame_id: Current frame number
            
        Returns:
            List of tracked objects with IDs:
            {
                'track_id': int,
                'bbox': (x1, y1, x2, y2),
                'confidence': float,
                'class': str,
                'class_id': int
            }
        """
        if not detections:
            # Update tracker with empty detections to age out old tracks
            return []
        
        # Convert to supervision.Detections format
        supervision_detections = self._convert_to_supervision_detections(detections)
        
        # Update tracker
        tracked_detections = self.tracker.update_with_detections(supervision_detections)
        
        # Convert back to our format
        tracks = self._convert_from_supervision_detections(tracked_detections, frame_id)
        
        return tracks
        
    def _convert_to_supervision_detections(self, detections: List[Dict]) -> Detections:
        """
        Convert detection list to supervision.Detections format.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            supervision.Detections object
        """
        if not detections:
            return Detections.empty()
            
        xyxy = []
        confidence = []
        class_id = []
        
        for det in detections:
            # Extract bbox
            if 'bbox' in det:
                # Assume bbox is [x, y, w, h]
                x, y, w, h = det['bbox']
                x1, y1, x2, y2 = x, y, x + w, y + h
            elif 'xyxy' in det:
                x1, y1, x2, y2 = det['xyxy']
            else:
                continue
                
            xyxy.append([x1, y1, x2, y2])
            confidence.append(det.get('confidence', 1.0))
            class_id.append(det.get('class_id', 0))
        
        if not xyxy:
            return Detections.empty()
            
        return Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id)
        )
    
    def _convert_from_supervision_detections(self, tracked_detections: Detections, 
                                           frame_id: int) -> List[Dict]:
        """
        Convert supervision.Detections back to track dictionaries.
        
        Args:
            tracked_detections: Detections with tracker_id from ByteTrack
            frame_id: Current frame ID
            
        Returns:
            List of track dictionaries
        """
        tracks = []
        
        if len(tracked_detections) == 0:
            return tracks
            
        for i in range(len(tracked_detections)):
            x1, y1, x2, y2 = tracked_detections.xyxy[i]
            track_id = tracked_detections.tracker_id[i] if tracked_detections.tracker_id is not None else i
            confidence = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 1.0
            class_id = tracked_detections.class_id[i] if tracked_detections.class_id is not None else 0
            
            # Store track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            centroid = [(x1 + x2) / 2, (y1 + y2) / 2]
            self.track_history[track_id].append({
                'frame_id': frame_id,
                'bbox': [x1, y1, x2, y2],
                'center': centroid,
                'confidence': confidence
            })
            
            # Keep only last 30 frames
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id] = self.track_history[track_id][-30:]
            
            # Map class_id to class name
            class_map = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            class_name = class_map.get(int(class_id), 'vehicle')
            
            # Create track dictionary
            track = {
                'track_id': int(track_id),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(confidence),
                'class': class_name,
                'class_id': int(class_id),
                'center': (int(centroid[0]), int(centroid[1]))
            }
            
            tracks.append(track)
            
        return tracks
        
    def get_track_history(self, track_id: int) -> List[Dict]:
        """Get movement history for a track."""
        return self.track_history.get(track_id, [])
        
    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()
        self.track_history.clear()
        self.logger.info("Tracker reset")