"""
Vehicle Speed Estimator

Estimates vehicle speeds using pixel-based tracking with real-world calibration.
Features:
- Per-track speed calculation (instantaneous, average, min, max)
- Speed violation detection with configurable limits per vehicle class
- Outlier rejection for robust estimation
- JSON export for speed data and violations
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict, deque


class SpeedEstimator:
    """
    Estimates vehicle speeds from tracked bounding boxes with real-world calibration.
    """
    
    def __init__(
        self,
        frame_width_meters: float,
        frame_height_meters: float,
        fps: int,
        speed_limits: Dict[str, float],
        min_frames_for_speed: int = 10,
        smoothing_window: int = 5,
        ignore_edge_frames: int = 3,
        min_distance_threshold: float = 0.5,
        speed_unit: str = 'kmh',
        output_dir: str = 'data/output/speed_data',
        outlier_rejection: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the speed estimator.
        
        Args:
            frame_width_meters: Real-world width that camera sees (meters)
            frame_height_meters: Real-world height that camera sees (meters)
            fps: Video frames per second
            speed_limits: Speed limits per vehicle class, e.g., {'car': 120, 'truck': 90}
            min_frames_for_speed: Minimum frames tracked before calculating speed
            smoothing_window: Number of frames for moving average speed calculation
            ignore_edge_frames: Ignore first/last N frames (unstable detections)
            min_distance_threshold: Minimum distance (meters) to calculate speed
            speed_unit: Output unit ('kmh' or 'mph')
            output_dir: Directory for saving speed data and violations
            outlier_rejection: Enable outlier rejection (remove extreme speeds)
            verbose: Print detailed logging
        """
        self.frame_width_meters = frame_width_meters
        self.frame_height_meters = frame_height_meters
        self.fps = fps
        self.speed_limits = speed_limits
        self.min_frames_for_speed = min_frames_for_speed
        self.smoothing_window = smoothing_window
        self.ignore_edge_frames = ignore_edge_frames
        self.min_distance_threshold = min_distance_threshold
        self.speed_unit = speed_unit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.outlier_rejection = outlier_rejection
        self.verbose = verbose
        
        # Pixel to meter conversion (will be set after first frame)
        self.pixels_per_meter_x = None
        self.pixels_per_meter_y = None
        self.frame_width_pixels = None
        self.frame_height_pixels = None
        
        # Track data: {track_id: TrackData}
        self.tracks: Dict[int, Dict] = {}
        
        # Violations log
        self.violations: List[Dict] = []
        
        # Statistics
        self.stats = {
            'total_tracks': 0,
            'tracks_with_speed': 0,
            'total_violations': 0,
            'violations_by_class': defaultdict(int)
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("SPEED ESTIMATOR INITIALIZED")
            print(f"{'='*70}")
            print(f"Calibration: {frame_width_meters}m × {frame_height_meters}m")
            print(f"FPS: {fps}")
            print(f"Speed unit: {speed_unit.upper()}")
            print(f"Speed limits: {speed_limits}")
            print(f"{'='*70}\n")
    
    def set_frame_dimensions(self, frame_width_pixels: int, frame_height_pixels: int):
        """
        Set frame dimensions and calculate pixel-to-meter conversion ratios.
        Should be called once with the first frame dimensions.
        """
        self.frame_width_pixels = frame_width_pixels
        self.frame_height_pixels = frame_height_pixels
        
        # Calculate pixels per meter
        self.pixels_per_meter_x = frame_width_pixels / self.frame_width_meters
        self.pixels_per_meter_y = frame_height_pixels / self.frame_height_meters
        
        if self.verbose:
            print(f"Frame calibrated: {frame_width_pixels}×{frame_height_pixels} pixels")
            print(f"Conversion: {self.pixels_per_meter_x:.2f} px/m (X), "
                  f"{self.pixels_per_meter_y:.2f} px/m (Y)\n")
    
    def update(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        class_name: str,
        frame_id: int,
        confidence: float = 1.0
    ):
        """
        Update track position and calculate speed.
        
        Args:
            track_id: Unique vehicle track ID
            bbox: Bounding box (x1, y1, x2, y2)
            class_name: Vehicle class (car, truck, etc.)
            frame_id: Current frame number
            confidence: Detection confidence (optional)
        """
        # Initialize frame dimensions if not set
        if self.pixels_per_meter_x is None:
            # Can't calculate without frame dimensions
            return
        
        # Initialize track if new
        if track_id not in self.tracks:
            self._initialize_track(track_id, class_name)
            self.stats['total_tracks'] += 1
        
        track = self.tracks[track_id]
        
        # Calculate center point of bbox
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Store position data
        position = {
            'frame_id': frame_id,
            'center_x': center_x,
            'center_y': center_y,
            'confidence': confidence
        }
        track['positions'].append(position)
        track['last_seen_frame'] = frame_id
        
        # Calculate speed if enough frames
        if len(track['positions']) >= self.min_frames_for_speed:
            speed = self._calculate_speed(track_id)
            
            if speed is not None:
                track['speeds'].append(speed)
                track['current_speed'] = speed
                
                # Update min/max
                if track['max_speed'] is None or speed > track['max_speed']:
                    track['max_speed'] = speed
                if track['min_speed'] is None or speed < track['min_speed']:
                    track['min_speed'] = speed
                
                # Check for violation
                self._check_violation(track_id, speed, class_name, frame_id)
    
    def _calculate_speed(self, track_id: int) -> Optional[float]:
        """
        Calculate current speed for a track using recent positions.
        
        Returns:
            Speed in configured unit (km/h or mph), or None if can't calculate
        """
        track = self.tracks[track_id]
        positions = track['positions']
        
        # Need at least 2 positions
        if len(positions) < 2:
            return None
        
        # Use smoothing window
        window_size = min(self.smoothing_window, len(positions))
        recent_positions = positions[-window_size:]
        
        # Ignore edge frames if specified
        if len(positions) <= self.ignore_edge_frames * 2:
            return None
        
        # Calculate displacement over window
        start_pos = recent_positions[0]
        end_pos = recent_positions[-1]
        
        # Pixel displacement
        dx_pixels = end_pos['center_x'] - start_pos['center_x']
        dy_pixels = end_pos['center_y'] - start_pos['center_y']
        
        # Convert to meters
        dx_meters = dx_pixels / self.pixels_per_meter_x
        dy_meters = dy_pixels / self.pixels_per_meter_y
        
        # Total distance (Euclidean)
        distance_meters = np.sqrt(dx_meters**2 + dy_meters**2)
        
        # Check minimum distance threshold
        if distance_meters < self.min_distance_threshold:
            return None
        
        # Time elapsed
        frame_diff = end_pos['frame_id'] - start_pos['frame_id']
        if frame_diff == 0:
            return None
        
        time_seconds = frame_diff / self.fps
        
        # Speed in m/s
        speed_ms = distance_meters / time_seconds
        
        # Convert to desired unit
        if self.speed_unit == 'kmh':
            speed = speed_ms * 3.6  # m/s to km/h
        elif self.speed_unit == 'mph':
            speed = speed_ms * 2.237  # m/s to mph
        else:
            speed = speed_ms
        
        # Outlier rejection
        if self.outlier_rejection and len(track['speeds']) > 0:
            # Reject if speed is more than 3x the median
            median_speed = np.median(track['speeds'])
            if speed > median_speed * 3 or speed > 300:  # Also reject unrealistic speeds
                return None
        
        return speed
    
    def _check_violation(self, track_id: int, speed: float, class_name: str, frame_id: int):
        """Check if current speed exceeds limit and log violation."""
        if class_name not in self.speed_limits:
            return
        
        speed_limit = self.speed_limits[class_name]
        
        if speed > speed_limit:
            track = self.tracks[track_id]
            
            # Only log once per track (first violation)
            if not track['violation_logged']:
                violation = {
                    'track_id': track_id,
                    'vehicle_class': class_name,
                    'speed': round(speed, 2),
                    'speed_limit': speed_limit,
                    'overspeed': round(speed - speed_limit, 2),
                    'frame_id': frame_id,
                    'unit': self.speed_unit
                }
                
                self.violations.append(violation)
                track['violation_logged'] = True
                self.stats['total_violations'] += 1
                self.stats['violations_by_class'][class_name] += 1
                
                if self.verbose:
                    print(f"⚠️  VIOLATION - Track {track_id} ({class_name}): "
                          f"{speed:.1f} {self.speed_unit} "
                          f"(limit: {speed_limit} {self.speed_unit}, "
                          f"+{speed - speed_limit:.1f}) at frame {frame_id}")
    
    def finalize_track(self, track_id: int) -> Optional[Dict]:
        """
        Finalize track when vehicle leaves frame.
        Calculate final statistics.
        
        Returns:
            Dictionary with speed statistics, or None if track doesn't exist
        """
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        
        # Calculate average speed
        if track['speeds']:
            track['avg_speed'] = np.mean(track['speeds'])
            track['finalized'] = True
            self.stats['tracks_with_speed'] += 1
            
            if self.verbose:
                print(f"✓ Track {track_id} ({track['class']}) finalized: "
                      f"Avg={track['avg_speed']:.1f} {self.speed_unit}, "
                      f"Max={track['max_speed']:.1f} {self.speed_unit}, "
                      f"Min={track['min_speed']:.1f} {self.speed_unit}")
        
        return self._get_track_summary(track_id)
    
    def get_current_speed(self, track_id: int) -> Optional[float]:
        """Get current/latest speed for a track."""
        if track_id in self.tracks:
            return self.tracks[track_id].get('current_speed')
        return None
    
    def get_average_speed(self, track_id: int) -> Optional[float]:
        """Get average speed for a track."""
        if track_id in self.tracks and self.tracks[track_id]['speeds']:
            return np.mean(self.tracks[track_id]['speeds'])
        return None
    
    def _get_track_summary(self, track_id: int) -> Optional[Dict]:
        """Get summary statistics for a track."""
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        
        return {
            'track_id': track_id,
            'vehicle_class': track['class'],
            'avg_speed': round(track['avg_speed'], 2) if track['avg_speed'] else None,
            'max_speed': round(track['max_speed'], 2) if track['max_speed'] else None,
            'min_speed': round(track['min_speed'], 2) if track['min_speed'] else None,
            'speed_unit': self.speed_unit,
            'num_measurements': len(track['speeds']),
            'violation': track['violation_logged'],
            'finalized': track['finalized']
        }
    
    def get_all_track_summaries(self) -> List[Dict]:
        """Get summaries for all tracks."""
        summaries = []
        for track_id in self.tracks.keys():
            summary = self._get_track_summary(track_id)
            if summary:
                summaries.append(summary)
        return summaries
    
    def get_violations(self) -> List[Dict]:
        """Get all speed violations."""
        return self.violations
    
    def export_json(self, filename: str = "speed_data.json"):
        """
        Export all speed data and violations to JSON.
        
        Args:
            filename: Output filename
        """
        filepath = self.output_dir / filename
        
        data = {
            'calibration': {
                'frame_width_meters': self.frame_width_meters,
                'frame_height_meters': self.frame_height_meters,
                'fps': self.fps,
                'speed_unit': self.speed_unit
            },
            'speed_limits': self.speed_limits,
            'statistics': self.stats,
            'tracks': self.get_all_track_summaries(),
            'violations': self.violations
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"\n✓ Speed data exported to: {filepath}")
        
        return str(filepath)
    
    def print_summary(self):
        """Print a comprehensive summary of speed estimations."""
        print("\n" + "="*70)
        print("SPEED ESTIMATION SUMMARY")
        print("="*70)
        
        print(f"\nStatistics:")
        print(f"  Total tracks: {self.stats['total_tracks']}")
        print(f"  Tracks with speed data: {self.stats['tracks_with_speed']}")
        print(f"  Total violations: {self.stats['total_violations']}")
        
        if self.stats['violations_by_class']:
            print(f"\n  Violations by class:")
            for class_name, count in self.stats['violations_by_class'].items():
                print(f"    {class_name}: {count}")
        
        print(f"\nSpeed Data by Track:")
        print("-" * 70)
        
        summaries = self.get_all_track_summaries()
        for summary in sorted(summaries, key=lambda x: x['track_id']):
            if summary['avg_speed']:
                violation_flag = "⚠️ " if summary['violation'] else "✓ "
                print(f"  {violation_flag}Track {summary['track_id']} ({summary['vehicle_class']}): "
                      f"Avg={summary['avg_speed']:.1f} {self.speed_unit}, "
                      f"Max={summary['max_speed']:.1f}, "
                      f"Min={summary['min_speed']:.1f} "
                      f"({summary['num_measurements']} measurements)")
        
        if self.violations:
            print(f"\nViolations:")
            print("-" * 70)
            for v in self.violations:
                print(f"  Track {v['track_id']} ({v['vehicle_class']}): "
                      f"{v['speed']} {v['unit']} in {v['speed_limit']} {v['unit']} zone "
                      f"(+{v['overspeed']} {v['unit']}) at frame {v['frame_id']}")
        
        print("="*70 + "\n")
    
    def _initialize_track(self, track_id: int, class_name: str):
        """Initialize data structure for a new track."""
        self.tracks[track_id] = {
            'class': class_name,
            'positions': [],
            'speeds': [],
            'current_speed': None,
            'avg_speed': None,
            'max_speed': None,
            'min_speed': None,
            'last_seen_frame': 0,
            'violation_logged': False,
            'finalized': False
        }


