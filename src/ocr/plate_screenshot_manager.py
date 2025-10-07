"""
License Plate Screenshot Manager

Handles intelligent saving of license plate screenshots with:
- Quality-based filtering
- Storage optimization
- Track-specific organization
- Perfect quality limiting (max 3 screenshots for quality >= 0.95)
"""

import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class PlateScreenshotManager:
    """
    Manages license plate screenshot saving with intelligent quality filtering
    and storage optimization.
    """
    
    def __init__(
        self,
        output_dir: str = "data/output/plate_screenshots",
        perfect_quality_threshold: float = 0.95,
        max_perfect_quality_shots: int = 3,
        min_quality_threshold: float = 0.6,
        verbose: bool = True
    ):
        """
        Initialize the screenshot manager.
        
        Args:
            output_dir: Root directory for saving screenshots
            perfect_quality_threshold: Quality score considered "perfect" (0.95 = 95%)
            max_perfect_quality_shots: Maximum screenshots to keep for perfect quality
            min_quality_threshold: Minimum quality to save any screenshot
            verbose: Print detailed logging
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.perfect_quality_threshold = perfect_quality_threshold
        self.max_perfect_quality_shots = max_perfect_quality_shots
        self.min_quality_threshold = min_quality_threshold
        self.verbose = verbose
        
        # Track data: {track_id: {screenshots, class, best_quality, quality_threshold}}
        self.track_data: Dict[int, Dict] = {}
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'saved': 0,
            'skipped_low_quality': 0,
            'skipped_below_threshold': 0,
            'skipped_perfect_limit': 0,
            'replaced': 0
        }
    
    def save_if_better(
        self,
        track_id: int,
        class_name: str,
        plate_crop: np.ndarray,
        quality_score: float,
        frame_id: int
    ) -> Tuple[bool, str]:
        """
        Save plate screenshot if it meets quality criteria.
        
        Args:
            track_id: Vehicle track ID
            class_name: Vehicle class (car, truck, etc.)
            plate_crop: Cropped plate image
            quality_score: Quality score from quality assessor (0.0 to 1.0)
            frame_id: Current frame number
            
        Returns:
            Tuple of (was_saved: bool, reason: str)
        """
        self.stats['total_attempts'] += 1
        
        # Initialize track if new
        if track_id not in self.track_data:
            self._initialize_track(track_id, class_name)
        
        track_info = self.track_data[track_id]
        
        # Check if we should save this screenshot
        should_save, reason = self._should_save_screenshot(
            track_info, quality_score
        )
        
        if should_save:
            # Save the screenshot
            filepath = self._save_screenshot(
                track_id, class_name, plate_crop, quality_score, frame_id
            )
            
            # Update track data
            screenshot_info = {
                'filepath': filepath,
                'quality': quality_score,
                'frame_id': frame_id
            }
            track_info['screenshots'].append(screenshot_info)
            
            # Update best quality
            if quality_score > track_info['best_quality']:
                track_info['best_quality'] = quality_score
                track_info['best_screenshot'] = filepath
            
            # Update quality threshold (minimum of all saved)
            qualities = [s['quality'] for s in track_info['screenshots']]
            track_info['quality_threshold'] = min(qualities)
            
            # Handle perfect quality limit
            if quality_score >= self.perfect_quality_threshold:
                self._enforce_perfect_quality_limit(track_info)
            
            self.stats['saved'] += 1
            
            if self.verbose:
                print(f"  ✓ Track {track_id} ({class_name}): Saved plate screenshot "
                      f"(Q:{quality_score:.2f}, Frame:{frame_id}) - {reason}")
            
            return True, reason
        else:
            # Update statistics based on reason
            if "below minimum" in reason:
                self.stats['skipped_low_quality'] += 1
            elif "below threshold" in reason:
                self.stats['skipped_below_threshold'] += 1
            elif "limit reached" in reason:
                self.stats['skipped_perfect_limit'] += 1
            
            if self.verbose and self.stats['total_attempts'] % 30 == 0:
                print(f"  ✗ Track {track_id}: Skipped (Q:{quality_score:.2f}) - {reason}")
            
            return False, reason
    
    def _should_save_screenshot(
        self, 
        track_info: Dict, 
        quality_score: float
    ) -> Tuple[bool, str]:
        """
        Determine if screenshot should be saved based on quality criteria.
        
        Returns:
            Tuple of (should_save: bool, reason: str)
        """
        # Check 1: Minimum quality threshold
        if quality_score < self.min_quality_threshold:
            return False, f"below minimum quality ({self.min_quality_threshold:.2f})"
        
        # Check 2: Perfect quality handling (>= 0.95)
        if quality_score >= self.perfect_quality_threshold:
            num_screenshots = len(track_info['screenshots'])
            
            # Allow up to max_perfect_quality_shots
            if num_screenshots < self.max_perfect_quality_shots:
                return True, f"perfect quality slot {num_screenshots+1}/{self.max_perfect_quality_shots}"
            
            # If we have max shots, only save if better than the worst
            elif quality_score > track_info['quality_threshold']:
                return True, "perfect quality, replacing worst"
            else:
                return False, f"perfect quality limit reached ({self.max_perfect_quality_shots} max)"
        
        # Check 3: Regular quality - must be better than current threshold
        if quality_score > track_info['quality_threshold']:
            return True, f"better than threshold ({track_info['quality_threshold']:.2f})"
        
        return False, f"below threshold ({track_info['quality_threshold']:.2f})"
    
    def _save_screenshot(
        self,
        track_id: int,
        class_name: str,
        plate_crop: np.ndarray,
        quality_score: float,
        frame_id: int
    ) -> str:
        """
        Save plate screenshot with track info overlay.
        
        Returns:
            Filepath of saved screenshot
        """
        # Create track-specific directory
        track_dir = self.output_dir / f"{class_name}_{track_id}"
        track_dir.mkdir(exist_ok=True)
        
        # Add track ID and type overlay on the plate image
        plate_with_info = plate_crop.copy()
        
        # Create info text
        info_text = f"ID:{track_id} | {class_name.upper()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            info_text, font, font_scale, thickness
        )
        
        # Create a bar at the top for the text
        bar_height = text_h + baseline + 10
        plate_with_bar = cv2.copyMakeBorder(
            plate_with_info,
            bar_height, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
        
        # Add text on the black bar
        cv2.putText(
            plate_with_bar,
            info_text,
            (5, text_h + 5),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )
        
        # Generate filename with quality score
        filename = f"frame_{frame_id:06d}_q{quality_score:.2f}.jpg"
        filepath = track_dir / filename
        
        # Save screenshot
        cv2.imwrite(str(filepath), plate_with_bar)
        
        return str(filepath)
    
    def _enforce_perfect_quality_limit(self, track_info: Dict):
        """
        Enforce maximum limit for perfect quality screenshots.
        Removes lowest quality screenshot if limit exceeded.
        """
        screenshots = track_info['screenshots']
        
        # Count perfect quality screenshots
        perfect_shots = [
            s for s in screenshots 
            if s['quality'] >= self.perfect_quality_threshold
        ]
        
        # If we exceed the limit, remove the worst one
        if len(perfect_shots) > self.max_perfect_quality_shots:
            worst_shot = min(perfect_shots, key=lambda x: x['quality'])
            
            # Delete the file
            try:
                Path(worst_shot['filepath']).unlink()
                screenshots.remove(worst_shot)
                self.stats['replaced'] += 1
                
                if self.verbose:
                    print(f"    Removed lower quality screenshot (Q:{worst_shot['quality']:.2f})")
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: Could not delete old screenshot: {e}")
    
    def _initialize_track(self, track_id: int, class_name: str):
        """Initialize data structure for a new track."""
        self.track_data[track_id] = {
            'screenshots': [],
            'class': class_name,
            'best_quality': 0.0,
            'best_screenshot': None,
            'quality_threshold': 0.0  # Start with no threshold
        }
    
    def get_track_screenshots(self, track_id: int) -> Optional[List[Dict]]:
        """
        Get all screenshots for a specific track.
        
        Returns:
            List of screenshot info dicts, or None if track doesn't exist
        """
        if track_id in self.track_data:
            return self.track_data[track_id]['screenshots']
        return None
    
    def get_best_screenshot(self, track_id: int) -> Optional[str]:
        """
        Get filepath of best quality screenshot for a track.
        
        Returns:
            Filepath string, or None if no screenshots exist
        """
        if track_id in self.track_data:
            return self.track_data[track_id]['best_screenshot']
        return None
    
    def get_statistics(self) -> Dict:
        """Get saving statistics."""
        stats = self.stats.copy()
        
        # Add derived statistics
        if stats['total_attempts'] > 0:
            stats['save_rate'] = stats['saved'] / stats['total_attempts']
        else:
            stats['save_rate'] = 0.0
        
        stats['total_tracks'] = len(self.track_data)
        stats['tracks_with_screenshots'] = sum(
            1 for t in self.track_data.values() if t['screenshots']
        )
        
        return stats
    
    def print_summary(self):
        """Print a summary of all saved screenshots."""
        print("\n" + "="*70)
        print("PLATE SCREENSHOT SUMMARY")
        print("="*70)
        
        stats = self.get_statistics()
        
        print(f"\nStatistics:")
        print(f"  Total save attempts: {stats['total_attempts']}")
        print(f"  Screenshots saved: {stats['saved']}")
        print(f"  Save rate: {stats['save_rate']*100:.1f}%")
        print(f"  Skipped (low quality): {stats['skipped_low_quality']}")
        print(f"  Skipped (below threshold): {stats['skipped_below_threshold']}")
        print(f"  Skipped (perfect limit): {stats['skipped_perfect_limit']}")
        print(f"  Replaced (better found): {stats['replaced']}")
        
        print(f"\nTracks:")
        print(f"  Total tracks: {stats['total_tracks']}")
        print(f"  Tracks with screenshots: {stats['tracks_with_screenshots']}")
        
        print(f"\nPer-Track Details:")
        print("-" * 70)
        
        for track_id, data in sorted(self.track_data.items()):
            num_screenshots = len(data['screenshots'])
            if num_screenshots > 0:
                best_quality = data['best_quality']
                print(f"  Track {track_id} ({data['class']}): "
                      f"{num_screenshots} screenshots (Best Q:{best_quality:.2f})")
                print(f"    Best: {data['best_screenshot']}")
        
        print("="*70 + "\n")
    
    def cleanup_empty_directories(self):
        """Remove empty track directories."""
        for track_dir in self.output_dir.iterdir():
            if track_dir.is_dir() and not any(track_dir.iterdir()):
                track_dir.rmdir()
                if self.verbose:
                    print(f"Removed empty directory: {track_dir}")


