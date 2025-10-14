"""
Upgraded ByteTrack Tracker Module
- More robust handling when detections are empty (always updates tracker)
- Configurable parameters via constructor
- Bounded deque for track history
- Stores frame_id and timestamp in history entries
- Keeps OCR-related metadata per track (ocr_attempts, best_plate, ocr_locked)
- Metrics counters for monitoring
- Callbacks hooks for track lifecycle events
- Defensive handling of supervision.Detections shapes
- Clear documented input/output formats
"""

from collections import deque
from typing import List, Dict, Optional, Callable
import logging
import time
import numpy as np

try:
    from supervision import ByteTrack, Detections
    SUPERVISION_AVAILABLE = True
except Exception:
    SUPERVISION_AVAILABLE = False


class VehicleTracker:
    """Tracks vehicles across frames using ByteTrack algorithm.

    Expected detection input format (list of dicts):
        {
            'bbox': (x, y, w, h)   # preferred
            # or 'xyxy': (x1, y1, x2, y2)
            'confidence': float,
            'class_id': int,
            'class': str (optional)
        }

    Output per tracked object (list of dicts):
        {
            'track_id': int,
            'bbox': (x1, y1, x2, y2),
            'confidence': float,
            'class': str,
            'class_id': int,
            'center': (cx, cy),
            'frame_id': int,
            'timestamp': float
        }
    """

    DEFAULT_CLASS_MAP = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    def __init__(
        self,
        track_thresh: float = 0.4,
        track_buffer: int = 90,
        match_thresh: float = 0.7,
        frame_rate: int = 30,
        max_history: int = 90,
        class_map: Optional[Dict[int, str]] = None,
        on_track_started: Optional[Callable[[int, Dict], None]] = None,
        on_track_updated: Optional[Callable[[int, Dict], None]] = None,
        on_track_lost: Optional[Callable[[int, Dict], None]] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not SUPERVISION_AVAILABLE:
            raise ImportError(
                "supervision library not installed. Install with: pip install supervision"
            )

        # Configurable parameters
        self.track_thresh = float(track_thresh)
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.frame_rate = int(frame_rate)
        self.max_history = int(max_history)
        self.class_map = class_map if class_map is not None else self.DEFAULT_CLASS_MAP

        # Instantiate ByteTrack
        self.tracker = ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=self.frame_rate,
            minimum_consecutive_frames=1,
        )

        # Track history and metadata
        # history: track_id -> deque of last N states
        self.track_history: Dict[int, deque] = {}
        # ocr metadata per track
        self.track_meta: Dict[int, Dict] = {}

        # Callbacks
        self.on_track_started = on_track_started
        self.on_track_updated = on_track_updated
        self.on_track_lost = on_track_lost

        # Metrics
        self.metrics = {
            'frames_processed': 0,
            'active_tracks': 0,
            'created_tracks': 0,
            'lost_tracks': 0,
            'last_update_time_ms': 0.0,
        }

        self.logger.info("ByteTrack tracker initialized (upgraded)")

    # ----- Public API -----
    def update(self, detections: List[Dict], frame_id: int, timestamp: Optional[float] = None) -> List[Dict]:
        """Update tracker with new detections and return tracked objects.

        Even when detections is empty, this will call the tracker with an empty Detections
        object so internal tracker state (aging/lost tracks) is updated.
        """
        if timestamp is None:
            timestamp = time.time()

        start = time.time()
        self.metrics['frames_processed'] += 1

        # Convert incoming detections to supervision.Detections
        sup_dets = self._convert_to_supervision_detections(detections)

        # Update tracker with (possibly empty) detections
        try:
            tracked = self.tracker.update_with_detections(sup_dets)
        except Exception as e:
            # Defensive fallback: log and return empty list
            self.logger.exception("ByteTrack update failed: %s", e)
            self.metrics['last_update_time_ms'] = (time.time() - start) * 1000.0
            return []

        # Convert tracked detections back to our format
        tracks = self._convert_from_supervision_detections(tracked, frame_id, timestamp)

        # update metrics
        self.metrics['last_update_time_ms'] = (time.time() - start) * 1000.0
        self.metrics['active_tracks'] = len(tracks)

        return tracks

    def reset(self):
        """Reset tracker state and metrics."""
        try:
            self.tracker.reset()
        except Exception:
            pass
        self.track_history.clear()
        self.track_meta.clear()
        for k in self.metrics:
            if isinstance(self.metrics[k], int):
                self.metrics[k] = 0
        self.logger.info("Tracker reset")

    def get_track_history(self, track_id: int) -> List[Dict]:
        return list(self.track_history.get(track_id, deque()))

    # ----- Internal helpers -----
    def _convert_to_supervision_detections(self, detections: List[Dict]) -> Detections:
        """Converts a list of detection dicts to supervision.Detections.

        Keeps defensive handling for missing fields and supports both (x,y,w,h) and xyxy formats.
        """
        if not detections:
            return Detections.empty()

        xyxy_list = []
        conf_list = []
        class_id_list = []

        for det in detections:
            try:
                if 'bbox' in det:
                    x, y, w, h = det['bbox']
                    x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
                elif 'xyxy' in det:
                    x1, y1, x2, y2 = map(float, det['xyxy'])
                else:
                    # skip malformed detection
                    continue

                conf = float(det.get('confidence', 1.0))
                cid = int(det.get('class_id', 0))

                xyxy_list.append([x1, y1, x2, y2])
                conf_list.append(conf)
                class_id_list.append(cid)
            except Exception:
                # skip individual malformed detection but continue
                continue

        if len(xyxy_list) == 0:
            return Detections.empty()

        return Detections(xyxy=np.array(xyxy_list), confidence=np.array(conf_list), class_id=np.array(class_id_list))

    def _convert_from_supervision_detections(self, tracked: Detections, frame_id: int, timestamp: float) -> List[Dict]:
        """Convert supervision tracked detections into our standardized track dicts.

        Also maintains internal track history and meta.
        """
        out = []

        # Defensive checks
        try:
            length = len(tracked)
        except Exception:
            # If tracked isn't iterable or has no length, return empty
            return out

        for i in range(length):
            try:
                # retrieve fields defensively
                xyxy = tracked.xyxy[i] if hasattr(tracked, 'xyxy') and tracked.xyxy is not None else None
                tid = tracked.tracker_id[i] if hasattr(tracked, 'tracker_id') and tracked.tracker_id is not None else None
                conf = tracked.confidence[i] if hasattr(tracked, 'confidence') and tracked.confidence is not None else 1.0
                cid = tracked.class_id[i] if hasattr(tracked, 'class_id') and tracked.class_id is not None else 0

                if xyxy is None:
                    continue

                x1, y1, x2, y2 = map(float, xyxy)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                track_id = int(tid) if tid is not None else i

                # create history entry
                hist_entry = {
                    'frame_id': int(frame_id),
                    'timestamp': float(timestamp),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(cx), float(cy)],
                    'confidence': float(conf)
                }

                if track_id not in self.track_history:
                    self.track_history[track_id] = deque(maxlen=self.max_history)
                    self.metrics['created_tracks'] += 1
                    # initialize meta for OCR attempts and locking
                    self.track_meta[track_id] = {'ocr_attempts': 0, 'best_plate': None, 'ocr_locked': False}
                    if callable(self.on_track_started):
                        try:
                            self.on_track_started(track_id, hist_entry.copy())
                        except Exception:
                            pass

                self.track_history[track_id].append(hist_entry)

                # call update hook
                if callable(self.on_track_updated):
                    try:
                        self.on_track_updated(track_id, hist_entry.copy())
                    except Exception:
                        pass

                # map class id to name
                class_name = self.class_map.get(int(cid), 'vehicle')

                out.append({
                    'track_id': track_id,
                    'bbox': (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    'confidence': float(conf),
                    'class': class_name,
                    'class_id': int(cid),
                    'center': (int(round(cx)), int(round(cy))),
                    'frame_id': int(frame_id),
                    'timestamp': float(timestamp),
                })

            except Exception:
                # skip this tracked index on any unexpected error
                continue

        return out

    # Optional utility: mark track as lost (for downstream systems)
    def _handle_lost_track(self, track_id: int):
        self.metrics['lost_tracks'] += 1
        if callable(self.on_track_lost):
            try:
                meta = {'track_id': track_id, 'meta': self.track_meta.get(track_id, {})}
                self.on_track_lost(track_id, meta)
            except Exception:
                pass


# End of module
