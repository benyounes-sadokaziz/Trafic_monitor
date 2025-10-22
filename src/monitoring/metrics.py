"""
Prometheus metrics for monitoring the application
"""

from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Container for all Prometheus metrics."""
    
    def __init__(self):
        # Application info
        self.app_info = Info('traffic_monitor_app', 'Application information')
        self.app_info.info({
            'version': '1.0.0',
            'component': 'traffic_monitor'
        })
        
        # API Metrics
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status']
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint']
        )
        
        # Job Metrics
        self.jobs_created = Counter(
            'jobs_created_total',
            'Total number of jobs created'
        )
        
        self.jobs_processing = Gauge(
            'jobs_processing_current',
            'Number of jobs currently processing'
        )
        
        self.job_completed = Counter(
            'jobs_completed_total',
            'Total number of completed jobs'
        )
        
        self.job_failed = Counter(
            'jobs_failed_total',
            'Total number of failed jobs'
        )
        
        self.job_duration = Histogram(
            'job_duration_seconds',
            'Job processing duration in seconds',
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
        )
        
        # Processing Metrics
        self.frames_processed = Counter(
            'frames_processed_total',
            'Total number of frames processed'
        )
        self.frame_processing_seconds = Histogram(
            'frame_processing_seconds',
            'Per-frame total processing time in seconds',
            buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        )
        
        self.vehicles_detected = Counter(
            'vehicles_detected_total',
            'Total number of vehicles detected'
        )
        
        self.vehicles_tracked = Counter(
            'vehicles_tracked_total',
            'Total number of unique vehicles tracked'
        )
        
        self.plates_detected = Counter(
            'plates_detected_total',
            'Total number of license plates detected'
        )
        
        self.screenshots_saved = Counter(
            'screenshots_saved_total',
            'Total number of plate screenshots saved'
        )
        
        self.speed_violations = Counter(
            'speed_violations_total',
            'Total number of speed violations detected',
            ['vehicle_class']
        )
        
        # Model Performance Metrics
        self.model_inference_duration = Histogram(
            'model_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        self.detection_confidence = Histogram(
            'detection_confidence_score',
            'Detection confidence scores',
            ['detection_type'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # System Metrics
        self.active_websocket_connections = Gauge(
            'active_websocket_connections',
            'Number of active WebSocket connections'
        )
        
        logger.info("✅ Prometheus metrics initialized")


# Global metrics instance
prometheus_metrics = PrometheusMetrics()


# Decorators for automatic tracking
def track_request(func):
    """Decorator to track API request metrics."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        status = 200
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = 500
            raise
        finally:
            duration = time.time() - start_time
            
            # Extract endpoint name
            endpoint = func.__name__
            
            prometheus_metrics.api_requests_total.labels(
                method='POST',
                endpoint=endpoint,
                status=status
            ).inc()
            
            prometheus_metrics.api_request_duration.labels(
                method='POST',
                endpoint=endpoint
            ).observe(duration)
    
    return wrapper


def track_processing_time(func):
    """Decorator to track job processing time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        job_id = kwargs.get('job_id')
        start_time = time.time()
        
        prometheus_metrics.jobs_processing.inc()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            prometheus_metrics.job_duration.observe(duration)
            prometheus_metrics.jobs_processing.dec()
    
    return wrapper


def track_model_inference(model_type: str):
    """Decorator to track model inference time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                prometheus_metrics.model_inference_duration.labels(
                    model_type=model_type
                ).observe(duration)
        
        return wrapper
    return decorator


# Helper functions for manual tracking
def record_vehicle_detection(count: int, avg_confidence: float):
    """Record vehicle detection metrics."""
    prometheus_metrics.vehicles_detected.inc(count)
    prometheus_metrics.detection_confidence.labels(
        detection_type='vehicle'
    ).observe(avg_confidence)


def record_plate_detection(count: int, avg_confidence: float):
    """Record plate detection metrics."""
    prometheus_metrics.plates_detected.inc(count)
    prometheus_metrics.detection_confidence.labels(
        detection_type='plate'
    ).observe(avg_confidence)


def record_speed_violation(vehicle_class: str):
    """Record speed violation."""
    prometheus_metrics.speed_violations.labels(
        vehicle_class=vehicle_class
    ).inc()


def record_frame_processed():
    """Record a processed frame."""
    prometheus_metrics.frames_processed.inc()


def record_screenshot_saved():
    """Record a saved screenshot."""
    prometheus_metrics.screenshots_saved.inc()