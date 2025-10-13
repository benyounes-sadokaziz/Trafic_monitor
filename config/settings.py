from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = Field(default="Traffic Monitor")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    
    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)
    
    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/traffic_monitor"
    )
    database_echo: bool = Field(default=False)
    
    # Storage paths
    data_dir: Path = Field(default=Path("./data"))
    input_dir: Path = Field(default=Path("./data/input"))
    output_dir: Path = Field(default=Path("./data/output"))
    plates_dir: Path = Field(default=Path("./data/plates"))
    
    # Models
    yolo_vehicle_model: Path = Field(default=Path("./models/yolov8n.pt"))
    yolo_plate_model: Path = Field(default=Path("./models/best.pt"))
    tracking_method: str = Field(default="bytetrack")
    
    # Processing
    max_queue_size: int = Field(default=100)
    worker_concurrency: int = Field(default=2)
    batch_size: int = Field(default=1)
    
    # Monitoring
    prometheus_port: int = Field(default=9090)
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    
    # WebSocket
    websocket_heartbeat_interval: int = Field(default=30)
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.data_dir, self.input_dir, self.output_dir, self.plates_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
settings.ensure_directories()