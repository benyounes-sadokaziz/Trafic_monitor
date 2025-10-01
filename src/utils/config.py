# src/utils/config.py

import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration manager for traffic monitoring system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'tracking', 'speed', 'ocr', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        return config
    
    def get(self, key: str, default=None):
        """Get configuration value by dot notation (e.g., 'model.name')"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str):
        """Allow dictionary-style access"""
        return self.config[key]
    
    def __repr__(self):
        return f"Config(path={self.config_path})"


# Singleton instance
_config = None

def get_config(config_path: str = "config/config.yaml") -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config