"""
WebSocket Client for Real-Time Updates
"""

import json
import asyncio
from typing import Callable, Optional
import websocket
import threading
import logging

logger = logging.getLogger(__name__)


class WebSocketClient:
    """WebSocket client for receiving real-time updates."""
    
    def __init__(self, job_id: str, base_url: str = "ws://localhost:8000"):
        """
        Initialize WebSocket client.
        
        Args:
            job_id: Job ID to monitor
            base_url: Base WebSocket URL
        """
        self.job_id = job_id
        self.ws_url = f"{base_url}/ws/{job_id}"
        self.ws = None
        self.running = False
        self.thread = None
        self.on_message_callback = None
        self.on_error_callback = None
        self.on_close_callback = None
    
    def connect(
        self,
        on_message: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_close: Optional[Callable] = None
    ):
        """
        Connect to WebSocket server.
        
        Args:
            on_message: Callback for messages (receives dict)
            on_error: Callback for errors
            on_close: Callback for connection close
        """
        self.on_message_callback = on_message
        self.on_error_callback = on_error
        self.on_close_callback = on_close
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self):
        """Run WebSocket connection in thread."""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            self.ws.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if self.on_error_callback:
                self.on_error_callback(e)
    
    def _on_open(self, ws):
        """Handle WebSocket connection open."""
        logger.info(f"WebSocket connected to {self.ws_url}")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            if self.on_message_callback:
                self.on_message_callback(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
        if self.on_error_callback:
            self.on_error_callback(error)
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self.on_close_callback:
            self.on_close_callback()
    
    def disconnect(self):
        """Disconnect from WebSocket."""
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=2)