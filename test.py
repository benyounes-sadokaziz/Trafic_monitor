import websocket
import json

# Test connection
ws = websocket.WebSocket()

try:
    print("Connecting to WebSocket...")
    ws.connect('ws://localhost:8000/ws/test-job-123')
    print("✅ Connected!")
    
    # Receive welcome message
    msg = ws.recv()
    print(f"Received: {msg}")
    
    # Send ping
    ws.send(json.dumps({"type": "ping"}))
    print("Sent ping")
    
    # Receive pong
    response = ws.recv()
    print(f"Received: {response}")
    
    ws.close()
    print("✅ WebSocket test passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")