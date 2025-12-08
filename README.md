# 🚗 Traffic Monitor - AI-Powered Vehicle Detection & Speed Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A production-ready traffic monitoring system that uses deep learning for real-time vehicle detection, tracking, license plate recognition, and accurate speed estimation with violation detection.

![Traffic Monitor Demo](docs/demo.gif)

## ✨ Features

- **🚙 Vehicle Detection**: YOLOv8-based detection for cars, trucks, buses, motorcycles, and bicycles
- **📍 Multi-Object Tracking**: ByteTrack algorithm for robust vehicle tracking across frames
- **📸 License Plate Recognition**: Automatic plate detection with quality-based screenshot management
- **⚡ Speed Estimation**: Homography-based real-world speed calculation with outlier rejection
- **🚨 Violation Detection**: Configurable speed limits per vehicle type with automatic violation logging
- **🎥 Real-time Processing**: WebSocket-based live video processing with progress updates
- **📊 Monitoring & Metrics**: Prometheus metrics + Grafana dashboards for system observability
- **🖥️ Interactive Dashboard**: Streamlit-based UI for video upload, monitoring, and visualization
- **🐳 Docker Support**: Fully containerized deployment with Docker Compose

## 🏗️ Architecture

![Traffic Monitor Architecture](docs/architecture.png)

### System Components

**Frontend Layer:**
- **Streamlit Dashboard**: User interface for video upload, real-time monitoring, results visualization, and job history

**API Layer:**
- **FastAPI Backend**: REST API endpoints, WebSocket manager for live updates, and background task processor

**Processing Pipeline:**
1. **YOLOv8 Vehicle Detector**: Detects vehicles and returns bounding boxes
2. **ByteTrack Tracker**: Assigns unique IDs and tracks vehicles across frames
3. **YOLOv8 Plate Detector**: Detects license plates within vehicle crops (parallel processing)
4. **Quality Assessor**: Evaluates plate image quality (sharpness, brightness, size)
5. **Smart Screenshot Storage**: Saves only the best quality plate images per vehicle
6. **Homography Speed Estimator**: Calculates real-world speeds using 9-point calibration
7. **Speed Checker**: Detects violations based on configurable limits per vehicle type

**Monitoring Layer:**
- **Prometheus**: Metrics collection (frame latency, detection rates, violations)
- **Grafana**: Real-time dashboards and visualization

**Storage:**
- Input videos, processed screenshots, speed data (JSON), violation logs

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for real-time processing)
- 8GB+ RAM
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/traffic_monitor.git
cd traffic_monitor
```

2. **Create virtual environment**
```powershell
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download YOLO models**

Place your trained models in the `models/` directory:
- `yolov8n.pt` - Vehicle detection model
- `best.pt` - License plate detection model

### Running the Application

#### Option 1: Local Development

**Terminal 1 - FastAPI Backend:**
```powershell
venv\Scripts\activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Streamlit Dashboard:**
```powershell
venv\Scripts\activate
streamlit run src/dashboard/app.py
```

**Terminal 3 - Observability Stack (Optional):**
```powershell
docker-compose -f docker/docker-compose.observability.yml up -d
```

#### Option 2: Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up -d
```

Access the services:
- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 📖 Usage

### 1. Upload Video for Processing

1. Open the Streamlit dashboard at http://localhost:8501
2. Navigate to the **Upload** tab
3. Configure processing parameters:
   - **Speed Limits**: Set limits for each vehicle type (car, truck, bus, motorcycle, bicycle)
   - **Max Frames**: Optionally limit processing for testing
   - **Save Output**: Enable to save annotated video
4. Upload your video file (MP4, AVI, MOV)
5. Click **Start Processing**

### 2. Monitor Processing

Switch to the **Monitor** tab to watch:
- Real-time frame processing
- Live vehicle tracking
- Speed measurements
- Violation detection
- Progress percentage

### 3. View Results

Navigate to the **Visualize** tab to:
- Review speed statistics
- Analyze violations
- View license plate screenshots
- Download speed data (JSON)

### 4. Check History

The **History** tab shows:
- All processed jobs
- Processing statistics
- Job status and errors

## ⚙️ Configuration

### Speed Limits

Configure speed limits per vehicle type via the Streamlit UI or API:

```json
{
  "car": 120,
  "truck": 90,
  "bus": 90,
  "motorcycle": 120,
  "bicycle": 30
}
```

### Homography Calibration

Speed estimation uses homography transformation for accurate real-world measurements. Configure calibration points in `src/pipeline/orchestrator.py`:

```python
homography_src_points = np.array([
    # Pixel coordinates on video frame (9-point grid)
    [280, 680], [550, 680], [820, 680],  # Bottom row
    [380, 480], [550, 480], [720, 480],  # Middle row
    [440, 280], [550, 280], [660, 280]   # Top row
], dtype=np.float32)

homography_dst_points = np.array([
    # Real-world coordinates in meters
    [0, 0], [3.6, 0], [7.2, 0],      # 0m depth
    [0, 9], [3.6, 9], [7.2, 9],      # 9m depth
    [0, 18], [3.6, 18], [7.2, 18]    # 18m depth
], dtype=np.float32)
```

### Environment Variables

Create a `.env` file:

```env
# Application
APP_NAME="Traffic Monitor"
DEBUG=False
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Models
YOLO_VEHICLE_MODEL=models/yolov8n.pt
YOLO_PLATE_MODEL=models/best.pt

# Paths
INPUT_DIR=data/input
OUTPUT_DIR=data/output
```

## 📊 API Endpoints

### Process Video
```http
POST /api/process
Content-Type: multipart/form-data

Parameters:
  - file: video file (MP4/AVI/MOV)
  - max_frames: int (optional)
  - save_output_video: bool
  - speed_limits: JSON string (optional)
```

### Get Job Status
```http
GET /api/jobs/{job_id}

Response:
{
  "job_id": "uuid",
  "status": "processing|completed|failed",
  "progress": 75.5,
  "stats": {
    "processed_frames": 1500,
    "total_tracks": 25,
    "violations_count": 3
  }
}
```

### Get Tracked Vehicles
```http
GET /api/jobs/{job_id}/tracks

Response: [
  {
    "track_id": 1,
    "class": "car",
    "speed": 95.5,
    "is_violation": false,
    "plate_screenshot": "data:image/jpeg;base64,..."
  }
]
```

### WebSocket Live Updates
```javascript
ws://localhost:8000/ws/{job_id}

Messages:
{
  "type": "frame_update",
  "frame_number": 150,
  "tracks": [...],
  "violations": [...]
}
```

## 🔧 Development

### Project Structure

```
traffic_monitor/
├── config/                  # Configuration files
│   ├── settings.py         # Pydantic settings
│   └── model_config.yaml   # Model configurations
├── src/
│   ├── api/                # FastAPI backend
│   │   ├── main.py        # API endpoints
│   │   ├── schemas.py     # Pydantic models
│   │   └── websocket.py   # WebSocket manager
│   ├── dashboard/          # Streamlit frontend
│   │   ├── app.py         # Main dashboard
│   │   └── components/    # UI components
│   ├── detection/          # Vehicle detection
│   ├── tracking/           # ByteTrack tracker
│   ├── ocr/                # Plate detection & OCR
│   ├── speed/              # Speed estimation
│   ├── pipeline/           # Main orchestrator
│   └── monitoring/         # Prometheus metrics
├── models/                 # YOLO model files
├── data/                   # Input/output data
├── docker/                 # Docker configurations
├── monitoring/             # Grafana dashboards
└── test/                   # Test files
```

### Running Tests

```bash
pytest test/ -v
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Format code
black src/
```

## 📈 Monitoring

Access Grafana dashboards at http://localhost:3000:

**Metrics tracked:**
- Frame processing latency
- Vehicle detection rate
- Model inference times
- Active WebSocket connections
- Jobs processed per minute
- Speed violations detected

## 🐛 Troubleshooting

### High/Inaccurate Speeds
- Verify homography calibration points match your camera view
- Check that reference points are accurately measured
- Ensure 9-point grid covers the entire road area

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU mode if needed (slower)
# Set device='cpu' in orchestrator.py
```

### WebSocket Connection Fails
- Ensure both API and dashboard are running
- Check firewall settings for port 8000
- Verify CORS settings in main.py

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [Supervision](https://github.com/roboflow/supervision) - Computer vision tools

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Always comply with local privacy laws and regulations when deploying traffic monitoring systems.