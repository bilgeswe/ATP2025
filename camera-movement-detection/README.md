# Camera Movement Detection System

**Detecting Significant Camera Movement Using Advanced Computer Vision Techniques**

This application detects significant camera movement in video sequences or image sequences using multiple computer vision algorithms. It distinguishes between camera movement (global motion) and object movement (local motion) within the scene.

## 🎯 Features

- **Multi-method Detection**: Combines frame differencing and feature matching for robust detection
- **Movement Classification**: Categorizes movement as translation, rotation, scale, or perspective change
- **Interactive Web Interface**: User-friendly Streamlit app with real-time analysis
- **Comprehensive Visualizations**: Timeline charts, movement type distribution, and frame analysis
- **Multiple Input Formats**: Supports video files and image sequences (ZIP)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenCV 4.8+
- Streamlit 1.28+

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd camera-movement-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📊 How It Works

### Detection Algorithm

The system uses a multi-stage approach to detect camera movement:

1. **Frame Differencing**
   - Computes absolute difference between consecutive frames
   - Applies Gaussian blur to reduce noise
   - Calculates percentage of changed pixels
   - Threshold-based detection for basic motion

2. **Feature Matching (ORB)**
   - Detects ORB (Oriented FAST and Rotated BRIEF) keypoints
   - Matches features between consecutive frames using FLANN
   - Applies Lowe's ratio test for robust matching
   - Estimates homography transformation matrix

3. **Homography Analysis**
   - Extracts translation, rotation, and scale from homography matrix
   - Calculates confidence based on transformation magnitude and match count
   - Distinguishes camera movement from object movement

4. **Movement Classification**
   - **Scale**: Significant zoom in/out (scale factor > 1.2 or < 0.8)
   - **Rotation**: Camera rotation (rotation angle > 5°)
   - **Translation**: Camera pan/tilt (translation magnitude > 20 pixels)
   - **Perspective**: Other geometric transformations

### Decision Fusion

The final movement detection combines both methods:
- Movement detected if **either** frame differencing OR feature matching indicates significant change
- Confidence score is the maximum of both method scores
- Movement type is determined by feature matching analysis

## 🎮 Usage

### Web Interface

1. **Upload Content**
   - Choose between video file or image sequence (ZIP)
   - Supported video formats: MP4, AVI, MOV, MKV
   - Supported image formats: PNG, JPG, JPEG, BMP, TIFF

2. **Configure Settings**
   - **Frame Difference Threshold**: Sensitivity for pixel-level changes (0.01-0.2)
   - **Feature Matching Threshold**: Confidence threshold for feature detection (0.1-0.8)
   - **Minimum Feature Matches**: Required matches for reliable detection (5-50)

3. **Run Analysis**
   - Click "Detect Camera Movement" button
   - View real-time progress and results summary

4. **Explore Results**
   - **Movement Timeline**: Interactive chart showing confidence scores over time
   - **Movement Type Distribution**: Pie chart of detected movement categories
   - **Frame Analysis**: Detailed view of frames with detected movement

### Programmatic Usage

```python
from movement_detector import MovementDetector
import cv2

# Initialize detector
detector = MovementDetector(
    diff_threshold=0.05,
    feature_threshold=0.3,
    min_match_count=10
)

# Load images
images = [cv2.imread(f"frame_{i}.jpg") for i in range(10)]

# Detect movement
results = detector.detect_movement_sequence(images)

# Get frames with movement
movement_frames = detector.get_movement_frames(results)
print(f"Movement detected in frames: {movement_frames}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_movement_detector.py
```

The test suite covers:
- Detector initialization and parameter validation
- Frame differencing with identical and different images
- Movement sequence detection with various inputs
- Movement classification for different transformation types
- Homography analysis with identity and translation matrices

## 📈 Performance

### Accuracy
- **High precision** for significant camera movements (translation > 20px, rotation > 5°)
- **Low false positives** due to multi-method validation
- **Robust to lighting changes** through feature-based detection

### Speed
- **Real-time processing** for standard video resolutions
- **Optimized frame sampling** (max 100 frames per video)
- **Efficient feature matching** using FLANN algorithm

### Limitations
- Requires sufficient texture/features in images for reliable detection
- May miss subtle movements below threshold levels
- Performance depends on image quality and resolution

## 🛠️ Technical Details

### Dependencies
- **OpenCV 4.8.1.78**: Computer vision algorithms
- **Streamlit 1.28.1**: Web application framework
- **NumPy 1.24.3**: Numerical computations
- **Plotly 5.17.0**: Interactive visualizations
- **Pillow 10.0.1**: Image processing

### Architecture
```
app.py                    # Streamlit web interface
├── main()               # Application entry point
├── setup_sidebar()      # Configuration controls
├── handle_upload_tab()  # File upload handling
├── handle_results_tab() # Results visualization
└── handle_about_tab()   # Information display

movement_detector.py      # Core detection logic
├── MovementDetector     # Main detection class
├── MovementType         # Movement classification enum
└── MovementResult       # Detection result dataclass

test_movement_detector.py # Comprehensive test suite
```

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Deploy with requirements.txt

### Local Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py

# Access at http://localhost:8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📝 Use Cases

- **Security Camera Monitoring**: Detect camera tampering or repositioning
- **Video Stabilization**: Identify frames requiring stabilization
- **Camera Mount Testing**: Verify stability of camera installations
- **Surveillance Analysis**: Monitor camera movement in security systems
- **Content Analysis**: Detect camera shake in video content

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision algorithms
- Streamlit team for the web framework
- Plotly for interactive visualizations

---

**Built with ❤️ for the ATP Core Talent 2025 Challenge**
