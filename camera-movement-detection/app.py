"""
Streamlit Web Application for Camera Movement Detection

This app allows users to upload image sequences or videos and detect
significant camera movement using computer vision techniques.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from typing import List
import plotly.graph_objects as go
import plotly.express as px
import zipfile

from movement_detector import MovementDetector, MovementResult, MovementType


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Camera Movement Detection",
        page_icon="📹",
        layout="wide"
    )

    st.title("📹 Camera Movement Detection")
    st.markdown("""
    This application detects significant camera movement in image sequences using advanced computer vision techniques.
    Upload a video file or a sequence of images to analyze camera motion patterns.
    """)

    # Sidebar for configuration
    setup_sidebar()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 Upload & Analyze", "📊 Results", "ℹ️ About"])

    with tab1:
        handle_upload_tab()

    with tab2:
        handle_results_tab()

    with tab3:
        handle_about_tab()


def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.header("⚙️ Detection Settings")

    st.session_state.diff_threshold = st.sidebar.slider(
        "Frame Difference Threshold",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Sensitivity for frame differencing detection"
    )

    st.session_state.feature_threshold = st.sidebar.slider(
        "Feature Matching Threshold",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.1,
        help="Confidence threshold for feature matching"
    )

    st.session_state.min_matches = st.sidebar.slider(
        "Minimum Feature Matches",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Minimum number of feature matches required"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Instructions")
    st.sidebar.markdown("""
    1. **Upload Content**: Choose video file or image sequence
    2. **Configure Settings**: Adjust detection sensitivity
    3. **Run Analysis**: Click 'Detect Movement' button
    4. **View Results**: Check the Results tab for detailed analysis
    """)


def handle_upload_tab():
    """Handle the upload and analysis tab."""
    st.header("Upload Your Content")

    # File upload options
    upload_option = st.radio(
        "Choose upload type:",
        ["📹 Video File", "🖼️ Image Sequence (ZIP)"],
        horizontal=True
    )

    if upload_option == "📹 Video File":
        handle_video_upload()
    else:
        handle_image_sequence_upload()


def handle_video_upload():
    """Handle video file upload and processing."""
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to extract frames and analyze camera movement"
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        try:
            # Extract frames from video
            with st.spinner("Extracting frames from video..."):
                frames = extract_frames_from_video(video_path)

            if frames:
                st.success(f"Extracted {len(frames)} frames from video")

                # Show video info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Frames", len(frames))
                with col2:
                    st.metric("Frame Size", f"{frames[0].shape[1]}×{frames[0].shape[0]}")

                # Display sample frames
                display_sample_frames(frames)

                # Run detection
                if st.button("🔍 Detect Camera Movement", type="primary"):
                    run_movement_detection(frames)
            else:
                st.error("Could not extract frames from video. Please check the file format.")

        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)


def handle_image_sequence_upload():
    """Handle image sequence upload and processing."""
    uploaded_file = st.file_uploader(
        "Choose a ZIP file containing images",
        type=['zip'],
        help="Upload a ZIP file containing a sequence of images"
    )

    if uploaded_file is not None:
        try:
            # Extract images from ZIP
            with st.spinner("Extracting images from ZIP file..."):
                images = extract_images_from_zip(uploaded_file)

            if images:
                st.success(f"Loaded {len(images)} images from ZIP file")

                # Show image info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Images", len(images))
                with col2:
                    st.metric("Image Size", f"{images[0].shape[1]}×{images[0].shape[0]}")

                # Display sample images
                display_sample_frames(images)

                # Run detection
                if st.button("🔍 Detect Camera Movement", type="primary"):
                    run_movement_detection(images)
            else:
                st.error("Could not load images from ZIP file. Please check the file contents.")

        except Exception as e:
            st.error(f"Error processing ZIP file: {str(e)}")


def extract_frames_from_video(video_path: str, max_frames: int = 100) -> List[np.ndarray]:
    """
    Extract frames from video file.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract

    Returns:
        List of frame arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)

        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_count += 1

    finally:
        cap.release()

    return frames


def extract_images_from_zip(zip_file) -> List[np.ndarray]:
    """
    Extract images from ZIP file.

    Args:
        zip_file: Uploaded ZIP file

    Returns:
        List of image arrays
    """
    images = []

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get list of image files
        image_files = [f for f in zip_ref.namelist()
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        # Sort files to maintain sequence
        image_files.sort()

        for file_name in image_files:
            with zip_ref.open(file_name) as img_file:
                # Read image
                img_data = img_file.read()
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return images


def display_sample_frames(frames: List[np.ndarray], num_samples: int = 4):
    """Display sample frames from the sequence."""
    st.subheader("Sample Frames")

    if len(frames) < num_samples:
        num_samples = len(frames)

    # Select evenly spaced frames
    indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)

    cols = st.columns(num_samples)
    for i, idx in enumerate(indices):
        with cols[i]:
            st.image(frames[idx], caption=f"Frame {idx + 1}", use_column_width=True)


def run_movement_detection(frames: List[np.ndarray]):
    """Run movement detection on the frame sequence."""
    if 'diff_threshold' not in st.session_state:
        st.session_state.diff_threshold = 0.05
    if 'feature_threshold' not in st.session_state:
        st.session_state.feature_threshold = 0.3
    if 'min_matches' not in st.session_state:
        st.session_state.min_matches = 10

    # Initialize detector with current settings
    detector = MovementDetector(
        diff_threshold=st.session_state.diff_threshold,
        feature_threshold=st.session_state.feature_threshold,
        min_match_count=st.session_state.min_matches
    )

    # Run detection
    with st.spinner("Analyzing camera movement..."):
        results = detector.detect_movement_sequence(frames)

    # Store results in session state
    st.session_state.detection_results = results
    st.session_state.frames = frames

    # Display summary
    movement_frames = detector.get_movement_frames(results)

    if movement_frames:
        st.success(f"✅ Detected significant camera movement in {len(movement_frames)} frames")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames", len(frames))
        with col2:
            st.metric("Movement Frames", len(movement_frames))
        with col3:
            st.metric("Movement Percentage", f"{len(movement_frames)/len(results)*100:.1f}%")

        st.info("📊 Check the **Results** tab for detailed analysis and visualizations")
    else:
        st.info("ℹ️ No significant camera movement detected in the sequence")


def handle_results_tab():
    """Handle the results display tab."""
    if 'detection_results' not in st.session_state:
        st.info("👆 Please upload and analyze content in the Upload tab first")
        return

    results = st.session_state.detection_results
    frames = st.session_state.frames

    st.header("📊 Detection Results")

    # Movement timeline
    create_movement_timeline(results)

    # Movement type distribution
    create_movement_type_chart(results)

    # Detailed frame analysis
    display_movement_frames(results, frames)


def create_movement_timeline(results: List[MovementResult]):
    """Create a timeline visualization of movement detection."""
    st.subheader("Movement Timeline")

    frame_indices = [r.frame_index for r in results]
    confidences = [r.confidence for r in results]
    movement_detected = [1 if r.movement_detected else 0 for r in results]

    fig = go.Figure()

    # Add confidence line
    fig.add_trace(go.Scatter(
        x=frame_indices,
        y=confidences,
        mode='lines+markers',
        name='Confidence Score',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))

    # Add movement detection markers
    movement_frames = [r.frame_index for r in results if r.movement_detected]
    movement_confidences = [r.confidence for r in results if r.movement_detected]

    if movement_frames:
        fig.add_trace(go.Scatter(
            x=movement_frames,
            y=movement_confidences,
            mode='markers',
            name='Movement Detected',
            marker=dict(color='red', size=8, symbol='diamond')
        ))

    fig.update_layout(
        title="Camera Movement Detection Timeline",
        xaxis_title="Frame Index",
        yaxis_title="Confidence Score",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def create_movement_type_chart(results: List[MovementResult]):
    """Create a chart showing distribution of movement types."""
    st.subheader("Movement Type Distribution")

    # Count movement types
    movement_counts = {}
    for result in results:
        if result.movement_detected:
            movement_type = result.movement_type.value
            movement_counts[movement_type] = movement_counts.get(movement_type, 0) + 1

    if movement_counts:
        # Create pie chart
        fig = px.pie(
            values=list(movement_counts.values()),
            names=list(movement_counts.keys()),
            title="Types of Camera Movement Detected"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No movement types detected in the sequence")


def display_movement_frames(results: List[MovementResult], frames: List[np.ndarray]):
    """Display frames where movement was detected."""
    st.subheader("Frames with Detected Movement")

    movement_results = [r for r in results if r.movement_detected]

    if not movement_results:
        st.info("No frames with significant movement detected")
        return

    # Display movement frames with details
    for i, result in enumerate(movement_results):
        frame_idx = result.frame_index - 1  # Convert to 0-based index
        
        if frame_idx < len(frames):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(frames[frame_idx], caption=f"Frame {result.frame_index}", use_column_width=True)
            
            with col2:
                st.markdown(f"**Movement Type:** {result.movement_type.value}")
                st.markdown(f"**Confidence:** {result.confidence:.3f}")
                st.markdown(f"**Translation:** ({result.translation[0]:.1f}, {result.translation[1]:.1f})")
                st.markdown(f"**Rotation:** {result.rotation_angle:.1f}°")
                st.markdown(f"**Scale:** {result.scale_factor:.3f}")
        
        if i < len(movement_results) - 1:
            st.markdown("---")


def handle_about_tab():
    """Handle the about tab."""
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### Camera Movement Detection
    
    This application uses advanced computer vision techniques to detect significant camera movement in video sequences or image sequences.
    
    #### How it works:
    1. **Frame Differencing**: Compares consecutive frames to detect pixel-level changes
    2. **Feature Matching**: Uses ORB (Oriented FAST and Rotated BRIEF) features to find corresponding points between frames
    3. **Homography Analysis**: Estimates transformation matrices to distinguish camera movement from object movement
    4. **Movement Classification**: Categorizes detected movement as translation, rotation, scale, or perspective change
    
    #### Technologies Used:
    - **OpenCV**: Computer vision algorithms and image processing
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    - **NumPy**: Numerical computations
    
    #### Use Cases:
    - Security camera monitoring
    - Video stabilization analysis
    - Camera mount stability testing
    - Motion detection in surveillance systems
    """)


if __name__ == "__main__":
    main()
