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
        st.info("Video upload functionality will be implemented in the next commit.")


def handle_image_sequence_upload():
    """Handle image sequence upload and processing."""
    uploaded_file = st.file_uploader(
        "Choose a ZIP file containing images",
        type=['zip'],
        help="Upload a ZIP file containing a sequence of images"
    )

    if uploaded_file is not None:
        st.info("Image sequence upload functionality will be implemented in the next commit.")


def handle_results_tab():
    """Handle the results display tab."""
    st.info("👆 Please upload and analyze content in the Upload tab first")


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
