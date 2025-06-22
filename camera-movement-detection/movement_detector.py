"""
Camera Movement Detection Module

This module provides functionality to detect significant camera movement
in image sequences using multiple computer vision techniques.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class MovementType(Enum):
    """Types of camera movement that can be detected."""
    NONE = "none"
    TRANSLATION = "translation"
    ROTATION = "rotation"
    SCALE = "scale"
    PERSPECTIVE = "perspective"


@dataclass
class MovementResult:
    """Result of movement detection for a frame pair."""
    frame_index: int
    movement_detected: bool
    movement_type: MovementType
    confidence: float
    translation: Tuple[float, float] = (0.0, 0.0)
    rotation_angle: float = 0.0
    scale_factor: float = 1.0

def detect_significant_movement(frames: List[np.ndarray], threshold: float = 50.0) -> List[int]:
    """
    Detect frames where significant camera movement occurs.
    Args:
        frames: List of image frames (as numpy arrays).
        threshold: Sensitivity threshold for detecting movement.
    Returns:
        List of indices where significant movement is detected.
    """
    movement_indices = []
    prev_gray = None
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = np.mean(diff)
            if score > threshold:
                movement_indices.append(idx)
        prev_gray = gray
    return movement_indices

class MovementDetector:
    """
    Main class for detecting camera movement in image sequences.
    
    Uses multiple techniques:
    - Frame differencing for basic motion detection
    - Feature matching (ORB) for robust keypoint-based detection
    - Homography estimation for transformation analysis
    - Optical flow for motion vector analysis
    """
    
    def __init__(
        self,
        diff_threshold: float = 0.05,
        feature_threshold: float = 0.3,
        min_match_count: int = 10,
        homography_threshold: float = 5.0
    ):
        """
        Initialize the movement detector.
        
        Args:
            diff_threshold: Threshold for frame differencing (0-1)
            feature_threshold: Threshold for feature matching confidence
            min_match_count: Minimum number of feature matches required
            homography_threshold: RANSAC threshold for homography estimation
        """
        self.diff_threshold = diff_threshold
        self.feature_threshold = feature_threshold
        self.min_match_count = min_match_count
        self.homography_threshold = homography_threshold
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Initialize FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def _frame_differencing(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Detect movement using frame differencing.
        
        Args:
            img1: First grayscale image
            img2: Second grayscale image
            
        Returns:
            Normalized difference score (0-1)
        """
        # Compute absolute difference
        diff = cv2.absdiff(img1, img2)
        
        # Apply Gaussian blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Threshold to binary
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate percentage of changed pixels
        changed_pixels = np.sum(thresh > 0)
        total_pixels = thresh.shape[0] * thresh.shape[1]
        
        return changed_pixels / total_pixels
    
    def _feature_matching(self, img1: np.ndarray, img2: np.ndarray) -> Dict:
        """
        Detect movement using feature matching and homography.
        
        Args:
            img1: First grayscale image
            img2: Second grayscale image
            
        Returns:
            Dictionary with movement analysis results
        """
        # Detect keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < self.min_match_count or len(des2) < self.min_match_count:
            return {
                'movement_detected': False,
                'confidence': 0.0,
                'translation': (0.0, 0.0),
                'rotation': 0.0,
                'scale': 1.0
            }
        
        # Match features
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.min_match_count:
            return {
                'movement_detected': False,
                'confidence': 0.0,
                'translation': (0.0, 0.0),
                'rotation': 0.0,
                'scale': 1.0
            }
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            self.homography_threshold
        )
        
        if homography is None:
            return {
                'movement_detected': False,
                'confidence': 0.0,
                'translation': (0.0, 0.0),
                'rotation': 0.0,
                'scale': 1.0
            }
        
        # Analyze homography matrix
        return self._analyze_homography(homography, len(good_matches))
