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
