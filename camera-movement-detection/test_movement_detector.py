"""
Unit tests for the MovementDetector class.
"""

import unittest
import numpy as np
import cv2
from movement_detector import MovementDetector, MovementType, MovementResult


class TestMovementDetector(unittest.TestCase):
    """Test cases for MovementDetector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = MovementDetector()
        
        # Create test images
        self.img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        self.img2 = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Add some features to the images
        cv2.rectangle(self.img1, (20, 20), (40, 40), (255, 255, 255), -1)
        cv2.rectangle(self.img2, (25, 25), (45, 45), (255, 255, 255), -1)  # Shifted
    
    def test_detector_initialization(self):
        """Test detector initialization with default parameters."""
        detector = MovementDetector()
        self.assertEqual(detector.diff_threshold, 0.05)
        self.assertEqual(detector.feature_threshold, 0.3)
        self.assertEqual(detector.min_match_count, 10)
        self.assertEqual(detector.homography_threshold, 5.0)
    
    def test_detector_custom_parameters(self):
        """Test detector initialization with custom parameters."""
        detector = MovementDetector(
            diff_threshold=0.1,
            feature_threshold=0.5,
            min_match_count=20,
            homography_threshold=10.0
        )
        self.assertEqual(detector.diff_threshold, 0.1)
        self.assertEqual(detector.feature_threshold, 0.5)
        self.assertEqual(detector.min_match_count, 20)
        self.assertEqual(detector.homography_threshold, 10.0)
    
    def test_frame_differencing_identical_images(self):
        """Test frame differencing with identical images."""
        img = np.zeros((50, 50), dtype=np.uint8)
        diff_score = self.detector._frame_differencing(img, img)
        self.assertEqual(diff_score, 0.0)
    
    def test_frame_differencing_different_images(self):
        """Test frame differencing with different images."""
        img1 = np.zeros((50, 50), dtype=np.uint8)
        img2 = np.ones((50, 50), dtype=np.uint8) * 255
        diff_score = self.detector._frame_differencing(img1, img2)
        self.assertGreater(diff_score, 0.0)
        self.assertLessEqual(diff_score, 1.0)
    
    def test_detect_movement_sequence_empty(self):
        """Test movement detection with empty sequence."""
        results = self.detector.detect_movement_sequence([])
        self.assertEqual(len(results), 0)
    
    def test_detect_movement_sequence_single_image(self):
        """Test movement detection with single image."""
        results = self.detector.detect_movement_sequence([self.img1])
        self.assertEqual(len(results), 0)
    
    def test_detect_movement_sequence_multiple_images(self):
        """Test movement detection with multiple images."""
        images = [self.img1, self.img2, self.img1]
        results = self.detector.detect_movement_sequence(images)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], MovementResult)
        self.assertIsInstance(results[1], MovementResult)
        self.assertEqual(results[0].frame_index, 1)
        self.assertEqual(results[1].frame_index, 2)
    
    def test_movement_result_attributes(self):
        """Test MovementResult object attributes."""
        result = MovementResult(
            frame_index=1,
            movement_detected=True,
            movement_type=MovementType.TRANSLATION,
            confidence=0.8,
            translation=(10.0, 5.0),
            rotation_angle=2.5,
            scale_factor=1.1
        )
        
        self.assertEqual(result.frame_index, 1)
        self.assertTrue(result.movement_detected)
        self.assertEqual(result.movement_type, MovementType.TRANSLATION)
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.translation, (10.0, 5.0))
        self.assertEqual(result.rotation_angle, 2.5)
        self.assertEqual(result.scale_factor, 1.1)
    
    def test_get_movement_frames(self):
        """Test extraction of movement frame indices."""
        results = [
            MovementResult(1, True, MovementType.TRANSLATION, 0.8),
            MovementResult(2, False, MovementType.NONE, 0.1),
            MovementResult(3, True, MovementType.ROTATION, 0.9),
        ]
        
        movement_frames = self.detector.get_movement_frames(results)
        self.assertEqual(movement_frames, [1, 3])
    
    def test_classify_movement_none(self):
        """Test movement classification for no movement."""
        feature_result = {'movement_detected': False}
        movement_type = self.detector._classify_movement(feature_result)
        self.assertEqual(movement_type, MovementType.NONE)
    
    def test_classify_movement_translation(self):
        """Test movement classification for translation."""
        feature_result = {
            'movement_detected': True,
            'translation_magnitude': 25.0,
            'rotation': 1.0,
            'scale': 1.05
        }
        movement_type = self.detector._classify_movement(feature_result)
        self.assertEqual(movement_type, MovementType.TRANSLATION)
    
    def test_classify_movement_rotation(self):
        """Test movement classification for rotation."""
        feature_result = {
            'movement_detected': True,
            'translation_magnitude': 5.0,
            'rotation': 10.0,
            'scale': 1.05
        }
        movement_type = self.detector._classify_movement(feature_result)
        self.assertEqual(movement_type, MovementType.ROTATION)
    
    def test_classify_movement_scale(self):
        """Test movement classification for scale."""
        feature_result = {
            'movement_detected': True,
            'translation_magnitude': 5.0,
            'rotation': 1.0,
            'scale': 1.3
        }
        movement_type = self.detector._classify_movement(feature_result)
        self.assertEqual(movement_type, MovementType.SCALE)
    
    def test_analyze_homography_identity(self):
        """Test homography analysis with identity matrix."""
        H = np.eye(3, dtype=np.float32)
        result = self.detector._analyze_homography(H, 20)
        
        self.assertFalse(result['movement_detected'])
        self.assertEqual(result['translation'], (0.0, 0.0))
        self.assertAlmostEqual(result['rotation'], 0.0, places=1)
        self.assertAlmostEqual(result['scale'], 1.0, places=1)
    
    def test_analyze_homography_translation(self):
        """Test homography analysis with translation."""
        H = np.eye(3, dtype=np.float32)
        H[0, 2] = 20.0  # Translation in x
        H[1, 2] = 15.0  # Translation in y
        
        result = self.detector._analyze_homography(H, 20)
        
        self.assertTrue(result['movement_detected'])
        self.assertEqual(result['translation'], (20.0, 15.0))
        self.assertGreater(result['translation_magnitude'], 15.0)


class TestMovementType(unittest.TestCase):
    """Test cases for MovementType enum."""
    
    def test_movement_type_values(self):
        """Test MovementType enum values."""
        self.assertEqual(MovementType.NONE.value, "none")
        self.assertEqual(MovementType.TRANSLATION.value, "translation")
        self.assertEqual(MovementType.ROTATION.value, "rotation")
        self.assertEqual(MovementType.SCALE.value, "scale")
        self.assertEqual(MovementType.PERSPECTIVE.value, "perspective")


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\nAll {result.testsRun} tests passed successfully!")
    else:
        print(f"\n{len(result.failures)} test(s) failed, {len(result.errors)} error(s)") 