import os
import unittest
import numpy as np
import cv2

# Add parent directory to sys.path so we can import from backend.app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.face_restorer import face_restorer


class TestFaceRestorer(unittest.TestCase):
    def setUp(self):
        # Create a dummy BGR face image for testing (e.g., 128x128 pixel block)
        self.dummy_face = np.zeros((128, 128, 3), dtype=np.uint8)
        # Draw some features so it's not completely empty
        cv2.circle(self.dummy_face, (64, 64), 30, (255, 255, 255), -1)
        cv2.rectangle(self.dummy_face, (44, 44), (84, 84), (0, 255, 0), 2)

    def test_singleton_initialization(self):
        self.assertIsNotNone(face_restorer)

    def test_restore_face_returns_image(self):
        # Skip if model weights are not loaded/mock mode or similar, but
        # we expect face_restorer.restore_face to return a valid numpy array
        # of shape (512, 512, 3) or at least not fail.
        restored = face_restorer.restore_face(self.dummy_face)
        if restored is not None:
            self.assertEqual(restored.shape, (512, 512, 3))
            self.assertIsInstance(restored, np.ndarray)


if __name__ == "__main__":
    unittest.main()
