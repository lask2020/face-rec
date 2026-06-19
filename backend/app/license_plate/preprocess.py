"""
Image Preprocessing Module for License Plate Recognition

Handles:
- Basic image enhancement (contrast, noise reduction)
- Perspective correction for tilted plates
- Color space conversion
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR image to grayscale."""
    if len(image.shape) == 3 and image.shape[2] in [3, 4]:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization for contrast enhancement."""
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur for noise reduction."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adaptive_threshold(image: np.ndarray, block_size: int = 11, c: float = 2.0) -> np.ndarray:
    """Apply adaptive thresholding for binarization."""
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )


def detect_plate_orientation(image: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Detect plate orientation using vanishing points.
    
    Returns:
        Tuple of (rotation_angle_x, rotation_angle_y) or None if not detected
    """
    # Convert to grayscale and threshold
    gray = convert_to_grayscale(image)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour (likely the plate)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit a rotated rectangle to find orientation
    rect = cv2.minAreaRect(largest_contour)
    if rect is None:
        return None
    
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    
    # Calculate angles from the rotated rectangle
    angle_x, angle_y = calculate_rotation_angles(box)
    
    return (angle_x, angle_y)


def calculate_rotation_angles(box_points: np.ndarray) -> Tuple[float, float]:
    """Calculate rotation angles from box points."""
    if len(box_points) < 4:
        return (0, 0)
    
    # Get the four corners of the plate
    corners = box_points.reshape(4, 2)
    
    # Calculate vectors between adjacent corners
    v1 = corners[1] - corners[0]
    v2 = corners[2] - corners[1]
    
    # Calculate angles using atan2
    angle_x = np.arctan2(v1[1], v1[0]) * 180 / np.pi
    angle_y = np.arctan2(v2[1], v2[0]) * 180 / np.pi
    
    return (angle_x, angle_y)


def apply_perspective_correction(image: np.ndarray, angles: Tuple[float, float]) -> Optional[np.ndarray]:
    """
    Apply perspective correction to normalize plate view.
    
    Args:
        image: Source image with tilted plate
        angles: Tuple of (x_angle, y_angle) from detection
    
    Returns:
        Corrected image or None if correction not possible
    """
    h, w = image.shape[:2]
    
    # Create transformation matrix for perspective warp
    # Source points: 4 corners of the image
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    # Destination points: apply shifts based on angles
    offset_x = np.tan(np.radians(angles[0])) * w / 4
    offset_y = np.tan(np.radians(angles[1])) * h / 4
    
    dst_pts = np.array([
        [0 - offset_x, 0 - offset_y],
        [w - offset_x, 0 + offset_y],
        [w + offset_x, h + offset_y],
        [0 + offset_x, h - offset_y]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply perspective transform
    try:
        corrected = cv2.warpPerspective(image, M, (w, h))
        return corrected
    except Exception as e:
        print(f"Perspective correction failed: {e}")
        return None


def preprocess_image(
    image: np.ndarray,
    use_perspective_correction: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline for license plate recognition.
    
    Args:
        image: Input image (BGR format from OpenCV)
        use_perspective_correction: Whether to apply perspective correction
    
    Returns:
        Tuple of (processed_image, metadata_dict)
    """
    # Convert to grayscale
    gray = convert_to_grayscale(image)
    
    # Apply histogram equalization for contrast enhancement
    enhanced = apply_histogram_equalization(gray)
    
    # Apply Gaussian blur for noise reduction
    blurred = apply_gaussian_blur(enhanced, kernel_size=5)
    
    metadata = {
        'original_shape': image.shape,
        'processed_shape': blurred.shape,
        'perspective_corrected': use_perspective_correction,
        'angles_applied': None
    }
    
    # Apply perspective correction if requested and angles detected
    if use_perspective_correction:
        angles = detect_plate_orientation(blurred)
        if angles:
            corrected = apply_perspective_correction(blurred, angles)
            if corrected is not None:
                metadata['angles_applied'] = angles
                return (corrected, metadata)
    
    # Return enhanced image without perspective correction if needed
    return (blurred, metadata)
