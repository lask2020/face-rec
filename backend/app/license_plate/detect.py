"""
License Plate Detection Module

Handles:
- Color-based plate detection (blue text on white/yellow background)
- Contour analysis and shape filtering
- Aspect ratio validation
- Tilt angle estimation
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class LicensePlateDetector:
    """Detect license plates in images using color segmentation and contour analysis."""
    
    def __init__(self):
        # HSV ranges for Thai license plate colors
        self.blue_text_ranges = [
            (100, 50, 50),   # Lower bound
            (125, 50, 50)    # Upper bound
        ]
        
        self.white_background_ranges = [
            (0, 0, 200),     # Lower bound
            (180, 0, 240)    # Upper bound
        ]
        
        self.yellow_background_ranges = [
            (20, 50, 150),   # Lower bound
            (35, 50, 240)    # Upper bound
        ]
    
    def convert_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to HSV color space."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def segment_by_color(self, hsv_image: np.ndarray, lower_bound: Tuple[int, int, int], 
                         upper_bound: Tuple[int, int, int]) -> np.ndarray:
        """Create mask based on HSV color range."""
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        return mask
    
    def detect_blue_text(self, hsv_image: np.ndarray) -> np.ndarray:
        """Detect blue text regions (standard Thai plates)."""
        # Combine multiple blue ranges for better coverage
        masks = []
        for i in range(0, 180, 25):
            lower = (i + 90, 40, 40)
            upper = (i + 130, 60, 255)
            mask = cv2.inRange(hsv_image, lower, upper)
            masks.append(mask)
        
        # Combine all blue text masks
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        return combined_mask
    
    def detect_yellow_background(self, hsv_image: np.ndarray) -> np.ndarray:
        """Detect yellow background plates (commercial/taxis)."""
        lower = (15, 40, 150)
        upper = (35, 255, 255)
        return cv2.inRange(hsv_image, lower, upper)
    
    def find_contours(self, mask: np.ndarray, min_area: int = 1000) -> List[np.ndarray]:
        """Find contours in mask and filter by minimum area."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                valid_contours.append(contour)
        
        return valid_contours
    
    def filter_plate_shape(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """
        Filter plate candidates by shape characteristics.
        
        Thai license plates typically have:
        - Aspect ratio between 1.5:1 and 5:1
        - Support for rotated/skewed angles (using minAreaRect)
        """
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        
        if w == 0 or h == 0:
            return False
            
        # Ensure w is the longer side since plates are horizontal
        if h > w:
            w, h = h, w
            
        # Calculate aspect ratio
        aspect_ratio = w / h
        
        # Typical Thai plate dimensions
        min_width = 80
        max_aspect_ratio = 5.0
        min_aspect_ratio = 1.3
        
        return (aspect_ratio >= min_aspect_ratio and 
                aspect_ratio <= max_aspect_ratio and
                w >= min_width)
    
    def estimate_tilt_angle(self, contour: np.ndarray) -> Optional[float]:
        """Estimate the tilt angle of a detected plate."""
        rect = cv2.minAreaRect(contour)
        
        if rect is None:
            return 0
        
        (center), (width, height), angle = rect
        
        # OpenCV returns angle in range [-90, 0) or [0, 90]
        # Normalize to positive angle for our use case
        if angle < 0:
            angle += 90
        
        return abs(angle)
    
    def detect_plate(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Detect license plates in an image.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of tuples (contour, confidence_score) sorted by confidence
        """
        hsv_image = self.convert_to_hsv(image)
        
        # Try blue text detection first (most common for Thai plates)
        blue_text_mask = self.detect_blue_text(hsv_image)
        
        # Apply morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(blue_text_mask, kernel, iterations=2)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
        
        # Find contours
        contours = self.find_contours(eroded_mask)
        
        plate_candidates = []
        
        for contour in contours:
            if not self.filter_plate_shape(contour, image.shape):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on aspect ratio and area
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # Ideal aspect ratio is around 3:1 for Thai plates
            ideal_aspect_ratio = 3.0
            aspect_diff = abs(aspect_ratio - ideal_aspect_ratio)
            
            # Calculate confidence score (higher is better)
            confidence = max(0, 1 - aspect_diff / 2.0) * 0.9
            
            plate_candidates.append((contour, confidence))
        
        # Sort by confidence (descending)
        plate_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return plate_candidates
    
    def detect_plate_with_color(self, image: np.ndarray) -> List[Tuple[np.ndarray, float, str]]:
        """
        Detect license plates with color classification.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of tuples (contour, confidence_score, plate_type)
            where plate_type is 'standard', 'commercial', or 'graphic'
        """
        hsv_image = self.convert_to_hsv(image)
        
        # Detect blue text plates (standard)
        blue_text_mask = self.detect_blue_text(hsv_image)
        dilated_mask = cv2.dilate(blue_text_mask, kernel=np.ones((3, 3), np.uint8), iterations=2)
        eroded_mask = cv2.erode(dilated_mask, kernel=np.ones((3, 3), np.uint8), iterations=1)
        
        # Detect yellow background plates (commercial/taxis)
        yellow_mask = self.detect_yellow_background(hsv_image)
        dilated_yellow = cv2.dilate(yellow_mask, kernel=np.ones((3, 3), np.uint8), iterations=2)
        eroded_yellow = cv2.erode(dilated_yellow, kernel=np.ones((3, 3), np.uint8), iterations=1)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(eroded_mask, eroded_yellow)
        
        contours = self.find_contours(combined_mask)
        
        plate_candidates = []
        
        for contour in contours:
            if not self.filter_plate_shape(contour, image.shape):
                continue
            
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rect_w, rect_h), angle = rect
            
            # Check if this region has yellow background using bounded ROI
            x, y, w, h = cv2.boundingRect(contour)
            roi_mask = combined_mask[max(0, y):min(combined_mask.shape[0], y+h), 
                                     max(0, x):min(combined_mask.shape[1], x+w)]
            yellow_ratio = np.sum(roi_mask > 0) / max(1, roi_mask.size)
            
            if yellow_ratio > 0.3:
                plate_type = 'commercial'
            else:
                plate_type = 'standard'
            
            # Use actual rotated rect dimensions
            if rect_h > rect_w:
                rect_w, rect_h = rect_h, rect_w
                
            aspect_ratio = rect_w / max(1, rect_h)
            ideal_aspect_ratio = 3.0
            aspect_diff = abs(aspect_ratio - ideal_aspect_ratio)
            confidence = max(0, 1 - aspect_diff / 2.0) * 0.9
            
            plate_candidates.append((contour, confidence, plate_type))
        
        # Sort by confidence (descending)
        plate_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return plate_candidates


def detect_plate(image_path: str) -> List[Tuple[str, float, str]]:
    """
    Convenience function to detect plates from image file.
    
    Args:
        image_path: Path to input image file
        
    Returns:
        List of tuples (image_path, confidence_score, plate_type)
    """
    detector = LicensePlateDetector()
    
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Detect plates
    candidates = detector.detect_plate_with_color(image)
    
    return [(image_path, conf, plate_type) for _, conf, plate_type in candidates]
