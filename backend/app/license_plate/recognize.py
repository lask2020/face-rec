"""
License Plate Recognition Module (OCR)

Handles:
- Tesseract OCR integration with Thai language support
- Character set validation for Thai plates
- Format-based pattern matching and correction
"""

import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
from typing import Tuple, Optional


class LicensePlateRecognizer:
    """Recognize text from license plate images using OCR."""
    
    # Thai consonants (ก-ฮ)
    THAI_CONSONANTS = set('กขคงจฉซฬมผฝภมาลวศษณฤตท')
    
    # Thai vowels and special characters
    THAI_VOWELS = set("'ัิใไเแโโอๅูืึะำา็์่้๊๋")
    
    # All valid Thai plate characters (consonants + vowels)
    THAI_CHARS = THAI_CONSONANTS | THAI_VOWELS
    
    def __init__(self, lang: str = 'chi_sim+eng+tha'):
        """
        Initialize recognizer with language settings.
        
        Args:
            lang: Tesseract language code (default includes Chinese, English, Thai)
        """
        self.lang = lang
        self.confidence_threshold = 0.5
    
    def recognize(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize text from a license plate ROI.
        
        Args:
            image: Preprocessed plate ROI (grayscale or BGR)
            
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        # Convert to PIL Image for Tesseract
        if len(image.shape) == 3 and image.shape[2] in [3, 4]:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        # Configure Tesseract settings for Thai text
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist:กขคงจฉซฬมผฝภมาลวศษณฤตท0123456789ัิใไเแโโอๅูืึะำา็์่้๊๋'
        
        # Perform OCR
        try:
            text = pytesseract.image_to_string(
                pil_image,
                lang=self.lang,
                config=config
            )
            
            # Clean up the result
            text = self._clean_text(text)
            
            # Calculate confidence based on character count and clarity
            confidence = self._calculate_confidence(image, text)
            
            return (text.strip(), confidence)
        except Exception as e:
            print(f"OCR error: {e}")
            return (None, 0.0)
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR output by removing noise and normalizing."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\sก-ฮ0-9ัิใไเแโโอๅูืึะำา็์่้๊๋]', '', text)
        
        return text
    
    def _calculate_confidence(self, image: np.ndarray, text: str) -> float:
        """Calculate confidence score based on image and text quality."""
        if not text:
            return 0.0
        
        # Base confidence from character count (Thai plates have 5-12 chars)
        char_count = len(text.replace(' ', ''))
        
        # Ideal length for standard plate is around 8 characters
        ideal_length = 8
        length_penalty = abs(char_count - ideal_length) / 10
        
        # Check if all characters are valid Thai/numeric
        valid_chars = sum(1 for c in text.replace(' ', '') 
                         if c in self.THAI_CHARS or c.isdigit())
        invalid_ratio = (len(text.replace(' ', '')) - valid_chars) / len(text.replace(' ', '')) if text else 0
        
        # Calculate final confidence
        base_confidence = max(0.5, 1.0 - length_penalty * 0.3)
        character_confidence = 1.0 - invalid_ratio * 0.4
        
        return min(0.95, base_confidence * character_confidence)
    
    def recognize_with_pil(self, pil_image: Image.Image) -> Tuple[Optional[str], float]:
        """
        Recognize text from PIL Image directly.
        
        Args:
            pil_image: PIL Image of the license plate ROI
            
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        # Convert to grayscale for better OCR results
        if len(pil_image.mode) == 4:  # RGBA
            pil_image = pil_image.convert('L')
        
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist:กขคงจฉซฬมผฝภมาลวศษณฤตท0123456789ัิใไเแโโอๅูืึะำา็์่้๊๋'
        
        try:
            text = pytesseract.image_to_string(
                pil_image,
                lang='tha+chi_sim+eng',
                config=config
            )
            
            text = self._clean_text(text)
            confidence = self._calculate_confidence_from_pil(pil_image, text)
            
            return (text.strip(), confidence)
        except Exception as e:
            print(f"OCR error with PIL: {e}")
            return (None, 0.0)
    
    def _calculate_confidence_from_pil(self, pil_image: Image.Image, text: str) -> float:
        """Calculate confidence from PIL image."""
        if not text:
            return 0.0
        
        width, height = pil_image.size
        
        # Smaller images tend to have lower confidence
        area_penalty = max(0, (1024 - min(width, height)) / 500)
        
        valid_chars = sum(1 for c in text.replace(' ', '') 
                         if c in self.THAI_CHARS or c.isdigit())
        invalid_ratio = (len(text.replace(' ', '')) - valid_chars) / len(text.replace(' ', '')) if text else 0
        
        base_confidence = max(0.5, 1.0 - area_penalty * 0.2)
        character_confidence = 1.0 - invalid_ratio * 0.4
        
        return min(0.95, base_confidence * character_confidence)


def recognize_plate(image_path: str) -> Tuple[Optional[str], float]:
    """
    Convenience function to recognize plate from image file.
    
    Args:
        image_path: Path to input image file
        
    Returns:
        Tuple of (recognized_text, confidence_score)
    """
    recognizer = LicensePlateRecognizer()
    
    # Read image with OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Convert to PIL for recognition
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    return recognizer.recognize_with_pil(pil_image)
