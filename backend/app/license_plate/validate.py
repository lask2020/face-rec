"""
License Plate Validation Module

Handles:
- Thai license plate format validation (old/new format)
- Motorcycle 3-row plate validation
- Commercial/graphic plate validation
- Common OCR error correction
"""

import re
from typing import Tuple, Optional


class LicensePlateValidator:
    """Validate and normalize Thai license plates."""
    
    # Old format pattern: [ก-ฮ]{2}-[1-9][0-9]{0,3}
    OLD_FORMAT_PATTERN = r'(?<![ก-ฮ0-9])([ก-ฮ]{2})\s*-\s*([1-9]\d{0,3})(?![ก-ฮ0-9])'
    
    # New format pattern: [1-9]-[ก-ฮ]{2}-[1-9][0-9]{0,3}
    NEW_FORMAT_PATTERN = r'(?<![ก-ฮ0-9])([1-9])\s*-?\s*([ก-ฮ]{2})\s*-\s*([1-9]\d{0,3})(?![ก-ฮ0-9])'
    
    # Motorcycle pattern: Row1-[ก-ฮ]+-Row2-[จังหวัด]-Row3-[1-9][0-9]{0,3}
    MOTORCYCLE_PATTERN = r'(?<![ก-ฮ0-9])([1-9]?[ก-ฮ]{1,2})\s*-\s*([ก-ฮ]+)\s*-\s*([1-9]\d{0,3})(?![ก-ฮ0-9])'
    
    # Commercial plate pattern (same as standard but with color indicator)
    COMMERCIAL_PATTERN = OLD_FORMAT_PATTERN
    
    # Valid Thai consonants for plates
    VALID_CONSONANTS = set('กขคงจฉซฬมผฝภมาลวศษณฤตท')
    
    # Valid digits
    VALID_DIGITS = set('0123456789')
    
    def __init__(self):
        """Initialize validator."""
        pass
    
    @staticmethod
    def is_valid_thai_char(char: str) -> bool:
        """Check if character is a valid Thai consonant for license plates."""
        return char in LicensePlateValidator.VALID_CONSONANTS or char.isdigit()
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by standardizing format without destroying word boundaries."""
        if not text:
            return ""
        # Clean obvious OCR artifacts but keep spaces so our Lookaround boundaries work!
        cleaned = re.sub(r'[^\w\sก-ฮ0-9\-]', '', text)
        return cleaned
    
    @staticmethod
    def validate_old_format(text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate old format plate: [ก-ฮ]{2}-[1-9][0-9]{0,3}
        
        Examples: กข 1234, คค 5678
        
        Returns:
            Tuple of (is_valid, normalized_plate)
        """
        text = LicensePlateValidator.normalize_text(text)
        
        # Match old format pattern
        match = re.search(LicensePlateValidator.OLD_FORMAT_PATTERN, text)
        
        if match:
            return (True, f"{match.group(1)}-{match.group(2)}")
        
        return (False, None)
    
    @staticmethod
    def validate_new_format(text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate new format plate: [1-9]-[ก-ฮ]{2}-[1-9][0-9]{0,3}
        
        Examples: 1-กข 1234, 5-คค 5678
        
        Returns:
            Tuple of (is_valid, normalized_plate)
        """
        text = LicensePlateValidator.normalize_text(text)
        
        # Match new format pattern
        match = re.search(LicensePlateValidator.NEW_FORMAT_PATTERN, text)
        
        if match:
            return (True, f"{match.group(1)}-{match.group(2)}-{match.group(3)}")
        
        return (False, None)
    
    @staticmethod
    def validate_motorcycle(text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate motorcycle plate with 3 rows.
        
        Examples: 1กข-กรุงเทพมหานคร-1234
        
        Returns:
            Tuple of (is_valid, normalized_plate)
        """
        text = LicensePlateValidator.normalize_text(text)
        
        # Try to match motorcycle pattern
        match = re.search(LicensePlateValidator.MOTORCYCLE_PATTERN, text)
        
        if match:
            row1 = match.group(1)
            row2 = match.group(2)  # Province name (abbreviated or full)
            row3 = match.group(3)
            
            # Normalize to standard format
            normalized = f"{row1}-{row2}-{row3}"
            return (True, normalized)
        
        return (False, None)
    
    @staticmethod
    def validate_commercial(text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate commercial plate (yellow background).
        
        Same format as standard plates.
        
        Returns:
            Tuple of (is_valid, normalized_plate)
        """
        return LicensePlateValidator.validate_old_format(text)
    
    @staticmethod
    def validate_graphic(text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate graphic plate (auction plates).
        
        Same format as standard plates.
        
        Returns:
            Tuple of (is_valid, normalized_plate)
        """
        return LicensePlateValidator.validate_old_format(text)
    
    @staticmethod
    def validate(text: str, plate_type: Optional[str] = None) -> Tuple[bool, Optional[str], float]:
        """
        Validate license plate with automatic type detection.
        
        Args:
            text: Raw OCR output
            plate_type: Optional plate type ('standard', 'motorcycle', etc.)
            
        Returns:
            Tuple of (is_valid, normalized_plate, confidence_score)
        """
        if not text or len(text.strip()) == 0:
            return (False, None, 0.0)
        
        # Try different validation methods based on plate type
        if plate_type == 'motorcycle':
            is_valid, normalized = LicensePlateValidator.validate_motorcycle(text)
            confidence = 0.85 if is_valid else 0.3
        elif plate_type == 'commercial':
            is_valid, normalized = LicensePlateValidator.validate_commercial(text)
            confidence = 0.9 if is_valid else 0.4
        elif plate_type == 'graphic':
            is_valid, normalized = LicensePlateValidator.validate_graphic(text)
            confidence = 0.85 if is_valid else 0.4
        else:
            # Auto-detect format
            new_format = LicensePlateValidator.validate_new_format(text)
            
            if new_format[0]:
                is_valid, normalized = new_format
                confidence = 0.92
            else:
                old_format = LicensePlateValidator.validate_old_format(text)
                
                if old_format[0]:
                    is_valid, normalized = old_format
                    confidence = 0.88
                else:
                    # Try motorcycle as fallback
                    moto = LicensePlateValidator.validate_motorcycle(text)
                    
                    if moto[0]:
                        is_valid, normalized = moto
                        confidence = 0.75
                    else:
                        return (False, None, 0.2)
        
        return (is_valid, normalized, confidence)

    @staticmethod
    def correct_common_errors(text: str) -> Optional[str]:
        """
        Apply common OCR error corrections.
        """
        if not text:
            return None
            
        text = LicensePlateValidator.normalize_text(text)
        
        # 1. SIMPLE HYPHEN RECOVERY (Run these FIRST before destructive artifact scaling!)
        # Add hyphens to old format plates without them: 'กข 1234' -> 'กข-1234'
        old_no_hyphen_pattern = r'(?<![ก-ฮ0-9])([ก-ฮ]{2})\s*([1-9]\d{1,3})(?![ก-ฮ0-9])'
        match = re.search(old_no_hyphen_pattern, text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        
        # Add hyphens to new format plates without them: '1กข 1234' -> '1-กข-1234'
        new_no_hyphen_pattern = r'(?<![ก-ฮ0-9])([1-9])\s*([ก-ฮ]{2})\s*([1-9]\d{1,3})(?![ก-ฮ0-9])'
        match = re.search(new_no_hyphen_pattern, text)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            
        # 2. STRIP HYPHENS — OCR often inserts fake hyphens mid-plate (e.g. 'SU0-1292')
        # Remove all hyphens/dashes so the artifact patterns can match the full string cleanly.
        stripped = re.sub(r'[\-–—]', '', text)

        # 3. CONSERVATIVE ARTIFACT RECOVERY (on hyphen-stripped text)
        # NOTE: every pattern here REQUIRES the two consonant slots to already be real
        # Thai consonants ([ก-ฮ]{2}). We deliberately do NOT fabricate consonants from
        # digits/latin glyphs — when the OCR reads a consonant slot as a number it means
        # the model genuinely failed, and inventing a letter produced plausible-looking
        # but wrong plates (e.g. '133327' -> '1-รร-327'). Better to return None and let
        # the caller drop it.
        error_patterns = [
            # Fix leading 'เ' misread as '4'
            (r'(?<![A-Za-zก-ฮ0-9])เ\s*([ก-ฮ]{2})\s*([1-9]\d{1,3})(?![A-Za-zก-ฮ0-9])',
             lambda m: f"4-{m.group(1)}-{m.group(2)}"),
            # Fix '8' read as 'B' (between two real consonants and the number block)
            (r'(?<![A-Za-zก-ฮ0-9])([ก-ฮ]{2})\s*B\s*([1-9]\d{1,3})(?![A-Za-zก-ฮ0-9])',
             lambda m: f"{m.group(1)}-8-{m.group(2)}"),
            # Fix '1' read as 'l' or 'I'
            (r'(?<![A-Za-zก-ฮ0-9])[lI|]\s*([ก-ฮ]{2})\s*([1-9]\d{1,3})(?![A-Za-zก-ฮ0-9])',
             lambda m: f"1-{m.group(1)}-{m.group(2)}"),
        ]

        for pattern, correction in error_patterns:
            match = re.search(pattern, stripped)
            if match:
                return correction(match)

        return None
    
    @staticmethod
    def get_plate_type(text: str) -> Optional[str]:
        """Determine plate type from text."""
        old_format = LicensePlateValidator.validate_old_format(text)
        new_format = LicensePlateValidator.validate_new_format(text)
        moto = LicensePlateValidator.validate_motorcycle(text)
        
        if new_format[0]:
            return 'standard'
        elif old_format[0]:
            return 'standard'
        elif moto[0]:
            return 'motorcycle'
        
        return None
