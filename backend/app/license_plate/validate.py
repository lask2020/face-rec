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
    
    # Characters that are actually used on Thai license plates (high-frequency consonants)
    PLATE_CONSONANTS = set('กขคฆงจฉชซญฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ')
    
    # Ranked candidate map: each OCR artifact maps to a LIST of Thai consonants
    # ordered by visual similarity AND frequency on Thai plates.
    # First match that is a valid plate consonant wins.
    SHAPE_CANDIDATES = {
        'n': ['ก', 'ณ'], 'N': ['ก', 'ณ'],
        'm': ['พ', 'ฟ'], 'M': ['พ', 'ฟ'],
        'w': ['พ', 'ฟ'], 'W': ['พ', 'ฟ'],
        'u': ['บ', 'ป'], 'U': ['บ', 'ป'],
        'o': ['ก', 'อ', 'ด'], 'O': ['ก', 'อ', 'ด'],  
        '0': ['ก', 'อ', 'ด'],  # 0 looks like ก (closed loop) more often than อ on plates!
        's': ['ร', 'ว'], 'S': ['ร', 'ว'],
        '5': ['ธ', 'ร'], '2': ['ว', 'ร'], '7': ['ว', 'ก'],
        '8': ['ข', 'ค'], '9': ['จ', 'ง'], '3': ['ร', 'ว'],
        'E': ['ย', 'ข'], 'e': ['ย', 'ข'],
        'c': ['ร', 'ว'], 'C': ['ร', 'ว'],
        'l': ['เ', 'ก'], 'I': ['เ', 'ก'],
        'd': ['ด', 'ค'], 'D': ['ด', 'ค'],
        'a': ['ค', 'ศ'], 'A': ['ค', 'ศ'],
        'b': ['ป', 'บ'], 'B': ['ป', 'บ'],
        'p': ['ม', 'น'], 'P': ['ม', 'น'],
        'v': ['บ', 'ป'], 'V': ['บ', 'ป'],
        'h': ['ห', 'น'], 'H': ['ห', 'น'],
        't': ['ท', 'ต'], 'T': ['ท', 'ต'],
    }

    @staticmethod
    def _map_to_thai(text: str) -> str:
        """
        Map OCR artifacts back to Thai consonants using ranked candidates.
        
        For a 2-char consonant slot, tries all permutations of candidate mappings
        and picks the first combo where BOTH characters are valid plate consonants.
        Falls back to first candidate if no perfect pair is found.
        """
        if len(text) == 2:
            # Get candidate lists for each character
            c1_candidates = LicensePlateValidator.SHAPE_CANDIDATES.get(text[0], [text[0]])
            c2_candidates = LicensePlateValidator.SHAPE_CANDIDATES.get(text[1], [text[1]])
            
            # If the character is already Thai, use it directly
            if text[0] in LicensePlateValidator.PLATE_CONSONANTS:
                c1_candidates = [text[0]]
            if text[1] in LicensePlateValidator.PLATE_CONSONANTS:
                c2_candidates = [text[1]]
            
            # Try all combos, prefer pairs where both are valid plate consonants
            for c1 in c1_candidates:
                for c2 in c2_candidates:
                    if (c1 in LicensePlateValidator.PLATE_CONSONANTS and 
                        c2 in LicensePlateValidator.PLATE_CONSONANTS):
                        return c1 + c2
            
            # Fallback: just use first candidate for each
            best_c1 = c1_candidates[0] if c1_candidates else text[0]
            best_c2 = c2_candidates[0] if c2_candidates else text[1]
            return best_c1 + best_c2
        
        # For single chars or longer strings, use simple first-candidate mapping
        result = []
        for c in text:
            candidates = LicensePlateValidator.SHAPE_CANDIDATES.get(c, [c])
            if c in LicensePlateValidator.PLATE_CONSONANTS:
                result.append(c)
            else:
                result.append(candidates[0] if candidates else c)
        return "".join(result)

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
        # Remove all hyphens/dashes so the aggressive patterns can match the full string cleanly.
        stripped = re.sub(r'[\-–—]', '', text)
        
        # 3. DESTRUCTIVE ARTIFACT RECOVERY (on hyphen-stripped text)
        error_patterns = [
            # Fix leading 'เ' misread as '4'
            (r'(?<![A-Za-zก-ฮ0-9])เ\s*([ก-ฮ]{2})\s*([1-9]\d{1,3})(?![A-Za-zก-ฮ0-9])', 
             lambda m: f"4-{m.group(1)}-{m.group(2)}"),
            # Fix '8' read as 'B'
            (r'(?<![A-Za-zก-ฮ0-9])([ก-ฮ]{2})\s*B\s*([1-9]\d{1,3})(?![A-Za-zก-ฮ0-9])', 
             lambda m: f"{m.group(1)}-8-{m.group(2)}"),
            # Fix '1' read as 'l' or 'I'
            (r'(?<![A-Za-zก-ฮ0-9])[lI|]\s*([ก-ฮ]{2})\s*([1-9]\d{1,3})(?![A-Za-zก-ฮ0-9])', 
             lambda m: f"1-{m.group(1)}-{m.group(2)}"),
            
            # --- AGGRESSIVE OCR ARTIFACT RECOVERY ---
            # New format: [1 digit] [2 ANY chars] [2-4 digits] — map middle 2 to Thai
            (r'(?<![A-Za-zก-ฮ0-9])([1-9])\s*([A-Za-zก-ฮ0-9]{2})\s*([1-9]\d{1,3})(?![A-Za-zก-ฮ0-9])', 
             lambda m: f"{m.group(1)}-{LicensePlateValidator._map_to_thai(m.group(2))}-{m.group(3)}"),
            # Old format: [2 ANY chars] [2-4 digits]
            (r'(?<![A-Za-zก-ฮ0-9])([A-Za-zก-ฮ0-9]{2})\s*([1-9]\d{1,3})(?![A-Za-zก-ฮ0-9])', 
             lambda m: f"{LicensePlateValidator._map_to_thai(m.group(1))}-{m.group(2)}"),
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
