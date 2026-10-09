import re

class VietnamPlateValidator:
    """
    Reinforces OCR text output against character confusion caused by dirt or smudges.
    Common confusion matrix:
    - '0' vs 'O' / 'D'
    - '1' vs 'I' / 'L'
    - '8' vs 'B'
    - '5' vs 'S'
    """
    
    # Official Vietnamese License Plate RegEx Syntax Patterns
    # Automobiles: e.g. 30E-697.70 or 51G-123.45
    AUTO_PATTERN = re.compile(r'^[1-9][0-9][A-Z]-[0-9]{3,5}(\.[0-9]{2})?$')
    
    # Motorcycles: e.g. 63-B1 123.45 or 29-H1 999.99
    MOTO_PATTERN = re.compile(r'^[1-9][0-9]-[A-Z][0-9]\s?[0-9]{3,5}(\.[0-9]{2})?$')

    @staticmethod
    def clean_and_correct(raw_text: str) -> dict:
        """
        Cleans raw OCR text and applies position-aware character substitution rules.
        """
        if not raw_text:
            return {'text': '', 'is_valid': False, 'confidence_boost': 0.0}
            
        # Uppercase and remove random symbols
        cleaned = re.sub(r'[^A-Z0-9\.\-]', '', raw_text.upper())
        
        # Position-based character correction:
        # First 2 positions are ALWAYS numbers (Province code 11 to 99)
        char_list = list(cleaned)
        if len(char_list) >= 2:
            num_map = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8', 'S': '5'}
            if char_list[0] in num_map:
                char_list[0] = num_map[char_list[0]]
            if char_list[1] in num_map:
                char_list[1] = num_map[char_list[1]]
                
        corrected_text = "".join(char_list)
        
        # Validate against official Vietnamese syntax patterns
        is_valid = bool(VietnamPlateValidator.AUTO_PATTERN.match(corrected_text) or 
                        VietnamPlateValidator.MOTO_PATTERN.match(corrected_text))
                        
        return {
            'text': corrected_text,
            'raw_text': raw_text,
            'is_valid': is_valid
        }

