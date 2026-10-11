import re

# Vietnamese License Plate Syntax Validator and Error Corrector
class VietnamPlateValidator:
    # Official Vietnamese License Plate RegEx Syntax Patterns
    # Automobiles (e.g. 30E-697.70 or 29M-6840): 2 digits + 1 letter + dash + 3-5 digits (optional .2 digits)
    AUTO_PATTERN = re.compile(r'^[1-9][0-9][A-Z]-[0-9]{3,5}(\.[0-9]{2})?$')
    
    # Motorcycles (e.g. 63-B1 123.45 or 29-H1 999.99): 2 digits + dash + 1 letter + 1 digit + space + 3-5 digits
    MOTO_PATTERN = re.compile(r'^[1-9][0-9]-[A-Z][0-9]\s?[0-9]{3,5}(\.[0-9]{2})?$')

    @staticmethod
    def clean_and_correct(raw_text: str) -> dict:
        if not raw_text:
            return {'text': '', 'raw_text': '', 'is_valid': False}
            
        # Step 1: Strip unwanted symbols and convert to uppercase
        cleaned = re.sub(r'[^A-Z0-9\.\-]', '', raw_text.upper())
        
        # Step 2: Strip leading/trailing dots and dashes (caused by mounting screw rivets)
        cleaned = re.sub(r'^[.\-]+|[.\-]+$', '', cleaned)
        
        # Step 3: Remove internal noise dots (e.g. '29M.-6840' -> '29M-6840')
        # Only preserve dot if it is a legitimate Vietnamese decimal separator (e.g. .70 in 697.70)
        parts = cleaned.split('-')
        cleaned_parts = []
        for p in parts:
            # If dot is at the end or start of a subpart, strip it
            p_clean = re.sub(r'^[.\-]+|[.\-]+$', '', p)
            cleaned_parts.append(p_clean)
        cleaned = "-".join(cleaned_parts)
        
        # Step 4: Position-Based Character Correction
        # First 2 characters are ALWAYS numeric province codes (11-99)
        char_list = list(cleaned)
        if len(char_list) >= 2:
            num_map = {
                'O': '0', 'Q': '0', 'D': '0', 
                'I': '1', 'L': '1', 
                'Z': '2', 
                'B': '8', 
                'S': '5'
            }
            if char_list[0] in num_map:
                char_list[0] = num_map[char_list[0]]
            if char_list[1] in num_map:
                char_list[1] = num_map[char_list[1]]
                
        corrected_text = "".join(char_list)
        
        # Step 5: Validate against official Vietnamese syntax patterns
        is_valid = bool(
            VietnamPlateValidator.AUTO_PATTERN.match(corrected_text) or 
            VietnamPlateValidator.MOTO_PATTERN.match(corrected_text)
        )
                        
        return {
            'text': corrected_text,
            'raw_text': raw_text,
            'is_valid': is_valid
        }
