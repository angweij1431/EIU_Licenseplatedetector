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
        
        # Step 4: Validate directly against official Vietnamese syntax patterns (No character replacement)
        is_valid = bool(
            VietnamPlateValidator.AUTO_PATTERN.match(cleaned) or 
            VietnamPlateValidator.MOTO_PATTERN.match(cleaned)
        )
        
        format_sim = VietnamPlateValidator.check_format_similarity(cleaned)
                        
        return {
            'text': cleaned,
            'raw_text': raw_text,
            'is_valid': is_valid,
            'format_similarity': format_sim['similarity'],
            'is_format_similar': format_sim['is_similar'],
            'mask': format_sim['mask']
        }

    # Calculates Levenshtein string edit distance
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        s1 = s1.upper().replace("-", "").replace(".", "").strip()
        s2 = s2.upper().replace("-", "").replace(".", "").strip()
        if len(s1) < len(s2):
            return VietnamPlateValidator.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]

    # Evaluates structural format similarity against Vietnamese license plate templates
    @staticmethod
    def check_format_similarity(text: str, min_similarity: float = 0.75) -> dict:
        if not text:
            return {'is_similar': False, 'similarity': 0.0, 'mask': '', 'closest_template': '', 'mismatches': 99}
            
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not cleaned:
            return {'is_similar': False, 'similarity': 0.0, 'mask': '', 'closest_template': '', 'mismatches': 99}
            
        # Build character mask: 'D' for digit, 'L' for letter
        mask = "".join(['D' if c.isdigit() else 'L' for c in cleaned])
        
        best_similarity = 0.0
        best_template = ""
        min_mismatches = 99
        
        # Standard Vietnamese plate templates (without symbols)
        templates = ["DDLDDDD", "DDLDDDDD", "DDLDDDDDD"]
        for t in templates:
            dist = VietnamPlateValidator.levenshtein_distance(mask, t)
            max_len = max(len(mask), len(t))
            similarity = max(0.0, (max_len - dist) / max_len)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_template = t
                min_mismatches = dist
                
        is_similar = (best_similarity >= min_similarity)
        
        return {
            'is_similar': is_similar,
            'similarity': round(best_similarity, 3),
            'mask': mask,
            'closest_template': best_template,
            'mismatches': min_mismatches
        }

    # Compares OCR extracted text against a registered student license plate
    @staticmethod
    def match_plate(ocr_text: str, registered_plate: str, max_distance: int = 2) -> dict:
        if not ocr_text or not registered_plate:
            return {'match': False, 'distance': 999, 'match_type': 'none'}
        dist = VietnamPlateValidator.levenshtein_distance(ocr_text, registered_plate)
        if dist == 0:
            return {'match': True, 'distance': 0, 'match_type': 'exact'}
        elif dist <= max_distance:
            return {'match': True, 'distance': dist, 'match_type': 'fuzzy'}
        return {'match': False, 'distance': dist, 'match_type': 'mismatch'}
