# License Plate Normalization Specification

Target Standard: Vietnamese Ministry of Public Security Circular 24/2023/TT-BCA  
Reference Module: `src/eiu_lpr/validator.py`  

## 1. Normalization Sequence

The raw optical character recognition (OCR) text is normalized through the following sequence:

1. **Uppercase Transformation**: All alphabetical characters are converted to standard uppercase.
2. **Vietnamese Diacritic Canonicalization**: The Vietnamese character `Đ` (`\u0110`, often predicted by OCR models on plates with mounting rivets or horizontal scratches across the letter `D`) is mapped to standard ASCII `D`. Raw evidence is preserved in `raw_plate` and `character_details`.
3. **Non-Alphanumeric Removal**: All spaces, hyphens (`-`), periods (`.`), rivets, underscores (`_`), and newline characters (`\n`) are removed.
4. **Line Order Preservation**: Dual-line motorcycle plates are joined sequentially from top line to bottom line.
5. **Character Fidelity Note**: Outside of standardizing `Đ` to canonical `D`, no ambiguous character substitution is made in the normalization function (such as swapping between letter 'O' and digit '0').

### Rationale
Automated character swapping can corrupt valid readings or hide visual ambiguity. By returning the exact recognized characters alongside per-character softmax probability arrays, downstream systems have full visual evidence to evaluate ambiguous characters against vehicle records. Meanwhile, canonicalizing `Đ` to `D` ensures that plates with rivet noise or OCR diacritic bias match standard vehicle registration databases and Circular 24/2023 syntax patterns without destroying the series letter.

## 2. Regulatory Syntax Patterns

Normalized text is validated against Circular 24/2023 patterns:

- **Automobiles**: Pattern `^[1-9][0-9][A-Z][0-9]{4,6}$` (e.g. `30E69770`, `51G74372`).
- **Motorcycles**: Pattern `^[1-9][0-9]([A-Z][0-9]|[A-Z]{2})[0-9]{4,6}$` (e.g. `59D242069`, `59VA24003`).

## 3. Output Schema & Scenario Examples

### A. Valid Single Plate
```json
{
  "schema_version": "1.0",
  "request_id": "req-001",
  "status": "OK",
  "candidates": [
    {
      "plate": "59D242069",
      "raw_plate": "59-D2 420.69",
      "confidence": 0.9935,
      "detection_confidence": 0.8312,
      "bbox": [212, 258, 284, 323],
      "is_valid": true,
      "ocr_min_confidence": 0.9719,
      "character_details": [
        {"char": "5", "confidence": 0.9998},
        {"char": "9", "confidence": 0.9996},
        {"char": "D", "confidence": 0.9997},
        {"char": "2", "confidence": 0.9946},
        {"char": "4", "confidence": 0.9994},
        {"char": "2", "confidence": 0.9996},
        {"char": "0", "confidence": 0.9719},
        {"char": "6", "confidence": 0.9989},
        {"char": "9", "confidence": 0.9978}
      ],
      "region": "Vietnam",
      "region_prob": 1.0,
      "pass_level": "enhanced_crop"
    }
  ],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 68.45
}
```

### B. Multiple Plates in Single Frame
```json
{
  "schema_version": "1.0",
  "request_id": "req-002",
  "status": "OK",
  "candidates": [
    {
      "plate": "30A69770",
      "raw_plate": "30A69770",
      "confidence": 0.9969,
      "detection_confidence": 0.8521,
      "bbox": [178, 234, 242, 266],
      "is_valid": true
    },
    {
      "plate": "51G74372",
      "raw_plate": "51G-743.72",
      "confidence": 0.9882,
      "detection_confidence": 0.7944,
      "bbox": [450, 210, 520, 248],
      "is_valid": true
    }
  ],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 94.12
}
```

### C. No Plate Detected (Clean Empty Scene)
```json
{
  "schema_version": "1.0",
  "request_id": "req-003",
  "status": "OK",
  "candidates": [],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 42.10
}
```

### D. Low Confidence or Invalid Regulatory Syntax
When a box is detected but fails Vietnamese regulatory regex (e.g. partial plate or roadside advertisement):
```json
{
  "schema_version": "1.0",
  "request_id": "req-004",
  "status": "OK",
  "candidates": [
    {
      "plate": "99ABC",
      "raw_plate": "99-ABC",
      "confidence": 0.4120,
      "detection_confidence": 0.2850,
      "bbox": [105, 340, 180, 390],
      "is_valid": false,
      "ocr_min_confidence": 0.2105,
      "character_details": [
        {"char": "9", "confidence": 0.6500},
        {"char": "9", "confidence": 0.5210},
        {"char": "A", "confidence": 0.3800},
        {"char": "B", "confidence": 0.2980},
        {"char": "C", "confidence": 0.2105}
      ],
      "region": "Unknown",
      "region_prob": 0.32,
      "pass_level": "base_640px"
    }
  ],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 55.30
}
```

### E. Score Conventions and Missing Score Values
- **Coordinate System**: Bounding box coordinates `[x1, y1, x2, y2]` are integer or float pixel coordinates in the original unpadded input image space, where `(0, 0)` is top-left.
- **Score Ranges**: All confidence metrics (`confidence`, `detection_confidence`, `ocr_min_confidence`, and per-character softmax) strictly span `[0.0, 1.0]`.
- **Missing Score Representation**: If an empty crop is evaluated or OCR cannot resolve characters, `confidence` and `detection_confidence` default to `0.0`, `character_details` defaults to `[]`, and `plate` defaults to `""` (empty string, never `null`).
- **Runtime Exceptions**: If input data is malformed (e.g. 2D array, non-uint8, or null pointer), the engine raises a typed [`InvalidInputError`](/src/eiu_lpr/exceptions.py#L10) rather than emitting a silent empty result.
