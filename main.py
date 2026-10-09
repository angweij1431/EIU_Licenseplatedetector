"""
=================================================================================
MOCK RFID & LPR BACKEND REST API SERVER (main.py)
=================================================================================

Educational Design Patterns Used:
1. REST API Architecture (FastAPI + Pydantic): Standard web API format used in production.
2. Event-Driven Trigger: Microcontroller/RFID sends POST request when card is tapped.
3. Event Pairing (UUID4): Generates unique transaction ID to pair student card with LPR crop.
4. Output Isolation: All visualization images are saved into the 'Outputs/' folder.
=================================================================================
"""

import os
import sys
import uuid
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our modular LPR Engine
from lpr_engine import YOLOv8Detector, PaddleOCRAdapter, LPRPipeline

# Ensure Windows UTF-8 output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure 'Outputs' directory exists
OUTPUTS_DIR = "Outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("[Server Startup] Initializing LPR Engine Models...")
# Initialize models ONCE during backend startup to keep per-tap latency low!
detector = YOLOv8Detector("Models_yolo/best.pt", target_classes=None, confidence_threshold=0.25)
ocr = PaddleOCRAdapter(lang='en')
pipeline = LPRPipeline(detector=detector, ocr=ocr)

# Initialize FastAPI app
app = FastAPI(
    title="Vietnam Vehicle LPR & RFID Gate API",
    description="Backend service pairing RFID student card taps with LPR camera snapshots.",
    version="1.0.0"
)


# =====================================================================
# PYDANTIC REQUEST & RESPONSE SCHEMAS
# =====================================================================
# WHY: Pydantic automatically validates JSON data types sent over HTTP.

class RFIDTapRequest(BaseModel):
    rfid_card_id: str                      # Student RFID card ID e.g., "STU_10492"
    camera_id: str = "GATE_01_CAM"         # Camera identifier
    image_source: str = "car_test_3.jpg"   # Snapshot image source for test simulation


class LPRTransactionResponse(BaseModel):
    event_id: str                          # Unique UUID transaction ID
    rfid_card_id: str                      # Student card ID
    status: str                            # "GATE_OPEN" or "ACCESS_DENIED"
    timestamp: str                         # ISO 8601 timestamp
    license_plate_text: str                # Extracted plate string e.g., "30E-697.70"
    confidence: float                      # Recognition confidence
    detection_latency_ms: float            # Detection latency in ms
    ocr_latency_ms: float                  # OCR latency in ms
    total_latency_ms: float                # Total pipeline latency in ms
    annotated_image_path: str              # Path to saved BBox image in Outputs/


# =====================================================================
# REST API ENDPOINTS
# =====================================================================

@app.get("/")
def health_check():
    """Health check endpoint to verify backend status."""
    return {"status": "ONLINE", "service": "LPR RFID Gate Backend", "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ')}


@app.post("/api/v1/rfid-tap", response_model=LPRTransactionResponse)
def handle_rfid_card_tap(request: RFIDTapRequest):
    """
    Simulates / handles a Student RFID Card Tap at the Barrier Gate.
    1. Generates unique transaction event_id (UUID).
    2. Triggers camera snapshot LPR processing.
    3. Saves visualization image inside 'Outputs/'.
    4. Pairs RFID student card with recognized plate string and returns response.
    """
    start_time = time.time()
    
    # 📚 LEARNING PATTERN: Generate unique UUID for pairing RFID event with LPR result
    event_id = f"evt_{uuid.uuid4().hex[:10]}"
    
    if not os.path.exists(request.image_source):
        raise HTTPException(status_code=404, detail=f"Camera snapshot image '{request.image_source}' not found.")
        
    output_image_filename = f"{event_id}_annotated.jpg"
    
    print(f"\n[HTTP Event] RFID Tap Received | Card: '{request.rfid_card_id}' | Event ID: '{event_id}'")
    
    # Process image through LPR pipeline (primary_only=True for speed)
    pipeline_res = pipeline.process(
        image_path=request.image_source,
        save_annotated_name=output_image_filename,
        sort_by='front',
        primary_only=True
    )
    
    # Extract recognized text string and confidence
    ocr_text = "NONE_RECOGNIZED"
    ocr_conf = 0.0
    if pipeline_res['ocr_results']:
        top_ocr = pipeline_res['ocr_results'][0]
        ocr_text = top_ocr['text']
        ocr_conf = top_ocr['confidence']
        
    total_latency = (time.time() - start_time) * 1000
    
    # Gate open status rule: Gate opens if plate recognized or confidence > 0.4
    gate_status = "GATE_OPEN" if (ocr_text != "NONE_RECOGNIZED" or ocr_conf >= 0.4) else "ACCESS_DENIED"
    
    return LPRTransactionResponse(
        event_id=event_id,
        rfid_card_id=request.rfid_card_id,
        status=gate_status,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        license_plate_text=ocr_text,
        confidence=round(ocr_conf, 4),
        detection_latency_ms=round(pipeline_res['detection_latency_ms'], 2),
        ocr_latency_ms=round(pipeline_res['ocr_latency_ms'], 2),
        total_latency_ms=round(total_latency, 2),
        annotated_image_path=pipeline_res['annotated_path']
    )


if __name__ == "__main__":
    import uvicorn
    print("\n[Server Launch] Starting FastAPI backend on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

