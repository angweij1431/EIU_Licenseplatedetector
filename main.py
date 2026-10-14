import os
import sys
import uuid
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import modular LPR engine
from lpr_core.validator import VietnamPlateValidator
from lpr_engine import YOLODetector, PaddleOCRAdapter, LPRPipeline

# Import-level fallback for Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure output directory exists
OUTPUTS_DIR = "Outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Initialize models once during startup to avoid reloading per request
# Primary detector: fine-tuned license plate ONNX model (Models_yolo/best.onnx)
model_path = "Models_yolo/best.onnx"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"ONNX model not found: {model_path}. Please place exported ONNX weights in Models_yolo/.")
primary_detector = YOLODetector(model_path, target_classes=['plate', 'license_plate'], confidence_threshold=0.25)

ocr = PaddleOCRAdapter(lang='en')
pipeline = LPRPipeline(
    detector=primary_detector, 
    ocr=ocr
)

# Initialize FastAPI application
app = FastAPI(
    title="Vietnam Vehicle LPR and RFID Gate API",
    description="Backend service pairing RFID student card taps with LPR camera snapshots",
    version="1.0.0"
)


# Request schema for RFID tap
class RFIDTapRequest(BaseModel):
    rfid_card_id: str
    camera_id: str = "GATE_01_CAM"
    image_source: str = "car_test_3.jpg"


from typing import Optional

# Response schema for RFID gate event
class LPRTransactionResponse(BaseModel):
    event_id: str
    rfid_card_id: str
    status: str
    mode: str
    try_level: str
    resolution: str
    audit_flag: bool
    flag_reason: Optional[str] = None
    timestamp: str
    license_plate_text: str
    confidence: float
    format_similarity: float
    detection_latency_ms: float
    ocr_latency_ms: float
    total_latency_ms: float
    annotated_image_path: str


# Health check endpoint
@app.get("/")
def health_check():
    return {
        "status": "YES ONLINE",
        "service": "LPR RFID Gate Backend",
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ')
    }


# Primary RFID tap transaction endpoint
@app.post("/api/v1/rfid-tap", response_model=LPRTransactionResponse)
def handle_rfid_card_tap(request: RFIDTapRequest):
    start_time = time.time()
    
    # Generate unique transaction event ID
    event_id = f"evt_{uuid.uuid4().hex[:10]}"
    
    if not os.path.exists(request.image_source):
        raise HTTPException(
            status_code=404, 
            detail=f"Image source not found: {request.image_source}"
        )
        
    output_image_filename = f"{event_id}_annotated.jpg"
    print(f"RFID Tap received: Card '{request.rfid_card_id}', Event ID '{event_id}'")
    
    # Process snapshot through LPR pipeline with 20 percent bounding box padding
    try:
        pipeline_res = pipeline.process(
            image_path=request.image_source,
            save_annotated_name=output_image_filename,
            sort_by='front',
            primary_only=True,
            padding_percent=0.20
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {e}")
    
    # Check detections and OCR results
    has_detections = bool(pipeline_res['detections'])
    ocr_text = "NONE_RECOGNIZED"
    ocr_conf = 0.0
    is_valid_format = False
    format_similarity = 0.0
    is_format_similar = False
    
    if has_detections and pipeline_res['ocr_results']:
        results = pipeline_res['ocr_results']
        
        if len(results) == 1:
            # Standard 1-line plate
            ocr_text = results[0]['validated_text']
            ocr_conf = results[0]['confidence']
            is_valid_format = results[0]['is_valid']
            format_similarity = results[0].get('format_similarity', 0.0)
            is_format_similar = results[0].get('is_format_similar', False)
        else:
            # 2-line plate: join lines in reading order (top row + bottom row)
            line_texts = [r['validated_text'] for r in results if r['validated_text']]
            ocr_text = "-".join(line_texts)
            ocr_conf = sum(r['confidence'] for r in results) / len(results)
            
            # Re-validate the joined string against Vietnamese regex and format similarity
            final_val = VietnamPlateValidator.clean_and_correct(ocr_text)
            ocr_text = final_val['text']
            is_valid_format = final_val['is_valid']
            format_similarity = final_val.get('format_similarity', 0.0)
            is_format_similar = final_val.get('is_format_similar', False)
        
    total_latency = (time.time() - start_time) * 1000
    pipe_try_level = pipeline_res.get('try_level', '1st try')
    
    # Authorized card check (cards starting with STU_ represent registered student cards)
    is_authorized_card = request.rfid_card_id.startswith("STU_")
    
    if not is_authorized_card:
        gate_status = "ACCESS_DENIED"
        mode = "DENIED"
        try_level = "denied"
        audit_flag = False
        flag_reason = "Unauthorized RFID card."
        print(f"SECURITY ALERT: Event {event_id} denied - Unauthorized Card '{request.rfid_card_id}'.")
    else:
        # Check if strict syntax passed on 1st try or 2nd try
        if is_valid_format and ocr_conf >= 0.50:
            gate_status = "GATE_OPEN"
            mode = pipe_try_level
            try_level = pipe_try_level
            audit_flag = False
            flag_reason = None
        # 3rd try: Format similarity tolerance (e.g. 301-69730 has 87.5% format similarity with DDLDDDDD)
        elif is_format_similar and ocr_conf >= 0.50:
            gate_status = "GATE_OPEN"
            mode = "3rd try"
            try_level = "3rd try"
            audit_flag = False
            flag_reason = f"Verified via format similarity ({format_similarity*100:.1f}% template match)."
            print(f"FORMAT MATCH: Event {event_id} verified via 3rd try format similarity ({format_similarity*100:.1f}%).")
        # Fallback: Valid student card, but plate unverified or unreadable
        else:
            gate_status = "GATE_OPEN_FLAGGED"
            mode = "CARD_ONLY"
            try_level = "fallback"
            audit_flag = True
            flag_reason = "Unverified plate; opened on student card."
            print(f"AUDIT ALERT: Event {event_id} flagged - Card '{request.rfid_card_id}' without verified plate.")
    
    return LPRTransactionResponse(
        event_id=event_id,
        rfid_card_id=request.rfid_card_id,
        status=gate_status,
        mode=mode,
        try_level=try_level,
        resolution=pipeline_res.get('resolution', '640px'),
        audit_flag=audit_flag,
        flag_reason=flag_reason,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        license_plate_text=ocr_text,
        confidence=round(ocr_conf, 4),
        format_similarity=round(format_similarity, 3),
        detection_latency_ms=round(pipeline_res['detection_latency_ms'], 2),
        ocr_latency_ms=round(pipeline_res['ocr_latency_ms'], 2),
        total_latency_ms=round(total_latency, 2),
        annotated_image_path=pipeline_res['annotated_path']
    )


# Local development server launch
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI backend on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
