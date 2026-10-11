import os
import sys
import uuid
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import modular LPR engine
from lpr_core.validator import VietnamPlateValidator
from lpr_engine import YOLOv8Detector, PaddleOCRAdapter, LPRPipeline

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
# Primary detector: fine-tuned license plate weights (Models_yolo/best.pt)
model_path = "Models_yolo/best.pt" if os.path.exists("Models_yolo/best.pt") else "Models_yolo/yolov8n.pt"
primary_detector = YOLOv8Detector(model_path, target_classes=['plate', 'license_plate'], confidence_threshold=0.10) # only detect license plates/plates

# Fallback detector: base YOLO model detecting vehicle bodies (car, truck, bus, motorcycle)
fallback_detector = YOLOv8Detector("Models_yolo/yolov8n.pt", target_classes=['car', 'motorcycle', 'bus', 'truck'], confidence_threshold=0.35)

ocr = PaddleOCRAdapter(lang='en')
pipeline = LPRPipeline(
    detector=primary_detector, 
    ocr=ocr, 
    fallback_detector=fallback_detector
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


# Response schema for RFID gate event
class LPRTransactionResponse(BaseModel):
    event_id: str
    rfid_card_id: str
    status: str
    timestamp: str
    license_plate_text: str
    confidence: float
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
    
    # Hard rule: if no bounding boxes are detected, raise error immediately to avoid false positive validation
    if not pipeline_res['detections']:
        raise HTTPException(
            status_code=422,
            detail="No vehicle or license plate bounding box detected in snapshot image."
        )
    
    # Extract OCR results and enforce official Vietnamese syntax validation
    ocr_text = "NONE_RECOGNIZED"
    ocr_conf = 0.0
    is_valid_format = False
    
    if pipeline_res['ocr_results']:
        results = pipeline_res['ocr_results']
        
        if len(results) == 1:
            # Standard 1-line plate
            ocr_text = results[0]['validated_text']
            ocr_conf = results[0]['confidence']
            is_valid_format = results[0]['is_valid']
        else:
            # 2-line plate: join lines in reading order (top row + bottom row)
            line_texts = [r['validated_text'] for r in results if r['validated_text']]
            ocr_text = "-".join(line_texts)
            ocr_conf = sum(r['confidence'] for r in results) / len(results)
            
            # Re-validate the joined string against Vietnamese regex
            final_validation = VietnamPlateValidator.clean_and_correct(ocr_text)
            ocr_text = final_validation['text']
            is_valid_format = final_validation['is_valid']
        
    total_latency = (time.time() - start_time) * 1000
    
    # Strict gate security rule:
    # Auto-reject any plate that does not match official Vietnamese regex format
    # The barrier opens ONLY when syntax is strictly valid AND confidence >= 0.50
    gate_status = "GATE_OPEN" if (is_valid_format and ocr_conf >= 0.50) else "ACCESS_DENIED"
    
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


# Local development server launch
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI backend on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
