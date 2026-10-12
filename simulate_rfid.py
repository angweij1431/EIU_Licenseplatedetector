import sys
import time
import argparse
import requests
import json

# Ensure Windows console handles UTF-8 characters without encoding crashes
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Backend API Endpoint URL
BACKEND_URL = "http://127.0.0.1:8000/api/v1/rfid-tap"

# Simulates physical student RFID card reader triggering gate transaction
def simulate_student_card_tap(rfid_card_id: str = "STU_10492", image_source: str = "test_images/car_test_3.jpg"):
    print(f"Simulating Student RFID Card Tap: Card '{rfid_card_id}', Image '{image_source}'")
    
    # Request Payload sent to FastAPI REST backend
    payload = {
        "rfid_card_id": rfid_card_id,
        "camera_id": "GATE_01_CAM",
        "image_source": image_source
    }
    
    start_time = time.time()
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=30)
        round_trip_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            print("Gate transaction completed successfully:")
            print(f"  Transaction Event ID : {data['event_id']}")
            print(f"  RFID Card ID         : {data['rfid_card_id']}")
            print(f"  Barrier Gate Status  : {data['status']}")
            print(f"  Try Level            : {data.get('try_level', 'N/A')}")
            print(f"  Resolution           : {data.get('resolution', 'N/A')}")
            print(f"  Format Match         : {data.get('format_similarity', 0.0)*100:.1f}%")
            print(f"  Audit Flagged        : {data.get('audit_flag', False)}")
            if data.get('flag_reason'):
                print(f"  Flag Reason          : {data['flag_reason']}")
            print(f"  Recognized Plate     : '{data['license_plate_text']}' (Confidence: {data['confidence']*100:.1f}%)")
            print(f"  Detection Latency    : {data['detection_latency_ms']:.2f} ms")
            print(f"  OCR Latency          : {data['ocr_latency_ms']:.2f} ms")
            print(f"  Total Roundtrip      : {round_trip_ms:.2f} ms")
            print(f"  Annotated Image Path : {data['annotated_image_path']}")
        else:
            print(f"Server Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Connection Error: Could not connect to backend server at http://127.0.0.1:8000.")
        print("Please start the backend server first in another terminal with:")
        print(".\\venv_models\\Scripts\\python.exe main.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Student RFID Card Tap")
    parser.add_argument("--card", default="STU_908412", help="Student RFID card ID")
    parser.add_argument("image", nargs="?", default="test_images/car_test_3.jpg", help="Path to input vehicle image (default: test_images/car_test_3.jpg)")
    args = parser.parse_args()
    
    simulate_student_card_tap(rfid_card_id=args.card, image_source=args.image)
