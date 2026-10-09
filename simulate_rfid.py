import sys
import time
import argparse
import requests
import json

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BACKEND_URL = "http://127.0.0.1:8000/api/v1/rfid-tap"

def simulate_student_card_tap(rfid_card_id: str = "STU_10492", image_source: str = "car_test_2.jpg"):
    print("=" * 60)
    print(f"SIMULATING STUDENT RFID CARD TAP")
    print(f"Card ID: '{rfid_card_id}' | Image: '{image_source}'")
    print("=" * 60)
    
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
            print("\n✅ GATE TRANSACTION SUCCESSFUL!")
            print(f"  * Transaction Event ID : {data['event_id']}")
            print(f"  * Student RFID Card ID : {data['rfid_card_id']}")
            print(f"  * Barrier Gate Status  : {data['status']}")
            print(f"  * Recognized Plate    : '{data['license_plate_text']}' (Conf: {data['confidence']*100:.1f}%)")
            print(f"  * Detection Latency    : {data['detection_latency_ms']} ms")
            print(f"  * OCR Latency          : {data['ocr_latency_ms']} ms")
            print(f"  * Total HTTP Roundtrip : {round_trip_ms:.2f} ms")
            print(f"  * BBox Visual Output   : {data['annotated_image_path']}")
        else:
            print(f"❌ Server Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR: Could not connect to backend server at http://127.0.0.1:8000.")
        print("   Please start the backend server first in another terminal with:")
        print("   .\\venv_models\\Scripts\\python.exe main.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Student RFID Card Tap")
    parser.add_argument("--card", default="STU_908412", help="Student RFID card ID")
    parser.add_argument("image", nargs="?", default="car_test_2.jpg", help="Path to input vehicle image (default: car_test_2.jpg)")
    args = parser.parse_args()
    
    simulate_student_card_tap(rfid_card_id=args.card, image_source=args.image)
