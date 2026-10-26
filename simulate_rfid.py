import os
import sys
import time
import argparse
import requests
import json
import shutil

# Ensure Windows console handles UTF-8 characters without encoding crashes
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BACKEND_URL = "http://127.0.0.1:8000/api/v1/rfid-tap"

# Known test mapping to simulate authorized student cards (falls back to guest taps)
STUDENT_CARD_MAPPING = {
    "car_test_4.jpg": "STU_1001",   # 51G74372
    "bike_test_9.jpg": "STU_1002",  # 59D242069
    "bike_test_10.jpg": "STU_1003", # 43D199999
    "bike_test_11.jpg": "STU_1004", # 29H133979
    "bike_test_12.jpg": "STU_1005", # 61H76854
    "car_test_5.jpg": "STU_1006",   # 51G77777
    "car_test_2.jpg": "STU_1007",   # 30A69770
}

def send_tap_request(rfid_card_id: str, image_path: str):
    payload = {
        "rfid_card_id": rfid_card_id,
        "camera_id": "GATE_01_CAM",
        "image_source": image_path.replace("\\", "/")
    }
    t0 = time.time()
    response = requests.post(BACKEND_URL, json=payload, timeout=30)
    round_trip_ms = (time.time() - t0) * 1000
    
    if response.status_code == 200:
        data = response.json()
        data['round_trip_ms'] = round_trip_ms
        return data
    else:
        raise RuntimeError(f"Server Error {response.status_code}: {response.text}")

def run_simulation(target_path: str, output_dir: str = "Outputs/rfid_simulation"):
    # 1. Resolve image list
    if os.path.isdir(target_path):
        valid_exts = ('.jpg', '.jpeg', '.png')
        images = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.lower().endswith(valid_exts)]
        images.sort()
    elif os.path.isfile(target_path):
        images = [target_path]
    else:
        raise FileNotFoundError(f"Input path not found: {target_path}")

    # 2. Setup output folder structure
    passed_dir = os.path.join(output_dir, "passed")
    rejected_dir = os.path.join(output_dir, "rejected")
    flagged_dir = os.path.join(output_dir, "flagged")
    for d in [passed_dir, rejected_dir, flagged_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Starting RFID Gate Simulation on {len(images)} image(s)...")
    print(f"Output Directory: {output_dir}\n")

    transactions = []
    extracted_records = []

    for idx, img_path in enumerate(images, 1):
        filename = os.path.basename(img_path)
        card_id = STUDENT_CARD_MAPPING.get(filename, f"STU_{9000 + idx}")

        try:
            res = send_tap_request(card_id, img_path)
            status = res['status']
            plate = res['license_plate_text']
            latency = res['round_trip_ms']

            # Route annotated image to bucket
            src_annotated = res['annotated_image_path']
            if os.path.exists(src_annotated):
                dest_folder = passed_dir if status == "GATE_OPEN" else (rejected_dir if status == "ACCESS_DENIED" else flagged_dir)
                dest_file = os.path.join(dest_folder, os.path.basename(src_annotated))
                shutil.copy2(src_annotated, dest_file)

            transactions.append({
                "idx": idx,
                "image": filename,
                "card": card_id,
                "plate": plate,
                "status": status,
                "reason": res.get("flag_reason") or "Verified OK",
                "latency_ms": latency
            })

            extracted_records.append({
                "timestamp": res['timestamp'],
                "image": filename,
                "rfid_card_id": card_id,
                "extracted_plate": plate,
                "confidence": res['confidence'],
                "status": status,
                "try_level": res['try_level'],
                "resolution": res['resolution']
            })

            print(f"  [{idx:02d}/{len(images):02d}] {filename:<20} | {card_id} | {plate:<12} | {status:<18} ({latency:.1f}ms)")

        except requests.exceptions.ConnectionError:
            print("\nError: Cannot connect to FastAPI server at http://127.0.0.1:8000.")
            print("Please run 'python main.py' in a separate terminal first.")
            return

    # 3. Print ordered summary scorecard
    print("\n" + "=" * 90)
    print(f"{'#':<3} | {'Image':<20} | {'Card ID':<10} | {'Extracted Plate':<15} | {'Gate Status':<18} | {'Latency'}")
    print("=" * 90)
    for t in transactions:
        print(f"{t['idx']:<3} | {t['image']:<20} | {t['card']:<10} | {t['plate']:<15} | {t['status']:<18} | {t['latency_ms']:.1f}ms")
    print("=" * 90)

    # 4. Save extracted license plates list
    report_file = os.path.join(output_dir, "extracted_plates.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(extracted_records, f, indent=2, ensure_ascii=False)
    print(f"\nExtracted plates and transaction log saved to: {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Multi-Image RFID Gate Streaming")
    parser.add_argument("input_path", nargs="?", default="test_images", help="Image file or directory path (default: test_images)")
    parser.add_argument("--output_dir", default="Outputs/rfid_simulation", help="Target output folder for organized images")
    args = parser.parse_args()

    run_simulation(args.input_path, args.output_dir)