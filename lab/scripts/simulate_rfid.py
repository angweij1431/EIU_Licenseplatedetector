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

            # Extract box and OCR metrics
            box_coords = [int(round(x)) for x in res.get("box_coordinates", [])]
            box_conf = res.get("box_confidence", 0.0)
            ocr_avg_conf = res.get("confidence", 0.0)
            ocr_min_conf = res.get("ocr_min_confidence", 0.0)
            char_details = res.get("character_details", [])
            
            # Compact per-character string, e.g. "5(100%) 9(100%) D(99%) 2(100%)"
            char_breakdown = " ".join(f"{c['char']}({c['confidence']*100:.0f}%)" for c in char_details) if char_details else "N/A"

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
                "box_str": f"{box_conf*100:.1f}% {box_coords}" if box_coords else "NO_BOX",
                "ocr_str": f"{ocr_avg_conf*100:.1f}% (min: {ocr_min_conf*100:.1f}%)" if plate != "NONE_RECOGNIZED" else "N/A",
                "char_breakdown": char_breakdown,
                "status": status,
                "latency_ms": latency
            })

            extracted_records.append({
                "timestamp": res['timestamp'],
                "image": filename,
                "rfid_card_id": card_id,
                "status": status,
                "try_level": res['try_level'],
                "resolution": res['resolution'],
                "box": {
                    "coordinates": box_coords,
                    "confidence": round(box_conf, 4)
                },
                "ocr": {
                    "extracted_plate": plate,
                    "average_confidence": round(ocr_avg_conf, 4),
                    "min_confidence": round(ocr_min_conf, 4),
                    "character_details": char_details
                }
            })

            # Live terminal print for each vehicle
            box_disp = f"{box_conf*100:.1f}% {box_coords}" if box_coords else "NO_BOX"
            ocr_disp = f"{ocr_avg_conf*100:.1f}% (min: {ocr_min_conf*100:.1f}%)" if plate != "NONE_RECOGNIZED" else "N/A"
            print(f"  [{idx:02d}/{len(images):02d}] {filename:<18} | Plate: {plate:<11} | Box: {box_disp:<26} | OCR: {ocr_disp:<21} | {status} ({latency:.0f}ms)")
            if char_details:
                print(f"         └─ Chars: {char_breakdown}")

        except requests.exceptions.ConnectionError:
            print("\nError: Cannot connect to FastAPI server at http://127.0.0.1:8000.")
            print("Please run 'python main.py' in a separate terminal first.")
            return

    # 3. Print ordered summary scorecard
    # 3. Print ordered summary scorecard with auto-fitted column widths
    if transactions:
        for t in transactions:
            t["idx_str"] = str(t["idx"])
            t["latency_str"] = f"{t['latency_ms']:.1f}ms"

        headers = {
            "idx_str": "#",
            "image": "Image",
            "plate": "Plate",
            "box_str": "Box (Conf / Coords)",
            "ocr_str": "OCR (Avg / Min)",
            "status": "Gate Status",
            "latency_str": "Latency"
        }

        # Calculate max width for each column dynamically + 2 spaces (~5% padding)
        col_w = {}
        for key, h_text in headers.items():
            max_data_len = max(len(str(t.get(key, ""))) for t in transactions)
            max_len = max(len(h_text), max_data_len)
            col_w[key] = max_len + 2

        # Format header and divider line
        header_line = " | ".join(f"{headers[k]:<{col_w[k]}}" for k in headers)
        divider = "=" * len(header_line)

        print("\n" + divider)
        print(header_line)
        print(divider)
        for t in transactions:
            row_line = " | ".join(f"{str(t.get(k, '')):<{col_w[k]}}" for k in headers)
            print(row_line)
        print(divider)

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