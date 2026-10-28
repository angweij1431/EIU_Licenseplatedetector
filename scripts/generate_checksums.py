# Utility script to generate SHA-256 checksums for models/ release assets
import hashlib
import os

MODELS_DIR = "models"
CHECKSUM_FILE = os.path.join(MODELS_DIR, "checksums.sha256")
TRACKED_FILES = [
    "detector.onnx",
    "cct_xs.onnx",
    "cct_xs_plate_config.yaml",
]

def calculate_sha256(filepath: str) -> str:
    # Reads file in 64 KB chunks to handle large binary model files efficiently
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    print("=" * 60)
    print("Generating SHA-256 Checksums for Release Models")
    print("=" * 60)

    lines = []
    for filename in TRACKED_FILES:
        filepath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[ERROR] Missing file: {filepath}")
            continue

        file_hash = calculate_sha256(filepath)
        # Format matching standard Linux sha256sum: '<hash>  <filename>'
        lines.append(f"{file_hash}  {filename}\n")
        print(f"  {filename:<26}: {file_hash}")

    with open(CHECKSUM_FILE, "w", encoding="utf-8") as out:
        out.writelines(lines)

    print("=" * 60)
    print(f"[OK] Saved checksum manifest to {CHECKSUM_FILE}\n")

if __name__ == "__main__":
    main()

