# Automated dataset download, fleet balancing, augmentation, and COCO conversion
import argparse
import glob
import json
import os
import random
import shutil
import sys
try:
    from PIL import Image
    import albumentations as A
    import cv2
    from roboflow import Roboflow
except ImportError:
    Image = None
    A = None
    cv2 = None
    Roboflow = None


def download_roboflow_datasets(api_key: str, download_dir: str = "raw_datasets"):
    os.makedirs(download_dir, exist_ok=True)
    rf = Roboflow(api_key=api_key)

    print("\n1/5 Downloading Car Fleet Dataset...")
    car_proj = rf.workspace("alphajack2006-gmail-com").project("vietnamese-car-license-plate-vlbzz")
    car_dataset = car_proj.version(1).download("yolov8", location=os.path.join(download_dir, "cars"))

    print("\n2/5 Downloading Motorcycle Fleet Dataset...")
    moto_proj = rf.workspace("alphajack2006-gmail-com").project("motorcycle-license-plate-sksvk")
    moto_dataset = moto_proj.version(1).download("yolov8", location=os.path.join(download_dir, "motorcycles"))

    print("\n3/5 Downloading Negative Background Dataset (Pavement/Roads)...")
    neg_proj = rf.workspace("alphajack2006-gmail-com").project("road-and-pavement-hc3qg-qsmcl")
    neg_dataset = neg_proj.version(1).download("yolov8", location=os.path.join(download_dir, "negatives"))

    return car_dataset.location, moto_dataset.location, neg_dataset.location


def merge_datasets(src_dir: str, prefix: str, unified_dir: str, max_train_samples: int = None, seed: int = 42):
    splits = ["train", "valid"]
    valid_exts = (".jpg", ".jpeg", ".png")

    for split in splits:
        img_src = os.path.join(src_dir, split, "images")
        lbl_src = os.path.join(src_dir, split, "labels")

        if not os.path.exists(img_src):
            alt_split = "valid"
            img_src = os.path.join(src_dir, alt_split, "images")
            lbl_src = os.path.join(src_dir, alt_split, "labels")

        all_images = [f for f in os.listdir(img_src) if f.lower().endswith(valid_exts)]

        if split == "train" and max_train_samples and len(all_images) > max_train_samples:
            random.seed(seed)
            selected_images = random.sample(all_images, max_train_samples)
            print(f"  [{prefix.upper()} - {split}] Downsampled from {len(all_images)} to {len(selected_images)} images")
        else:
            selected_images = all_images
            print(f"  [{prefix.upper()} - {split}] Importing all {len(selected_images)} images")

        for filename in selected_images:
            base_name, _ = os.path.splitext(filename)
            lbl_file = f"{base_name}.txt"

            src_img_path = os.path.join(img_src, filename)
            src_lbl_path = os.path.join(lbl_src, lbl_file)

            dest_img_path = os.path.join(unified_dir, split, "images", f"{prefix}_{filename}")
            dest_lbl_path = os.path.join(unified_dir, split, "labels", f"{prefix}_{lbl_file}")

            shutil.copy(src_img_path, dest_img_path)

            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                remapped = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0"  # Single class: 0 (plate)
                        remapped.append(" ".join(parts) + "\n")

                with open(dest_lbl_path, "w", encoding="utf-8") as f:
                    f.writelines(remapped)
            else:
                open(dest_lbl_path, "w", encoding="utf-8").close()


def generate_skewed_samples(unified_dir: str, prefix: str, sample_count: int = 140):
    img_dir = os.path.join(unified_dir, "train", "images")
    lbl_dir = os.path.join(unified_dir, "train", "labels")

    transform = A.Compose(
        [
            A.Perspective(scale=(0.02, 0.04), p=0.4, keep_size=True),
            A.Affine(scale=(0.95, 1.05), rotate=(-10, 10), translate_percent=(-0.03, 0.03), shear=(-4, 4), p=0.7),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )

    candidates = [f for f in os.listdir(img_dir) if f.startswith(prefix) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    selected = random.sample(candidates, min(sample_count, len(candidates)))

    generated = 0
    for fname in selected:
        stem, _ = os.path.splitext(fname)
        lbl_file = f"{stem}.txt"
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, lbl_file)

        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        bboxes = []
        class_labels = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

        if not bboxes:
            continue

        try:
            transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        except Exception:
            continue

        if transformed["bboxes"]:
            out_img_name = f"skew_{fname}"
            out_lbl_name = f"skew_{stem}.txt"
            cv2.imwrite(os.path.join(img_dir, out_img_name), transformed["image"])
            with open(os.path.join(lbl_dir, out_lbl_name), "w", encoding="utf-8") as f:
                for cls, box in zip(transformed["class_labels"], transformed["bboxes"]):
                    f.write(f"{int(float(cls))} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
            generated += 1

    print(f"  [{prefix.upper()}] Generated {generated} skewed augmentation samples.")


def add_external_as_negatives(dataset_dir: str, unified_dir: str, max_samples: int = 120, seed: int = 42):
    dest_img = os.path.join(unified_dir, "train", "images")
    dest_lbl = os.path.join(unified_dir, "train", "labels")

    src_img_dir = os.path.join(dataset_dir, "train", "images")
    if not os.path.exists(src_img_dir):
        src_img_dir = os.path.join(dataset_dir, "images")
        if not os.path.exists(src_img_dir):
            src_img_dir = dataset_dir

    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(valid_exts)]

    random.seed(seed)
    count = min(len(all_images), max_samples)
    selected = random.sample(all_images, count)

    for filename in selected:
        base_name, _ = os.path.splitext(filename)
        shutil.copy(os.path.join(src_img_dir, filename), os.path.join(dest_img, f"neg_{filename}"))
        open(os.path.join(dest_lbl, f"neg_{base_name}.txt"), "w", encoding="utf-8").close()

    print(f"  [NEGATIVES] Injected {count} negative background samples (0-byte labels).")


def convert_yolo_to_coco(unified_dir: str, coco_root: str, split_name: str, coco_split_name: str):
    src_img_dir = os.path.join(unified_dir, split_name, "images")
    src_lbl_dir = os.path.join(unified_dir, split_name, "labels")
    dest_img_dir = os.path.join(coco_root, coco_split_name)
    os.makedirs(dest_img_dir, exist_ok=True)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "plate", "supercategory": "none"}],
    }

    valid_exts = (".jpg", ".jpeg", ".png")
    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(valid_exts)]
    img_files.sort()

    ann_id = 1
    for img_id, filename in enumerate(img_files, 1):
        src_path = os.path.join(src_img_dir, filename)
        dest_path = os.path.join(dest_img_dir, filename)
        shutil.copy(src_path, dest_path)

        with Image.open(src_path) as im:
            width, height = im.size

        coco_data["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        stem, _ = os.path.splitext(filename)
        lbl_path = os.path.join(src_lbl_dir, f"{stem}.txt")
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        abs_x = (cx - bw / 2.0) * width
                        abs_y = (cy - bh / 2.0) * height
                        abs_w = bw * width
                        abs_h = bh * height
                        coco_data["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 1,
                            "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)],
                            "area": round(abs_w * abs_h, 2),
                            "iscrowd": 0,
                        })
                        ann_id += 1

    ann_file = os.path.join(coco_root, "annotations", f"instances_{coco_split_name}.json")
    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(coco_data, f)

    print(f"  [COCO] Converted {len(coco_data['images'])} images and {len(coco_data['annotations'])} boxes -> {ann_file}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare YOLOX COCO dataset from Roboflow")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY environment variable)",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_coco",
        help="Destination COCO dataset folder (default: dataset_coco)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep temporary raw Roboflow downloads",
    )
    args = parser.parse_args()

    if Image is None or Roboflow is None or cv2 is None or A is None:
        print("[ERROR] Missing required libraries. Please run: pip install -r training/requirements-train.txt")
        sys.exit(1)

    if not args.api_key:
        print("[ERROR] Roboflow API key is required. Run with: python training/prepare_dataset.py --api-key YOUR_KEY")
        sys.exit(1)

    print("=" * 65)
    print("EIU LPR: Dataset Download, Augmentation & COCO Conversion")
    print("=" * 65)

    raw_dir = "raw_datasets"
    unified_dir = "dataset_unified"
    coco_root = args.output_dir

    for split in ["train", "valid"]:
        os.makedirs(os.path.join(unified_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(unified_dir, split, "labels"), exist_ok=True)
    os.makedirs(os.path.join(coco_root, "annotations"), exist_ok=True)

    # 1. Download
    car_dir, moto_dir, neg_dir = download_roboflow_datasets(args.api_key, raw_dir)

    # 2. Merge & Balance (1,400 cars, 1,400 motorbikes)
    print("\n4/5 Merging & Balancing Fleet (1:1 Ratio)...")
    merge_datasets(car_dir, "car", unified_dir, max_train_samples=1400, seed=42)
    merge_datasets(moto_dir, "moto", unified_dir, max_train_samples=1400, seed=42)

    # 3. Augment with Skew
    print("\nGenerating Albumentations Perspective Skew Samples...")
    generate_skewed_samples(unified_dir, prefix="car", sample_count=140)
    generate_skewed_samples(unified_dir, prefix="moto", sample_count=140)

    # 4. Inject Negative Samples
    print("\nInjecting Negative Samples...")
    add_external_as_negatives(neg_dir, unified_dir, max_samples=120)

    # 5. Convert to COCO JSON
    print("\n5/5 Converting Unified Dataset to COCO JSON Format for YOLOX...")
    convert_yolo_to_coco(unified_dir, coco_root, "train", "train2017")
    convert_yolo_to_coco(unified_dir, coco_root, "valid", "val2017")

    # Cleanup temporary folders if not requested to keep
    if not args.keep_raw:
        print("\nCleaning up temporary working folders...")
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(unified_dir, ignore_errors=True)

    print("\n" + "=" * 65)
    print(f"[SUCCESS] Dataset ready at: {os.path.abspath(coco_root)}")
    print(f"Contains: {os.path.join(coco_root, 'annotations', 'instances_train2017.json')}")
    print(f"          {os.path.join(coco_root, 'annotations', 'instances_val2017.json')}")
    print("=" * 65)


if __name__ == "__main__":
    main()
