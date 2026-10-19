# ==============================================================================
# Vietnamese LPRNet OCR Training Pipeline with CTC Loss & ONNX Export
# Includes:
#   1. Automatic Perceptual Deduplication (dHash) & Annotation Synchronization
#   2. Multi-Row Aspect-Ratio Slicing (Motorbike plates sliced to 94x24 single strips)
#   3. Comprehensive Data Augmentation (Perspective tilt, bumper padding, color inversion, blur)
#   4. Cosine Annealing Learning Rate Schedule
#   5. Dynamic Batch Length CTC Loss & Robust ONNX Exporter
# ==============================================================================

import os
import time
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------------------
# Step 1: Environment & Device Setup
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Clone real Vietnamese dataset if not already present in environment
if not os.path.exists("PBL4_Deep-Learning"):
    print("Cloning real Vietnamese plate dataset from GitHub...")
    os.system("git clone --depth 1 https://github.com/lephamcong/PBL4_Deep-Learning.git")
else:
    print("Dataset already available in environment.")

# ------------------------------------------------------------------------------
# Step 2: Vietnamese Character Vocabulary
# ------------------------------------------------------------------------------
# Index 0 is reserved for CTC blank token '_'
# Remaining 32 chars: 10 digits + 21 valid Vietnamese letters + hyphen '-'
CHARS = ['_'] + list("0123456789ABCDEFGHKLMNPRSTUVXYZ-")
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

print(f"Total classes (including CTC blank): {NUM_CLASSES}")
print(f"Vocabulary: {' '.join(CHARS)}")

# ------------------------------------------------------------------------------
# Step 3: Dataset Deduplication & Synchronized Annotation Cleanup
# ------------------------------------------------------------------------------
def compute_dhash(img_path, hash_size=8):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return None

real_data_dir = "PBL4_Deep-Learning/Dataset/BiensoxeVietNam"
all_img_paths = glob.glob(os.path.join(real_data_dir, "**", "*.jpg"), recursive=True)
all_img_paths.extend(glob.glob(os.path.join(real_data_dir, "**", "*.png"), recursive=True))

print(f"\nInitial scan: Found {len(all_img_paths):,} images in {real_data_dir}")

seen_hashes = {}
deleted_images = 0
deleted_annotations = 0

for p in all_img_paths:
    h = compute_dhash(p)
    if h is None:
        continue
    if h in seen_hashes:
        try:
            if os.path.exists(p):
                os.remove(p)
                deleted_images += 1
            
            # Check and delete corresponding annotation file(s) in same directory
            stem, _ = os.path.splitext(p)
            for ext in ['.txt', '.xml', '.json', '.label']:
                annot_path = stem + ext
                if os.path.exists(annot_path):
                    os.remove(annot_path)
                    deleted_annotations += 1
            
            # Check parallel 'labels' folder if present
            dir_name = os.path.dirname(p)
            base_name = os.path.basename(stem)
            parent_dir = os.path.dirname(dir_name)
            parallel_labels = os.path.join(parent_dir, "labels")
            if os.path.exists(parallel_labels):
                for ext in ['.txt', '.xml', '.json', '.label']:
                    annot_path = os.path.join(parallel_labels, base_name + ext)
                    if os.path.exists(annot_path):
                        os.remove(annot_path)
                        deleted_annotations += 1
        except Exception as e:
            print(f"Error removing duplicate {p}: {e}")
    else:
        seen_hashes[h] = p

print("Deduplication complete:")
print(f"  - Purged duplicate images      : {deleted_images:,}")
print(f"  - Purged matching annotations  : {deleted_annotations:,}")
print(f"  - Unique clean images remaining: {len(seen_hashes):,}")

# ------------------------------------------------------------------------------
# Step 4: LPRNet Architecture Definition
# ------------------------------------------------------------------------------
class SmallBasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, class_num=NUM_CLASSES, dropout_rate=0.3):
        super().__init__()
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1)),
            SmallBasicBlock(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            SmallBasicBlock(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SmallBasicBlock(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(256, class_num, kernel_size=(1, 13), stride=1),
            nn.BatchNorm2d(class_num),
            nn.ReLU(inplace=True),
        )
        self.container = nn.Sequential(
            nn.Conv2d(64 + 128 + 256 + class_num, class_num, kernel_size=1, stride=1)
        )

    def forward(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)

        global_context = []
        target_size = (1, 76)
        for f in keep_features:
            if f.shape[2:] != target_size:
                f = nn.functional.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            global_context.append(f)

        merged = torch.cat(global_context, dim=1)
        logits = self.container(merged)
        return logits.squeeze(2) # [Batch, Class_Num, Seq_Len]

# ------------------------------------------------------------------------------
# Step 5: Multi-Row Plate Slicing, Normalization & Data Augmentation
# ------------------------------------------------------------------------------
def extract_plate_label(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    label = stem.split('_')[0].upper().replace('-', '').replace('.', '')
    return label

def split_motorbike_label(full_label):
    if len(full_label) >= 8 and full_label[2:4].isalnum() and not full_label[2:4].isdigit():
        return full_label[:4], full_label[4:]
    elif len(full_label) == 7 or len(full_label) == 8:
        return full_label[:3], full_label[3:]
    else:
        half = len(full_label) // 2
        return full_label[:half], full_label[half:]

class RealVietnameseDataset(Dataset):
    def __init__(self, sample_records, is_train=True):
        self.sample_records = sample_records
        self.is_train = is_train

    def __len__(self):
        return len(self.sample_records)

    def __getitem__(self, idx):
        img_path, slice_mode, label_str = self.sample_records[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((24, 94, 3), dtype=np.uint8)
        else:
            h, w = img.shape[:2]
            if slice_mode == 'top':
                img = img[0:max(1, int(h * 0.55)), :]
            elif slice_mode == 'bottom':
                img = img[max(0, int(h * 0.45)):h, :]

            if self.is_train:
                # 1. Random Perspective / Affine Tilt
                if random.random() < 0.6:
                    h_cur, w_cur = img.shape[:2]
                    pts1 = np.float32([[0, 0], [w_cur, 0], [0, h_cur], [w_cur, h_cur]])
                    dx = random.uniform(-0.08, 0.08) * w_cur
                    dy = random.uniform(-0.12, 0.12) * h_cur
                    pts2 = np.float32([
                        [max(0, dx), max(0, dy)],
                        [min(w_cur, w_cur - dx), max(0, -dy)],
                        [max(0, -dx), min(h_cur, h_cur + dy)],
                        [min(w_cur, w_cur + dx), min(h_cur, h_cur - dy)]
                    ])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    img = cv2.warpPerspective(img, matrix, (w_cur, h_cur), borderMode=cv2.BORDER_REPLICATE)

                # 2. Random Bumper Margin Simulation
                if random.random() < 0.4:
                    pad_top = random.randint(0, max(1, int(img.shape[0] * 0.12)))
                    pad_bot = random.randint(0, max(1, int(img.shape[0] * 0.12)))
                    pad_left = random.randint(0, max(1, int(img.shape[1] * 0.08)))
                    pad_right = random.randint(0, max(1, int(img.shape[1] * 0.08)))
                    bg_color = [random.randint(15, 60)] * 3
                    img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=bg_color)

                # 3. Color Inversion / Polarity Reversal
                if random.random() < 0.15:
                    img = 255 - img

                # 4. Random Lighting & Contrast Jitter
                if random.random() < 0.5:
                    alpha = random.uniform(0.75, 1.25)
                    beta = random.randint(-25, 25)
                    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

                # 5. Random Mild Blur
                if random.random() < 0.25:
                    ksize = random.choice([3, 5])
                    img = cv2.GaussianBlur(img, (ksize, ksize), 0)

            # Uniform resize to single-row LPRNet input dimension
            img = cv2.resize(img, (94, 24))

        norm = (img.astype(np.float32) - 127.5) * 0.0078125
        tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1)

        target = [CHARS_DICT[c] for c in label_str if c in CHARS_DICT]
        if len(target) == 0:
            target = [CHARS_DICT['0']]
        return tensor, torch.tensor(target, dtype=torch.long), len(target)

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    target_lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    flat_targets = torch.cat(targets)
    return images, flat_targets, target_lengths

# ------------------------------------------------------------------------------
# Build Dataset Records & DataLoaders
# ------------------------------------------------------------------------------
print("\nAnalyzing plate aspect ratios and creating single-row sample records...")
clean_files = [p for p in seen_hashes.values() if os.path.exists(p)]
random.seed(42)
random.shuffle(clean_files)

all_sample_records = []
for p in clean_files:
    full_lbl = extract_plate_label(p)
    if not full_lbl:
        continue
    img_tmp = cv2.imread(p)
    if img_tmp is None:
        continue
    h_tmp, w_tmp = img_tmp.shape[:2]
    ar = w_tmp / float(h_tmp) if h_tmp > 0 else 2.5
    
    if ar > 1.5:
        # 1-line rectangular plate (Car)
        all_sample_records.append((p, 'full', full_lbl))
    else:
        # 2-line square plate (Motorbike) -> split into 2 single-row strips
        lbl_top, lbl_bot = split_motorbike_label(full_lbl)
        all_sample_records.append((p, 'top', lbl_top))
        all_sample_records.append((p, 'bottom', lbl_bot))

random.shuffle(all_sample_records)
split_idx = int(len(all_sample_records) * 0.90)
train_records = all_sample_records[:split_idx]
val_records = all_sample_records[split_idx:]

train_dataset = RealVietnameseDataset(train_records, is_train=True)
val_dataset = RealVietnameseDataset(val_records, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=2)

print(f"Total single-row training samples generated: {len(all_sample_records):,}")
print(f"  - Training set   : {len(train_dataset):,} samples")
print(f"  - Validation set : {len(val_dataset):,} samples")

# ------------------------------------------------------------------------------
# Step 6: Training Loop with CTC Loss
# ------------------------------------------------------------------------------
model = LPRNet(class_num=NUM_CLASSES, dropout_rate=0.3).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

EPOCHS = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
best_loss = float("inf")

print("\nBeginning LPRNet Training on Real Vietnamese Plates...")
start_train_t = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for imgs, targets, target_lengths in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        optimizer.zero_grad()

        logits = model(imgs)
        seq_len = logits.shape[2]
        input_lengths = torch.full((imgs.shape[0],), seq_len, dtype=torch.long, device=device)

        log_probs = logits.permute(2, 0, 1).log_softmax(dim=2)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    avg_loss = train_loss / max(1, len(train_loader))

    # Validation pass
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, targets, target_lengths in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            target_lengths = target_lengths.to(device)
            logits = model(imgs)
            seq_len = logits.shape[2]
            val_input_lengths = torch.full((imgs.shape[0],), seq_len, dtype=torch.long, device=device)
            log_probs = logits.permute(2, 0, 1).log_softmax(dim=2)
            loss = criterion(log_probs, targets, val_input_lengths, target_lengths)
            val_loss += loss.item()

    avg_val = val_loss / max(1, len(val_loader))
    print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train CTC Loss: {avg_loss:.4f} | Val CTC Loss: {avg_val:.4f}")

    if avg_val < best_loss:
        best_loss = avg_val
        torch.save(model.state_dict(), "best_lprnet.pth")

train_mins = (time.time() - start_train_t) / 60.0
print(f"\nTraining finished in {train_mins:.1f} minutes. Best Val Loss: {best_loss:.4f}")

# ------------------------------------------------------------------------------
# Step 7: Test CTC Greedy Decoding on a Validation Sample
# ------------------------------------------------------------------------------
model.load_state_dict(torch.load("best_lprnet.pth"))
model.eval()

sample_path, sample_mode, true_label = val_records[0]
sample_img = cv2.imread(sample_path)
h_s, w_s = sample_img.shape[:2]
if sample_mode == 'top':
    sample_img = sample_img[0:max(1, int(h_s * 0.55)), :]
elif sample_mode == 'bottom':
    sample_img = sample_img[max(0, int(h_s * 0.45)):h_s, :]

sample_img = cv2.resize(sample_img, (94, 24))
test_norm = (sample_img.astype(np.float32) - 127.5) * 0.0078125
test_tensor = torch.tensor(test_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(test_tensor)
    best_path = logits.argmax(dim=1)[0].cpu().numpy()

# CTC Greedy Collapse
decoded = []
prev = 0
for idx in best_path:
    if idx != 0 and idx != prev:
        decoded.append(CHARS[idx])
    prev = idx
pred_str = "".join(decoded)

print(f"\nSample Path : {sample_path} (Mode: {sample_mode})")
print(f"Ground Truth: {true_label}")
print(f"Predicted   : {pred_str}")
print(f"Match       : {true_label == pred_str}")

# ------------------------------------------------------------------------------
# Step 8: Export directly to ONNX Format
# ------------------------------------------------------------------------------
model = LPRNet(class_num=NUM_CLASSES, dropout_rate=0.0).cpu()
model.load_state_dict(torch.load("best_lprnet.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 24, 94, dtype=torch.float32)
onnx_export_path = "lprnet.onnx"

try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_export_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        dynamo=False
    )
    print(f"\nSuccessfully exported: {onnx_export_path} ({os.path.getsize(onnx_export_path) / (1024*1024):.2f} MB)")
except Exception as e:
    print(f"Primary export notice: {e}. Retrying standard TorchScript exporter...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_export_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported via fallback: {onnx_export_path}")

# Verify exported ONNX model with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession(onnx_export_path, providers=['CPUExecutionProvider'])
ort_out = session.run(None, {"input": dummy_input.numpy()})[0]
print(f"ONNX Output Shape: {ort_out.shape} (Verified!)")

