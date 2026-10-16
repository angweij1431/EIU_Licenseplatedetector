import os
import time
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Device and Vocabulary Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Index 0 is reserved for CTC blank '_'
# Remaining 32 chars: 10 digits + 21 valid Vietnamese letters + hyphen '-'
CHARS = ['_'] + list("0123456789ABCDEFGHKLMNPRSTUVXYZ-")
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)

# 2. LPRNet Network Architecture
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
        for f in keep_features:
            if f.shape[2:] != keep_features[-1].shape[2:]:
                f = nn.functional.adaptive_avg_pool2d(f, keep_features[-1].shape[2:])
            global_context.append(f)

        merged = torch.cat(global_context, dim=1)
        logits = self.container(merged)
        return logits.squeeze(2) # [B, class_num, seq_len]


# 3. Authentic Vietnamese Synthetic Plate Generation
VALID_PROVINCES = ["29", "30", "31", "50", "51", "52", "59", "60", "61", "63", "43", "99"]
VALID_LETTERS = list("ABCDEFGHKLMNPRSTUVXYZ")

def generate_plate_text():
    prov = random.choice(VALID_PROVINCES)
    let = random.choice(VALID_LETTERS)
    if random.random() < 0.5:
        digits = "".join(str(random.randint(0, 9)) for _ in range(5))
        return f"{prov}{let}-{digits}"
    else:
        sub_d = str(random.randint(1, 9))
        digits = "".join(str(random.randint(0, 9)) for _ in range(5))
        return f"{prov}{let}{sub_d}-{digits}"

def render_plate_image(text, w=94, h=24):
    img = np.ones((h, w, 3), dtype=np.uint8) * random.randint(235, 255)
    clean_text = text.replace("-", "")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (t_w, _), _ = cv2.getTextSize(clean_text, font, font_scale, thickness)
    start_x = max(2, (w - t_w) // 2)
    start_y = int(h * 0.75)
    cv2.putText(img, clean_text, (start_x, start_y), font, font_scale, (15, 15, 15), thickness, cv2.LINE_AA)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

class SyntheticVietnamesePlateDataset(Dataset):
    def __init__(self, size=8000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        text = generate_plate_text().replace("-", "")
        img = render_plate_image(text)
        norm = (img.astype(np.float32) - 127.5) * 0.0078125
        tensor = torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1)
        targets = [CHARS_DICT[c] for c in text if c in CHARS_DICT]
        return tensor, torch.tensor(targets, dtype=torch.long), len(targets)

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    t_lens = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, torch.cat(targets), t_lens

def main():
    print(f"Initializing 33-class Vietnamese LPRNet...")
    model = LPRNet(class_num=NUM_CLASSES, dropout_rate=0.3).to(device)
    
    train_loader = DataLoader(SyntheticVietnamesePlateDataset(8000), batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SyntheticVietnamesePlateDataset(1000), batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    
    print("Training started for 25 epochs...")
    t0 = time.time()
    for epoch in range(1, 26):
        model.train()
        total_loss = 0.0
        for imgs, targets, t_lens in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            seq_len = logits.shape[2]
            in_lens = torch.full((imgs.shape[0],), seq_len, dtype=torch.long, device=device)
            log_probs = logits.permute(2, 0, 1).log_softmax(dim=2)
            loss = criterion(log_probs, targets, in_lens, t_lens)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 25:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch:02d}/25] | CTC Loss: {avg_loss:.4f}")
            
    print(f"Training completed in {(time.time() - t0)/60.0:.1f} minutes.")
    torch.save(model.state_dict(), "best_lprnet.pth")
    
    # Export to ONNX format
    model.eval()
    model.cpu()
    dummy = torch.randn(1, 3, 24, 94, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, "lprnet.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}}, opset_version=12
    )
    print("Successfully exported lprnet.onnx (Ready for Models_ocr/).")

if __name__ == "__main__":
    main()

