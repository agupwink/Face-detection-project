"""
Fine-tune ResNet18 for age regression on UTKFace Indian faces.

Usage:
  1. Download UTKFace from https://susanqq.github.io/UTKFace/
     (fill the Google Form → download the aligned&cropped images zip)
  2. Extract so images are in:  ./UTKFace/*.jpg
  3. Run:  python train_age.py

Output: age_model_indian.pth  (loaded automatically by face_detect.py)
"""

import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "UTKFace")
OUT_PATH   = os.path.join(os.path.dirname(__file__), "age_model_indian.pth")
EPOCHS     = 40
BATCH      = 32
LR         = 3e-4
VAL_SPLIT  = 0.15
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


class UTKFaceIndian(Dataset):
    """
    UTKFace filename format: [age]_[gender]_[race]_[datetime].jpg
    race codes: 0=White 1=Black 2=Asian 3=Indian 4=Others
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []
        for path in glob.glob(os.path.join(root_dir, "*.jpg")):
            parts = os.path.basename(path).split("_")
            if len(parts) < 3:
                continue
            try:
                age  = int(parts[0])
                race = int(parts[2])
            except ValueError:
                continue
            if race == 3 and 1 <= age <= 80:
                self.samples.append((path, float(age)))
        print(f"Indian face samples found: {len(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No Indian-labelled images found in {root_dir}.\n"
                "Make sure UTKFace/*.jpg images are present."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, age = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(age, dtype=torch.float32)


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)


def train():
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_ds  = UTKFaceIndian(DATA_DIR)
    n_val    = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_ds.dataset.transform = train_tf
    val_ds.dataset              = UTKFaceIndian(DATA_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)

    model     = build_model()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_mae  = float("inf")
    print(f"Training on {DEVICE}  |  train={n_train}  val={n_val}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for imgs, ages in train_loader:
            imgs, ages = imgs.to(DEVICE), ages.to(DEVICE)
            optimizer.zero_grad()
            pred = model(imgs).squeeze(1)
            loss = criterion(pred, ages)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        scheduler.step()

        # ── validate ──
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for imgs, ages in val_loader:
                imgs, ages = imgs.to(DEVICE), ages.to(DEVICE)
                pred = model(imgs).squeeze(1)
                val_mae += torch.abs(pred - ages).sum().item()

        train_loss /= n_train
        val_mae    /= n_val
        marker = " ← best" if val_mae < best_mae else ""
        print(f"Epoch {epoch:3d}/{EPOCHS}  train_MAE={train_loss:.2f}  val_MAE={val_mae:.2f}{marker}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), OUT_PATH)

    print(f"\nDone. Best val MAE: {best_mae:.2f} years")
    print(f"Model saved → {OUT_PATH}")


if __name__ == "__main__":
    train()
