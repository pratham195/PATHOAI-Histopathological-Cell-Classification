
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3


class NucleusDataset(Dataset):
    """Wraps the saved patch images + labels so PyTorch can load them in batches."""

    def __init__(self, split_name, augment=False):
        labels = pd.read_csv(PROCESSED_DIR / f"{split_name}_split_labels.csv")
        patches = np.load(PROCESSED_DIR / f"{split_name}_split_patches.npz")["patches"]

        # keep only rows that survived feature extraction (a few can be dropped
        # earlier if a mask measurement failed) — patches and labels must line up
        self.patches = patches[: len(labels)]
        self.labels = labels["is_cancerous"].values[: len(patches)]

        base_transform = [
            transforms.ToTensor(),  # HxWx3 uint8 numpy array -> 3xHxW float tensor, scaled to [0,1]
        ]
        if augment:
            # small, realistic augmentations only — a cell photographed at a
            # slightly different rotation or flipped orientation is still the
            # same cell, so these don't change what the label should be.
            # NOTE: these transforms need a PIL Image, not a raw numpy array,
            # so ToPILImage() must run first, then ToTensor() converts back
            # to a tensor at the very end.
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose(base_transform)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.patches[idx]
        label = self.labels[idx]
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class SmallCNN(nn.Module):
    """A deliberately small CNN: 3 conv blocks, then a small classifier head.
    Kept small on purpose — a bigger network would overfit this dataset size
    even with augmentation."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),  # 64->32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # randomly drops half the connections during training, another overfitting guard
            nn.Linear(64, 2),  # 2 output scores: non-cancerous, cancerous
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over the data. If optimizer is given, this trains; otherwise
    it only evaluates (used for validation/test)."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1, all_preds, all_labels


def main():
    print(f"Using device: {DEVICE}")

    train_ds = NucleusDataset("train", augment=True)
    val_ds = NucleusDataset("val", augment=False)
    test_ds = NucleusDataset("test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # same principle as the classical model comparison: handle class imbalance
    # explicitly rather than ignoring it, so the CNN doesn't just learn to
    # always predict the majority (non-cancerous) class
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_ds.labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = SmallCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch:2d}/{EPOCHS} | train loss {train_loss:.3f} f1 {train_f1:.3f} "
              f"| val loss {val_loss:.3f} f1 {val_f1:.3f}")

        # keep the best-performing version seen so far, in case later epochs
        # start overfitting (val score would get worse while train keeps improving)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest validation macro-F1: {best_val_f1:.3f} — restoring that checkpoint for final testing")
    model.load_state_dict(best_state)

    print("\n--- Final test-set evaluation ---")
    _, test_f1, test_preds, test_labels = run_epoch(model, test_loader, criterion)
    print(f"Test macro-F1: {test_f1:.3f}")
    print(classification_report(test_labels, test_preds, target_names=["non-cancerous", "cancerous"]))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODELS_DIR / "cnn_iscancerous.pt")
    print(f"\nModel saved to {MODELS_DIR / 'cnn_iscancerous.pt'}")


if __name__ == "__main__":
    main()