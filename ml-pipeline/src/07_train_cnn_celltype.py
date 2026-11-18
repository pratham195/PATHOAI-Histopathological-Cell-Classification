
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3

class NucleusDataset(Dataset):
    def __init__(self, split_name, label_encoder, augment=False):
        labels = pd.read_csv(PROCESSED_DIR / f"{split_name}_split_labels.csv")
        patches = np.load(PROCESSED_DIR / f"{split_name}_split_patches.npz")["patches"]

        self.patches = patches[: len(labels)]
        categories = labels["category"].values[: len(patches)]
        self.labels = label_encoder.transform(categories)  # e.g. "Neoplastic" -> 3

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(self.patches[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


class SmallCNN(nn.Module):
    """Identical architecture to the isCancerous CNN, except the final layer
    now outputs 5 scores (one per cell type) instead of 2."""

    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_epoch(model, loader, criterion, optimizer=None):
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

    train_labels_df = pd.read_csv(PROCESSED_DIR / "train_split_labels.csv")
    # fit once on the training categories, so "Neoplastic" always maps to
    # the same integer everywhere (train, val, test, and later at inference)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels_df["category"].values)
    class_names = list(label_encoder.classes_)
    print(f"Cell-type classes ({len(class_names)}): {class_names}")

    # check class balance before training — 5-way imbalance can be much
    # more severe than the 2-way isCancerous split was
    print("\nClass distribution in training set:")
    print(train_labels_df["category"].value_counts())

    train_ds = NucleusDataset("train", label_encoder, augment=True)
    val_ds = NucleusDataset("val", label_encoder, augment=False)
    test_ds = NucleusDataset("test", label_encoder, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(class_names)), y=train_ds.labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = SmallCNN(num_classes=len(class_names)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1, _, _ = run_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch:2d}/{EPOCHS} | train loss {train_loss:.3f} f1 {train_f1:.3f} "
              f"| val loss {val_loss:.3f} f1 {val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nBest validation macro-F1: {best_val_f1:.3f} — restoring that checkpoint for final testing")
    model.load_state_dict(best_state)

    print("\n--- Final test-set evaluation ---")
    _, test_f1, test_preds, test_labels = run_epoch(model, test_loader, criterion)
    print(f"Test macro-F1: {test_f1:.3f}")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    print("Confusion matrix (rows = actual, columns = predicted):")
    cm = confusion_matrix(test_labels, test_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODELS_DIR / "cnn_celltype.pt")
    joblib.dump(label_encoder, MODELS_DIR / "celltype_label_encoder.joblib")
    print(f"\nModel saved to {MODELS_DIR / 'cnn_celltype.pt'}")
    print(f"Label encoder saved to {MODELS_DIR / 'celltype_label_encoder.joblib'}")


if __name__ == "__main__":
    main()