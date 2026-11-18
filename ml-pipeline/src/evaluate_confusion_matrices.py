
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestDataset(Dataset):
    def __init__(self, task, encoder=None):
        labels = pd.read_csv(DATA_DIR / "test_split_labels.csv")
        patches = np.load(
            DATA_DIR / "test_split_patches.npz"
        )["patches"]

        self.patches = patches[:len(labels)]

        if task == "binary":
            self.labels = labels["is_cancerous"].to_numpy(dtype=int)
        else:
            self.labels = encoder.transform(
                labels["category"].to_numpy()
            )

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(self.patches[idx])
        return image, int(self.labels[idx])


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(task, model_path, class_names, encoder=None):
    dataset = TestDataset(task, encoder)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = SmallCNN(len(class_names)).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    actual, predicted = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().numpy()

            actual.extend(labels.numpy())
            predicted.extend(predictions)

    cm = confusion_matrix(
        actual,
        predicted,
        labels=list(range(len(class_names)))
    )

    result = pd.DataFrame(
        cm,
        index=[f"Actual: {name}" for name in class_names],
        columns=[f"Predicted: {name}" for name in class_names]
    )

    output_path = DATA_DIR / f"{task}_cnn_confusion_matrix.csv"
    result.to_csv(output_path)

    print(f"\n{task.upper()} CNN CONFUSION MATRIX")
    print(result)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    encoder = joblib.load(
        MODELS_DIR / "celltype_label_encoder.joblib"
    )

    evaluate(
        "binary",
        MODELS_DIR / "cnn_iscancerous.pt",
        ["Non-cancerous", "Cancerous"]
    )

    evaluate(
        "celltype",
        MODELS_DIR / "cnn_celltype.pt",
        list(encoder.classes_),
        encoder
    )
