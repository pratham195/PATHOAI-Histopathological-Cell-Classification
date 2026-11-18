
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "untested"


def main():
    labels = pd.read_csv(PROCESSED_DIR / "test_split_labels.csv")
    patches = np.load(PROCESSED_DIR / "test_split_patches.npz")["patches"]

    (SAMPLES_DIR / "cancerous").mkdir(parents=True, exist_ok=True)
    (SAMPLES_DIR / "non_cancerous").mkdir(parents=True, exist_ok=True)

    for is_cancerous, folder_name in [(1, "cancerous"), (0, "non_cancerous")]:
        subset = labels[labels["is_cancerous"] == is_cancerous]  # every matching row, no limit

        for i, row in enumerate(subset.itertuples()):
            image = Image.fromarray(patches[row.Index])
            filename = f"{row.category.replace(' ', '_')}_{i}.png"
            image.save(SAMPLES_DIR / folder_name / filename)

        print(f"Saved {len(subset)} images to {SAMPLES_DIR / folder_name}")

    print(f"\nDone. All images are from the held-out test set — no model has trained on them.")


if __name__ == "__main__":
    main()