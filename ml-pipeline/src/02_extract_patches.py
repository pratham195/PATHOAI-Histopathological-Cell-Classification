
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

CANCEROUS_CATEGORY = "Neoplastic"


def crop_patch(image, cy, cx, size):
    """Crop a size x size patch centered at (cy, cx), zero-padding at borders."""
    half = size // 2
    h, w = image.shape[:2]
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    pad_top, pad_left = max(0, -y0), max(0, -x0)
    pad_bottom, pad_right = max(0, y1 - h), max(0, x1 - w)
    y0c, y1c = max(0, y0), min(h, y1)
    x0c, x1c = max(0, x0), min(w, x1)
    crop = image[y0c:y1c, x0c:x1c]
    if crop.ndim == 2:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)))
    else:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
    return crop


def compute_morphology_features(mask_binary, image_rgb):
    """Compute shape + texture features for one nucleus mask, used later
    to generate the human-readable explanation (Module 9)."""
    lbl = label(mask_binary)
    props = regionprops(lbl)
    if not props:
        return None
    p = max(props, key=lambda r: r.area)  # largest component = the nucleus

    gray_patch = (image_rgb.mean(axis=-1)).astype(np.uint8)
    glcm = graycomatrix(gray_patch, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]

    return {
        "area": p.area,
        "eccentricity": p.eccentricity,      # 0 = circular, close to 1 = elongated/irregular
        "solidity": p.solidity,               # low solidity = irregular/jagged border
        "perimeter": p.perimeter,
        "mean_intensity": float(gray_patch[mask_binary].mean()) if mask_binary.any() else np.nan,
        "texture_contrast": contrast,         # higher = coarser chromatin texture
    }


def extract_split(split_name: str, patch_size: int):
    parquet_path = RAW_DIR / f"{split_name}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found. Run 01_acquire_data.py first.")

    df = pd.read_parquet(parquet_path)
    patches, records = [], []

    for img_idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name}"):
        image = np.array(row["image"])           # HxWx3
        instances = row["instances"]               # list of {mask, category}
        tissue = row["tissue"]
        image_id = f"{tissue}_{img_idx}"            # grouping unit for Module 2's split
                                                       # (PanNuke has no patient id; this is the
                                                       # finest available proxy — see module docstring)

        for nuc_idx, inst in enumerate(instances):
            mask = np.array(inst["mask"]).astype(bool)
            category = inst["category"]

            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            cy, cx = int(ys.mean()), int(xs.mean())

            img_patch = crop_patch(image, cy, cx, patch_size)
            mask_patch = crop_patch(mask.astype(np.uint8), cy, cx, patch_size).astype(bool)

            feats = compute_morphology_features(mask_patch, img_patch)
            if feats is None:
                continue

            patches.append(img_patch)
            records.append({
                "image_id": image_id,
                "image_idx": img_idx,
                "tissue": tissue,
                "nucleus_idx": nuc_idx,
                "category": category,
                "is_cancerous": int(category == CANCEROUS_CATEGORY),
                **feats,
            })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED_DIR / f"{split_name}_patches.npz", patches=np.array(patches, dtype=np.uint8))
    pd.DataFrame(records).to_csv(PROCESSED_DIR / f"{split_name}_labels.csv", index=False)
    print(f"{split_name}: {len(records)} nucleus patches saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--splits", nargs="+", default=["fold1", "fold2", "fold3"])
    args = parser.parse_args()

    for split in args.splits:
        extract_split(split, args.patch_size)
