
import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# PanNuke nucleus category mapping -> project's target labels
# PanNuke classes: Neoplastic, Inflammatory, Connective, Dead, Non-Neo Epithelial
PANNUKE_TO_CELLTYPE = {
    "Neoplastic": "epithelial",        # neoplastic cells are (malignant) epithelial-origin
    "Non-Neoplastic Epithelial": "epithelial",
    "Inflammatory": "inflammatory",
    "Connective": "fibroblast",        # connective/soft tissue ~ fibroblast-like
    "Dead": "other",
}
# isCancerous target: only "Neoplastic" nuclei are cancerous by PanNuke's own definition
CANCEROUS_CLASSES = {"Neoplastic"}


def download_pannuke(colon_only: bool = False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading PanNuke (RationAI/PanNuke) from Hugging Face...")
    ds = load_dataset("RationAI/PanNuke")

    for split_name, split in ds.items():
        print(f"Processing split: {split_name} ({len(split)} images)")
        out_path = RAW_DIR / f"{split_name}.parquet"
        if colon_only:
            split = split.filter(lambda ex: ex["tissue"] == "Colon")
            print(f"  -> filtered to Colon tissue: {len(split)} images")
        split.to_parquet(str(out_path))
        print(f"  saved -> {out_path}")

    print("\nDone. Raw PanNuke data saved under:", RAW_DIR)
    print("Next step: python src/02_extract_patches.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--colon-only",
        action="store_true",
        default=True,
        help="Keep only the Colon tissue subset (project scope: colorectal cancer). Default: True.",
    )
    parser.add_argument(
        "--all-tissue",
        dest="colon_only",
        action="store_false",
        help="Download all 19 tissue types instead of Colon-only.",
    )
    args = parser.parse_args()
    download_pannuke(colon_only=args.colon_only)
