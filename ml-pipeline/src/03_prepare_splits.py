
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SPLIT_SEED = 42
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# TEST_FRAC is the remainder (0.15)


def load_all_folds():
    label_dfs, patch_arrays = [], []
    total = 0
    for fold_labels_path in sorted(PROCESSED_DIR.glob("fold*_labels.csv")):
        fold_name = fold_labels_path.stem.replace("_labels", "")
        df = pd.read_csv(fold_labels_path)
        df["fold_source"] = fold_name
        df["global_idx"] = np.arange(total, total + len(df))
        total += len(df)

        patches = np.load(PROCESSED_DIR / f"{fold_name}_patches.npz")["patches"]
        assert len(patches) == len(df), f"Mismatch in {fold_name}: {len(patches)} patches vs {len(df)} labels"

        label_dfs.append(df)
        patch_arrays.append(patches)

    all_labels = pd.concat(label_dfs, ignore_index=True)
    all_patches = np.concatenate(patch_arrays, axis=0)
    # unique image id must be globally unique across folds/tissues
    all_labels["image_id"] = all_labels["fold_source"] + "_" + all_labels["image_id"].astype(str)
    return all_labels, all_patches


def image_level_split(labels: pd.DataFrame):
    groups = labels["image_id"].values

    gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=SPLIT_SEED)
    train_idx, temp_idx = next(gss1.split(labels, groups=groups))

    temp_labels = labels.iloc[temp_idx]
    remaining_frac_for_val = VAL_FRAC / (1 - TRAIN_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=remaining_frac_for_val, random_state=SPLIT_SEED)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_labels, groups=temp_labels["image_id"].values))

    val_idx = temp_labels.iloc[val_rel_idx].index.values
    test_idx = temp_labels.iloc[test_rel_idx].index.values

    # sanity check: zero image-id overlap
    train_i = set(labels.iloc[train_idx]["image_id"])
    val_i = set(labels.iloc[val_idx]["image_id"])
    test_i = set(labels.iloc[test_idx]["image_id"])
    assert not (train_i & val_i), "Image leakage between train/val!"
    assert not (train_i & test_i), "Image leakage between train/test!"
    assert not (val_i & test_i), "Image leakage between val/test!"

    return train_idx, val_idx, test_idx


def main():
    print("Loading all extracted folds...")
    labels, patches = load_all_folds()
    print(f"Total nuclei: {len(labels)}  |  Unique images: {labels['image_id'].nunique()}")

    train_idx, val_idx, test_idx = image_level_split(labels)

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_labels = labels.iloc[idx].reset_index(drop=True)
        split_patches = patches[idx]

        split_labels.to_csv(PROCESSED_DIR / f"{name}_split_labels.csv", index=False)
        np.savez_compressed(PROCESSED_DIR / f"{name}_split_patches.npz", patches=split_patches)

        print(f"\n{name.upper()}  — {len(split_labels)} nuclei, {split_labels['image_id'].nunique()} images")
        print("  is_cancerous distribution:\n", split_labels["is_cancerous"].value_counts(normalize=True).round(3))

    print("\nImage-level split complete with verified zero overlap.")
    print("Saved: {train,val,test}_split_labels.csv and _split_patches.npz in data/processed/")


if __name__ == "__main__":
    main()
