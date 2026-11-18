
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

FEATURE_COLS = ["area", "eccentricity", "solidity", "perimeter", "mean_intensity", "texture_contrast"]


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{name}_split_labels.csv")
    df = df.dropna(subset=FEATURE_COLS)
    return df


def main():
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    X_train, y_train = train_df[FEATURE_COLS].values, train_df["is_cancerous"].values
    X_val, y_val = val_df[FEATURE_COLS].values, val_df["is_cancerous"].values
    X_test, y_test = test_df[FEATURE_COLS].values, test_df["is_cancerous"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train class balance (is_cancerous=1): {y_train.mean():.3f}")

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(X_train_s, y_train)

    for split_name, X, y in [("VAL", X_val_s, y_val), ("TEST", X_test_s, y_test)]:
        preds = clf.predict(X)
        macro_f1 = f1_score(y, preds, average="macro")
        print(f"\n--- {split_name} ---")
        print(f"Macro-F1: {macro_f1:.4f}")
        print(classification_report(y, preds, target_names=["non-cancerous", "cancerous"]))
        print("Confusion matrix:\n", confusion_matrix(y, preds))

    print("\nFeature importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "scaler": scaler, "feature_cols": FEATURE_COLS}, MODELS_DIR / "iscancerous_rf.joblib")
    print(f"\nModel saved to {MODELS_DIR / 'iscancerous_rf.joblib'}")


if __name__ == "__main__":
    main()
