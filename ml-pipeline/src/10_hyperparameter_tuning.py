
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FEATURES = ["area", "eccentricity", "solidity", "perimeter", "mean_intensity", "texture_contrast"]

CANDIDATE_MAX_DEPTHS = [4, 8, 12, 16, 20, None]  # None = no depth limit


def load(name):
    df = pd.read_csv(PROCESSED_DIR / f"{name}_split_labels.csv")
    return df.dropna(subset=FEATURES)


def main():
    train, val, test = load("train"), load("val"), load("test")

    X_train, y_train = train[FEATURES], train["is_cancerous"]
    X_val, y_val = val[FEATURES], val["is_cancerous"]
    X_test, y_test = test[FEATURES], test["is_cancerous"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    print("Tuning max_depth for Random Forest (isCancerous task)\n")
    results = []
    best_depth, best_f1, best_model = None, -1, None

    for depth in CANDIDATE_MAX_DEPTHS:
        model = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42)
        model.fit(X_train_s, y_train, sample_weight=sample_weight)

        train_f1 = f1_score(y_train, model.predict(X_train_s), average="macro")
        val_f1 = f1_score(y_val, model.predict(X_val_s), average="macro")
        results.append({"max_depth": depth, "train_macro_f1": train_f1, "val_macro_f1": val_f1})
        print(f"max_depth={depth}: train macro-F1={train_f1:.3f}, val macro-F1={val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, best_depth, best_model = val_f1, depth, model

    print(f"\nBest max_depth on validation: {best_depth} (macro-F1 = {best_f1:.3f})")

    # a depth where train_f1 is much higher than val_f1 is a sign of overfitting —
    # worth calling out explicitly if it shows up in the results table above
    results_df = pd.DataFrame(results)
    results_df["overfit_gap"] = results_df["train_macro_f1"] - results_df["val_macro_f1"]
    results_df.to_csv(PROCESSED_DIR / "hyperparameter_tuning_results.csv", index=False)
    print(f"Full results saved to {PROCESSED_DIR / 'hyperparameter_tuning_results.csv'}")

    print(f"\n--- Final test-set evaluation (max_depth={best_depth} only) ---")
    test_preds = best_model.predict(X_test_s)
    print(f"Test macro-F1: {f1_score(y_test, test_preds, average='macro'):.3f}")
    print(classification_report(y_test, test_preds, target_names=["non-cancerous", "cancerous"]))


if __name__ == "__main__":
    main()