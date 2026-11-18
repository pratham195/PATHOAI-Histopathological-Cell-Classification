from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

FEATURES = ["area", "eccentricity", "solidity", "perimeter", "mean_intensity", "texture_contrast"]


def load(name):
    df = pd.read_csv(PROCESSED_DIR / f"{name}_split_labels.csv")
    return df.dropna(subset=FEATURES)


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }


def main():
    train, val, test = load("train"), load("val"), load("test")

    label_encoder = LabelEncoder()
    label_encoder.fit(train["category"])
    class_names = list(label_encoder.classes_)

    X_train, y_train = train[FEATURES], label_encoder.transform(train["category"])
    X_val, y_val = val[FEATURES], label_encoder.transform(val["category"])
    X_test, y_test = test[FEATURES], label_encoder.transform(test["category"])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    print(f"Classes: {class_names}")
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}\n")

    models = build_models()
    val_scores, trained = {}, {}

    for name, model in models.items():
        model.fit(X_train_s, y_train, sample_weight=sample_weight)
        preds = model.predict(X_val_s)
        macro_f1 = f1_score(y_val, preds, average="macro")
        val_scores[name] = macro_f1
        trained[name] = model
        print(f"{name}: validation macro-F1 = {macro_f1:.3f}")

    winner_name = max(val_scores, key=val_scores.get)
    winner = trained[winner_name]
    print(f"\nBest on validation: {winner_name} (macro-F1 = {val_scores[winner_name]:.3f})")

    print(f"\n--- Final test-set evaluation ({winner_name} only) ---")
    test_preds = winner.predict(X_test_s)
    print(f"Test macro-F1: {f1_score(y_test, test_preds, average='macro'):.3f}")
    print(classification_report(y_test, test_preds, target_names=class_names))

    print("Confusion matrix (rows = actual, columns = predicted):")
    cm = pd.DataFrame(confusion_matrix(y_test, test_preds), index=class_names, columns=class_names)
    print(cm)

    comparison = pd.DataFrame({
        "model": list(val_scores.keys()),
        "validation_macro_f1": list(val_scores.values()),
    }).sort_values("validation_macro_f1", ascending=False)
    comparison.to_csv(PROCESSED_DIR / "celltype_model_comparison.csv", index=False)
    print(f"\nComparison table saved to {PROCESSED_DIR / 'celltype_model_comparison.csv'}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": winner, "scaler": scaler, "features": FEATURES, "label_encoder": label_encoder, "model_name": winner_name},
        MODELS_DIR / "celltype_classical.joblib",
    )
    print(f"Winning model ({winner_name}) saved to {MODELS_DIR / 'celltype_classical.joblib'}")


if __name__ == "__main__":
    main()