"""
Runs THREE models on every uploaded image and reports all three opinions
honestly, rather than hiding disagreement between them:

  1. Classical ML (Random Forest or similar) on 6 hand-measured features
  2. CNN — isCancerous (binary), trained directly on pixels
  3. CNN — cell type (5-class), trained directly on pixels

The two CNNs are trained independently, so nothing guarantees they agree
with each other, or with the classical model. Disagreement is reported
explicitly rather than hidden.

Explanations describe the actual measured features (shape, border,
staining, texture) plainly — they state what was measured, not a claim
about what's "typical" for a category, since a canned claim can end up
contradicting the real numbers for a given image. This is supporting
context, not a readout of what the CNNs internally weighted (a CNN's real
reasoning lives in its pixel-level weights — Grad-CAM, planned next, is
the correct tool for that).
"""
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "ml-pipeline" / "models"
ISCANCEROUS_PATH = MODELS_DIR / "cnn_iscancerous.pt"
CELLTYPE_PATH = MODELS_DIR / "cnn_celltype.pt"
LABEL_ENCODER_PATH = MODELS_DIR / "celltype_label_encoder.joblib"
CLASSICAL_PATH = MODELS_DIR / "iscancerous_rf.joblib"

_iscancerous_model = None
_celltype_model = None
_label_encoder = None
_classical_bundle = None
_transform = transforms.Compose([transforms.ToTensor()])


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
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


def load_iscancerous_model():
    global _iscancerous_model
    if _iscancerous_model is None:
        if not ISCANCEROUS_PATH.exists():
            raise FileNotFoundError(f"No trained model at {ISCANCEROUS_PATH}. Run 06_train_cnn.py first.")
        model = SmallCNN(num_classes=2)
        model.load_state_dict(torch.load(ISCANCEROUS_PATH, map_location="cpu", weights_only=True))
        model.eval()
        _iscancerous_model = model
    return _iscancerous_model


def load_celltype_model():
    global _celltype_model, _label_encoder
    if _celltype_model is None:
        if not CELLTYPE_PATH.exists():
            raise FileNotFoundError(f"No trained model at {CELLTYPE_PATH}. Run 07_train_cnn_celltype.py first.")
        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(f"No label encoder at {LABEL_ENCODER_PATH}. Run 07_train_cnn_celltype.py first.")
        model = SmallCNN(num_classes=5)
        model.load_state_dict(torch.load(CELLTYPE_PATH, map_location="cpu", weights_only=True))
        model.eval()
        _celltype_model = model
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return _celltype_model, _label_encoder


def load_classical_bundle():
    global _classical_bundle
    if _classical_bundle is None:
        if not CLASSICAL_PATH.exists():
            raise FileNotFoundError(f"No classical model at {CLASSICAL_PATH}. Run 04_train_cancer_model.py first.")
        _classical_bundle = joblib.load(CLASSICAL_PATH)
    return _classical_bundle


def estimate_nucleus_mask(gray):
    """Otsu threshold + pick the connected component closest to center.
    An approximation — see module docstring for the honesty note on this."""
    if gray.min() == gray.max():
        return np.zeros_like(gray, dtype=bool)  # flat image: nothing to measure

    thresh = threshold_otsu(gray)
    binary = gray <= thresh  # <= so the darkest level isn't dropped
    if not binary.any():
        binary = gray < gray.mean()  # fallback if Otsu still selects nothing

    lbl = label(binary)
    props = regionprops(lbl)
    if not props:
        return binary
    center = np.array(gray.shape) / 2
    best = min(props, key=lambda p: np.linalg.norm(np.array(p.centroid) - center))
    return lbl == best.label


def measure_features(rgb_array):
    gray = rgb_array.mean(axis=-1).astype(np.uint8)
    mask = estimate_nucleus_mask(gray)
    lbl = label(mask)
    props = regionprops(lbl)
    if not props:
        return None
    p = max(props, key=lambda r: r.area)

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]

    return {
        "area": float(p.area),
        "eccentricity": float(p.eccentricity),
        "solidity": float(p.solidity),
        "perimeter": float(p.perimeter),
        "mean_intensity": float(gray[mask].mean()) if mask.any() else float("nan"),
        "texture_contrast": float(contrast),
    }


def describe_shape(features):
    """Plain description of what was actually measured — no claims about
    what's 'typical' for any category, so this can never contradict the
    real numbers for a given image."""
    shape = "an elongated, irregular shape" if features["eccentricity"] > 0.7 else "a round, regular shape"
    border = "a jagged border" if features["solidity"] < 0.85 else "a smooth border"
    staining = "darker staining" if features["mean_intensity"] < 100 else "lighter staining"
    texture = "a coarse, rough texture" if features["texture_contrast"] > 100 else "a fine, uniform texture"
    return shape, border, staining, texture


def explain_iscancerous(cnn_verdict, rf_verdict, features):
    shape, border, staining, texture = describe_shape(features)
    base = (
        f"The CNN classified this as {'cancerous' if cnn_verdict else 'non-cancerous'}. For reference, "
        f"this nucleus measured {shape}, {border}, {staining}, and {texture} — supporting context "
        f"computed from the image, not the CNN's literal internal reasoning, which operates on raw pixels."
    )
    if rf_verdict is None:
        return base
    if cnn_verdict == rf_verdict:
        agreement = f"The classical model agrees ({'cancerous' if rf_verdict else 'non-cancerous'})."
    else:
        agreement = (
            f"Note: the classical model disagrees, predicting "
            f"{'cancerous' if rf_verdict else 'non-cancerous'} instead — the two models don't always "
            f"align, which is worth treating as a signal of genuine ambiguity in borderline cases."
        )
    return f"{base} {agreement}"


def explain_celltype(predicted_type, features):
    shape, border, staining, texture = describe_shape(features)
    return (
        f"The cell-type CNN's top prediction is {predicted_type}. For reference, this nucleus measured "
        f"{shape}, {border}, {staining}, and {texture}. These are supporting measurements computed from "
        f"the image, offered as context — not a direct readout of what the CNN internally weighted, "
        f"which operates on raw pixels rather than these specific numbers."
    )


def predict(image):
    image = image.convert("RGB").resize((64, 64))
    rgb_array = np.array(image)
    tensor = _transform(image).unsqueeze(0)

    features = measure_features(rgb_array)

    iscancerous_model = load_iscancerous_model()
    with torch.no_grad():
        probs1 = torch.softmax(iscancerous_model(tensor), dim=1)[0]
    cnn_is_cancerous = bool(probs1.argmax() == 1)
    cnn_cancer_confidence = float(probs1[int(probs1.argmax())])

    celltype_model, label_encoder = load_celltype_model()
    with torch.no_grad():
        probs2 = torch.softmax(celltype_model(tensor), dim=1)[0]
    predicted_idx = int(probs2.argmax())
    predicted_type = label_encoder.inverse_transform([predicted_idx])[0]
    type_confidence = float(probs2[predicted_idx])
    all_probs = {label_encoder.inverse_transform([i])[0]: round(float(probs2[i]), 3) for i in range(len(probs2))}

    # Random Forest. Wrapped in try/except so a problem here never takes down
    # the whole response; the error is shown under the RF row in the UI instead.
    rf_is_cancerous, rf_confidence, rf_model_name, rf_error = None, None, None, None
    if features is not None:
        try:
            bundle = load_classical_bundle()
            feature_names = (
                bundle.get("feature_cols")
                or bundle.get("features")
                or bundle.get("feature_names")
                or bundle.get("columns")
            )
            if feature_names is None:
                raise KeyError(f"feature list not found in bundle (keys: {list(bundle.keys())})")

            X = np.array([[features[c] for c in feature_names]])
            X_scaled = bundle["scaler"].transform(X)
            rf_pred = bundle["model"].predict(X_scaled)[0]
            rf_proba = bundle["model"].predict_proba(X_scaled)[0]
            rf_is_cancerous = bool(rf_pred)
            rf_confidence = float(rf_proba[1] if rf_is_cancerous else rf_proba[0])
            rf_model_name = bundle.get("model_name", "Random Forest")
        except Exception as exc:
            rf_error = f"The Random Forest could not run: {exc}"
            print("RF error:", repr(exc))

    cnn_label = "cancerous" if cnn_is_cancerous else "non-cancerous"

    result = {
        "is_cancerous": cnn_is_cancerous,
        "confidence": round(cnn_cancer_confidence, 3),
        "explanation": (
            explain_iscancerous(cnn_is_cancerous, rf_is_cancerous, features)
            if features
            else f"The CNN classified this as {cnn_label}."
        ),
        "cell_type": {
            "predicted": predicted_type,
            "confidence": round(type_confidence, 3),
            "probabilities": all_probs,
            "explanation": explain_celltype(predicted_type, features) if features else None,
        },
        "classical_model": {
            "model_name": rf_model_name,
            "is_cancerous": rf_is_cancerous,
            "confidence": round(rf_confidence, 3) if rf_confidence is not None else None,
            "agrees_with_cnn": (rf_is_cancerous == cnn_is_cancerous) if rf_is_cancerous is not None else None,
            "explanation": rf_error or (None if features else "Could not measure supporting features for this image."),
        },
        "features": {k: round(v, 3) for k, v in features.items()} if features else {},
    }
    return result