
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

FEATURES = ["area", "eccentricity", "solidity", "perimeter", "mean_intensity", "texture_contrast"]


def main():
    df = pd.read_csv(PROCESSED_DIR / "train_split_labels.csv").dropna(subset=FEATURES)
    X = StandardScaler().fit_transform(df[FEATURES])

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    df["pc1"], df["pc2"] = components[:, 0], components[:, 1]

    print(f"Variance explained by 2 components: {pca.explained_variance_ratio_.sum():.1%}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: colored by cancerous / non-cancerous
    fig, ax = plt.subplots(figsize=(7, 6))
    for value, label, color in [(0, "Non-cancerous", "#2DE2FF"), (1, "Cancerous", "#FF3B5C")]:
        subset = df[df["is_cancerous"] == value]
        ax.scatter(subset["pc1"], subset["pc2"], s=6, alpha=0.4, label=label, color=color)
    ax.set_title("PCA of morphological features — cancerous vs. non-cancerous")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend()
    fig.savefig(FIGURES_DIR / "pca_iscancerous.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'pca_iscancerous.png'}")

    # Plot 2: colored by cell type
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = {"Neoplastic": "#FF3B5C", "Inflammatory": "#FFB020", "Connective": "#2DE2FF",
              "Dead": "#9A6BFF", "Epithelial": "#39FF8A"}
    for category, color in colors.items():
        subset = df[df["category"] == category]
        ax.scatter(subset["pc1"], subset["pc2"], s=6, alpha=0.4, label=category, color=color)
    ax.set_title("PCA of morphological features — by cell type")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend()
    fig.savefig(FIGURES_DIR / "pca_celltype.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / 'pca_celltype.png'}")

    print("\nDone. Use these two PNGs in your report's EDA section.")


if __name__ == "__main__":
    main()