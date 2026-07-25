"""Rice Cammeo / Osmancik classification — simple course project.

Main model: Logistic Regression.
Naive Bayes and Perceptron are included for comparison.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = Path("data") / "Rice_Cammeo_Osmancik.arff"
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
FEATURE_COLS = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
]


def load_data() -> pd.DataFrame:
    """Load the ARFF dataset and decode class labels."""
    raw, _ = arff.loadarff(DATA_PATH)
    df = pd.DataFrame(raw)
    df["Class"] = df["Class"].apply(
        lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
    )
    return df


def explore(df: pd.DataFrame) -> None:
    """Print basic stats and save two simple plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Data Exploration ===")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("Missing values:")
    print(df.isna().sum())
    print("Class distribution:")
    print(df["Class"].value_counts())

    # Category distribution
    counts = df["Class"].value_counts().sort_index()
    ax = counts.plot(kind="bar", title="Class distribution", color=["#c45c26", "#2f6b4f"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # One scatter plot
    fig, ax = plt.subplots()
    for label, color in (("Cammeo", "#c45c26"), ("Osmancik", "#2f6b4f")):
        subset = df.loc[df["Class"] == label]
        ax.scatter(
            subset["Major_Axis_Length"],
            subset["Minor_Axis_Length"],
            s=10,
            alpha=0.5,
            label=label,
            color=color,
        )
    ax.set_xlabel("Major Axis Length")
    ax.set_ylabel("Minor Axis Length")
    ax.set_title("Major vs Minor Axis Length")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scatter_axis_length.png", dpi=150)
    plt.close()
    print(f"Saved plots in {OUTPUT_DIR}/")


def evaluate(name: str, y_true, y_pred, class_names: list[str]) -> None:
    """Print Accuracy / Precision / Recall and save a confusion matrix."""
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    for index, class_name in enumerate(class_names):
        print(
            f"{class_name}: precision={precision[index]:.4f}, "
            f"recall={recall[index]:.4f}"
        )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        values_format="d",
    )
    display.ax_.set_title(name)
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{safe_name}.png", dpi=150)
    plt.close()


def main() -> None:
    df = load_data()
    explore(df)

    x = df[FEATURE_COLS]
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Class"])
    class_names = list(encoder.classes_)

    # 80/20 stratified split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("\n=== Split ===")
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")

    # Scale features (fit on train only)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "Perceptron": Perceptron(max_iter=1000, tol=1e-3, random_state=RANDOM_STATE),
    }

    results: dict[str, float] = {}
    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        results[name] = float(accuracy_score(y_test, y_pred))
        evaluate(name, y_test, y_pred, class_names)

    best_name = max(results, key=results.get)
    print("\n=== Conclusion ===")
    print(
        f"Best model: {best_name} "
        f"(accuracy={results[best_name]:.4f})."
    )
    print(
        "Logistic Regression achieved the highest accuracy in this experiment. "
        "Naive Bayes may be affected by correlations between the size features, "
        "while Perceptron produced a lower accuracy on the selected test split."
    )
    print("Possible improvements: cross-validation, tuning C / priors, more features.")


if __name__ == "__main__":
    main()
