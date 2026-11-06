"""
Decode Labs - Project 2: Data Classification Using AI
Iris flower classification using supervised learning and K-Nearest Neighbors (KNN).

Run:
    python main.py

The Iris dataset is loaded from scikit-learn, so no separate dataset download is needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_and_explore_data():
    """Load the Iris dataset and display a brief overview."""
    iris = load_iris(as_frame=True)
    dataframe = iris.frame.copy()
    dataframe["species"] = dataframe["target"].map(
        dict(enumerate(iris.target_names))
    )

    print("\n" + "=" * 65)
    print("DECODE LABS - DATA CLASSIFICATION USING AI")
    print("=" * 65)
    print("\nFirst five rows:")
    print(dataframe.head().to_string(index=False))
    print(f"\nDataset shape: {dataframe.shape}")
    print("\nFeature columns:")
    print(list(iris.feature_names))
    print("\nClass distribution:")
    print(dataframe["species"].value_counts().to_string())
    print("\nMissing values:")
    print(dataframe.isnull().sum().to_string())
    print("\nStatistical summary:")
    print(dataframe[iris.feature_names].describe().round(2).to_string())

    return iris, dataframe


def create_visualizations(dataframe, feature_names):
    """Save a class-distribution chart for the dataset."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=dataframe, x="species", order=[
        "setosa", "versicolor", "virginica"
    ])
    plt.title("Iris Dataset: Number of Samples per Species")
    plt.xlabel("Iris species")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    output_path = RESULTS_DIR / "class_distribution.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nSaved class distribution chart: {output_path}")


def train_and_evaluate(iris):
    """Split data, train a scaled KNN model, and evaluate its predictions."""
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Scaling is fitted only on the training data through this Pipeline.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    weighted_f1 = f1_score(y_test, predictions, average="weighted")
    matrix = confusion_matrix(y_test, predictions)

    print("\n" + "-" * 65)
    print("MODEL EVALUATION")
    print("-" * 65)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Accuracy:         {accuracy:.2%}")
    print(f"Weighted F1 score:{weighted_f1: .4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, target_names=iris.target_names
    ))

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )
    plt.title("KNN Confusion Matrix")
    plt.xlabel("Predicted species")
    plt.ylabel("Actual species")
    plt.tight_layout()
    matrix_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(matrix_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {matrix_path}")

    return model, iris


def predict_custom_flower(model, iris):
    """Ask the user for four measurements and predict the flower species."""
    print("\n" + "-" * 65)
    print("TRY A FLOWER PREDICTION")
    print("Enter measurements in centimeters, or press Enter to skip.")
    print("-" * 65)

    measurements = []
    prompts = [
        "Sepal length (cm): ",
        "Sepal width (cm): ",
        "Petal length (cm): ",
        "Petal width (cm): ",
    ]

    try:
        for prompt in prompts:
            value = input(prompt).strip()
            if value == "":
                print("Prediction skipped.")
                return
            number = float(value)
            if number <= 0:
                raise ValueError("Measurements must be greater than zero.")
            measurements.append(number)

        prediction = model.predict([measurements])[0]
        predicted_species = iris.target_names[prediction]
        print(f"\nPredicted Iris species: {predicted_species.title()}")

    except ValueError as error:
        print(f"\nInvalid input: {error}")
        print("Please run the program again and enter four positive numbers.")


def main():
    iris, dataframe = load_and_explore_data()
    create_visualizations(dataframe, iris.feature_names)
    model, iris = train_and_evaluate(iris)
    predict_custom_flower(model, iris)
    print("\nProject completed. Thank you for using PyBot's Iris classifier!")


if __name__ == "__main__":
    main()
