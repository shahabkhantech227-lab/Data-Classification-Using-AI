# Data Classification Using AI – Iris Flower Classification

**Decode Labs | Artificial Intelligence Internship | Project 2**

## Project overview

This project builds a supervised machine-learning classification model that predicts an Iris flower species from four measurements. It uses the Iris dataset and the K-Nearest Neighbors (KNN) algorithm.

The dataset contains 150 samples, three species (Setosa, Versicolor, and Virginica), and four features:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Project workflow

1. Load and explore the Iris dataset.
2. Inspect the first five rows, dataset dimensions, class counts, missing values, and descriptive statistics.
3. Separate the four input features (`X`) from the target (`y`).
4. Split the data into 80% training and 20% testing sets.
5. Scale features with `StandardScaler` inside a scikit-learn pipeline.
6. Train a KNN classifier with `k=5`.
7. Evaluate the model with accuracy, a classification report, weighted F1 score, and a confusion matrix.
8. Ask for four measurements and predict a species.

The split uses `random_state=42` and stratification to keep the class proportions similar in both sets. The scaler is fitted as part of the training pipeline to avoid using test-set information during training.

## Technologies

- Python 3.9 or later
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Folder structure

```text
Data-Classification-Using-AI/
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── results/
│   ├── class_distribution.png
│   └── confusion_matrix.png
└── screenshots/
    └── (add your project screenshots here)
```

The two result images are generated when you run the program. They are not included until then.

## Installation and execution

### 1. Install Python

Install Python 3.9 or later from [python.org](https://www.python.org/downloads/).

### 2. Open the project folder in a terminal

```bash
cd Data-Classification-Using-AI
```

### 3. (Optional) Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the project

```bash
python main.py
```

The Iris dataset is provided by scikit-learn, so you do not need to download a separate CSV file. The program saves charts in the `results/` folder.

## Example prediction input

```text
Sepal length (cm): 5.1
Sepal width (cm): 3.5
Petal length (cm): 1.4
Petal width (cm): 0.2

Predicted Iris species: Setosa
```

Model metrics are calculated when the program runs. Exact results are printed in the terminal rather than hard-coded in this README.

## Important notes

- This is a small educational demonstration, not a production flower-identification system.
- KNN is a supervised learning algorithm: it learns from labelled examples and classifies a new sample based on nearby training samples.
- The model uses standardized features because KNN relies on distances between observations.

## Possible future improvements

- Compare KNN with Decision Tree, Logistic Regression, and Support Vector Machine models.
- Tune the number of neighbors using cross-validation.
- Add a graphical user interface.
- Save and reload the trained model.
- Add more visual analysis of feature relationships.

## Author

Developed as an educational project for the Decode Labs Artificial Intelligence internship.

## License

This project is provided for educational and learning purposes.
