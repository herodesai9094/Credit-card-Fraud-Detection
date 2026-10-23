# Credit Card Fraud Detection with Exploratory Data Analysis (EDA)

A machine learning project that explores credit card transactions, visualizes patterns related to fraudulent transactions, and trains a **Random Forest Classifier** to identify potentially fraudulent transactions.

## Project Overview

Credit card fraud detection is a classification problem in which each transaction is classified as:
- **Not Fraud (`is_fraud = 0`)**
- **Fraud (`is_fraud = 1`)**

The notebook performs data cleaning, exploratory data analysis (EDA), feature engineering, class-imbalance handling, model training, evaluation, and model-file export.

## Features

- Cleans and prepares transaction data.
- Calculates customer age from date-of-birth and transaction-date fields.
- Extracts transaction month and year.
- Visualizes fraud distribution and patterns by gender, category, age, and transaction amount.
- Converts gender into a numeric feature and applies one-hot encoding to transaction categories.
- Splits the data into training and testing sets.
- Uses **random undersampling** on the training data to reduce class imbalance.
- Scales features with `StandardScaler`.
- Trains and evaluates a `RandomForestClassifier`.
- Saves the trained model and preprocessing information for reuse.

## Tech Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Dataset

The notebook loads a file named `fraudTrain.csv`:

```python
df = pd.read_csv("fraudTrain.csv")
```

Place the dataset in the same directory as the notebook, or update the file path in the notebook.

The notebook uses transaction fields such as transaction amount, gender, date, category, customer age, and the target column `is_fraud`. During cleaning, it renames `dob` to `age` and `amt` to `amount`, derives age and transaction month/year, and drops columns that are not used in the final model.

**Dataset note:** This repository README does not include the dataset itself. Make sure you have permission to use and redistribute the dataset before publishing it.

## Project Workflow

1. **Load data** — Read `fraudTrain.csv` into a Pandas DataFrame.
2. **Clean data** — Rename columns, combine first and last names, convert date fields, calculate age, create month/year features, and remove selected columns.
3. **EDA and visualization** — Explore fraud counts and transaction patterns using bar plots, count plots, pie charts, histograms, and scatter plots.
4. **Prepare features** — Encode gender numerically, remove selected identifier/text columns, and one-hot encode `category`.
5. **Split data** — Use `train_test_split` with a 25% test split and `random_state=42`.
6. **Handle class imbalance** — Randomly undersample the majority class in the training set.
7. **Scale features** — Fit `StandardScaler` on the undersampled training features and transform the test features.
8. **Train model** — Train a `RandomForestClassifier` with `random_state=42`.
9. **Evaluate model** — Generate a confusion matrix, classification report, accuracy score, and ROC-AUC score.
10. **Save artifacts** — Export the model, scaler, and training feature-column names using Joblib.

## Installation

Use Python 3.10+ if possible, then install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

## Run the Notebook

1. Clone or download this repository.
2. Put `fraudTrain.csv` beside the notebook (or update the dataset path).
3. Install the dependencies.
4. Launch Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `Credit Card Fraud Detection With EDA.ipynb` and run the cells from top to bottom.

## Model Evaluation

The notebook evaluates predictions on the held-out test set using:
- **Confusion matrix** — summarizes correct and incorrect predictions by class.
- **Classification report** — provides precision, recall, and F1-score.
- **Accuracy score** — reports overall accuracy.
- **ROC-AUC score** — measures how well the model separates the two classes.

The actual scores depend on the dataset and execution environment. Run the notebook to obtain the results; no metric values are claimed here.

## Saved Model Files

The notebook saves these files in the current working directory:

| File | Purpose |
|---|---|
| `fraud_model.pkl` | Trained Random Forest model |
| `scaler.pkl` | Fitted `StandardScaler` |
| `model_columns.pkl` | Feature-column names used during training |

These files can be loaded later with Joblib:

```python
import joblib

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
```

For inference, input data must use the same feature names and preprocessing steps as the training data.

## Important Notes

- The dataset is highly imbalanced in typical fraud-detection settings. Accuracy alone may therefore be misleading; review precision, recall, F1-score, the confusion matrix, and ROC-AUC.
- Undersampling discards some majority-class training examples. Results can vary depending on the sample.
- The scaler is fitted on the undersampled training features and then applied to the test features.
- This is an educational machine learning project, not a production-ready fraud-prevention system. Real-world deployment requires additional validation, monitoring, privacy safeguards, and careful handling of false positives and false negatives.

## Future Improvements

- Compare Random Forest with Logistic Regression, Gradient Boosting, and other suitable classifiers.
- Tune model hyperparameters using cross-validation.
- Evaluate precision-recall curves and select a threshold appropriate to the business use case.
- Test alternative imbalance-handling approaches.
- Build a user interface or API for transaction scoring, ensuring the same preprocessing pipeline is applied consistently.

## Author

**Your Name**

Update this section with your name, GitHub profile, or contact details before publishing.
"# Credit-card-Fraud-Detection" 
