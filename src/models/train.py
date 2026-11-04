import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.features.build_features import create_features


DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "artifacts/model"


def load_and_prepare_data():
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, sep="\t")

    print("Original shape:", df.shape)

    # Feature engineering
    df = create_features(df)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Convert numeric columns
    df["tenure"] = pd.to_numeric(
        df["tenure"],
        errors="coerce"
    )

    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    )

    # Remove customer ID
    df = df.drop(columns=["customerID"])

    # Convert target
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Remove rows where target is missing
    df = df.dropna(subset=["Churn"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    return preprocessor


def evaluate_model(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def train_models():

    X, y = load_and_prepare_data()

    print("\nFeatures:", X.shape[1])
    print("Target distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    preprocessor = build_preprocessor(X_train)

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=10
        )
    }

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print("\nTraining:", model_name)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        result = evaluate_model(
            pipeline,
            X_test,
            y_test,
            model_name
        )

        results.append(result)

        trained_models[model_name] = pipeline

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.sort_values(
            by="f1",
            ascending=False
        ).to_string(index=False)
    )

    # Select best model using F1-score
    best_model_name = results_df.loc[
        results_df["f1"].idxmax(),
        "model"
    ]

    best_model = trained_models[
        best_model_name
    ]

    print("\nBest Model:", best_model_name)

    print(
        "Best F1 Score:",
        round(
            results_df["f1"].max(),
            4
        )
    )

    # Create model directory
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save complete pipeline
    model_path = os.path.join(
        MODEL_DIR,
        "model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    # Save comparison results
    results_path = os.path.join(
        MODEL_DIR,
        "model_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\nModel saved to:", model_path)
    print("Results saved to:", results_path)


if __name__ == "__main__":
    train_models()