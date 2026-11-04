import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data(path=RAW_DATA_PATH):
    """
    Load the Telco Customer Churn dataset.
    The Kaggle file is tab-separated.
    """
    df = pd.read_csv(path, sep="\t")

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    return df


# --------------------------------------------------
# Basic Cleaning
# --------------------------------------------------

def clean_data(df):
    """
    Perform basic data cleaning.
    """

    df = df.copy()

    # Convert TotalCharges to numeric
    # Invalid/blank values become NaN
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove customerID because it is an identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Remove rows where target is missing
    df = df.dropna(subset=["Churn"])

    return df


# --------------------------------------------------
# Build Preprocessing Pipeline
# --------------------------------------------------

def build_preprocessor(X):
    """
    Create preprocessing pipeline for numerical
    and categorical features.
    """

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
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
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# --------------------------------------------------
# Prepare Train/Test Data
# --------------------------------------------------

def prepare_data(df, test_size=0.2, random_state=42):

    df = clean_data(df)

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# Save Processed Data
# --------------------------------------------------

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test
):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    train_df = X_train.copy()
    train_df["Churn"] = y_train.values

    test_df = X_test.copy()
    test_df["Churn"] = y_test.values

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print(
        f"Training data saved to: {TRAIN_PATH}"
    )

    print(
        f"Testing data saved to: {TEST_PATH}"
    )


# --------------------------------------------------
# Main Execution
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading dataset...")

    df = load_data()

    print("Original shape:", df.shape)

    print("\nCleaning dataset...")

    df = clean_data(df)

    print("Cleaned shape:", df.shape)

    print("\nPreparing train/test data...")

    X_train, X_test, y_train, y_test = prepare_data(df)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("\nBuilding preprocessing pipeline...")

    preprocessor = build_preprocessor(X_train)

    print("Preprocessing pipeline created successfully.")

    print("\nSaving processed datasets...")

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nPreprocessing completed successfully!")