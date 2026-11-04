import pandas as pd

from src.data.preprocess import (
    load_data,
    clean_data,
    prepare_data,
    build_preprocessor,
)


def test_load_data():
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "customerID" in df.columns
    assert "Churn" in df.columns


def test_clean_data():
    df = load_data()
    cleaned_df = clean_data(df)

    assert "customerID" not in cleaned_df.columns
    assert cleaned_df["Churn"].isin(["Yes", "No"]).all()
    assert pd.api.types.is_numeric_dtype(cleaned_df["TotalCharges"])


def test_prepare_data():
    df = load_data()
    df = clean_data(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_build_preprocessor():
    df = load_data()
    df = clean_data(df)

    X_train, X_test, y_train, y_test = prepare_data(df)

    preprocessor = build_preprocessor(X_train)

    transformed_data = preprocessor.fit_transform(X_train)

    assert transformed_data.shape[0] == X_train.shape[0]
    assert transformed_data.shape[1] > 0