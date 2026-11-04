import pandas as pd


def create_features(df):
    """
    Create additional features for the customer churn dataset.
    """

    df = df.copy()

    # --------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["tenure"] = pd.to_numeric(
        df["tenure"],
        errors="coerce"
    )

    df["MonthlyCharges"] = pd.to_numeric(
        df["MonthlyCharges"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 1. Tenure Group
    # --------------------------------------------------

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12",
            "13-24",
            "25-48",
            "49-72"
        ]
    )

    # --------------------------------------------------
    # 2. Monthly Charge Category
    # --------------------------------------------------

    df["monthly_charge_category"] = pd.cut(
        df["MonthlyCharges"],
        bins=[-1, 35, 70, float("inf")],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    # --------------------------------------------------
    # 3. Average Monthly Spend
    # --------------------------------------------------

    df["average_monthly_spend"] = (
        df["TotalCharges"]
        / df["tenure"].replace(0, 1)
    )

    # --------------------------------------------------
    # 4. Service Count
    # --------------------------------------------------

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    existing_service_columns = [
        col
        for col in service_columns
        if col in df.columns
    ]

    if existing_service_columns:
        df["service_count"] = (
            df[existing_service_columns]
            .apply(
                lambda row: sum(
                    1
                    for value in row
                    if value == "Yes"
                ),
                axis=1
            )
        )
    else:
        df["service_count"] = 0

    return df


if __name__ == "__main__":

    DATA_PATH = (
        "data/raw/"
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        sep="\t"
    )

    print("Original shape:", df.shape)

    print("\nCreating features...")

    df = create_features(df)

    print("\nFeature engineering completed.")

    print("\nNew columns:")

    original_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    ]

    new_columns = [
        col
        for col in df.columns
        if col not in original_columns
    ]

    for column in new_columns:
        print("-", column)

    print("\nFinal shape:", df.shape)

    print("\nSample engineered data:")

    print(
        df[
            [
                "tenure_group",
                "monthly_charge_category",
                "average_monthly_spend",
                "service_count"
            ]
        ].head()
    )