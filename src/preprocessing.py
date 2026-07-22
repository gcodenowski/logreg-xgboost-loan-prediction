import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess():
    # Section 1 - Cleaning
    # Load with income and outgoings as strings to preserve nil/sentinel values
    df = pd.read_csv(
            "data/previousApplicants.csv",
            dtype={"Income": str, "Outgoings": str},
            )

    print(f"Loaded {len(df)} rows")
    print()

    print(df.dtypes)
    print()

    print(df.head())
    print()

    # Dropping ghost rows (nil fields + sentinel outgoings)
    ghost_mask = (
     df ["ID"].isna()
     & df["Age"].isna()
     & df["Income"].isna()
     & df["Postcode"].isna()
     & df["Gender"].isna()
     & (df["Outgoings"] == "999999999999999999999999")
    )

    print(f"Ghost rows found: {ghost_mask.sum()}")
    df = df[~ghost_mask].reset_index(drop=True)
    print(f"Rows after ghost removal: {len(df)}")

    # Drop corrupted rows
    corrupted = (df["Income"] == "nil") | (df["Outgoings"] ==
                                           "999999999999999999999999")
    in_dupes = df.duplicated(subset="ID", keep=False)

    # Drop rows that are both corrupted AND in a duplicate group
    df = df[~(corrupted & in_dupes)].reset_index(drop=True)

    # Cast ID to int (for clarity)
    df["ID"] = df["ID"].astype(int)

    # Verify
    print(f"Rows after deduplication: {len(df)}")
    print(f"Duplicate IDs remaining: {df.duplicated(subset='ID').sum()}\n")

    # Replace nil strings and sentinel values with NaN for machine readability
    df["Income"] = df["Income"].replace("nil", pd.NA)
    df["Outgoings"] = df["Outgoings"].replace("999999999999999999999999", pd.NA)

    # Cast income and outgoings back to float
    df["Income"] = df["Income"].astype(float)
    df["Outgoings"] = df["Outgoings"].astype(float)

    # Print missing values
    print(df.isna().sum())

    # Section 2 - train/test split, feature choices

    # Split data into train/test before imputation (to avoid leakage)
    X = df.drop(columns=["Approved"])
    y = df["Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                    )

    # Get medians from the trainins set, impute empty values
    income_median = X_train["Income"].median()
    outgoings_median = X_train["Outgoings"].median()

    X_train["Income"] = X_train["Income"].fillna(income_median)
    X_train["Outgoings"] = X_train["Outgoings"].fillna(outgoings_median)

    # Impute test set with training medians
    X_test["Income"] = X_test["Income"].fillna(income_median)
    X_test["Outgoings"] = X_test["Outgoings"].fillna(outgoings_median)

    print()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Income median: {income_median}, Outgoings median: {outgoings_median}")

    # Drop ID (not a feature)
    X_train = X_train.drop(columns=["ID"])
    X_test = X_test.drop(columns=["ID"])

    # Drop Postcode (ethical + cardinality)
    X_train = X_train.drop(columns=["Postcode"])
    X_test = X_test.drop(columns=["Postcode"])

    # Encode gender
    gender_map = {"a": 0, "b": 1, "c": 2}
    X_train["Gender"] = X_train["Gender"].map(gender_map)
    X_test["Gender"] = X_test["Gender"].map(gender_map)

    # Encode Approved (target)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Section 3 - Normalisation

    # Standardise numeric features, applied to shared output
    # XGBoost is invariant to scaling
    scaler = StandardScaler()

    numeric_cols = ["Age", "Income", "Outgoings"]
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Summary
    print("\n=== Summary ===")
    print(f"Training set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    print(f"\nFeature columns: {list(X_train.columns)}")

    print("\nClass distribution (train):")
    print(y_train.value_counts(normalize=True))
    print("\nClass distribution (test):")
    print(y_test.value_counts(normalize=True))

    # Verify there's no NaN's left
    assert X_train.isna().sum().sum() == 0, "NaN values in training features"
    assert X_test.isna().sum().sum() == 0, "NaN values in test features"

    assert y_train.isna().sum() == 0, "NaN values in training target"
    assert y_test.isna().sum() == 0, "NaN values in test target"

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()
