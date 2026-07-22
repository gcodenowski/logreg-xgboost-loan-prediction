import pandas as pd

# Load with income and outgoings as strings to preserne nil/sentinel values
df = pd.read_csv(
        "data/previousApplicants.csv",
        dtype={"Income": str, "Outgoings": str},
        )

print(f"Loaded {len(df)} rows")
print(df.dtypes)
print(df.head())
