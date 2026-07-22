import pandas as pd

# Load with income and outgoings as strings to preserve nil/sentinel values
df = pd.read_csv(
        "data/previousApplicants.csv",
        dtype={"Income": str, "Outgoings": str},
        )

print(f"Loaded {len(df)} rows")
print(df.dtypes)
print() #newline

print(df.head())
print() #newline

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

# Verify
print(f"Duplicate IDs remaining: {df.duplicated(subset='ID').sum()}")

# Replace nil strings and sentinel values with NaN for machine readability
df["Income"] = df["Income"].replace("nil", pd.NA)
df["Outgoings"] = df["Income"].replace("999999999999999999999999", pd.NA)

# Cast income and outgoings to float
