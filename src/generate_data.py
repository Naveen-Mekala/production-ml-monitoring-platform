from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

RANDOM_STATE = 42
N_SAMPLES = 5000

FEATURE_NAMES = [
    "income",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "employment_years",
    "interest_rate",
]

def main():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=len(FEATURE_NAMES),
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        weights=[0.8, 0.2],
        class_sep=0.9,
        random_state=RANDOM_STATE,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    # Transform standardized values into realistic-looking numeric ranges.
    df["income"] = (45000 + df["income"] * 15000).clip(15000, 200000).round(2)
    df["credit_score"] = (680 + df["credit_score"] * 55).clip(300, 850).round().astype(int)
    df["debt_to_income"] = (0.30 + df["debt_to_income"] * 0.12).clip(0.01, 0.80).round(3)
    df["loan_amount"] = (20000 + df["loan_amount"] * 12000).clip(1000, 150000).round(2)
    df["employment_years"] = (7 + df["employment_years"] * 4).clip(0, 45).round(1)
    df["interest_rate"] = (9 + df["interest_rate"] * 2).clip(1, 30).round(2)

    df["default"] = y.astype(int)

    output_path = Path("data/loan_default.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nDefault distribution:")
    print(df["default"].value_counts(normalize=True).round(3))

    print("\nFirst five rows:")
    print(df.head())

if __name__ == "__main__":
    main()
