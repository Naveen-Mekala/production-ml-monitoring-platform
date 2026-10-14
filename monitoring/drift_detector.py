import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_PATH = Path("data/loan_default.csv")
LOG_PATH = Path("data/prediction_logs.jsonl")
OUTPUT_PATH = Path("data/drift_report.json")

FEATURES = [
    "income",
    "credit_score",
    "debt_to_income",
    "loan_amount",
    "employment_years",
    "interest_rate",
]


def main():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_PATH}")

    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Prediction log not found: {LOG_PATH}")

    reference = pd.read_csv(REFERENCE_PATH)

    records = []
    with LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                event = json.loads(line)
                records.append(event["features"])

    current = pd.DataFrame(records)

    if len(current) < 2:
        raise ValueError(
            "At least 2 prediction records are needed for drift analysis."
        )

    results = {}

    for feature in FEATURES:
        reference_values = pd.to_numeric(
            reference[feature], errors="coerce"
        ).dropna()
        current_values = pd.to_numeric(
            current[feature], errors="coerce"
        ).dropna()

        if current_values.empty or reference_values.empty:
            results[feature] = {
                "ks_statistic": None,
                "p_value": None,
                "drift_detected": None,
                "status": "insufficient_valid_data",
            }
            continue

        statistic, p_value = ks_2samp(
            reference_values,
            current_values,
        )

        results[feature] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": bool(p_value < 0.05),
            "status": "drift_detected" if p_value < 0.05 else "no_drift_detected",
        }

    report = {
        "method": "Kolmogorov-Smirnov two-sample test",
        "significance_level": 0.05,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "interpretation": (
            "Drift is flagged when the KS-test p-value is below 0.05. "
            "Small current samples can produce unreliable results."
        ),
        "features": results,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nDrift report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
