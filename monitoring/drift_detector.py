import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_PATH = Path("data/loan_default.csv")
LOG_PATH = Path("data/prediction_logs.jsonl")
OUTPUT_PATH = Path("data/drift_report.json")

SIGNIFICANCE_LEVEL = 0.05
MIN_CURRENT_ROWS = 30

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
            if not line.strip():
                continue
            event = json.loads(line)
            if isinstance(event.get("features"), dict):
                records.append(event["features"])

    current = pd.DataFrame(records)

    if current.empty:
        raise ValueError("No valid prediction feature records found.")

    insufficient_sample = len(current) < MIN_CURRENT_ROWS
    results = {}

    for feature in FEATURES:
        if feature not in reference.columns or feature not in current.columns:
            results[feature] = {
                "ks_statistic": None,
                "p_value": None,
                "drift_detected": None,
                "status": "missing_feature",
            }
            continue

        reference_values = pd.to_numeric(
            reference[feature], errors="coerce"
        ).dropna()
        current_values = pd.to_numeric(
            current[feature], errors="coerce"
        ).dropna()

        if reference_values.empty or current_values.empty:
            results[feature] = {
                "ks_statistic": None,
                "p_value": None,
                "drift_detected": None,
                "status": "insufficient_valid_data",
            }
            continue

        if insufficient_sample:
            results[feature] = {
                "ks_statistic": None,
                "p_value": None,
                "drift_detected": None,
                "status": "insufficient_sample_size",
            }
            continue

        statistic, p_value = ks_2samp(reference_values, current_values)
        drift_detected = bool(p_value < SIGNIFICANCE_LEVEL)

        results[feature] = {
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
            "drift_detected": drift_detected,
            "status": (
                "drift_detected" if drift_detected else "no_drift_detected"
            ),
        }

    report = {
        "method": "Kolmogorov-Smirnov two-sample test",
        "significance_level": SIGNIFICANCE_LEVEL,
        "minimum_current_rows": MIN_CURRENT_ROWS,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "interpretation": (
            f"Drift is evaluated only when at least {MIN_CURRENT_ROWS} "
            "prediction records are available. Below that threshold, "
            "feature results are marked insufficient_sample_size. "
            "This threshold is a safeguard, not a guarantee of statistical power."
        ),
        "features": results,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nDrift report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
