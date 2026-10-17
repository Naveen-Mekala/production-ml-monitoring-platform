import json

import pandas as pd

from monitoring import drift_detector


def test_drift_detector_marks_single_record_as_insufficient(
    tmp_path, monkeypatch
):
    reference_path = tmp_path / "reference.csv"
    log_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "drift_report.json"

    pd.DataFrame({
        feature: [1, 2, 3]
        for feature in drift_detector.FEATURES
    }).to_csv(reference_path, index=False)

    log_path.write_text(
        json.dumps({
            "features": {
                feature: 1 for feature in drift_detector.FEATURES
            }
        }) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(drift_detector, "REFERENCE_PATH", reference_path)
    monkeypatch.setattr(drift_detector, "LOG_PATH", log_path)
    monkeypatch.setattr(drift_detector, "OUTPUT_PATH", output_path)

    drift_detector.main()

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["current_rows"] == 1

    for result in report["features"].values():
        assert result["status"] == "insufficient_sample_size"
        assert result["drift_detected"] is None
        assert result["p_value"] is None


def test_drift_detector_generates_report(tmp_path, monkeypatch):
    reference_path = tmp_path / "reference.csv"
    log_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "drift_report.json"

    pd.DataFrame({
        feature: list(range(100))
        for feature in drift_detector.FEATURES
    }).to_csv(reference_path, index=False)

    records = [
        {
            "features": {
                feature: value
                for feature in drift_detector.FEATURES
            }
        }
        for value in range(10)
    ]

    log_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(drift_detector, "REFERENCE_PATH", reference_path)
    monkeypatch.setattr(drift_detector, "LOG_PATH", log_path)
    monkeypatch.setattr(drift_detector, "OUTPUT_PATH", output_path)

    drift_detector.main()

    assert output_path.exists()
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["reference_rows"] == 100
    assert report["current_rows"] == 10
    assert report["minimum_current_rows"] == 30
    assert set(report["features"]) == set(drift_detector.FEATURES)

    for result in report["features"].values():
        assert result["status"] == "insufficient_sample_size"
        assert result["drift_detected"] is None
        assert result["p_value"] is None
