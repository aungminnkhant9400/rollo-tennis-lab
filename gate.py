from pathlib import Path
import json
import math


ARTIFACT_DIR = Path("experiments/baseline_xgb")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
VAL_PREDICTIONS_PATH = ARTIFACT_DIR / "val_predictions.csv"
TEST_PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"
VAL_METRICS_PATH = ARTIFACT_DIR / "metrics.json"
TEST_METRICS_PATH = ARTIFACT_DIR / "test_metrics.json"
GATE_REPORT_PATH = ARTIFACT_DIR / "gate_report.json"

VAL_REQUIRED_KEYS = [
    "model_name",
    "n_train_rows",
    "n_val_rows",
    "log_loss",
    "accuracy",
    "feature_names",
]
TEST_REQUIRED_KEYS = [
    "model_name",
    "n_test_rows",
    "log_loss",
    "accuracy",
    "feature_names",
]


def add_check(checks: list, name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": passed, "detail": detail})


def check_file_exists(path: Path, checks: list) -> bool:
    exists = path.exists()
    detail = "found" if exists else f"missing: {path}"
    add_check(checks, f"file_exists:{path.name}", exists, detail)
    return exists


def load_json(path: Path, checks: list):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        add_check(checks, f"json_parse:{path.name}", False, f"invalid JSON: {error}")
        return None

    add_check(checks, f"json_parse:{path.name}", True, "parsed successfully")
    return data


def require_keys(data: dict, required_keys: list, label: str, checks: list) -> bool:
    missing_keys = [key for key in required_keys if key not in data]
    passed = not missing_keys
    detail = "all required keys present" if passed else f"missing keys: {', '.join(missing_keys)}"
    add_check(checks, f"required_keys:{label}", passed, detail)
    return passed


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def validate_positive_count(data: dict, key: str, label: str, checks: list) -> None:
    value = data.get(key)
    passed = isinstance(value, int) and value > 0
    detail = f"{key}={value}" if passed else f"{key} must be an integer > 0, got {value}"
    add_check(checks, f"metric:{label}:{key}", passed, detail)


def validate_log_loss(data: dict, label: str, checks: list) -> None:
    value = data.get("log_loss")
    passed = is_finite_number(value) and value >= 0
    detail = f"log_loss={value}" if passed else f"log_loss must be finite and >= 0, got {value}"
    add_check(checks, f"metric:{label}:log_loss", passed, detail)


def validate_accuracy(data: dict, label: str, checks: list) -> None:
    value = data.get("accuracy")
    passed = is_finite_number(value) and 0 <= value <= 1
    detail = f"accuracy={value}" if passed else f"accuracy must be finite and in [0, 1], got {value}"
    add_check(checks, f"metric:{label}:accuracy", passed, detail)


def validate_feature_names(data: dict, label: str, checks: list) -> None:
    value = data.get("feature_names")
    passed = isinstance(value, list) and len(value) > 0
    detail = f"feature_names count={len(value)}" if passed else f"feature_names must be a non-empty list, got {value}"
    add_check(checks, f"metric:{label}:feature_names", passed, detail)


def validate_metrics(data: dict, required_keys: list, label: str, checks: list) -> None:
    if data is None:
        add_check(checks, f"metrics_present:{label}", False, "metrics payload unavailable")
        return

    if not require_keys(data, required_keys, label, checks):
        return

    if label == "validation":
        validate_positive_count(data, "n_train_rows", label, checks)
        validate_positive_count(data, "n_val_rows", label, checks)
    if label == "test":
        validate_positive_count(data, "n_test_rows", label, checks)

    validate_log_loss(data, label, checks)
    validate_accuracy(data, label, checks)
    validate_feature_names(data, label, checks)


def build_report(checks: list, val_metrics, test_metrics) -> dict:
    status = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    return {
        "status": status,
        "checks": checks,
        "val_log_loss": None if val_metrics is None else val_metrics.get("log_loss"),
        "val_accuracy": None if val_metrics is None else val_metrics.get("accuracy"),
        "test_log_loss": None if test_metrics is None else test_metrics.get("log_loss"),
        "test_accuracy": None if test_metrics is None else test_metrics.get("accuracy"),
    }


def save_report(report: dict) -> None:
    GATE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GATE_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def format_metric(value) -> str:
    if is_finite_number(value):
        return f"{value:.6f}"
    return "n/a"


def print_summary(report: dict) -> None:
    print(f"Status: {report['status']}")
    print(
        "Validation metrics: "
        f"log_loss={format_metric(report['val_log_loss'])}, "
        f"accuracy={format_metric(report['val_accuracy'])}"
    )
    print(
        "Test metrics: "
        f"log_loss={format_metric(report['test_log_loss'])}, "
        f"accuracy={format_metric(report['test_accuracy'])}"
    )


def main() -> None:
    checks = []

    required_paths = [
        MODEL_PATH,
        VAL_PREDICTIONS_PATH,
        TEST_PREDICTIONS_PATH,
        VAL_METRICS_PATH,
        TEST_METRICS_PATH,
    ]
    for path in required_paths:
        check_file_exists(path, checks)

    val_metrics = load_json(VAL_METRICS_PATH, checks)
    test_metrics = load_json(TEST_METRICS_PATH, checks)

    validate_metrics(val_metrics, VAL_REQUIRED_KEYS, "validation", checks)
    validate_metrics(test_metrics, TEST_REQUIRED_KEYS, "test", checks)

    report = build_report(checks, val_metrics, test_metrics)
    save_report(report)
    print_summary(report)


if __name__ == "__main__":
    main()
