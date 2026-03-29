from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "baseline_xgb"
ACCEPTED_BASELINE_PATH = REPO_ROOT / "experiments" / "accepted_baseline.json"
LOG_DIR = REPO_ROOT / "logs"
STEPS = ["train.py", "evaluate.py", "gate.py"]


def run_step(step_name: str) -> None:
    print(f"=== Running {step_name} ===")
    subprocess.run([sys.executable, step_name], check=True, cwd=REPO_ROOT)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compare_against_baseline(current_val_log_loss: float) -> tuple[float | None, str | None]:
    if not ACCEPTED_BASELINE_PATH.exists():
        print(f"Accepted baseline file not found: {ACCEPTED_BASELINE_PATH}")
        return None, None

    accepted_baseline = read_json(ACCEPTED_BASELINE_PATH)
    accepted_val_log_loss = accepted_baseline["val_log_loss"]

    if current_val_log_loss < accepted_val_log_loss:
        comparison_result = "BEATS"
    elif current_val_log_loss == accepted_val_log_loss:
        comparison_result = "TIES"
    else:
        comparison_result = "FAILS TO BEAT"

    print(f"Accepted baseline val_log_loss: {accepted_val_log_loss}")
    print(f"Current val_log_loss: {current_val_log_loss}")
    print(f"Comparison result: {comparison_result}")
    return accepted_val_log_loss, comparison_result


def build_recommendation(comparison_result: str | None) -> str:
    if comparison_result == "BEATS":
        return "ACCEPT_CANDIDATE"
    if comparison_result in {"TIES", "FAILS TO BEAT"}:
        return "REJECT_CANDIDATE"
    return "NO_BASELINE"


def build_log_record(
    metrics: dict,
    test_metrics: dict,
    gate_report: dict,
    timestamp: datetime,
    accepted_val_log_loss,
    comparison_result,
    recommendation: str,
) -> dict:
    return {
        "timestamp": timestamp.isoformat(),
        "status": gate_report["status"],
        "val_log_loss": metrics["log_loss"],
        "val_accuracy": metrics["accuracy"],
        "test_log_loss": test_metrics["log_loss"],
        "test_accuracy": test_metrics["accuracy"],
        "model_name": metrics["model_name"],
        "feature_names": metrics["feature_names"],
        "accepted_val_log_loss": accepted_val_log_loss,
        "comparison_result": comparison_result,
        "recommendation": recommendation,
    }


def write_log_file(log_record: dict, timestamp: datetime) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"run_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(log_record, file, indent=2)
    return log_path


def main() -> None:
    try:
        for step_name in STEPS:
            run_step(step_name)
    except subprocess.CalledProcessError:
        print(f"Pipeline failed at step: {step_name}")
        sys.exit(1)

    metrics = read_json(EXPERIMENT_DIR / "metrics.json")
    test_metrics = read_json(EXPERIMENT_DIR / "test_metrics.json")
    gate_report = read_json(EXPERIMENT_DIR / "gate_report.json")

    if gate_report.get("status") != "PASS":
        print("Pipeline failed: gate status is not PASS.")
        sys.exit(1)

    timestamp = datetime.now()
    accepted_val_log_loss, comparison_result = compare_against_baseline(metrics["log_loss"])
    recommendation = build_recommendation(comparison_result)
    log_record = build_log_record(
        metrics,
        test_metrics,
        gate_report,
        timestamp,
        accepted_val_log_loss,
        comparison_result,
        recommendation,
    )
    log_path = write_log_file(log_record, timestamp)

    print(f"Recommendation: {recommendation}")
    print(f"Saved log: {log_path}")
    print("Pipeline succeeded.")


if __name__ == "__main__":
    main()
