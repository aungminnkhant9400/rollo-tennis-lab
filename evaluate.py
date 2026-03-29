from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("experiments/baseline_xgb/model.joblib")
OUTPUT_DIR = Path("experiments/baseline_xgb")
REQUIRED_COLUMNS = ["date", "player_1", "player_2", "target"]
FEATURE_NAMES = [
    "p1_win_rate",
    "p2_win_rate",
    "p1_matches",
    "p2_matches",
    "win_rate_diff",
    "matches_diff",
]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def validate_columns(df: pd.DataFrame, split_name: str) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in {split_name}: {missing_str}")


def validate_target(df: pd.DataFrame, split_name: str) -> None:
    valid_target = df["target"].isin([0, 1])
    if not valid_target.all():
        raise ValueError(f"Invalid target values found in {split_name}; expected only 0 or 1")


def build_player_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    player_1_stats = pd.DataFrame(
        {
            "player": train_df["player_1"],
            "matches": 1,
            "wins": train_df["target"],
        }
    )
    player_2_stats = pd.DataFrame(
        {
            "player": train_df["player_2"],
            "matches": 1,
            "wins": 1 - train_df["target"],
        }
    )

    player_stats = pd.concat([player_1_stats, player_2_stats], ignore_index=True)
    player_stats = player_stats.groupby("player", as_index=True)[["matches", "wins"]].sum()
    player_stats["win_rate"] = player_stats["wins"] / player_stats["matches"]
    return player_stats


def create_features(df: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
    matches_map = player_stats["matches"].to_dict()
    win_rate_map = player_stats["win_rate"].to_dict()

    p1_matches = df["player_1"].map(matches_map).fillna(0).astype(float)
    p2_matches = df["player_2"].map(matches_map).fillna(0).astype(float)
    p1_win_rate = df["player_1"].map(win_rate_map).fillna(0.5).astype(float)
    p2_win_rate = df["player_2"].map(win_rate_map).fillna(0.5).astype(float)

    features = pd.DataFrame(
        {
            "p1_win_rate": p1_win_rate,
            "p2_win_rate": p2_win_rate,
            "p1_matches": p1_matches,
            "p2_matches": p2_matches,
            "win_rate_diff": p1_win_rate - p2_win_rate,
            "matches_diff": p1_matches - p2_matches,
        }
    )
    return features[FEATURE_NAMES]


def evaluate_model(model, test_df: pd.DataFrame, test_features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    pred_proba = model.predict_proba(test_features)[:, 1]
    pred_label = (pred_proba >= 0.5).astype(int)

    metrics = {
        "model_name": "baseline_xgb",
        "n_test_rows": len(test_df),
        "log_loss": float(log_loss(test_df["target"], pred_proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(test_df["target"], pred_label)),
        "feature_names": FEATURE_NAMES,
    }

    predictions = test_df[["date", "player_1", "player_2", "target"]].copy()
    predictions["pred_proba"] = pred_proba
    predictions["pred_label"] = pred_label
    return metrics, predictions


def save_outputs(metrics: dict, predictions: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    with (OUTPUT_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def print_summary(metrics: dict) -> None:
    print(f"Test log_loss: {metrics['log_loss']:.6f}")
    print(f"Test accuracy: {metrics['accuracy']:.6f}")


def main() -> None:
    train_df = load_csv(TRAIN_PATH)
    test_df = load_csv(TEST_PATH)
    model = load_model(MODEL_PATH)

    validate_columns(train_df, "train")
    validate_columns(test_df, "test")
    validate_target(train_df, "train")
    validate_target(test_df, "test")

    player_stats = build_player_stats(train_df)
    test_features = create_features(test_df, player_stats)

    metrics, predictions = evaluate_model(model, test_df, test_features)
    save_outputs(metrics, predictions)
    print_summary(metrics)


if __name__ == "__main__":
    main()
