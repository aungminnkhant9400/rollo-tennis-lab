from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
OUTPUT_DIR = Path("experiments/baseline_xgb")
REQUIRED_COLUMNS = ["date", "player_1", "player_2", "target", "p1_rank", "p2_rank"]
ALPHA = 10.0
RECENT_ALPHA = 5.0
RECENT_COUNT_BETA = 5.0
FEATURE_NAMES = [
    "p1_win_rate",
    "p2_win_rate",
    "p1_matches",
    "p2_matches",
    "win_rate_diff",
    "matches_diff",
    "p1_rank",
    "p2_rank",
    "rank_diff",
    "p1_recent_win_rate",
    "p2_recent_win_rate",
    "recent_win_rate_diff",
    "p1_recent_count_strength",
    "p2_recent_count_strength",
    "recent_count_strength_diff",
]


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


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
    player_stats["win_rate"] = (player_stats["wins"] + ALPHA * 0.5) / (player_stats["matches"] + ALPHA)
    return player_stats


def build_recent_player_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    sorted_train_df = train_df.sort_values("date", ascending=True, kind="mergesort")
    player_1_stats = pd.DataFrame(
        {
            "player": sorted_train_df["player_1"],
            "date": sorted_train_df["date"],
            "win": sorted_train_df["target"],
        }
    )
    player_2_stats = pd.DataFrame(
        {
            "player": sorted_train_df["player_2"],
            "date": sorted_train_df["date"],
            "win": 1 - sorted_train_df["target"],
        }
    )

    recent_history = pd.concat([player_1_stats, player_2_stats], ignore_index=True)
    recent_history = recent_history.sort_values(["player", "date"], ascending=True, kind="mergesort")
    recent_history = recent_history.groupby("player", group_keys=False).tail(10)

    recent_stats = recent_history.groupby("player", as_index=True)["win"].agg(matches="size", wins="sum")
    recent_stats["win_rate"] = (recent_stats["wins"] + RECENT_ALPHA * 0.5) / (recent_stats["matches"] + RECENT_ALPHA)
    return recent_stats


def create_features(
    df: pd.DataFrame,
    player_stats: pd.DataFrame,
    recent_player_stats: pd.DataFrame,
) -> pd.DataFrame:
    matches_map = player_stats["matches"].to_dict()
    win_rate_map = player_stats["win_rate"].to_dict()
    recent_win_rate_map = recent_player_stats["win_rate"].to_dict()

    p1_matches = df["player_1"].map(matches_map).fillna(0).astype(float)
    p2_matches = df["player_2"].map(matches_map).fillna(0).astype(float)
    p1_win_rate = df["player_1"].map(win_rate_map).fillna(0.5).astype(float)
    p2_win_rate = df["player_2"].map(win_rate_map).fillna(0.5).astype(float)
    p1_rank = df["p1_rank"].astype(float)
    p2_rank = df["p2_rank"].astype(float)
    p1_recent_win_rate = df["player_1"].map(recent_win_rate_map).fillna(0.5).astype(float)
    p2_recent_win_rate = df["player_2"].map(recent_win_rate_map).fillna(0.5).astype(float)
    recent_matches_map = recent_player_stats["matches"].to_dict()
    p1_recent_matches = df["player_1"].map(recent_matches_map).fillna(0).astype(float)
    p2_recent_matches = df["player_2"].map(recent_matches_map).fillna(0).astype(float)
    p1_recent_count_strength = p1_recent_matches / (p1_recent_matches + RECENT_COUNT_BETA)
    p2_recent_count_strength = p2_recent_matches / (p2_recent_matches + RECENT_COUNT_BETA)

    features = pd.DataFrame(
        {
            "p1_win_rate": p1_win_rate,
            "p2_win_rate": p2_win_rate,
            "p1_matches": p1_matches,
            "p2_matches": p2_matches,
            "win_rate_diff": p1_win_rate - p2_win_rate,
            "matches_diff": p1_matches - p2_matches,
            "p1_rank": p1_rank,
            "p2_rank": p2_rank,
            "rank_diff": p2_rank - p1_rank,
            "p1_recent_win_rate": p1_recent_win_rate,
            "p2_recent_win_rate": p2_recent_win_rate,
            "recent_win_rate_diff": p1_recent_win_rate - p2_recent_win_rate,
            "p1_recent_count_strength": p1_recent_count_strength,
            "p2_recent_count_strength": p2_recent_count_strength,
            "recent_count_strength_diff": p1_recent_count_strength - p2_recent_count_strength,
        }
    )
    return features[FEATURE_NAMES]


def train_model(features: pd.DataFrame, target: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=1,
        objective="binary:logistic",
        tree_method="gpu_hist",
    )
    model.fit(features, target)
    return model


def evaluate_model(model: XGBClassifier, val_df: pd.DataFrame, val_features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    pred_proba = model.predict_proba(val_features)[:, 1]
    pred_label = (pred_proba >= 0.5).astype(int)

    metrics = {
        "model_name": "baseline_xgb",
        "n_train_rows": None,
        "n_val_rows": len(val_df),
        "log_loss": float(log_loss(val_df["target"], pred_proba)),
        "accuracy": float(accuracy_score(val_df["target"], pred_label)),
        "feature_names": FEATURE_NAMES,
    }

    predictions = val_df[["date", "player_1", "player_2", "target"]].copy()
    predictions["pred_proba"] = pred_proba
    predictions["pred_label"] = pred_label
    return metrics, predictions


def save_outputs(model: XGBClassifier, metrics: dict, predictions: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_DIR / "model.joblib")
    predictions.to_csv(OUTPUT_DIR / "val_predictions.csv", index=False)

    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def print_summary(metrics: dict) -> None:
    print(f"Validation log_loss: {metrics['log_loss']:.6f}")
    print(f"Validation accuracy: {metrics['accuracy']:.6f}")


def main() -> None:
    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)

    validate_columns(train_df, "train")
    validate_columns(val_df, "val")
    validate_target(train_df, "train")
    validate_target(val_df, "val")

    player_stats = build_player_stats(train_df)
    recent_player_stats = build_recent_player_stats(train_df)
    train_features = create_features(train_df, player_stats, recent_player_stats)
    val_features = create_features(val_df, player_stats, recent_player_stats)

    model = train_model(train_features, train_df["target"])
    metrics, predictions = evaluate_model(model, val_df, val_features)
    metrics["n_train_rows"] = len(train_df)

    save_outputs(model, metrics, predictions)
    print_summary(metrics)


if __name__ == "__main__":
    main()
