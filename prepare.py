from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/raw_matches.csv")
OUTPUT_DIR = Path("data/processed")
REQUIRED_COLUMNS = ["date", "player_1", "player_2", "winner", "p1_rank", "p2_rank"]
OUTPUT_COLUMNS = ["date", "player_1", "player_2", "winner", "p1_rank", "p2_rank", "target"]


def load_raw_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    return pd.read_csv(input_path)


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_str}")


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df[REQUIRED_COLUMNS].copy()
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date"])

    player_1_wins = cleaned["winner"] == cleaned["player_1"]
    player_2_wins = cleaned["winner"] == cleaned["player_2"]
    valid_winner = player_1_wins | player_2_wins

    cleaned = cleaned.loc[valid_winner].copy()
    cleaned["target"] = player_1_wins.loc[valid_winner].astype(int)

    cleaned = cleaned.sort_values("date", ascending=True, kind="mergesort")
    cleaned = cleaned.reset_index(drop=True)
    return cleaned[OUTPUT_COLUMNS]


def split_chronologically(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_rows = len(df)
    train_end = int(total_rows * 0.70)
    val_end = int(total_rows * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)


def format_date_range(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a"
    start = df["date"].min().date().isoformat()
    end = df["date"].max().date().isoformat()
    return f"{start} to {end}"


def print_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    total_rows = len(train_df) + len(val_df) + len(test_df)

    print(f"Total cleaned rows: {total_rows}")
    print(f"Train rows: {len(train_df)} | Date range: {format_date_range(train_df)}")
    print(f"Val rows: {len(val_df)} | Date range: {format_date_range(val_df)}")
    print(f"Test rows: {len(test_df)} | Date range: {format_date_range(test_df)}")


def main() -> None:
    raw_df = load_raw_data(INPUT_PATH)
    validate_columns(raw_df)
    cleaned_df = clean_matches(raw_df)
    train_df, val_df, test_df = split_chronologically(cleaned_df)
    save_splits(train_df, val_df, test_df)
    print_summary(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
