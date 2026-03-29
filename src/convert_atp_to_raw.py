from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/tennis_atp/atp_matches_2024.csv")
OUTPUT_PATH = Path("data/raw_matches.csv")
REQUIRED_COLUMNS = ["tourney_date", "winner_name", "loser_name", "winner_rank", "loser_rank"]
OUTPUT_COLUMNS = ["date", "player_1", "player_2", "winner", "p1_rank", "p2_rank"]


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_str}")


def convert_matches(df: pd.DataFrame) -> pd.DataFrame:
    converted = df[REQUIRED_COLUMNS].dropna(subset=REQUIRED_COLUMNS).copy()
    converted["date"] = pd.to_datetime(
        converted["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    converted = converted.dropna(subset=["date"]).copy()

    even_index = converted.index % 2 == 0
    converted["player_1"] = converted["loser_name"]
    converted["player_2"] = converted["winner_name"]
    converted["p1_rank"] = converted["loser_rank"]
    converted["p2_rank"] = converted["winner_rank"]
    converted.loc[even_index, "player_1"] = converted.loc[even_index, "winner_name"]
    converted.loc[even_index, "player_2"] = converted.loc[even_index, "loser_name"]
    converted.loc[even_index, "p1_rank"] = converted.loc[even_index, "winner_rank"]
    converted.loc[even_index, "p2_rank"] = converted.loc[even_index, "loser_rank"]
    converted["winner"] = converted["winner_name"]
    converted["date"] = converted["date"].dt.strftime("%Y-%m-%d")

    converted = converted.sort_values("date", ascending=True, kind="mergesort")
    converted = converted.reset_index(drop=True)
    return converted[OUTPUT_COLUMNS]


def save_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Rows written: 0")
        print("Min date: n/a")
        print("Max date: n/a")
        print("Winner equals player_1: 0")
        print("Winner equals player_2: 0")
        return

    winner_is_player_1 = int((df["winner"] == df["player_1"]).sum())
    winner_is_player_2 = int((df["winner"] == df["player_2"]).sum())

    print(f"Rows written: {len(df)}")
    print(f"Min date: {df['date'].min()}")
    print(f"Max date: {df['date'].max()}")
    print(f"Winner equals player_1: {winner_is_player_1}")
    print(f"Winner equals player_2: {winner_is_player_2}")


def main() -> None:
    raw_df = load_input(INPUT_PATH)
    validate_columns(raw_df)
    converted_df = convert_matches(raw_df)
    save_output(converted_df, OUTPUT_PATH)
    print_summary(converted_df)


if __name__ == "__main__":
    main()
