r"""Build a leakage-free modelling table for next-round fantasy prediction.

The golden rule: every feature for a given match must be computable *before*
that match is played. We therefore derive features only from a player's PRIOR
games (shifted by one), and the target is the CURRENT game's fantasy score.

Output: data/processed/regression_dataset.csv

Run:
    .\.venv\Scripts\python.exe src\features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "player_match_stats.csv"
OUT = ROOT / "data" / "processed" / "regression_dataset.csv"

# Feature columns the models will actually use (all known pre-match).
FEATURES = [
    "prev_score",
    "roll3_fantasy",
    "roll5_fantasy",
    "season_avg_fantasy",
    "career_avg_fantasy",
    "roll3_disposals",
    "roll5_disposals",
    "roll3_tackles",
    "games_played",
    "is_home",
]
TARGET = "target"


def build() -> pd.DataFrame:
    df = pd.read_csv(SRC, parse_dates=["date"])
    # Chronological order *within each player* is essential for the shifts below.
    df = df.sort_values(["player", "date"]).reset_index(drop=True)

    by_player = df.groupby("player", sort=False)

    # shift(1) => "previous game"; rolling on the shifted series never peeks at "now".
    def prior_roll(col: str, window: int) -> pd.Series:
        return by_player[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        )

    df["prev_score"] = by_player["fantasy_points"].shift(1)
    df["roll3_fantasy"] = prior_roll("fantasy_points", 3)
    df["roll5_fantasy"] = prior_roll("fantasy_points", 5)
    df["roll3_disposals"] = prior_roll("disposals", 3)
    df["roll5_disposals"] = prior_roll("disposals", 5)
    df["roll3_tackles"] = prior_roll("tackles", 3)

    # Expanding (all prior games) averages, shifted so "now" is excluded.
    df["career_avg_fantasy"] = by_player["fantasy_points"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["season_avg_fantasy"] = df.groupby(["player", "season"])["fantasy_points"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    df["games_played"] = by_player.cumcount()  # prior games count
    df["is_home"] = (df["home_away"] == "home").astype(int)
    df["target"] = df["fantasy_points"]

    keep = ["season", "date", "player", "team", "opponent"] + FEATURES + [TARGET]
    model_df = df[keep].copy()

    before = len(model_df)
    # Require a real recent-form window (>= 3 prior games) -> drops early-career rows.
    model_df = model_df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"rows: {before:,} -> {len(model_df):,} after dropping warm-up games "
          f"(need >=3 prior games)")
    return model_df


def main() -> None:
    model_df = build()
    model_df.to_csv(OUT, index=False)

    print(f"\nfeatures ({len(FEATURES)}): {FEATURES}")
    print(f"target: {TARGET}")
    print(f"\nseasons: {sorted(model_df['season'].unique())}")
    print("rows per season:")
    print(model_df.groupby("season").size().to_string())

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\nsample rows:")
    cols = ["season", "player", "prev_score", "roll3_fantasy", "roll5_fantasy",
            "season_avg_fantasy", "career_avg_fantasy", "games_played", "is_home", "target"]
    print(model_df[cols].head(8).round(1).to_string(index=False))

    print(f"\nwrote -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
