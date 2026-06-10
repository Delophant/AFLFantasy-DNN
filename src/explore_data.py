r"""Explore and sanity-check the scraped AFL player-match dataset.

Prints a compact data-quality + summary report and saves a few figures to
reports/figures/. The goal is to *understand the data and the target* before
we build any model:

* What does the AFL Fantasy score (our regression target) look like?
* Which raw stats drive fantasy points? (football intuition -> features)
* Do the numbers make sense (known stars on top, no obvious corruption)?

Run:
    .\.venv\Scripts\python.exe src\explore_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, don't try to open a window
import matplotlib.pyplot as plt
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "player_match_stats.csv"
FIG_DIR = ROOT / "reports" / "figures"

STAT_COLS = [
    "kicks", "handballs", "disposals", "marks", "goals", "behinds", "tackles",
    "hitouts", "goal_assists", "inside50s", "clearances", "clangers",
    "rebound50s", "frees_for", "frees_against",
]
TARGET = "fantasy_points"


def banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"])

    banner("DATASET OVERVIEW")
    print(f"rows (player-matches): {len(df):,}")
    print(f"matches              : {df['match_id'].nunique():,}")
    print(f"players              : {df['player'].nunique():,}")
    print(f"teams                : {df['team'].nunique()}")
    print(f"seasons              : {sorted(df['season'].unique())}")
    print(f"date range           : {df['date'].min().date()} -> {df['date'].max().date()}")
    print("\nplayer-rows per season:")
    print(df.groupby("season").size().to_string())

    banner("DATA QUALITY")
    missing = df.isna().sum()
    print("missing values per column (non-zero only):")
    print(missing[missing > 0].to_string() if (missing > 0).any() else "  none")
    dupes = df.duplicated(subset=["match_id", "team", "player"]).sum()
    print(f"\nduplicate (match, team, player) rows: {dupes}")
    games_per_player = df.groupby("player").size()
    print(f"\ngames per player: min={games_per_player.min()}, "
          f"median={int(games_per_player.median())}, max={games_per_player.max()}")
    print(f"players with < 5 games: {(games_per_player < 5).sum()}")

    banner("TARGET: AFL FANTASY POINTS")
    print(df[TARGET].describe().to_string())

    banner("WHICH RAW STATS DRIVE FANTASY POINTS? (Pearson r)")
    corr = df[STAT_COLS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    print(corr.to_string())

    banner("TOP 15 PLAYERS BY AVG FANTASY (min 20 games)")
    elig = games_per_player[games_per_player >= 20].index
    top = (
        df[df["player"].isin(elig)]
        .groupby("player")[TARGET]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(15)
    )
    print(top.round(1).to_string())

    # ---- Figures ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    df[TARGET].plot.hist(bins=60, color="#3b7dd8", edgecolor="white")
    plt.title("Distribution of AFL Fantasy scores (per player per match)")
    plt.xlabel("Fantasy points")
    plt.ylabel("Count")
    plt.tight_layout()
    f1 = FIG_DIR / "fantasy_distribution.png"
    plt.savefig(f1, dpi=110)
    plt.close()

    plt.figure(figsize=(8, 5))
    corr.sort_values().plot.barh(color="#2bb673")
    plt.title("Correlation of each raw stat with fantasy points")
    plt.xlabel("Pearson r")
    plt.tight_layout()
    f2 = FIG_DIR / "stat_fantasy_correlation.png"
    plt.savefig(f2, dpi=110)
    plt.close()

    print(f"\nsaved figures:\n  {f1.relative_to(ROOT)}\n  {f2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
