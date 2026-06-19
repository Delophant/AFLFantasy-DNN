r"""Audit player team assignments in player_match_stats.csv.

Reports confirmed trades between seasons and checks for actual data integrity
issues (duplicate match rows, split-team seasons). Does not treat trades as errors.

Run:
    .\.venv\Scripts\python.exe src\roster_audit.py
    .\.venv\Scripts\python.exe src\roster_audit.py --season 2026 --baseline 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "player_match_stats.csv"
TRADES_OUT = ROOT / "data" / "processed" / "roster_trades.csv"
ISSUES_OUT = ROOT / "data" / "processed" / "roster_audit.csv"


def primary_teams(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Most-played team per player in a season."""
    s = df[df["season"] == season]
    rows = []
    for player, g in s.groupby("player"):
        counts = g["team"].value_counts()
        rows.append({
            "player": player,
            "team": counts.index[0],
            "games": int(counts.iloc[0]),
            "teams_if_split": len(counts),
        })
    return pd.DataFrame(rows)


def season_scoring_rank(df: pd.DataFrame, season: int) -> pd.Series:
    return (
        df[df["season"] == season]
        .groupby("player")["fantasy_points"]
        .sum()
        .sort_values(ascending=False)
    )


def trades_table(
    df: pd.DataFrame,
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    season: int,
    scorer_rank: pd.Series,
) -> pd.DataFrame:
    """One row per player who changed primary team between seasons."""
    merged = baseline.merge(
        current,
        on="player",
        how="inner",
        suffixes=("_prior", "_current"),
    )
    changed = merged[merged["team_prior"] != merged["team_current"]].copy()
    if changed.empty:
        return changed

    changed = changed.rename(columns={
        "team_prior": "prior_team",
        "team_current": "current_team",
        "games_prior": "prior_games",
        "games_current": "current_games",
    })
    changed["prior_fantasy_total"] = changed["player"].map(scorer_rank).fillna(0)

    # Latest-squad context (useful for prediction squads, not a separate issue)
    s = df[df["season"] == season]
    in_latest = []
    for _, row in changed.iterrows():
        team_rows = s[(s["team"] == row["current_team"]) & (s["player"] == row["player"])]
        if team_rows.empty:
            in_latest.append(False)
            continue
        last_date = s[s["team"] == row["current_team"]]["date"].max()
        in_latest.append(bool((team_rows["date"] == last_date).any()))
    changed["in_latest_squad"] = in_latest

    cols = [
        "player", "prior_team", "current_team",
        "prior_games", "current_games", "prior_fantasy_total", "in_latest_squad",
    ]
    return changed[cols].sort_values("prior_fantasy_total", ascending=False)


def departed_regulars(
    df: pd.DataFrame,
    baseline_season: int,
    current_season: int,
    traded_players: set[str],
    min_games: int = 12,
) -> pd.DataFrame:
    """Regulars who left a team and are not active elsewhere (retired / missing)."""
    base = df[df["season"] == baseline_season]
    curr = df[df["season"] == current_season]
    rows = []
    for team in sorted(base["team"].unique()):
        regulars = (
            base[base["team"] == team]
            .groupby("player")
            .size()
            .loc[lambda s: s >= min_games]
            .index
        )
        still_there = set(curr[curr["team"] == team]["player"].unique())
        for player in regulars:
            if player in still_there or player in traded_players:
                continue
            rows.append({
                "player": player,
                "prior_team": team,
                "baseline_games": int(base[(base.team == team) & (base.player == player)].shape[0]),
            })
    return pd.DataFrame(rows)


def match_duplicates(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Same player listed for both teams in one match — actual scrape error."""
    s = df[df["season"] == season]
    rows = []
    for mid, g in s.groupby("match_id"):
        for player, pg in g.groupby("player"):
            teams = pg["team"].unique()
            if len(teams) > 1:
                rows.append({
                    "issue": "match_duplicate",
                    "player": player,
                    "detail": f"match {mid} ({pg['date'].iloc[0].date()}): {' / '.join(sorted(teams))}",
                })
    return pd.DataFrame(rows)


def split_team_seasons(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Player appeared for 2+ teams in the same season — possible scrape error."""
    s = df[df["season"] == season]
    rows = []
    for player, g in s.groupby("player"):
        teams = g["team"].value_counts()
        if len(teams) > 1:
            rows.append({
                "issue": "split_team_season",
                "player": player,
                "detail": ", ".join(f"{t} ({n}g)" for t, n in teams.items()),
            })
    return pd.DataFrame(rows)


def main(season: int, baseline: int) -> None:
    if not DATA.exists():
        print(f"ERROR: missing {DATA}")
        sys.exit(1)

    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["player", "season", "date"])
    print(f"dataset: {len(df):,} rows | seasons {sorted(df['season'].unique())}\n")

    if season not in df["season"].values:
        print(f"ERROR: no data for season {season}")
        sys.exit(1)

    base_pt = primary_teams(df, baseline) if baseline in df["season"].values else pd.DataFrame()
    curr_pt = primary_teams(df, season)
    scorer_rank = season_scoring_rank(df, baseline) if not base_pt.empty else pd.Series(dtype=float)

    print(f"=== TRADES {baseline} -> {season} ===")
    if base_pt.empty:
        print(f"(no baseline season {baseline})\n")
        trades = pd.DataFrame()
    else:
        trades = trades_table(df, base_pt, curr_pt, season, scorer_rank)
        print(f"{len(trades)} players changed primary team\n")
        if not trades.empty:
            print(trades.to_string(index=False))
        print()

    traded_players = set(trades["player"]) if not trades.empty else set()

    print(f"=== DEPARTED (regulars not traded, not on same team in {season}) ===")
    if base_pt.empty:
        departed = pd.DataFrame()
        print("(skipped)\n")
    else:
        departed = departed_regulars(df, baseline, season, traded_players)
        print(f"{len(departed)} players\n")
        if not departed.empty:
            print(departed.head(30).to_string(index=False))
            if len(departed) > 30:
                print(f"  ... and {len(departed) - 30} more")
        print()

    print(f"=== DATA INTEGRITY ({season}) ===")
    dups = match_duplicates(df, season)
    splits = split_team_seasons(df, season)
    issues = pd.concat([dups, splits], ignore_index=True)
    print(f"match duplicates: {len(dups)}")
    print(f"split-team seasons: {len(splits)}\n")
    if not issues.empty:
        print(issues.to_string(index=False))
        print()
    else:
        print("(no integrity issues found)\n")

    TRADES_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not trades.empty:
        trades.to_csv(TRADES_OUT, index=False)
        print(f"wrote {len(trades)} trades -> {TRADES_OUT.relative_to(ROOT)}")
    if not issues.empty:
        issues.to_csv(ISSUES_OUT, index=False)
        print(f"wrote {len(issues)} issues -> {ISSUES_OUT.relative_to(ROOT)}")
    elif ISSUES_OUT.exists():
        ISSUES_OUT.unlink()
        print(f"removed stale {ISSUES_OUT.relative_to(ROOT)} (no issues)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit trades and data integrity in scraped stats")
    parser.add_argument("--season", type=int, default=2026, help="Season to audit (default: 2026)")
    parser.add_argument("--baseline", type=int, default=2025, help="Compare against this season (default: 2025)")
    args = parser.parse_args()
    main(args.season, args.baseline)
